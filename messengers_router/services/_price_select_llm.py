"""LLM выбирает строки прайса, которые являются ровно той услугой из вопроса пациента.

Класс дефектов «цена не той услуги» (журнал, BUG-2026-09-30-SURGEON-PRICE-FOR-SPECIALIST):
правила подбирали строку прайса по похожим словам — «приём эндокринолога» → «Прием …
хирурга (эндокринологическое отделение)», «гастроскопия» → анализ «Гастрин», «приём
пластического хирурга» → обычный хирург. Решение владельца 30.09: что спросил пациент,
решает LLM, правила только собирают кандидатов.

Как устроено:
  1) кандидаты — строки прайса, которые находят правила: по реплике, по фразе услуги,
     по подсказкам (сущность NLU, выбор резолвера);
  2) LLM получает реплику и пронумерованный список и возвращает номера строк, которые
     являются той услугой; выдумать позицию она не может — только выбрать номер;
  3) не нашла — предлагает до трёх названий, под которыми услуга бывает в прайсе
     («гастроскопия» → «ФГДС»); по ним второй круг поиска и второй выбор. Такой термин
     пишется в журнал: синонимы заносит клиника, не мы (решение владельца 30.09);
  4) особые варианты (cito, капиллярная кровь, к.м.н., на дому…), которых пациент не
     просил, отбрасываются, если среди выбранных есть базовый.

Отказоустойчивость: сбой, таймаут, мусор, режим strict, kill-switch → None, и инструмент
отвечает как раньше. «Не нашла» тоже None: LLM заменяет ответ правил только тем, что
положительно выбрала (`reconcile_with_proposal`).

Выбор — на модуле `choose()` (ход (в), L-12), выключатель `[MESSENGER_ROUTER] choose_price`:
`off` (по умолчанию) — прежний вызов и разбор; `legacy` — прежний промпт и форма ответа через
модуль (перенос доказывается на записанных ответах прод-LLM: на стенде `replay` ответы обязаны
совпасть с `off`); `on` — роли system/user, JSON Schema, коды причин (промпт
`choose_price_service`). Политика сбоя во всех режимах прежняя; её меняет L-13 отдельным шагом.

Политика силы совпадения (L-13, решение владельца 07.10 — вариант B), выключатель
`[MESSENGER_ROUTER] price_match_policy` (по умолчанию `shadow` — решение только в телеметрию хода):
под `on` сильное совпадение по данным (дословное название, синоним МИС, код — `_strong_match`) — только
эти строки, без LLM; LLM не нашла услугу и во втором круге — честное «не нашёл»; LLM не ответила
внятно, а правила показали бы ОДНУ слабую строку — до 5 названий без цен (выбирает пациент), а нет таких — оператор;
показали бы СПИСОК (≥2 строк) — список правил остаётся (решение владельца 11.10 по замеру: строгий
вариант B прятал и списки с нужной строкой — форму 22.07 «при неоднозначности — список»). Политика
заменяет только слабое предложение правил: правила ничего не предложили или названо слишком общее —
их прежний ответ. Без LLM по замыслу (strict, ответ без своих слов об услуге, «по ОМС», выключатель)
— как раньше.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agent_logic_2.persist import DATA_DIR, ensure_dir

from .. import turn_parse, turn_telemetry
from ..choice import LegacyWire, Option, Task, choose
from ..llm_runtime import generate_text
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg
from ._common import _normalise_catalog_text
from ._prices_helpers import (
    _apply_service_synonyms,
    _extract_price_service_from_query,
    _price_query_tokens,
    _query_nonbase_price_flags,
    _rank_price_rows,
    _row_nonbase_price_flags,
    price_question_names_service,
    query_has_unsatisfiable_qualifier,
    query_names_service_in_other_words,
)
from ._strong_match import strong_rows

logger = logging.getLogger(__name__)

# Kill-switch: [MESSENGER_ROUTER] llm_price_service_select = false в config.ini
# (рестарт контейнера). Слой fail-open — по умолчанию ВКЛ.
_ENABLED: bool = bool(getattr(_cfg, "MR_LLM_PRICE_SERVICE_SELECT", True))

_POOL_LIMIT = 24
_PER_QUERY_LIMIT = 10
_PER_TERM_LIMIT = 8
_FEW_CANDIDATES = 5
_PROPOSAL_FIRST = 10
_MAX_TERMS = 3
_MAX_TERM_LEN = 60
# Ответ — несколько чисел в JSON; дольше ждать — значит LLM занята пациентами, и честнее
# ответить по правилам, чем держать человека.
_LLM_TIMEOUT_S = 12
_LLM_QUEUE_TIMEOUT_MS = 3000

_JOURNAL_NAME = "price_synonym_suggestions.jsonl"

# Задача модуля выбора: номера строк прайса, причина пустого выбора, до 3 терминов второго круга.
PRICE_TASK = Task(
    name="price_service",
    reasons=("other_service", "qualifier_absent", "too_general", "not_a_service"),
    max_terms=_MAX_TERMS,
    num_predict=192,
    timeout_s=_LLM_TIMEOUT_S,
    queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
)
_CHOOSE_MODES = ("off", "legacy", "on")
_POLICY_MODES = ("off", "shadow", "on")
_VARIANT_B_LIMIT = 5
# Причины пустого выбора, при которых LLM не признала в реплике названия услуги: «Цена анализа»
# (слишком общее), «сможете посчитать» (не услуга). Ни «не нашёл» пациенту, ни синоним клинике.
_NOT_A_SERVICE_NAME = frozenset({"too_general", "not_a_service"})


def match_policy() -> str:
    """Политика силы совпадения (L-13): off | shadow | on."""

    raw = str(getattr(_cfg, "MR_PRICE_MATCH_POLICY", "shadow") or "").strip().lower()
    return raw if raw in _POLICY_MODES else "shadow"


def _choose_mode() -> str:
    raw = str(getattr(_cfg, "MR_CHOOSE_PRICE", "off") or "").strip().lower()
    return raw if raw in _CHOOSE_MODES else "off"


@dataclass(frozen=True)
class PriceSelection:
    """Строки прайса, которые LLM признала той услугой (уже без непрошеных вариантов).

    `status` — решение политики силы совпадения (L-13; не `picked` — только под
    `price_match_policy = on`): `picked` — выбор LLM; `strong` — сильное совпадение по данным, без
    LLM; `none_fit` — LLM не нашла услугу и во втором круге; `unavailable` — LLM не ответила внятно
    на слабом совпадении, `names` — до 5 названий совпадений целым словом (вариант B).
    """

    rows: tuple[dict[str, Any], ...]
    search_terms: tuple[str, ...] = ()
    status: str = "picked"
    names: tuple[str, ...] = ()


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _normalise_catalog_text(str(row.get("serviceName") or row.get("name") or "")),
        str(row.get("serviceHomecode") or row.get("homecode") or "").strip(),
        str(row.get("cost") if row.get("cost") is not None else row.get("price") or ""),
    )


def _row_name(row: dict[str, Any]) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip()


def _row_cost(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("cost") if row.get("cost") is not None else row.get("price") or 0))
    except (TypeError, ValueError):
        return 0


def _build_pool(
    texts: Sequence[str],
    retail_rows: list[dict[str, Any]],
    *,
    per_text: int,
    first: Sequence[dict[str, Any]] = (),
    fallback_texts: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Кандидаты для LLM: ответ правил, затем поиск правил по каждому тексту.

    Один поиск по прайсу — ~120 мс CPU, поэтому одинаковые тексты не ищем дважды, а
    `fallback_texts` (отдельные слова) ищем, только если основных кандидатов мало.
    """

    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    for text in texts:
        norm = " ".join(sorted(_price_query_tokens(str(text or "")))) or _normalise_catalog_text(str(text or ""))
        if norm and norm not in seen_texts:
            seen_texts.add(norm)
            unique_texts.append(str(text))
    texts = unique_texts
    for row in first:
        key = _row_key(row) if isinstance(row, dict) else None
        if key is None or key in seen:
            continue
        seen.add(key)
        pool.append(row)
        if len(pool) >= _POOL_LIMIT:
            return pool
    for text in texts:
        if not str(text or "").strip():
            continue
        for row in _rank_price_rows(retail_rows, str(text), limit=per_text):
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            pool.append(row)
            if len(pool) >= _POOL_LIMIT:
                return pool
    if len(pool) < _FEW_CANDIDATES and fallback_texts:
        return _build_pool(fallback_texts, retail_rows, per_text=per_text, first=pool)
    return pool


def _build_prompt(question: str, pool: list[dict[str, Any]]) -> str:
    # Только реплика пациента. Сущность NLU и выбор правил в подсказку не идут: это
    # догадка по этой же реплике («биопсия» → «Биопсия вульвы»), и LLM послушно её
    # повторяла — общий вопрос сжимался до одной позиции (свип 01.10).
    rows_text = "\n".join(f"{i}. {_row_name(row)} — {_row_cost(row)} руб." for i, row in enumerate(pool, 1))
    return (
        load_prompt_text("price_service_select")
        .replace("<<QUESTION>>", question.strip())
        .replace("<<ROWS>>", rows_text or "(подходящих позиций не найдено)")
    ).strip()


def _parse_ids(items: Any, pool_size: int) -> list[int] | None:
    if not isinstance(items, list):
        return None
    ids: list[int] = []
    for item in items:
        try:
            num = int(item)
        except (TypeError, ValueError):
            return None
        if not 1 <= num <= pool_size:
            return None
        if num not in ids:
            ids.append(num)
    return ids


def _parse_reply(raw: str, pool_size: int) -> tuple[list[int], list[str]] | None:
    """(номера строк «та услуга», термины поиска) или None, если ответ не годится.

    Номер вне списка — признак, что LLM не держит список в голове; такому ответу не
    верим целиком, а не только этому номеру.
    """

    try:
        data = json.loads(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    match = _parse_ids(data.get("match", []), pool_size)
    terms = data.get("search_terms", [])
    if match is None or not isinstance(terms, list):
        return None
    clean_terms: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if text and len(text) <= _MAX_TERM_LEN and text not in clean_terms:
            clean_terms.append(text)
    return match, clean_terms[:_MAX_TERMS]


_TOKEN_DF_CACHE: dict[tuple[int, str], dict[str, int]] = {}


def _token_df(retail_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Во скольких названиях прайса встречается слово (кэш по отпечатку среза)."""

    key = (len(retail_rows), _row_name(retail_rows[0])[:48] if retail_rows else "")
    cached = _TOKEN_DF_CACHE.get(key)
    if cached is None:
        cached = {}
        for name in {_row_name(row) for row in retail_rows}:
            for token in set(_price_query_tokens(name)):
                cached[token] = cached.get(token, 0) + 1
        _TOKEN_DF_CACHE.clear()
        _TOKEN_DF_CACHE[key] = cached
    return cached


def _whole_word_names(question: str, pool: list[dict[str, Any]], retail_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Вариант B (L-13): названия строк, в которых есть самое различающее слово пациента целиком.

    Различающее — самое редкое в названиях прайса: родовые слова («анализ», «тест», «приём») часты и
    отпадают сами, без списка слов. По началу слова нельзя: на «гастроскопию» первым оказался бы
    «Гастрин»; слова нет в прайсе вовсе — названий нет (честный перевод). Порядок — как у кандидатов.
    """

    wanted = set(_price_query_tokens(question))
    if not wanted:
        return ()
    df = _token_df(retail_rows)
    key = min(wanted, key=lambda token: (df.get(token, 0), token))
    if not df.get(key):
        return ()
    names = [_row_name(row) for row in pool if key in set(_price_query_tokens(_row_name(row)))]
    return tuple(dict.fromkeys(name for name in names if name))[:_VARIANT_B_LIMIT]


def _policy_label(selection: PriceSelection | None) -> str:
    if selection is None:
        return "rules"
    if selection.status == "unavailable":
        return "names" if selection.names else "handoff"
    return {"none_fit": "not_found"}.get(selection.status, selection.status)


def _legacy_answer(raw: str, _options: Sequence[Option]) -> dict[str, Any] | None:
    """Прежняя форма ответа ({"match", "search_terms"}) → форма модуля; строгость разбора — модуля."""

    try:
        data = json.loads(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {"picks": data.get("match", []), "search_terms": data.get("search_terms", [])}


async def _ask(question: str, pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], str] | None:
    """Выбор LLM по пулу кандидатов: (строки «та услуга» в порядке пула, термины, код причины пустого
    выбора) или None — сбой; код причины есть только у промпта модуля (`on`)."""

    mode = _choose_mode()
    if mode == "off":
        return await _ask_directly(question, pool)
    options = [Option(id=str(index), text=f"{_row_name(row)} — {_row_cost(row)} руб.") for index, row in enumerate(pool)]
    legacy = (
        LegacyWire(prompt=lambda q, _opts: _build_prompt(q, pool), answer=_legacy_answer, fmt="json", options={"temperature": 0})
        if mode == "legacy"
        else None
    )
    choice = await choose(PRICE_TASK, question, options, legacy=legacy)
    if choice.failed:  # сбой и мусор — решение политики (L-13) или прежний ответ правил
        return None
    return [pool[index] for index in sorted(int(i) for i in choice.ids)], list(choice.terms), choice.reason


async def _ask_directly(question: str, pool: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], str] | None:
    """Прежний вызов и разбор (`choose_price = off`); уходит, когда модуль принят на стенде."""

    try:
        raw = await generate_text(
            _build_prompt(question, pool),
            timeout_s=_LLM_TIMEOUT_S,
            queue_timeout_ms=_LLM_QUEUE_TIMEOUT_MS,
            fmt="json",
            think=False,
            # Выбор, а не сочинение: одинаковый вопрос — одинаковые строки.
            options={"temperature": 0},
        )
    except Exception as exc:  # fail-open: любой сбой — ответ правил
        logger.warning("price_llm_select_failed: %s", type(exc).__name__)
        return None
    parsed = _parse_reply(raw, len(pool))
    if parsed is None:
        logger.warning("price_llm_select_bad_reply: %r", str(raw or "")[:200])
        return None
    match, terms = parsed
    return [pool[i - 1] for i in sorted(match)], terms, ""


# Условия, которые меняют саму услугу для пациента. «Комплекс/программа» сюда не входят:
# «УЗИ органов брюшной полости (комплексное)» — это основная услуга, а не пакет
# (свип 01.10: фильтр отбросил её и оставил «УЗИ печени и желчного пузыря»).
_HARD_VARIANT_FLAGS = frozenset({"cito", "capillary", "child", "home", "repeat", "kmn"})


def _prefer_base_variants(rows: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """Без cito/капиллярной/к.м.н./на дому…, если пациент их не просил и есть базовый вариант."""

    requested = _query_nonbase_price_flags(question)
    base = [row for row in rows if not ((_row_nonbase_price_flags(row) & _HARD_VARIANT_FLAGS) - requested)]
    return base or rows


def _journal_path() -> Path:
    return ensure_dir(DATA_DIR / "journals", "journals") / _JOURNAL_NAME


def _record_synonym_suggestion(phrase: str, terms: list[str], rows: list[dict[str, Any]], *, source: str = "") -> None:
    """Пациент назвал услугу словом, которого в прайсе нет.

    Это кандидат в синонимы МИС: клиника проверит и заведёт сама (поле синонимов МИС
    бот уже читает — синоним работает без правки кода). `found` — нашла ли услугу LLM
    по термину; промахи («спирометрия», а в прайсе «Спирография») клинике нужнее всего:
    без синонима их не найдёт никто. Пишем только фразу услуги — не всю реплику.

    `source` (под `turn_parse = on`, L-15b шаг 5) — откуда запись: `price` (выбор строки одной услуги),
    `basket` (слабая позиция корзины), `readiness` (слабая фраза вопроса о сроке). Поле новое и только
    под `on`: строки журнала без него читаются по-прежнему.
    """

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phrase": phrase,
        "found": bool(rows),
        "terms": terms,
        "services": [
            {"name": _row_name(row), "homecode": str(row.get("serviceHomecode") or row.get("homecode") or "").strip()}
            for row in rows
        ],
    }
    if source:
        entry["source"] = source
    try:
        with _journal_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:  # журнал — подсказка клинике, не повод ломать ответ
        logger.warning("price_synonym_journal_failed: %s", exc)


def _journal_phrase(question: str, rules_phrase: str) -> tuple[str, str]:
    """Что писать в журнал синонимов: (фраза, источник). Под `turn_parse = on` (L-15b, шаг 5) — фраза,
    которую выбрала LLM разбора хода (`ParsedPhrase.text`, срез реплики), а не вырезатель правил; разбор
    «ни одна фраза не услуга» (`none_fit`) — ничего: клинике не синоним. Разбор без выбора LLM (одна
    фраза-кандидат, данные, сбой) и прочие режимы — прежняя фраза правил, источник пустой."""

    parse = turn_parse.acting_for(question)
    if parse is None:
        return rules_phrase, ""
    if parse.status == turn_parse.ParseStatus.NONE_FIT:
        return "", "price"
    if parse.status == turn_parse.ParseStatus.PICKED and len(parse.items) == 1:
        return parse.items[0].text, "price"
    return rules_phrase, "price"


async def select_price_rows(
    question: str,
    retail_rows: list[dict[str, Any]],
    *,
    candidates: Sequence[dict[str, Any]] = (),
    shown: Sequence[dict[str, Any]] | None = None,
    hints: Sequence[str] = (),
    context: str = "",
    runtime_llm_mode: str = "",
) -> PriceSelection | None:
    """Строки прайса, которые являются той услугой, о которой спросил пациент.

    :param question: реплика пациента в этом ходе (как написал он сам)
    :param retail_rows: строки прайса региона
    :param candidates: ответ правил — всегда в списке для LLM первым, чтобы его можно
        было подтвердить («оак» → «Общий анализ крови», а не только строки со словом «ОАК»)
    :param shown: строки, которые правила покажут пациенту (политика L-13: список из ≥2 строк при
        недоступной LLM остаётся списком правил); None — правила покажут всех кандидатов
    :param hints: подсказки только для поиска кандидатов — выбор правил
    :param context: сущность NLU / услуга из диалога — тоже только для поиска
        кандидатов; в подсказку LLM не идёт
    :param runtime_llm_mode: strict|hybrid|rich; в strict LLM не зовём
    :return: выбор LLM; None — оставить ответ правил
    """

    if not _ENABLED or str(runtime_llm_mode or "").strip().lower() == "strict":
        return None
    question = str(question or "").strip()
    rows = [row for row in retail_rows if isinstance(row, dict)]
    if not question or not rows:
        return None
    # Ответ цифрой, «да», «а сколько стоит?» — своих слов об услуге нет, её держит контекст.
    if not price_question_names_service(question):
        return None
    # «по ОМС» и другие уточнения, которых в прайсе нет вовсе: правила честно говорят
    # «не нашёл» (решение владельца, П2) — не даём LLM найти «почти то». Но если
    # незнакомы ВСЕ слова об услуге («гастроскопия»), это не уточнение, а услуга
    # другим словом — её и ищет LLM (правило 3 промпта само отсекает «по ОМС»).
    if query_has_unsatisfiable_qualifier(question, rows) and not query_names_service_in_other_words(question, rows):
        return None

    phrase = str(_extract_price_service_from_query(question) or "").strip()
    phrase_norm = _normalise_catalog_text(phrase)
    journal_phrase, journal_source = _journal_phrase(question, phrase or question)
    synonymized = _apply_service_synonyms(question)
    texts = [question, synonymized, phrase, str(context or ""), *[str(h or "") for h in hints]]
    # По отдельным словам — если кандидатов мало: поиск правил требует всех слов сразу,
    # и «хгч срочно» не находит ни «ХГЧ», ни «Cito ХГЧ». Шум отсеет выбор LLM.
    words = [token for token in _price_query_tokens(question) if len(token) >= 3]
    # Ответ правил — первым, но не весь: длинный семейный список (до 50 строк) иначе
    # занимал весь список LLM, и лучшая находка поиска по словам в него не попадала
    # («панель женские наследственные» → чужие панели Геномеда, свип 01.10).
    candidates = [row for row in candidates if isinstance(row, dict)]
    shown = candidates if shown is None else [row for row in shown if isinstance(row, dict)]
    # Не `shown`: это первые кандидаты для LLM, а политика решает по строкам, которые увидит
    # пациент, — параметру `shown` (BUG-2026-10-13-L13-SHOWN-SHADOWED).
    proposal_first, rest = candidates[:_PROPOSAL_FIRST], candidates[_PROPOSAL_FIRST:]
    # Поиск по прайсу — CPU на сотни миллисекунд: не держим им цикл событий.
    pool = await asyncio.to_thread(
        _build_pool, texts, rows, per_text=_PER_QUERY_LIMIT, first=proposal_first, fallback_texts=words
    )
    pool += [row for row in rest if _row_key(row) not in {_row_key(r) for r in pool}][: max(0, _POOL_LIMIT - len(pool))]
    policy = match_policy()
    strong = strong_rows(question, rows, pool=pool)
    if strong and policy != "off":
        turn_telemetry.note(price_policy="strong")
        if policy == "on":
            return PriceSelection(rows=tuple(strong), status="strong")
    record = policy != "off" and not strong  # решение политики ещё не записано в телеметрию

    def settle(legacy: PriceSelection | None, decided: PriceSelection | None, label: str = "") -> PriceSelection | None:
        """Под on — решение политики (None — прежний ответ правил); под off / shadow — прежний
        результат, тень пишет решение в телеметрию."""

        if record:
            turn_telemetry.note(price_policy=label or _policy_label(decided))
        return decided if policy == "on" else legacy

    def unavailable() -> tuple[PriceSelection | None, str]:
        # Правила ничего не предложили — чужой цены не было, их честный ответ остаётся.
        if not candidates:
            return None, "rules"
        # Правила покажут список — при неоднозначности список, а не одна цена (решение 22.07).
        if len(shown) >= 2:
            return None, "rules_list"
        return PriceSelection(rows=(), status="unavailable", names=_whole_word_names(question, pool, rows)), ""

    def none_fit(reason: str) -> PriceSelection | None:
        # Названо слишком общее или не услуга — уточнение правил честнее, чем «не нашёл».
        if not candidates or reason in _NOT_A_SERVICE_NAME:
            return None
        return PriceSelection(rows=(), status="none_fit")

    def journal_miss(found_terms: list[str], *reasons: str) -> None:
        # Промах — услугу не нашли ни правила (`candidates` пуст), ни LLM: такое слово пациента —
        # кандидат в синонимы для клиники (см. `_record_synonym_suggestion`). Но только если LLM
        # признала в реплике название услуги: «слишком общее» и «не услуга» в любом из кругов —
        # не синоним (BUG-2026-10-13-SYNONYM-JOURNAL-IGNORES-REASON). Причину даёт только промпт
        # модуля (`choose_price = on`); прежний промпт её не знает — там пишем, как раньше.
        if not candidates and not _NOT_A_SERVICE_NAME.intersection(reasons) and journal_phrase:
            _record_synonym_suggestion(journal_phrase, found_terms, [], source=journal_source)

    # Дословное название позиции («ферритин» = «Ферритин»): правила не ошибаются.
    if phrase_norm and any(_normalise_catalog_text(_row_name(row)) == phrase_norm for row in pool):
        return None

    first = await _ask(question, pool)
    if first is None:
        return settle(None, *unavailable())
    chosen, terms, reason = first
    if chosen:
        picked = PriceSelection(rows=tuple(_prefer_base_variants(chosen, question)))
        return settle(picked, picked)
    if not terms:
        journal_miss([], reason)
        return settle(None, none_fit(reason))

    pool_by_terms = await asyncio.to_thread(_build_pool, terms, rows, per_text=_PER_TERM_LIMIT)
    second = await _ask(question, pool_by_terms)
    if second is None:
        return settle(None, *unavailable())
    if not second[0]:
        journal_miss(terms, reason, second[2])
        return settle(None, none_fit(second[2]))
    chosen = _prefer_base_variants(second[0], question)
    if journal_phrase:
        _record_synonym_suggestion(journal_phrase, terms, chosen, source=journal_source)
    picked = PriceSelection(rows=tuple(chosen), search_terms=tuple(terms))
    return settle(picked, picked)


def rows_overlap(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> bool:
    """Есть ли общие строки прайса (сравнение по названию, коду и цене, не по объекту)."""

    right_keys = {_row_key(row) for row in right if isinstance(row, dict)}
    return any(_row_key(row) in right_keys for row in left if isinstance(row, dict))


def reconcile_with_proposal(
    proposal: Sequence[dict[str, Any]] | None,
    selection: PriceSelection | None,
) -> list[dict[str, Any]] | None:
    """Что показать пациенту: ответ правил или строки, выбранные LLM.

    - LLM подтвердила ответ правил — он и остаётся (одна строка — с врачами, как было);
    - из списка правил остаются строки, которые LLM признала той услугой, плюс найденные
      ею варианты вне списка; чужие строки уходят;
    - LLM не признала ни одной строки правил — показываем её выбор;
    - выбора нет — ответ правил: «не нашла» от LLM слабее, чем найденное правилами.

    :param proposal: строки, которые показали бы правила
    :param selection: выбор LLM
    :return: None — показать ответ правил как есть; иначе строки к показу
    """

    if selection is None or not selection.rows:
        return None
    proposed = [row for row in (proposal or []) if isinstance(row, dict)]
    if not proposed:
        return list(selection.rows)
    selected_keys = {_row_key(row) for row in selection.rows}
    matched = [row for row in proposed if _row_key(row) in selected_keys]
    if not matched:
        return list(selection.rows)
    matched_keys = {_row_key(row) for row in matched}
    extras = [row for row in selection.rows if _row_key(row) not in matched_keys] if len(proposed) > 1 else []
    result = matched + extras
    if [_row_key(row) for row in result] == [_row_key(row) for row in proposed]:
        return None
    return result
