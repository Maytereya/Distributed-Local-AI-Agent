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
     («гастроскопия» → «ФГДС»); по ним (целиком, а мало — по их словам) второй круг поиска и
     второй выбор; пустой второй круг — «не нашёл» без вызова LLM. Такой термин
     пишется в журнал: синонимы заносит клиника, не мы (решение владельца 30.09);
  4) особые варианты (cito, капиллярная кровь, к.м.н., на дому…), которых пациент не
     просил, отбрасываются, если среди выбранных есть базовый.

Отказоустойчивость: сбой, таймаут, мусор, режим strict, kill-switch → None, и инструмент
отвечает как раньше. «Не нашла» тоже None: LLM заменяет ответ правил только тем, что
положительно выбрала (`reconcile_with_proposal`).

Выбор — на модуле `choose()` (ход (в), L-12): роли system/user, JSON Schema, коды причин (промпт
`choose_price_service`). Единственный путь с 17.10: выключатель `[MESSENGER_ROUTER] choose_price` (режимы `off` —
прежний вызов и разбор, `legacy` — прежний промпт `price_service_select` через модуль) и прежний промпт сняты
(STEP-2026-10-17-ONE-PATH-CHOOSE-PRICE; на проде `on` с 16.10).

Политика силы совпадения (L-13, решение владельца 07.10 — вариант B; единственный режим с 16.10 — T7 шаг 2,
выключатель `price_match_policy` и режимы off/shadow сняты, решение владельца):
сильное совпадение по данным (дословное название, синоним МИС, код — `_strong_match`) — только
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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from agent_logic_2.persist import DATA_DIR, ensure_dir

from .. import llm_runtime, turn_parse, turn_telemetry
from ..choice import Option, Task, choose
from ..runtime_config import config as _cfg
from ..russian_nlu import tokens_share_stem
from . import _biomaterial, _price_by_parse
from ._common import _normalise_catalog_text
from ._common import _normalise_input
from ._prices_helpers import (
    _PRICE_QUERY_STOPWORDS,
    _PRICE_TOKEN_RE,
    _apply_service_synonyms,
    _extract_price_service_from_query,
    _normalise_price_token,
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
# Та же задача с уточнённым правилом 3 промпта (BUG-2026-10-17-P2-JUDGE-COLLECTION-CONDITIONS): вопрос о сдаче, подготовке,
# записи, сроке, «условиях сбора» — не условие услуги. Под `price_judge_conditions = on`; своё имя — свой промпт, своё место в
# телеметрии и на стенде.
PRICE_CONDITIONS_TASK = replace(PRICE_TASK, name="price_service_conditions")
# Суд условия (гард П2 сработал, вопрос судит LLM; выключатель `price_candidates_by_meaning` снят 22.10): правило 3 называет условием и
# вариант выполнения, от которого зависит цена (срочно / экспресс, под наркозом, на дому), — карточка dev 20.10: «ФКС с
# наркозом» → 8 100 руб. без наркоза, «экспресс 17-ОН день в день» → обычная цена. Уточнение `price_judge_conditions`
# («как сдать, подготовиться…» — не условие) входит в тот же текст: строгое правило без него отсекало бы вопросы о сдаче.
PRICE_QUALIFIER_TASK = replace(PRICE_TASK, name="price_service_qualifier")
# Те же две задачи с правилами BUG-2026-10-24-FOREIGN-PRICE-D17 на пути ответа о цене (`price_select_rules`): `general` — слишком
# общее название («анализы», приём врача без специальности) — не группа услуг правила 5. Свои промпты.
_RULES_TASKS = {
    mode: (replace(PRICE_TASK, name=f"price_service_{mode}"), replace(PRICE_TASK, name=f"price_service_qualifier_{mode}"))
    for mode in ("general",)
}
_VARIANT_B_LIMIT = 5


def _select_rules() -> str:
    """`[MESSENGER_ROUTER] price_select_rules = off | general`; мусор — off (как было)."""

    mode = str(getattr(_cfg, "MR_PRICE_SELECT_RULES", "off") or "").strip().lower()
    return mode if mode in _RULES_TASKS else "off"


def general_clarify_on() -> bool:
    """`[MESSENGER_ROUTER] price_general_clarify = on` (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 3): «слишком общее» на пути
    цены — уточнение, несколько разных услуг выбора — список; мусор — off (как было)."""

    return str(getattr(_cfg, "MR_PRICE_GENERAL_CLARIFY", "off") or "").strip().lower() == "on"


def price_task(*, judge_qualifier: bool = False, price_answer: bool = True) -> Task:
    """Задача выбора строки прайса: суд условия — свой промпт; под `[MESSENGER_ROUTER] price_judge_conditions = on` — с
    уточнённым правилом 3; на пути ответа о цене (`price_answer`) под `price_select_rules` — промпты с правилами (уточнённое
    правило 3 `price_judge_conditions` в них не входит: на проде он off). Суд записи (`price_answer=False`) — прежний промпт:
    «запись к неврологу» правилом приёма без специальности не судится."""

    rules = _RULES_TASKS.get(_select_rules()) if price_answer else None
    if judge_qualifier:
        return rules[1] if rules else PRICE_QUALIFIER_TASK
    on = str(getattr(_cfg, "MR_PRICE_JUDGE_CONDITIONS", "off") or "").strip().lower() == "on"
    if on:
        return PRICE_CONDITIONS_TASK
    return rules[0] if rules else PRICE_TASK
# Причины пустого выбора, при которых LLM не признала в реплике названия услуги: «Цена анализа»
# (слишком общее), «сможете посчитать» (не услуга). Ни «не нашёл» пациенту, ни синоним клинике.
_NOT_A_SERVICE_NAME = frozenset({"too_general", "not_a_service"})


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
    reason: str = ""  # код причины пустого выбора (`none_fit`)


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


_TOKEN_NAMES_CACHE: dict[tuple[int, str], dict[str, frozenset[str]]] = {}


def _token_names(retail_rows: list[dict[str, Any]]) -> dict[str, frozenset[str]]:
    """В каких названиях прайса встречается слово (кэш по отпечатку среза, как у `_token_df`)."""

    key = (len(retail_rows), _row_name(retail_rows[0])[:48] if retail_rows else "")
    cached = _TOKEN_NAMES_CACHE.get(key)
    if cached is None:
        index: dict[str, set[str]] = {}
        for name in {_row_name(row) for row in retail_rows}:
            for token in set(_price_query_tokens(name)):
                index.setdefault(token, set()).add(name)
        cached = {token: frozenset(names) for token, names in index.items()}
        _TOKEN_NAMES_CACHE.clear()
        _TOKEN_NAMES_CACHE[key] = cached
    return cached


def _wordform_family_size(word: str, retail_rows: list[dict[str, Any]], *, same: Callable[[str, str], bool] | None = None) -> int:
    """Во скольких названиях прайса есть слово вместе со словоформами (`same`, по умолчанию `_same_word`)."""

    same = same or _same_word
    names: set[str] = set()
    for token, owners in _token_names(retail_rows).items():
        if same(word, token):
            names |= owners
    return len(names)


def _variant_b_wordforms_on() -> bool:
    """`[MESSENGER_ROUTER] variant_b_wordforms = off | on` (облако 16.10; по умолчанию on с 17.10 — STEP-2026-10-17-VBW-ON);
    мусор в значении — как было (off)."""

    return str(getattr(_cfg, "MR_VARIANT_B_WORDFORMS", "on") or "").strip().lower() == "on"


# Словоформа — одно слово (`russian_nlu.tokens_share_stem`, одно определение «одно ли это слово»): порог 6 — как у гарда
# отображения цены; «сифилис» = «сифилиса», но «гастроскопию» ≠ «гастрин» («гастро» / «гастри»). Только слова из одних
# букв: служебные токены кода («витамин_a», «витамин_b12») и коды с цифрами сравниваются целиком.
_WORDFORM_PREFIX = 6


def _same_word(a: str, b: str) -> bool:
    return a == b or (a.isalpha() and b.isalpha() and tokens_share_stem(a, b, length=_WORDFORM_PREFIX))


# Окончание словоформы — не длиннее трёх букв («-ами», «-ого»): у длинного слова общее начало словоформ — всё, кроме него.
_WORDFORM_ENDING = 3


def _same_word_strict(a: str, b: str) -> bool:
    """Словоформа синонима МИС (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 1): общее начало — длинное из двух слов без
    окончания, но не меньше порога `_same_word`. Шести букв мало длинному слову: «коагулограмма» и «коагуляция», «креатинина»
    и «креатинфосфокиназа», «прогестерон» и «прогестин» — разные слова (свип 24.10 по 1 068 вопросам эталона), «коагулограмма»
    и «коагулограммы» — одно; у слов до 9 букв — как `_same_word` («терапевт» = «терапевта»)."""

    if a == b:
        return True
    length = max(_WORDFORM_PREFIX, max(len(a), len(b)) - _WORDFORM_ENDING)
    return a.isalpha() and b.isalpha() and tokens_share_stem(a, b, length=length)


def _whole_word_names(question: str, pool: list[dict[str, Any]], retail_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Вариант B (L-13): названия строк, в которых есть самое различающее слово пациента целиком.

    Различающее — самое редкое в названиях прайса: родовые слова («анализ», «тест», «приём») часты и
    отпадают сами, без списка слов. По началу слова нельзя: на «гастроскопию» первым оказался бы
    «Гастрин»; слова нет в прайсе вовсе — названий нет (честный перевод). Порядок — как у кандидатов.

    Под `variant_b_wordforms = on`: различающее слово могло стать редким из-за словоформы — «сифилис» целиком есть только
    в чужой строке («Кровь на сифилис (…)»), а в верной — «сифилиса» («Экспресс-тест для определения сифилиса (антитела
    IgG и IgM …)»). Строка со словоформой различающего слова добавляется, только если покрывает слов пациента строго
    больше любой строки с ним целиком; порядок — по числу слов пациента в названии (словоформы — одно слово), при
    равенстве — как у кандидатов. Ключевое слово и «слова нет в прайсе — названий нет» прежние. Словоформы добавляют
    строки, только если слово вместе с ними различает (`_wordform_names`).
    """

    wanted = set(_price_query_tokens(question))
    if not wanted:
        return ()
    df = _token_df(retail_rows)
    key = min(wanted, key=lambda token: (df.get(token, 0), token))
    if not df.get(key):
        return ()
    if _variant_b_wordforms_on():
        return _wordform_names(wanted, key, pool, retail_rows)
    names = [_row_name(row) for row in pool if key in set(_price_query_tokens(_row_name(row)))]
    return tuple(dict.fromkeys(name for name in names if name))[:_VARIANT_B_LIMIT]


def _wordform_names(wanted: set[str], key: str, pool: list[dict[str, Any]], retail_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Строки варианта B под `variant_b_wordforms = on`. Словоформы ключевого слова — только когда слово вместе с ними
    различает: есть не больше чем в `_VARIANT_B_LIMIT` названиях прайса (список покажет их все). Иначе редкость слова —
    артефакт словоформы, а само слово общее (BUG-2026-10-17-VARIANT-B-COMMON-WORD: «стоимость консультации врача» —
    «консультации» дословно в одном названии, со словоформой «консультация» — в 163 приёмах; первые пять кандидатов —
    приёмы хирургов): строк по словоформе нет, как без словоформ. Порядок по числу слов пациента — всегда."""

    tokens = {name: set(_price_query_tokens(name)) for name in dict.fromkeys(_row_name(row) for row in pool) if name}

    def cover(name: str) -> int:
        return sum(1 for word in wanted if any(_same_word(word, token) for token in tokens[name]))

    exact = [name for name in tokens if key in tokens[name]]
    best = max((cover(name) for name in exact), default=0)
    distinguishes = _wordform_family_size(key, retail_rows) <= _VARIANT_B_LIMIT
    forms = [name for name in tokens if distinguishes and name not in exact and any(_same_word(key, t) for t in tokens[name]) and cover(name) > best]
    ranked = sorted([*forms, *exact], key=cover, reverse=True)  # устойчиво: при равенстве — порядок кандидатов
    return tuple(ranked[:_VARIANT_B_LIMIT])


def _budget_clarify_on() -> bool:
    """`[MESSENGER_ROUTER] price_budget_clarify = on`: отказ лимита решений хода на выборе строки — уточнение, не ответ правил."""

    return str(getattr(_cfg, "MR_PRICE_BUDGET_CLARIFY", "off") or "").strip().lower() == "on"


def _policy_label(selection: PriceSelection | None) -> str:
    if selection is None:
        return "rules"
    if selection.status == "unavailable":
        return "names" if selection.names else "handoff"
    return {"none_fit": "not_found"}.get(selection.status, selection.status)


# Кандидаты по редким словам: слово встречается не больше чем в стольких названиях прайса (или нет ни в одном —
# словоформа, «карбамазепина»), берём не больше `_RARE_WORDS` слов и `_PER_RARE_WORD` строк на слово.
_RARE_DF = 8
_RARE_WORDS = 3
_PER_RARE_WORD = 4


def _rare_words(texts: Sequence[str], retail_rows: list[dict[str, Any]], *, short_from: str = "") -> list[str]:
    """Самые редкие слова реплики (или терминов) в прайсе — сначала известные редкие, затем незнакомые прайсу
    (словоформы и сокращения): поиск по всем словам сразу отдаёт строку, только если совпали все слова, и пул забивают
    строки по частым словам («анализ крови на концентрацию карбамазепина» — ни одной строки «Карбамазепин»; диагностика
    п. 1, 19.10). Сокращения — ещё и из `short_from` (реплика как написал пациент): фраза правил приходит строчными,
    и «ЛГ» в ней уже не отличить от предлога."""

    df = _token_df(retail_rows)
    words: list[str] = []
    for text in texts:
        for token in _price_query_tokens(str(text or "")):
            if token not in words and not token.isdigit():
                words.append(token)
    short = [w for text in (short_from, *texts) for w in _short_name_words(str(text or "")) if w not in words]
    known = sorted((w for w in words if 0 < df.get(w, 0) <= _RARE_DF), key=lambda w: df[w])
    unknown = [w for w in words if w not in df and len(w) >= 4]
    return list(dict.fromkeys(short + known + unknown))[:_RARE_WORDS]


def _named_service_texts(question: str, phrase: str, hints: Sequence[str]) -> list[str]:
    """Откуда брать редкие слова: только то, что пациент назвал услугой, — фраза упоминания хода (L-14), без неё —
    фраза правил; понимание хода «услуги не названо» — ничего. Не вся реплика: «готов», «коду», «озвучить», «сердца»
    тянули в начало списка чужие строки («Консультация готовых гистологических стёкол», «Здоровье сердца и сосудов!»),
    и LLM выбирала их (карточка dev 20.10). Подсказки (`hints`) — названия услуг из выбора правил, их оставляем."""

    mentioned = _price_by_parse.phrase(question) if _price_by_parse.enabled() else _mention_phrase(question)
    named = phrase if mentioned is None else mentioned
    return [text for text in (named, *[str(h or "") for h in hints]) if str(text or "").strip()]


def _with_rare_tail(pool: list[dict[str, Any]], rare: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки по редким словам — в списке, но в его конце: место в начале списка LLM принимала за подсказку («ЭхоКГ узи
    сердца» → «Здоровье сердца и сосудов!» первым номером при «Эхокардиографии» вторым, карточка dev 20.10). Полный пул
    уступает им последние места — иначе строку «Карбамазепин» вытесняют частые слова (диагностика 19.10)."""

    keys = {_row_key(row) for row in pool}
    missing = [row for row in rare if _row_key(row) not in keys]
    if not missing:
        return pool
    room = max(0, _POOL_LIMIT - len(missing))
    return [*pool[:room], *missing][:_POOL_LIMIT]


def _rare_rows(texts: Sequence[str], retail_rows: list[dict[str, Any]], *, short_from: str = "") -> list[dict[str, Any]]:
    """Строки прайса по каждому редкому слову отдельно (кандидаты, не ответ: выбирает LLM)."""

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for word in _rare_words(texts, retail_rows, short_from=short_from):
        for row in _rank_price_rows(retail_rows, word, limit=_PER_RARE_WORD):
            key = _row_key(row)
            if key not in seen:
                seen.add(key)
                out.append(row)
    return out


def _strong_by_parse(question: str, rows: list[dict[str, Any]], pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Сильное совпадение под `price_by_parse = on` (цена, этап 2, Ш1): по фразе разбора этой реплики, а не по реплике,
    срезанной вырезателем. Разбор решил, что услуги нет, — сильного нет; разбора этой реплики нет (вход — не реплика хода,
    сбой LLM) — вход как есть: сильна только реплика, которая сама — название, синоним или код."""

    named = _price_by_parse.phrase(question)
    probe = question if named is None else named
    return strong_rows(probe, rows, pool=pool) if probe else []


def names_service(question: str) -> bool:
    """Назвал ли пациент в реплике услугу — одно определение для планировщика (не переспрашивать название) и выбора
    строки (звать ли LLM). Короткое сокращение с заглавной («ЛГ», «аТ ТГ») — тоже
    слова об услуге: её ищет LLM. До 20.10 это знал только выбор строки, а планировщик по прежнему определению
    переспрашивал «Скажите название услуги» раньше него (test-срез 20.10: «ЛГ стоимость», «Цена аТ ТГ»)."""

    if _price_by_parse.enabled():
        # Цена, этап 2, Ш3: одно решение хода — понимание этой реплики (`TurnUnderstanding.names_service`, ставит производитель:
        # фраза разбора есть — названа). Строчное «лг» буквы не видят (`_short_name_words` ждёт заглавной), и выбор строки не звался:
        # список правил с чужой строкой при работающей LLM. Понимания этой реплики нет — прежнее определение.
        from .. import turn_understanding  # понимание хода импортирует выбор строки — импорт на вызове

        understood = turn_understanding.of_reply(question)
        if understood is not None and understood.names_service is not None:
            return understood.names_service
    return price_question_names_service(question) or bool(_short_name_words(question))


# Строк по синонимам МИС внутри названной услуги — не больше стольких (у неоднозначного синонима их до 9: «вэб»).
_SYNONYM_ROWS = 6


def _synonym_rows(texts: Sequence[str], retail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки прайса услуг, чей синоним МИС входит в названную услугу (`mis_synonyms_within`), — кандидаты для LLM:
    «ЭхоКГ» клиника записала синонимом «Эхокардиографии», а буквами её строку не найти (g0501, 20.10).

    Только синонимы, которых нет в названиях прайса ни одним словом, ни словоформой, ни склеенными: их строку буквы не
    находят. Синоним из слова прайса («узи» → «УЗИ брюшной полости», «igg» → ЭБВ, «биохимия» → «Мочевая кислота»,
    «терапевт») буквы и так находят во всех его строках, а первым номером он ставил чужую строку — встречный свип 20.10
    на 335 размеченных вопросах стенда: без условия 27 чужих первых строк из 112, с ним 0 из 27."""

    df = _token_df(retail_rows)

    # Словоформа — строгая: «коагулограмма» не словоформа «коагуляции» (14 названий), её строки буквы не находят
    # (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 1; выключатель `synonym_wordforms` снят 31.10).
    def letters_miss(key: str) -> bool:
        glued = _normalise_price_token(_normalise_input(key).replace(" ", "").replace("-", ""))
        return not df.get(glued) and not any(
            _wordform_family_size(token, retail_rows, same=_same_word_strict) for token in _price_query_tokens(key)
        )

    wanted: list[str] = []
    for text in texts:
        # Синоним в другой словоформе («коагулограммы») — тот же довод клиники, что дословный (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 1).
        found = (
            *_biomaterial.mis_synonyms_within(str(text or "")),
            *_biomaterial.mis_synonyms_within_wordforms(str(text or ""), _same_word_strict),
        )
        for key, names in found:
            if not letters_miss(key):
                continue
            for name in names:
                norm = _normalise_catalog_text(name)
                if norm and norm not in wanted:
                    wanted.append(norm)
    if not wanted:
        return []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in retail_rows:
        by_name.setdefault(_normalise_catalog_text(_row_name(row)), []).append(row)
    return [row for key in wanted for row in by_name.get(key, ())][:_SYNONYM_ROWS]


def _short_name_words(text: str) -> list[str]:
    """Сокращения в две буквы, которые общий разбор цены отбрасывает как шум («ЛГ», «аТ ТГ»): написаны хотя бы с одной
    заглавной (предлоги «за», «из» пишут строчными) — без чисел и слов-вопросов."""

    out: list[str] = []
    for raw in _PRICE_TOKEN_RE.findall(str(text or "")):
        if len(raw) != 2 or raw.isdigit() or not any(ch.isupper() for ch in raw):
            continue
        token = _normalise_price_token(_normalise_input(raw))
        if len(token) == 2 and token not in _PRICE_QUERY_STOPWORDS and token not in out:
            out.append(token)
    return out


# Пометка МИС у позиции — не длиннее этого (знаков) и не больше двух названий каждого вида: длинный перечень материалов
# (до 1363 знаков у одной строки) сбивал выбор (прямой вызов 19.10). Все пометки списка — не длиннее `_MIS_NOTES_POOL_MAX`
# (в порядке списка: ответ правил первым): «кровь на ТТГ» пометила бы «кровь из вены» у каждой строки, а промпт — время LLM.
_MIS_NOTE_MAX = 120
_MIS_NOTE_ITEMS = 2
_MIS_NOTES_POOL_MAX = 1200


def _named_by_patient(texts: Sequence[str], wanted: set[str]) -> list[str]:
    """Названия МИС (материалы или синонимы), в которых больше всего слов пациента; ни одного — пусто. Слово — одно
    с учётом словоформы (`_same_word`)."""

    scored = []
    for text in texts:
        tokens = set(_price_query_tokens(text))
        scored.append((sum(1 for word in wanted if any(_same_word(word, token) for token in tokens)), text))
    best = max((score for score, _text in scored), default=0)
    return [text for score, text in scored if best and score == best][:_MIS_NOTE_ITEMS]


def _mis_note(row: dict[str, Any], wanted: set[str], vocab: Any = None) -> str:
    """Что МИС знает о строке и назвал пациент (BUG-2026-10-19-PRICE-ROUND-TWO-POOL, g0542): «Посев на микрофлору (…)»
    без материала в названии LLM отвергала на «мазок … с задней стенки глотки», хотя по `biomatNames` строка этот
    материал принимает. Только совпавшее со словами реплики: прочие материалы и синонимы судить не помогают, а длинный
    список сбивал выбор. Пусто — данных нет или пациент их не назвал: текст позиции как раньше."""

    if not wanted:
        return ""
    materials, synonyms = (_named_by_patient(texts, wanted) for texts in _biomaterial.service_mis_names(_row_name(row), vocab=vocab))
    # Не влезает — меньше названий, затем без синонима (материал — проверка правила 2 промпта), затем только синоним;
    # названия не режутся.
    for m, n in ((_MIS_NOTE_ITEMS, _MIS_NOTE_ITEMS), (1, 1), (1, 0), (0, 1)):
        parts = [f"{label} — {', '.join(items)}" for label, items in (("материал", materials[:m]), ("другое название", synonyms[:n])) if items]
        note = "по справочнику клиники: " + "; ".join(parts) if parts else ""
        if note and len(note) <= _MIS_NOTE_MAX:
            return note
    return ""


def _option_text(row: dict[str, Any], note: str) -> str:
    text = f"{_row_name(row)} — {_row_cost(row)} руб."
    return f"{text} ({note})" if note else text


def _options(question: str, pool: list[dict[str, Any]]) -> list[Option]:
    """Позиции для LLM: «название — цена» с пометкой МИС (`_mis_note`) в пределах `_MIS_NOTES_POOL_MAX` на список; число
    помеченных — в телеметрию хода (`price_mis_notes`). Выключатель `price_option_mis_notes` снят 22.10 (неделя `on` на проде)."""

    wanted = set(_price_query_tokens(question))
    vocab = _biomaterial._vocabularies() if wanted else None  # один раз на список: у словаря — проверка среза на диске
    notes, budget = [], _MIS_NOTES_POOL_MAX
    for row in pool:
        note = _mis_note(row, wanted, vocab)
        note = note if len(note) <= budget else ""
        budget -= len(note)
        notes.append(note)
    if wanted:  # за ход: оба круга выбора
        turn_telemetry.bump("price_mis_notes", sum(1 for note in notes if note))
    return [Option(id=str(index), text=_option_text(row, note)) for index, (row, note) in enumerate(zip(pool, notes))]


async def _ask(
    question: str, pool: list[dict[str, Any]], *, judge_qualifier: bool = False, price_answer: bool = True
) -> tuple[list[dict[str, Any]], list[str], str] | None:
    """Выбор LLM по пулу кандидатов: (строки «та услуга» в порядке пула, термины, код причины пустого
    выбора) или None — сбой."""

    options = _options(question, pool)
    choice = await choose(price_task(judge_qualifier=judge_qualifier, price_answer=price_answer), question, options)
    if choice.failed:  # сбой и мусор — решение политики (L-13) или прежний ответ правил
        return None
    return [pool[index] for index in sorted(int(i) for i in choice.ids)], list(choice.terms), choice.reason


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


def _mention_phrase(question: str) -> str | None:
    """Фраза одиночной услуги из понимания хода (L-14 шаг 2): упоминание этой
    реплики — его фраза (пациента или выбор LLM разбора, без речевой обвязки); понимание этой реплики без
    упоминаний — пусто: услуги пациент не назвал («Цена анализа»), клинике не синоним. None — понимания
    этой реплики нет (выключено, не ценовой ход, вход — не реплика хода) или в нём список: прежний путь. Под
    `turn_understanding_one = on` (ход (б), этап 2, U9) — только действующее понимание (`of_reply`), в тени — прежний путь."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — не на уровне модуля

    tu = turn_understanding.of_reply(question) if turn_understanding.one_enabled() else turn_understanding.current()
    if tu is None or not tu.for_text(question) or tu.is_list:
        return None
    single = tu.single
    return single.phrase if single is not None else ""


def _journal_phrase(question: str, rules_phrase: str) -> tuple[str, str]:
    """Что писать в журнал синонимов: (фраза, источник). Фраза упоминания хода (`_mention_phrase`, L-14, шаг 2;
    единственный путь с 17.10 — выключатель `journal_phrase` снят), если понимание этой реплики есть. Иначе
    (понимания нет: не ценовой ход, вход — не реплика хода, список) — фраза, которую выбрала LLM разбора хода
    (`ParsedPhrase.text`, срез реплики, L-15b шаг 5), а не вырезатель правил; разбор «ни одна фраза не услуга»
    (`none_fit`) — ничего: клинике не синоним. Разбор без выбора LLM (одна фраза-кандидат, данные, сбой) и вход
    без разбора — фраза правил.

    Под `price_by_parse = on` (цена, этап 2, Ш5) — только фраза разбора этой реплики (срез реплики, границы поставила LLM разметки
    или данные): разбор решил «услуги нет», не решён или его нет — ничего. Клинике не уходит ни фраза вырезателя правил, ни
    упоминание правил понимания: это буквы, а не слово пациента об услуге."""

    if _price_by_parse.enabled():
        return str(_price_by_parse.phrase(question) or ""), "price"
    mentioned = _mention_phrase(question)
    if mentioned is not None:
        return mentioned, "price"
    from .. import turn_understanding  # понимание хода импортирует сервисы — импорт на вызове
    parse = turn_understanding.parse_for(question)
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
    judge_qualifier: bool = False,
    one_round: bool = False,
    general_as_none_fit: bool = False,
    budget_clarify: bool = False,
    price_answer: bool = True,
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
    :param judge_qualifier: гард П2 («уточнения нет в прайсе») не отсекает реплику до LLM — её судит выбор строки
        (правило 3 промпта отсекает «по ОМС»). Только для фразы, которую LLM разбора выбрала одной услугой
        (класс A 15.10: гард по словарю каталога считал «карбамазепина» уточнением).
        Сбой LLM — ответ правил, а у них гард прежний: чужой цены нет. Пустой первый круг с кодом `qualifier_absent`
        («условия нет») — `none_fit` сразу, без второго круга (BUG-2026-10-16-QUALIFIER-ROUND-TWO-BUDGET).
    :param one_round: только первый круг (без поиска по терминам LLM): пустой выбор — `none_fit` с кодом причины
        (`reason`) при любых кандидатах и любом коде; None — только сбой. Для суда условия у сильной фразы разбора
        (BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS): второй круг искал бы «почти то», а под `turn_parse = on` упирался
        в лимит решений хода (`budget`) и превращал ответ «условия нет» в `unavailable`.
    :param general_as_none_fit: «слишком общее» и «не услуга» — тоже `none_fit` с кодом причины и названиями варианта B
        (`names`), а не None. Для записи (`_booking_service_select`): там «ответ правил» — молчаливая запись на догадку
        («массаж» → «Массаж век»), а не уточнение, как у цены (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS).
    :param price_answer: выбор отвечает на вопрос о цене — промпты с правилами `price_select_rules`; суд записи — False
        (прежний промпт: правила пути цены записи не касаются, карточка 24.10 — g0792)
    :return: выбор LLM; None — оставить ответ правил
    """

    if not _ENABLED or str(runtime_llm_mode or "").strip().lower() == "strict":
        return None
    question = str(question or "").strip()
    rows = [row for row in retail_rows if isinstance(row, dict)]
    if not question or not rows:
        return None
    # Ответ цифрой, «да», «а сколько стоит?» — своих слов об услуге нет, её держит контекст.
    if not names_service(question):
        return None
    # «по ОМС» и другие уточнения, которых в прайсе нет вовсе: правила честно говорят
    # «не нашёл» (решение владельца, П2) — не даём LLM найти «почти то». Но если
    # незнакомы ВСЕ слова об услуге («гастроскопия»), это не уточнение, а услуга
    # другим словом — её и ищет LLM (правило 3 промпта само отсекает «по ОМС»).
    guard_bypassed = False
    if judge_qualifier:
        turn_telemetry.note(price_guard="llm")
    elif query_has_unsatisfiable_qualifier(question, rows) and not query_names_service_in_other_words(question, rows):
        # Словарь прайса — повод спросить LLM, не вердикт: незнакомое прайсу слово чаще
        # сокращение или словоформа («бак посев», «ВМС», «карбамазепина»), чем «по ОМС» (п. 1 плана 19.10). Условие судит
        # правило 3 промпта; сбой LLM — ответ правил, как без выключателя (гард прежний: чужой цены нет).
        judge_qualifier = guard_bypassed = True
        turn_telemetry.note(price_guard="llm_bypass")

    phrase = str(_extract_price_service_from_query(question) or "").strip()
    journal_phrase, journal_source = _journal_phrase(question, phrase or question)
    parse_phrase = _price_by_parse.phrase(question) if _price_by_parse.enabled() else None
    synonymized = _apply_service_synonyms(question)
    texts = [question, synonymized, phrase, str(context or ""), *[str(h or "") for h in hints]]
    if parse_phrase is not None:
        # Цена, этап 2, Ш2 (слова поиска): названная услуга — фраза разбора, она уже без обвязки вопроса; реплика целиком и
        # вырезатель в поиск не идут.
        phrase = parse_phrase
        texts = [parse_phrase, _apply_service_synonyms(parse_phrase), str(context or ""), *[str(h or "") for h in hints]]
    # По отдельным словам — если кандидатов мало: поиск правил требует всех слов сразу,
    # и «хгч срочно» не находит ни «ХГЧ», ни «Cito ХГЧ». Шум отсеет выбор LLM.
    words = [token for token in _price_query_tokens(question if parse_phrase is None else parse_phrase) if len(token) >= 3]
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
    named = _named_service_texts(question, phrase, hints)
    # Строки по редким словам названной услуги — в конце списка: иначе пул из 24 строк заполняют частые слова.
    rare = await asyncio.to_thread(_rare_rows, named, rows, short_from=question)
    pool = _with_rare_tail(pool, rare)
    # Синоним МИС внутри названной услуги — первым: это слово клиники, довод сильнее букв ответа правил
    # («ЭхоКГ узи сердца»: правила предложили «…плода при многоплодной беременности», LLM выбрала его первым номером).
    by_synonym = await asyncio.to_thread(_synonym_rows, named, rows)
    if by_synonym:
        keys = {_row_key(row) for row in by_synonym}
        pool = [*by_synonym, *(row for row in pool if _row_key(row) not in keys)][:_POOL_LIMIT]
    if not _price_by_parse.enabled():
        strong = strong_rows(question, rows, pool=pool)
    else:
        # Цена, этап 2, Ш1: сильное — по фразе разбора; но не там, где выбор зовут судить условие (`judge_qualifier`: слово реплики,
        # которого нет в прайсе, или суд условия у сильной фразы). Фраза разбора бывает дословным названием без оторванного условия
        # («приём терапевта» из «… по ОМС»), и её строки — платная цена мимо условия (свип П2, карточка Ш7: 59 из 82).
        strong = [] if judge_qualifier else _strong_by_parse(question, rows, pool)
    if strong:
        turn_telemetry.note(price_policy="strong")
        return PriceSelection(rows=tuple(strong), status="strong")

    def settle(decided: PriceSelection | None, label: str = "") -> PriceSelection | None:
        """Решение политики L-13 (None — прежний ответ правил) и его метка в телеметрию хода."""

        turn_telemetry.note(price_policy=label or _policy_label(decided))
        return decided

    def unavailable() -> tuple[PriceSelection | None, str]:
        if budget_clarify and _budget_clarify_on() and llm_runtime.turn_budget_exhausted():
            # Только путь ответа о цене (`budget_clarify=True` из `prices.py`); суд записи и переспроса каталога
            # (`judge_booking_service`) держит свою политику отказа. Отказ лимита решений хода — не сбой модели: LLM в этом ходе уже ответила «не то», и ответ правил по буквам
            # давал цену чужой строки (g0577 → «Анализ крови на аминокислоты»). Уточнение вместо букв (решение владельца
            # 22.10, DESIGN-2026-10-22-TURN-DECISION-BUDGET, вариант А): названия целым словом — на выбор без цен; нет —
            # «не нашёл, уточните название». Цены нет ни в одном случае.
            names = _whole_word_names(phrase if parse_phrase is not None else question, pool, rows)
            if names:
                return PriceSelection(rows=(), status="unavailable", names=names), "budget_names"
            return PriceSelection(rows=(), status="none_fit", reason="budget"), "budget_clarify"
        # Правила ничего не предложили — чужой цены не было, их честный ответ остаётся.
        if not candidates and (not guard_bypassed or not _price_by_parse.enabled()):
            return None, "rules"
        if _price_by_parse.enabled():
            # Цена, этап 2, Ш4: без решения LLM строку не выбираем — ни ответ правил, ни их список с ценами (кандидаты правил — по
            # буквам: «лг» внутри «аналгезии», BUG-2026-10-25-RULES-LIST-FOREIGN-ROW); названия целым словом без цен или перевод.
            return PriceSelection(rows=(), status="unavailable", names=_whole_word_names(phrase if parse_phrase is not None else question, pool, rows)), ""
        # Словарный гард пропустил вопрос к LLM только ради её суда: без решения — ответ правил, как без выключателя.
        if guard_bypassed:
            return None, "rules"
        # Правила покажут список — при неоднозначности список, а не одна цена (решение 22.07).
        if len(shown) >= 2:
            return None, "rules_list"
        return PriceSelection(rows=(), status="unavailable", names=_whole_word_names(phrase if parse_phrase is not None else question, pool, rows)), ""

    def none_fit(reason: str) -> PriceSelection | None:
        # Названо слишком общее или не услуга — уточнение правил честнее, чем «не нашёл».
        if not candidates:
            return None
        if reason in _NOT_A_SERVICE_NAME:
            if not general_as_none_fit:
                return None
            return PriceSelection(rows=(), status="none_fit", reason=reason, names=_whole_word_names(phrase if parse_phrase is not None else question, pool, rows))
        return PriceSelection(rows=(), status="none_fit", reason=reason)

    def journal_miss(found_terms: list[str], *reasons: str) -> None:
        # Промах — услугу не нашли ни правила (`candidates` пуст), ни LLM: такое слово пациента —
        # кандидат в синонимы для клиники (см. `_record_synonym_suggestion`). Но только если LLM
        # признала в реплике название услуги: «слишком общее» и «не услуга» в любом из кругов —
        # не синоним (BUG-2026-10-13-SYNONYM-JOURNAL-IGNORES-REASON).
        if not candidates and not _NOT_A_SERVICE_NAME.intersection(reasons) and journal_phrase:
            _record_synonym_suggestion(journal_phrase, found_terms, [], source=journal_source)

    first = await _ask(question, pool, judge_qualifier=judge_qualifier, price_answer=price_answer)
    if first is None:
        return settle(*unavailable())
    chosen, terms, reason = first
    if chosen:
        picked = PriceSelection(rows=tuple(_prefer_base_variants(chosen, question)))
        return settle(picked)
    # Суд условия (`judge_qualifier`): «условия нет» (`qualifier_absent`, правило 3 промпта) — вердикт окончательный.
    # Второй круг искал бы «почти то» (это и запрещает П2), а у класса A под `turn_parse = on` упирался в лимит
    # решений хода (`budget`) и превращал «не нашёл» в ответ правил (BUG-2026-10-16-QUALIFIER-ROUND-TWO-BUDGET).
    # Прочие коды — второй круг, как раньше: он может найти услугу другим словом.
    if one_round or (judge_qualifier and reason == "qualifier_absent"):
        return settle(PriceSelection(rows=(), status="none_fit", reason=reason))
    if not terms:
        journal_miss([], reason)
        return settle(none_fit(reason))

    # По словам терминов — как в первом круге: поиск правил требует всех слов термина, и «Эзофагогастродуоденоскопия
    # с pH-метрией» не находит «Эзофагогастродуоденоскопия (ФГДС)» — второй круг выбирал из пустого или чужого списка
    # и отвечал «не нашёл» (BUG-2026-10-19-PRICE-ROUND-TWO-POOL).
    term_words = list(dict.fromkeys(token for term in terms for token in _price_query_tokens(term) if len(token) >= 3))
    # Редкие слова терминов — первыми (термины назвала сама LLM, слов пациента вне услуги в них нет: «резус фактор» →
    # «Rh- принадлежность», «эхо сердца» → «Эхокардиография» находились именно так, проба 20.10), но только если LLM
    # признала в реплике услугу: термины к «слишком общему» и «не услуге» («цены на анализы можешь озвучить?») — догадка
    # LLM, и строки по их редким словам она же потом выбирала (карточка dev 20.10: список «Биохимия…»).
    rare_by_terms = (
        await asyncio.to_thread(_rare_rows, terms, rows) if reason not in _NOT_A_SERVICE_NAME else []
    )
    pool_by_terms = await asyncio.to_thread(
        _build_pool, terms, rows, per_text=_PER_TERM_LIMIT, first=rare_by_terms, fallback_texts=term_words
    )
    # Пустой список — не вопрос к LLM: выбрать не из чего, ответ «не нашёл» тот же без решения из лимита хода.
    if not pool_by_terms:
        journal_miss(terms, reason)
        return settle(none_fit(reason))
    second = await _ask(question, pool_by_terms, judge_qualifier=judge_qualifier, price_answer=price_answer)
    if second is None:
        return settle(*unavailable())
    if not second[0]:
        journal_miss(terms, reason, second[2])
        return settle(none_fit(second[2]))
    chosen = _prefer_base_variants(second[0], question)
    if journal_phrase:
        _record_synonym_suggestion(journal_phrase, terms, chosen, source=journal_source)
    picked = PriceSelection(rows=tuple(chosen), search_terms=tuple(terms))
    return settle(picked)


def distinct_services(rows: Sequence[dict[str, Any]]) -> bool:
    """Строки — разные услуги, а не варианты одной: у выбранных строк больше одного названия (варианты одной услуги в прайсе —
    одно название с разными ценами и единицами). Сравнение — `_normalise_catalog_text`, как у сильного совпадения."""

    return len({_normalise_catalog_text(_row_name(row)) for row in rows if isinstance(row, dict)}) > 1


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
    - выбора нет — ответ правил: «не нашла» от LLM слабее, чем найденное правилами;
    - под `price_general_clarify = on`: правила показали бы одну строку, а LLM выбрала с ней и другие услуги (не варианты
      одной) — весь её выбор; иначе он сводился к строке правил («консультация врача» → приём хирурга,
      BUG-2026-10-24-FOREIGN-PRICE-D17, класс 3).

    :param proposal: строки, которые показали бы правила
    :param selection: выбор LLM
    :return: None — показать ответ правил как есть; иначе строки к показу
    """

    if selection is None or not selection.rows:
        return None
    proposed = [row for row in (proposal or []) if isinstance(row, dict)]
    if not proposed:
        return list(selection.rows)
    if len(proposed) == 1 and len(selection.rows) > 1 and general_clarify_on() and distinct_services(selection.rows):
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
