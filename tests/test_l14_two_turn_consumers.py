"""Потребители понимания услуги на двух ходах — характеризация до хода (б) L-14 (`ServiceMention`).

Ход (б) переводит «какая услуга» в один объект хода. Подробный разбор решения (кто решает, о чём реплика)
не увидел двух багов 13.10: журнал синонимов брал фразу у старого вырезателя, а состояние сессии хранило
строку, которой пациент не видел. Оба — потребители понятого, а не его производители (мастер-хэндофф §4 (б),
«Слепое пятно плана»; инвентаризация — `~/Dev/handoffs/dla/11_inventory_consumers.md`). Метрики стенда и
разметка смотрят на ответ ОДНОГО хода (`evalbench.run`: сессия на случай), поэтому такие баги живут между
ходами и в файлах — этот класс их и ловит.

Файл фиксирует поведение потребителей ДО перевода, без смены кода. Зелёные тесты — характеризация (их менять
только намеренно, вместе с переводом), красные — `xfail(strict=True)` с точной причиной:

1. Контекст = показанное (принципы 2 и 3: пациент не получает цену услуги, которую не видел).
   `test_context_is_what_was_shown`: после ответа X (одна услуга; семейство вариантов; «не нашёл»; названия
   без цен; список/корзина; перевод на оператора; срок; подбор анализов) в состоянии сессии (`service_name`,
   `test_name`) только показанное; после ответа без цен — ничего (пациент не выбирал). Весь класс зелёный
   (утечку L-13 закрыл `_sync_policy_context`) — тест держит её от возврата при переносе записи в состояние.
   `test_short_question_after_answer`: следующий короткий вопрос («сколько стоит?», «а цена?», «а как
   подготовиться?», «а где сдать?», «да», «1», «где сдать» → «сколько стоит?») не показывает цен строк, которых
   пациент не видел, и не печатает слова самого вопроса как название услуги — здесь красное.
   `test_shown_service_reaches_the_context`: обратное направление — услугу, которую показал выбор LLM или срок
   из МИС, состояние не получает (безопасно, но контекст ≠ показанное) — красное.
2. Журнал синонимов (`test_synonym_journal_phrase`): фраза — то, что назвал пациент и выбрала LLM, а не обрывок
   вырезателя правил; реплика без названия услуги — ничего. Шаг 2 (14.10): под `journal_phrase = mention` фраза —
   упоминание хода (`tests/test_l14_journal_mention.py`); под `rules` (по умолчанию) красное остаётся характеризацией.
3. Объект хода: `test_single_service_is_a_mention` (красное — объект строится только для списка) и
   `test_list_mentions_chosen_…` (у правил `chosen` — название строки прайса, у разбора — фраза пациента).
4. Остальные потребители, закреплённые как есть: состояние после списка / семейства / одной услуги / срока,
   анти-повтор (читает текст ответа, не услугу), телеметрия (числа и статусы, без фраз).

Мир и наборы выключателей — как в `test_price_one_or_many_class.py`: мини-прод, день, сеть закрыта, синонимы МИС
пусты; LLM гейта недоступна, где нужна живая — `_Scripted`. Наборы: `default` (по умолчанию), `deploy`
(прод сегодня: понимание хода, корзина, срок, симптом — on; цена через модуль off; L-13 — тень),
`deploy_parse` (набор деплоя с `turn_parse = on`, L-15b шаг 5), `next` (кандидат следующего деплоя: набор
деплоя с разбором, ценой через модуль и политикой L-13 — on; включать ли его — решение владельца, не здесь).

Классификатор LLM в гейте недоступен: короткие вопросы, которые правила не опознают («а сколько?», «а срок?»,
«1»), уходят в «не понял» — безопасный ответ, он тоже закреплён. Ходы с живым классификатором, который
перенёс бы услугу из `seeded`-контекста в сущности, гейт не воспроизводит — это стенд (`evalbench`, `cached`,
с диалогами из двух ходов).

Красное объявлено явно: на каждую пару (сценарий, вопрос, набор) — словарь `{код нарушения: причина}`.
`xfail(strict=True, raises=_Red)` значит, что нарушения ровно объявленные: позеленевший случай краснеет
(XPASS), а нарушение другого рода или частичное исправление — обычный провал, а не xfail. Отметки снимают в том
же коммите, который переводит потребителя на `ServiceMention`.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_one_or_many_class import _DEPLOY
from tests.test_price_turns_golden import FIXTURE, mini_prod_world  # noqa: F401 — фикстура модуля

_SETS: dict[str, dict[str, str]] = {
    "default": {},
    "deploy": _DEPLOY,
    "deploy_parse": {**_DEPLOY, "MR_TURN_PARSE": "on"},
    "next": {**_DEPLOY, "MR_TURN_PARSE": "on", "MR_CHOOSE_PRICE": "on", "MR_PRICE_MATCH_POLICY": "on"},
}
BASE_CONFIGS = tuple(_SETS)
# L-14, шаг 3 (14.10): «контекст = показанное» (`context_shown = on`) поверх деплоя и кандидата следующего деплоя.
_SETS["deploy_shown"] = {**_DEPLOY, "MR_CONTEXT_SHOWN": "on"}
_SETS["next_shown"] = {**_SETS["next"], "MR_CONTEXT_SHOWN": "on"}
SHOWN_CONFIGS = ("deploy_shown", "next_shown")
CONFIGS = tuple(_SETS)

_PHRASES_MARK = "понять, какие услуги назвал пациент"
_PRICE_MARK = "ответить пациенту на вопрос о цене"
_PRICED = re.compile(r"^\s*(?:\d+[.)]\s*|-\s*)?(?:Розничная цена:\s*)?(?P<name>.+?) — \d[\d\s]* руб", re.M)
_QUOTED = re.compile(r"«([^»]+)»")

# Продолжения диалога после ответа X: метка (она же — часть идентификатора теста) → реплики. Нарушения ищет
# последний ход; «показанное» — все ответы до него.
FOLLOWUPS: dict[str, tuple[str, ...]] = {
    "price": ("сколько стоит?",),
    "a_price": ("а цена?",),
    "prep": ("а как подготовиться?",),
    "where": ("а где сдать?",),
    "yes": ("да",),
    "one": ("1",),
    "where_price": ("а где сдать?", "сколько стоит?"),  # «где сдать» заменяет услугу в состоянии классом «анализы» (под `off`)
}

# --------------------------------------------------------------------------- мир, ходы, LLM


class _Red(AssertionError):
    """Нарушения класса — ровно объявленные: случай красный по причине L-14, а не по новой."""


class _Scripted:
    """LLM для двух мест: выбор фраз реплики (`cut`; None — недоступна) и выбор строки прайса (первый круг —
    без выбора и с терминами `terms`, второй — строки, чьё название начинается с `find_prefix`). Обе формы
    ответа выбора цены — модуль (`system` задан) и прежний вызов (`system` пуст). Прочие места — недоступна."""

    def __init__(self, cut: tuple[str, ...] | None = None, terms: tuple[str, ...] = (), find_prefix: str | None = None, reason: str = "other_service") -> None:
        self.cut = {_normalise_catalog_text(text) for text in cut} if cut is not None else None
        self.terms, self.find_prefix, self.reason = list(terms), find_prefix, reason
        self.price_calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _PHRASES_MARK in seen:
            if self.cut is None:
                raise ConnectionError("тест: LLM недоступна для выбора фраз")
            picks = [n for n, text in options if _normalise_catalog_text(text.strip("«»")) in self.cut]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if _PRICE_MARK in seen:
            self.price_calls += 1
            picks = []
            if self.price_calls >= 2 and self.find_prefix:
                picks = [n for n, text in options if text.rsplit(" — ", 1)[0].strip().startswith(self.find_prefix)]
            terms = [] if picks else self.terms
            if system is None:
                answer: dict[str, Any] = {"match": picks, "search_terms": terms}
            else:
                answer = {"picks": picks, "reason": "none" if picks else self.reason, "search_terms": terms}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _state_view(state: dict[str, Any]) -> dict[str, Any]:
    pending = state.get("_pending") if isinstance(state.get("_pending"), dict) else {}
    family = state.get("_price_family_context") if isinstance(state.get("_price_family_context"), dict) else {}
    return {
        "service_name": state.get("service_name"),
        "test_name": state.get("test_name"),
        "last_label": state.get("_last_label"),
        "pending": (pending.get("label"), tuple(pending.get("missing") or ())) if pending else None,
        "family_context": {"service_name": family.get("service_name"), "variants": len(family.get("family_variants") or [])} if family else None,
        "compound_pending": bool(state.get("_compound_price_pending")),
        "secondary_offer": bool(state.get("_secondary_offer_pending")),
    }


@dataclass
class _Turn:
    text: str
    handoff: bool
    label: str | None
    plan: list[str] | None
    service_name: str | None
    test_name: str | None
    understanding: dict[str, Any] | None
    state: dict[str, Any] = field(default_factory=dict)  # ключи состояния сессии, которые читают потребители услуги
    summary: str = ""  # сводка диалога (уходит в NLU как `_summary`)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    journal: list[dict[str, Any]] = field(default_factory=list)  # строки журнала синонимов, добавленные ходом
    shown: list[str] | None = None  # `_shown_services` состояния (L-14, шаг 3, `context_shown = on`)


_SYNONYM_FORK = {
    "оак": (
        "Общий анализ крови (Le, Er, Hb, СОЭ)",
        "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
        "Cito Общий анализ крови (Le, Er,Hb)",
    ),
}


def _play(
    turns: list[str], name: str, world_dir: Path, monkeypatch, config: str, llm: Any = None, render: str | None = None, fork: bool = False
) -> list[_Turn]:
    """Диалог через обработчик эндпоинта (`messenger_generate_once`, отладочный ответ) в мини-проде.

    :param render: фиксированный текст свободного ответа LLM (подбор анализов отвечает ею); None — LLM недоступна
    :param fork: словарь синонимов МИС с «оак» на три услуги
    """

    from messengers_router import renderer
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)  # как на стенде: решения не делятся между ходами
    if llm is not None:
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    if fork:
        from messengers_router.services import _biomaterial as bio

        hermetic = bio._vocabularies

        def with_fork():
            vocab = hermetic()
            vocab.synonym_candidates = dict(_SYNONYM_FORK)
            vocab.synonym_to_service = {}
            return vocab

        monkeypatch.setattr(bio, "_vocabularies", with_fork)
    if render is not None:

        async def fake_stream(*_args, **_kwargs):
            yield render

        monkeypatch.setattr(renderer, "render_stream", fake_stream)

    with ev_world.FrozenWorld(world_dir, "day") as world:

        def journal() -> list[dict[str, Any]]:
            path = world._journal
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []

        async def run() -> list[_Turn]:
            memory, services = MemoryStore(), Services()
            session = f"l14-{name}-{config}"
            out: list[_Turn] = []
            for text in turns:
                tel_before, journal_before = len(world.telemetry), len(journal())
                req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id="")
                body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
                dbg = (body.get("state_update") or {}).get("debug") or {}
                plan = dbg.get("plan") or {}
                state = dict(memory.get(session).last_entities or {})
                out.append(
                    _Turn(
                        text=body.get("text", ""),
                        handoff=bool(body.get("handoff")),
                        label=(dbg.get("decision") or {}).get("label"),
                        plan=[plan.get("label")] + [s.get("tool") for s in plan.get("steps") or []] if plan else None,
                        service_name=state.get("service_name"),
                        test_name=state.get("test_name"),
                        understanding=dbg.get("understanding"),
                        state=_state_view(state),
                        summary=str(memory.get(session).summary or ""),
                        telemetry=list(world.telemetry[tel_before:]),
                        journal=journal()[journal_before:],
                        shown=state.get("_shown_services"),
                    )
                )
            return out

        result = asyncio.run(run())
        assert list(world.net.attempts) == []
    return result


@functools.cache
def _catalog_names_raw() -> frozenset[str]:
    path = FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl"
    return frozenset(json.loads(line)["serviceName"] for line in path.open(encoding="utf-8") if line.strip())


@functools.cache
def _catalog_names() -> tuple[str, ...]:
    path = FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl"
    names = {_normalise_catalog_text(json.loads(line)["serviceName"]) for line in path.open(encoding="utf-8") if line.strip()}
    return tuple(sorted(names, key=len, reverse=True))


def _shown(text: str) -> set[str]:
    """Названия строк прайса, которые пациент видел в тексте ответа (нормализованные)."""

    norm = _normalise_catalog_text(text)
    return {name for name in _catalog_names() if name and name in norm}


def _priced(text: str) -> list[str]:
    return [_normalise_catalog_text(match.group("name")) for match in _PRICED.finditer(text)]


# --------------------------------------------------------------------------- инварианты


_CODES = {
    "ctx": "контекст сессии держит услугу, которой нет в показанном ответе",
    "rows": "короткий вопрос показал цены строк, которых пациент не видел",
    "phrase": "слова самого вопроса напечатаны как название услуги",
    "class": "класс walk-in («анализы») напечатан как название услуги",
}
_WALKIN_CLASS = _normalise_catalog_text("анализы")  # класс, который гард walk-in ставит вместо услуги контекста


def _context_violations(first: _Turn, *, named_one: bool = False) -> dict[str, str]:
    """Состояние после ответа X. Ответ с ценами: `service_name` / `test_name` — только показанное (название
    строки прайса из ответа или слова пациента, которые ответ повторил). Ответ без цен («не нашёл», названия
    без цен, уточнение, перевод на оператора, срок, подбор анализов): пациент ничего не выбрал — в состоянии
    услуги нет (решение владельца 13.10, BUG-2026-10-13-L13-CONTEXT-LEAK).

    :param named_one: правило владельца 14.10 (под `context_shown = on`): в контексте — услуга, которую ответ
        назвал пациенту одной строкой («Срок готовности «Ферритин» — 1–2 дня»); догадка — никогда."""

    held = [str(value) for value in (first.service_name, first.test_name) if value]
    if not held:
        return {}
    if not _priced(first.text):
        quoted = {_normalise_catalog_text(q) for q in _QUOTED.findall(first.text)}
        if named_one and len(held) == 1 and quoted == {_normalise_catalog_text(held[0])} and held[0] in _catalog_names_raw():
            return {}
        return {"ctx": f"{_CODES['ctx']}: ответ без цен, пациент ничего не выбрал, а в состоянии {held}"}
    shown, norm_text = _shown(first.text), _normalise_catalog_text(first.text)
    unseen = [value for value in held if _normalise_catalog_text(value) not in shown and _normalise_catalog_text(value) not in norm_text]
    return {"ctx": f"{_CODES['ctx']}: {unseen}"} if unseen else {}


def _followup_violations(earlier: list[_Turn], last: _Turn, said: str) -> dict[str, str]:
    """Последний ответ против показанного раньше (`earlier` — ответы X и промежуточных ходов)."""

    seen: set[str] = set()
    for turn in earlier:
        seen |= _shown(turn.text)
    out: dict[str, str] = {}
    foreign = [name for name in _priced(last.text) if name not in seen]
    if foreign:
        out["rows"] = f"{_CODES['rows']}: {foreign[:3]}"
    own = _normalise_catalog_text(said.strip("?! "))
    quoted = [frag for frag in _QUOTED.findall(last.text) if _normalise_catalog_text(frag) in {own, "а", "и"}]
    if quoted:
        out["phrase"] = f"{_CODES['phrase']}: {quoted}"
    if [frag for frag in _QUOTED.findall(last.text) if _normalise_catalog_text(frag) == _WALKIN_CLASS]:
        out["class"] = _CODES["class"]
    return out


def _settle(found: dict[str, str], declared: dict[str, str], dialog: str) -> None:
    """Нарушения ровно объявленные: объявленные есть — `_Red` (xfail), расходятся — обычный провал."""

    if set(found) != set(declared):
        raise AssertionError(f"нарушения {sorted(found)} не совпадают с объявленными {sorted(declared)}: {found}\n--- диалог:\n{dialog}")
    if found:
        raise _Red("; ".join(found.values()))


def _dialog_text(turns: list[str], played: list[_Turn]) -> str:
    return "\n".join(f"> {said}\n{turn.text[:700]}" for said, turn in zip(turns, played))


# --------------------------------------------------------------------------- сценарии: ответ X


@dataclass(frozen=True)
class Scenario:
    first: str
    followups: tuple[str, ...]
    llm: Callable[[], Any] | None = None
    render: str | None = None
    fork: bool = False  # синоним МИС «оак» на три услуги (словарь клиники задан явно: гейт держит синонимы пустыми)


_NOTHING_FOUND = lambda: _Scripted(terms=("ФГДС",))  # noqa: E731 — LLM выбора цены: «не та услуга» в обоих кругах, термин ФГДС
_FOUND_GGT = lambda: _Scripted(cut=("ггтп",), terms=("ГГТ",), find_prefix="ГГТ")  # noqa: E731 — «ггтп» → ГГТ по термину LLM

SCENARIOS: dict[str, Scenario] = {
    # одна услуга: строки прайса показаны, правила и данные МИС
    "one_ferritin": Scenario("сколько стоит ферритин", ("price", "a_price", "prep", "where", "where_price")),
    # семейство: правила показывают соседние строки; под разбором (STRONG) — только дословное название
    "family_glucose": Scenario("сколько стоит глюкоза", ("price", "a_price", "prep")),
    "family_oak": Scenario("сколько стоит оак", ("price", "a_price", "prep", "where", "one", "where_price")),
    # синоним МИС на несколько услуг: показаны ровно они
    "fork_oak": Scenario("сколько стоит оак", ("price", "a_price", "prep"), fork=True),
    # «не нашёл» (под L-13 — честное «не нашёл», иначе ответ правил: семья «Гастрин»)
    "not_found_gastro": Scenario("сколько стоит гастроскопия", ("price", "a_price", "one"), llm=_NOTHING_FOUND),
    # названия без цен (вариант B L-13) / одна цена правил
    "names_ttg": Scenario("сколько стоит анализ ттг", ("price", "a_price", "yes", "one")),
    # опечатка: ответ политики «перевод на оператора» (L-13) / цена правил
    "typo_ferritin": Scenario("сколько стоит ферритинн", ("price", "a_price")),
    # список (корзина): сильные позиции — по данным, слабые — выбор LLM или варианты кода
    "cart_three": Scenario("сколько стоит ферритин, ттг и витамин д", ("price", "a_price", "prep", "where")),
    # приём: пациент увидел бы одну строку из нескольких — под L-13 названия без цены
    "names_neurologist": Scenario("сколько стоит прием невролога", ("price", "a_price")),
    # две услуги через «и», одной из них нет в мини-проде («Не распознал»)
    "uzi_compound": Scenario("сколько стоит узи почек и щитовидной железы", ("price", "a_price", "prep")),
    # строку выбрала LLM по термину, правила её не находили
    "llm_pick_ggt": Scenario("сколько стоит ггтп", ("price", "a_price"), llm=_FOUND_GGT),
    # срок: из МИС кодом / честный перевод на оператора
    "readiness_ferritin": Scenario("сколько готовится ферритин", ("price", "prep")),
    "readiness_generic": Scenario("сколько готовятся анализы", ("price",)),
    # подбор анализов: свободный ответ LLM, данных об услуге в состоянии быть не должно
    "assist_prevention": Scenario("какие анализы сдать для профилактики", ("price", "a_price"), render="Могу подобрать анализы. Уточните, что именно вас беспокоит."),
}

# Красное. Контекст после ответа X: (сценарий, набор) → {код: причина}. Пусто — зелёный (на 13.10 весь класс зелёный:
# утечку L-13 закрыл `_sync_policy_context`; тест держит её от возврата при переносе).
KNOWN_CONTEXT: dict[tuple[str, str], dict[str, str]] = {}

_A_PRICE_ROWS = (
    "«а цена?»: вырезатель правил (`_extract_price_service_from_query`) отдаёт «а», а `_should_prefer_retail_query_candidate` "
    "ставит его выше услуги контекста (подстрока: «а» есть в «Глюкоза», «Гастрин», «Общий анализ крови»), ранжирование по "
    "букве «а» показывает цены аллергенов и шерсти"
)
_A_PRICE_PHRASE = "заголовок ответа «По услуге «а»»: слова вопроса напечатаны как название услуги"
_REPEAT_FAMILY = (
    "под разбором ответ X — одна строка «Глюкоза» (дословное название, STRONG), а повтор вопроса идёт правилами по "
    "`service_name` контекста и показывает семью с ценами «Глюкозы в моче» и «в синовиальной жидкости», которых пациент не видел"
)
_PREP_PHRASE = (
    "`_prepare_subject_hint` берёт самую длинную из фраз вопроса и контекста: «а как подготовиться» длиннее «Ферритин» и "
    "«Глюкоза» и печатается как предмет памятки («Памятки по подготовке к «а как подготовиться» у меня нет»); у длинных "
    "названий (ОАК, ТТГ) побеждает контекст — по длине, а не по смыслу"
)
_WHERE_PRICE_PHRASE = (
    "после «а где сдать?» walk-in заменяет `service_name` классом «анализы» (ходы 11 и 22 карты гардов), и «сколько стоит?» "
    "ищет по вырезанной фразе: «По услуге «сколько стоит»… цену определить не удалось… соединяю» — показанная услуга потеряна, "
    "слова вопроса напечатаны как название"
)

# (сценарии, вопрос, наборы, {код: причина}) → построчно в KNOWN_FOLLOWUP.
_FOLLOWUP_RED: tuple[tuple[tuple[str, ...], str, tuple[str, ...], dict[str, str]], ...] = (
    (("family_glucose", "family_oak", "fork_oak"), "a_price", BASE_CONFIGS, {"rows": _A_PRICE_ROWS, "phrase": _A_PRICE_PHRASE}),
    (("not_found_gastro",), "a_price", ("default", "deploy", "deploy_parse"), {"rows": _A_PRICE_ROWS, "phrase": _A_PRICE_PHRASE}),
    (("family_glucose",), "price", ("deploy_parse", "next"), {"rows": _REPEAT_FAMILY}),
    (("one_ferritin", "family_glucose"), "prep", BASE_CONFIGS, {"phrase": _PREP_PHRASE}),
    # Под `context_shown = on` закрыто (L-14, находка 3): после ответа адреса в контексте снова показанное.
    (("one_ferritin", "family_oak"), "where_price", BASE_CONFIGS, {"phrase": _WHERE_PRICE_PHRASE}),
)
# (сценарий, вопрос, набор) → {код: причина}. Пусто — зелёный. Вопрос — метка из FOLLOWUPS.
KNOWN_FOLLOWUP: dict[tuple[str, str, str], dict[str, str]] = {
    (scenario, question, config): codes for scenarios, question, configs, codes in _FOLLOWUP_RED for scenario in scenarios for config in configs
}


def _marks(declared: dict[str, str]):
    return [pytest.mark.xfail(strict=True, raises=_Red, reason="L-14: " + "; ".join(declared.values()))] if declared else []


def _context_params():
    for name in SCENARIOS:
        for config in CONFIGS:
            yield pytest.param(name, config, marks=_marks(KNOWN_CONTEXT.get((name, config), {})), id=f"{name}-{config}")


def _followup_params():
    for name, scenario in SCENARIOS.items():
        for followup in scenario.followups:
            for config in CONFIGS:
                yield pytest.param(name, followup, config, marks=_marks(KNOWN_FOLLOWUP.get((name, followup, config), {})), id=f"{name}-{followup}-{config}")


@pytest.mark.parametrize(("name", "config"), list(_context_params()))
def test_context_is_what_was_shown(name, config, mini_prod_world, monkeypatch):  # noqa: F811
    """После ответа X в состоянии нет услуги, которой пациент не видел (контекст = показанное)."""

    scenario = SCENARIOS[name]
    played = _play([scenario.first], name, mini_prod_world, monkeypatch, config, llm=scenario.llm() if scenario.llm else None, render=scenario.render)
    found = _context_violations(played[0], named_one=config in SHOWN_CONFIGS)
    _settle(found, KNOWN_CONTEXT.get((name, config), {}), _dialog_text([scenario.first], played))


@pytest.mark.parametrize(("name", "followup", "config"), list(_followup_params()))
def test_short_question_after_answer(name, followup, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Следующий короткий вопрос не показывает цен строк, которых в ответе X не было, и не печатает слова
    самого вопроса как название услуги; контекст после него — тоже только показанное."""

    scenario = SCENARIOS[name]
    turns = [scenario.first, *FOLLOWUPS[followup]]
    played = _play(turns, f"{name}-{followup}", mini_prod_world, monkeypatch, config, llm=scenario.llm() if scenario.llm else None, render=scenario.render, fork=scenario.fork)
    found = _followup_violations(played[:-1], played[-1], turns[-1])
    _settle(found, KNOWN_FOLLOWUP.get((name, followup, config), {}), _dialog_text(turns, played))


# --------------------------------------------------------------------------- журнал синонимов


@dataclass(frozen=True)
class JournalCase:
    said: str
    llm: Callable[[], _Scripted]
    expected: tuple[str, ...]  # фразы, которые должны попасть в журнал (нормализованные), по порядку; () — ничего


_TOO_GENERAL = lambda: _Scripted(reason="too_general")  # noqa: E731 — LLM выбора цены: «слишком общее» (причину знает промпт модуля)

JOURNAL_CASES: dict[str, JournalCase] = {
    # фраза пациента и есть то, что выбрала LLM
    "bare_term": JournalCase("сколько стоит ггтп", _FOUND_GGT, ("ггтп",)),
    # слова вопроса вокруг названия: журналу нужна фраза услуги, а не «а ггтп»
    "term_in_a_sentence": JournalCase("а подскажите пожалуйста сколько стоит ггтп сдать", _FOUND_GGT, ("ггтп",)),
    # слово, которого нет ни в прайсе, ни у LLM («спирометрия»): клинике нужнее всего
    "unknown_word": JournalCase(
        "подскажите пожалуйста сколько стоит спирометрия", lambda: _Scripted(cut=("спирометрия",), terms=("спирография",)), ("спирометрия",)
    ),
    # реплика без названия услуги — не синоним (прод 13.10: «Цена анализа», «сможете посчитать» в журнале)
    "no_service_named_1": JournalCase("Цена анализа", _TOO_GENERAL, ()),
    "no_service_named_2": JournalCase("сможете посчитать стоимость", _TOO_GENERAL, ()),
    "no_service_named_3": JournalCase("подскажите стоимость анализов", _TOO_GENERAL, ()),
}

_JOURNAL_RED = "журнал берёт фразу вырезателя правил (`_extract_price_service_from_query`), а не фразу пациента/выбор LLM"
_JOURNAL_SENTENCE = {
    "default": "вырезатель оставляет «а» перед термином («а ггтп»); фразы LLM нет (`turn_parse = off`)",
    "deploy": "вырезатель оставляет «а» перед термином («а ггтп»); фразы LLM нет (`turn_parse = off`)",
    "deploy_parse": "разбор вернул `single` (кандидат один — вся реплика): LLM выбора фраз не зовётся, и `_journal_phrase` возвращает фразу вырезателя «а ггтп»",
    "next": "разбор вернул `single` (кандидат один — вся реплика): LLM выбора фраз не зовётся, и `_journal_phrase` возвращает фразу вырезателя «а ггтп»",
}
_JOURNAL_SENTENCE["deploy_shown"], _JOURNAL_SENTENCE["next_shown"] = _JOURNAL_SENTENCE["deploy"], _JOURNAL_SENTENCE["next"]
_JOURNAL_NO_SERVICE = (
    "прежний промпт выбора цены (`choose_price = off`) не знает причины «слишком общее» (её даёт только промпт модуля), `journal_miss` "
    "пишет реплику как синоним; под разбором `single` (LLM фраз не звали) не отличает «нет услуги» от «слова без синонима»; "
    "закрыто только при `choose_price = on` (набор `next`)"
)
_JOURNAL_NO_SERVICE_WORDS = (
    "«сможете посчитать» — слова, которых нет ни в обвязке, ни в родовых словах: упоминание есть (слабое), и без причины "
    "«не услуга» от LLM (её знает только промпт модуля, `choose_price = on`) журнал не отличает его от слова без синонима; "
    "добавить слова в списки обвязки — решение владельца (правило под буквы)"
)
# (случай, набор, источник фразы журнала) → {код: причина}. Пусто — зелёный. Источник — `[MESSENGER_ROUTER]
# journal_phrase`: `rules` — прежний (по умолчанию), `mention` — упоминание хода (L-14, шаг 2).
KNOWN_JOURNAL: dict[tuple[str, str, str], dict[str, str]] = {
    **{("term_in_a_sentence", config, "rules"): {"journal": _JOURNAL_SENTENCE[config]} for config in CONFIGS},
    **{
        (case, config, "rules"): {"journal": _JOURNAL_NO_SERVICE}
        for case in ("no_service_named_1", "no_service_named_2", "no_service_named_3")
        for config in ("default", "deploy", "deploy_parse", "deploy_shown")
    },
    **{
        ("no_service_named_2", config, "mention"): {"journal": _JOURNAL_NO_SERVICE_WORDS}
        for config in ("default", "deploy", "deploy_parse", "deploy_shown")
    },
}
JOURNAL_SOURCES = ("rules", "mention")


def _journal_params():
    for name in JOURNAL_CASES:
        for config in CONFIGS:
            for source in JOURNAL_SOURCES:
                marks = _marks(KNOWN_JOURNAL.get((name, config, source), {}))
                yield pytest.param(name, config, source, marks=marks, id=f"{name}-{config}-{source}")


@pytest.mark.parametrize(("name", "config", "source"), list(_journal_params()))
def test_synonym_journal_phrase(name, config, source, mini_prod_world, monkeypatch):  # noqa: F811
    """Журнал синонимов (подсказка клинике) пишет фразу, которую назвал пациент и выбрала LLM; реплика без
    названия услуги — ничего. Под `journal_phrase = mention` фраза — упоминание хода (L-14, шаг 2)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_JOURNAL_PHRASE", source, raising=False)
    case = JOURNAL_CASES[name]
    [turn] = _play([case.said], f"{name}-{source}", mini_prod_world, monkeypatch, config, llm=case.llm())
    phrases = tuple(_normalise_catalog_text(str(row["phrase"])) for row in turn.journal)
    found: dict[str, str] = {}
    if phrases != case.expected:
        found["journal"] = f"{_JOURNAL_RED}: в журнале {list(phrases)}, ожидалось {list(case.expected)}"
    _settle(found, KNOWN_JOURNAL.get((name, config, source), {}), f"> {case.said}\n{turn.text[:400]}\nжурнал: {turn.journal}")


# --------------------------------------------------------------------------- объект хода: упоминания услуг

_LIST_REPLY = "сколько стоит ферритин, ттг и витамин д"
_LIST_RULE_NAMES = [
    "Ферритин",
    "ТТГ (TSH) тиреотропный гормон",
    "Витамин D суммарный (25-OH витамин D2 и 25-OH витамин D3, общий)",
]


@pytest.mark.parametrize("config", CONFIGS)
def test_list_mentions_chosen_is_catalog_name_by_rules_and_phrase_by_parse(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Список — упоминания хода (`understanding.services`). Что лежит в `chosen`, зависит от пути: у разреза
    правил — названия строк прайса, у разбора (`turn_parse = on`) — фразы пациента (слабые позиции строки
    ещё не выбрали). Единого смысла поля нет — перевод на `ServiceMention` его задаёт (кандидаты, сила, выбор)."""

    [turn] = _play([_LIST_REPLY], "mentions-list", mini_prod_world, monkeypatch, config)
    assert turn.understanding is not None and turn.understanding["facet"] == "price"
    if _SETS[config].get("MR_TURN_PARSE") == "on":
        assert turn.understanding["services"] == ["ферритин", "ттг", "витамин д"]
        assert turn.understanding["services_rule"] == "parse_unavailable"
    else:
        assert turn.understanding["services"] == _LIST_RULE_NAMES
        assert turn.understanding["services_rule"] == "list_split"


@pytest.mark.parametrize("config", CONFIGS)
def test_single_service_is_a_mention(config, mini_prod_world, monkeypatch):  # noqa: F811
    """L-14, шаг 0 (14.10): у одиночной услуги — упоминание хода (сильное по данным МИС — с выбором)."""

    [turn] = _play(["сколько стоит ферритин"], "mentions-single", mini_prod_world, monkeypatch, config)
    assert turn.understanding is not None and turn.understanding["services"] == ["Ферритин"]


def test_single_mention_by_rules_strength_and_choice(mini_prod_world):  # noqa: F811
    """Сила и выбор — по данным МИС: дословное название — сильное, выбор — оно; слово без названия и синонима —
    слабое, без кандидатов и без выбора (строку выбирает путь цены, как раньше); без фразы услуги — ничего."""

    from messengers_router import turn_understanding as tu

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        strong = tu.understand_price_turn("сколько стоит ферритин").single
        weak = tu.understand_price_turn("сколько стоит ттг").single
        empty = tu.understand_price_turn("сколько стоит?")
        listed = tu.understand_price_turn(_LIST_REPLY)
    assert (strong.source, strong.strength, strong.candidates, strong.chosen) == ("rules", "strong", ("Ферритин",), "Ферритин")
    assert (weak.source, weak.strength, weak.candidates, weak.chosen) == ("rules", "weak", (), None)
    assert empty.services == () and empty.single is None
    assert listed.single is None and listed.list_size == len(listed.services) >= 2


@pytest.mark.parametrize("config", CONFIGS)
def test_single_mention_is_an_observer_and_tu_list_keeps_its_meaning(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Упоминание одиночной услуги — наблюдатель: `tu_list` по-прежнему размер списка (0), а источник и сила —
    отдельными полями, без текста пациента."""

    from messengers_router.runtime_config import config as cfg
    from tests.test_price_turns_golden import run_dialog

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        run_dialog(["сколько стоит ферритин"], f"mention-telemetry-{config}")
        row = world.take_telemetry()
    if _SETS[config].get("MR_TURN_UNDERSTANDING_PRICE", "shadow") == "off":
        return
    assert row.get("tu_list") == 0
    assert row.get("tu_mention_strength") == "strong"
    assert row.get("tu_mention_src") in ("rules", "parse")


# Обратное направление «контекст = показанное»: услуга, которую ответ назвал пациенту и которую выбрала не
# заземляющая правила, а LLM (строка цены) или код (срок из МИС), в состояние не попадает — следующее «сколько
# стоит?» не знает, о чём шла речь (безопасно, но контекст ≠ показанное). (сценарий, наборы, показанная услуга)
_SHOWN_RED = (
    "L-14: показанную услугу выбрали LLM или код МИС (`service_name_effective`, срок), а в состояние сессии пишет "
    "только заземление правил по буквам (`merge_entities` в роутере; ранний возврат `_maybe_readiness_reply` идёт мимо него)"
)
_SHOWN_BUT_NOT_HELD = (
    ("llm_pick_ggt", CONFIGS, "ГГТ гамма-глютамилтрансфераза"),  # строку выбрала LLM по термину «ГГТ»
    ("readiness_ferritin", ("deploy", "deploy_parse", "next", *SHOWN_CONFIGS), "Ферритин"),  # срок «Ферритин» напечатал код из МИС
)
# Красное: (сценарий, набор) → причина. Под `context_shown = on` выбор LLM доходит до контекста (шаг 3).
_SHOWN_KNOWN = {
    **{("llm_pick_ggt", config): _SHOWN_RED for config in BASE_CONFIGS},
    **{("readiness_ferritin", config): _SHOWN_RED for config in ("deploy", "deploy_parse", "next")},
}


def _shown_params():
    for name, configs, service in _SHOWN_BUT_NOT_HELD:
        for config in configs:
            yield pytest.param(
                name,
                config,
                service,
                marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=_SHOWN_KNOWN[(name, config)])] if (name, config) in _SHOWN_KNOWN else [],
                id=f"{name}-{config}",
            )


@pytest.mark.parametrize(("name", "config", "service"), list(_shown_params()))
def test_shown_service_reaches_the_context(name, config, service, mini_prod_world, monkeypatch):  # noqa: F811
    scenario = SCENARIOS[name]
    turn = _state_after(scenario.first, config, mini_prod_world, monkeypatch, f"shown-{name}", llm=scenario.llm() if scenario.llm else None)
    assert _normalise_catalog_text(service) in _normalise_catalog_text(turn.text)  # услугу пациент действительно увидел
    assert turn.state["service_name"] == service


# --------------------------------------------------------------------------- состояние сессии, ожидания, анти-повтор, телеметрия
#
# Что остаётся в состоянии после ответа X — читатели: планировщик (выбор инструмента по `service_name`), цена и
# подготовка («контекст услуги»), заземление и дозаполнение, классификатор (`seeded` — контекст для NLU),
# сводка диалога. Эти тесты закрепляют записи как есть, чтобы перевод на `ServiceMention` менял их намеренно.


def _state_after(first: str, config: str, world_dir: Path, monkeypatch, name: str, **kwargs) -> _Turn:
    [turn] = _play([first], name, world_dir, monkeypatch, config, **kwargs)
    return turn


@pytest.mark.parametrize("config", CONFIGS)
def test_cart_leaves_no_single_service_and_no_continuation_in_state(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Список: планировщик снимает `service_name` / `test_name` прошлых ходов (следующее «а подготовка?» — не про
    одну из позиций), семейства и составного уточнения нет, ожидания нет; предложение «помогу выбрать» — есть."""

    turn = _state_after(_LIST_REPLY, config, mini_prod_world, monkeypatch, "state-cart")
    assert turn.state == {
        "service_name": None,
        "test_name": None,
        "last_label": "PRICE",
        "pending": None,
        "family_context": None,
        "compound_pending": False,
        "secondary_offer": True,
    }


@pytest.mark.parametrize("config", CONFIGS)
def test_family_list_keeps_the_first_variant_and_the_announced_family(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Семейство правил («оак»: два варианта и «ещё 6»): в состоянии первый показанный вариант (пациент не
    выбирал — его выбрало заземление) и семейство для «все» — ровно то, что ответ пообещал показать (8)."""

    turn = _state_after("сколько стоит оак", config, mini_prod_world, monkeypatch, "state-family")
    assert "найдено еще 6 вариантов" in turn.text
    assert turn.state["service_name"] == "Общий анализ крови (Le, Er, Hb, СОЭ)"
    assert turn.state["family_context"] == {"service_name": "общий анализ крови", "variants": 8}
    assert turn.state["service_name"] in turn.text and turn.state["pending"] is None


@pytest.mark.parametrize("config", CONFIGS)
def test_strong_single_service_is_the_context(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Одна сильная по данным услуга: в состоянии ровно показанная строка; семейства для «все» нет."""

    turn = _state_after("сколько стоит ферритин", config, mini_prod_world, monkeypatch, "state-strong")
    assert _priced(turn.text) == ["ферритин"]
    assert (turn.state["service_name"], turn.state["test_name"], turn.state["family_context"], turn.state["pending"]) == ("Ферритин", None, None, None)


@pytest.mark.parametrize("config", CONFIGS)
def test_readiness_answer_leaves_no_service_pending_or_label_in_state(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Срок из МИС кодом (ответ о «Ферритин»): ход уходит ранним возвратом (`_maybe_readiness_reply`) мимо
    слияния сущностей — в состоянии нет ни услуги, ни ожидания, ни метки хода. Следующее «сколько стоит?» не
    знает, о чём речь, и просит название (безопасно; но показанная услуга — «Ферритин» — контексту не досталась).

    Набор `default`: понимание хода в тени — вопрос о сроке не обрабатывается, ответ — перевод на оператора."""

    turn = _state_after("сколько готовится ферритин", config, mini_prod_world, monkeypatch, "state-readiness")
    if config == "default":
        assert turn.handoff
    else:
        assert turn.text == "Срок готовности «Ферритин» — 1–2 дня." and not turn.handoff
    # Под `context_shown = on` (правило владельца 14.10) срок назвал одну услугу — она в контексте.
    held = "Ферритин" if config in SHOWN_CONFIGS else None
    assert (turn.state["service_name"], turn.state["pending"], turn.state["last_label"]) == (held, None, None)


_SUMMARY_RED = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="L-14: сводка диалога (`context_summary.update_summary`, раз в 4 хода) пишет `service_name=…` из состояния и переживает его "
    "изменение и снятие: после ответа пятого хода состояние уже другое (или пусто), а `_summary`, который уходит в промпт классификатора, "
    "ещё называет прежнюю услугу (под `context_shown = on` сводка обновляется при расхождении — шаг 3)",
)


@pytest.mark.parametrize("config", [pytest.param(c, marks=_SUMMARY_RED if c in BASE_CONFIGS else [], id=c) for c in CONFIGS])
def test_summary_does_not_outlive_the_context_it_copied(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Сводка диалога копирует `service_name` из состояния на четвёртом ходу. Пятый ход — ответ политики «не нашёл»
    (L-13): услуга из состояния снята, а сводка ещё держит «Ферритин» до следующего обновления — потребитель
    контекста, который снятие не видит. Набор `next` — единственный, где пятый ход действительно снимает услугу;
    в остальных пятый ход показывает цену «Гастрина» (правила), и красное там — тот же нарушенный инвариант:
    сводка называет услугу, которой в ответе нет."""

    scenario = SCENARIOS["not_found_gastro"]
    turns = ["сколько стоит ферритин"] * 4 + [scenario.first]
    played = _play(turns, "summary", mini_prod_world, monkeypatch, config, llm=scenario.llm())
    last = played[-1]
    shown = _normalise_catalog_text(last.text)
    held = re.findall(r"service_name=([^,\n]+)", last.summary)
    assert "service_name=Ферритин" in played[3].summary  # четвёртый ход обновил сводку — до снятия услуги
    assert all(_normalise_catalog_text(value) in shown for value in held), (held, last.state["service_name"])


@pytest.mark.parametrize("config", CONFIGS)
def test_repeat_guard_counts_the_answer_text_not_the_service(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Анти-повтор читает текст ответа: два разных короткых вопроса подряд с одним уточнением — второе уточнение
    получает оффер оператора (`_maybe_offer_operator_on_repeat`). Ни услуга, ни понимание хода тут не читаются;
    если перевод изменит текст уточнения («по какой из услуг…»), счётчик повторов поедет вместе с ним."""

    played = _play([_LIST_REPLY, "сколько стоит?", "а цена?"], "repeat-guard", mini_prod_world, monkeypatch, config)
    clarify = "Скажите, пожалуйста, название услуги/анализа — я уточню стоимость."
    assert played[1].text == clarify and not played[1].handoff
    assert played[2].text.startswith(clarify) and "Перевести на оператора?" in played[2].text


@pytest.mark.parametrize("config", CONFIGS)
def test_turn_telemetry_carries_counts_and_statuses_not_phrases(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Телеметрия хода (`tu_*`, `price_policy`, `choose[]`) — числа и статусы: ни фраз пациента, ни названий услуг,
    ни самой реплики (ПДн и принцип «строка телеметрии не читается как текст»). Что понято и выбрано — нигде
    не записано: по телеметрии нельзя восстановить, о какой услуге ответил ход."""

    played = _play([_LIST_REPLY, "сколько стоит?", "сколько стоит ферритин"], "telemetry", mini_prod_world, monkeypatch, config)
    rows = [row for turn in played for row in turn.telemetry]
    assert len(rows) == 3
    dump = json.dumps(rows, ensure_ascii=False).lower()
    for word in ("ферритин", "ттг", "витамин", "глюкоз", "сколько", "стоит"):
        assert word not in dump, word
    assert rows[0]["tu_list"] == 3 and rows[2]["tu_list"] == 0
