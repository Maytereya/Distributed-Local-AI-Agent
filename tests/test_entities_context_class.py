"""Шаг 5.1 хода (б): контекст разговора после хода — только показанное или понятое из данных (STEP-2026-10-30-STEP5-1-CONTEXT).

Дизайн `~/Dev/handoffs/2026-10-30_step5_entities_design.md` (решение владельца 30.10: Ш5-1 — «контекст» и «форма» — разные поля,
Ш5-3 — один ключ `entities_by_understanding` со списком читателей). Класс П-1 / Д3 ночи 29→30.10: в состоянии оказывалось то, чего
пациент не видел (фраза «ВПЧ», «Нему подготовиться»), и следующий короткий вопрос («а сколько стоит?») отвечал о ней.

Инварианты (читатель `context`):
- И-5.5: значение `service_name` / `test_name`, записанное ходом, после хода — показанное пациенту (строка ответа или напечатано в
  тексте) или точное название из данных МИС (прайс, услуги врачей); фраза пациента — нет;
- И-5.2а: назвал новое, но ответ ничего не показал — контекст очищается вместе с показанным прошлым ответом;
- открытая форма (запись, вопрос-слот темы, вопрос о наркозе) — вход висящего вопроса, точка фиксации её не трогает (шаг 5.1б);
- значение, которое ход не менял, — не трогается; выключатель off — как было; неизвестный читатель — ошибка конфига при старте.
"""

from __future__ import annotations

import pytest

from messengers_router import button_flow, shown_context, turn_understanding
from messengers_router.mess_types import AppointmentPhase, SessionState
from messengers_router.runtime_config import config as rcfg
from messengers_router.services import _prices_helpers

_PRICE_ROWS = [{"serviceName": "Ферритин", "serviceId": 1}, {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceId": 2}]
_DOCTOR_ROWS = [{"serviceName": "Ультразвуковое исследование щитовидной железы", "doctorId": 7}]


@pytest.fixture
def ctx(monkeypatch):
    """Читатель `context` включён; данные МИС — маленький прайс и услуги врачей (без срезов)."""

    from agent_logic_2.nayka_api import api_price

    monkeypatch.setattr(rcfg, "MR_ENTITIES_BY_UNDERSTANDING", "context", raising=False)
    monkeypatch.setattr(_prices_helpers, "_catalog_rows", lambda rows=None: list(_PRICE_ROWS))
    monkeypatch.setattr(api_price, "load_doctor_prices", lambda: list(_DOCTOR_ROWS))
    return monkeypatch


def _turn(before: dict, write: dict, text: str, *, shown: list[str] | None = None) -> dict:
    """Ход: состояние на начало, что записал ход (`write`, `None` — снял), показал ли ответ строки (`shown`), текст ответа."""

    state = SessionState(session_id="ctx", last_entities=dict(before))
    shown_context.begin_turn(state)
    for key, value in write.items():
        if value is None:
            state.last_entities.pop(key, None)
        else:
            state.last_entities[key] = value
    if shown is not None:
        shown_context.remember(state, shown)
    shown_context.commit_context(state, text)
    return state.last_entities


def test_patient_phrase_not_shown_is_not_context(ctx):
    """«а ВПЧ?» → оффер: фраза не показана — не контекст; показанное прошлым ответом (ферритин) — тоже (И-5.2а)."""

    after = _turn({"service_name": "Ферритин", "_shown_services": ["Ферритин"]}, {"service_name": "ВПЧ"}, "По этому вопросу точно ответит оператор. Перевести?")
    assert "service_name" not in after and "_shown_services" not in after, after


@pytest.mark.parametrize(
    ("write", "text", "shown"),
    [
        ({"service_name": "Общий анализ крови (Le, Er, Hb, СОЭ)"}, "Цена — 390 руб.", ["Общий анализ крови (Le, Er, Hb, СОЭ)"]),
        ({"service_name": "Папилломавирус 16/18"}, "Памятка клиники «Папилломавирус 16/18»: соскоб …", None),
        ({"service_name": "Ферритин"}, "Анализы сдаются без записи. Адреса: …", None),
        ({"service_name": "Ультразвуковое исследование щитовидной железы"}, "В городе Самара доступны филиалы: …", None),
        ({"test_name": "Ферритин"}, "Как подготовиться: натощак.", None),
    ],
    ids=["shown_row", "printed_in_text", "price_name", "doctor_service_name", "test_name_from_data"],
)
def test_shown_or_understood_from_data_stays(ctx, write, text, shown):
    after = _turn({}, write, text, shown=shown)
    key = next(iter(write))
    assert after.get(key) == write[key], after


def test_named_new_but_other_rows_shown_keeps_shown(ctx):
    """Ход показал строки, а в `service_name` осталась фраза пациента: фраза снимается, показанное этого хода — нет."""

    after = _turn({}, {"service_name": "оак капиллярный"}, "Варианты: 1. … 2. …", shown=["Ферритин", "Общий анализ крови (Le, Er, Hb, СОЭ)"])
    assert "service_name" not in after and after["_shown_services"] == ["Ферритин", "Общий анализ крови (Le, Er, Hb, СОЭ)"], after


def test_value_not_changed_by_the_turn_is_untouched(ctx):
    after = _turn({"service_name": "узи сосудов шеи"}, {}, "Адреса: …")
    assert after["service_name"] == "узи сосудов шеи", after


@pytest.mark.parametrize(
    "before",
    [
        {"_pending": {"label": "ADDRESS", "missing": ["_any_of:city,branch_name,branch_id"]}},
        {"_pending": {"label": "PREPARE", "missing": ["service_name"]}},
        {"_last_label": "APPOINTMENT"},
        {"_prepare_variant_pending": {"with": "…", "without": "…"}},
    ],
    ids=["address_slot", "prepare_slot", "booking", "anesthesia_question"],
)
def test_open_form_keeps_its_input(ctx, before):
    """Открытая форма — значение её вход («анализы» для пунктов сдачи, цель записи): поле формы, шаг 5.1б."""

    after = _turn(before, {"service_name": "анализы"}, "Уточните город, пожалуйста.")
    assert after.get("service_name") == "анализы", after


def test_booking_phase_keeps_the_target(ctx):
    state = SessionState(session_id="ctx-book", last_entities={})
    state.dialog.phase = AppointmentPhase.COLLECTING
    shown_context.begin_turn(state)
    state.last_entities["service_name"] = "запись на узи"
    shown_context.commit_context(state, "Какой филиал вам удобен?")
    assert state.last_entities["service_name"] == "запись на узи"


def test_off_is_as_before(ctx):
    ctx.setattr(rcfg, "MR_ENTITIES_BY_UNDERSTANDING", "off", raising=False)
    after = _turn({"_shown_services": ["Ферритин"]}, {"service_name": "ВПЧ"}, "Перевести на оператора?")
    assert after["service_name"] == "ВПЧ" and after["_shown_services"] == ["Ферритин"], after


@pytest.mark.parametrize(("raw", "readers", "errors"), [("off", set(), 0), ("context", {"context"}, 0), ("context, slots", {"context"}, 1), ("", set(), 0)])
def test_reader_list_and_unknown_readers(monkeypatch, raw, readers, errors):
    """Ключ — список читателей; неизвестный читатель не действует и громко виден при старте (Н-7, `config_conflicts`)."""

    monkeypatch.setattr(rcfg, "MR_ENTITIES_BY_UNDERSTANDING", raw, raising=False)
    assert turn_understanding.entity_readers() == frozenset(readers)
    assert len(turn_understanding.entities_config_errors()) == errors
    assert sum("entities_by_understanding" in text for text in button_flow.config_conflicts()) == errors


def test_seam_every_plan_step_carries_the_turn_understanding(monkeypatch):
    """Шов шага 5 (STEP-2026-10-30-STEP5-SEAM): `router.build_plan` кладёт понимание хода во вход каждого шага плана — тот же объект,
    реплика хода; исполнитель его не читает (поведение — гейт), debug его не сериализует."""

    from messengers_router import router as router_mod
    from messengers_router.mess_types import Evidence
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services
    from tests.endpoint_turn import route_via_endpoint

    seen: list[list[object]] = []

    async def capture(plan, _state, _services):
        seen.append([step.understanding for step in plan.steps])
        return Evidence()

    monkeypatch.setattr(router_mod, "execute_plan", capture)
    state = SessionState(session_id="seam")
    route_via_endpoint("сколько стоит ферритин", state, Services(), MemoryStore())
    assert seen and seen[0], seen
    understood = {id(item) for item in seen[0]}
    assert len(understood) == 1 and seen[0][0] is not None and getattr(seen[0][0], "reply", "") == "сколько стоит ферритин"


# Шаг 5.1б — поле формы (STEP-2026-10-30-STEP5-1B-FORM): значения открытой формы — её вход; при закрытии формы они проверяются, как
# новые: вход формы, которого пациент не видел («анализы» для пунктов сдачи), контекстом следующего хода не становится.
_ADDRESS_FORM = {"_pending": {"label": "ADDRESS", "missing": ["_any_of:city,branch_name,branch_id"]}}


def _form_then_close(ctx, value: str, close_text: str) -> dict:
    opened = _turn(_ADDRESS_FORM, {"service_name": value}, "Уточните город, пожалуйста.")
    assert opened.get("_form") == {"service_name": value}, opened  # форма открыта — значение в поле формы
    closed = {key: val for key, val in opened.items() if key != "_pending"}  # ответ «Самара» закрыл вопрос-слот
    return _turn(closed, {}, close_text)


def test_form_input_not_shown_does_not_become_context(ctx):
    after = _form_then_close(ctx, "анализы", "В городе Самара пункты сдачи: ул. Пензенская, 71 …")
    assert "service_name" not in after and "_form" not in after, after


@pytest.mark.parametrize(("value", "text"), [("Ферритин", "Пункты сдачи: …"), ("ЭКГ", "ЭКГ делают по адресам: …")], ids=["data_name", "printed"])
def test_form_value_shown_or_from_data_stays_context(ctx, value, text):
    after = _form_then_close(ctx, value, text)
    assert after.get("service_name") == value and "_form" not in after, after


def test_form_value_dropped_keeps_what_was_shown_before(ctx):
    """Снят вход формы, но пациент ничего нового не называл: показанное раньше (ферритин) остаётся — это не И-5.2а."""

    opened = _turn({**_ADDRESS_FORM, "_shown_services": ["Ферритин"]}, {"service_name": "анализы"}, "Уточните город, пожалуйста.")
    closed = {key: val for key, val in opened.items() if key != "_pending"}
    after = _turn(closed, {}, "Пункты сдачи: …")
    assert "service_name" not in after and after.get("_shown_services") == ["Ферритин"], after


def test_form_field_never_reaches_the_tools(ctx):
    """`_form` — состояние хода: в сущности инструментов (и через `entities_used` — в промпт свободного ответа) не идёт."""

    from messengers_router import planner
    from messengers_router.mess_types import RouteDecision
    from messengers_router.memory import MemoryStore

    state = SessionState(session_id="form-tools", last_entities={"service_name": "Ферритин", "_form": {"service_name": "анализы"}})
    plan = planner.build_plan(RouteDecision(label="DOCTOR_INFO", confidence=0.9, entities={}), state, "кто делает?", MemoryStore())
    entities = [step.input.get("entities") or {} for step in plan.steps]
    assert entities and all("_form" not in ent and ent.get("service_name") == "Ферритин" for ent in entities), entities


# Шаг 5.3а — читатель `price` (STEP-2026-10-30-STEP5-3A-PRICE-CONTEXT): в реплике есть упоминание услуги (разбор хода), а в состоянии
# — контекст прошлого хода: контекст не идёт в путь цены (резолвер, «действующее» название, контекст выбора LLM).
def _price_plan(monkeypatch, readers: str, *, mention: bool, written_now: bool, state_out: dict | None = None) -> list[dict]:
    from messengers_router import planner
    from messengers_router.mess_types import RouteDecision
    from messengers_router.memory import MemoryStore
    from messengers_router.turn_understanding import ServiceMention, TurnUnderstanding

    reply = "а ещё ттг"
    monkeypatch.setattr(rcfg, "MR_ENTITIES_BY_UNDERSTANDING", readers, raising=False)
    monkeypatch.setattr(rcfg, "MR_TURN_UNDERSTANDING_ONE", "on", raising=False)
    monkeypatch.setattr(rcfg, "MR_TURN_UNDERSTANDING_PRICE", "on", raising=False)
    state = SessionState(session_id="p53", last_entities={"service_name": "Ферритин", "city": "Самара"})
    shown_context.begin_turn(state)
    if written_now:
        state.last_entities["service_name"] = "ТТГ (TSH) тиреотропный гормон"  # записал этот ход (подстановка каталога)
    services = (ServiceMention(phrase="ттг", candidates=(), strength="weak", chosen=None, source="parse"),) if mention else ()
    token = turn_understanding._CURRENT.set(TurnUnderstanding(services=services, facet="price", reply=reply, names_service=True))
    try:
        plan = planner.build_plan(RouteDecision(label="PRICE", confidence=0.9, entities={}), state, reply, MemoryStore())
    finally:
        turn_understanding._CURRENT.reset(token)
    if state_out is not None:
        state_out.update(state.last_entities)
    return [{"tool": step.tool, **(step.input.get("entities") or {})} for step in plan.steps]


def test_price_reader_drops_previous_context_when_the_reply_names_a_service(monkeypatch):
    """И из инструмента, и из состояния: рендер цены читает состояние — «Не нашёл… для «Ферритин»» на «а ггтп?» (стенд 30.10)."""

    after: dict = {}
    [step] = _price_plan(monkeypatch, "price", mention=True, written_now=False, state_out=after)
    assert step["tool"] == "price_info" and "service_name" not in step and "service_name" not in after, (step, after)


@pytest.mark.parametrize(
    ("readers", "mention", "written_now"),
    [("off", True, False), ("price", False, False), ("price", True, True)],
    ids=["off_as_before", "no_mention_condition_follow_up", "service_written_this_turn"],
)
def test_price_reader_keeps_the_service_otherwise(monkeypatch, readers, mention, written_now):
    """Выключатель off; в реплике нет упоминания услуги («а по омс?» — контекст и есть предмет); услугу записал этот ход — как было."""

    [step] = _price_plan(monkeypatch, readers, mention=mention, written_now=written_now)
    assert step.get("service_name"), step


# Шаг 5.3б — читатель `price_select` (STEP-2026-10-30-STEP5-3B-PRICE-SELECT): название этого хода подставлено по буквам, а в разборе
# реплики есть упоминание — строку выбирает LLM (`price_info`), а не правила карточки услуги (`service_bundle_info`).
@pytest.mark.parametrize(
    ("readers", "mention", "written_now", "tool"),
    [
        ("price_select", True, True, "price_info"),
        ("off", True, True, "service_bundle_info"),  # как было (консилиум 21.10, В2)
        ("price_select", False, True, "service_bundle_info"),  # разбор услуги не нашёл — как было
        ("price_select", True, False, "service_bundle_info"),  # контекст прошлого хода — читатель `price`, не этот
    ],
    ids=["letters_name_goes_to_llm_select", "off", "no_parse_mention", "carried_context"],
)
def test_price_select_reader(monkeypatch, readers, mention, written_now, tool):
    [step] = _price_plan(monkeypatch, readers, mention=mention, written_now=written_now)
    assert step["tool"] == tool and (tool != "price_info" or "service_name" not in step), step
    if tool == "price_info":
        # Буквенное название — кандидат пула выбора LLM, а не потерянное (карточка 30.10: «Immulite» выпадала из пула).
        assert step["__catalog_hint"] == "ТТГ (TSH) тиреотропный гормон", step
