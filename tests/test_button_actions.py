"""Шаг 2 кнопок: нажатие задаёт тему, подпись кнопки не становится содержанием (29.09).

Меню живёт у шлюза; сюда приходят кнопки-листья идентификатором `button_id`.
Подпись кнопки приходит в `text` (у API min_length=1), но это не слова
пациента. Зонд 29.09 по проду: подпись «Цена приёма врача», прочитанная как
текст, дала услугу «приём врача» и цену ХИРУРГА; «Подготовка к процедуре» —
поиск правил подготовки к «процедуре». Поэтому:

* кнопка-уточнение — тема из кнопки, сущностей из подписи ноль; недостающее
  спрашивает штатный планировщик и запоминает контекст (pending);
* кнопка-перевод — готовый текст и оператор;
* обычная кнопка — подпись — полноценный запрос, идёт как текст;
* любое нажатие — явная смена темы: висящий вопрос (дата записи, «да/нет»)
  подпись кнопки не получает.
"""

import asyncio
from typing import get_args

import pytest

from messengers_router import button_menu
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Label, ResponseEnvelope, RouteDecision, SessionState
from messengers_router.orchestrator import OrchestratorContext, nlu_route
from messengers_router.services import Services

# Контракт со шлюзом: «Таблица экранов, версия 1» — кнопки, которые доходят до бота.
GATEWAY_V1_TITLES = {
    "menu.address": "Адреса и часы работы",
    "menu.operator": "Оператор",
    "menu.appointment.book": "Записаться на приём",
    "menu.appointment.schedule": "Когда принимает врач",
    "menu.appointment.change": "Перенести или отменить",
    "menu.appointment.prepare": "Подготовка к процедуре",
    "menu.tests.where": "Где и до скольки принимают",
    "menu.tests.booking_needed": "Нужно ли записываться",
    "menu.tests.prepare": "Как подготовиться к анализу",
    "menu.tests.suggest": "Подобрать анализы",
    "menu.tests.price": "Сколько стоит анализ",
    "menu.results.lookup": "Ввести данные",
    "menu.results.missing": "Результат не пришёл",
    "menu.results.eta": "Когда будет готов",
    "menu.price.test": "Цена анализа",
    "menu.price.doctor": "Цена приёма врача",
    "menu.price.promo": "Акции и скидки",
    "menu.docs.tax": "Справка для вычета",
    "menu.docs.contract": "Копия договора, выписка",
    "menu.docs.sick_leave": "Больничный лист",
}


def _ids(mode: str) -> list[str]:
    return sorted(b for b, a in button_menu.BUTTONS.items() if a.mode == mode)


# --- таблица -------------------------------------------------------------------


def test_table_matches_gateway_screen_table_v1():
    assert set(button_menu.BUTTONS) == set(GATEWAY_V1_TITLES)


def test_every_button_maps_to_valid_label():
    labels = set(get_args(Label))
    bad = {b: a.label for b, a in button_menu.BUTTONS.items() if a.label not in labels}
    assert not bad


def test_handoff_buttons_have_patient_text():
    assert all(button_menu.BUTTONS[b].text for b in _ids("handoff"))


def test_unknown_button_id_has_no_action():
    assert button_menu.action_for("removed.button") is None
    assert button_menu.action_for("") is None


# --- решение на стадии nlu_route ----------------------------------------------


def _route(button_id: str, monkeypatch, nlu_decision: RouteDecision | None = None):
    calls = []

    async def fake_nlu(*, user_text, state, services, memory, runtime_options):
        calls.append(user_text)
        return (nlu_decision or RouteDecision(label="OTHER", confidence=0.5, source="nlu")), {}

    monkeypatch.setattr(router_mod, "_resolve_nlu_decision_before_doctor_guard", fake_nlu)
    title = GATEWAY_V1_TITLES.get(button_id, "Неизвестная кнопка")
    ctx = OrchestratorContext(text=title, state=SessionState(session_id="b"), button_id=button_id)
    ctx = asyncio.run(nlu_route(ctx, services=object(), memory=MemoryStore()))
    return ctx, calls


@pytest.mark.parametrize("button_id", _ids("slot"))
def test_slot_button_sets_topic_and_takes_nothing_from_title(button_id, monkeypatch):
    ctx, nlu_calls = _route(button_id, monkeypatch)
    action = button_menu.BUTTONS[button_id]
    assert nlu_calls == []  # подпись не разбирается NLU вовсе
    assert ctx.decision.label == action.label
    assert ctx.decision.entities == dict(action.entities)  # только смысл кнопки
    assert ctx.decision.source == "button"


@pytest.mark.parametrize("button_id", _ids("handoff"))
def test_handoff_button_answers_with_its_text(button_id, monkeypatch):
    ctx, nlu_calls = _route(button_id, monkeypatch)
    assert nlu_calls == []
    assert ctx.decision.needs_handoff is True
    assert ctx.decision.clarify_needed is True
    assert ctx.decision.clarify_reason == button_menu.BUTTONS[button_id].text


@pytest.mark.parametrize("button_id", [*_ids("pass"), "removed.button"])
def test_pass_and_unknown_buttons_go_through_nlu(button_id, monkeypatch):
    marker = RouteDecision(label="ADDRESS", confidence=0.9, source="nlu")
    ctx, nlu_calls = _route(button_id, monkeypatch, nlu_decision=marker)
    assert nlu_calls == [GATEWAY_V1_TITLES.get(button_id, "Неизвестная кнопка")]
    assert ctx.decision is marker


# --- класс: подпись кнопки никогда не ищется как цена ---------------------------


def _services_recording_prices(calls: list):
    # Цену планировщик ищет двумя инструментами: price_info и — когда услуга уже
    # «известна» — service_bundle_info. Записываем оба: подпись не должна уйти ни в один.
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    def recorder(tool):
        async def record(query, entities, *args, **kwargs):
            calls.append((tool, query))
            return {}

        return record

    services.price_info = recorder("price_info")
    services.service_bundle_info = recorder("service_bundle_info")
    return services


def _stream(text, state, services, memory, **kwargs):
    async def collect():
        return [e async for e in router_mod.patient_routing_stream(text, state, services, memory, **kwargs)]

    return asyncio.run(collect())


@pytest.mark.parametrize("button_id", [b for b in _ids("slot") if button_menu.BUTTONS[b].label == "PRICE"])
def test_price_button_asks_for_service_instead_of_searching_title(button_id):
    calls: list = []
    state = SessionState(session_id=f"p-{button_id}")
    memory = MemoryStore()
    _stream(GATEWAY_V1_TITLES[button_id], state, _services_recording_prices(calls), memory, button_id=button_id)
    assert calls == []  # по подписи цену не ищем
    pending = memory.get_pending(state) or {}
    assert pending.get("label") == "PRICE"
    assert "service_name" in (pending.get("missing") or [])


# --- нажатие — явная смена темы -------------------------------------------------


def _capture_pipeline(monkeypatch, seen: dict):
    async def fake_run_pipeline(text, state, services=None, memory=None, runtime_options=None, **kwargs):
        seen["entities"] = dict(state.last_entities)
        seen["pending"] = memory.get_pending(state) if memory else None
        ctx = OrchestratorContext(text=text, state=state)
        ctx.decision = RouteDecision(label="PRICE", confidence=1.0, source="button")
        ctx.response = ResponseEnvelope(text="Скажите название анализа.")
        return ctx

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)


def test_button_press_drops_previous_topic(monkeypatch):
    seen: dict = {}
    _capture_pipeline(monkeypatch, seen)
    state = SessionState(session_id="switch-1")
    memory = MemoryStore()
    state.last_entities["service_name"] = "Ферритин"
    memory.set_pending(state, label="PRICE", missing_slots=["_any_of:city,branch_name,branch_id"])
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    _stream("Цена анализа", state, services, memory, button_id="menu.price.test")
    assert "service_name" not in seen["entities"]
    assert seen["pending"] is None


def test_button_press_is_not_consumed_by_active_appointment_flow(monkeypatch):
    seen: dict = {}
    _capture_pipeline(monkeypatch, seen)
    state = SessionState(session_id="switch-2")
    memory = MemoryStore()
    state.last_entities.update({"appointment_flow_active": True, "specialty": "лор"})
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["_any_of:date_from,time_from,date_hint"])
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    out = _stream("Цена анализа", state, services, memory, button_id="menu.price.test")
    assert "entities" in seen  # дошли до конвейера — предпроверка записи не перехватила
    assert out[-1].text == "Скажите название анализа."


# --- класс: нажатие не запускает догадку каталога ------------------------------


def _press(button_id: str):
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    state = SessionState(session_id=f"press-{button_id}")
    out = _stream(GATEWAY_V1_TITLES[button_id], state, services, MemoryStore(), button_id=button_id)
    return "".join(e.text for e in out if e.text)


@pytest.mark.parametrize("button_id", sorted(GATEWAY_V1_TITLES))
def test_button_press_never_guesses_a_catalog_service(button_id):
    # Воронка услуг (CLASS-2026-09-28-SERVICE-FUNNEL-SWALLOWS-PROPERTY): слово
    # подписи или смысла кнопки («анализы») уходит в каталог, и бот предлагает
    # первую попавшуюся услугу — «Анализ крови на аминокислоты». Нажатие кнопки
    # такую догадку вызывать не вправе: пациент ещё ничего не назвал.
    assert "имели в виду" not in _press(button_id).lower()


def test_booking_needed_button_answers_walk_in_rule():
    # Кнопка из раздела «Сдать анализы»: штатный walk-in ответ, а не оффер записи.
    assert "без записи" in _press("menu.tests.booking_needed")


def test_eta_button_does_not_answer_something_else():
    # «Когда будет готов» — срок готовности анализа. Бот сроков не знает: даже
    # «когда будет готов анализ на ферритин» текстом даёт ЦЕНУ (замер 29.09), а
    # кнопка как slot TEST_ASSIST спрашивала «для какой цели подобрать анализы».
    # Пока срока нет в ответах бота — честный перевод на оператора.
    ctx_text = _press("menu.results.eta")
    assert "подобрать анализы" not in ctx_text.lower()
    assert button_menu.BUTTONS["menu.results.eta"].mode == "handoff"
