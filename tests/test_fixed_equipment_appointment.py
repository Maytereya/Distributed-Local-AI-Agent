"""Fixed-equipment (флюорограф/маммограф) перехватывает и ЗАПИСЬ, не только расписание.

Прод #668: «Записаться на флюорографию» прошёл полный booking-флоу («Да, можем
записать на Флюорография… ФИО… Подтверждаете?»), хотя у клиники нет приёмного
врача-радиолога и запись ведётся ТОЛЬКО через регистратуру Ленина 5.
Перехват `resolve_diagnostic_fixed_addresses` стоял в doctors_schedule_week
(расписание) и addresses (адрес), но НЕ в APPOINTMENT-пути.

Инвариант (класс): запись на процедуру с фиксированным оборудованием
(флюорография/маммография) не запускает сбор слотов, а сразу отдаёт honest
handoff «через регистратуру Ленина 5». Отмена такой записи (cancel) —
не перехватывается (обычный путь). Запись на обычную услугу — без изменений.
"""

from __future__ import annotations

from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.memory import MemoryStore
from messengers_router.planner import build_plan
from messengers_router.response_builder import build_appointment_step_response


def _decision(service: str, action: str = "book") -> RouteDecision:
    ent = {"appointment_action": action}
    if service:
        ent["service_name"] = service
    return RouteDecision(label="APPOINTMENT", confidence=0.9, entities=ent)


def _state() -> SessionState:
    return SessionState(session_id="fixed-equip")


def test_fluorography_booking_marks_fixed_equipment_no_slot_collection():
    state = _state()
    plan = build_plan(_decision("Флюорография"), state, "Записаться на флюорографию", memory=MemoryStore())
    assert state.last_entities.get("_appointment_fixed_equipment") is True
    assert plan.steps == [], "запись на флюорографию не должна собирать слоты"


def test_mammography_booking_also_intercepted():
    state = _state()
    build_plan(_decision("Маммография"), state, "хочу записаться на маммографию", memory=MemoryStore())
    assert state.last_entities.get("_appointment_fixed_equipment") is True


def test_fixed_equipment_by_user_text_without_service_entity():
    """Перехват работает и когда service_name пуст, но текст про флюорографию."""
    state = _state()
    build_plan(_decision(""), state, "запишите на флюорографию", memory=MemoryStore())
    assert state.last_entities.get("_appointment_fixed_equipment") is True


def test_ordinary_service_not_intercepted():
    state = _state()
    build_plan(_decision("УЗИ брюшной полости"), state, "записаться на узи брюшной полости", memory=MemoryStore())
    assert state.last_entities.get("_appointment_fixed_equipment") is not True


def test_cancel_fluorography_not_intercepted():
    """Отмена записи на флюорографию — обычный cancel, не fixed-equipment handoff."""
    state = _state()
    build_plan(_decision("Флюорография", action="cancel"), state, "отмени флюорографию", memory=MemoryStore())
    assert state.last_entities.get("_appointment_fixed_equipment") is not True


def test_response_builder_emits_registry_handoff():
    state = _state()
    state.last_entities["_appointment_fixed_equipment"] = True
    resp = build_appointment_step_response("APPOINTMENT", Evidence(), state, None, MemoryStore())
    assert resp is not None
    assert resp.handoff is True
    low = resp.text.lower()
    assert "регистратур" in low and "ленина" in low
    assert "_appointment_fixed_equipment" not in state.last_entities, "маркер снят"
