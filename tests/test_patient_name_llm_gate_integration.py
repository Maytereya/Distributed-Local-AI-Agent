"""Интеграция LLM-валидатора ФИО в router-prelock (прод #668).

Активная запись, бот ждёт ФИО. «Время изменени» проходит форму, но валидатор
(замокан) говорит «не ФИО» → router через штатный __clear_patient_name сбрасывает
имя и переспрашивает. Валидное «Иванов Иван» валидатор пропускает → запись идёт.

Гейт стоит в единой async-точке (_complete_route_after_doctor_guard), а не в
трёх формах-приёмниках — минимум касания хрупкого флоу.
"""

from __future__ import annotations

import asyncio

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState


def run(coro):
    return asyncio.run(coro)


def _state_waiting_name() -> tuple[SessionState, MemoryStore]:
    memory = MemoryStore()
    state = SessionState(session_id="pn-gate")
    state.last_entities.update({
        "appointment_flow_active": True,
        "doctor_name": "Трубин",
        "branch_name": "Ленина 5",
        "date_from": "2026-07-16",
        "time_from": "11:00",
        "_last_label": "APPOINTMENT",
    })
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])
    return state, memory


def _resolve(user_text, state, memory):
    services = router_mod.Services()
    decision, _dbg = run(
        router_mod._resolve_nlu_decision_before_doctor_guard(user_text, state, services, memory)
    )
    return decision


def test_non_name_reply_rejected_and_reasked(monkeypatch):
    """Валидатор «не ФИО» → patient_name не принят, запись переспрашивает."""
    async def _reject(_text):
        return False

    monkeypatch.setattr(router_mod, "is_patient_name_reply", _reject)

    state, memory = _state_waiting_name()
    decision = _resolve("Время изменени", state, memory)
    assert decision.entities.get("patient_name") in (None, ""), "не-ФИО не должно стать patient_name"
    assert state.last_entities.get("patient_name") in (None, "")


def test_valid_name_accepted(monkeypatch):
    """Валидатор «ФИО» → имя принято, запись движется к подтверждению."""
    async def _accept(_text):
        return True

    monkeypatch.setattr(router_mod, "is_patient_name_reply", _accept)

    state, memory = _state_waiting_name()
    decision = _resolve("Иванов Иван Иванович", state, memory)
    assert decision.entities.get("patient_name") == "Иванов Иван Иванович"


def test_validator_not_called_when_not_waiting_name(monkeypatch):
    """Анти-over-trigger: валидатор не зовётся, если ФИО не ждём (другой слот)."""
    called = {"n": 0}

    async def _spy(_text):
        called["n"] += 1
        return True

    monkeypatch.setattr(router_mod, "is_patient_name_reply", _spy)

    memory = MemoryStore()
    state = SessionState(session_id="pn-other-slot")
    state.last_entities.update({"appointment_flow_active": True, "doctor_name": "Трубин"})
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["date_from"])
    _resolve("на завтра в 10:00", state, memory)
    assert called["n"] == 0, "валидатор ФИО зовётся только когда ждём именно ФИО"
