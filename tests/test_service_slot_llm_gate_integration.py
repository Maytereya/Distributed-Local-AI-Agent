"""Интеграция LLM-валидатора слота услуги в router-prelock (аудит 2026-07-22).

Активная запись, бот ждёт услугу/врача. «Поменяй» проходит форму
`_SERVICE_SINGLE_WORD_RE`, но валидатор (замокан «OTHER») говорит «не услуга» →
router через штатный __clear_service_name сбрасывает услугу и переспрашивает.
Валидную услугу валидатор пропускает. Многословная услуга валидатор не дёргает
(командо-риск — только одно слово).

Гейт стоит в единой async-точке prelock-обработки (рядом с ФИО-гейтом), а не в
форме-приёмнике — минимум касания хрупкого флоу.
"""

from __future__ import annotations

import asyncio

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState


def run(coro):
    return asyncio.run(coro)


def _state_waiting_service() -> tuple[SessionState, MemoryStore]:
    memory = MemoryStore()
    state = SessionState(session_id="svc-gate")
    state.last_entities.update({
        "appointment_flow_active": True,
        "_last_label": "APPOINTMENT",
    })
    memory.set_pending(
        state,
        label="APPOINTMENT",
        missing_slots=["_any_of:doctor_id,doctor_name,specialty,service_name"],
    )
    return state, memory


def _resolve(user_text, state, memory):
    services = router_mod.Services()
    decision, _dbg = run(
        router_mod._resolve_nlu_decision_before_doctor_guard(user_text, state, services, memory)
    )
    return decision


def test_command_word_rejected_and_not_stored_as_service(monkeypatch):
    """Валидатор «OTHER» → командное слово не оседает в service_name."""
    async def _reject(_text):
        return False

    monkeypatch.setattr(router_mod, "is_service_name_reply", _reject)

    state, memory = _state_waiting_service()
    decision = _resolve("Поменяй", state, memory)
    assert decision.entities.get("service_name") in (None, ""), (
        f"команда осела в service_name: {decision.entities.get('service_name')!r}"
    )
    assert state.last_entities.get("service_name") in (None, "")


def test_valid_single_word_service_accepted(monkeypatch):
    """Валидатор «SERVICE» → однословная услуга принята."""
    async def _accept(_text):
        return True

    monkeypatch.setattr(router_mod, "is_service_name_reply", _accept)

    state, memory = _state_waiting_service()
    decision = _resolve("торакоцентез", state, memory)
    assert "торакоцентез" in str(decision.entities.get("service_name") or "").lower()


def test_multiword_service_does_not_call_validator(monkeypatch):
    """Командо-риск — только одно слово: многословную услугу валидатор не дёргает."""
    called = {"n": 0}

    async def _spy(_text):
        called["n"] += 1
        return True

    monkeypatch.setattr(router_mod, "is_service_name_reply", _spy)

    state, memory = _state_waiting_service()
    decision = _resolve("УЗИ брюшной полости", state, memory)
    assert called["n"] == 0, "валидатор не должен дёргаться на многословной услуге"
    assert "узи" in str(decision.entities.get("service_name") or "").lower()
