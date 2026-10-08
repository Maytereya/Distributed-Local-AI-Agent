"""Отказ стоп-листа — с оффером оператора (L-06в, ревью 05.10; решение владельца 08.10).

Стоп-лист «не оказываем» — список в коде и бывает неправ (трафик 08.10: отказ на рентген,
который в клинике есть). Инвариант: после отказа пациент может одним «да» уйти к оператору;
без «да» — ничего принудительно, обычный разговор продолжается.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.response_builder import UNSUPPORTED_CATALOG_OPERATOR_OFFER
from messengers_router.services import Services


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )


def _say(text, state, memory, services):
    async def go():
        out = [e async for e in router_mod.patient_routing_stream(text, state, services, memory)]
        return "".join(e.text for e in out if e.text), any(e.handoff for e in out)

    return asyncio.run(go())


@pytest.mark.parametrize("question", ["Где сделать МРТ?", "Нужна справка в ГИБДД", "Есть ли у вас психиатр?"])
def test_refusal_offers_operator_and_yes_hands_off(question):
    state, memory, services = SessionState(session_id=f"stop-{question}"), MemoryStore(), Services()
    services.ensure_background_refresh_started = lambda: None

    refusal, handoff = _say(question, state, memory, services)
    assert refusal.endswith(UNSUPPORTED_CATALOG_OPERATOR_OFFER), refusal
    assert handoff is False

    _answer, handoff = _say("да", state, memory, services)
    assert handoff is True
