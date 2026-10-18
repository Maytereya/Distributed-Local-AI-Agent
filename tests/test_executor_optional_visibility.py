"""Необязательный шаг плана: погашенный handoff и ошибка видны — ход (д), шаг 6а, DATA-4 (17.10).

Было: в `executor.py` не было логгера; необязательный шаг (`required=False` — например, расписание в активной записи),
который попросил оператора или упал, уходил в ключи `{tool}_optional_suppressed` / `{tool}_optional_error`, которые
никто не читает, — запись шла дальше без окон, и следа не оставалось ни в логе, ни в телеметрии.

Стало (поведение не меняется — гашение остаётся, его смена — шаг 6 под выключателем): `logger.warning` (у ошибки — с
трассировкой) и поле телеметрии хода `optional_suppressed` — список инструментов. Упавший обязательный шаг — тоже
`logger.warning` с трассировкой (перевод на оператора, как было).

Через эндпоинт этот путь в мире мини-прода не достижим (врачей в мире нет — планировщик не ставит шаг расписания);
сквозная проверка — стенд сбоев (шаг 2 плана хода (д)).
"""

from __future__ import annotations

import asyncio
import logging

from messengers_router import executor, turn_telemetry
from messengers_router.mess_types import Plan, PlanStep, SessionState


class _Svc:
    async def price_info(self, _q, _e):
        return {"handoff_required": True, "handoff_reason": "service_error", "note": "price unavailable", "prices": []}

    async def address_info(self, _q, _e):
        raise RuntimeError("МИС недоступна")


def _run(steps: list[PlanStep]) -> tuple[object, dict]:
    token = turn_telemetry.begin("test", session_id="optional-visibility")
    ev = asyncio.run(executor.execute_plan(Plan(label="APPOINTMENT", steps=steps), SessionState(session_id="optional-visibility"), _Svc()))
    row = turn_telemetry.finish(token, handoff=bool(ev.get("handoff_required")), chars=0)
    return ev, row or {}


def test_suppressed_handoff_and_failed_optional_step_are_logged_and_counted(caplog, monkeypatch):
    monkeypatch.setattr(turn_telemetry, "_emit", lambda row: None)
    steps = [
        PlanStep(tool="price_info", input={"query": "q", "entities": {}}, required=False),
        PlanStep(tool="address_info", input={"query": "q", "entities": {}}, required=False),
    ]

    with caplog.at_level(logging.WARNING, logger=executor.logger.name):
        ev, row = _run(steps)

    # Поведение прежнее: handoff погашен, ход продолжается.
    assert ev.get("handoff_required") is None
    assert ev.get("price_info_optional_suppressed")["handoff_reason"] == "service_error"
    assert "МИС недоступна" in ev.get("address_info_optional_error")["message"]
    # Видимость.
    assert row.get("optional_suppressed") == ["price_info", "address_info"]
    messages = [(r.getMessage(), r.exc_info is not None) for r in caplog.records]
    assert any("price_info" in m and "погашен" in m for m, _ in messages), messages
    assert any("address_info" in m and has_trace for m, has_trace in messages), messages


def test_failed_required_step_is_logged_and_still_hands_off(caplog, monkeypatch):
    monkeypatch.setattr(turn_telemetry, "_emit", lambda row: None)

    with caplog.at_level(logging.WARNING, logger=executor.logger.name):
        ev, row = _run([PlanStep(tool="address_info", input={"query": "q", "entities": {}}, required=True)])

    assert ev.get("handoff_required") is True and ev.get("handoff_reason") == "service_error"
    assert "optional_suppressed" not in row
    assert any("address_info" in r.getMessage() and r.exc_info for r in caplog.records)


def test_no_optional_trouble_leaves_no_field(monkeypatch):
    monkeypatch.setattr(turn_telemetry, "_emit", lambda row: None)

    class _Ok:
        async def price_info(self, _q, _e):
            return {"prices": [{"serviceName": "A", "price": 1}]}

    token = turn_telemetry.begin("test", session_id="optional-ok")
    asyncio.run(executor.execute_plan(Plan(label="PRICE", steps=[PlanStep(tool="price_info", input={"query": "q", "entities": {}}, required=False)]), SessionState(session_id="optional-ok"), _Ok()))
    row = turn_telemetry.finish(token, handoff=False, chars=0) or {}

    assert "optional_suppressed" not in row
