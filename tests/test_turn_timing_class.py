"""Класс: журнал хода показывает, где секунды вне LLM и где кончился лимит решений хода (ход (е), 22.10).

Инварианты:
- каждый шаг плана даёт время в `tool_ms` под именем инструмента — и удачный, и упавший, и необязательный с отказом;
  повтор инструмента суммируется; в поле только имена инструментов и числа, без запроса пациента;
- `tu_decisions_denied` — число вызовов LLM хода с исходом `budget` (лимит решений / время хода), тени не считаются;
  нет отказов — поля нет;
- `tu_decisions_spent` пишет эндпоинт (сколько решений хода потрачено).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import executor, turn_telemetry
from messengers_router.mess_types import Plan, PlanStep, SessionState


class _Services:
    async def price_info(self, q, ent):
        await asyncio.sleep(0.02)
        return {"ok": True}

    async def test_prepare(self, q, ent):
        raise RuntimeError("срез недоступен")


@pytest.fixture
def rows(monkeypatch):
    captured: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", captured.append)
    monkeypatch.setattr(turn_telemetry, "enabled", lambda: True)
    return captured


def _turn(plan: Plan) -> dict:
    async def run() -> dict:
        token = turn_telemetry.begin("test", session_id="s-timing")
        await executor.execute_plan(plan, SessionState(session_id="s-timing"), _Services())
        return turn_telemetry.finish(token, handoff=False, chars=0)

    return asyncio.run(run())


def test_tool_ms_for_ok_failed_and_repeated_steps(rows) -> None:
    secret = "сколько стоит ферритин у иванова"
    plan = Plan(
        label="PRICE",
        steps=[
            PlanStep(tool="price_info", input={"query": secret}),
            PlanStep(tool="price_info", input={"query": secret}),
            PlanStep(tool="test_prepare", input={"query": secret}, required=False),
        ],
    )
    row = _turn(plan)
    assert set(row["tool_ms"]) == {"price_info", "test_prepare"}
    assert row["tool_ms"]["price_info"] >= 30  # два шага по ~20 мс — сумма
    assert all(isinstance(v, int) for v in row["tool_ms"].values())
    assert secret not in str(row["tool_ms"])


def test_required_failed_step_still_timed(rows) -> None:
    row = _turn(Plan(label="PREPARE", steps=[PlanStep(tool="test_prepare", input={"query": "x"})]))
    assert "test_prepare" in row["tool_ms"]


def test_decisions_denied_counts_budget_not_shadow(rows) -> None:
    async def run() -> dict:
        token = turn_telemetry.begin("test", session_id="s-budget")
        turn_telemetry.note_llm("choose_price_service", "ok", 900)
        turn_telemetry.note_llm("choose_price_service_qualifier", "budget", 0)
        turn_telemetry.note_llm("choose_price_service_qualifier", "budget", 0)
        turn_telemetry.note_llm("classifier", "budget", 0, shadow=True)
        return turn_telemetry.finish(token, handoff=False, chars=0)

    row = asyncio.run(run())
    assert row["tu_decisions_denied"] == 2


def test_no_denied_field_without_budget_refusals(rows) -> None:
    async def run() -> dict:
        token = turn_telemetry.begin("test", session_id="s-ok")
        turn_telemetry.note_llm("classifier", "ok", 900)
        return turn_telemetry.finish(token, handoff=False, chars=0)

    assert "tu_decisions_denied" not in asyncio.run(run())
