"""Класс: отказ лимита решений хода отличим от сбоя LLM (ход (е), DESIGN-2026-10-22-TURN-DECISION-BUDGET).

Инварианты: `choose()` при исчерпанном лимите — UNAVAILABLE с причиной `budget` (сбой модели — без причины);
`llm_runtime.turn_budget_exhausted()` — True ровно когда решений хода больше нет; вне хода — False.
"""

from __future__ import annotations

import asyncio

from ollama import GenerateResponse

from messengers_router import booking_slots, choice, llm_runtime
from messengers_router.choice import Status

# Настоящая задача с промптом в двух локациях — у пробной без промпта `choose()` честно отвечает «промпт сломан».
_TASK = booking_slots.TEST_ASSIST_FACET_TASK
_OPTIONS = booking_slots._TEST_ASSIST_FACET_OPTIONS


class _Ok:
    async def generate(self, **_kw):
        return GenerateResponse(model="m", response='{"picks": [1]}', done=True, prompt_eval_count=1, eval_count=1)


class _Down:
    async def generate(self, **_kw):
        raise ConnectionError("down")


def _run(client, monkeypatch, *, limit: int, n: int) -> list:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_LLM_DECISION_TURN_CALLS", limit, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", client)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())

    async def run() -> list:
        token = llm_runtime.begin_turn_budget()
        try:
            out = []
            for i in range(n):
                choice._CACHE.clear()
                out.append((await choice.choose(_TASK, f"вопрос {i}", _OPTIONS), llm_runtime.turn_budget_exhausted()))
            return out
        finally:
            llm_runtime.end_turn_budget(token)

    return asyncio.run(run())


def test_budget_denial_has_reason_budget(monkeypatch) -> None:
    res = _run(_Ok(), monkeypatch, limit=2, n=3)
    assert [r[0].status for r in res] == [Status.PICKED, Status.PICKED, Status.UNAVAILABLE]
    assert res[2][0].reason == "budget"
    assert [r[1] for r in res] == [False, True, True]


def test_model_failure_is_not_budget(monkeypatch) -> None:
    res = _run(_Down(), monkeypatch, limit=3, n=1)
    assert res[0][0].status is Status.UNAVAILABLE and res[0][0].reason == ""
    assert res[0][1] is False


def test_outside_turn_not_exhausted() -> None:
    assert llm_runtime.turn_budget_exhausted() is False
