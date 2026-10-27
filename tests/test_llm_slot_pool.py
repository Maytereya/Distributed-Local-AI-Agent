"""Пул слотов LLM и очередь под дедлайном хода (BUG-2026-10-27-LLM-QUEUE-GIVES-UP, `llm_turn_deadline_s`).

Пул заменил `asyncio.Semaphore`: без дедлайна очередь та же — по приходу; под дедлайном слот получает ход, начатый раньше
(начатые ходы добегают, классификатор нового хода ждёт), ожидание — до «остаток дедлайна − p95 задачи», а не `queue_timeout_ms`
места вызова. Слот не теряется: ушедший по тайм-ауту не занимает его, переданный уходящему — передаётся дальше.
"""

from __future__ import annotations

import asyncio
import contextvars
import time

import pytest

from messengers_router import llm_runtime
from messengers_router.runtime_config import config as cfg


async def _order_of(pool: llm_runtime._SlotPool, keys: list[float | None]) -> list[int]:
    got: list[int] = []

    async def wait(i: int, key: float | None) -> None:
        if key is None:
            await pool.acquire()
        else:
            await pool.acquire_by(key)
        got.append(i)
        pool.release()

    await pool.acquire()  # слот занят — все ждут
    tasks = []
    for i, key in enumerate(keys):
        tasks.append(asyncio.ensure_future(wait(i, key)))
        await asyncio.sleep(0)
    pool.release()
    await asyncio.gather(*tasks)
    return got


def test_acquire_is_fifo_like_a_semaphore():
    assert asyncio.run(_order_of(llm_runtime._SlotPool(1), [None, None, None])) == [0, 1, 2]


def test_acquire_by_serves_the_smallest_key_first():
    assert asyncio.run(_order_of(llm_runtime._SlotPool(1), [3.0, 1.0, 2.0])) == [1, 2, 0]


def test_timed_out_waiter_leaves_no_trace():
    async def run() -> tuple[bool, bool]:
        pool = llm_runtime._SlotPool(1)
        await pool.acquire()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(pool.acquire_by(0.0), timeout=0.05)
        pool.release()
        free_after = not pool.locked()
        await asyncio.wait_for(pool.acquire(), timeout=0.05)  # слот свободен — без ожидания
        return free_after, pool.locked()

    assert asyncio.run(run()) == (True, True)


def test_slot_handed_to_a_leaving_waiter_passes_on():
    """Гонка: слот передан ждущему, а его в тот же миг отменили — слот уходит следующему, а не пропадает."""

    async def run() -> list[str]:
        pool = llm_runtime._SlotPool(1)
        await pool.acquire()
        got: list[str] = []

        async def wait(name: str, key: float) -> None:
            await pool.acquire_by(key)
            got.append(name)

        first = asyncio.ensure_future(wait("first", 1.0))
        second = asyncio.ensure_future(wait("second", 2.0))
        await asyncio.sleep(0)
        pool.release()  # слот передан «first»…
        first.cancel()  # …а он уходит раньше, чем проснулся
        await asyncio.sleep(0)
        await asyncio.wait_for(second, timeout=0.5)
        return got

    assert asyncio.run(run()) == ["second"]


def _in_turn(started_ago: float, deadline_s: float) -> llm_runtime._TurnBudget:
    now = time.monotonic()
    budget = llm_runtime._TurnBudget(started_at=now - started_ago, deadline_at=now - started_ago + deadline_s)
    llm_runtime._TURN_BUDGET.set(budget)
    return budget


def test_under_deadline_the_turn_started_earlier_gets_the_slot_first(monkeypatch):
    monkeypatch.setattr(llm_runtime, "_SEMAPHORE", llm_runtime._SlotPool(1))

    async def run() -> list[str]:
        pool = llm_runtime._SEMAPHORE
        await pool.acquire()  # сосед держит слот
        got: list[str] = []

        async def call(name: str, started_ago: float) -> None:
            _in_turn(started_ago, 30.0)
            async with llm_runtime.llm_slot(1500, reserve_s=1.0):
                got.append(name)

        fresh = asyncio.ensure_future(contextvars.copy_context().run(lambda: call("новый ход", 0.0)))
        await asyncio.sleep(0)
        older = asyncio.ensure_future(contextvars.copy_context().run(lambda: call("начатый ход", 5.0)))
        await asyncio.sleep(0)
        pool.release()
        await asyncio.gather(fresh, older)
        return got

    assert asyncio.run(run()) == ["начатый ход", "новый ход"]


def test_under_deadline_the_queue_waits_past_the_call_site_timeout(monkeypatch):
    """`queue_timeout_ms` места вызова (здесь 1 с — минимум прежней очереди) под дедлайном не читается."""

    monkeypatch.setattr(llm_runtime, "_SEMAPHORE", llm_runtime._SlotPool(1))

    async def run() -> float:
        budget = _in_turn(0.0, 5.0)
        pool = llm_runtime._SEMAPHORE
        await pool.acquire()
        asyncio.get_running_loop().call_later(1.3, pool.release)
        started = time.monotonic()
        async with llm_runtime.llm_slot(1000, reserve_s=1.0):
            waited = time.monotonic() - started
        assert budget.queue_gaveup == 0 and budget.deadline_hits == 0
        return waited

    assert asyncio.run(run()) >= 1.25


def test_no_time_left_for_the_task_is_refused_at_once(monkeypatch):
    monkeypatch.setattr(llm_runtime, "_SEMAPHORE", llm_runtime._SlotPool(1))

    async def run() -> llm_runtime._TurnBudget:
        budget = _in_turn(9.0, 10.0)  # осталась 1 с, задаче нужно 2
        with pytest.raises(llm_runtime.LLMDeadlineError):
            async with llm_runtime.llm_slot(30000, reserve_s=2.0):
                pass
        return budget

    budget = asyncio.run(run())
    assert budget.deadline_hits == 1 and budget.queue_gaveup == 0
    assert llm_runtime._outcome(llm_runtime.LLMDeadlineError()) == "deadline"
    assert isinstance(llm_runtime.LLMDeadlineError(), llm_runtime.LLM_UNAVAILABLE_ERRORS)  # политики сбоя мест вызова те же


def test_without_deadline_the_call_site_queue_timeout_holds(monkeypatch):
    monkeypatch.setattr(llm_runtime, "_SEMAPHORE", llm_runtime._SlotPool(1))

    async def run() -> llm_runtime._TurnBudget:
        llm_runtime._TURN_BUDGET.set(llm_runtime._TurnBudget())
        budget = llm_runtime._TURN_BUDGET.get()
        await llm_runtime._SEMAPHORE.acquire()
        with pytest.raises(llm_runtime.LLMQueueTimeoutError) as caught:
            async with llm_runtime.llm_slot(1000):
                pass
        assert not isinstance(caught.value, llm_runtime.LLMDeadlineError)
        return budget

    budget = asyncio.run(run())
    assert budget.queue_gaveup == 1 and budget.deadline_hits == 0


def test_shadow_calls_ignore_the_deadline(monkeypatch):
    """Тень (L-15b) — вне хода: ни дедлайна, ни приоритета, своя очередь места вызова."""

    monkeypatch.setattr(llm_runtime, "_SEMAPHORE", llm_runtime._SlotPool(1))

    async def run() -> llm_runtime._TurnBudget:
        budget = _in_turn(9.5, 10.0)
        async with llm_runtime.llm_slot(1000, reserve_s=2.0, shadow=True):
            pass
        return budget

    assert asyncio.run(run()).deadline_hits == 0


def test_begin_turn_budget_opens_the_deadline_from_config(monkeypatch):
    monkeypatch.setattr(cfg, "MR_LLM_TURN_DEADLINE_S", 50, raising=False)
    token = llm_runtime.begin_turn_budget()
    try:
        left = llm_runtime.turn_deadline_left()
        assert left is not None and 49.0 < left <= 50.0
        assert llm_runtime.turn_load_note() == {"llm_deadline_s": 50.0}
    finally:
        llm_runtime.end_turn_budget(token)
    monkeypatch.setattr(cfg, "MR_LLM_TURN_DEADLINE_S", 0, raising=False)
    token = llm_runtime.begin_turn_budget()
    try:
        assert llm_runtime.turn_deadline_left() is None and llm_runtime.turn_load_note() == {}
    finally:
        llm_runtime.end_turn_budget(token)


def test_reserve_is_the_task_p95_capped_by_its_generation_timeout():
    assert llm_runtime._reserve_s("classifier", 45) == llm_runtime._P95_S["classifier"]
    assert llm_runtime._reserve_s("choose_price_service", 8.0) == llm_runtime._P95_DEFAULT_S
    assert llm_runtime._reserve_s("choose_walkin", 1.5) == 1.5
