"""Прогрев после старта (03.10): справочники и модель LLM — до первого пациента.

Справочники МИС, словари синонимов, памятки базы знаний и (после перезапуска ollama)
модель LLM грузились в первом запросе пациента. Инварианты:
прогрев проходит все шаги, даже если один упал; старт сервиса его не ждёт; планировщики
ежедневных обновлений запускаются на старте, а не с первым запросом.
"""

from __future__ import annotations

import asyncio

from messengers_router import endpoint, warm_up


def test_warm_up_runs_every_step_even_if_one_fails(monkeypatch):
    done: list[str] = []

    def failing():
        raise ConnectionError("МИС недоступна")

    monkeypatch.setattr(
        warm_up,
        "_steps",
        lambda: (("прайс", lambda: done.append("прайс")), ("справочник", failing), ("врачи", lambda: done.append("врачи"))),
    )

    async def fake_llm():
        done.append("LLM")

    monkeypatch.setattr(warm_up, "_warm_llm", fake_llm)

    asyncio.run(warm_up.warm_up())

    assert done == ["прайс", "врачи", "LLM"]


def test_service_start_does_not_wait_for_warm_up(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_warm_up():
        started.set()
        await release.wait()

    monkeypatch.setattr(warm_up, "warm_up", slow_warm_up)
    monkeypatch.setattr(endpoint, "get_services", lambda: None)

    async def service_start():
        await asyncio.wait_for(endpoint._on_startup(), timeout=1)  # не ждёт прогрева
        await asyncio.wait_for(started.wait(), timeout=1)  # а прогрев уже идёт
        release.set()

    asyncio.run(service_start())


def test_startup_handler_is_registered():
    assert endpoint._on_startup in endpoint.router.on_startup
