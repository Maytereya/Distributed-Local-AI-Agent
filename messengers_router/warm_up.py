"""Прогрев после старта: справочники и модель LLM загружаются до первого пациента.

Замер 03.10: первый запрос после старта процесса — 66 с против 25 с у прогретого. В
запросе пациента разбирались справочники МИС (serviceInfoAll — около 45 МБ), строились
словари синонимов, тянулись памятки базы знаний; после перезапуска ollama ещё и модель
грузилась в память GPU. Прогрев делает это фоновой задачей при старте сервиса: старт его
не ждёт, сбой шага не мешает остальным.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

_tasks: set[asyncio.Task[Any]] = set()


def _steps() -> tuple[tuple[str, Callable[[], Any]], ...]:
    from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info

    from .services import _biomaterial
    from .services._prepare_select_llm import kb_patient_memos

    return (
        ("прайс Самары", lambda: api_price.load_price_by_region(api_price.BOT_PRICE_REGION_ID)),
        ("справочник услуг МИС", api_service_info.load_service_info),
        ("словари синонимов и биоматериалов", _biomaterial._vocabularies),
        ("цены врачей", api_price.load_doctor_prices),
        ("врачи", api_nayka.get_cached_doctors_data),
        ("памятки базы знаний", kb_patient_memos),
    )


async def _warm_llm() -> None:
    """Модель — в память GPU: после перезапуска ollama первый вызов грузит её десятки секунд."""

    from .llm_runtime import generate_text

    started = time.monotonic()
    try:
        await generate_text("Ответь одним словом: да", timeout_s=120, queue_timeout_ms=5000, think=False)
        log.info("warm_up: модель LLM — %.1f с", time.monotonic() - started)
    except Exception as exc:
        log.warning("warm_up: модель LLM не прогрета: %s", type(exc).__name__)


async def warm_up() -> None:
    """Справочники по очереди (в потоке — event loop свободен), затем модель LLM."""

    for name, step in _steps():
        started = time.monotonic()
        try:
            await asyncio.to_thread(step)
            log.info("warm_up: %s — %.1f с", name, time.monotonic() - started)
        except Exception as exc:
            log.warning("warm_up: %s не прогрет: %s", name, exc)
    await _warm_llm()


def schedule_warm_up() -> None:
    """Запустить прогрев фоновой задачей; ссылка хранится, чтобы задачу не собрал GC."""

    task = asyncio.get_running_loop().create_task(warm_up())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
