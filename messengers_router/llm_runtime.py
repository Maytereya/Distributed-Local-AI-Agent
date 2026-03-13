"""Единый runtime для вызовов локальной LLM с ограничением конкуренции.

Ответственность модуля: оборачивать вызовы Ollama, контролировать очередь/таймауты
и отдавать безопасные helper-функции для text/json/stream генерации.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from ollama import AsyncClient

from agent_logic_2 import config as c, ollama_settings
from agent_logic_2.ollama_settings import LLMName

_DEFAULT_MAX_CONCURRENCY = 5
try:
    _MAX_CONCURRENCY = int(str(c.MR_LLM_MAX_CONCURRENCY).strip() or _DEFAULT_MAX_CONCURRENCY)
except Exception:
    _MAX_CONCURRENCY = _DEFAULT_MAX_CONCURRENCY
_SEMAPHORE = asyncio.Semaphore(max(1, _MAX_CONCURRENCY))
_OLLAMA_CLIENT = AsyncClient(c.ollama_url)


class LLMQueueTimeoutError(RuntimeError):
    pass


def _extract_response_text(res: Any) -> str:
    if isinstance(res, dict):
        raw = res.get("response")
    else:
        raw = getattr(res, "response", None)
        if raw is None and hasattr(res, "model_dump"):
            try:
                dumped = res.model_dump()
                if isinstance(dumped, dict):
                    raw = dumped.get("response")
            except Exception:
                raw = None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return raw
    return ""


@asynccontextmanager
async def llm_slot(queue_timeout_ms: int) -> AsyncGenerator[None, None]:
    timeout_s = max(1.0, float(queue_timeout_ms) / 1000.0)
    try:
        await asyncio.wait_for(_SEMAPHORE.acquire(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise LLMQueueTimeoutError("llm_queue_timeout") from e
    try:
        yield
    finally:
        _SEMAPHORE.release()


async def generate_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    llm: str | None = None,
    think: bool | None = None,
) -> str:
    model = llm or LLMName.get()
    resolved_think = ollama_settings.resolve_think(think)
    async with llm_slot(queue_timeout_ms):
        res = await asyncio.wait_for(
            _OLLAMA_CLIENT.generate(
                model=model,
                prompt=prompt,
                options=ollama_settings.options_set(),
                format=fmt,
                keep_alive=-1,
                think=resolved_think,
            ),
            timeout=max(5, int(timeout_s)),
        )
    return _extract_response_text(res)


async def generate_stream_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    llm: str | None = None,
    think: bool | None = None,
) -> AsyncGenerator[str, None]:
    model = llm or LLMName.get()
    resolved_think = ollama_settings.resolve_think(think)
    async with llm_slot(queue_timeout_ms):
        stream = await asyncio.wait_for(
            _OLLAMA_CLIENT.generate(
                model=model,
                prompt=prompt,
                options=ollama_settings.options_set(),
                stream=True,
                keep_alive=-1,
                think=resolved_think,
            ),
            timeout=max(5, int(timeout_s)),
        )
        async for chunk in stream:
            text = _extract_response_text(chunk)
            if text:
                yield text
