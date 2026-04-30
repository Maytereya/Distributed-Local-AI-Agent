"""Единый runtime для вызовов локальной LLM с ограничением конкуренции.

Ответственность модуля: оборачивать вызовы Ollama, контролировать очередь/таймауты
и отдавать безопасные helper-функции для text/json/stream генерации.
"""

from __future__ import annotations

import asyncio
import json as _json
import os as _os
import time as _time
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path as _Path
from typing import Any, AsyncGenerator

from ollama import AsyncClient

from agent_logic_2 import ollama_settings
from agent_logic_2.ollama_settings import LLMName

from .runtime_config import config as c

_DEFAULT_MAX_CONCURRENCY = 5
try:
    _MAX_CONCURRENCY = int(str(c.MR_LLM_MAX_CONCURRENCY).strip() or _DEFAULT_MAX_CONCURRENCY)
except Exception:
    _MAX_CONCURRENCY = _DEFAULT_MAX_CONCURRENCY
_SEMAPHORE = asyncio.Semaphore(max(1, _MAX_CONCURRENCY))
_OLLAMA_CLIENT = AsyncClient(c.ollama_url)


# --- DEBUG: файловый лог промптов и ответов ---
# Включается env-переменной MR_LLM_LOG_PROMPTS=1.
# Пишет одну JSON-line на каждое событие в /tmp/llm_prompts.jsonl
# События: prompt_request (до вызова), prompt_response (после успеха),
#          prompt_timeout (на TimeoutError), prompt_error (на любое исключение).
# Цель: захватить реальный промпт даже при зависании LLM, сравнить
# отличие между быстрыми ("Расписание хирург") и медленными
# ("Расписание уролог") запросами.
_PROMPT_LOG_PATH = _Path(_os.getenv("MR_LLM_LOG_PATH", "/tmp/llm_prompts.jsonl"))
# Маркер-файл: если /tmp/llm_prompts.enabled существует → пишем prompt-лог.
# Так не нужно перезапускать сервер — touch/rm включает/выключает на лету.
_PROMPT_LOG_MARKER = _Path("/tmp/llm_prompts.enabled")


def _prompt_log_enabled() -> bool:
    if str(_os.getenv("MR_LLM_LOG_PROMPTS", "")).strip() in {"1", "true", "True"}:
        return True
    try:
        return _PROMPT_LOG_MARKER.exists()
    except Exception:
        return False


def _log_prompt_event(event: str, payload: dict[str, Any]) -> None:
    if not _prompt_log_enabled():
        return
    record = {"ts": _time.time(), "event": event, **payload}
    try:
        _PROMPT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROMPT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    except Exception:
        # Не падаем, если лог-файл недоступен — debug-фича.
        pass


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


def _extract_usage(res: Any) -> dict[str, int]:
    if isinstance(res, dict):
        dumped = res
    elif hasattr(res, "model_dump"):
        try:
            raw = res.model_dump()
            dumped = raw if isinstance(raw, dict) else {}
        except Exception:
            dumped = {}
    else:
        dumped = {}

    usage: dict[str, int] = {}
    for key in ("prompt_eval_count", "eval_count", "total_duration", "prompt_eval_duration", "eval_duration"):
        value = dumped.get(key)
        try:
            parsed = int(value)
        except Exception:
            continue
        usage[key] = parsed
    return usage


async def generate_text_with_usage(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    llm: str | None = None,
    think: bool | None = None,
) -> tuple[str, dict[str, int]]:
    model = llm or LLMName.get()
    resolved_think = ollama_settings.resolve_think(think)
    request_id = _uuid.uuid4().hex[:12] if _prompt_log_enabled() else ""
    started_at = _time.monotonic()
    if _prompt_log_enabled():
        _log_prompt_event(
            "prompt_request",
            {
                "id": request_id,
                "model": model,
                "fmt": fmt,
                "timeout_s": int(timeout_s),
                "prompt_len": len(prompt),
                "prompt": prompt,
            },
        )
    try:
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
    except asyncio.TimeoutError:
        if _prompt_log_enabled():
            _log_prompt_event(
                "prompt_timeout",
                {"id": request_id, "duration_s": round(_time.monotonic() - started_at, 3)},
            )
        raise
    except Exception as exc:
        if _prompt_log_enabled():
            _log_prompt_event(
                "prompt_error",
                {
                    "id": request_id,
                    "duration_s": round(_time.monotonic() - started_at, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                },
            )
        raise
    text = _extract_response_text(res)
    usage = _extract_usage(res)
    if _prompt_log_enabled():
        _log_prompt_event(
            "prompt_response",
            {
                "id": request_id,
                "duration_s": round(_time.monotonic() - started_at, 3),
                "response_len": len(text),
                "response": text,
                "usage": usage,
            },
        )
    return text, usage


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
    text, _usage = await generate_text_with_usage(
        prompt,
        timeout_s=timeout_s,
        queue_timeout_ms=queue_timeout_ms,
        fmt=fmt,
        llm=llm,
        think=think,
    )
    return text


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
