"""Port to legacy LLM runtime for FreeTalk."""

from __future__ import annotations

import importlib
import logging

from .legacy import import_legacy_alias
from .observability import log_port_event

import_legacy_alias("messengers_router")


async def generate_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    think: bool | None = False,
) -> str:
    try:
        llm_runtime = importlib.import_module("localragagent.messengers_router.llm_runtime")
        return await llm_runtime.generate_text(
            prompt,
            timeout_s=int(timeout_s),
            queue_timeout_ms=int(queue_timeout_ms),
            fmt=fmt,
            think=think,
        )
    except Exception as exc:
        log_port_event(
            "llm_port_generate_text_failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
        )
        raise


async def generate_text_with_usage(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    think: bool | None = False,
) -> tuple[str, dict[str, int]]:
    try:
        llm_runtime = importlib.import_module("localragagent.messengers_router.llm_runtime")
    except Exception as exc:
        log_port_event(
            "llm_port_runtime_import_failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
        )
        raise
    if hasattr(llm_runtime, "generate_text_with_usage"):
        text, usage = await llm_runtime.generate_text_with_usage(
            prompt,
            timeout_s=int(timeout_s),
            queue_timeout_ms=int(queue_timeout_ms),
            fmt=fmt,
            think=think,
        )
        usage_map = usage if isinstance(usage, dict) else {}
        return str(text or ""), {str(k): int(v) for k, v in usage_map.items() if isinstance(k, str)}

    text = await llm_runtime.generate_text(
        prompt,
        timeout_s=int(timeout_s),
        queue_timeout_ms=int(queue_timeout_ms),
        fmt=fmt,
        think=think,
    )
    return str(text or ""), {}
