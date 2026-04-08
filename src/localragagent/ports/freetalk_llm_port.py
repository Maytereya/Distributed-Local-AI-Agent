"""Port to legacy LLM runtime for FreeTalk."""

from __future__ import annotations

import importlib

from .legacy import import_legacy_alias

import_legacy_alias("messengers_router")


async def generate_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    think: bool | None = False,
) -> str:
    llm_runtime = importlib.import_module("localragagent.messengers_router.llm_runtime")
    return await llm_runtime.generate_text(
        prompt,
        timeout_s=int(timeout_s),
        queue_timeout_ms=int(queue_timeout_ms),
        fmt=fmt,
        think=think,
    )


async def generate_text_with_usage(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    think: bool | None = False,
) -> tuple[str, dict[str, int]]:
    llm_runtime = importlib.import_module("localragagent.messengers_router.llm_runtime")
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
