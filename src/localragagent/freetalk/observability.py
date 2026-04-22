"""Logging helpers for FreeTalk package."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("localragagent.freetalk")


def _render_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(fields):
        value = fields.get(key)
        if value is None:
            continue
        text = str(value).replace("\n", "\\n").strip()
        if not text:
            continue
        if len(text) > 300:
            text = text[:300] + "...(truncated)"
        parts.append(f"{key}={text}")
    return " ".join(parts)


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    payload = _render_fields(fields)
    message = f"FreeTalkAI component=freetalk event={event}"
    if payload:
        message = f"{message} {payload}"
    LOGGER.log(level, message)
