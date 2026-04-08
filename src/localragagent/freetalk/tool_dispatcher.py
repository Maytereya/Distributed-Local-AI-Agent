"""Tool call dispatcher for FreeTalk."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .contracts import ToolCallResult

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _has_useful_data(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        if text.lower() in {"no query", "missing_result_fields"}:
            return False
        return True
    if isinstance(value, dict):
        ignored = {"note", "entities_used", "status", "reason", "checked_at", "source_counts", "handoff_required"}
        for key, item in value.items():
            if key in ignored:
                continue
            if _has_useful_data(item):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        if not value:
            return False
        return any(_has_useful_data(item) for item in value)
    return True


class ToolDispatcher:
    def __init__(self, handlers: dict[str, ToolHandler]) -> None:
        self._handlers = dict(handlers)

    async def call(self, tool_name: str, query: str, entities: dict[str, Any] | None = None) -> ToolCallResult:
        payload: dict[str, Any] = {}
        args = dict(entities or {})
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolCallResult(tool_name=tool_name, payload={}, found=False, error=f"tool_not_registered:{tool_name}")
        try:
            raw_payload = await handler(query, args)
            if isinstance(raw_payload, dict):
                payload = raw_payload
            else:
                payload = {"result": raw_payload}
        except Exception as exc:
            return ToolCallResult(tool_name=tool_name, payload={}, found=False, error=f"{type(exc).__name__}:{exc}")

        return ToolCallResult(
            tool_name=tool_name,
            payload=payload,
            found=_has_useful_data(payload),
            error="",
        )

