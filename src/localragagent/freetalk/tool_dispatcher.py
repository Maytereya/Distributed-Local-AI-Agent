"""Tool call dispatcher for FreeTalk."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from .contracts import (
    TOOL_OUTCOME_ERROR,
    TOOL_OUTCOME_NOT_FOUND,
    TOOL_OUTCOME_OK,
    TOOL_OUTCOME_TECH_UNAVAILABLE,
    ToolCallResult,
)
from .observability import log_event

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
        ignored = {
            "note",
            "entities_used",
            "status",
            "reason",
            "checked_at",
            "source_counts",
            "handoff_required",
            "cache_file",
            "top_n_applied",
            "service_kind",
            "service_name",
            "prepare_wrap_status",
            "prepare_wrap_reason",
        }
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


def _payload_requests_handoff(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if bool(payload.get("handoff_required")):
        return True
    return bool(str(payload.get("handoff_message") or "").strip())


def _payload_attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("attachments")
    if not isinstance(raw, list):
        raw = payload.get("result_attachments")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _payload_outcome(payload: dict[str, Any], *, found: bool) -> str:
    if not isinstance(payload, dict):
        return TOOL_OUTCOME_NOT_FOUND
    explicit = str(payload.get("ft_outcome") or payload.get("outcome") or "").strip().lower()
    if explicit in {
        TOOL_OUTCOME_OK,
        TOOL_OUTCOME_NOT_FOUND,
        TOOL_OUTCOME_TECH_UNAVAILABLE,
        TOOL_OUTCOME_ERROR,
    }:
        return explicit

    note = str(payload.get("note") or "").strip().lower()
    status = str(payload.get("status") or payload.get("reason") or "").strip().lower()
    handoff_reason = str(payload.get("handoff_reason") or "").strip().lower()
    markers = " ".join(part for part in (note, status, handoff_reason) if part)
    if "tech_unavailable" in markers or "source unavailable" in markers:
        return TOOL_OUTCOME_TECH_UNAVAILABLE
    if "not_found" in markers or "not_ready" in markers:
        return TOOL_OUTCOME_NOT_FOUND
    if "missing_result_fields" in markers:
        return TOOL_OUTCOME_OK
    return TOOL_OUTCOME_OK if found else TOOL_OUTCOME_NOT_FOUND


class ToolDispatcher:
    def __init__(self, handlers: dict[str, ToolHandler]) -> None:
        self._handlers = dict(handlers)

    async def call(self, tool_name: str, query: str, entities: dict[str, Any] | None = None) -> ToolCallResult:
        payload: dict[str, Any] = {}
        args = dict(entities or {})
        handler = self._handlers.get(tool_name)
        if handler is None:
            log_event(
                "tool_not_registered",
                level=logging.ERROR,
                tool_name=tool_name,
            )
            return ToolCallResult(
                tool_name=tool_name,
                payload={},
                found=False,
                error=f"tool_not_registered:{tool_name}",
                outcome=TOOL_OUTCOME_ERROR,
            )
        try:
            raw_payload = await handler(query, args)
            if isinstance(raw_payload, dict):
                payload = raw_payload
            else:
                payload = {"result": raw_payload}
        except Exception as exc:
            log_event(
                "tool_call_failed",
                level=logging.ERROR,
                tool_name=tool_name,
                error_type=type(exc).__name__,
            )
            return ToolCallResult(
                tool_name=tool_name,
                payload={},
                found=False,
                error=f"{type(exc).__name__}:{exc}",
                outcome=TOOL_OUTCOME_TECH_UNAVAILABLE,
            )

        found = _has_useful_data(payload)
        return ToolCallResult(
            tool_name=tool_name,
            payload=payload,
            found=found,
            error="",
            outcome=_payload_outcome(payload, found=found),
            degraded=bool(payload.get("degraded")) if isinstance(payload, dict) else False,
            handoff=_payload_requests_handoff(payload),
            attachments=_payload_attachments(payload),
        )
