"""Contracts for FreeTalk orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SessionContext:
    session_id: str
    summary: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ToolCallResult:
    tool_name: str
    payload: dict[str, Any]
    found: bool
    error: str = ""


@dataclass(slots=True)
class AgentReply:
    text: str
    source: str
    tool_name: str = ""
    tool_payload: dict[str, Any] = field(default_factory=dict)
    next_session_id: str = ""
