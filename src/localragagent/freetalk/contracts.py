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
class DialogAct:
    route: str = "general"
    intent: str = "unknown"
    entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    missing_slots: list[str] = field(default_factory=list)
    clarify_question: str = ""
    tool_plan: list[str] = field(default_factory=list)
    response_policy: str = "general_only"
    source: str = "llm_router"
    fallback_reason: str = ""


@dataclass(slots=True)
class PostToolVerification:
    enough_data: bool = True
    should_clarify: bool = False
    clarify_question: str = ""
    answer_policy: str = "direct"
    source: str = "heuristic"


@dataclass(slots=True)
class AgentReply:
    text: str
    source: str
    tool_name: str = ""
    tool_payload: dict[str, Any] = field(default_factory=dict)
    source_fragments: list[dict[str, str]] = field(default_factory=list)
    next_session_id: str = ""
