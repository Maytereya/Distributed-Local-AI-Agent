"""Contracts for FreeTalk orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOOL_OUTCOME_OK = "ok"
TOOL_OUTCOME_NOT_FOUND = "not_found"
TOOL_OUTCOME_TECH_UNAVAILABLE = "tech_unavailable"
TOOL_OUTCOME_ERROR = "error"


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
    outcome: str = TOOL_OUTCOME_OK
    degraded: bool = False
    handoff: bool = False
    attachments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class DialogAct:
    route: str = "general"
    intent: str = "unknown"
    entities: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    missing_slots: list[str] = field(default_factory=list)
    clarify_type: str = ""
    clarify_question: str = ""
    tool_plan: list[str] = field(default_factory=list)
    response_policy: str = "general_only"
    source: str = "llm_router"
    fallback_reason: str = ""


@dataclass(slots=True)
class DialogState:
    route: str = "general"
    intent: str = "unknown"
    entities: dict[str, Any] = field(default_factory=dict)
    candidate_entities: dict[str, Any] = field(default_factory=dict)
    confirmation_target: str = ""
    missing_slots: list[str] = field(default_factory=list)
    clarify_type: str = ""
    tool_plan: list[str] = field(default_factory=list)
    response_policy: str = "general_only"
    confidence: float = 0.0
    clarify_count: int = 0
    last_tool: str = ""
    phase: str = ""
    open_question: str = ""
    flow_active: bool = False
    flow_kind: str = ""
    flow_stage: str = ""
    flow_interruptible: bool = False
    flow_resume_question: str = ""
    expected_slots: list[str] = field(default_factory=list)
    flow_non_answer_count: int = 0
    flow_non_answer_kind: str = ""


@dataclass(slots=True)
class PostToolVerification:
    enough_data: bool = True
    should_clarify: bool = False
    clarify_type: str = ""
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
    handoff: bool = False
    outcome: str = ""
    degraded: bool = False
    attachments: list[dict[str, Any]] = field(default_factory=list)
