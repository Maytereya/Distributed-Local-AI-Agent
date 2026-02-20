"""Доменные типы мессенджерного роутера.

Содержит канонический список labels и dataclass-контракты между модулями:
RouteDecision, Plan, Evidence, SessionState, ResponseEnvelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PATIENT_LABEL_PRIORITY = [
    "URGENT",
    "COMPLAINT",
    "TEST_RESULT",
    "TEST_ASSIST",
    "DOCTOR_SCHEDULE",
    "DOCTOR_INFO",
    "APPOINTMENT",
    "PRICE",
    "ADDRESS",
    "PREPARE",
    "NEWS",
    "MEDICAL_ADVICE",
    "OTHER",
]

Label = Literal[
    "APPOINTMENT",
    "TEST_ASSIST",
    "TEST_RESULT",
    "DOCTOR_INFO",
    "DOCTOR_SCHEDULE",
    "PRICE",
    "ADDRESS",
    "PREPARE",
    "NEWS",
    "COMPLAINT",
    "URGENT",
    "MEDICAL_ADVICE",
    "OTHER",
]

ContextAction = Literal["continue", "overwrite_doctor", "new_topic", "cancel_flow"]

AuthLevel = Literal["none", "patient_token"]


@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)  # [{"role":"user|assistant","text":...}]
    last_entities: dict[str, Any] = field(default_factory=dict)
    summary: str = field(default="")
    is_authenticated: bool = field(default=False)
    # Важно: токены лучше не хранить тут в явном виде
    auth_ref: str | None = None  # id сессии авторизации или что-то подобное


@dataclass
class RouteDecision:
    label: Label
    confidence: float = 0.0
    entities: dict[str, Any] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)  # {"urgent","complaint","auth_required",...}
    needs_handoff: bool = field(default=False) # потребность в переключении на оператора
    context_action: ContextAction = "continue"


@dataclass
class PlanStep:
    tool: str
    input: dict[str, Any] = field(default_factory=dict)
    required: bool = True
    auth: AuthLevel = "none"


@dataclass
class Plan:
    label: Label
    steps: list[PlanStep] = field(default_factory=list)


@dataclass
class Evidence:
    items: dict[str, Any] = field(default_factory=dict)
    debug_trace: list[dict[str, Any]] = field(default_factory=list)

    def put(self, key: str, value: Any) -> None:
        self.items[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.items.get(key, default)


@dataclass
class ResponseEnvelope:
    text: str
    attachments: list[dict[str, Any]] = field(default_factory=list)  # [{"type":"pdf","name":"...","url":"..."}]
    handoff: bool = False
    state_update: dict[str, Any] = field(default_factory=dict)
