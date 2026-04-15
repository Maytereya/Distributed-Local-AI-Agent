"""Доменные типы мессенджерного роутера.

Содержит канонический список labels и dataclass-контракты между модулями:
RouteDecision, Plan, Evidence, SessionState, ResponseEnvelope.
Ответственность модуля: единый межмодульный контракт типов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ConfidencePolicy:
    """Named confidence thresholds — single source of truth.

    Replace every scattered magic float in nlu_pipeline / classifier
    with a reference to the appropriate field here so thresholds are
    tuned in one place.

    Threshold semantics
    -------------------
    llm_promote_rich    — _merge(): promote rule over LLM when LLM conf is
                          below this value and engine mode is "rich".
    llm_promote_hybrid  — same for "hybrid" mode (stricter threshold).
    refine_min          — minimum LLM-refine confidence to override base decision.
    rule_hardcode       — confidence assigned to unambiguous deterministic hits
                          (safety labels, strong keyword rules).
    high                — high-confidence rule-based label assignment.
    moderate            — moderate-confidence rule-based label assignment.
    price_floor         — minimum confidence floor applied to PRICE decisions
                          before emitting them.
    promoted_floor      — confidence floor used when rule is promoted over LLM.
    llm_default         — default confidence returned when LLM is unavailable
                          or returns malformed JSON.
    """

    llm_promote_rich: float = 0.45
    llm_promote_hybrid: float = 0.55
    refine_min: float = 0.45
    rule_hardcode: float = 0.85
    high: float = 0.75
    moderate: float = 0.70
    price_floor: float = 0.55
    promoted_floor: float = 0.60
    llm_default: float = 0.20


# Module-level singleton — import this everywhere instead of sprinkling floats.
CONFIDENCE = ConfidencePolicy()


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
    source: str = "guardrail"
    clarify_needed: bool = False
    clarify_reason: str = ""
    clarify_slots: list[str] = field(default_factory=list)
    intent_candidates: list[str] = field(default_factory=list)


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
