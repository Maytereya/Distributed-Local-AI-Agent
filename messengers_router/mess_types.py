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


class AppointmentPhase:
    """String constants for appointment flow phases.

    Used as the ``DialogState.phase`` value to track where a patient is
    in the appointment booking / cancellation / rescheduling state machine.
    These replace 20+ scattered boolean flags in ``last_entities``.

    Lifecycle
    ---------
    IDLE               → not in any appointment sub-flow
    COLLECTING         → gathering required slots (doctor, date/time, etc.)
    CONFIRM            → all slots collected; awaiting patient confirmation
    CONFIRMED          → patient confirmed; handoff to booking system
    CANCEL_CONFIRM     → awaiting confirmation before cancelling appointment
    """

    IDLE: str = ""
    COLLECTING: str = "appointment_collecting"
    CONFIRM: str = "appointment_confirm"
    CONFIRMED: str = "appointment_confirmed"
    CANCEL_CONFIRM: str = "appointment_cancel_confirm"

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return the set of all valid phase strings (excluding IDLE)."""
        return frozenset({cls.COLLECTING, cls.CONFIRM, cls.CONFIRMED, cls.CANCEL_CONFIRM})

    @classmethod
    def is_active(cls, phase: str) -> bool:
        """Return True when the phase represents an in-progress appointment flow."""
        return phase in cls.values()


@dataclass
class DialogState:
    """Typed conversation state for a single active flow.

    Replaces the unbounded ``last_entities`` god-object for all structured
    multi-turn flows (appointment, test_result, etc.).  ``last_entities``
    continues to carry raw session memory; ``DialogState`` carries the
    semantic intent + slot state for the *current active intent*.

    Methods
    -------
    is_active()         — True when a non-trivial label is currently held.
    clear()             — Reset all fields to defaults (on topic switch / handoff).
    merge_entities(new) — Merge new entities without overwriting existing values.
    """

    label: str = "OTHER"
    phase: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    candidate_entities: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    clarify_count: int = 0
    open_question: str = ""
    confidence: float = 0.0

    _INACTIVE_LABELS: frozenset[str] = field(
        default_factory=lambda: frozenset({"OTHER", ""}),
        init=False,
        repr=False,
        compare=False,
    )

    def is_active(self) -> bool:
        """Return True when an intent is being tracked (not OTHER / empty)."""
        return self.label not in self._INACTIVE_LABELS

    def clear(self) -> None:
        """Reset all state to defaults — call on explicit topic switch or handoff."""
        self.label = "OTHER"
        self.phase = ""
        self.entities = {}
        self.candidate_entities = {}
        self.missing_slots = []
        self.clarify_count = 0
        self.open_question = ""
        self.confidence = 0.0

    def merge_entities(self, new: dict[str, Any]) -> None:
        """Merge ``new`` entities into ``self.entities``.

        Existing values are preserved; only genuinely missing keys are added.
        Call this after each NLU pass to accumulate extracted slots.
        """
        for k, v in new.items():
            if k not in self.entities and v not in (None, "", []):
                self.entities[k] = v


@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)  # [{"role":"user|assistant","text":...}]
    last_entities: dict[str, Any] = field(default_factory=dict)
    summary: str = field(default="")
    is_authenticated: bool = field(default=False)
    # Важно: токены лучше не хранить тут в явном виде
    auth_ref: str | None = None  # id сессии авторизации или что-то подобное
    dialog: DialogState = field(default_factory=DialogState)  # typed intent state

    def __post_init__(self) -> None:
        # Migration bridge: derive dialog.phase from legacy last_entities flags when
        # dialog.phase is not already set.  Allows code that only writes last_entities
        # (tests, restored sessions) to work correctly during Phase 2.
        # Remove once all writers set dialog.phase directly.
        if not self.dialog.phase:
            if self.last_entities.get("appointment_confirmed"):
                self.dialog.phase = AppointmentPhase.CONFIRMED
            elif self.last_entities.get("appointment_confirm_pending"):
                self.dialog.phase = AppointmentPhase.CONFIRM
            elif self.last_entities.get("appointment_flow_active"):
                self.dialog.phase = AppointmentPhase.COLLECTING


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
