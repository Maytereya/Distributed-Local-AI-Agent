"""Контракты prompt/JSON-схемы для роутера пациента.

Ответственность модуля:
1) Зафиксировать допустимую структуру JSON-ответа классификатора.
2) Ограничить допустимые действия/ключи entities.
3) Санитизировать LLM-JSON без исключений, чтобы не ломать pipeline.

Эти контракты отделяют "что можно вернуть из LLM" от логики роутера.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mess_types import PATIENT_LABEL_PRIORITY

ALLOWED_CONTEXT_ACTIONS = {"continue", "overwrite_doctor", "new_topic", "cancel_flow"}

ALLOWED_ENTITY_KEYS = {
    "doctor_name",
    "doctor_id",
    "specialty",
    "branch_name",
    "branch_id",
    "city",
    "service_name",
    "appointment_action",
    "patient_name",
    "test_name",
    "test_goal",
    "surname",
    "year",
    "filial",
    "number",
    "order_id",
    "result_action",
    "lang",
    "insurance_type",
    "accepts_children",
    "child_age",
    "date_hint",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "secondary_intents",
}


@dataclass(frozen=True)
class PromptSchemaVersion:
    name: str
    version: str
    required_top_keys: tuple[str, ...]


CLASSIFIER_SCHEMA_V2 = PromptSchemaVersion(
    name="classifier_patient",
    version="v2",
    required_top_keys=(
        "label",
        "confidence",
        "context_action",
        "entities",
        "flags",
        "clarify_needed",
        "clarify_reason",
        "clarify_slots",
        "intent_candidates",
    ),
)


def sanitize_classifier_json(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Легкий sanitizer для JSON, возвращаемого LLM-классификатором.
    Не бросает исключений, чтобы не ломать маршрут.
    """
    out: dict[str, Any] = {}

    label = payload.get("label")
    if isinstance(label, str) and label in PATIENT_LABEL_PRIORITY:
        out["label"] = label
    else:
        out["label"] = "OTHER"

    try:
        conf = float(payload.get("confidence", 0.2))
    except Exception:
        conf = 0.2
    out["confidence"] = max(0.0, min(1.0, conf))

    action = payload.get("context_action")
    if isinstance(action, str) and action in ALLOWED_CONTEXT_ACTIONS:
        out["context_action"] = action
    else:
        out["context_action"] = "continue"

    entities = payload.get("entities")
    if not isinstance(entities, dict):
        entities = {}
    out["entities"] = {k: v for k, v in entities.items() if k in ALLOWED_ENTITY_KEYS}

    flags = payload.get("flags")
    if not isinstance(flags, list):
        flags = []
    out["flags"] = [str(x) for x in flags[:16] if str(x).strip()]

    out["clarify_needed"] = bool(payload.get("clarify_needed"))

    reason = payload.get("clarify_reason")
    out["clarify_reason"] = str(reason).strip()[:120] if isinstance(reason, str) else ""

    raw_slots = payload.get("clarify_slots")
    if not isinstance(raw_slots, list):
        raw_slots = []
    out["clarify_slots"] = [str(x).strip() for x in raw_slots[:8] if str(x).strip()]

    raw_candidates = payload.get("intent_candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    out["intent_candidates"] = [
        str(x).strip()
        for x in raw_candidates[:4]
        if isinstance(x, str) and str(x).strip() in PATIENT_LABEL_PRIORITY
    ]

    return out


def sanitize_self_check_json(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["aligned_with_user_intent"] = bool(payload.get("aligned_with_user_intent"))
    out["fact_consistency"] = bool(payload.get("fact_consistency"))
    out["state_consistency"] = bool(payload.get("state_consistency"))
    out["unsafe_or_policy_violation"] = bool(payload.get("unsafe_or_policy_violation"))
    try:
        score = float(payload.get("score", 0.0))
    except Exception:
        score = 0.0
    out["score"] = max(0.0, min(score, 1.0))
    reason = payload.get("reason_short")
    out["reason_short"] = str(reason).strip()[:240] if isinstance(reason, str) else ""
    return out
