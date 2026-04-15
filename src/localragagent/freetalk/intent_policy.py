"""Deterministic intent/entity policy for FreeTalk."""

from __future__ import annotations

import re
from typing import Any


_DOCTOR_IN_SERVICE_CONTEXT_RE = re.compile(r"\b(врач|доктор|у\s+[а-яё\\-]{3,})\b", re.I)


def apply_intent_entity_policy(
    *,
    user_message: str,
    intent: str,
    entities: dict[str, Any],
) -> dict[str, Any]:
    filtered = dict(entities or {})
    normalized_intent = str(intent or "").strip().lower()
    message = str(user_message or "")

    # Doctor-oriented requests should not keep accidental service grounding.
    if normalized_intent in {"appointment", "doctor_info", "doctor_schedule"}:
        has_doctor_context = bool(str(filtered.get("doctor_name") or "").strip()) or bool(
            _DOCTOR_IN_SERVICE_CONTEXT_RE.search(message)
        )
        if has_doctor_context:
            filtered.pop("service_name", None)
            filtered.pop("test_name", None)
            filtered.pop("service_variant", None)

    return filtered
