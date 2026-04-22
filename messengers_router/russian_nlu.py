"""Russian NLU utilities — single source of truth.

Every module in messengers_router MUST import from here instead of
inlining `.lower().replace("ё", "е")` and related patterns.
This eliminates 20+ scattered copies of the same normalization idiom.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4096)
def normalize_ru(text: str | None) -> str:
    """Lowercase + ё→е normalization. Returns ``""`` for None.

    Use everywhere in place of the inline idiom:
        str(x).lower().replace("ё", "е").strip()
    """
    return str(text or "").lower().replace("ё", "е").strip()


# Canonical entity whitelist — single source of truth for all three former copies:
#   prompt_contracts.ALLOWED_ENTITY_KEYS
#   classifier._ALLOWED_ENTITY_KEYS
#   entity_grounder._LABEL_ENTITY_WHITELIST keys (union)
ENTITY_WHITELIST: frozenset[str] = frozenset({
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
    "include_promos",
    "time_flexible",
})
