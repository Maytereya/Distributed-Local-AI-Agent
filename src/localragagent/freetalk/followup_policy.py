"""Deterministic follow-up and contextual extraction helpers."""

from __future__ import annotations

from typing import Any

from .signal_parsers import (
    extract_contextual_entities,
    extract_branch_reference,
    extract_city_reference,
    extract_date_filters,
    extract_service_variant,
    extract_time_filters,
    looks_like_doctor_followup_message,
    looks_like_service_followup_message,
)


def looks_like_contextual_clinical_followup(
    *,
    user_message: str,
    remembered_doctor: str,
    memory_entities: dict[str, Any],
) -> bool:
    if not memory_entities and not str(remembered_doctor or "").strip():
        return False
    if extract_contextual_entities(user_message):
        return True
    remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
    if remembered_service and looks_like_service_followup_message(
        user_message=user_message,
        remembered_service=remembered_service,
    ):
        return True
    return looks_like_doctor_followup_message(
        user_message=user_message,
        remembered_doctor=remembered_doctor,
    )


def contextual_followup_tool_plan(
    *,
    user_message: str,
    remembered_doctor: str,
    memory_entities: dict[str, Any],
) -> list[str]:
    contextual_entities = extract_contextual_entities(user_message)
    has_schedule_filters = bool(
        {"branch_name", "city", "date", "date_from", "date_to", "time", "time_from", "time_to"}
        & set(contextual_entities.keys())
    )
    remembered_doctor = str(remembered_doctor or memory_entities.get("doctor_name") or "").strip()
    remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
    if remembered_doctor and has_schedule_filters:
        return ["doctors_schedule_week", "doctors_info"]
    if remembered_service:
        if "branch_name" in contextual_entities:
            return ["address_info", "service_bundle_info"]
        if "service_variant" in contextual_entities:
            return ["service_bundle_info", "price_info", "address_info"]
    return []
