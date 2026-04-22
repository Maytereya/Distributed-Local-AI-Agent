"""Catalog grounding policy for FreeTalk."""

from __future__ import annotations

import logging
from typing import Any

from .observability import log_event


async def ground_entities(
    services: Any,
    user_message: str,
    *,
    intent_hint: str = "",
    current_service_name: str = "",
) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    intent = str(intent_hint or "").strip().lower()
    doctor_focus = intent in {"doctor_info", "doctor_schedule"}
    service_focus = intent in {"price", "prepare", "tests", "service_info"}
    should_try_service = service_focus or not doctor_focus

    if should_try_service:
        try:
            service_match = await services.match_catalog_service(
                user_message,
                current_service_name=str(current_service_name or "").strip(),
            )
        except Exception as exc:
            log_event(
                "entity_ground_service_failed",
                level=logging.WARNING,
                error_type=type(exc).__name__,
            )
            service_match = {}
        if isinstance(service_match, dict):
            status = str(service_match.get("status") or "")
            canonical = str(service_match.get("canonical") or "").strip()
            entities["_ft_service_match_status"] = status
            entities["_ft_service_match_query"] = str(service_match.get("query") or "").strip()
            if status == "exact" and canonical:
                entities["service_name"] = canonical
            elif status == "fuzzy" and canonical:
                entities["service_name_candidate"] = canonical
    else:
        entities["_ft_service_match_status"] = "skipped_doctor_focus"
        entities["_ft_service_match_query"] = ""

    try:
        doctor_match = await services.match_catalog_doctor(user_message)
    except Exception as exc:
        log_event(
            "entity_ground_doctor_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
        )
        doctor_match = {}
    if isinstance(doctor_match, dict):
        status = str(doctor_match.get("status") or "")
        canonical = str(doctor_match.get("canonical") or "").strip()
        entities["_ft_doctor_match_status"] = status
        entities["_ft_doctor_match_query"] = str(doctor_match.get("query") or "").strip()
        if status == "exact" and canonical:
            entities["doctor_name"] = canonical
            entities["doctor_name_match_status"] = status
        elif status == "fuzzy" and canonical:
            entities["doctor_name_match_status"] = status
            entities["doctor_name_candidate"] = canonical

    return entities
