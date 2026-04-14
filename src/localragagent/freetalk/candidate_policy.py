"""Candidate confirmation policy for FreeTalk."""

from __future__ import annotations

from typing import Any


def candidate_entities_from_entities(
    entities: dict[str, Any],
    *,
    current: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(current or {})
    service_candidate = str(entities.get("service_name_candidate") or "").strip()
    doctor_candidate = str(entities.get("doctor_name_candidate") or "").strip()
    if service_candidate:
        out["service_name"] = service_candidate
    if doctor_candidate:
        out["doctor_name"] = doctor_candidate
    return out


def candidate_confirmation_target(
    *,
    intent: str,
    tool_plan: list[str],
    entities: dict[str, Any],
    candidate_entities: dict[str, Any],
    current_target: str = "",
) -> str:
    target = str(current_target or "").strip()
    if target and str(candidate_entities.get(target) or "").strip():
        if target == "doctor_name" and not str(entities.get("doctor_name") or "").strip():
            return target
        if target == "service_name" and not str(entities.get("service_name") or "").strip():
            return target

    plan = set(tool_plan or [])
    route_intent = str(intent or "").strip().lower()
    service_candidate = str(candidate_entities.get("service_name") or "").strip()
    doctor_candidate = str(candidate_entities.get("doctor_name") or "").strip()
    service_exact = bool(str(entities.get("service_name") or "").strip())
    doctor_exact = bool(str(entities.get("doctor_name") or "").strip())

    if (
        service_candidate
        and not service_exact
        and (
            route_intent in {"price", "prepare", "tests", "service_info", "address"}
            or bool({"price_info", "service_bundle_info", "test_prepare", "test_assist", "address_info"} & plan)
        )
    ):
        return "service_name"
    if (
        doctor_candidate
        and not doctor_exact
        and (
            route_intent in {"doctor_info", "doctor_schedule"}
            or bool({"doctors_info", "doctors_schedule_week"} & plan)
        )
    ):
        return "doctor_name"
    return ""


def candidate_confirmation_question(*, target: str, candidate_value: str) -> str:
    label = str(candidate_value or "").strip()
    if not label:
        return ""
    if target == "doctor_name":
        return f"Правильно понял, что нужен врач «{label}»? Ответьте: Да или Нет."
    if target == "service_name":
        return f"Правильно понял, что нужна услуга «{label}»? Ответьте: Да или Нет."
    return ""


def candidate_rejected_question(target: str) -> str:
    if target == "doctor_name":
        return "Хорошо. Уточните, пожалуйста, фамилию врача или специальность."
    if target == "service_name":
        return "Хорошо. Уточните, пожалуйста, точное название услуги или анализа."
    return "Хорошо. Уточните запрос чуть подробнее."


def promote_confirmed_candidate(
    *,
    entities: dict[str, Any],
    candidate_entities: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    out = dict(entities or {})
    value = str(candidate_entities.get(target) or "").strip()
    if not value:
        return out
    out[target] = value
    if target == "doctor_name":
        out["doctor_name_match_status"] = "fuzzy_confirmed"
    if target == "service_name":
        out["_ft_service_match_status"] = "fuzzy_confirmed"
    return out
