"""Candidate confirmation policy for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .signal_parsers import (
    extract_confirmation_head,
    extract_doctor_reference_candidate,
    extract_service_reference_candidate,
    split_mixed_utterance,
)


_TOPIC_QUESTION_RE = re.compile(
    r"\b(кто|что|где|когда|сколько|как|почему|зачем|расскажи|покажи|дай|найди|поищи|объясни|подскажи)\b",
    re.I,
)


@dataclass(slots=True)
class CandidateConfirmationParse:
    decision: str = "unknown"
    correction_value: str = ""
    switch_message: str = ""
    continue_message: str = ""


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


def parse_candidate_confirmation_message(text: str, *, target: str) -> CandidateConfirmationParse:
    decision, remainder = extract_confirmation_head(text)
    if decision not in {"yes", "no"}:
        return CandidateConfirmationParse()
    tail = str(remainder or "").strip()
    if decision == "yes":
        switch_message = _normalize_switch_candidate(tail) if _looks_like_switch_tail(tail) else ""
        return CandidateConfirmationParse(
            decision="yes",
            correction_value="",
            switch_message=switch_message,
            continue_message="Да",
        )

    correction_value = ""
    switch_message = ""
    correction_source = tail
    flow_part, split_switch = split_mixed_utterance(tail)
    if flow_part and split_switch:
        correction_source = flow_part
        switch_message = _normalize_switch_candidate(split_switch)
    elif _looks_like_switch_tail(tail):
        correction_source = ""
        switch_message = _normalize_switch_candidate(tail)

    if str(target or "").strip() == "doctor_name":
        correction_value = extract_doctor_reference_candidate(correction_source)
    elif str(target or "").strip() == "service_name":
        correction_value = extract_service_reference_candidate(correction_source)
        if not correction_value and _looks_like_short_service_phrase(correction_source):
            correction_value = str(correction_source or "").strip(" ,.?!")

    return CandidateConfirmationParse(
        decision="no",
        correction_value=str(correction_value or "").strip(),
        switch_message=str(switch_message or "").strip(),
        continue_message=str(correction_value or "").strip(),
    )


def _looks_like_switch_tail(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    if split_mixed_utterance(probe) != ("", ""):
        return True
    return bool(_TOPIC_QUESTION_RE.search(probe) and len(probe.split()) >= 2)


def _normalize_switch_candidate(text: str) -> str:
    probe = str(text or "").strip(" ,")
    if not probe:
        return ""
    probe = re.sub(r"^\s*(?:а\s+|и\s+|но\s+|ладно[, ]*|тогда\s+|кстати[, ]*)", "", probe, count=1, flags=re.I)
    return probe.strip(" ,")


def _looks_like_short_service_phrase(text: str) -> bool:
    probe = str(text or "").strip(" ,.?!")
    if not probe:
        return False
    if len(probe.split()) > 5:
        return False
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]", probe))
