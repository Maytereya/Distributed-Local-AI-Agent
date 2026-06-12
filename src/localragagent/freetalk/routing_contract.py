"""Clinical intent routing contract for FreeTalk clarification-first flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CLINICAL_INTENTS: tuple[str, ...] = (
    "appointment",
    "doctor_schedule",
    "doctor_info",
    "price",
    "prepare",
    "tests",
    "test_result",
    "address",
    "clinic_documents",
    "clinic_news",
    "service_info",
    "unknown",
)

CLARIFY_TYPES: tuple[str, ...] = (
    "identify",
    "confirm_candidate",
    "narrow_choice",
    "missing_auth_data",
    "other",
)


_KNOWN_TOOLS: set[str] = {
    "service_bundle_info",
    "price_info",
    "test_prepare",
    "test_assist",
    "doctors_info",
    "doctors_schedule_week",
    "address_info",
    "test_result_status",
    "main_index_info",
    "news_info",
}


_INTENT_ALIASES: dict[str, str] = {
    "appointment": "appointment",
    "book": "appointment",
    "doctor_schedule": "doctor_schedule",
    "schedule": "doctor_schedule",
    "doctor_info": "doctor_info",
    "doctor": "doctor_info",
    "price": "price",
    "cost": "price",
    "prepare": "prepare",
    "preparation": "prepare",
    "tests": "tests",
    "test_assist": "tests",
    "test_result": "test_result",
    "result": "test_result",
    "address": "address",
    "branches": "address",
    "clinic_documents": "clinic_documents",
    "documents": "clinic_documents",
    "clinic_news": "clinic_news",
    "news": "clinic_news",
    "service_info": "service_info",
    "service": "service_info",
    "unknown": "unknown",
    "other": "unknown",
}


_INTENT_TO_PLAN: dict[str, list[str]] = {
    "appointment": ["doctors_schedule_week", "doctors_info", "address_info"],
    "doctor_schedule": ["doctors_schedule_week", "doctors_info"],
    "doctor_info": ["doctors_info", "doctors_schedule_week"],
    "price": ["price_info", "service_bundle_info", "test_assist"],
    "prepare": ["test_prepare", "test_assist", "service_bundle_info"],
    "tests": ["test_assist", "test_prepare", "price_info"],
    "test_result": ["test_result_status", "test_assist"],
    "address": ["address_info", "service_bundle_info"],
    "service_info": ["service_bundle_info", "price_info", "address_info"],
    "clinic_documents": ["main_index_info", "news_info"],
    "clinic_news": ["news_info", "main_index_info"],
    "unknown": [],
}


_ENTITY_KEYS: set[str] = {
    "appointment_action",
    "doctor_name",
    "specialty",
    "service_name",
    "service_variant",
    "test_name",
    "city",
    "branch_name",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "date",
    "time",
    "doctor_id",
    "patient_name",
    "child_age",
    "result_surname",
    "result_year_of_birth",
    "result_analysis_code",
    "result_analysis_number",
}

_KNOWN_MISSING_SLOTS: set[str] = {
    "appointment_action",
    "doctor_name",
    "specialty",
    "service_or_analysis_name",
    "branch_or_city",
    "branch_name",
    "city",
    "date",
    "date_from",
    "date_to",
    "time",
    "time_from",
    "time_to",
    "patient_name",
    "result_surname",
    "result_year_of_birth",
    "result_analysis_code",
    "result_analysis_number",
}

_MISSING_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "doctor": ("doctor_name",),
    "doctor_id": ("doctor_name",),
    "doctor_or_specialty": ("doctor_name", "specialty"),
    "doctor_name_or_specialty": ("doctor_name", "specialty"),
    "doctor_name/specialty": ("doctor_name", "specialty"),
    "service": ("service_or_analysis_name",),
    "service_name": ("service_or_analysis_name",),
    "test_name": ("service_or_analysis_name",),
    "analysis_name": ("service_or_analysis_name",),
    "service_or_test_name": ("service_or_analysis_name",),
    "branch": ("branch_or_city",),
    "region": ("branch_or_city",),
    "filial": ("branch_or_city",),
    "surname": ("result_surname",),
    "result_last_name": ("result_surname",),
    "birth_year": ("result_year_of_birth",),
    "year": ("result_year_of_birth",),
    "result_year": ("result_year_of_birth",),
    "analysis_code": ("result_analysis_code",),
    "result_filial": ("result_analysis_code",),
    "result_analysis_filial": ("result_analysis_code",),
    "analysis_number": ("result_analysis_number",),
    "number": ("result_analysis_number",),
    "result_number": ("result_analysis_number",),
    "order_number": ("result_analysis_number",),
}


@dataclass(slots=True)
class ClinicalDecision:
    intent: str
    confidence: float = 0.0
    entities: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    clarify_type: str = ""
    clarify_question: str = ""
    tool_plan: list[str] = field(default_factory=list)
    source: str = "heuristic"


def normalize_intent(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    return _INTENT_ALIASES.get(raw, "unknown")


def _clip_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _coerce_entities(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key or "").strip()
        if name not in _ENTITY_KEYS:
            continue
        if isinstance(raw, (str, int, float, bool)):
            text = str(raw).strip()
            if text:
                out[name] = text
    return out


def normalize_missing_slots(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        slots = _MISSING_SLOT_ALIASES.get(raw, (raw,))
        for slot in slots:
            if slot not in _KNOWN_MISSING_SLOTS or slot in seen:
                continue
            seen.add(slot)
            out.append(slot)
    return out


def _coerce_missing_slots(value: Any) -> list[str]:
    return normalize_missing_slots(value)


def _coerce_tool_plan(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        tool = str(item or "").strip()
        if tool in _KNOWN_TOOLS and tool not in out:
            out.append(tool)
    return out


def normalize_clarify_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in CLARIFY_TYPES:
        return raw
    return ""


def tool_plan_for_intent(intent: str, *, include_meili_tools: bool) -> list[str]:
    plan = list(_INTENT_TO_PLAN.get(str(intent or "").strip().lower(), []))
    if not include_meili_tools:
        plan = [item for item in plan if item not in {"main_index_info", "news_info"}]
    return plan


def merge_missing_slots_from_plan(tool_plan: list[str], entities: dict[str, Any], *, intent: str = "") -> list[str]:
    missing: list[str] = []
    plan = list(tool_plan or [])
    normalized_intent = str(intent or "").strip().lower()
    doctor_known = bool(str(entities.get("doctor_name") or entities.get("doctor_id") or "").strip())
    specialty_known = bool(str(entities.get("specialty") or "").strip())
    service_known = bool(
        str(
            entities.get("service_name")
            or entities.get("test_name")
            or entities.get("service_variant")
            or ""
        ).strip()
    )
    result_surname = bool(str(entities.get("result_surname") or "").strip())
    result_year = bool(str(entities.get("result_year_of_birth") or "").strip())
    result_code = bool(str(entities.get("result_analysis_code") or "").strip())
    result_number = bool(str(entities.get("result_analysis_number") or "").strip())
    patient_name = bool(str(entities.get("patient_name") or "").strip())
    appointment_action = bool(str(entities.get("appointment_action") or "").strip())
    branch_known = bool(str(entities.get("branch_name") or entities.get("city") or "").strip())
    date_known = bool(str(entities.get("date") or entities.get("date_from") or "").strip())
    time_known = bool(str(entities.get("time") or entities.get("time_from") or "").strip())

    if normalized_intent == "appointment":
        has_windows = bool(entities.get("appointment_windows"))
        has_selected_datetime = date_known or time_known
        if not appointment_action:
            missing.append("appointment_action")
        if not doctor_known and not specialty_known:
            missing.append("doctor_name")
            missing.append("specialty")
        if not branch_known and str(entities.get("appointment_branch_options") or "").strip():
            missing.append("branch_or_city")
        if has_windows or has_selected_datetime:
            if not date_known:
                missing.append("date")
            if not time_known:
                missing.append("time")
        if (has_windows or (date_known and time_known)) and not patient_name:
            missing.append("patient_name")

    if any(tool in {"doctors_info", "doctors_schedule_week"} for tool in plan):
        if not doctor_known and not specialty_known:
            missing.append("doctor_name")
            missing.append("specialty")

    if any(tool in {"price_info", "service_bundle_info", "test_prepare", "test_assist"} for tool in plan):
        if not service_known and not doctor_known:
            missing.append("service_or_analysis_name")

    if "test_result_status" in plan:
        if not result_surname:
            missing.append("result_surname")
        if not result_year:
            missing.append("result_year_of_birth")
        if not result_code:
            missing.append("result_analysis_code")
        if not result_number:
            missing.append("result_analysis_number")

    dedup: list[str] = []
    seen: set[str] = set()
    for slot in missing:
        if slot in seen:
            continue
        seen.add(slot)
        dedup.append(slot)
    return dedup


def clarify_question_for_slots(intent: str, missing_slots: list[str]) -> str:
    slots = set(str(slot or "").strip().lower() for slot in (missing_slots or []))
    if "appointment_action" in slots:
        return "Уточните, пожалуйста, что нужно сделать: записаться, перенести или отменить запись."
    if str(intent or "").strip().lower() == "appointment" and slots & {"doctor_name", "specialty"}:
        return "Уточните, пожалуйста, к какому врачу или по какой специальности нужна запись."
    if slots & {"doctor_name", "specialty"}:
        if intent == "doctor_schedule":
            return "Уточните, пожалуйста, фамилию врача или специальность, чтобы показать расписание."
        return "Уточните, пожалуйста, фамилию врача или специальность."
    if "service_or_analysis_name" in slots:
        return "Уточните, пожалуйста, точное название услуги или анализа."
    if "branch_or_city" in slots:
        return "Уточните, пожалуйста, удобный филиал для записи."
    if "date" in slots and "time" in slots and str(intent or "").strip().lower() == "appointment":
        return "Выберите, пожалуйста, дату и время для записи."
    if "date" in slots and str(intent or "").strip().lower() == "appointment":
        return "Уточните, пожалуйста, удобную дату для записи."
    if "time" in slots and str(intent or "").strip().lower() == "appointment":
        return "Уточните, пожалуйста, удобное время для записи."
    if "patient_name" in slots:
        return "Сообщите, пожалуйста, ваше ФИО для записи."
    result_labels: list[str] = []
    if "result_surname" in slots:
        result_labels.append("фамилию пациента")
    if "result_year_of_birth" in slots:
        result_labels.append("год рождения")
    if "result_analysis_code" in slots:
        result_labels.append("код анализа")
    if "result_analysis_number" in slots:
        result_labels.append("номер анализа")
    if result_labels:
        return "Для проверки результата уточните: " + ", ".join(result_labels) + "."
    return "Уточните, пожалуйста, ваш запрос по клинике, чтобы я корректно выполнил поиск."


def clarify_type_for_slots(intent: str, missing_slots: list[str]) -> str:
    _ = intent
    slots = set(str(slot or "").strip().lower() for slot in (missing_slots or []))
    if not slots:
        return ""
    if slots & {"result_surname", "result_year_of_birth", "result_analysis_code", "result_analysis_number", "patient_name"}:
        return "missing_auth_data"
    if slots & {"appointment_action", "doctor_name", "specialty", "service_or_analysis_name"}:
        return "identify"
    if slots & {"branch_name", "branch_or_city", "city", "date", "date_from", "date_to", "time", "time_from", "time_to"}:
        return "narrow_choice"
    return "other"


def parse_clinical_decision(
    payload: dict[str, Any] | None,
    *,
    include_meili_tools: bool,
) -> ClinicalDecision:
    data = payload if isinstance(payload, dict) else {}
    intent = normalize_intent(data.get("intent"))
    confidence = _clip_float(data.get("confidence"), default=0.0)
    entities = _coerce_entities(data.get("entities"))
    missing_slots = _coerce_missing_slots(data.get("missing_slots"))
    clarify_type = normalize_clarify_type(data.get("clarify_type"))
    clarify_question = str(data.get("clarify_question") or "").strip()
    explicit_plan = _coerce_tool_plan(data.get("tool_plan"))
    mapped_plan = tool_plan_for_intent(intent, include_meili_tools=include_meili_tools)
    if not clarify_type and (missing_slots or clarify_question):
        clarify_type = clarify_type_for_slots(intent, missing_slots)

    if explicit_plan:
        tool_plan = [tool for tool in explicit_plan if tool in mapped_plan or tool in _KNOWN_TOOLS]
        if not include_meili_tools:
            tool_plan = [tool for tool in tool_plan if tool not in {"main_index_info", "news_info"}]
    else:
        tool_plan = mapped_plan

    if intent == "price":
        blocked_slots = {"branch_name", "branch_or_city", "city"}
        sanitized_slots = [slot for slot in missing_slots if slot not in blocked_slots]
        if len(sanitized_slots) != len(missing_slots):
            missing_slots = sanitized_slots
            clarify_type = ""
            clarify_question = ""

    return ClinicalDecision(
        intent=intent,
        confidence=confidence,
        entities=entities,
        missing_slots=missing_slots,
        clarify_type=clarify_type,
        clarify_question=clarify_question,
        tool_plan=tool_plan,
        source="llm_router",
    )
