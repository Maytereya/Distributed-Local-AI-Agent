"""Session-memory policy helpers for FreeTalk."""

from __future__ import annotations

from typing import Any, Callable

from .contracts import DialogState
from .dialog_state import SESSION_MEMORY_ENTITY_KEYS
from .observability import log_event
from .signal_parsers import DOCTOR_ANAPHORA_RE


LAST_DOCTOR_NAME_KEY = "last_doctor_name"
FLOW_SCOPED_META_KEYS: tuple[str, ...] = (LAST_DOCTOR_NAME_KEY,)


def flow_scoped_memory_keys() -> tuple[str, ...]:
    return tuple(SESSION_MEMORY_ENTITY_KEYS)


def flow_scoped_meta_keys() -> tuple[str, ...]:
    return FLOW_SCOPED_META_KEYS


def extract_primary_doctor_name(tool_name: str, payload: dict[str, Any]) -> str:
    entities_used_ft = payload.get("entities_used_ft") if isinstance(payload, dict) else {}
    if isinstance(entities_used_ft, dict):
        value = str(entities_used_ft.get("doctor_name") or "").strip()
        if value:
            return value

    doctors = payload.get("doctors") if isinstance(payload, dict) else []
    if isinstance(doctors, list):
        names: list[str] = []
        for row in doctors:
            if not isinstance(row, dict):
                continue
            fio = str(row.get("fio") or "").strip()
            if fio:
                names.append(fio)
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) == 1:
            return unique_names[0]

    schedule = payload.get("schedule") if isinstance(payload, dict) else []
    if isinstance(schedule, list):
        names = []
        for row in schedule:
            if not isinstance(row, dict):
                continue
            fio = str(row.get("fio") or "").strip()
            if fio:
                names.append(fio)
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) == 1:
            return unique_names[0]

    if tool_name == "doctors_schedule_week":
        direct = str(payload.get("doctor_name") or payload.get("fio") or "").strip()
        if direct:
            return direct

    return ""


def extract_doctor_options(payload: dict[str, Any], *, limit: int = 6) -> list[dict[str, str]]:
    doctors = payload.get("doctors") if isinstance(payload, dict) else []
    if not isinstance(doctors, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in doctors:
        if not isinstance(row, dict):
            continue
        fio = str(row.get("fio") or "").strip()
        if not fio or fio in seen:
            continue
        seen.add(fio)
        option = {"fio": fio}
        spec = str(row.get("specialization") or row.get("specialty") or "").strip()
        if spec:
            option["specialization"] = spec
        regions = row.get("regions") or row.get("addresses")
        if isinstance(regions, list):
            clean_regions = [str(x).strip() for x in regions if str(x).strip()]
            if clean_regions:
                option["branches"] = "; ".join(clean_regions[:3])
        out.append(option)
        if len(out) >= max(1, int(limit)):
            break
    return out


def doctor_choice_clarification_text(memory_entities: dict[str, Any]) -> str:
    specialty = str(memory_entities.get("specialty") or "").strip()
    options = memory_entities.get("doctor_options")
    doctors = [x for x in options if isinstance(x, dict)] if isinstance(options, list) else []
    if specialty:
        head = f"Выберите, пожалуйста, конкретного врача по специальности {specialty}, и я покажу его расписание."
    else:
        head = "Выберите, пожалуйста, конкретного врача, и я покажу его расписание."
    if not doctors:
        return head
    lines = [head]
    for row in doctors[:6]:
        fio = str(row.get("fio") or "").strip()
        if not fio:
            continue
        spec = str(row.get("specialization") or "").strip()
        branches = str(row.get("branches") or "").strip()
        suffix_parts = [part for part in (spec, branches) if part]
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        lines.append(f"- {fio}{suffix}")
    return "\n".join(lines)


def should_clarify_doctor_choice_from_memory(
    *,
    user_message: str,
    tool_plan: list[str],
    entities: dict[str, Any],
    memory_entities: dict[str, Any],
    remembered_doctor: str,
) -> bool:
    if not DOCTOR_ANAPHORA_RE.search(str(user_message or "")):
        return False
    if not ({"doctors_schedule_week", "doctors_info"} & set(tool_plan or [])):
        return False
    if str(entities.get("doctor_name") or entities.get("doctor_id") or remembered_doctor or "").strip():
        return False
    if str(memory_entities.get("doctor_name") or "").strip():
        return False
    return bool(str(memory_entities.get("specialty") or "").strip() or memory_entities.get("doctor_options"))


def extract_schedule_memory_entities(payload: dict[str, Any]) -> dict[str, Any]:
    schedule = payload.get("schedule") if isinstance(payload, dict) else []
    if not isinstance(schedule, list):
        return {}
    out: dict[str, Any] = {}
    for row in schedule:
        if not isinstance(row, dict):
            continue
        row_schedule = row.get("schedule")
        if not isinstance(row_schedule, dict):
            continue
        for region_name, days in row_schedule.items():
            branch_name = str(region_name or "").strip()
            if branch_name:
                out["branch_name"] = branch_name
            if not isinstance(days, list):
                return out
            for day in days:
                if not isinstance(day, dict):
                    continue
                date_value = str(day.get("date") or "").strip()
                if date_value:
                    out["date"] = date_value
                    out["date_from"] = date_value
                    out["date_to"] = date_value
                slots = [str(x).strip() for x in (day.get("slots") or []) if str(x).strip()]
                if slots:
                    out["time"] = slots[0][:5]
                    out["time_from"] = slots[0][:5]
                else:
                    start = str(day.get("start") or "").strip()
                    end = str(day.get("end") or "").strip()
                    if start:
                        out["time_from"] = start[:5]
                    if end:
                        out["time_to"] = end[:5]
                return out
            return out
    return out


def tool_payload_memory_entities(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    doctor_name = extract_primary_doctor_name(tool_name, payload)
    if doctor_name:
        out["doctor_name"] = doctor_name
    if tool_name in {"doctors_info", "doctors_schedule_week"}:
        entities_used_ft = payload.get("entities_used_ft") if isinstance(payload, dict) else {}
        if isinstance(entities_used_ft, dict):
            specialty = str(entities_used_ft.get("specialty") or "").strip()
            if specialty:
                out["specialty"] = specialty
        options = extract_doctor_options(payload)
        if options:
            out["doctor_options"] = options
    if tool_name in {"price_info", "service_bundle_info", "test_prepare", "test_assist"}:
        entities_used_ft = payload.get("entities_used_ft") if isinstance(payload, dict) else {}
        if isinstance(entities_used_ft, dict):
            service_name = str(entities_used_ft.get("service_name") or "").strip()
            test_name = str(entities_used_ft.get("test_name") or "").strip()
            if service_name:
                out["service_name"] = service_name
            if test_name:
                out["test_name"] = test_name
    if tool_name == "doctors_schedule_week":
        out.update(extract_schedule_memory_entities(payload))
        appointment_windows = payload.get("appointment_windows") if isinstance(payload, dict) else None
        appointment_branch_options = payload.get("appointment_branch_options") if isinstance(payload, dict) else None
        if isinstance(appointment_windows, list) and appointment_windows:
            out["appointment_windows"] = appointment_windows
        if isinstance(appointment_branch_options, list) and appointment_branch_options:
            out["appointment_branch_options"] = appointment_branch_options
    if tool_name == "address_info":
        entities_used_ft = payload.get("entities_used_ft") if isinstance(payload, dict) else {}
        if isinstance(entities_used_ft, dict):
            branch_name = str(entities_used_ft.get("branch_name") or "").strip()
            city = str(entities_used_ft.get("city") or "").strip()
            service_name = str(entities_used_ft.get("service_name") or "").strip()
            test_name = str(entities_used_ft.get("test_name") or "").strip()
            if branch_name:
                out["branch_name"] = branch_name
            if city:
                out["city"] = city
            if service_name:
                out["service_name"] = service_name
            if test_name:
                out["test_name"] = test_name
        addresses = [str(x).strip() for x in (payload.get("addresses") or []) if str(x).strip()]
        if addresses:
            out["branch_name"] = addresses[0]
            out["appointment_branch_options"] = addresses
    return out


async def enrich_entities_from_session_memory(
    *,
    memory: Any,
    session_id: str,
    user_message: str,
    tool_plan: list[str],
    entities: dict[str, Any],
    dialog_state: DialogState | None,
    memory_entities: dict[str, Any] | None,
    contextual_entities: dict[str, Any] | None,
    load_session_entity_memory: Callable[[Any, str], Any],
    dialog_state_is_active: Callable[[DialogState | None], bool],
    looks_like_doctor_followup_message: Callable[..., bool],
    looks_like_service_followup_message: Callable[..., bool],
    compose_service_with_variant: Callable[[str, str], str],
) -> dict[str, Any]:
    out = dict(entities or {})
    memory_entities = dict(memory_entities or {})
    if not memory_entities:
        memory_entities = await load_session_entity_memory(memory, session_id)
    contextual_entities = dict(contextual_entities or {})
    active_state = dialog_state_is_active(dialog_state)
    enriched_keys: list[str] = []

    plan = set(tool_plan or [])
    uses_doctor_tools = bool({"doctors_schedule_week", "doctors_info"} & plan)
    uses_service_tools = bool({"price_info", "service_bundle_info", "test_prepare", "test_assist"} & plan)
    uses_address_tools = "address_info" in plan
    uses_result_tool = "test_result_status" in plan
    has_contextual_entities = bool(contextual_entities)

    if uses_doctor_tools and not str(out.get("doctor_name") or "").strip():
        remembered = await memory.get_meta_str(session_id, LAST_DOCTOR_NAME_KEY, "")
        remembered = str(remembered or memory_entities.get("doctor_name") or "").strip()
        if remembered and (
            active_state
            or has_contextual_entities
            or DOCTOR_ANAPHORA_RE.search(str(user_message or ""))
            or looks_like_doctor_followup_message(
                user_message=user_message,
                remembered_doctor=remembered,
            )
        ):
            out["doctor_name"] = remembered
            out["doctor_name_source"] = "session_memory"
            enriched_keys.append("doctor_name")

    remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
    if (
        (uses_service_tools or uses_address_tools)
        and not str(out.get("service_name") or out.get("test_name") or "").strip()
    ):
        if remembered_service and (
            active_state
            or has_contextual_entities
            or looks_like_service_followup_message(
                user_message=user_message,
                remembered_service=remembered_service,
            )
        ):
            out["service_name"] = remembered_service
            enriched_keys.append("service_name")
    if remembered_service and str(out.get("service_variant") or "").strip():
        combined = compose_service_with_variant(
            str(out.get("service_name") or remembered_service),
            str(out.get("service_variant") or ""),
        )
        if combined and combined != str(out.get("service_name") or "").strip():
            out["service_name"] = combined
            enriched_keys.append("service_name")

    if uses_result_tool:
        for key in (
            "result_surname",
            "result_year_of_birth",
            "result_analysis_code",
            "result_analysis_number",
        ):
            if str(out.get(key) or "").strip():
                continue
            value = str(memory_entities.get(key) or "").strip()
            if not value:
                continue
            out[key] = value
            enriched_keys.append(key)

    if active_state or has_contextual_entities:
        for key in ("branch_name", "city", "date", "date_from", "date_to", "time", "time_from", "time_to"):
            if str(out.get(key) or "").strip():
                continue
            value = str(memory_entities.get(key) or "").strip()
            if not value:
                continue
            out[key] = value
            enriched_keys.append(key)

    if enriched_keys:
        log_event(
            "medical_entities_enriched_from_memory",
            session_id=session_id,
            keys=",".join(sorted(set(enriched_keys))),
            message=user_message[:140],
        )
    return out


async def remember_doctor_from_tool_result(
    *,
    memory: Any,
    session_id: str,
    tool_name: str,
    payload: dict[str, Any],
) -> None:
    if memory is None:
        return
    doctor = extract_primary_doctor_name(tool_name, payload)
    if not doctor:
        return
    await memory.set_meta_str(session_id, LAST_DOCTOR_NAME_KEY, doctor)
    log_event(
        "doctor_context_stored",
        session_id=session_id,
        tool_name=tool_name,
        doctor_name=doctor,
    )
