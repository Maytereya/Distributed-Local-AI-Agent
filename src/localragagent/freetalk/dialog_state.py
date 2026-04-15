"""Dialog-state and session-memory helpers for FreeTalk."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from .routing_contract import normalize_clarify_type
from .contracts import DialogState


CLINICAL_DIALOG_STATE_KEY = "clinical_dialog_state"
CLINICAL_ENTITY_MEMORY_KEY = "clinical_entity_memory"
SESSION_MEMORY_ENTITY_KEYS: tuple[str, ...] = (
    "appointment_action",
    "doctor_name",
    "specialty",
    "service_name",
    "test_name",
    "service_variant",
    "city",
    "branch_name",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "date",
    "time",
    "result_surname",
    "result_year_of_birth",
    "result_analysis_code",
    "result_analysis_number",
    "doctor_id",
    "appointment_windows",
    "appointment_branch_options",
)


def merge_missing_slots(primary: list[str], secondary: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for bucket in (primary or [], secondary or []):
        values = bucket if isinstance(bucket, list) else [bucket]
        for source in values:
            slot = str(source or "").strip().lower()
            if not slot or slot in seen:
                continue
            seen.add(slot)
            out.append(slot)
    return out


def filter_missing_slots_by_entities(missing_slots: list[str], entities: dict[str, Any]) -> list[str]:
    out: list[str] = []
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
    appointment_action = bool(str(entities.get("appointment_action") or "").strip())
    patient_name = bool(str(entities.get("patient_name") or "").strip())
    branch_known = bool(str(entities.get("branch_name") or entities.get("city") or "").strip())
    date_known = bool(str(entities.get("date") or entities.get("date_from") or "").strip())
    time_known = bool(str(entities.get("time") or entities.get("time_from") or "").strip())
    for slot in (missing_slots or []):
        name = str(slot or "").strip().lower()
        if not name:
            continue
        if name in {"doctor_name", "specialty"} and (doctor_known or specialty_known):
            continue
        if name == "service_or_analysis_name" and (service_known or doctor_known):
            continue
        if name == "appointment_action" and appointment_action:
            continue
        if name == "branch_or_city" and branch_known:
            continue
        if name == "date" and date_known:
            continue
        if name == "time" and time_known:
            continue
        if name == "patient_name" and patient_name:
            continue
        if name == "result_surname" and result_surname:
            continue
        if name == "result_year_of_birth" and result_year:
            continue
        if name == "result_analysis_code" and result_code:
            continue
        if name == "result_analysis_number" and result_number:
            continue
        out.append(name)
    return out


def merge_entities(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (updates or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        text = str(value or "").strip()
        if not text:
            continue
        out[name] = value
    return out


def copy_dialog_state(dialog_state: DialogState | None) -> DialogState:
    if not isinstance(dialog_state, DialogState):
        return DialogState()
    return DialogState(
        route=str(dialog_state.route or ""),
        intent=str(dialog_state.intent or ""),
        entities=dict(dialog_state.entities or {}),
        candidate_entities=dict(dialog_state.candidate_entities or {}),
        confirmation_target=str(dialog_state.confirmation_target or ""),
        missing_slots=list(dialog_state.missing_slots or []),
        clarify_type=normalize_clarify_type(dialog_state.clarify_type),
        tool_plan=list(dialog_state.tool_plan or []),
        response_policy=str(dialog_state.response_policy or ""),
        confidence=float(dialog_state.confidence or 0.0),
        clarify_count=int(dialog_state.clarify_count or 0),
        last_tool=str(dialog_state.last_tool or ""),
        phase=str(dialog_state.phase or ""),
        open_question=str(dialog_state.open_question or ""),
    )


def dialog_state_is_active(dialog_state: DialogState | None) -> bool:
    if not isinstance(dialog_state, DialogState):
        return False
    intent = str(dialog_state.intent or "").strip().lower()
    return bool(
        (intent and intent != "unknown")
        or dialog_state.entities
        or dialog_state.missing_slots
        or str(dialog_state.clarify_type or "").strip()
        or dialog_state.tool_plan
        or str(dialog_state.confirmation_target or "").strip()
        or str(dialog_state.phase or "").strip()
    )


def dialog_state_payload(dialog_state: DialogState) -> dict[str, Any]:
    payload = asdict(dialog_state) if isinstance(dialog_state, DialogState) else {}
    return payload if isinstance(payload, dict) else {}


def dialog_state_from_payload(payload: dict[str, Any] | None) -> DialogState:
    data = payload if isinstance(payload, dict) else {}
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    candidate_entities = (
        data.get("candidate_entities")
        if isinstance(data.get("candidate_entities"), dict)
        else {}
    )
    missing_slots = data.get("missing_slots") if isinstance(data.get("missing_slots"), list) else []
    tool_plan = data.get("tool_plan") if isinstance(data.get("tool_plan"), list) else []
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    try:
        clarify_count = int(data.get("clarify_count", 0))
    except Exception:
        clarify_count = 0
    return DialogState(
        route=str(data.get("route") or "").strip() or "general",
        intent=str(data.get("intent") or "").strip() or "unknown",
        entities={str(k): v for k, v in entities.items() if str(k).strip() and str(v or "").strip()},
        candidate_entities={
            str(k): v for k, v in candidate_entities.items() if str(k).strip() and str(v or "").strip()
        },
        confirmation_target=str(data.get("confirmation_target") or "").strip(),
        missing_slots=[str(x).strip() for x in missing_slots if str(x).strip()],
        clarify_type=normalize_clarify_type(data.get("clarify_type")),
        tool_plan=[str(x).strip() for x in tool_plan if str(x).strip()],
        response_policy=str(data.get("response_policy") or "").strip() or "general_only",
        confidence=max(0.0, min(1.0, confidence)),
        clarify_count=max(0, clarify_count),
        last_tool=str(data.get("last_tool") or "").strip(),
        phase=str(data.get("phase") or "").strip(),
        open_question=str(data.get("open_question") or "").strip(),
    )


async def load_dialog_state(memory: Any, session_id: str) -> DialogState:
    if memory is None:
        return DialogState()
    raw = await memory.get_meta_str(session_id, CLINICAL_DIALOG_STATE_KEY, "")
    text = str(raw or "").strip()
    if text:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {}
        state = dialog_state_from_payload(parsed if isinstance(parsed, dict) else {})
        if dialog_state_is_active(state):
            return state
    return DialogState()


async def save_dialog_state(memory: Any, session_id: str, dialog_state: DialogState) -> None:
    if memory is None:
        return
    payload = dialog_state_payload(dialog_state)
    try:
        encoded = json.dumps(payload, ensure_ascii=False)
    except Exception:
        encoded = "{}"
    await memory.set_meta_str(session_id, CLINICAL_DIALOG_STATE_KEY, encoded)


async def clear_dialog_state(memory: Any, session_id: str) -> None:
    if memory is None:
        return
    await memory.set_meta_str(session_id, CLINICAL_DIALOG_STATE_KEY, "")


async def load_session_entity_memory(memory: Any, session_id: str) -> dict[str, Any]:
    if memory is None:
        return {}
    raw = await memory.get_meta_str(session_id, CLINICAL_ENTITY_MEMORY_KEY, "")
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): v for k, v in parsed.items() if str(k).strip() and str(v or "").strip()}


async def save_session_entity_memory(memory: Any, session_id: str, entities: dict[str, Any]) -> None:
    if memory is None:
        return
    current = await load_session_entity_memory(memory, session_id)
    updates: dict[str, Any] = {}
    for key in SESSION_MEMORY_ENTITY_KEYS:
        value = entities.get(key)
        if not str(value or "").strip():
            continue
        updates[key] = value
    merged = merge_entities(current, updates)
    if not merged:
        return
    try:
        encoded = json.dumps(merged, ensure_ascii=False)
    except Exception:
        encoded = "{}"
    await memory.set_meta_str(session_id, CLINICAL_ENTITY_MEMORY_KEY, encoded)


async def clear_session_entity_memory_keys(memory: Any, session_id: str, keys: list[str] | tuple[str, ...]) -> None:
    if memory is None:
        return
    current = await load_session_entity_memory(memory, session_id)
    if not current:
        return
    updated = dict(current)
    changed = False
    for key in (keys or []):
        name = str(key or "").strip()
        if not name:
            continue
        if name in updated:
            updated.pop(name, None)
            changed = True
    if not changed:
        return
    if not updated:
        await memory.set_meta_str(session_id, CLINICAL_ENTITY_MEMORY_KEY, "")
        return
    try:
        encoded = json.dumps(updated, ensure_ascii=False)
    except Exception:
        encoded = "{}"
    await memory.set_meta_str(session_id, CLINICAL_ENTITY_MEMORY_KEY, encoded)
