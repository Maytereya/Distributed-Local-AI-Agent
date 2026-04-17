"""Deterministic appointment flow policy for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .contracts import DialogState
from .signal_parsers import extract_person_name, parse_yes_no


_BOOK_RE = re.compile(r"\b(запис\w*|запись)\b", re.I)
_RESCHEDULE_RE = re.compile(r"\b(перенест\w*|перезапис\w*)\b", re.I)
_CANCEL_BOOKING_RE = re.compile(r"\b(отмен\w*\s+запис|отмена\s+запис)\b", re.I)
_SCHEDULE_PREVIEW_RE = re.compile(
    r"\b(расписан\w*|график|свободн\w*\s+(?:окн\w*|слот\w*)|слот\w*|окн\w*)\b",
    re.I,
)

_MONTHS_RU = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}

APPOINTMENT_MEMORY_CLEAR_KEYS: tuple[str, ...] = (
    "appointment_action",
    "appointment_windows",
    "appointment_branch_options",
    "patient_name",
    "date",
    "date_from",
    "date_to",
    "time",
    "time_from",
    "time_to",
    "branch_name",
)


@dataclass(slots=True)
class AppointmentPrecheckResult:
    handled: bool = False
    reply_text: str = ""
    next_state: DialogState | None = None
    clear_state: bool = False
    handoff: bool = False
    clear_memory_keys: tuple[str, ...] = ()
    save_memory_entities: dict[str, Any] = field(default_factory=dict)


def looks_like_appointment_intent_message(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    return bool(_BOOK_RE.search(probe) or _RESCHEDULE_RE.search(probe) or _CANCEL_BOOKING_RE.search(probe))


def detect_appointment_action(text: str, *, current_action: str = "") -> str:
    probe = str(text or "").strip()
    if _RESCHEDULE_RE.search(probe):
        return "reschedule"
    if _CANCEL_BOOKING_RE.search(probe):
        return "cancel"
    if _BOOK_RE.search(probe):
        return "book"
    return str(current_action or "").strip().lower()


def appointment_confirmation_transition(text: str) -> str:
    return parse_yes_no(text, profile="strict")


def extract_patient_name(text: str) -> str:
    probe = str(text or "").strip()
    if not probe or any(ch.isdigit() for ch in probe):
        return ""
    if _BOOK_RE.search(probe) or _RESCHEDULE_RE.search(probe) or _CANCEL_BOOKING_RE.search(probe):
        return ""
    return extract_person_name(probe)


def looks_like_schedule_preview_request(text: str) -> bool:
    return bool(_SCHEDULE_PREVIEW_RE.search(str(text or "").strip()))


def has_schedule_context(memory_entities: dict[str, Any]) -> bool:
    doctor_name = str((memory_entities or {}).get("doctor_name") or "").strip()
    windows = memory_entities.get("appointment_windows")
    return bool(doctor_name and isinstance(windows, list) and windows)


def should_start_appointment_from_context(text: str, memory_entities: dict[str, Any], contextual_entities: dict[str, Any]) -> bool:
    if looks_like_appointment_intent_message(text):
        return has_schedule_context(memory_entities)
    has_datetime = bool(str((contextual_entities or {}).get("date") or "").strip()) and bool(
        str((contextual_entities or {}).get("time") or "").strip()
    )
    return has_datetime and has_schedule_context(memory_entities)


def merge_appointment_entities(
    *,
    user_message: str,
    base_entities: dict[str, Any],
    memory_entities: dict[str, Any],
    contextual_entities: dict[str, Any],
) -> dict[str, Any]:
    out = dict(base_entities or {})
    for key in (
        "doctor_name",
        "specialty",
        "service_name",
        "branch_name",
        "city",
        "appointment_windows",
        "appointment_branch_options",
    ):
        if str(out.get(key) or "").strip():
            continue
        value = memory_entities.get(key)
        if str(value or "").strip():
            out[key] = value
    for key, value in (contextual_entities or {}).items():
        if str(value or "").strip():
            out[key] = value
    patient_name = extract_patient_name(user_message)
    if patient_name:
        out["patient_name"] = patient_name
    action = detect_appointment_action(user_message, current_action=str(out.get("appointment_action") or ""))
    if action:
        out["appointment_action"] = action
    if not str(out.get("appointment_action") or "").strip():
        out["appointment_action"] = "book"
    _apply_branch_from_slot(out)
    return out


def appointment_missing_slots(entities: dict[str, Any]) -> list[str]:
    data = dict(entities or {})
    action = str(data.get("appointment_action") or "book").strip().lower() or "book"
    missing: list[str] = []
    doctor_known = bool(str(data.get("doctor_name") or data.get("doctor_id") or "").strip())
    specialty_known = bool(str(data.get("specialty") or "").strip())
    branch_known = bool(str(data.get("branch_name") or data.get("city") or "").strip())
    date_known = bool(str(data.get("date") or data.get("date_from") or "").strip())
    time_known = bool(str(data.get("time") or data.get("time_from") or "").strip())
    patient_name = bool(str(data.get("patient_name") or "").strip())

    if not str(data.get("appointment_action") or "").strip():
        missing.append("appointment_action")
    if not doctor_known and not specialty_known:
        missing.extend(["doctor_name", "specialty"])
    branch_options = data.get("appointment_branch_options")
    if not branch_known and isinstance(branch_options, list) and len(branch_options) > 1:
        missing.append("branch_or_city")
    if action in {"book", "reschedule"}:
        if not date_known:
            missing.append("date")
        if not time_known:
            missing.append("time")
    if not patient_name:
        missing.append("patient_name")

    dedup: list[str] = []
    seen: set[str] = set()
    for slot in missing:
        if slot in seen:
            continue
        seen.add(slot)
        dedup.append(slot)
    return dedup


def appointment_clarify_question(missing_slots: list[str], entities: dict[str, Any]) -> str:
    slots = set(str(slot or "").strip().lower() for slot in (missing_slots or []))
    action = str((entities or {}).get("appointment_action") or "book").strip().lower() or "book"
    if "appointment_action" in slots:
        return "Уточните, пожалуйста, что нужно сделать: записаться, перенести или отменить запись."
    if slots & {"doctor_name", "specialty"}:
        if action == "cancel":
            return "Уточните, пожалуйста, к какому врачу нужно отменить запись."
        if action == "reschedule":
            return "Уточните, пожалуйста, к какому врачу или по какой специальности нужно перенести запись."
        return "Уточните, пожалуйста, к какому врачу или по какой специальности нужна запись."
    if "branch_or_city" in slots:
        return "Уточните, пожалуйста, удобный филиал для записи."
    if "date" in slots and "time" in slots:
        return "Выберите, пожалуйста, дату и время для записи."
    if "date" in slots:
        return "Уточните, пожалуйста, удобную дату для записи."
    if "time" in slots:
        return "Уточните, пожалуйста, удобное время для записи."
    if "patient_name" in slots:
        if action == "cancel":
            return "Сообщите, пожалуйста, ваше ФИО для отмены записи."
        if action == "reschedule":
            return "Сообщите, пожалуйста, ваше ФИО для переноса записи."
        return "Сообщите, пожалуйста, ваше ФИО для записи."
    return "Уточните, пожалуйста, данные для записи."


def appointment_confirmation_text(entities: dict[str, Any]) -> str:
    data = dict(entities or {})
    action = str(data.get("appointment_action") or "book").strip().lower() or "book"
    doctor_name = str(data.get("doctor_name") or data.get("specialty") or "врач").strip()
    branch_name = str(data.get("branch_name") or "").strip()
    date_label = _format_human_date(str(data.get("date") or data.get("date_from") or ""))
    time_label = str(data.get("time") or data.get("time_from") or "").strip()
    patient_name = str(data.get("patient_name") or "").strip()

    parts: list[str] = []
    if action == "cancel":
        parts.append("Подтверждаю отмену записи")
    elif action == "reschedule":
        parts.append("Подтверждаю перенос записи")
    else:
        parts.append("Вы выбрали")
    if date_label and time_label:
        parts.append(f"{date_label}, {time_label}")
    elif date_label:
        parts.append(date_label)
    elif time_label:
        parts.append(time_label)
    if doctor_name:
        parts.append(f"врач {doctor_name}")
    if branch_name:
        parts.append(branch_name)
    if action == "book" and parts:
        summary = f"{parts[0]} {', '.join(parts[1:])}".strip()
    else:
        summary = ", ".join(parts).strip(", ")
    if patient_name:
        summary += f". ФИО: {patient_name}"
    if action == "cancel":
        return f"{summary}. Верно? Ответьте: да или нет."
    if action == "reschedule":
        return f"{summary}. Перенести запись? Ответьте: да или нет."
    return f"{summary}. Продолжить запись? Ответьте: да или нет."


def appointment_handoff_text(entities: dict[str, Any]) -> str:
    action = str((entities or {}).get("appointment_action") or "book").strip().lower() or "book"
    if action == "cancel":
        return "Запрос на отмену записи зафиксирован. Передаю его оператору для подтверждения."
    if action == "reschedule":
        return "Запрос на перенос записи зафиксирован. Передаю его оператору для подтверждения."
    return "Запрос на запись зафиксирован. Передаю его оператору для подтверждения."


def appointment_resume_question(dialog_state: DialogState, entities: dict[str, Any]) -> str:
    open_question = str(dialog_state.open_question or "").strip()
    if open_question:
        return open_question
    return appointment_clarify_question(list(dialog_state.missing_slots or []), entities)


def apply_appointment_precheck(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
    contextual_entities: dict[str, Any],
) -> AppointmentPrecheckResult:
    current_state = dialog_state if isinstance(dialog_state, DialogState) else DialogState()
    active = str(current_state.intent or "").strip().lower() == "appointment"
    message = str(user_message or "").strip()
    memory_entities = dict(memory_entities or {})
    contextual_entities = dict(contextual_entities or {})

    if active:
        return _apply_active_appointment_turn(
            user_message=message,
            dialog_state=current_state,
            memory_entities=memory_entities,
            contextual_entities=contextual_entities,
        )

    if should_start_appointment_from_context(message, memory_entities, contextual_entities):
        entities = merge_appointment_entities(
            user_message=message,
            base_entities={},
            memory_entities=memory_entities,
            contextual_entities=contextual_entities,
        )
        missing_slots = appointment_missing_slots(entities)
        if missing_slots:
            clarify_text = appointment_clarify_question(missing_slots, entities)
            return AppointmentPrecheckResult(
                handled=True,
                reply_text=clarify_text,
                next_state=build_appointment_state(
                    entities=entities,
                    missing_slots=missing_slots,
                    phase="appointment_collecting",
                    open_question=clarify_text,
                ),
            )
        confirm_text = appointment_confirmation_text(entities)
        return AppointmentPrecheckResult(
            handled=True,
            reply_text=confirm_text,
            next_state=build_appointment_state(
                entities=entities,
                missing_slots=[],
                phase="appointment_confirm",
                open_question=confirm_text,
            ),
        )

    return AppointmentPrecheckResult()


def _apply_active_appointment_turn(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
    contextual_entities: dict[str, Any],
) -> AppointmentPrecheckResult:
    entities = merge_appointment_entities(
        user_message=user_message,
        base_entities=dict(dialog_state.entities or {}),
        memory_entities=memory_entities,
        contextual_entities=contextual_entities,
    )
    phase = str(dialog_state.phase or "").strip().lower()
    transition = appointment_confirmation_transition(user_message)

    if phase == "appointment_confirm":
        if transition == "yes":
            return AppointmentPrecheckResult(
                handled=True,
                reply_text=appointment_handoff_text(entities),
                clear_state=True,
                handoff=True,
                clear_memory_keys=APPOINTMENT_MEMORY_CLEAR_KEYS,
            )
        if transition == "no":
            clarify_text = "Что нужно изменить в записи: дату, время, филиал или ФИО?"
            next_missing = appointment_missing_slots(entities)
            return AppointmentPrecheckResult(
                handled=True,
                reply_text=clarify_text,
                next_state=build_appointment_state(
                    entities=entities,
                    missing_slots=next_missing,
                    phase="appointment_collecting",
                    open_question=clarify_text,
                ),
            )
        return AppointmentPrecheckResult(
            handled=True,
            reply_text=str(dialog_state.open_question or appointment_confirmation_text(entities)),
            next_state=build_appointment_state(
                entities=entities,
                missing_slots=[],
                phase="appointment_confirm",
                open_question=str(dialog_state.open_question or appointment_confirmation_text(entities)),
            ),
        )

    if looks_like_schedule_preview_request(user_message):
        preview_text = render_cached_appointment_schedule_preview(entities)
        if preview_text:
            resume_text = appointment_resume_question(dialog_state, entities)
            next_state = build_appointment_state(
                entities=entities,
                missing_slots=list(dialog_state.missing_slots or appointment_missing_slots(entities)),
                phase="appointment_collecting",
                open_question=resume_text,
            )
            return AppointmentPrecheckResult(
                handled=True,
                reply_text=preview_text,
                next_state=next_state,
            )

    if phase == "appointment_collecting" and not _looks_like_active_appointment_followup(
        user_message=user_message,
        contextual_entities=contextual_entities,
        entities=entities,
    ):
        return AppointmentPrecheckResult()

    missing_slots = appointment_missing_slots(entities)
    if missing_slots:
        clarify_text = appointment_clarify_question(missing_slots, entities)
        return AppointmentPrecheckResult(
            handled=True,
            reply_text=clarify_text,
            next_state=build_appointment_state(
                entities=entities,
                missing_slots=missing_slots,
                phase="appointment_collecting",
                open_question=clarify_text,
            ),
        )

    confirm_text = appointment_confirmation_text(entities)
    return AppointmentPrecheckResult(
        handled=True,
        reply_text=confirm_text,
        next_state=build_appointment_state(
            entities=entities,
            missing_slots=[],
            phase="appointment_confirm",
            open_question=confirm_text,
        ),
    )


def _looks_like_active_appointment_followup(
    *,
    user_message: str,
    contextual_entities: dict[str, Any],
    entities: dict[str, Any],
) -> bool:
    if looks_like_appointment_intent_message(user_message):
        return True
    if looks_like_schedule_preview_request(user_message):
        return True
    if extract_patient_name(user_message):
        return True
    if contextual_entities:
        return True
    return bool(str((entities or {}).get("patient_name") or "").strip() and appointment_confirmation_transition(user_message) != "unknown")


def build_appointment_state(
    *,
    entities: dict[str, Any],
    missing_slots: list[str],
    phase: str,
    open_question: str,
) -> DialogState:
    phase_value = str(phase or "").strip().lower()
    flow_stage = "confirm" if "confirm" in phase_value else "collecting"
    return DialogState(
        route="clinical",
        intent="appointment",
        entities=dict(entities or {}),
        candidate_entities={},
        confirmation_target="",
        missing_slots=list(missing_slots or []),
        clarify_type="missing_auth_data" if "patient_name" in set(missing_slots or []) else "identify",
        tool_plan=[],
        response_policy="tool_only",
        confidence=1.0,
        clarify_count=1,
        last_tool="",
        phase=str(phase or ""),
        open_question=str(open_question or ""),
        flow_active=True,
        flow_kind="appointment",
        flow_stage=flow_stage,
        flow_interruptible=True,
        flow_resume_question=str(open_question or ""),
        expected_slots=[str(slot).strip() for slot in list(missing_slots or []) if str(slot).strip()],
    )

def _apply_branch_from_slot(entities: dict[str, Any]) -> None:
    branch_name = str(entities.get("branch_name") or "").strip()
    if branch_name:
        return
    windows = entities.get("appointment_windows")
    date_value = str(entities.get("date") or entities.get("date_from") or "").strip()
    time_value = str(entities.get("time") or entities.get("time_from") or "").strip()
    if isinstance(windows, list) and date_value and time_value:
        matched_branches: list[str] = []
        for item in windows:
            if not isinstance(item, dict):
                continue
            item_date = str(item.get("date") or "").strip()
            item_time = str(item.get("time") or "").strip()
            item_branch = str(item.get("branch_name") or item.get("branch") or "").strip()
            if item_date == date_value and item_time[:5] == time_value[:5] and item_branch:
                matched_branches.append(item_branch)
        unique = sorted(set(matched_branches))
        if len(unique) == 1:
            entities["branch_name"] = unique[0]
            return
    branch_options = entities.get("appointment_branch_options")
    if isinstance(branch_options, list) and len(branch_options) == 1:
        only = str(branch_options[0] or "").strip()
        if only:
            entities["branch_name"] = only


def _format_human_date(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = raw.split("-")
    if len(parts) >= 3:
        try:
            month_num = int(parts[1])
            day_num = int(parts[2])
        except Exception:
            return raw
        month = _MONTHS_RU.get(month_num)
        if month:
            return f"{day_num} {month}"
    return raw


def render_cached_appointment_schedule_preview(entities: dict[str, Any]) -> str:
    windows = entities.get("appointment_windows")
    if not isinstance(windows, list) or not windows:
        return ""
    doctor_name = str(entities.get("doctor_name") or "").strip() or str(windows[0].get("doctor_name") or "").strip()
    grouped: dict[str, dict[str, list[str]]] = {}
    for item in windows:
        if not isinstance(item, dict):
            continue
        branch_name = str(item.get("branch_name") or item.get("branch") or "").strip() or "Филиал"
        date_value = str(item.get("date") or "").strip()
        time_value = str(item.get("time") or "").strip()
        if not (date_value and time_value):
            continue
        branch_bucket = grouped.setdefault(branch_name, {})
        branch_bucket.setdefault(date_value, [])
        if time_value not in branch_bucket[date_value]:
            branch_bucket[date_value].append(time_value)
    if not grouped:
        return ""

    lines: list[str] = ["Нашел расписание:"]
    if doctor_name:
        lines.append(doctor_name)
    rendered_days = 0
    for branch_name, days in grouped.items():
        lines.append(branch_name)
        for date_value in sorted(days.keys()):
            rendered_days += 1
            if rendered_days > 4:
                break
            human_date = _format_human_date(date_value) or date_value
            slots = ", ".join(days[date_value][:6])
            lines.append(f"• {human_date}: {slots}")
        if rendered_days > 4:
            break
    lines.append("Если хотите записаться, выберите дату и время из предложенных, и я продолжу запись.")
    return "\n".join(line for line in lines if str(line).strip())
