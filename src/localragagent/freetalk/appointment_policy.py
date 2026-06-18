"""Deterministic appointment flow policy for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .contracts import DialogState
from .signal_parsers import (
    extract_branch_reference,
    extract_confirmation_head,
    extract_contextual_entities,
    extract_doctor_reference_candidate,
    extract_person_name,
    extract_specialty_reference,
    looks_like_specific_doctor_reference,
    parse_yes_no,
)


_BOOK_RE = re.compile(r"\b(запис\w*|запись)\b", re.I)
_RESCHEDULE_RE = re.compile(r"\b(перенест\w*|перезапис\w*)\b", re.I)
_CANCEL_BOOKING_RE = re.compile(r"\b(отмен\w*\s+запис|отмена\s+запис)\b", re.I)
_SCHEDULE_PREVIEW_RE = re.compile(
    r"\b(расписан\w*|график|свободн\w*\s+(?:окн\w*|слот\w*)|слот\w*|окн\w*)\b",
    re.I,
)
_REPAIR_MARKER_RE = re.compile(
    r"\b(не\s+тот|не\s+та|не\s+то|не\s+эт|друг\w+|лучше|поменя\w*|измени\w*|замен\w*)\b",
    re.I,
)
_REPLACEMENT_TAIL_RE = re.compile(r"\bа\s+(.+)$", re.I)
_PATIENT_LABEL_RE = re.compile(r"\b(фио|пациент\w*|имя)\b\s*[:\-]?\s*(.+)$", re.I)
_BRANCH_REPAIR_CUE_RE = re.compile(r"\b(филиал\w*|адрес\w*|клиник\w*)\b", re.I)
_DOCTOR_REPAIR_CUE_RE = re.compile(r"\b(врач\w*|доктор\w*|к\s+врачу|к\s+доктору)\b", re.I)
_NEGATED_DOCTOR_SPECIALTY_RE = re.compile(
    r"\b([А-ЯЁA-Z][а-яёa-z\-]+)\s+не\s+([а-яёa-z\- ]{3,48})\b",
    re.I,
)
_TIME_EXACT_RE = re.compile(r"^\d{1,2}:\d{2}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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


@dataclass(slots=True)
class AppointmentRepairPatch:
    updates: dict[str, Any] = field(default_factory=dict)
    clear_keys: set[str] = field(default_factory=set)


@dataclass(slots=True)
class AppointmentWindowCheck:
    ok: bool = True
    entities: dict[str, Any] = field(default_factory=dict)
    reply_text: str = ""
    missing_slots: list[str] = field(default_factory=list)


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
    repair_patch = _extract_appointment_repair_patch(
        user_message=user_message,
        current_entities=dict(dialog_state.entities or {}),
        contextual_entities=contextual_entities,
    )
    repair_applied = bool(repair_patch.updates or repair_patch.clear_keys)
    if repair_applied:
        entities = _apply_appointment_repair_patch(entities, repair_patch)
        _apply_branch_from_slot(entities)

    if phase == "appointment_confirm":
        if transition == "yes":
            return AppointmentPrecheckResult(
                handled=True,
                reply_text=appointment_handoff_text(entities),
                clear_state=True,
                handoff=True,
                clear_memory_keys=APPOINTMENT_MEMORY_CLEAR_KEYS,
            )
        if transition == "no" and not repair_applied:
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
        if not repair_applied:
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

    if not repair_applied and looks_like_schedule_preview_request(user_message):
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

    if phase == "appointment_collecting" and not repair_applied and not _looks_like_active_appointment_followup(
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

    window_check = _check_appointment_window(entities)
    if not window_check.ok:
        return AppointmentPrecheckResult(
            handled=True,
            reply_text=window_check.reply_text,
            next_state=build_appointment_state(
                entities=window_check.entities,
                missing_slots=window_check.missing_slots,
                phase="appointment_collecting",
                open_question=window_check.reply_text,
            ),
        )
    if window_check.entities:
        entities = window_check.entities
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


def _extract_appointment_repair_patch(
    *,
    user_message: str,
    current_entities: dict[str, Any],
    contextual_entities: dict[str, Any],
) -> AppointmentRepairPatch:
    message = str(user_message or "").strip()
    if not message:
        return AppointmentRepairPatch()
    decision, remainder = extract_confirmation_head(message)
    work = str(remainder or "").strip() if decision == "no" and str(remainder or "").strip() else message
    source = _appointment_repair_source(work)
    context_source = extract_contextual_entities(source) if source and source != message else dict(contextual_entities or {})
    patch = AppointmentRepairPatch()

    patient_name = _extract_patient_name_repair(work=work, source=source)
    if patient_name:
        patch.updates["patient_name"] = patient_name

    negated_doctor, negated_specialty = _extract_negated_doctor_specialty(work)
    if negated_doctor:
        patch.updates["doctor_name"] = negated_doctor
    if negated_specialty:
        patch.clear_keys.add("specialty")

    specialty = extract_specialty_reference(source)
    if specialty and not negated_specialty:
        patch.updates["specialty"] = specialty

    doctor_name = _extract_doctor_repair_candidate(
        work=work,
        source=source,
        current_entities=current_entities,
        patient_name=patient_name,
        specialty=specialty,
    )
    if doctor_name:
        patch.updates["doctor_name"] = doctor_name

    branch_name = str(context_source.get("branch_name") or "").strip()
    if not branch_name:
        branch_name = _extract_short_branch_repair(
            work=work,
            source=source,
            current_entities=current_entities,
            has_patient=bool(patient_name),
            has_doctor=bool(doctor_name),
            has_specialty=bool(specialty or negated_specialty),
        )
    if branch_name:
        patch.updates["branch_name"] = branch_name
    city = str(context_source.get("city") or "").strip()
    if city:
        patch.updates["city"] = city

    for key in ("date", "date_from", "date_to", "time", "time_from", "time_to"):
        value = str(context_source.get(key) or "").strip()
        if value:
            patch.updates[key] = value

    _apply_repair_dependency_clears(patch, current_entities=current_entities)
    return patch


def _appointment_repair_source(text: str) -> str:
    source = str(text or "").strip(" ,.")
    if not source:
        return ""
    replacement = _REPLACEMENT_TAIL_RE.search(source)
    if replacement and (_REPAIR_MARKER_RE.search(source) or re.search(r"\bне\b", source, re.I)):
        return str(replacement.group(1) or "").strip(" ,.")
    return source


def _extract_patient_name_repair(*, work: str, source: str) -> str:
    for candidate in (source, work):
        match = _PATIENT_LABEL_RE.search(str(candidate or "").strip())
        if not match:
            continue
        value = str(match.group(2) or "").strip(" ,.")
        replacement = _appointment_repair_source(value)
        person = extract_person_name(replacement)
        if person:
            return person
    if _PATIENT_LABEL_RE.search(str(work or "")):
        person = extract_person_name(source)
        if person:
            return person
    return ""


def _extract_negated_doctor_specialty(text: str) -> tuple[str, str]:
    match = _NEGATED_DOCTOR_SPECIALTY_RE.search(str(text or "").strip())
    if not match:
        return "", ""
    doctor_candidate = str(match.group(1) or "").strip()
    if not looks_like_specific_doctor_reference(doctor_candidate):
        return "", ""
    specialty = extract_specialty_reference(str(match.group(2) or "").strip())
    if not specialty:
        return "", ""
    return doctor_candidate, specialty


def _extract_doctor_repair_candidate(
    *,
    work: str,
    source: str,
    current_entities: dict[str, Any],
    patient_name: str,
    specialty: str,
) -> str:
    if patient_name or specialty:
        return ""
    candidate = extract_doctor_reference_candidate(source)
    if not candidate:
        return ""
    if _DOCTOR_REPAIR_CUE_RE.search(work):
        return candidate
    current_doctor = str((current_entities or {}).get("doctor_name") or "").strip()
    if current_doctor and len(candidate.split()) <= 2 and (_REPAIR_MARKER_RE.search(work) or re.search(r"\bне\b", work, re.I)):
        return candidate
    return ""


def _extract_short_branch_repair(
    *,
    work: str,
    source: str,
    current_entities: dict[str, Any],
    has_patient: bool,
    has_doctor: bool,
    has_specialty: bool,
) -> str:
    if has_patient or has_doctor or has_specialty:
        return ""
    if extract_contextual_entities(source):
        return ""
    explicit = extract_branch_reference(source)
    if explicit:
        return explicit
    current_branch = str((current_entities or {}).get("branch_name") or "").strip()
    if not (
        _BRANCH_REPAIR_CUE_RE.search(work)
        or (current_branch and (_REPAIR_MARKER_RE.search(work) or re.search(r"\bне\b", work, re.I)))
    ):
        return ""
    candidate = str(source or "").strip(" ,.")
    if not candidate or len(candidate.split()) > 5:
        return ""
    if extract_person_name(candidate) or extract_specialty_reference(candidate):
        return ""
    return candidate


def _apply_repair_dependency_clears(patch: AppointmentRepairPatch, *, current_entities: dict[str, Any]) -> None:
    updates = patch.updates
    current = dict(current_entities or {})
    if "doctor_name" in updates and _normalized_text(updates.get("doctor_name")) != _normalized_text(current.get("doctor_name")):
        patch.clear_keys.update(
            {
                "doctor_id",
                "appointment_windows",
                "appointment_branch_options",
                "date",
                "date_from",
                "date_to",
                "time",
                "time_from",
                "time_to",
                "branch_name",
            }
        )
        if "specialty" not in updates:
            patch.clear_keys.add("specialty")
    if "specialty" in updates and _normalized_text(updates.get("specialty")) != _normalized_text(current.get("specialty")):
        if "doctor_name" not in updates:
            patch.clear_keys.update({"doctor_name", "doctor_id"})
        patch.clear_keys.update(
            {
                "appointment_windows",
                "appointment_branch_options",
                "date",
                "date_from",
                "date_to",
                "time",
                "time_from",
                "time_to",
                "branch_name",
            }
        )


def _apply_appointment_repair_patch(entities: dict[str, Any], patch: AppointmentRepairPatch) -> dict[str, Any]:
    out = dict(entities or {})
    for key in sorted(patch.clear_keys):
        if key in patch.updates:
            continue
        out.pop(key, None)
    for key, value in patch.updates.items():
        if str(value or "").strip():
            out[key] = value
    return out


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("ё", "е"))


def _check_appointment_window(entities: dict[str, Any]) -> AppointmentWindowCheck:
    data = dict(entities or {})
    rows = _appointment_window_rows(data.get("appointment_windows"))
    if not rows:
        return AppointmentWindowCheck(ok=True, entities=data)
    if not _appointment_has_date_filter(data) or not _appointment_has_time_filter(data):
        return AppointmentWindowCheck(ok=True, entities=data)

    matching = [row for row in rows if _appointment_row_matches(row, data)]
    if matching:
        chosen = matching[0]
        updated = dict(data)
        updated["date"] = chosen["date"]
        updated["date_from"] = chosen["date"]
        updated["date_to"] = chosen["date"]
        updated["time"] = chosen["time"]
        updated["time_from"] = chosen["time"]
        updated.pop("time_to", None)
        if chosen.get("branch_name"):
            updated["branch_name"] = chosen["branch_name"]
        return AppointmentWindowCheck(ok=True, entities=updated)

    repaired = dict(data)
    date_has_any = any(_appointment_row_matches_date(row, data) for row in rows)
    branch_has_any = any(_appointment_row_matches_branch(row, data) for row in rows)
    if not date_has_any:
        for key in ("date", "date_from", "date_to"):
            repaired.pop(key, None)
    if not branch_has_any and str(data.get("branch_name") or "").strip():
        repaired.pop("branch_name", None)
    for key in ("time", "time_from", "time_to"):
        repaired.pop(key, None)

    missing_slots = appointment_missing_slots(repaired)
    if "time" not in missing_slots:
        missing_slots.append("time")
    reply_text = _render_no_appointment_window_reply(rows=rows, requested=data)
    return AppointmentWindowCheck(
        ok=False,
        entities=repaired,
        reply_text=reply_text,
        missing_slots=missing_slots,
    )


def _appointment_window_rows(windows: Any) -> list[dict[str, str]]:
    if not isinstance(windows, list):
        return []
    rows: list[dict[str, str]] = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        date_value = str(item.get("date") or "").strip()
        time_value = str(item.get("time") or "").strip()
        branch_name = str(item.get("branch_name") or item.get("branch") or "").strip()
        if not (date_value and time_value):
            continue
        rows.append({"date": date_value, "time": time_value[:5], "branch_name": branch_name})
    rows.sort(key=lambda row: (row.get("date", ""), row.get("time", ""), row.get("branch_name", "")))
    return rows


def _appointment_has_date_filter(entities: dict[str, Any]) -> bool:
    return bool(str(entities.get("date") or entities.get("date_from") or entities.get("date_to") or "").strip())


def _appointment_has_time_filter(entities: dict[str, Any]) -> bool:
    return bool(str(entities.get("time") or entities.get("time_from") or entities.get("time_to") or "").strip())


def _appointment_row_matches(row: dict[str, str], entities: dict[str, Any]) -> bool:
    return (
        _appointment_row_matches_branch(row, entities)
        and _appointment_row_matches_date(row, entities)
        and _appointment_row_matches_time(row, entities)
    )


def _appointment_row_matches_branch(row: dict[str, str], entities: dict[str, Any]) -> bool:
    requested = str(entities.get("branch_name") or "").strip()
    if not requested:
        return True
    actual = str(row.get("branch_name") or "").strip()
    if not actual:
        return False
    req_norm = _normalized_text(requested)
    actual_norm = _normalized_text(actual)
    return bool(req_norm == actual_norm or req_norm in actual_norm or actual_norm in req_norm)


def _appointment_row_matches_date(row: dict[str, str], entities: dict[str, Any]) -> bool:
    row_date = str(row.get("date") or "").strip()
    if not row_date:
        return False
    requested = str(entities.get("date") or "").strip()
    if requested and requested not in {"weekend", "next_week", "this_week"}:
        return row_date == requested or _same_month_day(row_date, requested)
    date_from = str(entities.get("date_from") or "").strip()
    date_to = str(entities.get("date_to") or "").strip() or date_from
    if date_from and date_to and _ISO_DATE_RE.match(row_date) and _ISO_DATE_RE.match(date_from) and _ISO_DATE_RE.match(date_to):
        return date_from <= row_date <= date_to
    if date_from:
        return row_date == date_from or _same_month_day(row_date, date_from)
    return True


def _appointment_row_matches_time(row: dict[str, str], entities: dict[str, Any]) -> bool:
    row_time = str(row.get("time") or "").strip()[:5]
    requested = str(entities.get("time") or "").strip()
    if requested and _TIME_EXACT_RE.match(requested):
        return row_time == requested[:5]
    time_from = str(entities.get("time_from") or "").strip()
    time_to = str(entities.get("time_to") or "").strip()
    if time_from and time_to:
        return time_from[:5] <= row_time <= time_to[:5]
    if time_from and _TIME_EXACT_RE.match(time_from):
        return row_time == time_from[:5]
    if time_to:
        return row_time <= time_to[:5]
    return True


def _same_month_day(left: str, right: str) -> bool:
    return bool(_ISO_DATE_RE.match(left) and _ISO_DATE_RE.match(right) and left[5:] == right[5:])


def _render_no_appointment_window_reply(*, rows: list[dict[str, str]], requested: dict[str, Any]) -> str:
    alternatives = _appointment_window_alternatives(rows=rows, requested=requested)
    lines = ["На выбранные дату и время нет записи."]
    if alternatives:
        lines.append("Ближайшие варианты:")
        lines.extend(alternatives)
    lines.append("Выберите, пожалуйста, другой доступный слот.")
    return "\n".join(lines)


def _appointment_window_alternatives(*, rows: list[dict[str, str]], requested: dict[str, Any]) -> list[str]:
    scoped = [row for row in rows if _appointment_row_matches_branch(row, requested)]
    dated = [row for row in scoped if _appointment_row_matches_date(row, requested)]
    candidates = dated or scoped or list(rows)
    out: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for row in candidates[:6]:
        key = (row.get("date", ""), row.get("time", ""), row.get("branch_name", ""))
        if key in seen:
            continue
        seen.add(key)
        date_label = _format_human_date(row.get("date", "")) or row.get("date", "")
        branch = str(row.get("branch_name") or "").strip()
        suffix = f", {branch}" if branch else ""
        out.append(f"• {date_label}, {row.get('time', '')}{suffix}")
        if len(out) >= 4:
            break
    return out


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
