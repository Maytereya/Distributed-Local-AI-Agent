"""Flow-local deterministic handling for active FT flows."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .appointment_policy import (
    appointment_clarify_question,
    appointment_confirmation_text,
    appointment_handoff_text,
    appointment_missing_slots,
    build_appointment_state,
)
from .candidate_policy import candidate_rejected_question
from .contracts import DialogState
from .dialog_state import copy_dialog_state, filter_missing_slots_by_entities
from .memory_policy import flow_scoped_memory_keys, flow_scoped_meta_keys
from .medical_pretool_policy import rejected_candidate_slots
from .routing_contract import clarify_question_for_slots, clarify_type_for_slots
from .signal_parsers import (
    extract_expected_slot_entities,
    extract_result_lookup_fields,
    looks_like_no_preference_answer,
    looks_like_uncertainty_answer,
    split_mixed_utterance,
)
from .tool_planning import is_about_agent_query, select_tool_plan, should_use_web_search


_RESULT_REQUIRED_SLOTS: tuple[str, ...] = (
    "result_surname",
    "result_year_of_birth",
    "result_analysis_code",
    "result_analysis_number",
)


@dataclass(slots=True)
class FlowLocalPrecheckResult:
    handled: bool = False
    reply_text: str = ""
    next_state: DialogState | None = None
    clear_state: bool = False
    clear_memory_keys: tuple[str, ...] = ()
    clear_meta_keys: tuple[str, ...] = ()
    handoff: bool = False
    save_memory_entities: dict[str, Any] = field(default_factory=dict)
    reprocess_current_message: bool = False
    pending_topic_switch_message: str = ""
    pending_continue_message: str = ""


_TOPIC_QUESTION_RE = re.compile(
    r"\b(кто|что|где|когда|сколько|как|почему|зачем|расскажи|покажи|дай|найди|поищи|объясни|подскажи)\b",
    re.I,
)


def apply_flow_local_precheck(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    state = copy_dialog_state(dialog_state)
    if not _has_active_interruptible_flow(state):
        return FlowLocalPrecheckResult()

    mixed = _apply_mixed_utterance_flow_local(
        user_message=user_message,
        dialog_state=state,
        memory_entities=memory_entities,
    )
    if mixed.handled:
        return mixed

    flow_kind = str(state.flow_kind or "").strip().lower()
    if flow_kind == "appointment":
        result = _apply_appointment_flow_local(
            user_message=user_message,
            dialog_state=state,
            memory_entities=memory_entities,
        )
        if result.handled:
            return result
    if flow_kind in {"clarify", "confirmation"}:
        result = _apply_generic_flow_local(
            user_message=user_message,
            dialog_state=state,
        )
        if result.handled:
            return result
    if flow_kind == "result_lookup":
        result = _apply_result_lookup_flow_local(
            user_message=user_message,
            dialog_state=state,
            memory_entities=memory_entities,
        )
        if result.handled:
            return result

    if looks_like_uncertainty_answer(user_message):
        return _build_resume_reply(
            dialog_state=state,
            prefix=_uncertainty_prefix(state),
        )
    if looks_like_no_preference_answer(user_message):
        return _build_resume_reply(
            dialog_state=state,
            prefix=_no_preference_prefix(state),
        )
    return FlowLocalPrecheckResult()


def _apply_mixed_utterance_flow_local(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    flow_part, switch_part = split_mixed_utterance(user_message)
    if not flow_part or not switch_part:
        return FlowLocalPrecheckResult()
    if not _looks_like_explicit_switch_part(switch_part):
        return FlowLocalPrecheckResult()

    expected_slots = list(dialog_state.expected_slots or dialog_state.missing_slots or [])
    flow_entities = extract_expected_slot_entities(flow_part, expected_slots=expected_slots)
    if not flow_entities:
        return FlowLocalPrecheckResult()

    flow_kind = str(dialog_state.flow_kind or "").strip().lower()
    if flow_kind == "appointment":
        flow_result = _apply_appointment_slot_entities(
            dialog_state=dialog_state,
            memory_entities=memory_entities,
            flow_entities=flow_entities,
        )
    elif flow_kind == "clarify":
        flow_result = _apply_clarify_slot_entities(
            dialog_state=dialog_state,
            memory_entities=memory_entities,
            flow_entities=flow_entities,
        )
    elif flow_kind == "result_lookup":
        flow_result = _apply_result_lookup_slot_entities(
            dialog_state=dialog_state,
            memory_entities=memory_entities,
            flow_entities=flow_entities,
        )
    else:
        return FlowLocalPrecheckResult()

    if not _has_meaningful_flow_update(flow_result, dialog_state):
        return FlowLocalPrecheckResult()
    if flow_result.handoff or flow_result.clear_state:
        return FlowLocalPrecheckResult()

    previous_state = flow_result.next_state if isinstance(flow_result.next_state, DialogState) else copy_dialog_state(dialog_state)
    return FlowLocalPrecheckResult(
        handled=True,
        next_state=previous_state,
        save_memory_entities=dict(flow_result.save_memory_entities or {}),
        pending_topic_switch_message=switch_part,
        pending_continue_message=flow_part if flow_result.reprocess_current_message else "",
    )


def _apply_appointment_flow_local(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    if looks_like_uncertainty_answer(user_message):
        return _appointment_non_answer(dialog_state, kind="uncertainty")
    if not looks_like_no_preference_answer(user_message):
        return FlowLocalPrecheckResult()

    entities = _merged_appointment_entities(dialog_state, memory_entities)
    auto_entities = _autofill_appointment_no_preference(
        base_entities=entities,
        expected_slots=list(dialog_state.expected_slots or dialog_state.missing_slots or []),
    )
    if not auto_entities:
        return _appointment_non_answer(dialog_state, kind=_appointment_no_preference_kind(dialog_state))

    entities.update(auto_entities)
    missing_slots = appointment_missing_slots(entities)
    if missing_slots:
        clarify_text = appointment_clarify_question(missing_slots, entities)
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text=clarify_text,
            next_state=build_appointment_state(
                entities=entities,
                missing_slots=missing_slots,
                phase="appointment_collecting",
                open_question=clarify_text,
            ),
            save_memory_entities=entities,
        )

    confirm_text = appointment_confirmation_text(entities)
    return FlowLocalPrecheckResult(
        handled=True,
        reply_text=confirm_text,
        next_state=build_appointment_state(
            entities=entities,
            missing_slots=[],
            phase="appointment_confirm",
            open_question=confirm_text,
        ),
        save_memory_entities=entities,
    )


def _apply_appointment_slot_entities(
    *,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
    flow_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    entities = _merged_appointment_entities(dialog_state, memory_entities)
    entities.update(flow_entities)
    missing_slots = appointment_missing_slots(entities)
    if missing_slots:
        clarify_text = appointment_clarify_question(missing_slots, entities)
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text=clarify_text,
            next_state=build_appointment_state(
                entities=entities,
                missing_slots=missing_slots,
                phase="appointment_collecting",
                open_question=clarify_text,
            ),
            save_memory_entities=entities,
        )

    confirm_text = appointment_confirmation_text(entities)
    return FlowLocalPrecheckResult(
        handled=True,
        reply_text=confirm_text,
        next_state=build_appointment_state(
            entities=entities,
            missing_slots=[],
            phase="appointment_confirm",
            open_question=confirm_text,
        ),
        save_memory_entities=entities,
    )


def _apply_result_lookup_flow_local(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    if looks_like_uncertainty_answer(user_message) or looks_like_no_preference_answer(user_message):
        return _result_lookup_non_answer(dialog_state)

    expected_slots = _result_expected_slots(dialog_state)
    parsed = extract_result_lookup_fields(user_message, expected_slots=expected_slots)
    if not parsed:
        return FlowLocalPrecheckResult()

    return _apply_result_lookup_slot_entities(
        dialog_state=dialog_state,
        memory_entities=memory_entities,
        flow_entities=parsed,
    )


def _apply_result_lookup_slot_entities(
    *,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
    flow_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    expected_slots = _result_expected_slots(dialog_state)
    parsed = dict(flow_entities or {})
    if not parsed:
        return FlowLocalPrecheckResult()

    entities = _merged_flow_entities(dialog_state, memory_entities)
    entities.update(parsed)
    missing_slots = filter_missing_slots_by_entities(expected_slots, entities)
    if missing_slots:
        clarify_text = clarify_question_for_slots("test_result", missing_slots)
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text=clarify_text,
            next_state=_build_result_lookup_state(
                dialog_state=dialog_state,
                entities=entities,
                missing_slots=missing_slots,
                open_question=clarify_text,
            ),
            save_memory_entities=entities,
        )

    return FlowLocalPrecheckResult(
        handled=True,
        next_state=_build_result_lookup_state(
            dialog_state=dialog_state,
            entities=entities,
            missing_slots=[],
            open_question="",
        ),
        save_memory_entities=entities,
        reprocess_current_message=True,
    )


def _apply_clarify_slot_entities(
    *,
    dialog_state: DialogState,
    memory_entities: dict[str, Any],
    flow_entities: dict[str, Any],
) -> FlowLocalPrecheckResult:
    parsed = dict(flow_entities or {})
    if not parsed:
        return FlowLocalPrecheckResult()
    entities = _merged_flow_entities(dialog_state, memory_entities)
    entities.update(parsed)
    current_slots = [str(slot).strip() for slot in (dialog_state.expected_slots or dialog_state.missing_slots or []) if str(slot).strip()]
    missing_slots = filter_missing_slots_by_entities(current_slots, entities)
    if missing_slots:
        clarify_text = clarify_question_for_slots(str(dialog_state.intent or "").strip(), missing_slots)
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text=clarify_text,
            next_state=_build_generic_clarify_state(
                dialog_state=dialog_state,
                entities=entities,
                missing_slots=missing_slots,
                open_question=clarify_text,
            ),
            save_memory_entities=entities,
        )
    return FlowLocalPrecheckResult(
        handled=True,
        next_state=_build_generic_clarify_state(
            dialog_state=dialog_state,
            entities=entities,
            missing_slots=[],
            open_question="",
        ),
        save_memory_entities=entities,
        reprocess_current_message=True,
    )


def _apply_generic_flow_local(
    *,
    user_message: str,
    dialog_state: DialogState,
) -> FlowLocalPrecheckResult:
    if looks_like_uncertainty_answer(user_message):
        return _generic_non_answer(dialog_state, kind="uncertainty")
    if looks_like_no_preference_answer(user_message):
        if str(dialog_state.flow_kind or "").strip().lower() == "confirmation":
            confirmation_result = _apply_confirmation_no_preference(dialog_state)
            if confirmation_result.handled:
                return confirmation_result
        return _generic_non_answer(dialog_state, kind="no_preference")
    return FlowLocalPrecheckResult()


def _has_active_interruptible_flow(dialog_state: DialogState) -> bool:
    return bool(
        (
            dialog_state.flow_active
            or str(dialog_state.flow_kind or "").strip()
            or dialog_state.expected_slots
            or str(dialog_state.flow_stage or "").strip()
        )
        and dialog_state.flow_interruptible
    )


def _looks_like_explicit_switch_part(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    if is_about_agent_query(probe):
        return True
    if should_use_web_search(probe, allow_for_medical=True):
        return True
    if select_tool_plan(probe, include_meili_tools=True):
        return True
    return bool(_TOPIC_QUESTION_RE.search(probe) and len(probe.split()) >= 2)


def _has_meaningful_flow_update(result: FlowLocalPrecheckResult, dialog_state: DialogState) -> bool:
    if result.reprocess_current_message or result.handoff or result.clear_state:
        return True
    if result.save_memory_entities:
        return True
    next_state = result.next_state if isinstance(result.next_state, DialogState) else None
    if next_state is None:
        return False
    current = copy_dialog_state(dialog_state)
    return bool(
        dict(next_state.entities or {}) != dict(current.entities or {})
        or list(next_state.missing_slots or []) != list(current.missing_slots or [])
        or list(next_state.expected_slots or []) != list(current.expected_slots or [])
        or str(next_state.phase or "") != str(current.phase or "")
        or str(next_state.open_question or "") != str(current.open_question or "")
    )


def _build_resume_reply(*, dialog_state: DialogState, prefix: str) -> FlowLocalPrecheckResult:
    state = copy_dialog_state(dialog_state)
    state.flow_non_answer_count = 0
    state.flow_non_answer_kind = ""
    resume = _resume_question(state)
    text = str(prefix or "").strip()
    if resume and resume not in text:
        text = f"{text} {resume}".strip()
    return FlowLocalPrecheckResult(
        handled=True,
        reply_text=text,
        next_state=state,
    )


def _resume_question(dialog_state: DialogState) -> str:
    return str(dialog_state.flow_resume_question or dialog_state.open_question or "").strip()


def _uncertainty_prefix(dialog_state: DialogState) -> str:
    expected = {str(slot or "").strip().lower() for slot in (dialog_state.expected_slots or dialog_state.missing_slots or [])}
    if str(dialog_state.flow_kind or "").strip().lower() == "appointment" and "patient_name" in expected:
        return "Для записи нужно ваше ФИО."
    if str(dialog_state.flow_kind or "").strip().lower() == "result_lookup":
        return "Для проверки результата нужны точные данные."
    return "Нужна более точная информация, чтобы продолжить."


def _no_preference_prefix(dialog_state: DialogState) -> str:
    if str(dialog_state.flow_kind or "").strip().lower() == "appointment":
        return "Чтобы продолжить запись, нужно выбрать подходящий вариант."
    if str(dialog_state.flow_kind or "").strip().lower() == "result_lookup":
        return "Для проверки результата нельзя использовать произвольные данные."
    return "Чтобы продолжить, нужна более конкретная информация."


def _merged_flow_entities(dialog_state: DialogState, memory_entities: dict[str, Any]) -> dict[str, Any]:
    merged = dict(memory_entities or {})
    merged.update(dict(dialog_state.entities or {}))
    return merged


def _merged_appointment_entities(dialog_state: DialogState, memory_entities: dict[str, Any]) -> dict[str, Any]:
    merged = dict(dialog_state.entities or {})
    for key in (
        "appointment_action",
        "doctor_name",
        "doctor_id",
        "specialty",
        "service_name",
        "branch_name",
        "city",
        "appointment_windows",
        "appointment_branch_options",
    ):
        if str(merged.get(key) or "").strip():
            continue
        value = memory_entities.get(key)
        if isinstance(value, list):
            if value:
                merged[key] = value
            continue
        if str(value or "").strip():
            merged[key] = value
    return merged


def _autofill_appointment_no_preference(
    *,
    base_entities: dict[str, Any],
    expected_slots: list[str],
) -> dict[str, Any]:
    expected = {str(slot or "").strip().lower() for slot in expected_slots if str(slot or "").strip()}
    updates: dict[str, Any] = {}
    windows = base_entities.get("appointment_windows")
    rows = _normalized_appointment_windows(windows, branch_name=str(base_entities.get("branch_name") or "").strip())
    if rows and expected & {"branch_or_city", "date", "time"}:
        first = rows[0]
        branch_name = str(first.get("branch_name") or "").strip()
        date_value = str(first.get("date") or "").strip()
        time_value = str(first.get("time") or "").strip()
        if branch_name:
            updates["branch_name"] = branch_name
        if date_value:
            updates["date"] = date_value
            updates["date_from"] = date_value
            updates["date_to"] = date_value
        if time_value:
            updates["time"] = time_value[:5]
            updates["time_from"] = time_value[:5]
        return updates

    branch_options = base_entities.get("appointment_branch_options")
    if "branch_or_city" in expected and isinstance(branch_options, list):
        for value in branch_options:
            branch_name = str(value or "").strip()
            if branch_name:
                updates["branch_name"] = branch_name
                return updates
    return updates


def _normalized_appointment_windows(windows: Any, *, branch_name: str) -> list[dict[str, str]]:
    if not isinstance(windows, list):
        return []
    rows: list[dict[str, str]] = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        date_value = str(item.get("date") or "").strip()
        time_value = str(item.get("time") or "").strip()
        row_branch = str(item.get("branch_name") or item.get("branch") or "").strip()
        if not (date_value and time_value):
            continue
        rows.append(
            {
                "date": date_value,
                "time": time_value,
                "branch_name": row_branch,
            }
        )
    if branch_name:
        filtered = [row for row in rows if row.get("branch_name") == branch_name]
        if filtered:
            rows = filtered
    rows.sort(key=lambda row: (row.get("date", ""), row.get("time", ""), row.get("branch_name", "")))
    return rows


def _result_expected_slots(dialog_state: DialogState) -> list[str]:
    slots = [str(slot).strip() for slot in (dialog_state.expected_slots or dialog_state.missing_slots or []) if str(slot).strip()]
    if slots:
        return slots
    return list(_RESULT_REQUIRED_SLOTS)


def _build_result_lookup_state(
    *,
    dialog_state: DialogState,
    entities: dict[str, Any],
    missing_slots: list[str],
    open_question: str,
) -> DialogState:
    state = copy_dialog_state(dialog_state)
    state.route = "clinical"
    state.intent = "test_result"
    state.entities = dict(entities or {})
    state.missing_slots = list(missing_slots or [])
    state.expected_slots = list(missing_slots or [])
    state.clarify_type = clarify_type_for_slots("test_result", missing_slots)
    state.phase = "collecting" if missing_slots else ""
    state.open_question = str(open_question or "").strip()
    state.flow_active = bool(missing_slots)
    state.flow_kind = "result_lookup" if missing_slots else ""
    state.flow_stage = "collecting" if missing_slots else ""
    state.flow_interruptible = bool(missing_slots)
    state.flow_resume_question = str(open_question or "").strip()
    state.flow_non_answer_count = 0
    state.flow_non_answer_kind = ""
    return state


def _build_generic_clarify_state(
    *,
    dialog_state: DialogState,
    entities: dict[str, Any],
    missing_slots: list[str],
    open_question: str,
) -> DialogState:
    state = copy_dialog_state(dialog_state)
    state.entities = dict(entities or {})
    state.missing_slots = list(missing_slots or [])
    state.expected_slots = list(missing_slots or [])
    state.clarify_type = (
        clarify_type_for_slots(str(dialog_state.intent or "").strip(), missing_slots) if missing_slots else ""
    )
    state.phase = "collecting" if missing_slots else ""
    state.open_question = str(open_question or "").strip()
    state.flow_active = bool(missing_slots)
    state.flow_kind = "clarify" if missing_slots else ""
    state.flow_stage = "collecting" if missing_slots else ""
    state.flow_interruptible = bool(missing_slots)
    state.flow_resume_question = str(open_question or "").strip()
    state.flow_non_answer_count = 0
    state.flow_non_answer_kind = ""
    return state


def _appointment_non_answer(dialog_state: DialogState, *, kind: str) -> FlowLocalPrecheckResult:
    state = _bump_non_answer_state(dialog_state, kind=kind)
    count = int(state.flow_non_answer_count or 0)
    if count >= 3:
        entities = _merged_flow_entities(dialog_state, {})
        action = str(entities.get("appointment_action") or "").strip().lower()
        if action not in {"book", "cancel", "reschedule"}:
            entities["appointment_action"] = "book"
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text=_appointment_non_answer_handoff_text(state),
            handoff=True,
        )
    if count == 2:
        return _resume_with_state(
            state,
            prefix="Без этих данных я не смогу завершить запись автоматически. Если вы не сможете их указать, я передам диалог оператору.",
        )
    return _resume_with_state(
        state,
        prefix=_uncertainty_prefix(dialog_state) if kind == "uncertainty" else _no_preference_prefix(dialog_state),
    )


def _result_lookup_non_answer(dialog_state: DialogState) -> FlowLocalPrecheckResult:
    state = _bump_non_answer_state(dialog_state, kind="uncertainty")
    count = int(state.flow_non_answer_count or 0)
    if count >= 3:
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text="Без точных данных я не смогу проверить результат автоматически. Текущий сценарий остановлен. Можете задать другой вопрос.",
            clear_state=True,
            clear_memory_keys=flow_scoped_memory_keys(),
            clear_meta_keys=flow_scoped_meta_keys(),
        )
    if count == 2:
        return _resume_with_state(
            state,
            prefix="Без фамилии, года рождения, кода и номера анализа я не смогу проверить результат автоматически.",
        )
    return _resume_with_state(
        state,
        prefix="Для проверки результата нужны точные данные.",
    )


def _generic_non_answer(dialog_state: DialogState, *, kind: str) -> FlowLocalPrecheckResult:
    state = _bump_non_answer_state(dialog_state, kind=kind)
    count = int(state.flow_non_answer_count or 0)
    if count >= 3:
        return FlowLocalPrecheckResult(
            handled=True,
            reply_text="Останавливаю текущий сценарий. Можете задать вопрос по-другому или перейти к другой теме.",
            clear_state=True,
            clear_memory_keys=flow_scoped_memory_keys(),
            clear_meta_keys=flow_scoped_meta_keys(),
        )
    if count == 2:
        if str(dialog_state.flow_kind or "").strip().lower() == "confirmation":
            return _resume_with_state(
                state,
                prefix="Если сейчас не готовы подтвердить решение, можем остановить текущий сценарий. Пока нужен однозначный ответ.",
            )
        return _resume_with_state(
            state,
            prefix="Если сейчас неудобно отвечать на уточнение, можно задать вопрос иначе или перейти к другой теме.",
        )
    if str(dialog_state.flow_kind or "").strip().lower() == "confirmation":
        return _resume_with_state(
            state,
            prefix="Нужен однозначный ответ." if kind == "uncertainty" else "Нужно выбрать конкретный вариант.",
        )
    return _resume_with_state(
        state,
        prefix=_uncertainty_prefix(dialog_state) if kind == "uncertainty" else _no_preference_prefix(dialog_state),
    )


def _apply_confirmation_no_preference(dialog_state: DialogState) -> FlowLocalPrecheckResult:
    target = str(dialog_state.confirmation_target or "").strip()
    if not target:
        return FlowLocalPrecheckResult()
    state = copy_dialog_state(dialog_state)
    clarify_slots = rejected_candidate_slots(target)
    clarify_text = candidate_rejected_question(target)
    state.candidate_entities = dict(state.candidate_entities or {})
    state.candidate_entities.pop(target, None)
    state.confirmation_target = ""
    state.missing_slots = list(clarify_slots or [])
    state.expected_slots = list(clarify_slots or [])
    state.clarify_type = "identify"
    state.phase = "collecting"
    state.open_question = clarify_text
    state.flow_active = True
    state.flow_kind = "clarify"
    state.flow_stage = "collecting"
    state.flow_interruptible = True
    state.flow_resume_question = clarify_text
    state.flow_non_answer_count = 0
    state.flow_non_answer_kind = ""
    return FlowLocalPrecheckResult(
        handled=True,
        reply_text=clarify_text,
        next_state=state,
    )


def _resume_with_state(dialog_state: DialogState, *, prefix: str) -> FlowLocalPrecheckResult:
    state = copy_dialog_state(dialog_state)
    resume = _resume_question(state)
    text = str(prefix or "").strip()
    if resume and resume not in text:
        text = f"{text} {resume}".strip()
    return FlowLocalPrecheckResult(
        handled=True,
        reply_text=text,
        next_state=state,
    )


def _bump_non_answer_state(dialog_state: DialogState, *, kind: str) -> DialogState:
    state = copy_dialog_state(dialog_state)
    current_kind = str(state.flow_non_answer_kind or "").strip().lower()
    current_count = max(0, int(state.flow_non_answer_count or 0))
    next_kind = str(kind or "").strip().lower() or "uncertainty"
    if current_kind == next_kind:
        state.flow_non_answer_count = current_count + 1
    else:
        state.flow_non_answer_count = 1
    state.flow_non_answer_kind = next_kind
    return state


def _appointment_no_preference_kind(dialog_state: DialogState) -> str:
    expected = {str(slot or "").strip().lower() for slot in (dialog_state.expected_slots or dialog_state.missing_slots or [])}
    if "patient_name" in expected:
        return "uncertainty"
    return "no_preference"


def _appointment_non_answer_handoff_text(dialog_state: DialogState) -> str:
    flow_stage = str(dialog_state.flow_stage or "").strip().lower()
    if flow_stage == "confirm":
        return appointment_handoff_text(dict(dialog_state.entities or {}))
    return "Не удалось получить данные для записи автоматически. Передаю диалог оператору."
