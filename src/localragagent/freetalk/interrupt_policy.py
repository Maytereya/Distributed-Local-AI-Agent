"""Global user interrupt/reset policy for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .contracts import DialogState
from .candidate_policy import parse_candidate_confirmation_message
from .dialog_state import (
    copy_dialog_state,
    dialog_state_from_payload,
    dialog_state_is_active,
    dialog_state_payload,
)
from .memory_policy import flow_scoped_memory_keys, flow_scoped_meta_keys
from .signal_parsers import (
    extract_branch_reference,
    extract_city_reference,
    extract_date_filters,
    extract_doctor_reference_candidate,
    extract_expected_slot_entities,
    extract_person_name,
    extract_result_lookup_fields,
    extract_service_reference_candidate,
    extract_service_variant,
    extract_time_filters,
    looks_like_no_preference_answer,
    looks_like_slot_correction,
    looks_like_uncertainty_answer,
    parse_yes_no,
    split_mixed_utterance,
)
from .tool_planning import is_about_agent_query, select_tool_plan, should_use_web_search


_HARD_RESET_RE = re.compile(
    r"\b("
    r"reset|"
    r"сброс(?:ь|ить)?|"
    r"очист(?:и|ить)\s+(?:диалог|память)|"
    r"забудь\s+вс[её]|"
    r"нов(?:ый|ую)\s+диалог|"
    r"начн[её]м\s+нов(?:ый|ую)\s+диалог"
    r")\b",
    re.I,
)
_INTERRUPT_RE = re.compile(
    r"\b("
    r"стоп|"
    r"останов(?:ись|ить|ись\!)|"
    r"прекрат\w*|"
    r"подожди(?:те)?|"
    r"давай\s+сначала|"
    r"начн[её]м\s+сначала|"
    r"не\s+то|"
    r"неправильн\w*"
    r")\b",
    re.I,
)
_NEGATIVE_FEEDBACK_RE = re.compile(
    r"\b("
    r"бред\w*|"
    r"что\s+ты\s+нес(?:е|ё)ш\w*|"
    r"ты\s+не\s+то\s+пиш\w*|"
    r"неправильн\w*\s+ответ"
    r")\b",
    re.I,
)
FLOW_INTERRUPT_CONFIRM_TEXT = "Прекратить текущий сценарий? Ответьте: да или нет."
FLOW_INTERRUPTED_TEXT = "Остановил текущий сценарий. Можем начать заново или перейти к другому вопросу."
TOPIC_SWITCH_CONFIRM_TEXT = "Прервать текущий сценарий и перейти к новому вопросу? Ответьте: да или нет."
SESSION_RESET_CONFIRM_TEXT = "Очистить весь диалог и начать новую сессию? Ответьте: да или нет."
SESSION_RESET_TEXT = "Диалог очищен. Начинаем новую сессию."
NO_ACTIVE_FLOW_TEXT = "Сейчас нет активного сценария. Можете задать новый вопрос."
CONTINUE_DIALOG_TEXT = "Хорошо, продолжаем текущий диалог."
INTERRUPT_ARBITER_DECISIONS: tuple[str, ...] = (
    "continue",
    "correct",
    "switch",
    "interrupt",
    "hard_reset",
    "unknown",
)

_PREV_STATE_KEY = "_ft_interrupt_previous_state"
_RESUME_QUESTION_KEY = "_ft_interrupt_resume_question"
_PENDING_USER_MESSAGE_KEY = "_ft_interrupt_pending_user_message"
_PENDING_CONTINUE_MESSAGE_KEY = "_ft_interrupt_pending_continue_message"
_YEAR_RE = re.compile(r"^\s*(19|20)\d{2}\s*[.!?]?\s*$")
_RESULT_CODE_RE = re.compile(r"^\s*[A-Za-zА-Яа-яЁё]{1,4}\s*[.!?]?\s*$")
_RESULT_NUMBER_RE = re.compile(r"^\s*\d{2,12}\s*[.!?]?\s*$")
_TOPIC_QUESTION_RE = re.compile(
    r"\b(кто|что|где|когда|сколько|как|почему|зачем|расскажи|покажи|дай|найди|поищи|объясни|подскажи)\b",
    re.I,
)
_SCHEDULE_PREVIEW_RE = re.compile(
    r"\b(расписан\w*|график|свободн\w*\s+(?:окн\w*|слот\w*)|слот\w*|окн\w*)\b",
    re.I,
)
_LEADING_YES_RE = re.compile(r"^\s*(да|yes|y)\b", re.I)
_LEADING_NO_RE = re.compile(r"^\s*(нет|не|no|n)\b", re.I)


@dataclass(slots=True)
class InterruptPrecheckResult:
    handled: bool = False
    reply_text: str = ""
    next_state: DialogState | None = None
    clear_state: bool = False
    clear_memory_keys: tuple[str, ...] = ()
    clear_meta_keys: tuple[str, ...] = ()
    hard_reset: bool = False
    next_session_id: str = ""
    reentry_message: str = ""
    arbiter_needed: bool = False


def detect_interrupt_intent(text: str) -> str:
    probe = str(text or "").strip()
    if not probe:
        return "none"
    if looks_like_slot_correction(probe):
        return "none"
    if _HARD_RESET_RE.search(probe):
        return "hard_reset"
    if _INTERRUPT_RE.search(probe):
        return "interrupt"
    if _NEGATIVE_FEEDBACK_RE.search(probe):
        return "feedback"
    return "none"


def apply_interrupt_precheck(
    *,
    user_message: str,
    dialog_state: DialogState,
    memory_entities: dict[str, Any] | None = None,
) -> InterruptPrecheckResult:
    state = copy_dialog_state(dialog_state)
    phase = str(state.phase or "").strip().lower()
    intent = detect_interrupt_intent(user_message)

    if phase in {"interrupt_confirm_flow", "interrupt_confirm_session", "interrupt_confirm_topic_switch"}:
        return _apply_interrupt_confirmation(user_message=user_message, dialog_state=state)

    active_flow = _has_interruptible_flow(state)
    if intent == "hard_reset":
        return InterruptPrecheckResult(
            handled=True,
            reply_text=SESSION_RESET_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="session",
                previous_state=state,
                resume_question=_resume_question(state),
            ),
        )
    if intent == "interrupt":
        if not active_flow:
            return InterruptPrecheckResult(
                handled=True,
                reply_text=NO_ACTIVE_FLOW_TEXT,
            )
        return InterruptPrecheckResult(
            handled=True,
            reply_text=FLOW_INTERRUPT_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="flow",
                previous_state=state,
                resume_question=_resume_question(state),
            ),
        )
    if intent == "feedback" and active_flow:
        return InterruptPrecheckResult(
            handled=True,
            reply_text=FLOW_INTERRUPT_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="flow",
                previous_state=state,
                resume_question=_resume_question(state),
            ),
        )
    if active_flow and _looks_like_topic_switch_candidate(user_message=user_message, dialog_state=state):
        return InterruptPrecheckResult(
            handled=True,
            reply_text=TOPIC_SWITCH_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="topic_switch",
                previous_state=state,
                resume_question=_resume_question(state),
                pending_user_message=user_message,
            ),
        )
    if active_flow and _should_request_interrupt_arbiter(user_message=user_message, dialog_state=state):
        return InterruptPrecheckResult(arbiter_needed=True)
    _ = memory_entities
    return InterruptPrecheckResult()


def apply_interrupt_arbiter_decision(
    *,
    decision: str,
    user_message: str,
    dialog_state: DialogState,
) -> InterruptPrecheckResult:
    normalized = str(decision or "").strip().lower()
    state = copy_dialog_state(dialog_state)
    if normalized in {"continue", "correct", "unknown"}:
        return InterruptPrecheckResult()
    if normalized == "switch" and _has_interruptible_flow(state):
        return InterruptPrecheckResult(
            handled=True,
            reply_text=TOPIC_SWITCH_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="topic_switch",
                previous_state=state,
                resume_question=_resume_question(state),
                pending_user_message=user_message,
            ),
        )
    if normalized == "interrupt":
        if not _has_interruptible_flow(state):
            return InterruptPrecheckResult(
                handled=True,
                reply_text=NO_ACTIVE_FLOW_TEXT,
            )
        return InterruptPrecheckResult(
            handled=True,
            reply_text=FLOW_INTERRUPT_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="flow",
                previous_state=state,
                resume_question=_resume_question(state),
            ),
        )
    if normalized == "hard_reset":
        return InterruptPrecheckResult(
            handled=True,
            reply_text=SESSION_RESET_CONFIRM_TEXT,
            next_state=_build_interrupt_state(
                scope="session",
                previous_state=state,
                resume_question=_resume_question(state),
            ),
        )
    return InterruptPrecheckResult()


def parse_interrupt_arbiter_payload(payload: dict[str, Any] | None) -> str:
    data = payload if isinstance(payload, dict) else {}
    decision = str(data.get("decision") or "").strip().lower()
    if decision in INTERRUPT_ARBITER_DECISIONS:
        return decision
    return "unknown"


def _apply_interrupt_confirmation(
    *,
    user_message: str,
    dialog_state: DialogState,
) -> InterruptPrecheckResult:
    transition = _yes_no_transition(user_message)
    phase = str(dialog_state.phase or "").strip().lower()
    previous_state = _previous_state(dialog_state)
    resume_text = _resume_text(dialog_state, previous_state)

    if phase == "interrupt_confirm_flow":
        if transition == "yes":
            return InterruptPrecheckResult(
                handled=True,
                reply_text=FLOW_INTERRUPTED_TEXT,
                clear_state=True,
                clear_memory_keys=flow_scoped_memory_keys(),
                clear_meta_keys=flow_scoped_meta_keys(),
            )
        if transition == "no":
            if dialog_state_is_active(previous_state):
                return InterruptPrecheckResult(
                    handled=True,
                    reply_text=resume_text,
                    next_state=previous_state,
                )
            return InterruptPrecheckResult(
                handled=True,
                reply_text=CONTINUE_DIALOG_TEXT,
                clear_state=True,
            )
        return InterruptPrecheckResult(
            handled=True,
            reply_text=FLOW_INTERRUPT_CONFIRM_TEXT,
            next_state=dialog_state,
        )

    if phase == "interrupt_confirm_topic_switch":
        continue_message = _pending_continue_message(dialog_state)
        if transition == "yes":
            return InterruptPrecheckResult(
                handled=True,
                reply_text="",
                clear_state=True,
                clear_memory_keys=flow_scoped_memory_keys(),
                clear_meta_keys=flow_scoped_meta_keys(),
                reentry_message=_pending_user_message(dialog_state),
            )
        if transition == "no":
            if continue_message:
                return InterruptPrecheckResult(
                    handled=True,
                    reply_text="",
                    next_state=previous_state,
                    reentry_message=continue_message,
                )
            if dialog_state_is_active(previous_state):
                return InterruptPrecheckResult(
                    handled=True,
                    reply_text=resume_text,
                    next_state=previous_state,
                )
            return InterruptPrecheckResult(
                handled=True,
                reply_text=CONTINUE_DIALOG_TEXT,
                clear_state=True,
            )
        return InterruptPrecheckResult(
            handled=True,
            reply_text=TOPIC_SWITCH_CONFIRM_TEXT,
            next_state=dialog_state,
        )

    if transition == "yes":
        return InterruptPrecheckResult(
            handled=True,
            reply_text=SESSION_RESET_TEXT,
            clear_state=True,
            hard_reset=True,
        )
    if transition == "no":
        if dialog_state_is_active(previous_state):
            return InterruptPrecheckResult(
                handled=True,
                reply_text=resume_text,
                next_state=previous_state,
            )
        return InterruptPrecheckResult(
            handled=True,
            reply_text=CONTINUE_DIALOG_TEXT,
            clear_state=True,
        )
    return InterruptPrecheckResult(
        handled=True,
        reply_text=SESSION_RESET_CONFIRM_TEXT,
        next_state=dialog_state,
    )


def _build_interrupt_state(
    *,
    scope: str,
    previous_state: DialogState,
    resume_question: str,
    pending_user_message: str = "",
    continue_message: str = "",
) -> DialogState:
    previous_payload = dialog_state_payload(previous_state)
    if scope == "flow":
        question = FLOW_INTERRUPT_CONFIRM_TEXT
        phase = "interrupt_confirm_flow"
    elif scope == "topic_switch":
        question = TOPIC_SWITCH_CONFIRM_TEXT
        phase = "interrupt_confirm_topic_switch"
    else:
        question = SESSION_RESET_CONFIRM_TEXT
        phase = "interrupt_confirm_session"
    entities: dict[str, Any] = {
        _PREV_STATE_KEY: previous_payload,
        _RESUME_QUESTION_KEY: str(resume_question or "").strip(),
    }
    if str(pending_user_message or "").strip():
        entities[_PENDING_USER_MESSAGE_KEY] = str(pending_user_message or "").strip()
    if str(continue_message or "").strip():
        entities[_PENDING_CONTINUE_MESSAGE_KEY] = str(continue_message or "").strip()
    return DialogState(
        route="general",
        intent="interrupt",
        entities=entities,
        candidate_entities={},
        confirmation_target="",
        missing_slots=[],
        clarify_type="other",
        tool_plan=[],
        response_policy="general_only",
        confidence=1.0,
        clarify_count=1,
        last_tool="",
        phase=phase,
        open_question=question,
        flow_active=True,
        flow_kind="interrupt",
        flow_stage="confirm",
        flow_interruptible=False,
        flow_resume_question=question,
    )


def _previous_state(dialog_state: DialogState) -> DialogState:
    entities = dialog_state.entities if isinstance(dialog_state.entities, dict) else {}
    payload = entities.get(_PREV_STATE_KEY) if isinstance(entities, dict) else None
    if isinstance(payload, dict):
        return dialog_state_from_payload(payload)
    return DialogState()


def _resume_text(dialog_state: DialogState, previous_state: DialogState) -> str:
    entities = dialog_state.entities if isinstance(dialog_state.entities, dict) else {}
    saved = str(entities.get(_RESUME_QUESTION_KEY) or "").strip()
    if saved:
        return saved
    return _resume_question(previous_state)


def _resume_question(dialog_state: DialogState) -> str:
    question = str(dialog_state.flow_resume_question or dialog_state.open_question or "").strip()
    if question and question not in {FLOW_INTERRUPT_CONFIRM_TEXT, TOPIC_SWITCH_CONFIRM_TEXT, SESSION_RESET_CONFIRM_TEXT}:
        return question
    return CONTINUE_DIALOG_TEXT


def _yes_no_transition(text: str) -> str:
    return parse_yes_no(text, profile="strict")


def _pending_user_message(dialog_state: DialogState) -> str:
    entities = dialog_state.entities if isinstance(dialog_state.entities, dict) else {}
    return str(entities.get(_PENDING_USER_MESSAGE_KEY) or "").strip()


def _pending_continue_message(dialog_state: DialogState) -> str:
    entities = dialog_state.entities if isinstance(dialog_state.entities, dict) else {}
    return str(entities.get(_PENDING_CONTINUE_MESSAGE_KEY) or "").strip()


def _has_interruptible_flow(dialog_state: DialogState) -> bool:
    descriptor_present = bool(
        str(dialog_state.flow_kind or "").strip()
        or dialog_state.expected_slots
        or str(dialog_state.flow_stage or "").strip()
        or str(dialog_state.flow_resume_question or "").strip()
        or dialog_state.flow_active
    )
    if descriptor_present:
        active = bool(
            dialog_state.flow_active
            or str(dialog_state.flow_kind or "").strip()
            or dialog_state.expected_slots
            or str(dialog_state.flow_stage or "").strip()
        )
        return bool(active and dialog_state.flow_interruptible)
    return dialog_state_is_active(dialog_state)


def _looks_like_topic_switch_candidate(*, user_message: str, dialog_state: DialogState) -> bool:
    probe = str(user_message or "").strip()
    if not probe:
        return False
    if _is_same_flow_continuation(probe, dialog_state):
        return False
    if _looks_like_flow_local_signal(probe, dialog_state):
        return False
    if looks_like_slot_correction(probe):
        return False
    if _looks_like_preview_inside_flow(probe, dialog_state):
        return False
    return _looks_like_new_topic_message(probe)


def _should_request_interrupt_arbiter(*, user_message: str, dialog_state: DialogState) -> bool:
    probe = str(user_message or "").strip()
    if not probe or len(probe) < 3:
        return False
    if _is_same_flow_continuation(probe, dialog_state):
        return False
    if _looks_like_flow_local_signal(probe, dialog_state):
        return False
    if looks_like_slot_correction(probe):
        return False
    if _looks_like_preview_inside_flow(probe, dialog_state):
        return False
    if _looks_like_new_topic_message(probe):
        return False
    return True


def _is_same_flow_continuation(text: str, dialog_state: DialogState) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    if _expects_confirmation(dialog_state) and _looks_like_confirmation_answer(probe):
        return True

    expected = {str(slot or "").strip().lower() for slot in (dialog_state.expected_slots or dialog_state.missing_slots or [])}
    if not expected:
        return False

    if "patient_name" in expected and extract_person_name(probe):
        return True
    if expected & {"date"} and extract_date_filters(probe):
        return True
    if expected & {"time"} and extract_time_filters(probe):
        return True
    if expected & {"branch_or_city"} and (extract_branch_reference(probe) or extract_city_reference(probe)):
        return True
    if expected & {"doctor_name", "result_surname"} and extract_doctor_reference_candidate(probe):
        return True
    if "specialty" in expected and (
        extract_doctor_reference_candidate(probe) or _looks_like_short_slot_phrase(probe)
    ):
        return True
    if expected & {"service_or_analysis_name"}:
        if extract_service_reference_candidate(probe) or extract_service_variant(probe):
            return True
        if _looks_like_short_slot_phrase(probe):
            return True
    if "result_year_of_birth" in expected and _YEAR_RE.match(probe):
        return True
    if "result_analysis_code" in expected and _RESULT_CODE_RE.match(probe):
        return True
    if "result_analysis_number" in expected and _RESULT_NUMBER_RE.match(probe):
        return True
    return False


def _looks_like_flow_local_signal(text: str, dialog_state: DialogState) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    if _expects_confirmation(dialog_state):
        confirmation_target = str(dialog_state.confirmation_target or "").strip()
        if confirmation_target:
            parsed = parse_candidate_confirmation_message(probe, target=confirmation_target)
            if parsed.decision in {"yes", "no"}:
                return True
    if looks_like_uncertainty_answer(probe) or looks_like_no_preference_answer(probe):
        return True
    if str(dialog_state.flow_kind or "").strip().lower() == "result_lookup":
        return bool(extract_result_lookup_fields(probe, expected_slots=_result_slots_for_flow(dialog_state)))
    flow_part, switch_part = split_mixed_utterance(probe)
    if flow_part and switch_part:
        expected_slots = list(dialog_state.expected_slots or dialog_state.missing_slots or [])
        return bool(extract_expected_slot_entities(flow_part, expected_slots=expected_slots))
    return False


def build_topic_switch_confirm_state(
    *,
    previous_state: DialogState,
    pending_user_message: str,
    continue_message: str = "",
) -> DialogState:
    return _build_interrupt_state(
        scope="topic_switch",
        previous_state=previous_state,
        resume_question=_resume_question(previous_state),
        pending_user_message=str(pending_user_message or "").strip(),
        continue_message=str(continue_message or "").strip(),
    )


def _result_slots_for_flow(dialog_state: DialogState) -> list[str]:
    slots = [str(slot).strip() for slot in (dialog_state.expected_slots or dialog_state.missing_slots or []) if str(slot).strip()]
    if slots:
        return slots
    return [
        "result_surname",
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    ]


def _expects_confirmation(dialog_state: DialogState) -> bool:
    phase = str(dialog_state.phase or "").strip().lower()
    return bool(
        str(dialog_state.flow_stage or "").strip().lower() == "confirm"
        or phase in {
            "appointment_confirm",
            "confirm_candidate",
        }
        or str(dialog_state.flow_kind or "").strip().lower() in {"confirmation", "interrupt"}
    )


def _looks_like_preview_inside_flow(text: str, dialog_state: DialogState) -> bool:
    if str(dialog_state.flow_kind or "").strip().lower() != "appointment":
        return False
    return bool(_SCHEDULE_PREVIEW_RE.search(str(text or "").strip()))


def _looks_like_new_topic_message(text: str) -> bool:
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


def _looks_like_short_slot_phrase(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe or "?" in probe or len(probe.split()) > 8:
        return False
    if _TOPIC_QUESTION_RE.search(probe):
        return False
    return True


def _looks_like_confirmation_answer(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe:
        return False
    if parse_yes_no(probe, profile="strict") != "unknown":
        return True
    if _LEADING_YES_RE.match(probe):
        tail = _strip_leading_confirmation_token(probe)
        return not tail or _looks_like_short_slot_phrase(tail)
    if _LEADING_NO_RE.match(probe):
        tail = _strip_leading_confirmation_token(probe)
        if not tail:
            return True
        if looks_like_slot_correction(tail):
            return True
        if extract_doctor_reference_candidate(tail):
            return True
        if extract_service_reference_candidate(tail):
            return True
        if _looks_like_short_slot_phrase(tail):
            return True
    return False


def _strip_leading_confirmation_token(text: str) -> str:
    probe = str(text or "").strip()
    probe = re.sub(r"^\s*(да|yes|y|нет|не|no|n)\s*[,.-]?\s*", "", probe, count=1, flags=re.I)
    return probe.strip()
