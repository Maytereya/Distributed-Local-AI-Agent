"""Guard-политики для активного APPOINTMENT flow.

Выносит из router.py state-machine для confirm/cancel/topic-switch в записи.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .city import match_city
from .flow_policy import looks_like_patient_fio
from .memory import MemoryStore
from .mess_types import ResponseEnvelope, SessionState
from .policies import (
    APPOINTMENT_CONFIRM_NO,
    APPOINTMENT_CONFIRM_YES,
    appointment_confirmation_transition,
    appointment_summary,
    appointment_text_cancel_confirm,
    appointment_text_cancelled,
    appointment_text_confirm_prompt,
    appointment_text_confirmed_handoff,
    appointment_text_reask_confirm,
    appointment_text_reask_datetime,
    appointment_text_topic_switch_confirm,
    clarification_question,
    detect_address_intent,
    detect_doc_request_intent,
    detect_doctor_info_intent,
    detect_news_intent,
    detect_nonbookable_walkin_intent,
    detect_prepare_intent,
    detect_price_intent,
    detect_schedule_intent,
    detect_test_assist_intent,
    detect_test_result_intent,
    has_datetime_signal,
    looks_like_branch_hint,
    missing_slots,
)
from .recovery_policy import contextual_reply_kind

_PATIENT_NAME_FRAGMENT_RE = re.compile(r"^\s*[А-ЯЁа-яё\-]{2,}\s+[А-ЯЁа-яё\-]{1,}\s*$")
_APPOINTMENT_CANCEL_OR_RESTART_RE = re.compile(
    r"\b("
    r"отмен\w*|передумал\w*|не\s+надо|стоп|отбой|сброс\w*|"
    r"заново|сначала|начать\s+заново|нов\w*\s+запис\w*"
    r")\b",
    re.I,
)
_APPOINTMENT_SOFT_PAUSE_RE = re.compile(
    r"\b("
    r"подожд(?:ите|и)\w*|"
    r"пока\s+нет|"
    r"пока\s+не\s+буду|"
    r"извините|"
    r"не\s+то|"
    r"не\s+это|"
    r"я\s+друг(ое|ой)\s+хотел\w*"
    r")\b",
    re.I,
)
_APPOINTMENT_SOFT_PAUSE_EXACT = {"нет", "ладно"}
_APPOINTMENT_SOFT_PAUSE_PUNCT_RE = re.compile(r"[!.,?;:]+")

_APPOINTMENT_RUNTIME_KEYS: tuple[str, ...] = (
    "appointment_flow_active",
    "appointment_confirm_pending",
    "appointment_confirmed",
    "appointment_cancel_pending",
    "appointment_topic_switch_pending",
    "appointment_selection_mode",
    "appointment_windows",
    "appointment_branch_options",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "time_flexible",
    "date_hint",
    "branch_id",
    "branch_name",
    "patient_name",
)
_APPOINTMENT_FULL_CONTEXT_KEYS: tuple[str, ...] = (
    "doctor_id",
    "doctor_name",
    "specialty",
    "service_name",
    "test_name",
)


def reset_appointment_runtime_state(state: SessionState) -> None:
    for key in _APPOINTMENT_RUNTIME_KEYS:
        state.last_entities.pop(key, None)


def clear_appointment_flow_context(state: SessionState, memory: MemoryStore) -> None:
    reset_appointment_runtime_state(state)
    for key in _APPOINTMENT_FULL_CONTEXT_KEYS:
        state.last_entities.pop(key, None)
    memory.clear_pending(state)


def should_keep_appointment_flow_override(user_text: str) -> bool:
    """
    Разрешаем OTHER->APPOINTMENT override только для реплик,
    похожих на продолжение сценария записи.
    """
    text = str(user_text or "").strip()
    if not text:
        return False

    low = text.lower()
    if (
        detect_test_result_intent(low)
        or detect_test_assist_intent(low)
        or detect_prepare_intent(low)
        or detect_price_intent(low)
        or detect_address_intent(low)
        or detect_doc_request_intent(low)
        or detect_nonbookable_walkin_intent(text)
        or detect_schedule_intent(low)
        or detect_doctor_info_intent(low)
    ):
        return False

    reply_kind = contextual_reply_kind(text)
    if reply_kind in {"yes", "no"}:
        return True
    if has_datetime_signal(text):
        return True
    if match_city(text):
        return True
    if looks_like_branch_hint(text):
        return True
    if looks_like_patient_fio(text):
        return True
    if _PATIENT_NAME_FRAGMENT_RE.fullmatch(text):
        return True
    return False


def is_new_topic_while_confirm_pending(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    if contextual_reply_kind(text) in {"yes", "no"}:
        return False

    low = text.lower()
    if (
        detect_prepare_intent(low)
        or detect_price_intent(low)
        or detect_test_assist_intent(low)
        or detect_test_result_intent(low)
        or detect_address_intent(low)
        or detect_schedule_intent(low)
        or detect_doctor_info_intent(low)
        or detect_doc_request_intent(low)
        or detect_nonbookable_walkin_intent(text)
    ):
        return True

    return ("?" in text) and (len(text.split()) >= 4)


def is_appointment_cancel_or_restart_request(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    return bool(_APPOINTMENT_CANCEL_OR_RESTART_RE.search(text))


def is_appointment_soft_pause_request(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    norm = _APPOINTMENT_SOFT_PAUSE_PUNCT_RE.sub(" ", text.lower().replace("ё", "е")).strip()
    norm = re.sub(r"\s+", " ", norm)
    if norm in _APPOINTMENT_SOFT_PAUSE_EXACT:
        return True
    return bool(_APPOINTMENT_SOFT_PAUSE_RE.search(norm))


def is_likely_topic_switch_from_appointment(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    if should_keep_appointment_flow_override(text):
        return False
    if is_appointment_cancel_or_restart_request(text):
        return False
    if is_appointment_soft_pause_request(text):
        return False
    low = text.lower()
    if (
        detect_test_result_intent(low)
        or detect_test_assist_intent(low)
        or detect_prepare_intent(low)
        or detect_doc_request_intent(low)
        or detect_price_intent(low)
        or detect_news_intent(low)
    ):
        return True
    return ("?" in text) and (len(text.split()) >= 4)


def _appointment_resume_prompt(state: SessionState, memory: MemoryStore) -> str:
    pending = memory.get_pending(state)
    missing: list[str] = []
    if isinstance(pending, dict) and pending.get("label") == "APPOINTMENT":
        raw_missing = pending.get("missing")
        if isinstance(raw_missing, list):
            missing = [str(x) for x in raw_missing if str(x).strip()]
    if not missing:
        missing = missing_slots("APPOINTMENT", state.last_entities)
    if missing:
        memory.set_pending(state, label="APPOINTMENT", missing_slots=missing)
        return clarification_question("APPOINTMENT", missing)

    if state.last_entities.get("appointment_confirm_pending"):
        return appointment_text_reask_confirm()

    summary = appointment_summary(state.last_entities)
    state.last_entities["appointment_confirm_pending"] = True
    return appointment_text_confirm_prompt(summary)


def run_appointment_precheck(
    *,
    user_text: str,
    state: SessionState,
    memory: MemoryStore,
    debug: bool,
    debug_state_update_factory: Callable[..., dict[str, Any]],
) -> ResponseEnvelope | None:
    if state.last_entities.get("appointment_cancel_pending") or state.last_entities.get("appointment_topic_switch_pending"):
        reply_kind = contextual_reply_kind(user_text)
        if reply_kind == "yes":
            clear_appointment_flow_context(state, memory)
            return ResponseEnvelope(
                text=appointment_text_cancelled(),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_flow_cancelled"},
                    context_action="new_topic",
                    confidence=1.0,
                ),
            )

        if reply_kind == "no":
            state.last_entities.pop("appointment_cancel_pending", None)
            state.last_entities.pop("appointment_topic_switch_pending", None)
            state.last_entities["appointment_flow_active"] = True
            return ResponseEnvelope(
                text=_appointment_resume_prompt(state, memory),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_flow_cancel_rejected"},
                    context_action="continue",
                    confidence=1.0,
                ),
            )

        prompt = (
            appointment_text_topic_switch_confirm()
            if state.last_entities.get("appointment_topic_switch_pending")
            else appointment_text_cancel_confirm()
        )
        return ResponseEnvelope(
            text=prompt,
            handoff=False,
            state_update=debug_state_update_factory(
                debug,
                label="APPOINTMENT",
                handoff=False,
                flags={"appointment_flow_cancel_confirm_reask"},
                context_action="continue",
                confidence=1.0,
            ),
        )

    pending = memory.get_pending(state)
    appointment_pending = isinstance(pending, dict) and pending.get("label") == "APPOINTMENT"
    appointment_flow_active = bool(state.last_entities.get("appointment_flow_active"))
    if (appointment_flow_active or appointment_pending) and not state.last_entities.get("appointment_confirm_pending"):
        if is_appointment_cancel_or_restart_request(user_text) or is_appointment_soft_pause_request(user_text):
            state.last_entities["appointment_cancel_pending"] = True
            return ResponseEnvelope(
                text=appointment_text_cancel_confirm(),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_flow_cancel_requested"},
                    context_action="continue",
                    confidence=1.0,
                ),
            )
        if is_likely_topic_switch_from_appointment(user_text):
            state.last_entities["appointment_topic_switch_pending"] = True
            return ResponseEnvelope(
                text=appointment_text_topic_switch_confirm(),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_flow_topic_switch_requested"},
                    context_action="continue",
                    confidence=1.0,
                ),
            )

    if state.last_entities.get("appointment_confirm_pending"):
        confirm_transition = appointment_confirmation_transition(user_text)
        if confirm_transition == APPOINTMENT_CONFIRM_YES:
            summary = appointment_summary(state.last_entities)
            state.last_entities["appointment_confirmed"] = True
            state.last_entities.pop("appointment_confirm_pending", None)
            state.last_entities.pop("appointment_flow_active", None)
            return ResponseEnvelope(
                text=appointment_text_confirmed_handoff(summary),
                handoff=True,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=True,
                    flags={"appointment_confirm_yes"},
                    confidence=1.0,
                ),
            )
        if confirm_transition == APPOINTMENT_CONFIRM_NO:
            state.last_entities["appointment_confirmed"] = False
            state.last_entities.pop("appointment_confirm_pending", None)
            for key in ("date_from", "date_to", "time_from", "time_to", "date_hint"):
                state.last_entities.pop(key, None)
            state.last_entities["appointment_flow_active"] = True
            return ResponseEnvelope(
                text=appointment_text_reask_datetime(),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_confirm_no"},
                    confidence=1.0,
                ),
            )
        if is_appointment_cancel_or_restart_request(user_text) or is_appointment_soft_pause_request(user_text):
            state.last_entities["appointment_cancel_pending"] = True
            return ResponseEnvelope(
                text=appointment_text_cancel_confirm(),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_flow_cancel_requested"},
                    context_action="continue",
                    confidence=1.0,
                ),
            )
        if is_new_topic_while_confirm_pending(user_text):
            state.last_entities["appointment_topic_switch_pending"] = True
            return ResponseEnvelope(
                text=appointment_text_topic_switch_confirm(),
                handoff=False,
                state_update=debug_state_update_factory(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_flow_topic_switch_requested"},
                    context_action="continue",
                    confidence=1.0,
                ),
            )

        return ResponseEnvelope(
            text=appointment_text_reask_confirm(),
            handoff=False,
            state_update=debug_state_update_factory(
                debug,
                label="APPOINTMENT",
                handoff=False,
                flags={"appointment_confirm_reask"},
                confidence=1.0,
            ),
        )

    return None
