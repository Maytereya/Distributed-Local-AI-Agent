"""Guard-политики для активного APPOINTMENT flow.

Выносит из router.py state-machine для confirm/cancel/topic-switch в записи.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .city import match_city
from .flow_policy import clear_on_appointment_end, looks_like_patient_fio
from .memory import MemoryStore
from .mess_types import AppointmentPhase, ResponseEnvelope, SessionState
from .state_mutations import (
    finalize_appointment_confirmation,
    mark_appointment_cancel_pending,
    mark_appointment_confirm_pending,
    mark_appointment_topic_switch_pending,
    reject_appointment_confirmation,
)
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
from .russian_nlu import normalize_ru
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
    r"не\s+туда|"
    r"не\s+так|"
    r"останов(?:ись|ите)\w*|"
    r"бред\w*|"
    r"ошибк\w*|"
    r"извините|"
    r"не\s+то|"
    r"не\s+это|"
    r"я\s+друг(ое|ой)\s+хотел\w*"
    r")\b",
    re.I,
)
_APPOINTMENT_SOFT_PAUSE_EXACT = {"нет", "ладно"}
_APPOINTMENT_SOFT_PAUSE_PUNCT_RE = re.compile(r"[!.,?;:]+")

# Extended appointment-context keys (doctor identity / service context) are
# cleared by flow_policy.clear_on_appointment_end(), not by runtime reset alone.
def _is_topic_switch_intent(text: str) -> bool:
    """Определяет, что реплика уводит из сценария записи в другую тему.

    :param text: исходный текст пользователя
    :return: True, если сработал любой intent-детектор нового топика
    """
    raw = str(text or "").strip()
    if not raw:
        return False

    low = raw.lower()
    return any(
        (
            detect_test_result_intent(low),
            detect_test_assist_intent(low),
            detect_prepare_intent(low),
            detect_price_intent(low),
            detect_address_intent(low),
            detect_schedule_intent(low),
            detect_doctor_info_intent(low),
            detect_doc_request_intent(low),
            detect_nonbookable_walkin_intent(raw),
            detect_news_intent(low),
        )
    )


def should_keep_appointment_flow_override(user_text: str) -> bool:
    """
    Разрешаем OTHER->APPOINTMENT override только для реплик,
    похожих на продолжение сценария записи.
    """
    text = str(user_text or "").strip()
    if not text:
        return False

    if _is_topic_switch_intent(text):
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

    if _is_topic_switch_intent(text):
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
    norm = _APPOINTMENT_SOFT_PAUSE_PUNCT_RE.sub(" ", normalize_ru(text)).strip()
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
    if _is_topic_switch_intent(text):
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
        return clarification_question("APPOINTMENT", missing, state.last_entities)

    if state.dialog.phase == AppointmentPhase.CONFIRM:
        return appointment_text_reask_confirm()

    summary = appointment_summary(state.last_entities)
    mark_appointment_confirm_pending(state)
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
            clear_on_appointment_end(state, memory)
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
            reject_appointment_confirmation(state, extra_keys=("appointment_cancel_pending", "appointment_topic_switch_pending"))
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
    appointment_flow_active = state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
    pending_missing: list[str] = []
    if appointment_pending:
        raw_missing = pending.get("missing")
        if isinstance(raw_missing, list):
            pending_missing = [str(x) for x in raw_missing if str(x).strip()]
    waiting_action_choice = "appointment_action" in pending_missing
    reply_kind = contextual_reply_kind(user_text)
    if (appointment_flow_active or appointment_pending) and state.dialog.phase != AppointmentPhase.CONFIRM:
        # Когда ждем именно выбор действия (отмена/перенос), короткие "да/нет"
        # не считаем soft-pause/cancel, чтобы обработка шла в pending-ветке роутера.
        if waiting_action_choice and reply_kind in {"yes", "no"}:
            return None
        if is_appointment_cancel_or_restart_request(user_text) or is_appointment_soft_pause_request(user_text):
            mark_appointment_cancel_pending(state)
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
            mark_appointment_topic_switch_pending(state)
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

    if state.dialog.phase == AppointmentPhase.CONFIRM:
        confirm_transition = appointment_confirmation_transition(user_text)
        if confirm_transition == APPOINTMENT_CONFIRM_YES:
            summary = appointment_summary(state.last_entities)
            finalize_appointment_confirmation(state)
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
            reject_appointment_confirmation(
                state,
                extra_keys=("date_from", "date_to", "time_from", "time_to", "date_hint"),
            )
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
            mark_appointment_cancel_pending(state)
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
            mark_appointment_topic_switch_pending(state)
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
