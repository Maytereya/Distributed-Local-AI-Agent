"""Guard-политики для активного APPOINTMENT flow.

Выносит из router.py state-machine для cancel/topic-switch в записи (шага «Подтвердите запись» нет с 08.10, вариант A;
обработчик фазы CONFIRM снят 18.10 — решение владельца Q2, класс 5).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from . import expects, pending_answer, pending_reply
from .city import match_city
from .flow_policy import clear_on_appointment_end, looks_like_patient_fio
from .memory import MemoryStore
from .mess_types import AppointmentPhase, ResponseEnvelope, SessionState
from .state_mutations import (
    mark_appointment_cancel_pending,
    mark_appointment_topic_switch_pending,
    reject_appointment_confirmation,
    set_appointment_selection_mode,
)
from .policies import (
    REQUIRED_SLOTS,
    appointment_service_display,
    appointment_text_cancel_confirm,
    appointment_text_cancelled,
    appointment_text_topic_switch_confirm,
    appointment_text_wish_prompt,
    clarification_question,
    detect_address_intent,
    detect_appointment_action,
    detect_doc_request_intent,
    detect_doctor_info_intent,
    detect_news_intent,
    detect_nonbookable_walkin_intent,
    detect_prepare_intent,
    detect_price_intent,
    detect_schedule_intent,
    detect_test_assist_intent,
    detect_test_result_intent,
    extract_branch_hint,
    has_datetime_signal,
    looks_like_branch_hint,
    missing_slots,
    normalize_appointment_action,
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


def _answers_branch_question(state: SessionState, user_text: str) -> bool:
    """Ответ на вопрос бота «Какой филиал вам удобен?» — не смена темы (класс 5, решение владельца 18.10).

    Бот знает, чего ждёт (`_expects` прошлого хода — `appointment_branch`). Ответ разбирает LLM (`pending_reply`,
    решение этого хода): «другая тема» — прежняя проверка смены темы; выбран адрес — это филиал; «любой филиал» —
    филиал «любой филиал», оператор получает его как есть; «список врачей» — список; иной исход — запись идёт дальше.
    LLM не решала (выключатель, сбой) — запасной путь, правила только сужают: ответ о филиалах, адресах, врачах или
    показанный вариант — в домене вопроса. Иначе — прежняя проверка.

    :return: True — ответ на вопрос о филиале (смены темы нет); False — проверять смену темы, как было
    """

    raw = state.last_entities.get(expects.EXPECTS_KEY)
    if not isinstance(raw, dict) or raw.get("type") != expects.Question.APPOINTMENT_BRANCH.value:
        return False
    # Одна функция ответа на висящий вопрос (без неё — прежний клиент): пункт — филиал / «список врачей» / «любой
    # филиал» (оператору как есть, решение 18.10); другая тема — прежняя проверка смены темы; «ничего из этого» по
    # промпту — в том числе уточняющий вопрос о самом вопросе («а какой ближе к …?») — ответ в домене вопроса.
    decided = pending_answer.branch_answer(user_text)
    if decided is not None:
        outcome, picked = decided
        if outcome == "new_topic":
            return False
        if picked == expects.DOCTOR_LIST_OPTION:
            set_appointment_selection_mode(state, "doctor")
        elif picked:
            state.last_entities["branch_name"] = picked
        return True
    low = str(user_text or "").lower()
    shown = state.last_entities.get("appointment_branch_options")
    hint = extract_branch_hint(user_text, state.last_entities)
    answers = bool(
        detect_address_intent(low)
        or detect_doctor_info_intent(low)
        or (isinstance(shown, list) and hint in shown)
    )
    if answers:
        pending_reply.note_rules_branch_answer(user_text)
    return answers


def _nothing_chosen_new_topic(pending_missing: list[str]) -> bool:
    """На вопрос цели записи одна функция решила «другая тема», а цель записи (врач / специальность / услуга) ещё не выбрана
    (`booking_target_slot_llm = on`)."""

    found = pending_answer.slot_answer()
    return (
        pending_answer.booking_target_on()
        and found is not None
        and found.outcome is pending_answer.Outcome.NEW_TOPIC
        and REQUIRED_SLOTS["APPOINTMENT"][0] in pending_missing
    )


def _appointment_wish_resume_prompt(state: SessionState, memory: MemoryStore) -> str:
    """Вариант A (08.10): вернуться к записи — значит снова спросить пожелание по времени."""

    entities = state.last_entities
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["appointment_wish"])
    entities["_appointment_wish_asked"] = True
    action = str(entities.get("appointment_action") or "").strip().lower()
    branch = str(entities.get("branch_name") or entities.get("city") or "выбранном филиале").strip()
    return appointment_text_wish_prompt(
        appointment_service_display(entities), branch, None, reschedule=action == "reschedule"
    )


def _appointment_resume_prompt(state: SessionState, memory: MemoryStore) -> str:
    pending = memory.get_pending(state)
    missing: list[str] = []
    if isinstance(pending, dict) and pending.get("label") == "APPOINTMENT":
        raw_missing = pending.get("missing")
        if isinstance(raw_missing, list):
            missing = [str(x) for x in raw_missing if str(x).strip()]
    if "appointment_wish" in missing:
        return _appointment_wish_resume_prompt(state, memory)
    if not missing:
        missing = missing_slots("APPOINTMENT", state.last_entities)
    if missing:
        memory.set_pending(state, label="APPOINTMENT", missing_slots=missing)
        if "appointment_action" in missing:  # «Хотите отменить или перенести запись?» — поле ожидания (D2.1)
            expects.ask(expects.Question.APPOINTMENT_ACTION, kind=expects.PICK, options=expects.APPOINTMENT_ACTION_OPTIONS)
        return clarification_question("APPOINTMENT", missing, state.last_entities)

    # Вариант A: карточки «Подтверждаете?» нет — цель и филиал есть, спрашиваем пожелание.
    return _appointment_wish_resume_prompt(state, memory)


_UNDECIDED = object()
_FLOW_QUESTIONS = (expects.Question.APPOINTMENT_ACTION, expects.Question.APPOINTMENT_BRANCH, expects.Question.BOOKING_SERVICE)


def _cancel_confirm(state: SessionState, debug: bool, factory: Callable[..., dict[str, Any]], *, topic_switch: bool) -> ResponseEnvelope:
    """Вопрос «Отменить текущий процесс записи?» (или «…перейти к новому вопросу?») — собранное пока не трогаем."""

    (mark_appointment_topic_switch_pending if topic_switch else mark_appointment_cancel_pending)(state)
    expects.ask(expects.Question.APPOINTMENT_ABORT)
    return ResponseEnvelope(
        text=appointment_text_topic_switch_confirm() if topic_switch else appointment_text_cancel_confirm(),
        handoff=False,
        state_update=factory(
            debug,
            label="APPOINTMENT",
            handoff=False,
            flags={"appointment_flow_topic_switch_requested" if topic_switch else "appointment_flow_cancel_requested"},
            context_action="continue",
            confidence=1.0,
        ),
    )


def _flow_answer(
    state: SessionState,
    memory: MemoryStore,
    user_text: str,
    waiting_action_choice: bool,
    debug: bool,
    factory: Callable[..., dict[str, Any]],
) -> ResponseEnvelope | None | object:
    """Ответ на вопрос шага записи (действие с записью, филиал, услуга), решённый одной функцией (класс 5): решение
    читает предпроверка, а не свои правила паузы и смены темы (консилиум 18.10, «Рефакторинг Next» А2).

    :return: ответ хода (вопрос «Отменить?»), None — ход идёт дальше (пункт исполнят обработчики и NLU записи), или
        `_UNDECIDED` — решения нет (выключатель, сбой LLM, «ничего из этого») — прежние правила
    """

    slot = pending_answer.slot_answer()
    if slot is not None and (slot.in_topic or slot.outcome is pending_answer.Outcome.NO):
        # Ответ на вопрос-слот («Уточните название услуги» после «нет» на «Похоже, вы имели в виду …?»), решённый одной
        # функцией (`pending_buttons`): его исполняют обработчики и NLU, не правила паузы и отмены записи.
        return None
    answered = next((found for q in _FLOW_QUESTIONS if (found := pending_answer.current(q)) is not None), None)
    if answered is None or not answered.decided:
        return _UNDECIDED
    outcome = answered.outcome
    if answered.type == expects.Question.APPOINTMENT_ACTION.value and waiting_action_choice:
        # Ответ на «Хотите отменить или перенести запись?» целиком разбирает обработчик выбора действия: распознанная
        # отмена — всегда оператор (вето правил при любом решении модели), пункт, «нет», пауза, другая тема.
        return None
    if outcome in (pending_answer.Outcome.YES, pending_answer.Outcome.PICK):
        if answered.type == expects.Question.APPOINTMENT_BRANCH.value:
            _answers_branch_question(state, user_text)
        return None
    if outcome in (pending_answer.Outcome.NO, pending_answer.Outcome.PAUSE):
        return _cancel_confirm(state, debug, factory, topic_switch=False)
    # «Другая тема» от модели — повод, не вердикт: смену темы подтверждают прежние проверки ниже (правила сужают), иначе
    # запись идёт дальше («массаж спины» на список услуг — не новая тема).
    return _UNDECIDED


def run_appointment_precheck(
    *,
    user_text: str,
    state: SessionState,
    memory: MemoryStore,
    debug: bool,
    debug_state_update_factory: Callable[..., dict[str, Any]],
) -> ResponseEnvelope | None:
    if state.last_entities.get("appointment_cancel_pending") or state.last_entities.get("appointment_topic_switch_pending"):
        abort_q = expects.Question.APPOINTMENT_ABORT
        reply_kind = pending_answer.kind(abort_q, lambda: contextual_reply_kind(user_text))
        if pending_answer.passes(abort_q) or reply_kind == "pause":
            # Другая тема, «я подумаю» или хвост («нет, а где филиал?»): собранное цело (стирает только явное «да»),
            # снят только вопрос, ход идёт обычным путём (консилиум 18.10, МЕГА 1(а)).
            reject_appointment_confirmation(state, extra_keys=("appointment_cancel_pending", "appointment_topic_switch_pending"))
            return None
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
        expects.ask(expects.Question.APPOINTMENT_ABORT, reask=True)  # ответ не разобран — тот же вопрос снова
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
    if appointment_flow_active or appointment_pending:
        flow_reply = _flow_answer(state, memory, user_text, waiting_action_choice, debug, debug_state_update_factory)
        if flow_reply is not _UNDECIDED:
            return flow_reply
        # Когда ждем именно выбор действия (отмена/перенос), короткие "да/нет"
        # не считаем soft-pause/cancel, чтобы обработка шла в pending-ветке роутера.
        if waiting_action_choice and reply_kind in {"yes", "no"}:
            return None
        # BUG-2026-10-14-CANCEL-EXISTING-LOST: на вопрос «Хотите отменить или перенести запись?» «отменить» — выбор
        # отмены СУЩЕСТВУЮЩЕЙ записи, а не «прервать оформление»: его разбирает обработчик выбора действия
        # (`router._handle_appointment_action_pending` → сразу оператор, решение владельца 04.10/08.10).
        if waiting_action_choice and normalize_appointment_action(detect_appointment_action(user_text), user_text) == "cancel":
            return None
        if is_appointment_cancel_or_restart_request(user_text) or is_appointment_soft_pause_request(user_text):
            mark_appointment_cancel_pending(state)
            expects.ask(expects.Question.APPOINTMENT_ABORT)
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
        if _nothing_chosen_new_topic(pending_missing):
            # Отменять нечего: на вопрос цели записи — другая тема (решение одной функции), а врач, специальность и услуга
            # ещё не выбраны. Без вопроса «Отменить оформление?» реплика идёт так же, как набранная в новой сессии: запись,
            # в которой ничего нет, снимается — иначе ответом другой темы становился повтор «к какому врачу?» (класс E).
            # Ошибся ли выбор модели («болит стопа, нужно записаться на Ленина 5» — «другая тема»), решает обычный путь, как
            # для текста (BUG-2026-10-25-BOOK-BUTTON-REPLY-AS-TOPIC-SWITCH, `booking_target_slot_llm`).
            clear_on_appointment_end(state, memory)
            return None
        if not _answers_branch_question(state, user_text) and is_likely_topic_switch_from_appointment(user_text):
            mark_appointment_topic_switch_pending(state)
            expects.ask(expects.Question.APPOINTMENT_ABORT)
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

    return None
