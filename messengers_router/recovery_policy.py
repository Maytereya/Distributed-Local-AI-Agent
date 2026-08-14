"""Recovery/HITL-политика для неуверенных и неоднозначных входов.

Ответственность модуля:
1) Единообразно обработать "не понял" без размазывания логики по роутеру.
2) Поддержать мягкую эскалацию: clarify -> clarify+options -> handoff.
3) Обработать явный запрос на оператора и контекстные yes/no-ответы.

Модуль не принимает решений о бизнес-интентах (APPOINTMENT/PRICE/...),
а только управляет восстановлением диалога, когда классификация неустойчива.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .mess_types import RouteDecision
from .policies import clarification_question, handoff_message, is_context_affirmative, is_context_negative
from .prompt_registry import load_prompt_text
from .text_templates import LOW_CONF_CLARIFY_TEXT

# Явный запрос «нужен живой человек / не бот». Сознательно НЕ ловим голое
# «человек» (иначе ложно срабатывает «цена с человека»): только в связке с
# глаголом (позовите/переведите/соедините…) или явными маркерами.
_OPERATOR_REQUEST_RE = re.compile(
    r"(?:"
    r"\bоператор\w*"  # оператор, оператора, к оператору
    r"|\bжив(?:ой|ого|ым|ому|ом)\s+человек\w*"  # живой человек
    r"|\bне\s+бот\w*"  # не бот
    r"|\bне\s+робот\w*"  # не робот
    r"|\bобычн\w+\s+чат\w*"  # обычный чат (не бот)
    r"|\bменеджер\w*"  # менеджер
    r"|\b(?:позов\w+|перевед\w+|переключ\w+|соедин\w+|свяж\w+)\b.{0,15}\bчеловек\w*"  # переведите на человека
    r")",
    re.I,
)
_LOW_CONF_CLARIFY_OPTIONS_TEXT = (
    "Чтобы помочь быстрее, выберите вариант: "
    "1) запись к врачу, 2) расписание врача, 3) стоимость, 4) адреса филиалов, 5) результаты анализов."
)


RecoveryKind = Literal["none", "clarify", "handoff"]


# П6-3 (решение владельца 14.08): после ДВУХ непонятых подряд предлагаем
# оператора ВОПРОСОМ, а не переводим принудительно — принудительный перевод
# сбрасывал людей, которым бот отвечает нормально. Формулировка совпадает с
# анти-залипным гардом `_maybe_offer_operator_on_repeat`, чтобы пациент видел
# один и тот же оффер независимо от того, какой механизм сработал.
OPERATOR_OFFER_AFTER_UNCLEAR = 2
UNCLEAR_OPERATOR_OFFER_TEXT = "Похоже, мне не удаётся помочь с этим в чате. Перевести на оператора?"


@dataclass
class RecoveryAction:
    kind: RecoveryKind
    text: str = ""
    handoff: bool = False
    reason: str = ""
    unclear_count: int = 0
    # True → вызывающий обязан поставить operator-offer pending: следующий ход
    # обработает «да/нет» через `_handle_operator_offer_pending`.
    offer_operator: bool = False


def explicit_operator_requested(user_text: str) -> bool:
    return bool(_OPERATOR_REQUEST_RE.search(str(user_text or "")))


def contextual_reply_kind(user_text: str) -> Literal["yes", "no", "other"]:
    if is_context_affirmative(user_text):
        return "yes"
    if is_context_negative(user_text):
        return "no"
    return "other"


def _build_recovery_text(user_text: str, summary: str, last_label: str) -> str:
    """
    LLM-assisted текст для второго уточнения. При ошибке — статический fallback.
    """
    try:
        tmpl = load_prompt_text("recovery_patient")
        prompt = (
            tmpl.replace("<<USER_TEXT>>", str(user_text or ""))
            .replace("<<SUMMARY>>", str(summary or ""))
            .replace("<<LAST_LABEL>>", str(last_label or "OTHER"))
        )
    except Exception:
        return _LOW_CONF_CLARIFY_OPTIONS_TEXT

    # На этом этапе не подключаем отдельный LLM call, чтобы не увеличивать latency.
    # Используем шаблон как fallback-текст.
    _ = prompt
    return _LOW_CONF_CLARIFY_OPTIONS_TEXT


def _label_patient_option(label: str) -> str:
    mapping = {
        "APPOINTMENT": "запись к врачу",
        "DOCTOR_SCHEDULE": "расписание врача",
        "PRICE": "стоимость услуги",
        "ADDRESS": "адреса филиалов",
        "TEST_RESULT": "результаты анализов",
        "DOCTOR_INFO": "информация о враче",
        "TEST_ASSIST": "подбор анализов",
        "PREPARE": "подготовка к исследованию",
        "NEWS": "акции и предложения",
    }
    return mapping.get(label, "уточнение запроса")


def _intent_disambiguation_text(candidates: list[str]) -> str:
    clean = [c for c in candidates if c]
    if not clean:
        return _LOW_CONF_CLARIFY_OPTIONS_TEXT
    if len(clean) == 1:
        return f"Уточните, пожалуйста: вам нужна {_label_patient_option(clean[0])}?"
    human = [f"{i}. {_label_patient_option(label)}" for i, label in enumerate(clean[:3], 1)]
    return "Уточните, пожалуйста, что именно вам нужно:\n" + "\n".join(human)


def _structured_clarify_text(decision: RouteDecision, flow_label: str, summary: str, user_text: str) -> str:
    if decision.clarify_reason in {"slot_request", "context_repair"} and decision.clarify_slots:
        return clarification_question(flow_label or decision.label, list(decision.clarify_slots), decision.entities)
    if decision.clarify_reason in {"intent_disambiguation", "low_confidence"}:
        candidates = list(decision.intent_candidates)
        if decision.label != "OTHER" and decision.label not in candidates:
            candidates.insert(0, decision.label)
        return _intent_disambiguation_text(candidates)
    return _build_recovery_text(user_text, summary, flow_label)


def evaluate_recovery(
    *,
    user_text: str,
    decision: RouteDecision,
    flow_label: str,
    pending_exists: bool,
    flow_active: bool,
    state_entities: dict[str, Any],
    summary: str = "",
    max_unclear: int = 3,
) -> RecoveryAction:
    """
    Мягкая политика:
    - явный запрос оператора -> handoff сразу
    - low confidence (OTHER/TEST_ASSIST) -> clarify 1/2, handoff на 3-м
    """
    if explicit_operator_requested(user_text):
        state_entities["_nlu_unclear_count"] = 0
        return RecoveryAction(
            kind="handoff",
            text=handoff_message("manual_operator"),
            handoff=True,
            reason="manual_operator",
            unclear_count=0,
        )

    is_structured_clarify = bool(
        decision.clarify_needed
        and flow_label == decision.label
        and not pending_exists
        and not flow_active
    )
    is_low_conf_case = (
        not is_structured_clarify
        and "low_confidence" in decision.flags
        and decision.label in {"OTHER", "TEST_ASSIST"}
        and flow_label == decision.label
        and not pending_exists
        and not flow_active
    )
    if not (is_structured_clarify or is_low_conf_case):
        state_entities["_nlu_unclear_count"] = 0
        return RecoveryAction(kind="none", unclear_count=0)

    prev = state_entities.get("_nlu_unclear_count")
    try:
        n = int(prev) if prev is not None else 0
    except Exception:
        n = 0
    n += 1
    state_entities["_nlu_unclear_count"] = n

    # П6-3: на ВТОРОМ непонятом подряд — оффер оператора ВОПРОСОМ. Раньше здесь
    # на третьем ходу стоял принудительный `handoff=True`; владелец 14.08 решил
    # не переводить принудительно. Текст уточнения сохраняем: пациент видит и
    # переспрос, и предложение — и сам выбирает.
    if n >= max(1, min(int(max_unclear), OPERATOR_OFFER_AFTER_UNCLEAR)):
        state_entities["_nlu_unclear_count"] = 0
        clarify_text = (
            _structured_clarify_text(decision, flow_label, summary, user_text)
            if is_structured_clarify
            else _build_recovery_text(user_text, summary, flow_label)
        )
        return RecoveryAction(
            kind="clarify",
            text=f"{clarify_text}\n\n{UNCLEAR_OPERATOR_OFFER_TEXT}",
            handoff=False,
            reason="low_confidence_operator_offer",
            unclear_count=n,
            offer_operator=True,
        )

    if n == 1:
        return RecoveryAction(
            kind="clarify",
            text=_structured_clarify_text(decision, flow_label, summary, user_text)
            if is_structured_clarify
            else LOW_CONF_CLARIFY_TEXT,
            handoff=False,
            reason=decision.clarify_reason or "low_confidence_clarify_1",
            unclear_count=n,
        )

    return RecoveryAction(
        kind="clarify",
        text=_structured_clarify_text(decision, flow_label, summary, user_text)
        if is_structured_clarify
        else _build_recovery_text(user_text, summary, flow_label),
        handoff=False,
        reason=decision.clarify_reason or "low_confidence_clarify_2",
        unclear_count=n,
    )
