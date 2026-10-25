"""Кнопка меню после нажатия: тема ответа на её вопрос известна (решение владельца 25.10 — ~90 % пациентов придут через меню).

Нажатие листа меню — явный выбор темы (`button_menu`). Кнопка режима `slot` спрашивает своё («Какой анализ интересует?»,
«К какой процедуре нужна подготовка?»), и висящий вопрос (`_pending`) помнит, какая кнопка его задала, —
`_pending[QUESTION_KEY]`. Пока вопрос висит, тема ответа на него известна из нажатия; кто этим пользуется — каждый под своим
выключателем, по умолчанию off (как было):

* `button_answer_topic` (BUG-2026-10-25-BUTTON-ANSWER-CLASSIFIER) — LLM-классификатор темы ответ на вопрос кнопки не
  переспрашивает. Журнал хода прода 14–24.10: после нажатия кнопки-темы следующая реплика ждала классификатор в 35 % ходов,
  с 21.10 — в 59 % (p50 3,7 с, до 12 с), и в половине из них он возвращал ту же тему. Одна функция ответа на висящий вопрос
  (`pending_answer`, вид `free`) решила «это ответ» — тему решает кнопка там, где решала бы LLM-классификатор (правило
  не сработало или его ветка в `topic_by_llm`); надёжные правила и срочное — как всегда. У кнопок цены своего вопроса-слота
  не было — его ответ судила LLM-классификатор; под on его судит та же функция (≈ 1 с вместо 3,7 с), и это решение лимита
  решений хода не тратит: оно — шаг NLU вместо классификатора, а пути цены остаются её решения, как без кнопки.
"""

from __future__ import annotations

from typing import Any, Iterable

from . import button_menu, pending_answer, turn_telemetry
from .expects import Question
from .runtime_config import config as _cfg

QUESTION_KEY = "button"  # `_pending[QUESTION_KEY]` — лист меню, который задал висящий вопрос
# Темы вопросов кнопок, ответ на которые судит одна функция только под `button_answer_topic` (остальные — `SLOT_TOPICS`).
_JUDGED_UNDER_SWITCH = frozenset({"PRICE"})


def _on(key: str) -> bool:
    return str(getattr(_cfg, key, "off") or "").strip().lower() == "on"


def answer_topic_on() -> bool:
    """`[MESSENGER_ROUTER] button_answer_topic = on`: тему ответа на вопрос кнопки решает кнопка, не LLM-классификатор."""

    return _on("MR_BUTTON_ANSWER_TOPIC")


def marks_questions() -> bool:
    """Висящий вопрос кнопки помечается её листом — только когда пометку кто-то читает (off — состояние как было)."""

    return answer_topic_on()


def pressed(flags: Iterable[Any] | None) -> str:
    """Лист меню, нажатый на этом ходе, — из флага решения кнопки `button:<id>` (`orchestrator._button_decision`)."""

    for flag in flags or ():
        if str(flag).startswith("button:"):
            return str(flag).split(":", 1)[1]
    return ""


def question_button(pending: Any) -> str:
    """Лист меню, который задал висящий вопрос; вопрос не кнопки — пусто."""

    return str(pending.get(QUESTION_KEY) or "") if isinstance(pending, dict) else ""


def mark_question(pending: Any, button_id: str) -> None:
    """Пометить висящий вопрос листом кнопки — если вопрос её темы (кнопка `slot` спрашивает недостающее своей метки)."""

    action = button_menu.action_for(button_id) if button_id else None
    if isinstance(pending, dict) and action is not None and action.mode == "slot" and pending.get("label") == action.label:
        pending[QUESTION_KEY] = button_id


def judges_answer(label: str, pending: Any) -> bool:
    """Вопрос кнопки этой темы получает свободный ответ одной функции под `button_answer_topic` (кнопки цены): вопрос
    `SLOT_TOPICS` получает его и без выключателя (`pending_buttons`)."""

    return answer_topic_on() and label in _JUDGED_UNDER_SWITCH and bool(question_button(pending))


def answer_topic(pending: Any) -> str:
    """Тема ответа на вопрос кнопки, если её знает кнопка: выключатель, вопрос задала кнопка, одна функция решила «это
    ответ» (правилами или LLM). Иначе пусто — тему решает NLU, как было."""

    if not answer_topic_on() or not question_button(pending):
        return ""
    found = pending_answer.slot_answer()
    if found is None or found.outcome is not pending_answer.Outcome.ANSWER:
        return ""
    return str(pending.get("label") or "")


def refund_topic_judge(answered: pending_answer.Answer | None, pending: Any) -> None:
    """Решение одной функции об ответе на вопрос кнопки цены — шаг NLU вместо классификатора: лимит решений хода не тратит,
    пути цены остаются её решения, как без кнопки (`button_answer_topic`)."""

    from . import llm_runtime

    if (
        answered is None
        or answered.source is not pending_answer.Source.LLM
        or answered.outcome is not pending_answer.Outcome.ANSWER  # прочие исходы ход не закрывают — их уже вернула функция (Q5)
        or answered.type != Question.SLOT_VALUE.value
        or not judges_answer(str((pending or {}).get("label") or ""), pending)
    ):
        return
    if llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_button_refund=1)
