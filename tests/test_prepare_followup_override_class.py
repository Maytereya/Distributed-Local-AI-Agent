"""Короткая реплика после памятки: тему другого вопроса не переклеивает правило по буквам (BUG-2026-10-28-PREPARE-FOLLOWUP-LETTERS).

Проверка всех слоёв кнопок 28.10 (четвёртый ход): «Цена анализа» → «ферритин» → «а как к нему подготовиться?» (памятка) →
«а где его сдать?» → та же памятка вместо адресов; так же «Как подготовиться к анализу» → «к нему» → «а где сдать?». Правило
`flow_prepare_followup_override` (`router._apply_post_nlu_guardrails`) держит подготовку для любой реплики до пяти слов, где
буквы не нашли цену, адрес или справку, — и перебивает тему, которую решила LLM: «где его сдать» детектор адреса не видит.

Инвариант класса: после ответа о подготовке тема вопроса о другом (адрес, запись, результаты), которую решила LLM, правилом по
буквам не переклеивается в PREPARE; встречная сторона — название анализа без темы («вульвоскопия» после памятки) остаётся
подготовкой, как задумано правилом. Без LLM: решение NLU — готовое, правило зовётся напрямую. Провал — xfail strict (чинить —
по слову владельца, LLM-first: решение темы — за LLM).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.services import Services


def _after_prepare(text: str, decision: RouteDecision, service: str = "Ферритин") -> RouteDecision:
    state = SessionState(session_id=f"prep-followup-{text}")
    state.last_entities.update({"_last_label": "PREPARE", **({"service_name": service} if service else {})})
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    return asyncio.run(router._apply_post_nlu_guardrails(decision, state, text, services, MemoryStore()))


# Вопросы о другом, которые буквы детекторов цены / адреса / справки не узнают; тема — решение LLM.
_OTHER_TOPIC = [("а где его сдать?", "ADDRESS"), ("а где сдать?", "ADDRESS"), ("а запишите меня", "APPOINTMENT"), ("а результаты когда?", "TEST_RESULT")]
# Услуга в состоянии: настоящая; слова вопроса, которые разбор положил услугой (живой прогон: «а как к нему подготовиться?» →
# «Нему подготовиться»); пусто. С настоящим анализом «где сдать» спасает детектор walk-in (тоже буквы), с мусором — нет.
_CONTEXTS = ["Ферритин", "Нему подготовиться", ""]
_RELABELED = {("Ферритин", "а запишите меня")} | {(c, t) for c in ("Нему подготовиться", "") for t in ("а где его сдать?", "а где сдать?", "а запишите меня")}


@pytest.mark.parametrize(
    ("service", "text", "label"),
    [
        pytest.param(c, t, lab, marks=pytest.mark.xfail(strict=True, reason="BUG-2026-10-28-PREPARE-FOLLOWUP-LETTERS"))
        if (c, t) in _RELABELED
        else pytest.param(c, t, lab)
        for c in _CONTEXTS
        for t, lab in _OTHER_TOPIC
    ],
)
def test_llm_topic_of_other_question_is_not_relabeled_to_prepare(service, text, label):
    decision = RouteDecision(label=label, confidence=0.8, source="llm_primary", flags={"topic_by_llm"})

    assert _after_prepare(text, decision, service).label == label


def test_bare_name_after_memo_stays_prepare():
    # Встречная сторона — то, ради чего правило написано: название без темы после памятки — подготовка, а не подбор анализов.
    decision = RouteDecision(label="TEST_ASSIST", confidence=0.6, source="llm_primary")

    assert _after_prepare("вульвоскопия", decision).label == "PREPARE"
