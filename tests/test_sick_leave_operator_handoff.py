"""П9 (faithfulness): «больничный лист» — бот галлюцинировал «клиника НЕ оформляет».

Инцидент 25.06: на вопрос про больничный бот выдумал «клиника не оформляет
больничные» (на деле оформляет: любой врач на приёме при показаниях). Пока факты
не подтверждены владельцем — решение: БЛ-интент → ПЕРЕВОД НА ОПЕРАТОРА, не
факт-ответ (честный handoff > выдумка).

Архитектура: legacy_v2 — deterministic_rule_decision не-None короткозамыкает LLM,
правило драйвит прод. Ветка по образцу website_help (BUG-2026-06-09-06): clarify
с детерминированным текстом, но с needs_handoff=True (envelope.handoff=True —
сигнал агрегатору переключить на оператора).

Инвариант (класс): вопрос про больничный лист / лист(ок) нетрудоспособности →
handoff на оператора с честным текстом, НЕ утверждение «оформляем/не оформляем».
Анти-FP: «Больничная улица» (есть филиал на ул. Больничная 44Г в Красном Яру!),
«лист назначений», «прайс-лист» — НЕ больничный. Грамматика различает: sick-leave
= маскулинные формы («больничный/больничного…»), улица = феминные («Больничная»).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.classifier import deterministic_rule_decision
from messengers_router.mess_types import SessionState
from messengers_router.orchestrator import run_pipeline
from messengers_router.policies import SICK_LEAVE_HANDOFF_TEXT, detect_sick_leave_intent


def run(coro):
    return asyncio.run(coro)


SICK_LEAVE_POSITIVE = [
    "Вы оформляете больничный?",
    "нужен больничный лист",
    "можно у вас открыть больничный?",
    "выдаёте листок нетрудоспособности?",
    "продлить больничный на 3 дня",
    "занимаетесь ли больничными листами?",
    "лист нетрудоспособности делаете?",
    "нужна справка о нетрудоспособности",
    "Больничный оформляете?",
]

SICK_LEAVE_NEGATIVE = [
    "есть филиал на ул. Больничная 44Г?",          # адрес филиала в Красном Яру
    "Больничная улица где находится?",              # адрес
    "как доехать до Больничной 44Г",                # адрес, косвенный падеж
    "запишите на Больничную",                       # адрес
    "лист назначений где взять?",                   # другой документ
    "прайс-лист пришлите",                          # не «лист нетрудоспособности»
    "запишите к врачу",
    "сколько стоит ОАК",
]


@pytest.mark.parametrize("text", SICK_LEAVE_POSITIVE, ids=[t[:28] for t in SICK_LEAVE_POSITIVE])
def test_detect_sick_leave_positive(text):
    assert detect_sick_leave_intent(text) is True, f"{text!r} not detected as sick-leave"


@pytest.mark.parametrize("text", SICK_LEAVE_NEGATIVE, ids=[t[:28] for t in SICK_LEAVE_NEGATIVE])
def test_detect_sick_leave_negative(text):
    assert detect_sick_leave_intent(text) is False, f"{text!r} wrongly detected as sick-leave"


def test_rule_decision_sick_leave_hands_off_to_operator():
    d = run(deterministic_rule_decision("Вы оформляете больничный лист?", {}))
    assert d is not None
    assert "rule_sick_leave" in d.flags
    assert d.needs_handoff is True
    assert d.clarify_needed is True
    assert d.clarify_reason == SICK_LEAVE_HANDOFF_TEXT
    assert "оператор" in d.clarify_reason.lower()


def test_rule_decision_sick_leave_wins_over_doc_request():
    """«справка о нетрудоспособности» = больничный → оператор, НЕ doc_request-флоу."""
    d = run(deterministic_rule_decision("нужна справка о нетрудоспособности", {}))
    assert d is not None
    assert "rule_sick_leave" in d.flags
    assert "doc_request_main_index" not in d.flags


def test_rule_decision_bolnichnaya_street_is_not_sick_leave():
    """Анти-FP: адрес филиала на ул. Больничная НЕ уводит на оператора."""
    d = run(deterministic_rule_decision("Где находится филиал на Больничной 44Г?", {}))
    if d is not None:
        assert "rule_sick_leave" not in d.flags
        assert d.needs_handoff is False


def test_pipeline_sick_leave_envelope_carries_handoff():
    """Envelope-инвариант: текст оператора + handoff=True доезжают до агрегатора."""
    state = SessionState(session_id="sick-leave-envelope")
    ctx = run(run_pipeline("Вы оформляете больничный?", state))
    assert ctx.response is not None
    assert ctx.response.text == SICK_LEAVE_HANDOFF_TEXT
    assert ctx.response.handoff is True


def test_pipeline_sick_leave_never_denies_service():
    """Ядро П9: НИКОГДА не утверждать «не оформляем/не занимаемся» про больничный."""
    state = SessionState(session_id="sick-leave-no-denial")
    ctx = run(run_pipeline("Занимаетесь ли вы больничными?", state))
    assert ctx.response is not None
    low = ctx.response.text.lower()
    assert "не оформля" not in low
    assert "не занима" not in low
    assert "оператор" in low
