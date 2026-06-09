"""BUG-F: «Надо записываться?» на лаб-анализы → бот ошибочно предлагает запись.

Прод-диалог (ЮГ-2): в контексте сдачи анализов (живая очередь, без записи) на
follow-up «Надо записываться?» бот ответил «Есть возможность записи… Ленина 5.
Какой филиал?» — противоречит собственному «анализы без записи».

Корень (офлайн-repro): bare «Надо записываться?» — нет слова «анализ» в самом
сообщении и нет submit-verb («сдать»), поэтому `detect_nonbookable_walkin_intent`
возвращал False (follow-up branch требовал submit-verb) → «записываться» ловил
APPOINTMENT-rule → оффер записи. «нужно ли записываться на АНАЛИЗЫ» (со словом
«анализ») уже работало (ADDRESS).

Инвариант (класс): ВОПРОС о необходимости записи («надо/нужно ли записываться?»,
«нужна ли запись?») в активном ЛАБ-контексте (test_name/анализ-услуга) → честный
walk-in ответ (ADDRESS «без записи»), НЕ APPOINTMENT. Это ВОПРОС, не действие
записи. Doctor-контекст (specialty/ФИО) исключён — к врачу запись нужна. Без
лаб-контекста поведение не меняется. Класс по сигнатуре «необходимость+запись +
лаб-контекст», не инстанс.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.classifier import deterministic_rule_decision
from messengers_router.policies import (
    detect_booking_necessity_question,
    detect_nonbookable_walkin_intent,
)


def run(coro):
    return asyncio.run(coro)


NECESSITY_POSITIVE = [
    "Надо записываться?",
    "нужно ли записываться",
    "нужна ли запись",
    "обязательно ли записываться",
    "запись обязательна?",
    "надо ли запись делать",
]
NECESSITY_NEGATIVE = [
    "запишите меня",
    "записаться к кардиологу",
    "хочу записаться на УЗИ",
    "сколько стоит ОАК",
    "где сдать анализ",
]


@pytest.mark.parametrize("text", NECESSITY_POSITIVE, ids=[t[:20] for t in NECESSITY_POSITIVE])
def test_detect_booking_necessity_positive(text):
    assert detect_booking_necessity_question(text) is True, f"{text!r} not detected"


@pytest.mark.parametrize("text", NECESSITY_NEGATIVE, ids=[t[:20] for t in NECESSITY_NEGATIVE])
def test_detect_booking_necessity_negative(text):
    assert detect_booking_necessity_question(text) is False, f"{text!r} wrongly detected"


def test_nonbookable_walkin_necessity_question_with_lab_context():
    """«Надо записываться?» + активный лаб-контекст → walk-in (True)."""
    assert detect_nonbookable_walkin_intent("Надо записываться?", {"test_name": "общий анализ крови"}) is True


def test_nonbookable_walkin_necessity_doctor_context_is_false():
    """Регресс: тот же вопрос в doctor-контексте → НЕ walk-in (к врачу запись нужна)."""
    assert detect_nonbookable_walkin_intent("Надо записываться?", {"specialty": "кардиолог"}) is False


def test_nonbookable_walkin_necessity_no_context_unchanged():
    """Без контекста поведение не меняется (нет лаб-контекста → не walk-in)."""
    assert detect_nonbookable_walkin_intent("Надо записываться?", {}) is False


def test_deterministic_rule_decision_necessity_lab_context_walkin_not_appointment():
    d = run(deterministic_rule_decision("Надо записываться?", {"test_name": "ОАК"}))
    assert d is not None
    assert d.label == "ADDRESS", f"lab-context necessity wrongly routed to {d.label}"
    assert d.label != "APPOINTMENT"
    assert "rule_nonbookable_walkin" in d.flags


def test_deterministic_rule_decision_real_appointment_unaffected():
    """Анти-регресс: настоящая запись к врачу по-прежнему APPOINTMENT."""
    d = run(deterministic_rule_decision("записаться к кардиологу", {}))
    assert d is not None
    assert d.label == "APPOINTMENT"
