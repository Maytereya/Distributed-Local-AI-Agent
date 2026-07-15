"""Интеграция различителя сроков результата в test_result_status (прод #675).

Справочный вопрос о сроках (валидатор→TIMING) при отсутствии данных пациента →
honest handoff «сроки подскажет оператор», без просьбы фамилии/года. Запрос
своего результата (валидатор→LOOKUP, дефолт) → обычный clarify «укажите данные».
"""

from __future__ import annotations

import asyncio

from messengers_router import evidence_keys as ek
from messengers_router.mess_types import Evidence
from messengers_router.response_builder import build_test_result_response
from messengers_router.services import lab_tests as lab
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


def test_timing_question_returns_operator_handoff(monkeypatch):
    async def _timing(_text):
        return True

    monkeypatch.setattr(lab, "is_result_timing_question", _timing)

    payload = run(Services().test_result_status("Результат флюорографии отдают сразу?", {}))
    assert payload.get("timing_to_operator") is True
    assert "missing_fields" not in payload, "не просим данные при справке о сроках"

    ev = Evidence()
    ev.put(ek.TEST_RESULT_STATUS, payload)
    resp = build_test_result_response("TEST_RESULT", ev)
    assert resp is not None and resp.handoff is True
    low = resp.text.lower()
    assert "оператор" in low and "срок" in low
    assert "фамили" not in low, "не просим фамилию/год"


def test_lookup_stays_clarify(monkeypatch):
    """Дефолт (валидатор LOOKUP) — обычный clarify «укажите данные», не оператор."""
    async def _lookup(_text):
        return False

    monkeypatch.setattr(lab, "is_result_timing_question", _lookup)

    payload = run(Services().test_result_status("хочу узнать результаты анализов", {}))
    assert payload.get("timing_to_operator") is not True
    assert payload.get("missing_fields")

    ev = Evidence()
    ev.put(ek.TEST_RESULT_STATUS, payload)
    resp = build_test_result_response("TEST_RESULT", ev)
    assert resp is not None and resp.handoff is False
    assert "фамили" in resp.text.lower()


def test_validator_not_called_when_data_present(monkeypatch):
    """Анти-over-trigger: данные пациента есть → lookup, валидатор не зовётся."""
    called = {"n": 0}

    async def _spy(_text):
        called["n"] += 1
        return True

    monkeypatch.setattr(lab, "is_result_timing_question", _spy)

    payload = run(Services().test_result_status(
        "результат", {"surname": "Иванов", "year": "1985", "filial": "Самара", "number": "123"}
    ))
    assert called["n"] == 0, "при полных данных валидатор сроков не нужен"
    assert payload.get("timing_to_operator") is not True
