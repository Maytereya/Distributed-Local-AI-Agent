"""LLM-различитель: справочный вопрос о СРОКАХ результата vs запрос своего результата.

Прод #675: «Результат флюорографии отдают сразу?» / «Через сколько готов
результат?» → бот просил фамилию/год/код (result-lookup), хотя пациент
спрашивал про СРОКИ. Наблюдение владельца: 95% «результат» = свои результаты,
справка о сроках редка и её даёт оператор (сроки достоверно не знаем —
выдумывать нельзя).

Fail-safe В СТОРОНУ LOOKUP: валидатор отклоняет в «сроки» (True) ТОЛЬКО на
уверенном «TIMING» от LLM; при сомнении/сбое/kill-switch → False (дефолт
lookup, не штрафуем 95%).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import _result_timing_validator as rtv


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_and_enable(monkeypatch):
    rtv._CACHE.clear()
    monkeypatch.setattr(rtv, "_ENABLED", True)
    yield
    rtv._CACHE.clear()


def _mock_llm(monkeypatch, answer):
    async def fake_generate(prompt, *, timeout_s=None, queue_timeout_ms=None):
        _ = prompt, timeout_s, queue_timeout_ms
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(rtv, "generate_text", fake_generate)


def test_timing_question_detected(monkeypatch):
    _mock_llm(monkeypatch, "TIMING")
    assert run(rtv.is_result_timing_question("Результат флюорографии отдают сразу?")) is True


def test_lookup_stays_lookup(monkeypatch):
    _mock_llm(monkeypatch, "LOOKUP")
    assert run(rtv.is_result_timing_question("хочу узнать результаты анализов")) is False


def test_kill_switch_off_is_lookup_without_llm(monkeypatch):
    monkeypatch.setattr(rtv, "_ENABLED", False)
    called = {"n": 0}

    async def fake_generate(*a, **k):
        called["n"] += 1
        return "TIMING"

    monkeypatch.setattr(rtv, "generate_text", fake_generate)
    assert run(rtv.is_result_timing_question("отдают сразу?")) is False
    assert called["n"] == 0


def test_llm_error_fails_safe_to_lookup(monkeypatch):
    _mock_llm(monkeypatch, RuntimeError("llm down"))
    assert run(rtv.is_result_timing_question("через сколько готов результат")) is False


def test_llm_garbage_fails_safe_to_lookup(monkeypatch):
    _mock_llm(monkeypatch, "хм, наверное про сроки")
    assert run(rtv.is_result_timing_question("когда готов")) is False


def test_cached(monkeypatch):
    calls = {"n": 0}

    async def fake_generate(*a, **k):
        calls["n"] += 1
        return "TIMING"

    monkeypatch.setattr(rtv, "generate_text", fake_generate)
    run(rtv.is_result_timing_question("отдают сразу?"))
    run(rtv.is_result_timing_question("отдают сразу?"))
    assert calls["n"] == 1
