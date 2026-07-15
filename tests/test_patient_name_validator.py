"""LLM-валидатор слота ФИО: отличает имя от команды/коррекции.

Прод #668 (флюорография-запись): бот спросил ФИО, пациент написал «Время
изменени» (хотел сменить время) → форма `_looks_like_patient_fio` приняла как
ФИО → «Запись: Время Изменени…». Форма проверяет только 2+ кириллических слова
не из стоп-листа; семантику (имя vs команда) не различает. Решение владельца —
LLM-first валидатор (не бесконечный стоп-лист).

Инварианты (класс):
1. LLM уверенно «OTHER» (команда/коррекция) → отвергнуть (False).
2. LLM «FIO» → принять (True).
3. FAIL-OPEN: kill-switch off / таймаут / сбой / мусор от LLM → True (принять
   по форме, текущее поведение). Запись НИКОГДА не блокируется валидатором.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import _patient_name_validator as pnv


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_cache_and_enable(monkeypatch):
    pnv._CACHE.clear()
    monkeypatch.setattr(pnv, "_ENABLED", True)
    yield
    pnv._CACHE.clear()


def _mock_llm(monkeypatch, answer):
    async def fake_generate(prompt, *, timeout_s=None, queue_timeout_ms=None):
        _ = prompt, timeout_s, queue_timeout_ms
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(pnv, "generate_text", fake_generate)


def test_llm_says_other_rejects(monkeypatch):
    _mock_llm(monkeypatch, "OTHER")
    assert run(pnv.is_patient_name_reply("Время изменени")) is False


def test_llm_says_fio_accepts(monkeypatch):
    _mock_llm(monkeypatch, "FIO")
    assert run(pnv.is_patient_name_reply("Иванов Иван Иванович")) is True


def test_kill_switch_off_accepts_without_llm(monkeypatch):
    monkeypatch.setattr(pnv, "_ENABLED", False)
    called = {"n": 0}

    async def fake_generate(*a, **k):
        called["n"] += 1
        return "OTHER"

    monkeypatch.setattr(pnv, "generate_text", fake_generate)
    assert run(pnv.is_patient_name_reply("Время изменени")) is True
    assert called["n"] == 0, "kill-switch off — LLM не зовём"


def test_llm_timeout_fails_open_accepts(monkeypatch):
    _mock_llm(monkeypatch, asyncio.TimeoutError())
    assert run(pnv.is_patient_name_reply("Иванов Иван")) is True


def test_llm_error_fails_open_accepts(monkeypatch):
    _mock_llm(monkeypatch, RuntimeError("llm down"))
    assert run(pnv.is_patient_name_reply("Петров Пётр")) is True


def test_llm_garbage_answer_fails_open_accepts(monkeypatch):
    _mock_llm(monkeypatch, "не знаю, возможно это имя а может и нет")
    assert run(pnv.is_patient_name_reply("Сидоров Сидор")) is True


def test_result_is_cached(monkeypatch):
    calls = {"n": 0}

    async def fake_generate(*a, **k):
        calls["n"] += 1
        return "OTHER"

    monkeypatch.setattr(pnv, "generate_text", fake_generate)
    run(pnv.is_patient_name_reply("Время изменени"))
    run(pnv.is_patient_name_reply("Время изменени"))
    assert calls["n"] == 1, "повторная валидация той же реплики — из кэша"
