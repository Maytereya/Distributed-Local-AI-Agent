"""LLM-валидатор слота услуги: отличает услугу от команды/навигации.

Аудит 2026-07-22 (класс `slot_accepts_command_as_data`): в активной записи бот
спросил услугу, пациент ответил командой «Стоп»/«Поменяй»/«Заново» → форма
`_SERVICE_SINGLE_WORD_RE` (любое слово ≥4 букв не из стоп-листа) приняла как
услугу → «Запись: … Поменяй … Подтверждаете?». Решение владельца — LLM-first
валидатор (не бесконечный стоп-лист), тот же паттерн, что для ФИО.

Инварианты (класс):
1. LLM уверенно «OTHER» (команда/навигация) → отвергнуть (False).
2. LLM «SERVICE» → принять (True).
3. FAIL-OPEN: kill-switch off / таймаут / сбой / мусор от LLM → True (принять по
   форме, текущее поведение). Запись НИКОГДА не блокируется валидатором.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import _service_slot_validator as ssv


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_cache_and_enable(monkeypatch):
    ssv._CACHE.clear()
    monkeypatch.setattr(ssv, "_ENABLED", True)
    yield
    ssv._CACHE.clear()


def _mock_llm(monkeypatch, answer):
    async def fake_generate(prompt, *, timeout_s=None, queue_timeout_ms=None):
        _ = prompt, timeout_s, queue_timeout_ms
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(ssv, "generate_text", fake_generate)


@pytest.mark.parametrize("cmd", ["Стоп", "Поменяй", "Заново", "Другое", "Забудь", "Назад"])
def test_llm_says_other_rejects_command_words(monkeypatch, cmd):
    _mock_llm(monkeypatch, "OTHER")
    assert run(ssv.is_service_name_reply(cmd)) is False


def test_llm_says_service_accepts(monkeypatch):
    _mock_llm(monkeypatch, "SERVICE")
    assert run(ssv.is_service_name_reply("торакоцентез")) is True


def test_kill_switch_off_accepts_without_llm(monkeypatch):
    monkeypatch.setattr(ssv, "_ENABLED", False)
    called = {"n": 0}

    async def fake_generate(*a, **k):
        called["n"] += 1
        return "OTHER"

    monkeypatch.setattr(ssv, "generate_text", fake_generate)
    assert run(ssv.is_service_name_reply("Стоп")) is True
    assert called["n"] == 0, "kill-switch off — LLM не зовём"


def test_llm_timeout_fails_open_accepts(monkeypatch):
    _mock_llm(monkeypatch, asyncio.TimeoutError())
    assert run(ssv.is_service_name_reply("Стоп")) is True


def test_llm_error_fails_open_accepts(monkeypatch):
    _mock_llm(monkeypatch, RuntimeError("llm down"))
    assert run(ssv.is_service_name_reply("Поменяй")) is True


def test_llm_garbage_answer_fails_open_accepts(monkeypatch):
    _mock_llm(monkeypatch, "хм, возможно услуга а может команда")
    assert run(ssv.is_service_name_reply("Стоп")) is True


def test_result_is_cached(monkeypatch):
    calls = {"n": 0}

    async def fake_generate(*a, **k):
        calls["n"] += 1
        return "OTHER"

    monkeypatch.setattr(ssv, "generate_text", fake_generate)
    run(ssv.is_service_name_reply("Поменяй"))
    run(ssv.is_service_name_reply("Поменяй"))
    assert calls["n"] == 1, "повторная валидация той же реплики — из кэша"


def test_prompt_loads_from_both_locations():
    """Dual-location: промпт есть и в bundle, и в host (mr_)."""
    from messengers_router.prompt_registry import _PROMPTS_DIR, _RUNTIME_PROMPTS_DIR

    bundle = _PROMPTS_DIR / "service_slot_validator.txt"
    host = _RUNTIME_PROMPTS_DIR / "mr_service_slot_validator.txt"
    for path in (bundle, host):
        assert path.exists(), f"prompt missing: {path}"
        assert "<<TEXT>>" in path.read_text(encoding="utf-8"), f"placeholder missing in {path}"
