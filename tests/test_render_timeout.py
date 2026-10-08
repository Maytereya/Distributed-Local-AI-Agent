"""Свободный ответ LLM ограничен бюджетом целиком (03.10).

Раньше тайм-аут 300 с стоял только на старте потока: ollama с `stream=True` отдаёт
генератор без обращения к сети, и само чтение потока не ограничивалось ничем. Пациент
мог ждать сколько угодно — например, пока LLM разбирает очередь чужих запросов.

Решение владельца 03.10: 60 с на ответ целиком (очередь, разбор промпта, генерация);
истекли — передача оператору тем же текстом, что при сбое.
"""

from __future__ import annotations

import pytest

import asyncio

from messengers_router import renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.policies import handoff_message
from messengers_router.services import Services


@pytest.fixture(autouse=True)
def _free_text_path_under_test(monkeypatch):
    """Файл проверяет механику свободного текста LLM; ход здесь без данных, а такие ходы с L-05
    (08.10) до свободного текста не доходят — правило L-05 отключаем явно."""
    from messengers_router.services import _empty_evidence_reply

    monkeypatch.setattr(_empty_evidence_reply, "enabled", lambda: False)



def _route_to_free_answer(monkeypatch, stream):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )
    monkeypatch.setattr(renderer, "render_stream", stream)


def _answer(text: str) -> tuple[str, bool]:
    async def go():
        state, memory = SessionState(session_id="render-timeout"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return asyncio.run(go())


def test_answer_over_budget_goes_to_the_operator(monkeypatch):
    async def slow_stream(*_args, **_kwargs):
        yield "Начинаю отвечать"
        await asyncio.sleep(5)  # LLM застряла: очередь или очень длинная генерация
        yield " и не заканчиваю"

    _route_to_free_answer(monkeypatch, slow_stream)
    monkeypatch.setattr(renderer, "RENDER_TIMEOUT_S", 0.05)

    text, handoff = _answer("расскажите о клинике")

    assert text == handoff_message("service_error")
    assert handoff is True


def test_answer_within_budget_is_unchanged(monkeypatch):
    async def quick_stream(*_args, **_kwargs):
        yield "Клиника «Наука» "
        yield "работает каждый день."

    _route_to_free_answer(monkeypatch, quick_stream)

    text, handoff = _answer("расскажите о клинике")

    assert "работает каждый день" in text
    assert handoff is False


def test_budget_is_sixty_seconds():
    assert renderer.RENDER_TIMEOUT_S == 60
