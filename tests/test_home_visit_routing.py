"""«Вызвать на дом» — услуга клиники, а не экстренный случай (BUG-2026-09-25-HOME-VISIT-AS-URGENT).

## Что произошло

«Вызвать на дом» → «клиника не предоставляет услугу вызова врача на дом». Выезд есть
(владелец 02.10: терапевт, дерматолог, невролог), у клиники есть скрипт «Вызов на дом
специалиста» в базе знаний.

## Механизм

Тема `service_home_visit` ловила только «вызвать врача на дом», без слова «врача» — мимо.
LLM-классификатор связал «вызвать» со скорой (у URGENT один пример — боль в груди) и
поставил URGENT. Шаблон срочного ответа включался только для URGENT от правил, URGENT от
LLM уходил в свободную генерацию, и рендер на пустых данных написал «не предоставляет».

## Решение владельца 02.10

Вызов на дом оформляет оператор. В базе знаний по теме — скрипт администратора
(«Алгоритм консультации по услуге Вызов на дом специалиста»), пересказывать его пациенту
нельзя. Тема в реестре получила стратегию `handoff`: какие темы ведёт оператор, решают
данные реестра, а не код.

## Инварианты

- `llm_safety_without_template`: URGENT — шаблоном всегда, от правил и от LLM;
- фразы вызова на дом ведут в тему, тема — сразу к оператору, своим текстом;
- у каждой темы со стратегией `handoff` есть свой текст передачи.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router import renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.services import Services
from messengers_router.policies import HANDOFF_REASON_MATRIX, handoff_message
from messengers_router.topic_registry import list_topics, match_topic

_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text",
    ["Вызвать на дом", "вызов на дом терапевта", "можно врача на дом", "нужен выезд на дом", "вызвать врача на дом"],
)
def test_home_visit_phrases_reach_the_topic(text: str):
    match = match_topic(text)
    assert match is not None and match.topic_id == "service_home_visit", match


def test_llm_urgent_is_answered_by_the_template_not_free_text(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="URGENT", confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm_safety",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    async def must_not_generate(*_args, **_kwargs):
        raise AssertionError("URGENT не должен уходить в свободную генерацию")

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(router_mod, "_env_flag", lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default))
    monkeypatch.setattr(renderer, "render_stream", must_not_generate, raising=False)

    async def go():
        state, memory = SessionState(session_id="urgent-template"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream("Вызвать на дом", state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    text, handoff = asyncio.run(go())
    assert text == renderer.render_urgent().text
    assert "не предоставляет" not in text
    assert handoff is True


@pytest.mark.parametrize(
    "path", ["messengers_router/prompts/classifier_patient.txt", "app_data/prompts/mr_classifier_patient.txt"]
)
def test_classifier_knows_home_visit_is_not_urgent(path: str):
    text = (_REPO / path).read_text(encoding="utf-8")
    assert "Текст: Вызвать на дом\nLabel: OTHER" in text
    assert "а НЕ экстренный случай" in text


@pytest.mark.parametrize("text", ["Вызвать на дом", "можно вызвать терапевта на дом?", "нужен выезд на дом"])
def test_home_visit_goes_straight_to_the_operator(monkeypatch, text: str):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.8, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def must_not_search(_plan, _state, _services):
        raise AssertionError("вызов на дом не ищет ответ в базе знаний")

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", must_not_search)
    monkeypatch.setattr(router_mod, "_env_flag", lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default))

    async def go():
        state, memory = SessionState(session_id="home-visit-handoff"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    answer, handoff = asyncio.run(go())
    assert answer == handoff_message("home_visit")
    assert handoff is True


def test_every_handoff_topic_has_its_own_message():
    handoff_topics = [
        topic for topic in list_topics() if str((topic.get("route") or {}).get("strategy") or "").lower() == "handoff"
    ]
    assert handoff_topics, "предусловие: в реестре есть тема со стратегией handoff"
    for topic in handoff_topics:
        assert topic["route"].get("message_key") in HANDOFF_REASON_MATRIX, topic["topic_id"]
