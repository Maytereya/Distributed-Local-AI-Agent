"""Ход без данных: свободный текст LLM не пишется (L-05; BUG-2026-10-08-FREE-TEXT-ON-EMPTY-DATA).

Класс 4 журнала: данных нет, а модель пишет «клиника не занимается больничными», «не
предоставляем выезд». На трафике 08.10 — 28% провальных диалогов с 03.09.

Инварианты класса (по полному тексту ответа):
- на ходу без данных свободный текст LLM не вызывается вовсе;
- вид реплики выбирает LLM из закрытого списка, текст печатает код: благодарность, прощание,
  «ок», вопрос о боте — шаблон; бессмыслица — «уточните»; вопрос — честный оффер оператора,
  «да» → перевод;
- LLM не ответила — нейтральная фраза без утверждений о клинике;
- есть данные — путь свободного текста прежний; выключатель снят — откат на свободный текст.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router import orchestrator, renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.services import Services
from messengers_router.services import _empty_evidence_reply as reply
from messengers_router.text_templates import INTRO_TEXT, LOW_CONF_CLARIFY_TEXT

_ROOT = Path(__file__).resolve().parents[1]
# Настоящий различитель — до autouse-подмены `_hermetic_empty_evidence_reply` в conftest.
_REAL_CLASSIFY = reply.classify_empty_evidence_turn
_FORBIDDEN = ("не оказыва", "не занимается", "не предоставля", "нет информации", "не проводит")


@pytest.fixture
def world(monkeypatch):
    """NLU говорит «прочее», инструменты ничего не нашли, свободный текст LLM — шпион."""

    free_text_calls: list[str] = []
    evidence_box = {"evidence": Evidence()}

    async def fake_nlu(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.6, entities={}, flags={"rule_none"}),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return evidence_box["evidence"]

    async def spy_stream(text, *_args, **_kwargs):
        free_text_calls.append(text)
        yield "Клиника не оказывает данную услугу."

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_nlu)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )
    monkeypatch.setattr(renderer, "render_stream", spy_stream)
    return free_text_calls, evidence_box


def _kind_is(monkeypatch, kind):
    async def classify(_text):
        return kind

    monkeypatch.setattr(reply, "classify_empty_evidence_turn", classify)


def _say(text, state, memory):
    async def go():
        services = Services()
        services.ensure_background_refresh_started = lambda: None
        out = [e async for e in router_mod.patient_routing_stream(text, state, services, memory)]
        return "".join(e.text for e in out if e.text), any(e.handoff for e in out)

    return asyncio.run(go())


@pytest.mark.parametrize(
    ("kind", "text", "expected"),
    [
        ("THANKS", "спасибо", orchestrator.EMPTY_EVIDENCE_REPLIES["THANKS"]),
        ("BYE", "до свидания", orchestrator.EMPTY_EVIDENCE_REPLIES["BYE"]),
        ("ACK", "понятно", orchestrator.EMPTY_EVIDENCE_REPLIES["ACK"]),
        ("ABOUT_BOT", "что ты умеешь?", INTRO_TEXT),
        ("NOISE", "????", LOW_CONF_CLARIFY_TEXT),
        (None, "кто директор клиники", orchestrator.EMPTY_EVIDENCE_NEUTRAL),
    ],
)
def test_empty_evidence_turn_is_template_not_free_text(world, monkeypatch, kind, text, expected):
    free_text_calls, _ = world
    _kind_is(monkeypatch, kind)

    answer, handoff = _say(text, SessionState(session_id=f"l05-{kind}"), MemoryStore())

    assert answer == expected
    assert handoff is False
    assert free_text_calls == []
    for banned in _FORBIDDEN:
        assert banned not in answer.lower()


@pytest.mark.parametrize(
    "question",
    ["можно ли пройти медосмотр для медкнижки?", "Что делать если потеряла бланк", "Электромиография", "Капельницу"],
)
def test_question_without_data_offers_operator_and_yes_hands_off(world, monkeypatch, question):
    free_text_calls, _ = world
    _kind_is(monkeypatch, "QUESTION")
    state, memory = SessionState(session_id=f"l05-q-{question}"), MemoryStore()

    answer, handoff = _say(question, state, memory)

    assert answer == orchestrator.EMPTY_EVIDENCE_REPLIES["QUESTION"]
    assert handoff is False
    assert free_text_calls == []
    _, handoff = _say("да", state, memory)
    assert handoff is True


def test_turn_with_data_keeps_free_text_path(world, monkeypatch):
    free_text_calls, evidence_box = world
    evidence_box["evidence"] = Evidence(items={"main_index": {"documents": [{"title": "Справка 086/у"}]}})
    asked: list[str] = []

    async def classify(text):
        asked.append(text)
        return "QUESTION"

    monkeypatch.setattr(reply, "classify_empty_evidence_turn", classify)

    _say("нужна справка 086", SessionState(session_id="l05-data"), MemoryStore())

    assert asked == []
    assert free_text_calls == ["нужна справка 086"]


def test_kill_switch_restores_free_text(world, monkeypatch):
    free_text_calls, _ = world
    monkeypatch.setattr(reply, "enabled", lambda: False)

    _say("кто директор клиники", SessionState(session_id="l05-off"), MemoryStore())

    assert free_text_calls == ["кто директор клиники"]


# --- различитель: строгий разбор, кэш, копии промпта ------------------------------------

def _generate_answers(monkeypatch, answer):
    prompts: list[str] = []

    async def fake_generate_text(prompt, **_kwargs):
        prompts.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(reply, "_CACHE", {})
    monkeypatch.setattr(reply, "generate_text", fake_generate_text)
    return prompts


@pytest.mark.parametrize(
    ("answer", "kind"),
    [("QUESTION", "QUESTION"), ("thanks.", "THANKS"), ("«ABOUT_BOT»\nпотому что", "ABOUT_BOT"), ("Не знаю", None), ("", None), (TimeoutError(), None)],
)
def test_classifier_parses_strictly(monkeypatch, answer, kind):
    prompts = _generate_answers(monkeypatch, answer)

    assert asyncio.run(_REAL_CLASSIFY("Электромиография")) == kind
    assert "«Электромиография»" in prompts[0]


def test_classifier_caches_and_empty_text_is_noise(monkeypatch):
    prompts = _generate_answers(monkeypatch, "ACK")

    for text in ("Понятно", "  понятно "):
        assert asyncio.run(_REAL_CLASSIFY(text)) == "ACK"
    assert asyncio.run(_REAL_CLASSIFY("   ")) == "NOISE"
    assert len(prompts) == 1


def test_prompt_copies_are_identical():
    bundle = (_ROOT / "messengers_router" / "prompts" / "empty_evidence_kind.txt").read_text(encoding="utf-8")
    host = (_ROOT / "app_data" / "prompts" / "mr_empty_evidence_kind.txt").read_text(encoding="utf-8")
    assert bundle == host
    assert "<<TEXT>>" in bundle
