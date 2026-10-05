"""Класс `llm_safety_without_template`: метка безопасности — всегда шаблон, а не свободный ответ.

URGENT отвечал шаблоном всегда, а MEDICAL_ADVICE — только когда метку ставило правило
(short-circuit). Та же метка от LLM уходила в свободную генерацию: на «болит голова три
дня, что делать?» модель ответила фразой для непрофильных вопросов «По техническим
вопросам обратитесь к администратору клиники» (прод, 04.10). Решение владельца 04.10:
медвопрос — шаблоном всегда (BUG-2026-10-04-MEDICAL-ADVICE-FREE-TEXT).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.services import Services

_FREE_ANSWER = "По техническим вопросам обратитесь к администратору клиники."


def _route_llm_label(monkeypatch, label: str) -> None:
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label=label, confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    async def free_stream(*_args, **_kwargs):
        yield _FREE_ANSWER

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )
    monkeypatch.setattr(renderer, "render_stream", free_stream)


def _answer(text: str) -> tuple[str, bool]:
    async def go():
        state, memory = SessionState(session_id="safety-template"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return asyncio.run(go())


@pytest.mark.parametrize(
    ("label", "template"),
    [("MEDICAL_ADVICE", renderer.render_medical_advice), ("URGENT", renderer.render_urgent)],
)
def test_safety_label_from_llm_answers_with_template(monkeypatch, label, template):
    _route_llm_label(monkeypatch, label)

    text, handoff = _answer("болит голова три дня, что делать?")

    expected = template()
    assert text == expected.text
    assert handoff is expected.handoff
