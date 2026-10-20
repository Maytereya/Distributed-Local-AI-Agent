"""Служебные коды уточнения разбора LLM не доходят до пациента (BUG-2026-10-20-CLARIFY-CODE-LEAK, ход (и), шаг 1).

Под `nlu_engine = llm_primary` разбор LLM ставит `clarify_reason` кодом контракта («slot_request»,
«intent_disambiguation»), а ворота уточнения (`orchestrator.clarify_gate`) печатали `clarify_reason` пациенту как есть:
на «хочу записаться» бот отвечал «slot_request», при сбое LLM — «intent_disambiguation». Правила кладут в то же поле готовый
текст (больничный, сайт, «узнать анализы») — его печатаем как есть.

Инвариант: текст ответа никогда не совпадает со служебным кодом и не содержит его — ни при каком ответе LLM, ни при её
сбое; готовый текст правил — без изменений. Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from messengers_router.mess_types import RouteDecision
from messengers_router.policies import SICK_LEAVE_HANDOFF_TEXT
from messengers_router.recovery_policy import CLARIFY_REASON_CODES, patient_clarify_text
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_price_turns_golden import mini_prod_world, synonym_fork  # noqa: F401 — фикстуры

_CLASSIFIER_MARK = "Ты классификатор запросов пациента"


def _button_dialog(services, monkeypatch, turns, *, switches: dict[str, str], llm: _LLM) -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта (перенесён из `test_booking_slots_by_meaning`: ветка `pending_buttons` сняла путь
    `slot_reply` вместе с его тестами, слияние деплоя 11)."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg

    session = f"slot-{json.dumps(switches)}-{turns[-1]}"

    async def run() -> list[dict[str, Any]]:
        memory, out = MemoryStore(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            dbg = (body.get("state_update") or {}).get("debug") or {}
            decision = dbg.get("decision") or {}
            out.append({"text": str(body.get("text") or ""), "label": decision.get("label"), "flags": decision.get("flags") or []})
        return out

    with monkeypatch.context() as patch:
        for key, value in switches.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        choice._CACHE.clear()
        return asyncio.run(run())


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


_CLASSIFIER_ANSWERS = {
    "slot_request": json.dumps({"label": "APPOINTMENT", "confidence": 0.9, "context_action": "new_topic", "entities": {}}),
    "low_confidence": json.dumps({"label": "OTHER", "confidence": 0.2, "context_action": "new_topic", "entities": {}}),
    "llm_says_code": json.dumps(
        {"label": "PRICE", "confidence": 0.8, "context_action": "new_topic", "entities": {}, "clarify_needed": True, "clarify_reason": "context_repair"}
    ),
    "invalid": "не json",
    "down": _down,
}
_REPLIES = ["хочу записаться", "мне бы узнать насчёт анализов и врача", "сколько стоит", "здравствуйте, подскажите"]


def _turn(world_dir, monkeypatch, reply: str, *, engine: str, answer) -> dict:
    from messengers_router.services import Services

    with ev_world.FrozenWorld(world_dir, "day"):
        return _button_dialog(Services(), monkeypatch, [reply], switches={"MR_NLU_ENGINE": engine}, llm=_LLM({_CLASSIFIER_MARK: answer}))[-1]


@pytest.mark.parametrize("case", sorted(_CLASSIFIER_ANSWERS))
@pytest.mark.parametrize("reply", _REPLIES)
def test_clarify_code_never_reaches_the_patient(case, reply, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _turn(mini_prod_world, monkeypatch, reply, engine="llm_primary", answer=_CLASSIFIER_ANSWERS[case])

    text = turn["text"]
    assert text.strip(), turn
    assert not any(code in text for code in CLARIFY_REASON_CODES), text


def test_rule_text_in_the_reason_is_printed_as_is():
    decision = RouteDecision(label="OTHER", confidence=1.0, clarify_needed=True, clarify_reason=SICK_LEAVE_HANDOFF_TEXT)

    assert patient_clarify_text(decision, "нужен больничный") == SICK_LEAVE_HANDOFF_TEXT


@pytest.mark.parametrize(
    ("reason", "slots", "candidates", "expected"),
    [
        ("slot_request", ["service_name"], [], "название услуги"),
        ("intent_disambiguation", [], ["APPOINTMENT", "PRICE"], "1. запись к врачу"),
        ("intent_disambiguation", [], ["TEST_ASSIST"], "вас интересует подбор анализов?"),
        ("", [], [], "выберите вариант"),
    ],
)
def test_code_becomes_a_question(reason, slots, candidates, expected):
    label = "PRICE" if reason == "slot_request" else "OTHER"
    decision = RouteDecision(label=label, confidence=0.3, clarify_needed=True, clarify_reason=reason, clarify_slots=slots, intent_candidates=candidates)

    text = patient_clarify_text(decision, "непонятная реплика")
    assert expected.lower() in text.lower() and text not in CLARIFY_REASON_CODES, text
