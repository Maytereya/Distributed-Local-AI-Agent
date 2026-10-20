"""Тень LLM рядом с правилом темы (ход (б), решения владельца 20.10 (б) 1, 3; STEP-2026-10-21-B-TOPIC-SHADOW).

Инвариант класса: под `topic_shadow_llm = off` ход не зовёт тень и не пишет её полей; под `on` на ходе, где тему решило
правило `legacy_v2`, тень решает тему тем же LLM-классификатором вне бюджета хода (`llm[]` — `shadow: true`, сайт
`classifier_shadow`, `nlu_llm` хода остаётся false), пишет `tu_topic_shadow` / `tu_topic_shadow_rule` /
`tu_topic_shadow_agree`, а ответ хода совпадает с ответом под `off` побайтно — и при согласии, и при несогласии, и при
сбое LLM. На ходе, где тему решала LLM (правила нет), тени нет. Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime, topic_shadow
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_CLASSIFIER_MARK = "Ты классификатор запросов пациента"


def _answer(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


def _turn(world_dir, monkeypatch, reply: str, *, shadow: str, answer: Any) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    async def run() -> str:
        req = MessengerGenerateRequest(session_id=f"shadow-{shadow}-{reply}", text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        await topic_shadow.settle(5.0)
        return str(body.get("text") or "")

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", "legacy_v2", raising=False)
        patch.setattr(cfg, "MR_TOPIC_SHADOW_LLM", shadow, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM({_CLASSIFIER_MARK: answer}))
        choice._CACHE.clear()
        text = asyncio.run(run())
        tel = world.take_telemetry() or {}
    return {"text": text, "telemetry": tel}


_RULE_REPLIES = ("сколько стоит ферритин", "какой у вас адрес", "хочу записаться к кардиологу")


@pytest.mark.parametrize("reply", _RULE_REPLIES)
def test_off_has_no_shadow(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _turn(mini_prod_world, monkeypatch, reply, shadow="off", answer=_answer("OTHER"))["telemetry"]
    assert not any(k.startswith("tu_topic_shadow") for k in tel), tel
    assert not any(call.get("site") == "classifier_shadow" for call in tel.get("llm") or []), tel


@pytest.mark.parametrize("llm_label", ["PRICE", "PREPARE"])
@pytest.mark.parametrize("reply", _RULE_REPLIES)
def test_shadow_on_rule_turn_writes_verdict_and_keeps_answer(reply, llm_label, mini_prod_world, monkeypatch):  # noqa: F811
    off = _turn(mini_prod_world, monkeypatch, reply, shadow="off", answer=_answer(llm_label))
    on = _turn(mini_prod_world, monkeypatch, reply, shadow="on", answer=_answer(llm_label))
    tel = on["telemetry"]
    assert tel.get("nlu_src") == "rule", tel  # корпус: ходы, где тему решило правило
    assert on["text"] == off["text"]
    assert tel.get("tu_topic_shadow") == llm_label, tel
    assert tel.get("tu_topic_shadow_rule") == tel.get("tu_label_nlu"), tel
    assert tel.get("tu_topic_shadow_agree") is (llm_label == tel.get("tu_label_nlu")), tel
    shadow_calls = [c for c in tel.get("llm") or [] if c.get("site") == "classifier_shadow"]
    assert shadow_calls and all(c.get("shadow") for c in shadow_calls), tel
    assert tel.get("nlu_llm") is False, tel


@pytest.mark.parametrize("reply", _RULE_REPLIES)
def test_shadow_failure_keeps_answer(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off = _turn(mini_prod_world, monkeypatch, reply, shadow="off", answer=_down)
    on = _turn(mini_prod_world, monkeypatch, reply, shadow="on", answer=_down)
    assert on["text"] == off["text"]
    assert on["telemetry"].get("tu_topic_shadow") in {"unavailable", "error", "invalid"}, on["telemetry"]
    assert "tu_topic_shadow_agree" not in on["telemetry"]


def test_no_shadow_when_llm_decided_topic(mini_prod_world, monkeypatch):  # noqa: F811
    tel = _turn(mini_prod_world, monkeypatch, "а вот у меня такой вопрос насчёт дочки", shadow="on", answer=_answer("OTHER"))["telemetry"]
    assert tel.get("nlu_src") == "llm", tel
    assert "tu_topic_shadow" not in tel, tel
