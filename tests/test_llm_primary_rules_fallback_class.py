"""`llm_primary` при сбое разбора LLM — запасной путь правил (BUG-2026-10-20-LLM-PRIMARY-NO-RULES-FALLBACK, ход (и)).

Решение владельца 20.10, п. 2: при `unavailable` / `invalid` разбора LLM ход решают правила, как `legacy_v2`, с отметкой в
телеметрии; решает LLM всегда, когда отвечает; код — под выключателем `[MESSENGER_ROUTER] llm_primary_rules_fallback`.
До фикса под `llm_primary` при недоступной LLM даже «сколько стоит ферритин» получал «Уточните, что именно вам нужно»,
а `legacy_v2` в той же ситуации отвечал ценой правил.

Инвариант: под `llm_primary` с `llm_primary_rules_fallback = on` при любом сбое разбора (недоступна, таймаут, не JSON,
JSON вне контракта) ответ хода совпадает с ответом `legacy_v2` при том же сбое, служебных кодов нет, срочное остаётся
срочным; в телеметрии хода — `nlu_fallback = rules` и вид сбоя. Выключатель off — как было; LLM ответила внятно —
решает она. Через обработчик эндпоинта, LLM — подделка настоящего типа (`llm_runtime._OLLAMA_CLIENT`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from messengers_router.recovery_policy import CLARIFY_REASON_CODES
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_CLASSIFIER_MARK = "Ты классификатор запросов пациента"


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


def _timeout(_prompt: str) -> str:
    raise asyncio.TimeoutError()


# Вид сбоя → (ответ подделки классификатору, вид в телеметрии хода).
_FAILURES = {
    "unavailable": (_down, "unavailable"),
    "timeout": (_timeout, "unavailable"),
    "not_json": ("не json", "invalid"),
    "off_contract": (json.dumps({"label": "ЦЕНА_АНАЛИЗА", "confidence": 0.9, "entities": {}}), "invalid"),
}
_REPLIES = {
    "price": "сколько стоит ферритин",
    "booking": "хочу записаться к кардиологу",
    "address": "какой у вас адрес",
    "urgent": "у меня сильная боль в груди и мне трудно дышать",
}


def _turn(world_dir, monkeypatch, reply: str, *, engine: str, fallback: str, answer) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=f"fallback-{engine}-{fallback}-{reply}", text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        dbg = (body.get("state_update") or {}).get("debug") or {}
        return {"text": str(body.get("text") or ""), "decision": dbg.get("decision") or {}}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", engine, raising=False)
        patch.setattr(cfg, "MR_LLM_PRIMARY_RULES_FALLBACK", fallback, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM({_CLASSIFIER_MARK: answer}))
        choice._CACHE.clear()
        out = asyncio.run(run())
        out["telemetry"] = world.take_telemetry() or {}
    # отладочное решение есть не на каждой ветке (у уточнения его нет) — метка и подпись NLU из строки телеметрии хода
    out["label"] = out["decision"].get("label") or out["telemetry"].get("label")
    out["source"] = out["decision"].get("source") or out["telemetry"].get("decision_source")
    return out


@pytest.mark.parametrize("failure", sorted(_FAILURES))
@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_llm_failure_turn_is_decided_by_rules_like_legacy(failure, reply, mini_prod_world, monkeypatch):  # noqa: F811
    answer, kind = _FAILURES[failure]
    text = _REPLIES[reply]

    primary = _turn(mini_prod_world, monkeypatch, text, engine="llm_primary", fallback="on", answer=answer)
    legacy = _turn(mini_prod_world, monkeypatch, text, engine="legacy_v2", fallback="off", answer=answer)

    assert primary["text"] == legacy["text"], (primary["text"], legacy["text"])
    assert primary["label"] == legacy["label"]
    assert not any(code in primary["text"] for code in CLARIFY_REASON_CODES), primary["text"]
    if primary["telemetry"].get("nlu_llm"):  # guardrail мог решить ход без вызова LLM — тогда и сбоя не было
        assert (primary["telemetry"].get("nlu_fallback"), primary["telemetry"].get("nlu_fail")) == ("rules", kind)


@pytest.mark.parametrize("failure", sorted(_FAILURES))
def test_urgent_stays_urgent_when_llm_fails(failure, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _turn(mini_prod_world, monkeypatch, _REPLIES["urgent"], engine="llm_primary", fallback="on", answer=_FAILURES[failure][0])

    assert turn["label"] == "URGENT", turn
    assert turn["telemetry"].get("handoff"), turn


def test_price_of_named_service_survives_llm_outage(mini_prod_world, monkeypatch):  # noqa: F811
    """Исходный симптом: при недоступной LLM «сколько стоит ферритин» — цена, а не меню «Уточните, что именно…»."""

    turn = _turn(mini_prod_world, monkeypatch, _REPLIES["price"], engine="llm_primary", fallback="on", answer=_down)

    assert turn["label"] == "PRICE", turn
    assert "руб" in turn["text"].lower(), turn["text"]


@pytest.mark.parametrize("failure", sorted(_FAILURES))
def test_switch_off_keeps_llm_primary_as_it_was(failure, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _turn(mini_prod_world, monkeypatch, _REPLIES["price"], engine="llm_primary", fallback="off", answer=_FAILURES[failure][0])

    assert "nlu_fallback" not in turn["telemetry"], turn["telemetry"]
    assert turn["label"] != "PRICE", turn  # без запасного пути разбор сбоя — OTHER, как до фикса


def test_garbage_switch_value_means_off(mini_prod_world, monkeypatch):  # noqa: F811
    turn = _turn(mini_prod_world, monkeypatch, _REPLIES["price"], engine="llm_primary", fallback="да", answer=_down)

    assert "nlu_fallback" not in turn["telemetry"], turn["telemetry"]


def test_legacy_engine_ignores_the_switch(mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(mini_prod_world, monkeypatch, _REPLIES["price"], engine="legacy_v2", fallback="on", answer=_down)
    off = _turn(mini_prod_world, monkeypatch, _REPLIES["price"], engine="legacy_v2", fallback="off", answer=_down)

    assert on["text"] == off["text"]
    assert "nlu_fallback" not in on["telemetry"]


@pytest.mark.parametrize("reply", ["price", "booking"])
def test_clear_llm_answer_decides_the_turn(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """LLM ответила внятно — решает она, даже если правила думают иначе: запасной путь только для сбоя."""

    answer = json.dumps({"label": "TEST_ASSIST", "confidence": 0.95, "context_action": "new_topic", "entities": {}})
    turn = _turn(mini_prod_world, monkeypatch, _REPLIES[reply], engine="llm_primary", fallback="on", answer=answer)

    assert turn["label"] == "TEST_ASSIST", turn
    assert turn["source"] == "llm_primary", turn
    assert "nlu_fallback" not in turn["telemetry"], turn["telemetry"]
