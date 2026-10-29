"""Общие помощники тестов хода через эндпоинт с подделкой LLM настоящего типа (`llm_runtime._OLLAMA_CLIENT`).

Перенесены 29.10 из `tests/test_llm_primary_rules_fallback_class.py` вместе со снятием движка NLU `llm_primary` (К2 карты
мёртвого кода): движок один — `legacy_v2`, поэтому у `_turn` больше нет параметров движка и запасного пути правил.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.test_booking_slots_by_meaning import _LLM

# Начало промпта LLM-классификатора темы (`classifier._build_classify_prompt`) — ключ подделки LLM.
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


def _turn(world_dir, monkeypatch, reply: str, *, answer, spans: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Один ход через эндпоинт; классификатор отвечает `answer`, прочие места LLM — «недоступна».

    :param spans: фразы-услуги для разметки хода (`turn_spans_llm = on`) — оракул `tests/spans_oracle.py`; None (по умолчанию) —
        разметка недоступна, как всё прочее
    """

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=f"turn-{reply}", text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        dbg = (body.get("state_update") or {}).get("debug") or {}
        return {"text": str(body.get("text") or ""), "decision": dbg.get("decision") or {}}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        client = _LLM({_CLASSIFIER_MARK: answer})
        if spans is not None:
            from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed

            client = SpansOracle(spans, inner=client)
            keep_llm_breaker_closed(patch)
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", client)
        choice._CACHE.clear()
        out = asyncio.run(run())
        out["telemetry"] = world.take_telemetry() or {}
    # отладочное решение есть не на каждой ветке (у уточнения его нет) — метка и подпись NLU из строки телеметрии хода
    out["label"] = out["decision"].get("label") or out["telemetry"].get("label")
    out["source"] = out["decision"].get("source") or out["telemetry"].get("decision_source")
    return out
