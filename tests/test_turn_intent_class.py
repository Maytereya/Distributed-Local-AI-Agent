"""Тема хода в понимании — ход (б), этап 1, наблюдатель (STEP-2026-10-21-B-INTENT).

Инвариант класса «понимание = итог хода»: у хода, дошедшего до NLU, тема (`turn_understanding.intent()`) построена в одной
точке сразу после NLU — с меткой NLU (`tu_label_nlu`), источником (`nlu_src`) и флагами ветки правила; её переклейки —
ровно сработавшие строки таблицы политик (`tu_relabel_src`); метка, по которой планировщик строит план, равна итоговой
метке темы (`tu_intent_mismatch` не пишется). Читатели не меняются, ответы — тоже. Через обработчик эндпоинта.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_LLM_ANSWER = json.dumps({"label": "OTHER", "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _turn(world_dir, monkeypatch, reply: str, *, engine: str) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=f"intent-{engine}-{reply}", text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        return (body.get("state_update") or {}).get("debug") or {}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", engine, raising=False)
        patch.setattr(cfg, "MR_LLM_PRIMARY_RULES_FALLBACK", "on", raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM({_CLASSIFIER_MARK: _LLM_ANSWER}))
        choice._CACHE.clear()
        debug = asyncio.run(run())
        return {"debug": debug, "telemetry": world.take_telemetry() or {}}

_REPLIES = (
    "есть ли у вас анализ на ферритин",
    "анализ на ферритин и глюкозу есть?",
    "сколько стоит ферритин",
    "мне нужно к лору",
    "хочу записаться к кардиологу",
    "какой у вас адрес",
    "где сдать ферритин без записи",
    "подготовка к анализу на ферритин",
    "а вот у меня такой вопрос насчёт дочки",
    "спасибо большое",
)


@pytest.mark.parametrize("engine", ["legacy_v2", "llm_primary"])
def test_intent_is_built_once_after_nlu_and_matches_the_plan(engine, mini_prod_world, monkeypatch):  # noqa: F811
    seen = 0
    for reply in _REPLIES:
        out = _turn(mini_prod_world, monkeypatch, reply, engine=engine)
        tel = out["telemetry"]
        assert "tu_intent_mismatch" not in tel, (reply, tel)
        if "tu_label_nlu" not in tel:
            continue
        seen += 1
        intent = out["debug"].get("intent")
        assert intent, (reply, sorted(out["debug"]))
        assert intent["label"] == tel["tu_label_nlu"], (reply, intent, tel)
        assert intent["source"] == tel.get("nlu_src", "?"), (reply, intent, tel)
        assert intent["overrides"] == (tel.get("tu_relabel_src") or []), (reply, intent, tel)
        assert intent["final"] == tel["tu_label_final"], (reply, intent, tel)
    assert seen >= 6, "корпус вырожден"
