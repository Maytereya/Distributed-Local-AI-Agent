"""П1 на ходах `topic_by_llm`: свойство вопроса об услуге решает LLM (ход (б), STEP-2026-10-21-B-P1-FACET).

Карточка 21.10 (`n21_tbl2_cached_test`): LLM решила тему TEST_ASSIST («есть ли анализ ПЦР на аскаридоз отдельно», «до какого
времени можно сдать ХГЧ»), а строка П1 `test_assist_catalog_hit_is_price` переклеила её в голую цену — чужая цена g0811 и цена
вместо часов g0816. Инвариант класса: на ходе `topic_by_llm` с услугой из каталога П1 действует по свойству вопроса от LLM —
`price` и сбой — цена (как было), `availability` — метка LLM остаётся (без голой цены и без воронки «для какой цели»),
`where_when` — адреса (строка таблицы `p1_where_when`); без `topic_by_llm` свойство не спрашивается и П1 — как было.
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

_FACET_MARK = "что именно пациент хочет узнать об анализе"
_FACETS = {"price": 1, "availability": 2, "where_when": 3}
_REPLY = "есть ли у вас анализ на ферритин"


def _turn(world_dir, monkeypatch, *, rules: str, facet: Any) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    answers = {_CLASSIFIER_MARK: json.dumps({"label": "TEST_ASSIST", "confidence": 0.9, "context_action": "new_topic", "entities": {}})}
    if facet is not None:
        answers[_FACET_MARK] = facet if callable(facet) else json.dumps({"picks": [_FACETS[facet]]})
    fake = _LLM(answers)

    async def run() -> str:
        req = MessengerGenerateRequest(session_id=f"p1-{rules}-{facet}", text=_REPLY, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        return str(body.get("text") or "")

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", "legacy_v2", raising=False)
        patch.setattr(cfg, "MR_TOPIC_BY_LLM", rules, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
        choice._CACHE.clear()
        text = asyncio.run(run())
        tel = world.take_telemetry() or {}
    return {"text": text, "telemetry": tel, "asked_facet": _FACET_MARK in fake.calls}


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


def test_without_topic_by_llm_p1_is_unchanged(mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, rules="", facet="availability")
    assert not out["asked_facet"]
    assert out["telemetry"].get("tu_label_final") == "PRICE", out["telemetry"]  # П1: услуга найдена — цена


@pytest.mark.parametrize("facet", ["price", _down])
def test_price_or_failure_keeps_p1(facet, mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet=facet)
    assert out["asked_facet"]
    assert out["telemetry"].get("tu_label_final") == "PRICE", out["telemetry"]


def test_availability_is_not_a_bare_price(mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet="availability")
    tel = out["telemetry"]
    assert tel.get("tu_p1_facet") == "availability", tel
    assert not any(s.startswith("test_assist_catalog_hit_is_price:") for s in tel.get("tu_relabel_src") or []), tel
    assert tel.get("tu_label_final") != "PRICE", tel
    assert "цел" not in out["text"].lower(), out["text"]  # не воронка подбора «для какой цели» (П1)


def test_where_when_goes_to_addresses(mini_prod_world, monkeypatch):  # noqa: F811
    tel = _turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet="where_when")["telemetry"]
    assert tel.get("tu_label_final") == "ADDRESS", tel
    assert "p1_where_when:TEST_ASSIST>ADDRESS" in (tel.get("tu_relabel_src") or []), tel
