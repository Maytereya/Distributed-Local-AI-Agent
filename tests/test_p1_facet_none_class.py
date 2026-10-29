"""П1: свойство вопроса «ни одно» не ведёт в цену (BUG-2026-10-22-P1-FACET-NONE-AS-PRICE, выключатель `p1_facet_none_offer`).

Симптом (стенд 22.10, `topic_by_llm` + ветка `rule_address_work_hours`): g0577 «Со скольки лет берут анализ крови из вены?» → LLM
тема TEST_ASSIST, свойство вопроса «ни одно» (спрашивают возраст, не цену), а строка П1 `test_assist_catalog_hit_is_price`
переклеила в цену чужой строки («Анализ крови на аминокислоты»).

Инварианты класса (ход `topic_by_llm`, услуга найдена каталогом):
- off — как было: «ни одно» → цена (П1);
- on: «ни одно» → честный оффер оператора (метка OTHER, строка таблицы `p1_facet_none_offer`), цены нет;
- on: «цена» → цена; сбой LLM свойства → цена (как было) — «ни одно» и сбой различаются;
- on: «срок готовности», «подготовка», «просто назвал» — свои варианты полной задачи (карточка 22.10: первая версия слила их с
  «ни одно» — оффер на «Сколько готовится ОАМ?», «ТТГ»), путь — как было.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.test_booking_slots_by_meaning import _LLM
from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_p1_facet_class import _FACET_MARK, _FACETS, _down
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

# Реплика ветки `rule_test_assist` с услугой каталога; свойство вопроса решает подделка LLM.
_REPLY = "есть ли у вас анализ на ферритин"
_FULL = {"readiness": 4, "prepare": 5, "bare": 6}  # номера вариантов полной задачи `test_assist_facet_full`


def _turn(world_dir, monkeypatch, *, mode: str, facet: Any) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    answers = {_CLASSIFIER_MARK: json.dumps({"label": "TEST_ASSIST", "confidence": 0.9, "context_action": "new_topic", "entities": {}})}
    if facet == "none":
        answers[_FACET_MARK] = json.dumps({"picks": []})
    elif callable(facet):
        answers[_FACET_MARK] = facet
    elif facet in _FULL:
        answers[_FACET_MARK] = json.dumps({"picks": [_FULL[facet]]})
    else:
        answers[_FACET_MARK] = json.dumps({"picks": [_FACETS[facet]]})
    fake = _LLM(answers)

    async def run() -> str:
        req = MessengerGenerateRequest(session_id=f"p1none-{mode}-{facet}", text=_REPLY, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        return str(body.get("text") or "")

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_TOPIC_BY_LLM", "rule_test_assist", raising=False)
        patch.setattr(cfg, "MR_P1_FACET_NONE_OFFER", mode, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
        choice._CACHE.clear()
        text = asyncio.run(run())
        tel = world.take_telemetry() or {}
    return {"text": text, "telemetry": tel, "asked_facet": _FACET_MARK in fake.calls}


def test_off_none_keeps_p1_price(mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, mode="off", facet="none")
    assert out["asked_facet"]
    assert out["telemetry"].get("tu_p1_facet") == "none", out["telemetry"]
    assert out["telemetry"].get("tu_label_final") == "PRICE", out["telemetry"]


def test_on_none_is_honest_offer_not_price(mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, mode="on", facet="none")
    tel = out["telemetry"]
    assert tel.get("tu_label_final") != "PRICE", tel
    assert any(s.startswith("p1_facet_none_offer:") for s in tel.get("tu_relabel_src") or []), tel
    assert "руб" not in out["text"], out["text"]
    assert "оператор" in out["text"].lower(), out["text"]


@pytest.mark.parametrize("facet", ["price", _down])
def test_on_price_or_failure_keeps_p1(facet, mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, mode="on", facet=facet)
    tel = out["telemetry"]
    assert tel.get("tu_label_final") == "PRICE", tel
    assert tel.get("tu_p1_facet") == ("price" if facet == "price" else "failed"), tel


@pytest.mark.parametrize("facet", ["readiness", "prepare", "bare"])
def test_on_full_facets_keep_old_path(facet, mini_prod_world, monkeypatch):  # noqa: F811
    out = _turn(mini_prod_world, monkeypatch, mode="on", facet=facet)
    tel = out["telemetry"]
    assert tel.get("tu_p1_facet") == facet, tel
    assert not any(s.startswith("p1_facet_none_offer:") for s in tel.get("tu_relabel_src") or []), tel
    assert tel.get("tu_label_final") == "PRICE", tel  # как было: П1
