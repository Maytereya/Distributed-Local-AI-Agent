"""Кнопка `pass` тему у LLM не переспрашивает (BUG-2026-10-24-BUTTON-PASS-TOPIC-LLM, выключатель `button_pass_rules`).

Прод 23.10 (журнал хода, STEP-2026-10-24-PROD-TELEMETRY): «Где и до скольки принимают» (`menu.tests.where`) с деплоя 15 шла в
LLM-классификатор — ветку часов `rule_address_work_hours` добавили в `topic_by_llm` ради набранного текста (g0577 «со скольки лет
берут кровь»), и нажатие ждало классификатор 11–12 с вместо 0,15–0,5 с. Тему кнопки пациент выбрал нажатием (`button_menu`).

Инвариант класса: под `button_pass_rules = on` ход-кнопка `pass` с любой веткой правила в `topic_by_llm` — тот же ход, что без
списка (текст, метка, классификатор не зван); off — как было (LLM решает тему); набранный руками тот же текст — по-прежнему
решает LLM. Через обработчик эндпоинта, LLM — подделка настоящего типа, отвечает чужой темой (OTHER), чтобы вызов был виден.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import button_menu, choice, llm_runtime
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

# Подписи кнопок — контракт со шлюзом («Таблица экранов, версия 1»).
from tests.test_button_actions import GATEWAY_V1_TITLES

_PASS = sorted(b for b, a in button_menu.BUTTONS.items() if a.mode == "pass")
_OTHER = json.dumps({"label": "OTHER", "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _turn(world_dir, monkeypatch, button_id: str, *, rules: str, switch: str, typed: bool = False) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    text = GATEWAY_V1_TITLES[button_id]
    req = MessengerGenerateRequest(
        session_id=f"pass-{button_id}-{rules}-{switch}-{typed}", text=text, debug=True, button_id="" if typed else button_id
    )

    async def run() -> dict[str, Any]:
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        return {"text": str(body.get("text") or "")}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", "legacy_v2", raising=False)
        patch.setattr(cfg, "MR_TOPIC_BY_LLM", rules, raising=False)
        patch.setattr(cfg, "MR_BUTTON_PASS_RULES", switch, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM({_CLASSIFIER_MARK: _OTHER}))
        choice._CACHE.clear()
        out = asyncio.run(run())
        out["telemetry"] = world.take_telemetry() or {}
    return out


def _branch(world_dir, monkeypatch, button_id: str) -> str:
    """Ветки правила, которые ставит подпись кнопки (`nlu_rule`), — их и кладём в `topic_by_llm`."""

    tel = _turn(world_dir, monkeypatch, button_id, rules="", switch="off")["telemetry"]
    return ",".join(part for part in str(tel.get("nlu_rule") or "").split("+") if part and part != "?")


def _same_turn(a: dict[str, Any], b: dict[str, Any]) -> tuple:
    keys = ("tu_label_nlu", "tu_label_final", "nlu_llm", "nlu_src", "answer_source", "handoff")
    return (a["text"], *(a["telemetry"].get(k) for k in keys)), (b["text"], *(b["telemetry"].get(k) for k in keys))


@pytest.mark.parametrize("button_id", _PASS)
def test_pass_button_turn_ignores_topic_by_llm_of_its_branch(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    branch = _branch(mini_prod_world, monkeypatch, button_id)
    base = _turn(mini_prod_world, monkeypatch, button_id, rules="", switch="on")
    on = _turn(mini_prod_world, monkeypatch, button_id, rules=branch or "rule_test_assist", switch="on")
    got, want = _same_turn(on, base)
    assert got == want, (button_id, branch)
    if branch:  # ветка правила была в списке — ход отметил, что тему решила кнопка
        assert on["telemetry"].get("nlu_topic_by_llm") == "button", on["telemetry"]
        assert on["telemetry"].get("nlu_llm") is False, on["telemetry"]


def test_hours_button_off_keeps_llm_topic_like_prod(mini_prod_world, monkeypatch):  # noqa: F811
    """off — как на проде с деплоя 15: ветка часов в списке — нажатие ждёт классификатор, тему решает LLM."""

    off = _turn(mini_prod_world, monkeypatch, "menu.tests.where", rules="rule_address_work_hours", switch="off")["telemetry"]
    assert off.get("nlu_llm") is True and off.get("nlu_topic_by_llm") in {"llm", "other_offer"}, off
    on = _turn(mini_prod_world, monkeypatch, "menu.tests.where", rules="rule_address_work_hours", switch="on")["telemetry"]
    assert (on.get("nlu_llm"), on.get("nlu_src"), on.get("tu_label_nlu")) == (False, "rule", "ADDRESS"), on


@pytest.mark.parametrize("button_id", _PASS)
def test_typed_caption_still_goes_to_llm_under_topic_by_llm(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: тот же текст, набранный руками (без `button_id`), — выключатель не глушит `topic_by_llm` для текста."""

    branch = _branch(mini_prod_world, monkeypatch, button_id)
    if not branch:
        pytest.skip(f"подпись {button_id} не ставит ветки правила — `topic_by_llm` к ней не применим")
    typed = _turn(mini_prod_world, monkeypatch, button_id, rules=branch, switch="on", typed=True)["telemetry"]
    assert typed.get("nlu_llm") is True and typed.get("nlu_topic_by_llm") in {"llm", "other_offer"}, typed
