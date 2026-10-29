"""Узлы сбоя LLM — честный оффер оператора вместо «молчаливого да» правил (`llm_fail_offer`; BUG-2026-10-27-LLM-QUEUE-GIVES-UP,
шаг 4; проверка ответов без LLM 27.10 `~/Dev/handoffs/2026-10-27_fallback_safety_d18.md`, таблица узлов).

На сбое выбора LLM код шёл «как было» и печатал общий ответ по ключевому слову — весь список филиалов, «без записи» во всех
пунктах, «есть запись к X», цену антител на вопрос о ПЦР: условие вопроса (праздник, ребёнок, ДМС, конкретный филиал) терялось
(58 опасных ответов на 2 076 ходов отказа, 39 — при занятой очереди).

Инвариант класса: в узле из `llm_fail_offer` сбой LLM (отказ соединения, тайм-аут, мусор) даёт честный оффер оператора вопросом
(«да» → оператор), без факта правил; решение LLM, которое состоялось, — тот же ответ, что без выключателя; узел не из списка и
пустой список — как было. Узлы: `address_followup` (реплика после адресного листа), `topic` (ветка `topic_by_llm`), `facet`
(свойство вопроса П1), `role_pick` (роль врача — отдельно, решение владельца по карточке). Через обработчик эндпоинта, LLM —
подделка настоящего типа.
"""

from __future__ import annotations

import pytest

from messengers_router import address_flow
from messengers_router.response_builder import _LLM_FAIL_TEXT, UNSUPPORTED_CATALOG_OPERATOR_OFFER
from messengers_router.runtime_config import config as cfg
from tests.test_address_followup_class import _answer as _address_answer
from tests.test_address_followup_class import _dialog as _address_dialog
from tests.test_address_followup_class import _listed, _press
from tests._llm_helpers import _down
from tests._llm_helpers import _turn as _nlu_turn
from tests.test_p1_facet_class import _turn as _facet_turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_TEST_ASSIST = "есть ли у вас анализ на ферритин"
_FAIL_OFFER = f"{_LLM_FAIL_TEXT['llm_fail_offer']}\n\n{UNSUPPORTED_CATALOG_OPERATOR_OFFER}"


def _nodes(monkeypatch, nodes: str) -> None:
    monkeypatch.setattr(cfg, "MR_LLM_FAIL_OFFER", nodes, raising=False)


# --------------------------------------------------------------------------- address_followup


@pytest.mark.parametrize("followup", [None, "не JSON"], ids=["down", "invalid"])
def test_address_followup_failure_is_an_honest_offer_not_the_whole_list(followup, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press("menu.address"), "где вы находитесь?"]
    _nodes(monkeypatch, "address_followup")
    on = _address_dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=followup)
    _nodes(monkeypatch, "")
    off = _address_dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=followup)

    assert len(_listed(off[1]["text"])) >= 2, off[1]["text"]  # как было: весь список (дефект класса)
    assert on[1]["text"] == address_flow.FAILED_TEXT, on[1]["text"]
    assert on[1]["state"].get("_operator_offer_pending"), on[1]["state"]
    assert on[1]["telemetry"].get("address_followup") == "failed_offer", on[1]["telemetry"]


@pytest.mark.parametrize("outcome", ["list", "other_topic"])
def test_address_followup_decided_by_llm_is_unchanged(outcome, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press("menu.address"), "где вы находитесь?"]
    _nodes(monkeypatch, "address_followup,topic,facet,role_pick")
    on = _address_dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=_address_answer(outcome))
    _nodes(monkeypatch, "")
    off = _address_dialog(mini_prod_world, monkeypatch, turns, switch="on", followup=_address_answer(outcome))

    assert on[1]["text"] == off[1]["text"]


# --------------------------------------------------------------------------- topic


def _topic(world, monkeypatch, reply: str, *, nodes: str, answer) -> dict:
    _nodes(monkeypatch, nodes)
    monkeypatch.setattr(cfg, "MR_TOPIC_BY_LLM", "rule_test_assist", raising=False)
    return _nlu_turn(world, monkeypatch, reply, answer=answer)


@pytest.mark.parametrize("failure", [_down, "не json"], ids=["down", "invalid"])
def test_topic_failure_is_an_honest_offer_not_the_rule_answer(failure, mini_prod_world, monkeypatch):  # noqa: F811
    on = _topic(mini_prod_world, monkeypatch, _TEST_ASSIST, nodes="topic", answer=failure)
    off = _topic(mini_prod_world, monkeypatch, _TEST_ASSIST, nodes="", answer=failure)

    assert off["telemetry"].get("nlu_topic_by_llm") == "rules_fallback" and off["telemetry"].get("tu_label_nlu") == "TEST_ASSIST"
    assert on["telemetry"].get("nlu_topic_by_llm") == "fail_offer", on["telemetry"]
    assert on["text"] == _FAIL_OFFER and not on["telemetry"].get("handoff"), on["text"]


@pytest.mark.parametrize("reply", ["сколько стоит ферритин", "у меня сильная боль в груди и мне трудно дышать"])
def test_topic_failure_outside_topic_by_llm_and_urgent_are_unchanged(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _topic(mini_prod_world, monkeypatch, reply, nodes="topic", answer=_down)
    off = _topic(mini_prod_world, monkeypatch, reply, nodes="", answer=_down)

    assert on["text"] == off["text"]


# --------------------------------------------------------------------------- facet


def test_facet_failure_is_an_honest_offer_not_a_price(mini_prod_world, monkeypatch):  # noqa: F811
    _nodes(monkeypatch, "facet")
    on = _facet_turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet=_down)
    _nodes(monkeypatch, "")
    off = _facet_turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet=_down)

    assert off["telemetry"].get("tu_label_final") == "PRICE", off["telemetry"]  # как было: цена
    assert on["telemetry"].get("tu_label_final") == "OTHER", on["telemetry"]
    assert on["text"] == _FAIL_OFFER, on["text"]


def test_facet_answered_by_llm_is_unchanged(mini_prod_world, monkeypatch):  # noqa: F811
    _nodes(monkeypatch, "facet")
    on = _facet_turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet="price")
    _nodes(monkeypatch, "")
    off = _facet_turn(mini_prod_world, monkeypatch, rules="rule_test_assist", facet="price")

    assert on["text"] == off["text"] and on["telemetry"].get("tu_label_final") == "PRICE"


def test_switch_defaults_to_empty():
    from agent_logic_2 import config as host_config

    assert host_config.settings.messenger_router.llm_fail_offer == ""


# --------------------------------------------------------------------------- role_pick (отдельно: решение владельца по карточке)


@pytest.mark.parametrize("text", ["запишите к эндокринологу ребёнку 12 лет", "хочу к гинекологу по ДМС"])
def test_role_pick_failure_is_an_offer_only_under_its_own_node(text, monkeypatch):
    """В мини-мире нет врачей — узел проверяется обвязкой теста класса роли (каталог подменён): сбой выбора роли под
    `role_pick` — флаг `role_fail_offer` (путь отказа каталога: текст «подберёт администратор» и оффер); без узла — как было,
    в том числе когда включены остальные узлы."""

    from messengers_router.mess_types import Evidence
    from messengers_router.response_builder import build_unsupported_catalog_response
    from messengers_router.router import _unsupported_catalog_kind
    from tests.test_role_pick_on_class import _apply
    from tests.test_role_pick_shadow_class import _down as _role_down

    _nodes(monkeypatch, "address_followup,topic,facet")
    before, others, _calls = _apply(monkeypatch, text, _role_down, rules="эндокринолог")
    assert others is before
    _nodes(monkeypatch, "role_pick")
    _before, on, _calls = _apply(monkeypatch, text, _role_down, rules="эндокринолог")
    assert "role_fail_offer" in on.flags and _unsupported_catalog_kind(set(on.flags)) == "role_fail_offer"
    reply = build_unsupported_catalog_response(Evidence(items={"unsupported_catalog": {"kind": "role_fail_offer"}}))
    assert reply is not None and reply.text == _LLM_FAIL_TEXT["role_fail_offer"] and not reply.handoff


def test_role_pick_decided_by_llm_is_unchanged(monkeypatch):
    from tests.test_role_pick_on_class import _apply
    from tests.test_role_pick_shadow_class import _picks

    _nodes(monkeypatch, "role_pick")
    _before, on, _calls = _apply(monkeypatch, "нужен окулист", _picks("офтальмолог"))
    _nodes(monkeypatch, "")
    _before, off, _calls = _apply(monkeypatch, "нужен окулист", _picks("офтальмолог"))
    assert on.entities == off.entities and set(on.flags) == set(off.flags)
