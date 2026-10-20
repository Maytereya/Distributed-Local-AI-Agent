"""Ход (б), этап 4, вариант Б: тему решает LLM, правило — кандидат, по одному правилу (STEP-2026-10-21-B-TOPIC-BY-LLM).

Решения владельца 20.10 (б) 1, 5: LLM решает тему там, где слои спорят (`rule_test_assist`), выключатель — по правилу
(`topic_by_llm = rule_test_assist`). Инвариант класса: ветка правила из списка тему сама не решает — зовётся
LLM-классификатор, его метка побеждает (правило дарит сущности, `_merge`); сбой LLM — ответ правила, как без
выключателя; правило не из списка и срочное — как было, без вызова LLM; пустой список — ход не меняется ни на такт.
Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import json

import pytest

from tests.test_llm_primary_rules_fallback_class import _down, _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_TEST_ASSIST = "есть ли у вас анализ на ферритин"


def _answer(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _run(world, monkeypatch, reply: str, *, rules: str, answer) -> dict:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TOPIC_BY_LLM", rules, raising=False)
    return _turn(world, monkeypatch, reply, engine="legacy_v2", fallback="off", answer=answer)


def test_empty_switch_keeps_rule_without_llm(mini_prod_world, monkeypatch):  # noqa: F811
    tel = _run(mini_prod_world, monkeypatch, _TEST_ASSIST, rules="", answer=_answer("OTHER"))["telemetry"]
    assert (tel.get("nlu_src"), tel.get("nlu_llm"), tel.get("tu_label_nlu")) == ("rule", False, "TEST_ASSIST"), tel
    assert "nlu_topic_by_llm" not in tel


@pytest.mark.parametrize("llm_label", ["PRICE", "TEST_ASSIST", "PREPARE"])
def test_listed_rule_is_candidate_and_llm_decides(llm_label, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _run(mini_prod_world, monkeypatch, _TEST_ASSIST, rules="rule_test_assist", answer=_answer(llm_label))["telemetry"]
    assert tel.get("nlu_llm") is True and tel.get("nlu_topic_by_llm") == "llm", tel
    assert tel.get("nlu_src") == "llm" and "rule_test_assist" in (tel.get("nlu_rule") or ""), tel
    assert tel.get("tu_label_nlu") == llm_label, tel


@pytest.mark.parametrize("failure", [_down, "не json"])
def test_llm_failure_falls_back_to_rule_answer(failure, mini_prod_world, monkeypatch):  # noqa: F811
    off = _run(mini_prod_world, monkeypatch, _TEST_ASSIST, rules="", answer=failure)
    on = _run(mini_prod_world, monkeypatch, _TEST_ASSIST, rules="rule_test_assist", answer=failure)
    assert on["text"] == off["text"]
    assert on["telemetry"].get("nlu_topic_by_llm") == "rules_fallback", on["telemetry"]
    assert on["telemetry"].get("tu_label_nlu") == "TEST_ASSIST"


@pytest.mark.parametrize(
    "reply", ["сколько стоит ферритин", "у меня сильная боль в груди и мне трудно дышать"]
)
def test_other_rules_and_safety_are_untouched(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off = _run(mini_prod_world, monkeypatch, reply, rules="", answer=_answer("OTHER"))
    on = _run(mini_prod_world, monkeypatch, reply, rules="rule_test_assist,urgent,rule_urgent", answer=_answer("OTHER"))
    assert on["text"] == off["text"]
    assert "nlu_topic_by_llm" not in on["telemetry"], on["telemetry"]


@pytest.mark.parametrize("llm_label", ["TEST_ASSIST", "ADDRESS"])
def test_walkin_detector_does_not_override_llm_topic(llm_label, mini_prod_world, monkeypatch):  # noqa: F811
    """Правило — кандидат и после NLU: детектор «без записи» (guardrail) не возвращает метку правила поверх LLM."""

    reply = "нужно ли записываться на анализ крови"
    off = _run(mini_prod_world, monkeypatch, reply, rules="", answer=_answer(llm_label))["telemetry"]
    assert off.get("tu_label_final") == "ADDRESS", off  # как было: решает правило walk-in
    on = _run(mini_prod_world, monkeypatch, reply, rules="rule_nonbookable_walkin", answer=_answer(llm_label))["telemetry"]
    assert not any(s.startswith("policy_nonbookable_walkin:") for s in on.get("tu_relabel_src") or []), on
    assert on.get("tu_label_nlu") == llm_label, on
