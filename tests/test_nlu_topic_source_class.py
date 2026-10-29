"""Кто решил тему хода — телеметрия `nlu_src` / `nlu_rule` (STEP-2026-10-20-B-TOPIC-SOURCE, ход (б), DEBT-1).

Подпись решения NLU (`decision_source`) не различает правило и LLM: решение ветки правила `legacy_v2` проходит слияние
(`nlu_pipeline._merge`) с подписью `llm_primary`. Замер-основа хода (б) (`evalbench.topic_source`, план 20.10) поэтому
восстанавливал источник по косвенным признакам; на проде их нет.

Инвариант класса: у каждого хода, дошедшего до NLU, в телеметрии есть `nlu_src`; `nlu_src = rule` ⇔ LLM-классификатор на
ходе не звали, и тогда `nlu_rule` — непустые флаги ветки без PII и вторичных намерений; `nlu_src = llm` — без `nlu_rule`.
Ответ хода от записи не меняется. Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import json


from tests._llm_helpers import _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_LLM_ANSWER = json.dumps({"label": "OTHER", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
_REPLIES = (
    "сколько стоит ферритин",
    "хочу записаться к кардиологу",
    "какой у вас адрес",
    "у меня сильная боль в груди и мне трудно дышать",
    "мой телефон 89991234567, хочу записаться к кардиологу",
    "а вот у меня такой вопрос насчёт дочки",
    "спасибо большое",
)


def _turns(world, monkeypatch) -> list[dict]:
    return [_turn(world, monkeypatch, r, answer=_LLM_ANSWER) for r in _REPLIES]


def test_every_nlu_turn_says_who_decided_the_topic(mini_prod_world, monkeypatch):  # noqa: F811
    turns = [t for t in _turns(mini_prod_world, monkeypatch) if "nlu_src" in t["telemetry"]]
    assert turns, "ни один ход не записал источник темы"
    for t in turns:
        tel = t["telemetry"]
        if tel["nlu_src"] == "rule":
            assert not tel.get("nlu_llm"), tel
            branch = tel.get("nlu_rule") or ""
            assert branch and branch != "?", tel
            assert not any(f.startswith("pii_") or f == "multi_intent_detected" for f in branch.split("+")), tel
        else:
            assert tel["nlu_src"] == "llm", tel
            assert "nlu_rule" not in tel, tel


def test_legacy_corpus_has_both_sources(mini_prod_world, monkeypatch):  # noqa: F811
    """Корпус не вырожден: у `legacy_v2` есть и ходы правил, и ходы LLM — инвариант выше проверяет обе ветки."""

    sources = {t["telemetry"].get("nlu_src") for t in _turns(mini_prod_world, monkeypatch)}
    assert {"rule", "llm"} <= sources, sources


def test_legacy_rule_turn_is_signed_llm_primary_but_counted_as_rule(mini_prod_world, monkeypatch):  # noqa: F811
    """Исходный симптом DEBT-1: подпись `llm_primary` у решения правила; источник темы — правило, LLM не звали."""

    turn = _turn(mini_prod_world, monkeypatch, "сколько стоит ферритин", answer=_LLM_ANSWER)
    tel = turn["telemetry"]

    assert (tel.get("nlu_src"), tel.get("nlu_llm")) == ("rule", False), tel
    assert "rule_price" in tel.get("nlu_rule", "").split("+"), tel
