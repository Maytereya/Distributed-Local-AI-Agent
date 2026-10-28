"""С6, шаг 1: ветки цепочки правил, не сработавшие на трафике, сняты — тему их реплик решает LLM (STEP-2026-10-28-C6-RULES-NEVER-FIRED).

Направление владельца 27.10 (LLM-first, снятие правил по замеру): ветка `deterministic_rule_decision`, которая не сработала ни на
одном ходе прода 14–28.10 и ни на одном вопросе стенда (`s19all_btn / dev / test`) и не держит работу без LLM (не SAFETY и не в
`topic_by_llm`), снимается — на виденном трафике поведение не меняется. Инвариант класса: реплика, которую ловила снятая ветка,
доходит до LLM-классификатора, и тему хода решает его метка; при упавшей LLM — ближайшая ветка из `topic_by_llm` (запасной путь), а не
снятое правило. Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import json

import pytest

from tests.test_llm_primary_rules_fallback_class import _down, _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура


def _answer(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _run(world, monkeypatch, reply: str, answer) -> dict:
    return _turn(world, monkeypatch, reply, engine="legacy_v2", fallback="off", answer=answer)["telemetry"]


# «Специальность + ближайший / поскорее / самое раннее» — ветка `rule_schedule_nearest` решала DOCTOR_SCHEDULE по буквам без LLM.
_NEAREST = ["гастроэнтеролог ближайший", "ближайший уролог", "кардиолог поскорее", "самое раннее к лору"]


@pytest.mark.parametrize("reply", _NEAREST)
def test_nearest_specialty_topic_is_decided_by_llm(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _run(mini_prod_world, monkeypatch, reply, _answer("DOCTOR_SCHEDULE"))
    assert "rule_schedule_nearest" not in str(tel.get("nlu_rule") or ""), tel
    assert tel.get("nlu_llm") is True and tel.get("nlu_topic_by_llm") == "llm", tel
    assert tel.get("tu_label_nlu") == "DOCTOR_SCHEDULE", tel


@pytest.mark.parametrize("reply", _NEAREST)
def test_nearest_specialty_without_llm_takes_the_listed_fallback(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _run(mini_prod_world, monkeypatch, reply, _down)
    assert tel.get("nlu_topic_by_llm") == "rules_fallback", tel
    assert "rule_doctor_info_specialty" in str(tel.get("nlu_rule") or ""), tel
