"""С6, шаг 2: ветки правил, отвечающие раньше LLM на простых вопросах, — в `topic_by_llm` списком (STEP-2026-10-28-C6-BRANCHES-TO-LLM).

Решение владельца 27.10 (порядок починки, п. 2): `rule_address`, `rule_news`, `rule_appointment` — по одной в LLM-first, переводить
по карточке. Механизм — уже принятый список `topic_by_llm` (ход (б), этап 4): ветка не удаляется, а становится кандидатом — разбор
аудита правил предупреждает, что ветка, вынутая из цепочки, отдала бы реплику нижестоящим правилам (`rule_price`). Инвариант класса для
каждой из трёх веток: в списке — тему хода решает метка LLM-классификатора, в том числе OTHER (честный оффер, не ответ правила);
упала LLM — ответ правила байт в байт, как без списка; без ветки в списке (умолчание = прод) — правило без вызова LLM, как было.
Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import json

import pytest

from tests._llm_helpers import _down, _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_PROD = "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,rule_website_help,rule_address_work_hours,rule_test_result,rule_prepare"

# ветка → (реплики, которые она ловит, метка правила, метка LLM «по смыслу» не той темы)
_BRANCHES = {
    "rule_address": (["где находится ваш филиал на Ленина", "по какому адресу вы находитесь"], "ADDRESS", "NEWS"),
    "rule_news": (["какие у вас есть акции", "есть скидка на первый прием?"], "NEWS", "PRICE"),
    "rule_appointment": (["хочу записаться к кардиологу", "запишите меня на приём к терапевту"], "APPOINTMENT", "DOCTOR_SCHEDULE"),
}
_CASES = [(branch, reply) for branch, (replies, _, _) in _BRANCHES.items() for reply in replies]


def _answer(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _run(world, monkeypatch, reply: str, *, rules: str, answer) -> dict:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TOPIC_BY_LLM", rules, raising=False)
    return _turn(world, monkeypatch, reply, answer=answer)


@pytest.mark.parametrize("branch,reply", _CASES)
def test_prod_default_keeps_rule_without_llm(branch, reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _run(mini_prod_world, monkeypatch, reply, rules=_PROD, answer=_answer("OTHER"))["telemetry"]
    assert branch in str(tel.get("nlu_rule") or "").split("+"), tel
    assert (tel.get("nlu_src"), tel.get("nlu_llm"), tel.get("tu_label_nlu")) == ("rule", False, _BRANCHES[branch][1]), tel


@pytest.mark.parametrize("branch,reply", _CASES)
def test_listed_branch_is_candidate_and_llm_label_decides(branch, reply, mini_prod_world, monkeypatch):  # noqa: F811
    other = _BRANCHES[branch][2]
    tel = _run(mini_prod_world, monkeypatch, reply, rules=f"{_PROD},{branch}", answer=_answer(other))["telemetry"]
    assert tel.get("nlu_llm") is True and tel.get("nlu_topic_by_llm") == "llm", tel
    assert tel.get("nlu_src") == "llm" and branch in str(tel.get("nlu_rule") or "").split("+"), tel
    assert tel.get("tu_label_nlu") == other, tel


@pytest.mark.parametrize("branch,reply", _CASES)
def test_llm_other_is_an_honest_offer_not_the_rule_answer(branch, reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _run(mini_prod_world, monkeypatch, reply, rules=f"{_PROD},{branch}", answer=_answer("OTHER"))
    assert on["telemetry"].get("nlu_topic_by_llm") == "other_offer", on["telemetry"]
    assert "точно ответит оператор" in on["text"] and "перевести" in on["text"].lower(), on["text"]


@pytest.mark.parametrize("branch,reply", _CASES)
def test_llm_failure_gives_the_rule_answer_as_before(branch, reply, mini_prod_world, monkeypatch):  # noqa: F811
    off = _run(mini_prod_world, monkeypatch, reply, rules=_PROD, answer=_down)
    on = _run(mini_prod_world, monkeypatch, reply, rules=f"{_PROD},{branch}", answer=_down)
    assert on["text"] == off["text"]
    assert on["telemetry"].get("nlu_topic_by_llm") == "rules_fallback", on["telemetry"]
    assert on["telemetry"].get("tu_label_nlu") == _BRANCHES[branch][1], on["telemetry"]
