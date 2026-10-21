"""Голое название услуги → максимум по данным клиники (класс 3, решение владельца 21.10, принцип разметки 7).

g0759 «Анализ T-SPOT» (разметка: подготовка) получал адреса «анализы без записи», «Анализ на Впч» — голые цены, «Фгдс» —
список врачей. Инвариант под `service_answer_from_data = on`: короткая реплика без висящего вопроса и без свойства вопроса —
кандидат; голое ли это название и какие строки прайса названы, решает LLM; одна строка — максимум из данных (цена и срок —
прайс, что это и подготовка — справочник МИС, где — правило клиники), несколько — варианты с ценами и вопрос «какой»; вопрос
в реплике, «не услуга», сбой — как было. Off — как было, без вызова.
"""

from __future__ import annotations

import json

import pytest

from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_service_answer_class import CHECK_MARK, _classifier, _number_of, answer_world, confirm_all, turn  # noqa: F401

_OVERVIEW_MARK = "прислал ли пациент только название услуги"
_T_SPOT = "T-SPOT (диагностика латентной и активной туберкулезной инфекции у детей и взрослых)"
_REPLY = "Анализ T-SPOT"


def _overview(*names: str, outcome: str = "none"):
    return lambda prompt: json.dumps({"picks": [_number_of(prompt, name) for name in names], "outcome": outcome})


def _answers(overview) -> dict:
    # Классификатор — как правило реплики (адреса), чтобы «как было» не зависело от класса 4 (тему ветки walk-in решает LLM).
    return {_CLASSIFIER_MARK: _classifier("ADDRESS"), _OVERVIEW_MARK: overview, CHECK_MARK: confirm_all}


def test_bare_name_gets_the_maximum_from_data(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, _REPLY, switch="on", answers=_answers(_overview(_T_SPOT)))

    text = out["text"]
    assert out["telemetry"].get("svc_overview") == "one", out["telemetry"]
    assert text.startswith(f"«{_T_SPOT}»:"), text
    assert "— цена: 7 900 руб." in text and "— срок готовности: до 12" in text, text  # прайс
    assert "— что это: Тест выявляет туберкулёзную инфекцию" in text, text  # справочник МИС
    assert "— подготовка: Кровь сдаётся утром." in text, text
    assert "— где: в лабораторных филиалах клиники, без записи" in text, text  # правило клиники: анализ (есть срок)
    assert not out["handoff"]


def test_several_variants_ask_which_one(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "Чекап", switch="on", answers=_answers(_overview("Ежегодный Чекап", "Женский чекап Базовый")))

    assert out["text"].startswith("В прайсе клиники несколько вариантов:"), out["text"]
    assert "— Ежегодный Чекап — 2 055 руб." in out["text"] and "Какой вариант интересует?" in out["text"], out["text"]


@pytest.mark.parametrize(
    "overview",
    [
        json.dumps({"picks": [], "outcome": "asks"}),
        json.dumps({"picks": [], "outcome": "not_service"}),
        json.dumps({"picks": [], "outcome": "none"}),
        "не json",
        ConnectionError("тест: LLM недоступна"),
    ],
)
def test_not_a_bare_name_or_failure_is_as_before(overview, answer_world, monkeypatch):  # noqa: F811
    off = turn(answer_world, monkeypatch, _REPLY, switch="off", answers=_answers(overview))
    on = turn(answer_world, monkeypatch, _REPLY, switch="on", answers=_answers(overview))

    assert _OVERVIEW_MARK not in off["calls"]
    assert on["text"] == off["text"]


@pytest.mark.parametrize("reply", ["сколько стоит T-SPOT", "как подготовиться к T-SPOT"])
def test_explicit_question_is_not_asked(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_overview(_T_SPOT)))

    assert _OVERVIEW_MARK not in out["calls"], out["telemetry"]


def test_where_question_is_decided_by_llm(answer_world, monkeypatch):  # noqa: F811
    reply = "где сдать T-SPOT"
    asks = json.dumps({"picks": [], "outcome": "asks"})
    off = turn(answer_world, monkeypatch, reply, switch="off", answers=_answers(asks))
    on = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(asks))

    assert _OVERVIEW_MARK in on["calls"] and on["telemetry"].get("svc_overview") == "asks", on["telemetry"]
    assert on["text"] == off["text"]


def test_decision_that_changes_nothing_returns_the_turn_budget(answer_world, monkeypatch):  # noqa: F811
    asks = json.dumps({"picks": [], "outcome": "asks"})
    out = turn(answer_world, monkeypatch, _REPLY, switch="on", answers=_answers(asks), cfg_extra={"MR_LLM_DECISION_TURN_CALLS": 1})

    assert out["telemetry"].get("svc_overview") == "asks" and out["telemetry"].get("svc_overview_refund") == 1, out["telemetry"]
