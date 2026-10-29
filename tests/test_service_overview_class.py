"""Голое название услуги → максимум по данным клиники (класс 3, решение владельца 21.10, принцип разметки 7).

g0759 «Анализ T-SPOT» (разметка: подготовка) получал адреса «анализы без записи», «Анализ на Впч» — голые цены, «Фгдс» —
список врачей. Инвариант под `bare_name_overview = on` (до 26.10 — под `service_answer_from_data`; связку владелец развязал 26.10,
п. 4 топ-5 разбора 95 цен): короткая реплика без висящего вопроса и без свойства вопроса —
кандидат; голое ли это название и какие строки прайса названы, решает LLM; одна строка — максимум из данных (цена и срок —
прайс, что это и подготовка — справочник МИС, где — правило клиники), несколько — варианты с ценами и вопрос «какой»; вопрос
в реплике, «не услуга», сбой — как было. Off — как было, без вызова. С 26.10 — и на темах записи, адресов и врачей; там строки
приёма врача не кандидаты (цену приёма к специальности не называем, пока врач не выбран — 25.09); без срока готовности «где»
не утверждает «по записи» (ЭКГ — без записи по правилу клиники).
"""

from __future__ import annotations

import json

import pytest

from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_service_answer_class import _classifier, _number_of, answer_world  # noqa: F401
from tests.test_service_answer_class import turn as _svc_turn


def turn(world_dir, monkeypatch, reply: str, *, switch: str, answers: dict, cfg_extra: dict | None = None) -> dict:
    """Ход с `bare_name_overview = switch`; ответ по данным (`service_answer_from_data`) — off, как на проде."""

    return _svc_turn(world_dir, monkeypatch, reply, switch="off", answers=answers, cfg_extra={"MR_BARE_NAME_OVERVIEW": switch, **(cfg_extra or {})})

_OVERVIEW_MARK = "прислал ли пациент только название услуги"
_T_SPOT = "T-SPOT (диагностика латентной и активной туберкулезной инфекции у детей и взрослых)"
_REPLY = "Анализ T-SPOT"


def _overview(*names: str, outcome: str = "none"):
    return lambda prompt: json.dumps({"picks": [_number_of(prompt, name) for name in names], "outcome": outcome})


def _answers(overview, label: str = "ADDRESS") -> dict:
    # Классификатор — как правило реплики (адреса), чтобы «как было» не зависело от класса 4 (тему ветки walk-in решает LLM).
    return {_CLASSIFIER_MARK: _classifier(label), _OVERVIEW_MARK: overview}


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
    # Голое название на теме врачей (правило ставит «Узи лимфоузлов» в DOCTOR_INFO): несколько вариантов — с ценами и вопросом.
    lymph = ("УЗИ периферических лимфоузлов (1 область) с эластографией", "УЗИ периферических лимфоузлов комплексно - все зоны (детское)")
    out = turn(answer_world, monkeypatch, "Узи лимфоузлов", switch="on", answers=_answers(_overview(*lymph), label="APPOINTMENT"))

    assert out["text"].startswith("В прайсе клиники несколько вариантов:"), out["text"]
    assert f"— {lymph[0]} — 1 800 руб." in out["text"] and "Какой вариант интересует?" in out["text"], out["text"]


@pytest.mark.parametrize("reply", ["Чекап", "Анализ T-SPOT"])
def test_price_and_test_assist_topics_keep_their_own_data_answer(reply, answer_world, monkeypatch):  # noqa: F811
    # На цене и подборе анализов у хода свой ответ по данным (выбор строки цены, ответ подбора) — строже; обзор его не перебивает
    # (карточка 26.10: «мазок на онкоцитологию» → «Мазок на эозинофилы…», «соскоб на лишай» → «ЗППП: соскоб»).
    for label in ("PRICE", "TEST_ASSIST"):
        out = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_overview(_T_SPOT), label=label))
        if out["telemetry"].get("tu_label_final") in {"PRICE", "TEST_ASSIST"}:
            assert _OVERVIEW_MARK not in out["calls"], (label, out["telemetry"])


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


# --------------------------------------------------------------------------- п. 4 топ-5 разбора 95 цен (26.10)

_UZI = "УЗИ органов мошонки + ЦДК сосудов"


@pytest.mark.parametrize("label", ["APPOINTMENT", "DOCTOR_INFO", "ADDRESS"])
def test_bare_procedure_name_on_booking_doctors_or_address_gets_the_price(label, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "УЗИ мошонки", switch="on", answers=_answers(_overview(_UZI), label=label))

    assert out["telemetry"].get("svc_overview") == "one", out["telemetry"]
    assert out["text"].startswith(f"«{_UZI}»:") and "— цена: 1 850 руб." in out["text"], out["text"]
    assert "по записи" not in out["text"], out["text"]  # без срока готовности «где» не утверждает запись (ЭКГ — без записи)


@pytest.mark.parametrize("reply", ["Невролог", "Приём невролога", "Консультация уролога"])
def test_bare_specialty_on_booking_and_doctors_gets_no_reception_price(reply, answer_world, monkeypatch):  # noqa: F811
    # Цену приёма к специальности не называем, пока врач не выбран (решение 25.09): строки приёма — не кандидаты обзора, и
    # у названия специальности их не остаётся — ход идёт как было, без вызова LLM обзора.
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_answers(_overview(_UZI), label="APPOINTMENT"))

    assert out["telemetry"].get("tu_label_final") in {"APPOINTMENT", "DOCTOR_INFO"}, out["telemetry"]
    assert out["telemetry"].get("svc_overview") == "no_candidates" and _OVERVIEW_MARK not in out["calls"], out["telemetry"]
    assert "2 700 руб." not in out["text"] and "3 100 руб." not in out["text"], out["text"]


@pytest.mark.parametrize(("without", "expected"), [(True, [_UZI]), (False, ["Прием (осмотр, консультация) врача-уролога", _UZI])])
def test_reception_rows_are_dropped_only_on_booking_and_doctors(without, expected, monkeypatch):
    import asyncio

    from messengers_router import choice, service_answer, service_overview
    from messengers_router.services import _prices_helpers

    rows = [{"serviceName": "Прием (осмотр, консультация) врача-уролога", "cost": 3100.0}, {"serviceName": _UZI, "cost": 1850.0}]
    seen: list[list[str]] = []

    async def fake_choose(task, question, options, **_kw):
        seen.append([o.text for o in options])
        return choice.Choice(status=choice.Status.OUTCOME, outcome="asks")

    monkeypatch.setattr(_prices_helpers, "_catalog_rows", lambda _region: rows)
    monkeypatch.setattr(service_answer, "_price_rows", lambda texts, question, catalog, first: list(catalog))
    monkeypatch.setattr(service_overview, "choose", fake_choose)
    asyncio.run(service_overview.decide("уролог узи", without_reception=without))

    assert seen == [expected]


def test_service_answer_alone_does_not_turn_overview_on(answer_world, monkeypatch):  # noqa: F811
    # Связка 21.10 развязана (решение владельца 26.10): у ответа по данным и у обзора — свои выключатели и карточки.
    out = _svc_turn(answer_world, monkeypatch, _REPLY, switch="on", answers=_answers(_overview(_T_SPOT)))

    assert _OVERVIEW_MARK not in out["calls"], out["calls"]
