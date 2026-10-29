"""Путь цены сдался — сначала данные клиники (STEP-2026-10-29-PRICE-NOT-FOUND-BY-DATA, выключатель `price_not_found_by_data`).

Стенд 29.10 (кнопки цены под рамкой `frame`): 32 провала ответом «Не нашёл актуальную стоимость … Уточните название услуги» — и где
услуга в данных есть (второй круг поиска по терминам LLM её находит), и где её нет, а нужен честный оффер оператора, не переспрос.

Инвариант класса: под on ход цены, где путь цены сдался (цен нет, не перевод, не варианты, не «в таком виде нет»), отвечает по данным
клиники (`service_answer`, место `price_not_found`, без «похожего»): строка из данных — с ценой из прайса; нет данных — честный оффер;
сбой LLM и «не о клинике» — прежний ответ цены; off — как было. Через обработчик эндпоинта; LLM — подделка настоящего типа.
"""

from __future__ import annotations

import pytest

from messengers_router import evidence_keys as ek
from messengers_router import service_answer
from messengers_router.mess_types import Evidence, RouteDecision
from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_service_answer_class import _SERVICE_ANSWER_MARK, _classifier, _number_of, answer_world, turn, verdict  # noqa: F401

_QUESTION = "сколько стоит устранение лапоухости"
_NOT_FOUND = "Не нашёл актуальную стоимость"


def _answers(svc) -> dict:
    # Классификатор — цена; выбор строки цены и разбор реплики — «ничего не подходит»: путь цены сдаётся.
    return {_SERVICE_ANSWER_MARK: svc, _CLASSIFIER_MARK: _classifier("PRICE"), "Сообщение пациента": verdict()}


def _turn(world, monkeypatch, svc, *, switch: str, calls: int = 3) -> dict:
    extra = {"MR_PRICE_NOT_FOUND_BY_DATA": switch, "MR_LLM_DECISION_TURN_CALLS": calls}
    return turn(world, monkeypatch, _QUESTION, switch="off", answers=_answers(svc), cfg_extra=extra)


def test_off_gives_up_as_before(answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, verdict(outcome="no_data"), switch="off")
    assert _SERVICE_ANSWER_MARK not in out["calls"], out
    assert out["text"].startswith(_NOT_FOUND) or "Уточните" in out["text"], out["text"]


def test_no_data_is_an_honest_operator_offer(answer_world, monkeypatch):  # noqa: F811
    out = _turn(answer_world, monkeypatch, verdict(outcome="no_data"), switch="on")
    assert out["telemetry"].get("svc_answer_site") == "price_not_found", out["telemetry"]
    assert out["text"].startswith(service_answer.NO_DATA_TEXT) and out["pending_offer"], out["text"]


def _second_round(prompt: str) -> str:
    """Первый круг — «нет данных» с термином поиска; второй круг нашёл строку прайса — её и выбираем."""

    if "Прайс клиники: Ежегодный Чекап" in prompt:
        return verdict(picks=(_number_of(prompt, "Прайс клиники: Ежегодный Чекап"),))
    return verdict(outcome="no_data", terms=("Ежегодный Чекап",))


def test_data_row_found_in_the_second_round_is_printed_from_the_price(answer_world, monkeypatch):  # noqa: F811
    # Второму кругу нужно решение хода сверх лимита прода (3: разбор реплики, выбор строки, первый круг) — механизм проверяем с 5.
    out = _turn(answer_world, monkeypatch, _second_round, switch="on", calls=5)
    assert out["telemetry"].get("svc_answer_rounds") == 2, out["telemetry"]
    assert out["text"].startswith(service_answer.ANSWER_HEADING), out["text"]
    assert "— Ежегодный Чекап — 2 055 руб." in out["text"], out["text"]  # название и цена — из прайса, не из LLM


@pytest.mark.parametrize("svc", [verdict(outcome="not_about_clinic"), "не json"])
def test_failure_or_not_about_clinic_keeps_the_price_answer(svc, answer_world, monkeypatch):  # noqa: F811
    on = _turn(answer_world, monkeypatch, svc, switch="on")
    off = _turn(answer_world, monkeypatch, svc, switch="off")
    assert on["text"] == off["text"]


def _price(payload: dict) -> Evidence:
    return Evidence(items={ek.PRICE: payload})


@pytest.mark.parametrize(
    "payload",
    [
        {"prices": [{"name": "x"}]},
        {"prices": [], "handoff_required": True},
        {"prices": [], "service_kind": "family_query"},
        {"prices": [], "qualifiers_missing": ["суммарные"]},
        {"prices": [], "clarify_text": "Уточните", "note": "price_match_policy: parse_failed"},
    ],
)
def test_price_that_answered_or_failed_to_parse_is_not_given_up(payload, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_NOT_FOUND_BY_DATA", "on", raising=False)
    decision = RouteDecision(label="PRICE", confidence=0.9)
    assert not service_answer.price_gave_up(decision, _price(payload))
    assert service_answer.price_gave_up(decision, _price({"prices": [], "note": "price_match_policy: none_fit", "clarify_text": "x"}))
