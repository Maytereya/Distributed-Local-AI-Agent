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


@pytest.mark.parametrize("svc", [verdict(outcome="not_about_clinic"), verdict(outcome="general"), "не json"])
def test_failure_not_about_clinic_or_general_keeps_the_price_answer(svc, answer_world, monkeypatch):  # noqa: F811
    """Сбой, «не о клинике» и «общий вопрос» (услуга не названа — «Стоимость анализов») — уточнение пути цены, не оффер."""

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


def test_first_round_candidates_come_from_the_turn_parse(monkeypatch):
    """Кандидаты первого круга — фразы-услуги разбора хода и их строки, а не только слова реплики целиком."""

    from messengers_router import turn_parse
    from messengers_router.turn_parse import ParsedPhrase, ParseStatus, PhraseParse

    question = "У вас есть чекап для мужчин 60+? Что входит и какая цена?"
    row = {"serviceId": 1, "serviceName": "Мужской чекап"}
    parse = PhraseParse(
        reply=question, status=ParseStatus.PICKED, source="llm",
        items=(ParsedPhrase(text="чекап для мужчин", start=10, end=26, strength="weak", rows=(row,), full="чекап для мужчин 60+"),),
    )
    monkeypatch.setattr(turn_parse, "phrases_now", lambda: parse)
    terms, rows = service_answer.turn_parse_terms(question)
    assert terms == ["чекап для мужчин 60+"] and rows == [row]
    assert service_answer.turn_parse_terms("другая реплика") == ([], [])  # разбор не этой реплики — поиск по реплике, как было


def _similar(prompt: str) -> str:
    if "Прайс клиники: Ежегодный Чекап" in prompt:
        return verdict(outcome="similar", picks=(_number_of(prompt, "Прайс клиники: Ежегодный Чекап"),))
    return verdict(outcome="no_data", terms=("Ежегодный Чекап",))


def _checked(monkeypatch, check):
    from messengers_router.services import _qualifier_pool

    async def fake(question, rows):
        return check

    monkeypatch.setattr(_qualifier_pool, "check_rows", fake)


def test_similar_confirmed_by_the_c3_check_is_shown_with_an_honest_heading(answer_world, monkeypatch):  # noqa: F811
    """«Похожее», которое проверка С3 признала той же услугой без условия из вопроса, — строки данных с заголовком «точно такой
    нет, есть похожие» и оффером, как «в таком виде нет» у цены, а не скрытое «нет данных»."""

    from messengers_router.services import _qualifier_pool

    _checked(monkeypatch, _qualifier_pool.RowsCheck(service=True, missing=("60+",)))
    out = _turn(answer_world, monkeypatch, _similar, switch="on", calls=5)
    assert out["text"].startswith(service_answer.SIMILAR_HEADING), out["text"]
    assert "— Ежегодный Чекап — 2 055 руб." in out["text"] and out["pending_offer"], out["text"]


@pytest.mark.parametrize("check", ["другая услуга", None])
def test_similar_not_confirmed_is_an_honest_offer_without_foreign_rows(check, answer_world, monkeypatch):  # noqa: F811
    """Проверка С3 сказала «другая услуга», не ответила или уточнений нет — «похожее» не печатаем: честное «нет данных», без чужих строк."""

    from messengers_router.services import _qualifier_pool

    _checked(monkeypatch, _qualifier_pool.RowsCheck(service=False) if check else None)
    out = _turn(answer_world, monkeypatch, _similar, switch="on", calls=5)
    assert out["text"].startswith(service_answer.NO_DATA_TEXT) and "Ежегодный Чекап" not in out["text"], out["text"]
