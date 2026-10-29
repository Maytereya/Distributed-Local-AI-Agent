"""У каждого типа вопроса на ходе цены — свой читатель (деплой 22, STEP-2026-10-30-FACET-READERS, `facet_by_llm`).

Было (стенд 30.10, `s22on_dev`): тип вопроса решила LLM, а читателя у значения на ходе цены не было — ход шёл в шаблон цены.
«Кровь на ХГЧ могу ли сдать через 15 минут? И сколько ждать результат?» (`booking`) и «Откуда берут биоматериал для ВПЧ?»
(`prepare`) → «Не нашёл актуальную стоимость. Уточните название…»; «Время забора крови» (`where`) → цены взятия крови. Срок у
услуги из двух строк прайса с разными сроками («Посев мочи с идентификацией культур»: 6 и 3–5 дней) — перевод на оператора, хотя
оба срока в данных есть.

Инвариант класса под `facet_by_llm = on` (таблица «значение → читатель»):
- цена, «просто назвал», «есть ли» — путь цены; срок — срок из МИС;
- подготовка — памятка (PREPARE); «где / когда» и «нужна ли запись» — адреса (ADDRESS, как вариант П1 `where_when`);
  «ни одно» — данные клиники или честный оффер (OTHER, как у П1 `p1_facet_none_offer`);
- ответ на вопрос кнопки цены — цена: тему задала кнопка;
- строки одной услуги с разными сроками — каждая со своей ценой и сроком (как цена списком), не перевод; у строки нет срока —
  перевод, как было;
- off — как было.
Через обработчик эндпоинта; LLM — подделка клиента Ollama, решение проходит настоящий `choose()`.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router, turn_understanding
from messengers_router.mess_types import RouteDecision
from tests.test_facet_by_llm_class import _FACET_MARK, _dialog, _relabels
from tests.spans_oracle import SPANS_MARK, spans_response
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_PRICE_TEMPLATE = "Не нашёл актуальную стоимость"


@pytest.mark.parametrize(
    "reply,facet,label,policy",
    [
        ("кровь на глюкозу могу ли сдать через 15 минут?", "booking", "ADDRESS", "turn_facet_where"),
        ("глюкозу во сколько можно сдать?", "where", "ADDRESS", "turn_facet_where"),
        ("глюкоза — что нужно сделать перед ней?", "prepare", "PREPARE", "turn_facet_prepare"),
        ("глюкоза — это из вены берут или из пальца?", "none", "OTHER", "turn_facet_none_offer"),
    ],
)
def test_price_turn_facet_without_price_reader_goes_to_its_reader(reply, facet, label, policy, mini_prod_world, monkeypatch):  # noqa: F811
    out, _ = _dialog(mini_prod_world, monkeypatch, [(reply, "PRICE", facet)], switch="on")
    turn = out[0]
    tel = turn["telemetry"]
    assert tel.get("tu_facet") == facet and tel.get("tu_facet_src") == "llm", tel
    assert tel.get("tu_label_final") == label, (tel, turn["text"])
    assert f"{policy}:PRICE>{label}" in _relabels(turn), _relabels(turn)
    assert _PRICE_TEMPLATE not in turn["text"], turn["text"]


@pytest.mark.parametrize("facet", ["price", "bare", "availability"])
def test_price_reader_facets_stay_a_price(facet, mini_prod_world, monkeypatch):  # noqa: F811
    out, _ = _dialog(mini_prod_world, monkeypatch, [("глюкоза", "PRICE", facet)], switch="on")
    tel = out[0]["telemetry"]
    assert tel.get("tu_label_final") == "PRICE", tel
    assert not [step for step in _relabels(out[0]) if step.startswith("turn_facet_")], _relabels(out[0])


def test_off_keeps_the_price_path(mini_prod_world, monkeypatch):  # noqa: F811
    out, fake = _dialog(mini_prod_world, monkeypatch, [("кровь на глюкозу могу ли сдать через 15 минут?", "PRICE", "booking")], switch="off")
    assert not [c for c in fake.calls if _FACET_MARK in c], "при off вопрос о типе не задаётся"
    assert not [step for step in _relabels(out[0]) if step.startswith("turn_facet_")], _relabels(out[0])


def test_answer_to_price_button_question_stays_a_price(monkeypatch):
    # «ЭКГ самара» после «Цена анализа»: LLM видит «где», но тему задала кнопка — путь цены.
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_FACET_BY_LLM", "on", raising=False)

    async def where(_text: str) -> str:
        return "where"

    monkeypatch.setattr(turn_understanding, "llm_facet", where)
    decision = RouteDecision(label="PRICE", flags={"button_answer_topic"})
    assert asyncio.run(router._route_by_turn_facet(decision, user_text="ЭКГ самара")) is decision
    plain = RouteDecision(label="PRICE")
    assert asyncio.run(router._route_by_turn_facet(plain, user_text="ЭКГ самара")).label == "ADDRESS"


_CULTURE_NAME = "Посев мочи с идентификацией культур"
_CULTURE = f"{_CULTURE_NAME} — какой срок получения результата?"
_CULTURE_SPANS = {SPANS_MARK: lambda prompt: spans_response(prompt, [_CULTURE_NAME]).response}


def test_readiness_of_service_with_rows_of_different_deadlines_lists_them(mini_prod_world, monkeypatch):  # noqa: F811
    out, _ = _dialog(mini_prod_world, monkeypatch, [(_CULTURE, "PRICE", "readiness")], switch="on", extra_answers=_CULTURE_SPANS)
    turn = out[0]
    assert "readiness_from_mis" in (turn["telemetry"].get("tu_acted") or []), turn["telemetry"]
    text = turn["text"]
    assert "Посев мочи с идентификацией культур" in text, text
    assert "1 100 руб. — 6 дней" in text and "1 050 руб. — 3–5 дней" in text, text
    assert "оператор" not in text.lower(), text


def test_readiness_of_rows_with_different_deadlines_off_is_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    # Off — как было: у строк одного названия сроки разные — честный перевод.
    from messengers_router.services import _readiness_select as rs

    rows = [
        {"serviceName": "Посев мочи с идентификацией культур", "deadline": "6", "cost": 1100.0},
        {"serviceName": "Посев мочи с идентификацией культур", "deadline": "3-5", "cost": 1050.0},
    ]
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_FACET_BY_LLM", "off", raising=False)
    assert rs._deadline_items("Посев мочи с идентификацией культур", rows) == []
    monkeypatch.setattr(cfg, "MR_FACET_BY_LLM", "on", raising=False)
    assert [i["variant"] for i in rs._deadline_items("Посев мочи с идентификацией культур", rows)] == ["1 100 руб.", "1 050 руб."]
    rows[1]["deadline"] = " "
    assert rs._deadline_items("Посев мочи с идентификацией культур", rows) == []  # у строки нет срока — перевод, как было
