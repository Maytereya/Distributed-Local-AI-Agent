"""«Есть ли услуга» — решает LLM по данным клиники; оператор — только если данных нет (класс 2, решение владельца 21.10).

BUG-2026-10-21-SERVICE-AVAILABILITY-BY-DATA, g0277 «Делают ли ФКС с биопсией?»: строка прайса «Биопсия при ФКС» и строка во
всех трёх памятках ФКС есть, а бот отвечал «врачей не нашёл» (правила) или переводил на оператора (OTHER от LLM). Инвариант
под `service_answer_from_data = on`: в местах, где бот сдаётся без данных, — OTHER от LLM (`topic_other_offer`), «врачей по
услуге не нашёл», вопрос без данных хода (L-05), — сначала данные клиники: ответ или близкое — из пунктов прайса и памяток,
выбранных LLM (текст печатает код, цена — из прайса, из памятки — без цены); «нет данных», «не о клинике», сбой — прежний
ответ места. Off — как было, без вызова.
"""

from __future__ import annotations

import json

import pytest

from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_service_answer_class import CHECK_MARK, _SERVICE_ANSWER_MARK, _classifier, _number_of, answer_world, confirm_all, turn, verdict  # noqa: F401

_G0277 = "Делают ли фкс с биопсией?"
_PRICE_ROW = "Прайс клиники: Биопсия при ФКС"
_MEMO_ROW = "Памятка «ПАМЯТКА ФКС без наркоза»: Биопсия при КС"
_OTHER_OFFER = "По этому вопросу у меня нет данных — точно ответит оператор."
_TOPIC_BY_LLM = {"MR_TOPIC_BY_LLM": "rule_doctor_info_specialty"}


def _pick(*rows: str):
    return lambda prompt: verdict(picks=tuple(_number_of(prompt, row) for row in rows))


def _other(service_answer) -> dict:
    return {_CLASSIFIER_MARK: _classifier("OTHER"), _SERVICE_ANSWER_MARK: service_answer, CHECK_MARK: confirm_all}


# --------------------------------------------------------------------------- OTHER от LLM (`topic_by_llm`)


def test_other_offer_answers_from_price_row(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(_pick(_PRICE_ROW)), cfg_extra=_TOPIC_BY_LLM)

    assert out["telemetry"].get("svc_answer_site") == "other_offer", out["telemetry"]
    assert out["text"].startswith("Да, по данным клиники это есть:"), out["text"]
    assert "— Биопсия при ФКС (без стоимости исследования) — 1 200 руб." in out["text"], out["text"]
    assert _OTHER_OFFER not in out["text"] and not out["handoff"]


def test_other_offer_answers_from_memo_line_without_its_price(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(_pick(_MEMO_ROW)), cfg_extra=_TOPIC_BY_LLM)

    assert "в памятке клиники «ПАМЯТКА ФКС без наркоза»: «Биопсия при КС (без стоимости исследования)»" in out["text"], out["text"]
    assert "900" not in out["text"]  # цена памятки — не цена: цены печатает прайс


def test_similar_is_shown_with_an_honest_offer(answer_world, monkeypatch):  # noqa: F811
    similar = lambda prompt: json.dumps({"picks": [_number_of(prompt, _PRICE_ROW)], "outcome": "similar", "search_terms": []})  # noqa: E731
    out = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(similar), cfg_extra=_TOPIC_BY_LLM)

    assert out["text"].startswith("Точно такой услуги в данных клиники не нашёл. Есть похожие:"), out["text"]
    assert "Перевести?" in out["text"] and out["pending_offer"]


@pytest.mark.parametrize(
    "service_answer", [verdict("no_data"), verdict("not_about_clinic"), "не json", ConnectionError("тест: LLM недоступна")]
)
def test_without_data_the_offer_stays(service_answer, answer_world, monkeypatch):  # noqa: F811
    off = turn(answer_world, monkeypatch, _G0277, switch="off", answers=_other(verdict("no_data")), cfg_extra=_TOPIC_BY_LLM)
    on = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(service_answer), cfg_extra=_TOPIC_BY_LLM)

    assert off["text"].startswith(_OTHER_OFFER), off["text"]
    assert on["text"] == off["text"]
    assert _SERVICE_ANSWER_MARK not in off["calls"]


def test_second_round_searches_the_names_suggested_by_llm(answer_world, monkeypatch):  # noqa: F811
    asked: list[str] = []

    def answer(prompt: str) -> str:
        asked.append(prompt)
        if "Ежегодный Чекап" not in prompt:  # первый круг: пункта нет — LLM называет, как услуга записана в прайсе
            return verdict("no_data", terms=("Ежегодный Чекап",))
        return verdict(picks=(_number_of(prompt, "Прайс клиники: Ежегодный Чекап"),))

    out = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(answer), cfg_extra=_TOPIC_BY_LLM)

    assert len(asked) == 2 and out["telemetry"].get("svc_answer_rounds") == 2, out["telemetry"]
    assert "— Ежегодный Чекап — 2 055 руб." in out["text"], out["text"]


# --------------------------------------------------------------------------- «врачей по услуге не нашёл» (без `topic_by_llm`)


def test_doctors_not_found_answers_from_data(answer_world, monkeypatch):  # noqa: F811
    off = turn(answer_world, monkeypatch, _G0277, switch="off", answers=_other(_pick(_PRICE_ROW)))
    on = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(_pick(_PRICE_ROW)))

    assert off["telemetry"].get("tu_label_final") == "DOCTOR_INFO" and "Биопсия" not in off["text"], off["text"]
    assert on["telemetry"].get("svc_answer_site") == "doctors_by_service", on["telemetry"]
    assert "— Биопсия при ФКС (без стоимости исследования) — 1 200 руб." in on["text"], on["text"]


def test_doctors_not_found_without_data_is_as_before(answer_world, monkeypatch):  # noqa: F811
    off = turn(answer_world, monkeypatch, _G0277, switch="off", answers=_other(verdict("no_data")))
    on = turn(answer_world, monkeypatch, _G0277, switch="on", answers=_other(verdict("no_data")))

    assert on["text"] == off["text"]


# --------------------------------------------------------------------------- вопрос без данных хода (L-05)


def _question_kind(monkeypatch):
    from messengers_router.services import _empty_evidence_reply

    async def question(_text: str) -> str:
        return "QUESTION"

    monkeypatch.setattr(_empty_evidence_reply, "classify_empty_evidence_turn", question)


@pytest.mark.parametrize("service_answer", [verdict("no_data"), ConnectionError("тест: LLM недоступна")])
def test_empty_evidence_question_without_data_keeps_the_offer(service_answer, answer_world, monkeypatch):  # noqa: F811
    _question_kind(monkeypatch)
    reply = "Есть ли у вас парковка?"
    off = turn(answer_world, monkeypatch, reply, switch="off", answers=_other(verdict("no_data")))
    on = turn(answer_world, monkeypatch, reply, switch="on", answers=_other(service_answer))

    assert off["telemetry"].get("answer_source") == "empty_evidence", off["telemetry"]
    assert on["text"] == off["text"] and "нет данных" in on["text"]


def test_empty_evidence_question_with_data_answers(answer_world, monkeypatch):  # noqa: F811
    _question_kind(monkeypatch)

    def answer(prompt: str) -> str:
        if "Ежегодный Чекап" not in prompt:
            return verdict("no_data", terms=("Ежегодный Чекап",))
        return verdict(picks=(_number_of(prompt, "Прайс клиники: Ежегодный Чекап"),))

    out = turn(answer_world, monkeypatch, "Есть ли у вас парковка?", switch="on", answers=_other(answer))

    assert out["telemetry"].get("svc_answer_site") == "empty_evidence", out["telemetry"]
    assert out["text"].startswith("Да, по данным клиники это есть:"), out["text"]
