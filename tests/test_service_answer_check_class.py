"""Ответ по данным: «да» и «похожее» — только после проверки исхода (BUG-2026-10-21-SVC-SIMILAR-NOISE).

Живые карточки деплоя 13 (`d13_svc_*`, `service_answer_from_data = on`): «Есть ли ПЦР на аскаридоз отдельно?» → «Да, есть:
Ascaris IgG» (антитела, не ПЦР); «похожее» — по слову («поставить систему» → «индукционная система SIS»), справки на «кровь
ребёнку», строки памяток о подготовке; голое «Уздг бцс ребёнку 8 лет» → справка 079/у; «где: по записи» у ЭКГ. Инвариант
класса: первый выбор держит полноту кандидатов, проверка исхода (`choose(service_answer_check)`) — точность: «да» — только
пункты, отвечающие полностью (тот же объект, всё уточнение — метод, материал, возраст); «похожее» — только строки прайса того
же объекта в другом варианте, строки памяток — никогда; не подтверждено — честный оффер; сбой проверки — как было у места.
Голое название — максимум только по подтверждённой строке; «где» у неанализа не утверждается.
"""

from __future__ import annotations

import json

import pytest

from tests.test_service_answer_class import (  # noqa: F401 — фикстура
    _SERVICE_ANSWER_MARK,
    CHECK_MARK,
    FABRICATED,
    _classifier,
    _number_of,
    _RENDER_MARK,
    answer_world,
    confirm_all,
    turn,
    verdict,
)
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK

_G0277 = "Делают ли фкс с биопсией?"
_ROW = "Прайс клиники: Биопсия при ФКС"
_OTHER_ROW = "Прайс клиники: Тотальная фиброколоноскопия"
_MEMO = "Памятка «ПАМЯТКА ФКС без наркоза»: Биопсия при КС"
_TOPIC_BY_LLM = {"MR_TOPIC_BY_LLM": "rule_doctor_info_specialty"}
_CLAIM = "Вы берёте анализы крови у детей 2 лет?"


def _answers(first, check, label: str = "OTHER") -> dict:
    return {_CLASSIFIER_MARK: _classifier(label), _SERVICE_ANSWER_MARK: first, CHECK_MARK: check, _RENDER_MARK: FABRICATED}


def _pick(*rows: str, outcome: str = "none"):
    return lambda prompt: json.dumps({"picks": [_number_of(prompt, row) for row in rows], "outcome": outcome, "search_terms": []})


def _first_option(prompt: str) -> str:
    return json.dumps({"picks": [1], "outcome": "none", "search_terms": []})


def _other_offer(world, monkeypatch, first, check):
    return turn(world, monkeypatch, _G0277, switch="on", answers=_answers(first, check), cfg_extra=_TOPIC_BY_LLM)


def test_yes_needs_the_check_to_confirm(answer_world, monkeypatch):  # noqa: F811
    out = _other_offer(answer_world, monkeypatch, _pick(_ROW), verdict())

    assert out["telemetry"].get("svc_check") == "none", out["telemetry"]
    assert not out["text"].startswith("Да"), out["text"]  # проверка не подтвердила — «да» нельзя
    assert out["text"].startswith("По этому вопросу у меня нет данных"), out["text"]  # прежний честный оффер места


def test_same_object_other_variant_is_similar_not_yes(answer_world, monkeypatch):  # noqa: F811
    out = _other_offer(answer_world, monkeypatch, _pick(_ROW), _pick(_ROW, outcome="variant"))

    assert out["telemetry"].get("svc_answer") == "similar", out["telemetry"]
    assert out["text"].startswith("Точно такой услуги в данных клиники не нашёл. Есть похожие:"), out["text"]
    assert "Перевести?" in out["text"], out["text"]


def test_only_confirmed_rows_are_shown(answer_world, monkeypatch):  # noqa: F811
    out = _other_offer(answer_world, monkeypatch, _pick(_ROW, _OTHER_ROW), _pick(_ROW))

    assert out["text"].startswith("Да, по данным клиники это есть:"), out["text"]
    assert "Биопсия при ФКС" in out["text"] and "Тотальная фиброколоноскопия" not in out["text"], out["text"]


def test_memo_line_is_never_similar(answer_world, monkeypatch):  # noqa: F811
    out = _other_offer(answer_world, monkeypatch, _pick(_MEMO, outcome="similar"), _pick(_MEMO, outcome="variant"))

    assert "памятке" not in out["text"], out["text"]
    assert out["telemetry"].get("svc_answer") == "no_data", out["telemetry"]


def test_unconfirmed_claim_in_free_text_is_an_offer(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, _CLAIM, switch="on", answers=_answers(_first_option, verdict(), label="TEST_ASSIST"))

    assert out["telemetry"].get("answer_source") == "service_answer", out["telemetry"]
    assert not out["text"].startswith("Да") and FABRICATED not in out["text"], out["text"]
    assert "нет данных" in out["text"] and "Перевести?" in out["text"], out["text"]


@pytest.mark.parametrize("failure", [ConnectionError("тест: LLM недоступна"), "не json"])
def test_check_failure_is_as_before(failure, answer_world, monkeypatch):  # noqa: F811
    off = turn(answer_world, monkeypatch, _CLAIM, switch="off", answers=_answers(_first_option, confirm_all, label="TEST_ASSIST"))
    on = turn(answer_world, monkeypatch, _CLAIM, switch="on", answers=_answers(_first_option, failure, label="TEST_ASSIST"))

    assert on["text"] == off["text"]  # свободный текст, как без выключателя
    assert on["telemetry"].get("svc_check") == "failed", on["telemetry"]


# --------------------------------------------------------------------------- голое название


_OVERVIEW_MARK = "прислал ли пациент только название услуги"


def _overview_answers(pick: str, check) -> dict:
    overview = lambda prompt: json.dumps({"picks": [_number_of(prompt, pick)], "outcome": "none"})  # noqa: E731
    return {_CLASSIFIER_MARK: _classifier("ADDRESS"), _OVERVIEW_MARK: overview, CHECK_MARK: check}


def test_bare_name_needs_a_confirmed_row(answer_world, monkeypatch):  # noqa: F811
    off = turn(answer_world, monkeypatch, "ЭКГ", switch="off", answers=_overview_answers("ЭКГ", confirm_all))
    on = turn(answer_world, monkeypatch, "ЭКГ", switch="on", answers=_overview_answers("ЭКГ", verdict()))

    assert on["telemetry"].get("svc_overview") == "not_confirmed", on["telemetry"]
    assert on["text"] == off["text"]


def test_bare_name_of_not_an_analysis_claims_no_booking_rule(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "ЭКГ", switch="on", answers=_overview_answers("ЭКГ", confirm_all))

    assert out["text"].startswith("«ЭКГ»:"), out["text"]
    assert "по записи" not in out["text"], out["text"]  # ЭКГ — без записи; правила о записи в данных строки нет
    assert "напишите «записаться» или «где сделать»" in out["text"], out["text"]
