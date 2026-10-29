"""Ответ о клинике по данным: «похожее» пациенту не показывается (BUG-2026-10-21-SVC-SIMILAR-NOISE; п. 5 топ-5 разбора 95 цен;
выключатель `service_answer_no_similar`, off; включение — после окна (е), решение владельца).

Карточка деплоя 13 (`service_answer_from_data = on`): исход `similar` показывал несвязанные строки и чужую цену — «Уздг бцс
ребёнку» → справка 079/у с ценой, «ПЦР на аскаридоз отдельно» → «Да, есть: Ascaris IgG». Класс C разбора 95 («есть ли у вас X»
вне анализов; 19 вопросов) ответ по данным получает только вместе с ним. Карточка 26.10 первой версии (задача без исхода
«похожее») показала обратное: без него модель называет близкое ответом «да» («3D/4D УЗИ» → обычное УЗИ беременных), а на ходе
подбора анализов ответ о клинике перехватывал ответ подбора по данным («анализы на мочу» → «Да, есть: Лебеда (W15)»).

Инвариант под on: у модели «похожее» есть, пациенту — нет: близкое — честный оффер; ответ — только выбранные строки (название и
цена — из прайса); ход подбора анализов отвечает своим ответом по данным; промпт — в двух локациях, без вопросов эталона; off — как
было. Через обработчик эндпоинта; LLM — подделка настоящего типа.
"""

from __future__ import annotations

import pathlib

import pytest

from messengers_router import service_answer
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_service_answer_class import (  # noqa: F401 — фикстура
    CLAIM_QUESTIONS,
    FABRICATED,
    _RENDER_MARK,
    _SERVICE_ANSWER_MARK,
    _classifier,
    _free_text_answers,
    _number_of,
    answer_world,
    turn,
    verdict,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ON = {"MR_SERVICE_ANSWER_NO_SIMILAR": "on"}
_CHECKUP = "Прайс клиники: Ежегодный Чекап"


def _text(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def _similar(prompt: str) -> str:
    return verdict("similar", picks=(_number_of(prompt, _CHECKUP),))


def _picked(prompt: str) -> str:
    return verdict(picks=(_number_of(prompt, _CHECKUP),))


def _other_answers(service_answer) -> dict:
    """Место «OTHER от LLM» (`topic_other_offer`): под `test_assist_from_data = on` (прод с 24.10) ход подбора анализов под on
    уступает своему ответу по данным (`test_test_assist_turn_keeps_its_own_data_answer_under_on`), и исходы самого ответа о
    клинике видны там, где он на проде действует, — на OTHER от классификатора (`topic_by_llm`, прод с 22.10)."""

    return {_CLASSIFIER_MARK: _classifier("OTHER"), _SERVICE_ANSWER_MARK: service_answer, _RENDER_MARK: FABRICATED}


def test_exact_prompt_is_the_old_one_with_synthetic_examples_in_both_places():
    exact = _text("messengers_router/prompts/choose_service_answer_exact.txt")
    assert exact == _text("app_data/prompts/mr_choose_service_answer_exact.txt")
    base = _text("messengers_router/prompts/choose_service_answer.txt").splitlines()
    # Отличия — примеры 2–3: синтетика вместо двух вопросов эталона (g0509, g0277; Гудхарт) — и исход «general» (услуга не названа,
    # STEP-2026-10-29-PRICE-NOT-FOUND-BY-DATA).
    assert [line[:2] for line in exact.splitlines() if line not in base] == ["- ", "2.", "3."]
    assert '"general"' in exact
    assert "ФКС с биопсией" not in exact and "детей 2 лет" not in exact
    assert '"similar"' in exact  # исход у модели остаётся


@pytest.mark.parametrize(("value", "task"), [("on", "service_answer_exact"), ("off", "service_answer"), ("мусор", "service_answer")])
def test_task_by_switch(value, task, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_SERVICE_ANSWER_NO_SIMILAR", value, raising=False)
    assert (service_answer.EXACT_TASK if service_answer.no_similar() else service_answer.TASK).name == task


def test_similar_is_an_honest_offer_not_rows_under_on(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "Чекап для собаки делаете?", switch="on", answers=_other_answers(_similar), cfg_extra=_ON)

    assert out["telemetry"].get("svc_answer") == "no_data" and out["telemetry"].get("svc_answer_similar_hidden") == 1, out["telemetry"]
    assert "Есть похожие" not in out["text"] and "2 055 руб." not in out["text"], out["text"]
    assert "нет данных" in out["text"] and "Перевести?" in out["text"] and out["pending_offer"], out["text"]


def test_similar_is_shown_as_before_under_off(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "Чекап для собаки делаете?", switch="on", answers=_free_text_answers(_similar))

    assert out["telemetry"].get("svc_answer") == "similar", out["telemetry"]
    assert "Есть похожие" in out["text"], out["text"]


@pytest.mark.parametrize("reply", CLAIM_QUESTIONS[:4])
def test_no_data_is_an_honest_offer_under_on(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_other_answers(verdict("no_data")), cfg_extra=_ON)

    assert out["telemetry"].get("svc_answer") == "no_data", out["telemetry"]
    assert "нет данных" in out["text"] and "Перевести?" in out["text"] and out["pending_offer"], out["text"]
    assert FABRICATED not in out["text"] and _RENDER_MARK not in out["calls"]


def test_answer_prints_name_and_price_from_the_row_under_on(answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, "Чекап делаете?", switch="on", answers=_other_answers(_picked), cfg_extra=_ON)

    assert _SERVICE_ANSWER_MARK in out["calls"]
    assert out["text"].startswith(service_answer.ANSWER_HEADING) and "— Ежегодный Чекап — 2 055 руб." in out["text"], out["text"]


@pytest.mark.parametrize(("extra", "source"), [(_ON, "test_assist_data"), ({}, "service_answer")])
def test_test_assist_turn_keeps_its_own_data_answer_under_on(extra, source, answer_world, monkeypatch):  # noqa: F811
    # Ход подбора анализов (TEST_ASSIST, `test_assist_from_data = on`) под on отвечает своим ответом по данным: ответ о клинике
    # его не перехватывает (карточка 26.10: «анализы на мочу» → «Да, есть: Лебеда (W15)»). Под off — как было (ответ о клинике).
    out = turn(
        answer_world, monkeypatch, CLAIM_QUESTIONS[0], switch="on", answers=_free_text_answers(verdict("no_data")),
        cfg_extra={"MR_TEST_ASSIST_FROM_DATA": "on", **extra},
    )

    assert out["telemetry"].get("answer_source") == source, out["telemetry"]
