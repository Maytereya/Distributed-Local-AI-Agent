"""Ответ о клинике по данным без исхода «похожее» (BUG-2026-10-21-SVC-SIMILAR-NOISE; п. 5 топ-5 разбора 95 цен; выключатель
`service_answer_no_similar`, off; включение — после окна (е), решение владельца).

Карточка деплоя 13 (`service_answer_from_data = on`): исход `similar` показывал несвязанные строки и чужую цену — «Уздг бцс
ребёнку» → справка 079/у с ценой, «ПЦР на аскаридоз отдельно» → «Да, есть: Ascaris IgG». Класс C разбора 95 («есть ли у вас X»
вне анализов, OTHER → «нет данных → оператор»; 19 вопросов) ответ по данным получает только вместе с ним.

Инвариант под on: исходов два — ответ по выбранным строкам (название и цена — из прайса, печатает код) или «нет данных» с честным
оффером; «похожее», которое модель всё же вернула, — не ответ (мусор: как сбой, место делает как было); промпт без «похожего»
лежит в двух локациях; off — прежняя задача с «похожим». Через обработчик эндпоинта; LLM — подделка настоящего типа.
"""

from __future__ import annotations

import pathlib

import pytest

from messengers_router import service_answer
from tests.test_service_answer_class import (  # noqa: F401 — фикстура
    CLAIM_QUESTIONS,
    FABRICATED,
    _RENDER_MARK,
    _SERVICE_ANSWER_MARK,
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


def test_exact_prompt_has_no_similar_outcome_and_lives_in_both_places():
    exact = _text("messengers_router/prompts/choose_service_answer_exact.txt")
    assert exact == _text("app_data/prompts/mr_choose_service_answer_exact.txt")
    assert '"similar"' not in exact and "похожие" not in exact.lower()
    assert "Близкая, но другая услуга" in exact and '"no_data", даже если других пунктов нет' in exact
    base = _text("messengers_router/prompts/choose_service_answer.txt").splitlines()
    # Отличия от прежнего промпта: правило 1 без «похожего» и примеры 2–3 — синтетика вместо двух вопросов эталона (Гудхарт).
    assert [line[:2] for line in exact.splitlines() if line not in base] == ["1.", "2.", "3."]
    assert "ФКС с биопсией" not in exact and "детей 2 лет" not in exact


@pytest.mark.parametrize(("value", "task"), [("on", "service_answer_exact"), ("off", "service_answer"), ("мусор", "service_answer")])
def test_task_by_switch(value, task, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_SERVICE_ANSWER_NO_SIMILAR", value, raising=False)
    assert (service_answer.EXACT_TASK if service_answer.no_similar() else service_answer.TASK).name == task
    assert ("similar" in service_answer.EXACT_TASK.outcomes) is False


def test_similar_from_the_model_is_not_an_answer_under_on(answer_world, monkeypatch):  # noqa: F811
    # Модель всё же вернула «похожее» с номером строки: на задаче без этого исхода — мусор, ответа «похожие» нет.
    similar = lambda prompt: verdict("similar", picks=(_number_of(prompt, _CHECKUP),))  # noqa: E731
    out = turn(answer_world, monkeypatch, "Чекап для собаки делаете?", switch="on", answers=_free_text_answers(similar), cfg_extra=_ON)

    assert out["telemetry"].get("svc_answer") == "failed", out["telemetry"]
    assert "Есть похожие" not in out["text"] and "2 055 руб." not in out["text"], out["text"]


def test_similar_is_shown_as_before_under_off(answer_world, monkeypatch):  # noqa: F811
    similar = lambda prompt: verdict("similar", picks=(_number_of(prompt, _CHECKUP),))  # noqa: E731
    out = turn(answer_world, monkeypatch, "Чекап для собаки делаете?", switch="on", answers=_free_text_answers(similar))

    assert out["telemetry"].get("svc_answer") == "similar", out["telemetry"]
    assert "Есть похожие" in out["text"], out["text"]


@pytest.mark.parametrize("reply", CLAIM_QUESTIONS[:4])
def test_no_data_is_an_honest_offer_under_on(reply, answer_world, monkeypatch):  # noqa: F811
    out = turn(answer_world, monkeypatch, reply, switch="on", answers=_free_text_answers(verdict("no_data")), cfg_extra=_ON)

    assert out["telemetry"].get("answer_source") == "service_answer", out["telemetry"]
    assert "нет данных" in out["text"] and "Перевести?" in out["text"] and out["pending_offer"], out["text"]
    assert FABRICATED not in out["text"] and _RENDER_MARK not in out["calls"]


def test_answer_prints_name_and_price_from_the_row_under_on(answer_world, monkeypatch):  # noqa: F811
    picked = lambda prompt: verdict(picks=(_number_of(prompt, _CHECKUP),))  # noqa: E731
    out = turn(answer_world, monkeypatch, "Чекап делаете?", switch="on", answers=_free_text_answers(picked), cfg_extra=_ON)

    assert _SERVICE_ANSWER_MARK in out["calls"]
    assert out["text"].startswith(service_answer.ANSWER_HEADING) and "— Ежегодный Чекап — 2 055 руб." in out["text"], out["text"]
