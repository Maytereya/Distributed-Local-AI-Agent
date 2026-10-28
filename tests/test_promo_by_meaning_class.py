"""Какая акция отвечает на вопрос — решает LLM по смыслу, а не счёт совпадений слов (STEP-2026-10-29-PROMO-BY-MEANING,
выключатель `promo_by_llm`).

Стенд деплоя 20 (`s20con_btn`, лист «Акции и скидки»): 17 провалов, из них 9 — шаблон или чужая акция: поиск акции считал
совпадения слов вопроса с названием и текстом (`news._score_promo`), и «А на первый приём есть скидка?» получал акцию на витамин
D, «Что входит в акцию госпитализация?» — весь список.

Инвариант класса: под `promo_by_llm = on` акция в ответе — та, что выбрала LLM среди действующих (`detail`); «такой нет» —
честный заголовок и список действующих (`absent`), без эха вопроса; общий вопрос — список; номер из показанного списка и «все» —
как было, без LLM; сбой LLM и «не об акциях» — прежний поиск по словам; off — как было. Сервис акций — тем же путём, что ход;
LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from messengers_router import choice, llm_runtime
from messengers_router.renderer import format_news_for_patient
from messengers_router.services import news as news_mod
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_promotions_flow import promo_env  # noqa: F401 — фикстура

_MARK = "Пациент спросил об акциях"


def _info(monkeypatch, text: str, *, switch: str, answer: str | None, entities: dict | None = None) -> tuple[dict, _LLM]:
    from messengers_router.runtime_config import config as cfg

    llm = _LLM({_MARK: answer} if answer is not None else {})
    monkeypatch.setattr(cfg, "MR_PROMO_BY_LLM", switch, raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    choice._CACHE.clear()
    return asyncio.run(news_mod.news_info(None, text, entities or {})), llm


def _titles(payload: dict) -> list[str]:
    return [x["title"] for x in payload["news"]]


def _option(prompt_titles: list[str], title: str) -> int:
    return prompt_titles.index(title) + 1


# Действующие акции фикстуры в порядке сервиса: чекап + витамин D, популярные анализы, офис на Ленина, доктор Шубин.
_ACTIVE = ["ЧЕКАП + ВИТАМИН D", "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%", "Скидка в офисе на Ленина", "Доктор Шубин в Самаре"]


def test_off_scores_words_as_before(promo_env, monkeypatch):  # noqa: F811
    payload, llm = _info(monkeypatch, "акция чекап", switch="off", answer=json.dumps({"picks": [2], "outcome": "promo"}))
    assert payload["mode"] == "detail" and _titles(payload)[0] == "ЧЕКАП + ВИТАМИН D"
    assert llm.calls == []


def test_promo_picked_by_meaning_is_printed(promo_env, monkeypatch):  # noqa: F811
    """Вопрос без слов названия: акцию выбирает смысл, код печатает её карточку."""

    pick = _option(_ACTIVE, "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%")
    payload, llm = _info(monkeypatch, "дешевле сдать кровь у вас можно?", switch="on", answer=json.dumps({"picks": [pick], "outcome": "promo"}))
    assert llm.calls == [_MARK]
    assert payload["mode"] == "detail" and _titles(payload) == ["ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%"]


def test_absent_promo_is_honest_without_echo(promo_env, monkeypatch):  # noqa: F811
    text = "есть скидка пенсионерам на массаж?"
    payload, _ = _info(monkeypatch, text, switch="on", answer=json.dumps({"picks": [], "outcome": "absent"}))
    assert payload["mode"] == "absent" and _titles(payload)
    rendered = format_news_for_patient(payload, {})
    assert rendered.startswith("Такой акции или скидки сейчас нет") and text not in rendered


def test_general_question_is_the_list(promo_env, monkeypatch):  # noqa: F811
    payload, _ = _info(monkeypatch, "что у вас по скидкам сейчас", switch="on", answer=json.dumps({"picks": [], "outcome": "list"}))
    assert payload["mode"] == "list" and set(_titles(payload)) == set(_ACTIVE)


@pytest.mark.parametrize("answer", [None, json.dumps({"picks": [], "outcome": "other_topic"}), "не json"])
def test_failure_or_other_topic_keeps_word_search(answer, promo_env, monkeypatch):  # noqa: F811
    on, _ = _info(monkeypatch, "акция чекап", switch="on", answer=answer)
    off, _ = _info(monkeypatch, "акция чекап", switch="off", answer=None)
    assert on == off


@pytest.mark.parametrize("text", ["все", "2"])
def test_structural_replies_skip_llm(text, promo_env, monkeypatch):  # noqa: F811
    context = {"_promo_context": {"titles": list(_ACTIVE)}}
    on, llm = _info(monkeypatch, text, switch="on", answer=json.dumps({"picks": [1], "outcome": "promo"}), entities=context)
    off, _ = _info(monkeypatch, text, switch="off", answer=None, entities=context)
    assert llm.calls == [] and on == off


def test_pick_outside_the_list_is_ignored(promo_env, monkeypatch):  # noqa: F811
    on, _ = _info(monkeypatch, "акция чекап", switch="on", answer=json.dumps({"picks": [99], "outcome": "promo"}))
    off, _ = _info(monkeypatch, "акция чекап", switch="off", answer=None)
    assert on == off
