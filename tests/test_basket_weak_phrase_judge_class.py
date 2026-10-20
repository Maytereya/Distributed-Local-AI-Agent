"""Слабая фраза корзины со словом вне прайса — суд LLM, не вердикт гарда по буквам (BUG-2026-10-20-BASKET-WEAK-PHRASE-GUARD).

Кикофф 20.10, §2, узкое место (4): «ферритин первый раз и глюкоза» → только глюкоза. Окна разбора на проде выключены
(решение 16.10), фраза разбора — «ферритин первый раз»; она слабая (дословного названия в прайсе нет), а словарный гард П2
(`query_has_unsatisfiable_qualifier`) считает «первый», «раз» уточнением, которого в прайсе нет, — «Не распознал» без LLM.

Под `basket_weak_phrase_judge = on`: внутри слабой фразы — сильное по данным МИС название («ферритин»), позиция идёт на суд LLM
корзины, как сильная фраза с условием (`judge`, правило 5 промпта): разговор («первый раз») — цена, условие («по ОМС») —
нет. Инварианты (через обработчик эндпоинта, мини-прод, набор деплоя, оракул LLM с правилом 5):
- разговорное слово рядом с названием — цена обеих позиций; под off — как было (ферритин без цены);
- условие, которого клиника не оказывает, — без цены, «Не распознал» словами пациента, как под off;
- сбой выбора строк — позиция без цены (не цена «почти той» услуги), как под off;
- список без слов вне прайса — ответ тот же;
- суд не сужает список (дополнение 20.10, карточка корзин dev): слова рядом с названием, которые сами называют услугу — сильное
  окно другого размера («бактериоскопия мазка цитология …») или слова прайса («тимофеевка луговая ферритин …»), — не условие:
  пациент склеил перечень без запятой, ответ как под off; «Не распознал» — словами пациента в его порядке.
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from messengers_router.services._turn_phrases import strong_inside
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _PHRASES_MARK, _ROWS_MARKS

_FERRITIN = "Ферритин — 530"
_GLUCOSE = "Глюкоза — 190"
_CONVERSATIONAL = (
    "сколько стоит ферритин первый раз и глюкоза",
    "сколько стоит ферритин первый раз, глюкоза",
    "ферритин первый раз и глюкоза сколько стоят",
)
_CONDITION = ("сколько стоит ферритин по омс и глюкоза",)


class _RuleFive:
    """LLM: разбор выбирает фразы, в которых ровно одна из услуг; выбор строк — правило 5 (позицию с условием «омс» не
    брать); `down` — выбор строк недоступен."""

    def __init__(self, *, down: bool = False) -> None:
        self.down = down
        self.rows_prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _PHRASES_MARK in seen:
            picks = [n for n, text in options if bool(re.search("ферритин", text.lower())) != bool(re.search("глюкоз", text.lower()))]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if any(mark in seen for mark in _ROWS_MARKS) and not self.down:
            self.rows_prompts.append(prompt)
            picks = [n for n, text in options if "омс" not in text.split(" → ", 1)[0].lower()]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _turn(reply: str, world, monkeypatch, *, judge: str, llm: _RuleFive) -> dict:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    monkeypatch.setattr(cfg, "MR_BASKET_WEAK_PHRASE_JUDGE", judge, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(world, "day") as frozen:
        session = "bwpj-" + judge + "-" + re.sub(r"\W+", "_", reply)
        [turn] = run_dialog([reply], session)
        assert list(frozen.net.attempts) == []
    return turn


@pytest.mark.parametrize("reply", _CONVERSATIONAL)
def test_conversational_word_beside_the_name_keeps_both_prices(reply, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _RuleFive()
    on = _turn(reply, mini_prod_world, monkeypatch, judge="on", llm=llm)
    off = _turn(reply, mini_prod_world, monkeypatch, judge="off", llm=_RuleFive())

    assert _FERRITIN in on["text"] and _GLUCOSE in on["text"], on["text"]
    assert "[добавлено: «первый раз»]" in llm.rows_prompts[0]
    assert _FERRITIN not in off["text"], off["text"]  # как было: «Не распознал: ферритин первый раз»


@pytest.mark.parametrize("reply", _CONDITION)
def test_a_condition_the_clinic_does_not_offer_gives_no_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(reply, mini_prod_world, monkeypatch, judge="on", llm=_RuleFive())
    off = _turn(reply, mini_prod_world, monkeypatch, judge="off", llm=_RuleFive())

    assert _FERRITIN not in on["text"] and _GLUCOSE in on["text"], on["text"]
    assert "Не распознал: ферритин по омс" in on["text"], on["text"]  # словами пациента
    assert _FERRITIN not in off["text"]


@pytest.mark.parametrize("reply", _CONVERSATIONAL[:2])
def test_failed_row_choice_gives_no_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(reply, mini_prod_world, monkeypatch, judge="on", llm=_RuleFive(down=True))

    assert _FERRITIN not in on["text"] and "Ферритин —" not in on["text"], on["text"]


@pytest.mark.parametrize("reply", ["сколько стоит ферритин и глюкоза", "сколько стоит ферритин, глюкоза"])
def test_lists_without_outside_words_do_not_change(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(reply, mini_prod_world, monkeypatch, judge="on", llm=_RuleFive())
    off = _turn(reply, mini_prod_world, monkeypatch, judge="off", llm=_RuleFive())

    assert on["text"] == off["text"] and _FERRITIN in on["text"]


def test_strong_inside_finds_one_name_and_the_words_beside(mini_prod_world):  # noqa: F811
    from agent_logic_2.nayka_api import api_price

    from messengers_router.services import _prices_helpers as H

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        name, found, rest = strong_inside("ферритин первый раз", rows)
        assert (name, rest) == ("ферритин", "первый раз") and found
        assert strong_inside("ферритин", rows) is None  # слов вокруг нет — суда нет
        assert strong_inside("ферритин и глюкоза", rows) is None  # два разных названия — склейка, как было
        assert strong_inside("первый раз", rows) is None


class _GluedList:
    """LLM: разбор выбирает ровно фразы `cut` (склейку двух услуг без запятой — одной фразой, как прод-LLM на r0121), выбор
    строк корзины отмечает все варианты — самый уступчивый суд: что ему подали, то и станет ценой."""

    def __init__(self, cut: tuple[str, ...]) -> None:
        self.cut = {_normalise_catalog_text(text) for text in cut}

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _PHRASES_MARK in seen:
            picks = [n for n, text in options if _normalise_catalog_text(text) in self.cut]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if any(mark in seen for mark in _ROWS_MARKS):
            return GenerateResponse(model="m", response=json.dumps({"picks": [n for n, _text in options]}), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


# Перечень без разделителя внутри фразы разбора: рядом с сильным названием — другая услуга. Сильное окно другого размера
# («цитология» при «бактериоскопия мазка») и названия словами прайса («тимофеевка луговая», «антитела к рецепторам ттг»).
_GLUED = (
    ("сколько стоит бактериоскопия мазка цитология после менструации, глюкоза", ("бактериоскопия мазка цитология после менструации", "глюкоза")),
    ("сколько стоит тимофеевка луговая ферритин первый раз, глюкоза", ("тимофеевка луговая ферритин первый раз", "глюкоза")),
    ("сколько стоит антитела к рецепторам ттг ферритин первый раз, глюкоза", ("антитела к рецепторам ттг ферритин первый раз", "глюкоза")),
)


@pytest.mark.parametrize("reply, cut", _GLUED)
def test_a_service_beside_the_name_is_not_a_condition(reply, cut, mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(reply, mini_prod_world, monkeypatch, judge="on", llm=_GluedList(cut))
    off = _turn(reply, mini_prod_world, monkeypatch, judge="off", llm=_GluedList(cut))

    assert on["text"] == off["text"], on["text"]  # суд не сужает список: вторая услуга не пропадает молча
    assert f"Не распознал: {cut[0]}" in on["text"], on["text"]  # словами пациента, целиком
    assert _GLUCOSE in on["text"]


@pytest.mark.parametrize("reply", ["сколько стоит по омс ферритин и глюкоза"])
def test_not_found_keeps_the_patient_word_order(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _turn(reply, mini_prod_world, monkeypatch, judge="on", llm=_RuleFive())

    assert _FERRITIN not in on["text"] and _GLUCOSE in on["text"], on["text"]
    assert "Не распознал: омс ферритин." in on["text"], on["text"]  # фраза разбора как есть; было «ферритин омс» — окно + условие


def test_strong_inside_takes_no_other_service_for_a_condition(mini_prod_world):  # noqa: F811
    from agent_logic_2.nayka_api import api_price

    from messengers_router.services import _prices_helpers as H

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        for _reply, (glued, _glucose) in _GLUED:
            assert strong_inside(glued, rows) is None, glued
        assert strong_inside("ферритин первый раз", rows)[2] == "первый раз"  # слова вне прайса — суд, как раньше
