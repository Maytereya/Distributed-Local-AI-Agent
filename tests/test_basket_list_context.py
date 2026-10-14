"""Корзина из фраз: позиция списка с общей частью соседей (L-15b, шаг 6; класс «общая голова списка»).

Перечисление держит общее слово один раз: «витамин б9 и б12», «Витамины В1, В2, В6, В7» — у «б12», «В7» нет
слова «витамин»; «т3 и т4 свободные» — у «т3» нет «свободные». По самой позиции код не находит ничего, и
ответ — «Не распознал: б12» (g0054), хотя «Витамин B12» в прайсе есть. Это класс, а не анализ: общая часть
берётся у соседей по списку, позиция остаётся срезом реплики (контракт C), строку выбирает LLM. Инварианты:
- (а) при живой LLM позиция без кандидатов получает кандидатов по «голове соседа + позиция» / «позиция +
  хвост соседа», выбранная строка — с ценой, чужой цены нет;
- (б) при сбое LLM — «Не распознал», а не строка соседа с ценой;
- (в) позиция, у которой кандидаты по себе есть, контекст не получает (ничего не меняется);
- (г) второй короткий вопрос не берёт услугу, которой пациент не видел.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import _basket_select as bs
from tests.test_price_one_or_many_class import Case, _configure, _OracleLLM, _violations
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_B12 = "Витамин B12"
_FERRITIN = "Ферритин"
_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)
_REPLY = "сколько стоит ферритин, витамин д и б12"
_CASE = Case(_REPLY, "list", {"ферритин": (_FERRITIN,), "витамин д": (), "б12": (_B12,)}, strong=frozenset({"ферритин"}))


def _priced(text: str) -> set[str]:
    return {" ".join(m.group("name").split()) for m in _PRICED.finditer(text)}


def _items(*fragments: str) -> list[dict]:
    return [{"fragment": f, "service_name": f} for f in fragments]


@pytest.fixture
def catalog(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        from messengers_router.services import _prices_helpers as H

        yield H._catalog_rows(None)


def test_head_of_an_earlier_position_finds_the_row(catalog):
    from messengers_router.services._prices_helpers import _rank_price_rows

    assert _rank_price_rows(catalog, "б12", limit=6) == []  # позиция сама по себе — ничего
    names = [r["serviceName"] for r in bs._context_candidates(2, _items("ферритин", "витамин д", "б12"), catalog)]
    assert _B12 in names, names


def test_head_is_taken_from_the_nearest_position_with_two_words(catalog):
    items = _items("Витамины В1", "В2", "б12")
    names = [r["serviceName"] for r in bs._context_candidates(2, items, catalog)]
    assert _B12 in names, names


def test_single_word_neighbours_give_no_context(catalog):
    assert bs._context_candidates(1, _items("ферритин", "б12"), catalog) == []


def test_with_the_model_the_position_gets_its_own_price(mini_prod_world, monkeypatch):  # noqa: F811
    _configure(monkeypatch, "deploy_parse")
    oracle = _OracleLLM(_CASE, tuple(_CASE.positions))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog([_REPLY, "а сколько?"], "basket-list-context")
        assert list(world.net.attempts) == []
    assert _B12 in _priced(first["text"]), first["text"]
    violations = [v for v in _violations(_CASE, first["text"], "oracle", oracle.picked) if "витамин д" not in v]
    assert not violations, "\n".join([*violations, "--- ответ:", first["text"]])
    assert sorted(_priced(second["text"]) - _priced(first["text"])) == []  # контекст = показанное


def test_without_the_model_the_position_stays_unrecognized(mini_prod_world, monkeypatch):  # noqa: F811
    _configure(monkeypatch, "deploy_parse")
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        [turn] = run_dialog([_REPLY], "basket-list-context-down")
    assert _B12 not in _priced(turn["text"]), turn["text"]
