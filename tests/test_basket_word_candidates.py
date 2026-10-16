"""Корзина из фраз: кандидаты слабой позиции по отдельным словам, когда по фразе целиком код не нашёл
ничего (L-15b, шаг 6; живой прогон `on`, 14.10).

Под `turn_parse = on` бот выделяет позицию «Т4 свободный», «кальций общий», но строгий поиск по фразе целиком
(`_rank_price_rows`) не находит строк — в прайсе «сТ4 (FT4) Свободный тироксин», «Са+ (кальций)», — и ответ
«Не распознал» (g0145, g0175, g0348, g0518). Это дыра поиска кандидатов, а не синонимов: слова позиции в
названиях есть. Инварианты:
- (а) при живой LLM позиция без кандидатов по фразе получает кандидатов по своим словам, строку выбирает LLM
  (оракул) — цена своей строки, чужой нет;
- (б) при сбое LLM — как раньше: «Не распознал», а не варианты по словам (по словам — не та услуга с ценой);
- (в) уточнение, которого в прайсе нет («приём терапевта по ОМС», П2), кандидатов по словам не получает;
- (г) второй короткий вопрос («а сколько?») не берёт услугу, которой пациент не видел.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import _basket_select as bs
from tests.test_price_one_or_many_class import Case, _configure, _OracleLLM, _violations
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_FT4 = "сТ4 (FT4) Свободный тироксин"
_FERRITIN = "Ферритин"
_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)
_REPLY = "сколько стоит ферритин и т4 свободный"
_CASE = Case(_REPLY, "list", {"ферритин": (_FERRITIN,), "т4 свободный": (_FT4,)}, strong=frozenset({"ферритин"}))


def _priced(text: str) -> set[str]:
    return {" ".join(m.group("name").split()) for m in _PRICED.finditer(text)}


@pytest.fixture
def catalog(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        from messengers_router.services import _prices_helpers as H

        yield H._catalog_rows(None)


@pytest.mark.parametrize(("fragment", "wanted"), [("т4 свободный", _FT4), ("кальций общий", "Са+ (кальций)")])
def test_words_of_the_position_find_rows_the_whole_phrase_misses(fragment, wanted, catalog):
    from messengers_router.services._prices_helpers import _rank_price_rows

    assert _rank_price_rows(catalog, fragment, limit=6) == []  # фраза целиком — ничего
    names = [row["serviceName"] for row in bs._word_candidates(fragment, catalog)]
    assert wanted in names, names
    assert len(names) <= bs._PER_ITEM * 2


def test_generic_words_bring_no_candidates(catalog):
    assert bs._word_candidates("анализ на", catalog) == []


def test_with_the_model_the_position_gets_its_own_price(mini_prod_world, monkeypatch):  # noqa: F811
    _configure(monkeypatch, "deploy")
    oracle = _OracleLLM(_CASE, tuple(_CASE.positions))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog([_REPLY, "а сколько?"], "basket-word-candidates")
        assert list(world.net.attempts) == []
    violations = _violations(_CASE, first["text"], "oracle", oracle.picked)
    assert not violations, "\n".join([*violations, "--- ответ:", first["text"]])
    assert _FT4 in _priced(first["text"])
    assert sorted(_priced(second["text"]) - _priced(first["text"])) == []  # контекст = показанное


def test_without_the_model_the_position_stays_unrecognized(mini_prod_world, monkeypatch):  # noqa: F811
    _configure(monkeypatch, "deploy")
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        [turn] = run_dialog([_REPLY], "basket-word-candidates-down")
    assert _FT4 not in _priced(turn["text"])
    assert not (_priced(turn["text"]) - {_FERRITIN}), turn["text"]


def test_unsatisfiable_qualifier_gets_no_word_candidates(mini_prod_world, monkeypatch):  # noqa: F811
    reply = "сколько стоит ферритин и приём терапевта по ОМС"
    case = Case(reply, "list", {"ферритин": (_FERRITIN,), "приём терапевта по ОМС": ()}, strong=frozenset({"ферритин"}))
    _configure(monkeypatch, "deploy")
    oracle = _OracleLLM(case, tuple(case.positions))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        [turn] = run_dialog([reply], "basket-word-candidates-oms")
    assert not [name for name in _priced(turn["text"]) if "терапевт" in name.lower()], turn["text"]
