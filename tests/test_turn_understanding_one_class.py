"""Ход (б), этап 2: одно понимание хода — «список ли» решается один раз (STEP-2026-10-21-B-STAGE2-ONE).

Консилиум 20.10 §1–2, решения владельца 20.10 (б) 4 и 21.10 (этап 2 — с правкой K1). Инвариант класса:
- один публичный объект: разбор реплики (`PhraseParse`) читатели берут только через понимание хода
  (`turn_understanding.parse_for`) — `turn_parse.acting_for` / `phrases_now` / `phrases` вне производителя — красный;
- под `turn_understanding_one = on` «список ли» на пути правил — одно решение (`turn_understanding.rules_list`): вердикт
  понимания этой реплики, без него — правило понимания с защитами K1; корзина правил (R1) и подбор анализов (R7) его читают и
  сами не решают (у R7 «не список» — кандидаты сегментатора для ответа LLM, K2 не снимается) (разбор 21.10: в 19 из 21 расхождения корзина без K1 резала одну услугу на чужие строки);
- защита K1 «одно название покрывает реплику» не для явного перечня своих названий («Витамин В1, Витамин В6» — не
  «Расширенный комплексный анализ на витамины (…)»), а для одного названия с запятыми — в силе;
- под `off` — как было.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from messengers_router import turn_understanding
from messengers_router.services import _prices_helpers as H
from messengers_router.services import lab_tests, prices
from messengers_router.turn_understanding import TurnUnderstanding

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"
_VITAMINS = [
    {"serviceId": 8350, "serviceHomecode": "5420", "cost": 22500.0, "deadline": "7-10",
     "serviceName": "Расширенный комплексный анализ на витамины (A, бета-каротин, D, E, K, C, B1, B2, B3, B5, B6, B9, B12)"},
    {"serviceId": 8365, "serviceHomecode": "5435", "cost": 2240.0, "deadline": "6-8", "serviceName": "Витамин B6 (пиридоксаль-5-фосфат)"},
    {"serviceId": 8367, "serviceHomecode": "5437", "cost": 2260.0, "deadline": "6-8", "serviceName": "Витамин B1(тиаминпирофосфат)"},
    {"serviceId": 1, "serviceHomecode": "1", "cost": 900.0, "deadline": "1", "serviceName": "Прием (осмотр, консультация) врача-терапевта первичный"},
    {"serviceId": 2, "serviceHomecode": "2", "cost": 800.0, "deadline": "1", "serviceName": "Осмотр (консультация) врача-терапевта повторный"},
]


@pytest.fixture
def one(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    def switch(value: str) -> None:
        monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_ONE", value, raising=False)

    return switch


def test_readers_take_the_parse_only_through_the_understanding():
    banned = {"acting_for", "phrases_now", "phrases"}
    found = []
    for path in sorted(_CORE.rglob("*.py")):
        if path.name in {"turn_parse.py", "turn_understanding.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in banned and getattr(node.value, "id", "") == "turn_parse":
                found.append(f"{path.name}:{node.lineno} turn_parse.{node.attr}")
    assert not found, f"разбор мимо понимания хода (нужно turn_understanding.parse_for): {found}"


@pytest.mark.parametrize(
    "text",
    ["Подскажите, пожалуйста, цены на анализы: Витамин В1, Витамин В6", "Подскажите цены на анализы:\nВитамин В1\nВитамин В6"],
)
def test_explicit_list_of_own_names_is_a_list_under_on(one, text):
    one("off")
    assert H.price_list_services(text, rows=_VITAMINS) == ((), "one_catalog_name")
    one("on")
    names, rule = H.price_list_services(text, rows=_VITAMINS)
    assert rule.startswith("list_") and len(names) == 2, (names, rule)


def test_one_name_with_commas_keeps_k1_under_on(one):
    one("on")
    assert H.price_list_services("сколько стоит прием, осмотр и консультация терапевта", rows=_VITAMINS) == ((), "one_catalog_name")


def test_rules_list_reads_the_understanding_of_this_reply(one, monkeypatch):
    text = "цена анализа на дефицит антитромбина"
    monkeypatch.setattr(turn_understanding, "price_list_services", lambda *_a, **_k: pytest.fail("решено повторно"))
    token = turn_understanding._CURRENT.set(TurnUnderstanding(services_rule="segment_unrecognized", reply=text))
    try:
        assert turn_understanding.rules_list(text) is False
        turn_understanding._CURRENT.set(TurnUnderstanding(services_rule="list_split", reply=text))
        assert turn_understanding.rules_list(text) is True
    finally:
        turn_understanding._CURRENT.reset(token)


def test_rules_list_without_understanding_uses_k1_rule(monkeypatch):
    calls = []
    monkeypatch.setattr(turn_understanding, "price_list_services", lambda text, rows=None: calls.append(text) or ((), "one_catalog_name"))
    assert turn_understanding.rules_list("не реплика хода") is False and calls == ["не реплика хода"]


def _basket(monkeypatch, *, verdict: bool):
    items = [
        {"service_name": "Антитела в подарок", "prices": [_VITAMINS[1]], "fragment": "на антитела"},
        {"service_name": "Лямблии IgG", "prices": [_VITAMINS[2]], "fragment": "IgG"},
    ]
    monkeypatch.setattr(prices, "_multi_price_items", lambda *_a, **_k: (items, []))
    monkeypatch.setattr(turn_understanding, "parse_for", lambda _text: None)
    monkeypatch.setattr(turn_understanding, "rules_list", lambda *_a, **_k: verdict)

    async def rows(query, items, unrecognized, retail_rows, llm_allowed):
        return items, unrecognized

    monkeypatch.setattr(prices, "choose_basket_rows", rows)
    return asyncio.run(prices._multi_price_payload("цена анализа на антитела IgG", "цена анализа на антитела IgG", [], {}))


def test_basket_reader_obeys_the_one_verdict_under_on(one, monkeypatch):
    one("on")
    assert _basket(monkeypatch, verdict=False) is None
    one("off")
    assert _basket(monkeypatch, verdict=False) is not None  # как было: корзина решает сама


def _test_assist(monkeypatch, *, verdict: bool) -> dict:
    monkeypatch.setattr(lab_tests.api_price, "load_price_by_region", lambda _region: list(_VITAMINS))
    monkeypatch.setattr(lab_tests, "_resolve_multi_price_items", lambda *_a, **_k: ([], []))  # не список — кандидаты сегментатора
    monkeypatch.setattr(turn_understanding, "parse_for", lambda _text: None)
    monkeypatch.setattr(turn_understanding, "rules_list", lambda *_a, **_k: verdict)
    items = [{"service_name": n, "fragment": n, "prices": [r]} for n, r in (("B1", _VITAMINS[2]), ("B6", _VITAMINS[1]))]
    monkeypatch.setattr(lab_tests, "_multi_price_items", lambda *_a, **_k: (items, []))
    return asyncio.run(lab_tests.test_assist(None, "Витамин В1, Витамин В6", {}))


def test_test_assist_reader_obeys_the_one_verdict_under_on(one, monkeypatch):
    """Список — позиции корзины (с K3) по одному решению; не список — не позиции корзины (кандидаты сегментатора, K2)."""

    one("on")
    assert "multi-item" in str(_test_assist(monkeypatch, verdict=True).get("note"))
    assert "multi-item" not in str(_test_assist(monkeypatch, verdict=False).get("note"))
