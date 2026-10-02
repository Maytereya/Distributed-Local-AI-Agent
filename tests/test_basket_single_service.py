"""Одна услуга не режется на «корзину» (BUG-2026-10-01-BASKET-SPLITS-ONE-SERVICE).

## Что произошло

«сколько стоит общий анализ крови полный» → корзина: общий анализ крови и «Полный
съемный протез системы «Экватор»». «приём хирурга на дому» → хирург, челюстно-лицевой
хирург и «Физиотерапия на дому». «узи щитовидной железы узи молочных желез» → в списке
УЗИ брюшной полости.

## Механизм

Реплику без запятых и переносов режет на позиции `_segment_items_by_catalog` (S2
корзины — для списков вида «ОАК ОАМ ГЛЮКОЗА ХОЛЕСТЕРИН»). До резки гард «анти-шинковки»
спрашивает, не одна ли это услуга: вся фраза находит строку прайса, и в её названии есть
каждое слово фразы. Название гард делил ПО ПРОБЕЛАМ: «(полный)(соэ,le,er» и
«врача-хирурга» считались одним словом, и «полный», «хирурга» в них не находились.
Гард отвечал «это список», и каждое слово-уточнение искалось в прайсе само: «полный» →
зубной протез, «дому» → физиотерапия. 01.10 так резались 82 из 234 названий прайса,
набранных словами.

Обратная сторона той же проверки — окно из нескольких слов списка «покрыто» составной
услугой, у которой эти слова стоят в скобках среди компонентов: «креатинин мочевина» →
«Оценка риска камнеобразования… (… креатинин, мочевина …)». Поэтому хотя бы одно слово
обязано совпасть с НАЗВАНИЕМ услуги, а не только со скобками.

## Инвариант

Фраза, которую прайс целиком называет одной строкой (все её слова есть в названии этой
строки), — не корзина. Позиция списка — услуга, которую она называет, а не составная
услуга, где это слово перечислено в скобках.

Не покрыто (решение владельца): уточнение, которого нет ни в одном названии прайса
(«узи брюшной полости ПОЛНОЕ», «общий анализ крови РАЗВЕРНУТЫЙ»), — фраза целиком не
находит строку со всеми словами, и правила не отличают уточнение от второй позиции.
"""

from __future__ import annotations

import re

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services._common import _normalise_input
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _build_multi_price_payload,
    _resolve_multi_price_items,
    resolve_price_service_name_from_catalog,
)

# Свип по названиям прайса; выборка по всему каталогу, а не его начало.
_SWEEP_SIZE = 50
_WORD_RE = re.compile(r"[0-9a-zа-яё]+")


def _rows():
    return [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_normalise_input(text))


def _names_typed_as_words(rows, limit: int) -> list[str]:
    """Первые пять слов названий прайса — без скобок и запятых, как пишет пациент.

    Пять слов — чтобы включилась резка сплошной строки (она начинается с четырёх).
    """

    seen: set[str] = set()
    phrases: list[str] = []
    for row in rows:
        words = [w for w in _words(str(row.get("serviceName") or "")) if len(w) > 1]
        if len(words) < 4:
            continue
        phrase = " ".join(words[:5])
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    step = max(1, len(phrases) // limit)
    return phrases[::step][:limit]


def test_phrase_named_by_one_catalog_row_is_not_a_basket():
    """Инвариант класса на живом каталоге."""
    rows = _rows()
    assert len(rows) > 1000, "нужен живой прайс региона"
    checked = 0
    split: list[tuple[str, str, list[str]]] = []
    for phrase in _names_typed_as_words(rows, _SWEEP_SIZE):
        whole = resolve_price_service_name_from_catalog(phrase, rows=rows)
        if not whole:
            continue
        row_words = set(_words(whole))
        if not all(w in row_words for w in _words(phrase) if len(w) >= 3):
            continue  # фраза не называет одну строку целиком — не этот класс
        checked += 1
        items, _ = _resolve_multi_price_items(phrase, rows)
        if items:
            split.append((phrase, whole, [str(i["service_name"]) for i in items]))
    assert checked >= 20, f"свип проверил слишком мало фраз ({checked}) — выборка не о том"
    assert not split, f"одна услуга стала корзиной в {len(split)} из {checked}, например: {split[:3]}"


@pytest.mark.parametrize(
    "question",
    [
        "сколько стоит общий анализ крови полный",  # → «Полный съемный протез»
        "сколько стоит приём хирурга на дому",  # → «Физиотерапия на дому»
        "сколько стоит прием врача акушера гинеколога",  # → приём хирурга
        "сколько стоит общего анализа крови с лейкоформулой",  # → «Общий белок»
    ],
)
def test_reported_single_services_are_not_baskets(question: str):
    assert _build_multi_price_payload(question, _rows()) is None


def test_list_item_is_the_service_it_names_not_a_panel_listing_it():
    """«креатинин мочевина» — два анализа, а не составная услуга со скобками."""
    items, _ = _resolve_multi_price_items("общий белок глюкоза креатинин мочевина", _rows())
    names = [_normalise_input(str(i["service_name"])) for i in items]
    assert len(names) == 4, names
    assert any(n.startswith("креатинин") for n in names), names
    assert any(n.startswith("мочевина") for n in names), names


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # Анти-регрессия: настоящие списки без разделителей режутся по-прежнему.
        ("общий анализ крови ферритин глюкоза", ("общий анализ крови", "ферритин", "глюкоза")),
        ("приём хирурга приём уролога приём кардиолога", ("хирург", "уролог", "кардиолог")),
        ("узи щитовидной железы узи молочных желез", ("щитовидн", "молочн")),
    ],
)
def test_real_lists_without_separators_still_split(question: str, expected: tuple[str, ...]):
    items, _ = _resolve_multi_price_items(question, _rows())
    names = [_normalise_input(str(i["service_name"])) for i in items]
    assert len(names) == len(expected), names
    for need in expected:
        assert any(need in n for n in names), (need, names)
