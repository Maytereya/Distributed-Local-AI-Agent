"""Класс `word_order_decides_existence`: перестановка слов не может превращать
найденную услугу в «услуги нет».

## Что произошло

Пациент пишет «т3 общий» вместо «общий т3» — и получает «не распознано», хотя
«Общий трийодтиронин - T3» есть в прайсе. Порядок слов смысла не несёт, и
пациент не обязан угадывать тот, который бот примет.

Механизм оказался не в матче, а в стыке двух правил:

* `_price_row_score` даёт **+90 за «головной» матч** — бонус введён в мае 2026,
  чтобы «ТТГ» выдавал «ТТГ (TSH) тиреотропный гормон», а не «Антитела к
  рецепторам ТТГ». Это подсказка РАНЖИРОВАНИЯ: она разводит двух кандидатов
  между собой.
* `_resolve_price_service_core` отсекает по АБСОЛЮТНОМУ порогу
  `best_score < 100 → None`.

Бонус сравнивался только с ПЕРВЫМ токеном запроса, поэтому «общий т3» набирал
180, а «т3 общий» — 90. Через порог подсказка ранжирования превращалась в
вердикт о существовании услуги.

## Почему LLM здесь не спасает

Грундер извлекает ОДНО название услуги на реплику. Позиции внутри корзины
резолвятся пофрагментно и **чисто лексически** — LLM в этом пути нет. Значит
лексический слой обязан быть устойчив к порядку слов сам.

## Инвариант

Если запрос из двух значащих слов находит услугу, то и он же с переставленными
словами обязан находить услугу. Свип идёт по ЖИВОМУ каталогу, а не по списку
примеров: класс шире, чем Т3 и Т4 (те же потери были у аллергенов с кодом —
«бензокаин с86» находился, «с86 бензокаин» нет).
"""

from __future__ import annotations

import re

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services._common import _normalise_input
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    resolve_price_service_name_from_catalog,
)

# Свип стоит секунды на запрос, а класс ловится уже на десятках: берём выборку
# достаточную, чтобы поймать механизм, и не растим гейт на минуты.
_SWEEP_SIZE = 60


def _rows():
    return [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _two_word_queries(rows, limit: int):
    """Двухсловные запросы из САМИХ названий каталога.

    :param rows: строки прайса
    :param limit: сколько пар вернуть
    :return: список пар (прямой порядок, обратный порядок)
    """

    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for row in rows:
        tokens = [t for t in re.findall(r"[a-zа-яё0-9]+", _normalise_input(str(row.get("serviceName") or ""))) if len(t) > 1]
        if len(tokens) < 2:
            continue
        forward = f"{tokens[0]} {tokens[1]}"
        if forward in seen:
            continue
        seen.add(forward)
        candidates.append((forward, f"{tokens[1]} {tokens[0]}"))
    if len(candidates) <= limit:
        return candidates
    # Берём выборку ПО ВСЕМУ каталогу, а не его начало. Первая версия теста
    # брала первые 60 строк и проходила даже без починки: потери сидели у
    # аллергенов с кодом («бензокаин с86»), которые лежат дальше по алфавиту.
    step = len(candidates) / limit
    return [candidates[int(i * step)] for i in range(limit)]


def test_catalog_sanity_for_word_order_sweep():
    """Свип бессмыслен на пустом или подменённом каталоге."""
    rows = _rows()
    assert len(rows) > 1000, "нужен живой прайс региона"
    assert len(_two_word_queries(rows, _SWEEP_SIZE)) == _SWEEP_SIZE


def test_word_order_does_not_decide_whether_service_exists():
    """Инвариант класса: перестановка слов не меняет факт нахождения услуги."""
    rows = _rows()
    lost: list[tuple[str, str, str]] = []
    for forward, reverse in _two_word_queries(rows, _SWEEP_SIZE):
        found_forward = resolve_price_service_name_from_catalog(forward, rows=rows)
        if not found_forward:
            continue
        if not resolve_price_service_name_from_catalog(reverse, rows=rows):
            lost.append((forward, reverse, found_forward))
    assert not lost, (
        "перестановка слов потеряла услугу в "
        f"{len(lost)} запросах, например: {lost[:3]}"
    )


@pytest.mark.parametrize(
    ("direct", "reversed_"),
    [
        # Кейс заказчика 16.09: умолчание «свободный» верное, но пациент вправе
        # уточнить «общий» в любом порядке.
        ("общий т3", "т3 общий"),
        ("общий тироксин", "тироксин общий"),
    ],
)
def test_explicit_qualifier_works_in_both_orders(direct: str, reversed_: str):
    """Уточнение «общий» обязано работать независимо от порядка слов."""
    rows = _rows()
    assert resolve_price_service_name_from_catalog(direct, rows=rows)
    assert resolve_price_service_name_from_catalog(reversed_, rows=rows)


def test_head_bonus_still_prefers_base_service_over_subspecialty():
    """Анти-регрессия: бонус не потерял своей исходной цели.

    Жалоба заказчика 2026-05-05: «ТТГ» отдавал «Антитела к рецепторам ТТГ»
    вместо простого ТТГ. Ослабление привязки к первому токену не должно было
    это вернуть — у «Антитела к рецепторам ТТГ» головное слово «антитела», и в
    запросе «ттг» его нет ни первым, ни любым другим.
    """

    found = resolve_price_service_name_from_catalog("ттг", rows=_rows())
    assert found, "«ттг» обязан находить услугу"
    assert "антител" not in _normalise_input(found), f"«ттг» снова уводит в субспециальное: {found!r}"


def test_same_short_word_matches_itself():
    """`tokens_share_stem` с порогом 4 не вправе развести слово с самим собой.

    Сверка объединённого правила со старыми шестью формами поймала 725 таких
    пар на живом словаре каталога: «оак» короче порога и переставал совпадать
    сам с собой. Поэтому в правиле есть явное короткое замыкание по равенству.
    """

    from messengers_router.russian_nlu import tokens_share_stem

    assert tokens_share_stem("оак", "оак", length=4)
    assert tokens_share_stem("оам", "оам", length=6)
    assert not tokens_share_stem("оак", "оам", length=4)
    assert not tokens_share_stem("", "", length=4), "пустые строки словом не считаются"
    # Порог обязан оставаться явным: одинаковое начало короче порога — не матч.
    assert not tokens_share_stem("кров", "кровь", length=5)
    assert tokens_share_stem("кровь", "кровяной", length=4)
