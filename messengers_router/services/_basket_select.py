"""Корзина: строки каждой позиции списка выбирает LLM — одним вызовом на весь список (ход (в), L-15a).

Корзина правил брала на позицию то, что угадал буквенный резолвер, и вторую строку «на всякий
случай» (`_select_patient_price_rows(limit=2)`): рядом с правильной строкой вставали чужие — «Глюкоза в
моче» к глюкозе, «СРБ в синовиальной жидкости» к СРБ (карточка хода (б)-1: строк вне разметки 101 →
222). Здесь нарезка на позиции остаётся за правилами (генератор кандидатов), а какие строки — та
услуга, решает модель:

- для каждой позиции — её кандидаты (строки корзины правил и поиск по словам позиции);
- сильное совпадение по данным (дословное название, синоним МИС, код — `_strong_match`) — без LLM (L-13);
- остальные позиции — один вызов `choose()`: плоский нумерованный список, строка помечена своей
  позицией («позиция» → название — цена); модель отмечает номера; ничего для позиции — «не нашёл»;
- сбой LLM: сильные позиции — по-прежнему их строки по данным, слабые — строки корзины правил, как
  раньше; сильных нет — None, корзина правил целиком.

Под `turn_parse = on` (L-15b, шаг 5) позиции — фразы разбора хода (`basket_from_phrases`): одна услуга
или несколько уже решили данные МИС или выбор LLM, сила фразы — из разбора; выбор строк слабых позиций —
тот же вызов. Политика сбоя под `on` (поправка владельца 13.10 к (ii)): LLM строки не выбрала (сбой,
мусор, вызов запрещён) — сильные позиции с ценой по данным, слабая — несколько вариантов кода (до 5) с
полными названиями и ценами; одна-единственная найденная строка ценой «той самой» не подаётся — её
название без цены (вариант B L-13, `price_names_without_prices_text`); форму решит замер шага 6.

Выключатель `[MESSENGER_ROUTER] choose_basket = off | on` (по умолчанию off; тени нет — это вызов LLM).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..choice import Option, Task, choose
from ..runtime_config import config as _cfg
from ._common import _normalise_catalog_text
from ._price_select_llm import _prefer_base_variants, _record_synonym_suggestion, _row_cost, _row_key, _row_name
from ._prices_helpers import _annotate_price_rows_with_care_context, _rank_price_rows, phrase_price_items
from ._strong_match import strong_rows

logger = logging.getLogger(__name__)

BASKET_TASK = Task(name="price_basket", num_predict=192, timeout_s=12, queue_timeout_ms=3000)
_PER_ITEM = 6
_MAX_OPTIONS = 30
_VARIANTS = 5  # слабая позиция без выбора LLM — до 5 вариантов кода (поправка владельца 13.10)


def basket_mode() -> str:
    raw = str(getattr(_cfg, "MR_CHOOSE_BASKET", "off") or "").strip().lower()
    return raw if raw in ("off", "on") else "off"


def _candidates(item: dict[str, Any], retail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in [*(item.get("prices") or []), *_rank_price_rows(retail_rows, str(item.get("fragment") or item["service_name"]), limit=_PER_ITEM)]:
        if isinstance(row, dict) and _row_key(row) not in seen:
            seen.add(_row_key(row))
            out.append(row)
    return out


def _position(fragment: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    name = _row_name(rows[0]) if len(rows) == 1 else fragment
    return {"service_name": name, "prices": _annotate_price_rows_with_care_context(list(rows)), "fragment": fragment}


def _variants(fragment: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Слабая позиция без выбора LLM (политика сбоя под `turn_parse = on`): (строки с ценой, название без цены).

    Кандидаты кода без непрошеных cito/капиллярной/на дому — по одной строке на название, до `_VARIANTS`:
    два названия и больше — варианты с полными названиями и ценами; одно — его название без цены (вариант
    B L-13: одна найденная строка — не «та самая цена»); ни одного — («», «») — «Не распознал».
    """

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _prefer_base_variants(list(candidates), fragment):
        key = _normalise_catalog_text(_row_name(row))
        if key and key not in seen and len(rows) < _VARIANTS:
            seen.add(key)
            rows.append(row)
    if len(rows) >= 2:
        return rows, ""
    return [], _row_name(rows[0]) if rows else ""


async def _pick(
    question: str,
    items: list[dict[str, Any]],
    retail_rows: list[dict[str, Any]],
    *,
    llm_allowed: bool = True,
) -> tuple[dict[int, list[dict[str, Any]]], list[tuple[int, str, list[dict[str, Any]]]], bool]:
    """Строки позиций: сильные — по данным, слабые — выбор LLM одним вызовом.

    :return: (выбранные строки по номеру позиции, слабые позиции с кандидатами, сбой LLM)
    """

    chosen: dict[int, list[dict[str, Any]]] = {}
    weak: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, item in enumerate(items):
        fragment = str(item.get("fragment") or item["service_name"])
        candidates = _candidates(item, retail_rows)
        # Сила позиции из разбора хода (`strong`, весь срез) — не пересчитывается по кандидатам.
        strong = list(item["strong"]) if "strong" in item else strong_rows(fragment, retail_rows, pool=candidates)
        if strong:
            chosen[index] = strong
        else:
            weak.append((index, fragment, candidates))
    if not weak:
        return chosen, weak, False
    per_item = max(2, _MAX_OPTIONS // len(weak))
    options: list[Option] = []
    owners: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, fragment, candidates in weak:
        for row in candidates[:per_item]:
            option_id = str(len(options))
            owners[option_id] = (index, row)
            options.append(Option(id=option_id, text=f"«{fragment}» → {_row_name(row)} — {_row_cost(row)} руб."))
    if not options:
        return chosen, weak, False  # у слабых позиций нет ни одного кандидата — выбирать не из чего
    if not llm_allowed:
        return chosen, weak, True
    choice = await choose(BASKET_TASK, question, options)
    if choice.failed:
        return chosen, weak, True
    for option_id in sorted(choice.ids, key=int):
        index, row = owners[option_id]
        chosen.setdefault(index, []).append(row)
    return chosen, weak, False


def _positions(
    items: list[dict[str, Any]], chosen: dict[int, list[dict[str, Any]]], unpriced: dict[int, str] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    positions: list[dict[str, Any]] = []
    not_found: list[str] = []
    for index, item in enumerate(items):
        fragment = str(item.get("fragment") or item["service_name"])
        rows = chosen.get(index)
        if rows:
            positions.append(_position(fragment, _prefer_base_variants(rows, fragment)))
        elif unpriced and unpriced.get(index):
            # Одна найденная строка без выбора LLM — название без цены (`multi_price_payload_from_items`).
            positions.append({"service_name": fragment, "prices": [], "fragment": fragment, "unpriced": unpriced[index]})
        else:
            not_found.append(fragment)
    return positions, not_found


def _settle_weak_as_variants(
    weak: list[tuple[int, str, list[dict[str, Any]]]], chosen: dict[int, list[dict[str, Any]]]
) -> dict[int, str]:
    """Политика сбоя под `on`: слабые позиции — варианты кода (`_variants`); названия без цены — по номеру позиции."""

    unpriced: dict[int, str] = {}
    for index, fragment, candidates in weak:
        rows, name = _variants(fragment, candidates)
        if rows:
            chosen[index] = rows
        elif name:
            unpriced[index] = name
    return unpriced


async def choose_basket_rows(
    question: str,
    items: list[dict[str, Any]],
    unrecognized: list[str],
    retail_rows: list[dict[str, Any]],
    *,
    llm_allowed: bool = True,
    failure_variants: bool = False,
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Позиции корзины с выбранными строками и нераспознанное; None — оставить корзину правил.

    :param question: реплика пациента (как написал он сам)
    :param items: позиции корзины правил (`fragment`, `service_name`, `prices`)
    :param unrecognized: позиции, которые правила не распознали
    :param retail_rows: строки прайса региона
    :param llm_allowed: можно ли звать LLM; нельзя — как сбой
    :param failure_variants: политика сбоя под `turn_parse = on` — слабые позиции без выбора LLM —
        варианты кода (`_variants`), а не строки корзины правил; None тогда не возвращается
    """

    chosen, weak, failed = await _pick(question, items, retail_rows, llm_allowed=llm_allowed)
    unpriced: dict[int, str] = {}
    if failed and failure_variants:
        unpriced = _settle_weak_as_variants(weak, chosen)
    elif failed:
        # Сильные позиции — данные, от LLM не зависят; слабые — как в корзине правил.
        if not chosen:
            return None
        for index, _fragment, _rows in weak:
            chosen[index] = list(items[index].get("prices") or [])
    positions, not_found = _positions(items, chosen, unpriced)
    return positions, [*unrecognized, *not_found]


async def basket_from_phrases(
    question: str,
    phrases: Sequence[Any],
    leftover: Sequence[str],
    retail_rows: list[dict[str, Any]],
    *,
    llm_allowed: bool = True,
    journal: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Корзина из фраз разбора хода (`turn_parse = on`, L-15b, шаг 5): позиции — фразы в порядке реплики.

    Сильная фраза — её строки по данным МИС без LLM; слабые — один вызов `choose()` по кандидатам правил
    (`phrase_price_items`); LLM строки не выбрала (сбой, мусор, вызов запрещён) — политика сбоя: варианты
    кода или название без цены (`_variants`). Фрагменты по разделителям, не покрытые разбором
    (`leftover`), и позиции без строк — «Не распознал».

    :param question: реплика пациента (как написал он сам)
    :param phrases: фразы разбора (`turn_parse.ParsedPhrase`)
    :param leftover: фрагменты реплики вне выбранных фраз
    :param retail_rows: строки прайса региона
    :param llm_allowed: можно ли звать LLM (режим `strict` — нельзя)
    :param journal: фразы выбрала LLM разбора — слабые позиции и решение LLM по ним (выбранные строки или
        «не нашла», в том числе когда кандидатов нет вовсе) пишутся в журнал синонимов (`source: basket`);
        без решения LLM по строкам (сбой) — не пишутся
    :return: (позиции с выбранными строками, нераспознанное)
    """

    items = phrase_price_items(phrases, retail_rows)
    chosen, weak, failed = await _pick(question, items, retail_rows, llm_allowed=llm_allowed)
    if journal and not failed:
        for index, fragment, _candidates in weak:
            _record_synonym_suggestion(fragment, [], list(chosen.get(index) or []), source="basket")
    unpriced = _settle_weak_as_variants(weak, chosen) if failed else {}
    positions, not_found = _positions(items, chosen, unpriced)
    return positions, [*leftover, *not_found]
