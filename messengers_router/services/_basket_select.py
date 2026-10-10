"""Корзина: строки каждой позиции списка выбирает LLM — одним вызовом на весь список (ход (в), L-15a).

Корзина правил брала на позицию то, что угадал буквенный резолвер, и вторую строку «на всякий
случай» (`_select_patient_price_rows(limit=2)`): рядом с правильной строкой вставали чужие — «Глюкоза в
моче» к глюкозе, «СРБ в синовиальной жидкости» к СРБ (карточка хода (б)-1: строк вне разметки 101 →
222). Здесь нарезка на позиции остаётся за правилами (генератор кандидатов), а какие строки — та
услуга, решает модель:

- для каждой позиции — её кандидаты (строки корзины правил и поиск по словам позиции);
- сильное совпадение по данным (дословное название, синоним МИС, код) — без LLM (L-13);
- остальные позиции — один вызов `choose()`: плоский нумерованный список, строка помечена своей
  позицией («позиция» → название — цена); модель отмечает номера; ничего для позиции — «не нашёл»;
- сбой LLM — None: корзина правил как раньше (политика корзины при сбое — отдельный шаг L-13).

Выключатель `[MESSENGER_ROUTER] choose_basket = off | on` (по умолчанию off; тени нет — это вызов LLM).
"""

from __future__ import annotations

import logging
from typing import Any

from ..choice import Option, Task, choose
from ..runtime_config import config as _cfg
from ._price_select_llm import _prefer_base_variants, _row_cost, _row_key, _row_name, _strong_rows
from ._prices_helpers import _annotate_price_rows_with_care_context, _rank_price_rows

logger = logging.getLogger(__name__)

BASKET_TASK = Task(name="price_basket", num_predict=192, timeout_s=12, queue_timeout_ms=3000)
_PER_ITEM = 6
_MAX_OPTIONS = 30


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


async def choose_basket_rows(
    question: str,
    items: list[dict[str, Any]],
    unrecognized: list[str],
    retail_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Позиции корзины с выбранными строками и нераспознанное; None — оставить корзину правил.

    :param question: реплика пациента (как написал он сам)
    :param items: позиции корзины правил (`fragment`, `service_name`, `prices`)
    :param unrecognized: позиции, которые правила не распознали
    :param retail_rows: строки прайса региона
    """

    chosen: dict[int, list[dict[str, Any]]] = {}
    weak: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, item in enumerate(items):
        fragment = str(item.get("fragment") or item["service_name"])
        candidates = _candidates(item, retail_rows)
        strong = _strong_rows(fragment, candidates, retail_rows)
        if strong:
            chosen[index] = strong
        else:
            weak.append((index, fragment, candidates))
    if weak:
        per_item = max(2, _MAX_OPTIONS // len(weak))
        options: list[Option] = []
        owners: dict[str, tuple[int, dict[str, Any]]] = {}
        for index, fragment, candidates in weak:
            for row in candidates[:per_item]:
                option_id = str(len(options))
                owners[option_id] = (index, row)
                options.append(Option(id=option_id, text=f"«{fragment}» → {_row_name(row)} — {_row_cost(row)} руб."))
        choice = await choose(BASKET_TASK, question, options)
        if choice.failed:
            return None
        for option_id in sorted(choice.ids, key=int):
            index, row = owners[option_id]
            chosen.setdefault(index, []).append(row)
    positions: list[dict[str, Any]] = []
    not_found: list[str] = []
    for index, item in enumerate(items):
        fragment = str(item.get("fragment") or item["service_name"])
        rows = chosen.get(index)
        if rows:
            positions.append(_position(fragment, _prefer_base_variants(rows, fragment)))
        else:
            not_found.append(fragment)
    return positions, [*unrecognized, *not_found]
