"""Показанное — по строке прайса, а не по названию: короткий вопрос о цене после показа отвечает ценой показанной строки
(BUG-2026-10-30-SHOWN-BY-NAME).

В прайсе региона одно название бывает у нескольких `serviceId` с разной ценой (срез 08.10: 50 названий, у 32 цены разные —
«Посев мочи с идентификацией культур»: 1 100 и 1 050 руб.). Контекст сессии хранит показанное названиями (`_shown_services`,
`shown_context.remember`), а `prices._rows_shown_before` находит строку по названию первой в прайсе (`setdefault`). Инварианты класса:
- **показана одна строка** (выбор LLM) → «а сколько стоит?» → цена именно этой строки, чужой строки с тем же названием нет;
- **показаны обе строки** → «а сколько стоит?» → обе цены (пациент видел две — ответ не сводит их к одной).

Группы — из герметичного прайса (`tests/fixtures/mini_prod`), не список фраз: новая пара в срезе сразу попадает в тест. Диалог —
через эндпоинт, харнесс `tests/understanding_contract.py` (подделка LLM настоящего типа). Дефект — xfail strict с ID: строки, которые
в прайсе не первые со своим названием, отвечают ценой первой.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

import pytest

from tests import understanding_contract as uc
from tests.test_price_turns_golden import FIXTURE
from tests.understanding_contract import d21_world  # noqa: F401 — фикстура

_BUG = "BUG-2026-10-30-SHOWN-BY-NAME: показанное хранится названием, «а сколько стоит?» берёт первую строку прайса с этим названием"
# Фраза, на которую выбор LLM (`choose_price_service_general`) показывает строки группы по одной: у лабораторных пар — общее начало.
_GENERAL = {"Посев мочи с идентификацией культур": "посев мочи", "Посев на микрофлору (идентификация агента и чувствительность к а\\б)": "посев на микрофлору"}
_FOLLOWUP = uc.say("а сколько стоит?", services=())


@lru_cache(maxsize=1)
def _groups() -> dict[str, tuple[dict[str, Any], ...]]:
    """Название → строки прайса в порядке файла: только названия с разными `serviceId` и разной ценой."""

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((FIXTURE / "price_by_region").glob("price_region_3_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else None
            if isinstance(row, dict) and str(row.get("serviceName") or "").strip():
                by_name[str(row["serviceName"]).strip()].append(row)
    return {
        name: tuple(rows)
        for name, rows in sorted(by_name.items())
        if len({row["serviceId"] for row in rows}) > 1 and len({row["cost"] for row in rows}) > 1
    }


def _shown_costs(text: str, rows: tuple[dict[str, Any], ...]) -> set[float]:
    """Цены строк группы, которые печатает ответ («1 050 руб.»)."""

    found = set()
    for row in rows:
        digits = str(int(row["cost"]))
        spaced = r"\s?".join([digits[: len(digits) % 3 or 3]] + [digits[i : i + 3] for i in range(len(digits) % 3 or 3, len(digits), 3)])
        if re.search(rf"(?<!\d)(?<!\d\s){spaced}\s*руб", text):
            found.add(row["cost"])
    return found


def test_fixture_price_has_duplicate_names_with_different_prices() -> None:
    """Тест класса не пустой: в герметичном прайсе есть пары и обе лабораторные группы выбора LLM."""

    assert len(_groups()) >= 3
    assert set(_GENERAL) <= set(_groups())


@pytest.mark.parametrize("switch", [pytest.param("off", marks=pytest.mark.xfail(strict=True, reason=_BUG)), "on"])
@pytest.mark.parametrize("name", sorted(_groups()))
def test_followup_after_both_rows_answers_every_shown_price(d21_world, monkeypatch, name: str, switch: str) -> None:  # noqa: F811
    rows = _groups()[name]
    turns = [uc.say(f"сколько стоит {name.lower()}", services=(name.lower(),)), _FOLLOWUP]
    record = uc.run(d21_world, monkeypatch, turns, cfg={"MR_SHOWN_BY_ID": switch})
    shown = _shown_costs(record[0]["text"], rows)
    assert shown == {row["cost"] for row in rows}, record[0]["text"]  # предусловие: пациент видел все строки названия
    assert _shown_costs(record[1]["text"], rows) == shown, record[1]["text"]


@pytest.mark.parametrize(
    "name, index, switch",
    [
        pytest.param(name, k, switch, id=f"{phrase}-{k}-{switch}", marks=pytest.mark.xfail(strict=True, reason=_BUG) if k and switch == "off" else ())
        for name, phrase in sorted(_GENERAL.items())
        for k in range(len(_groups().get(name, ())))
        for switch in ("off", "on")
    ],
)
def test_followup_after_one_shown_row_answers_its_price(d21_world, monkeypatch, name: str, index: int, switch: str) -> None:  # noqa: F811
    """Выбор LLM показал одну строку группы — «а сколько стоит?» отвечает её ценой. Под off первая в прайсе проходит случайно
    (`setdefault` берёт её), остальные — чужая цена; под `shown_by_id = on` — каждая своей."""

    rows = _groups()[name]
    row = rows[index]
    pick = uc.picked(f"{name} — {int(row['cost'])} руб.")
    phrase = _GENERAL[name]
    turns = [uc.say(f"сколько стоит {phrase}", services=(phrase,), choices={"price_service_general": pick}), _FOLLOWUP]
    record = uc.run(d21_world, monkeypatch, turns, cfg={"MR_SHOWN_BY_ID": switch})
    assert "choose_price_service_general" in record[0]["asked"]
    assert _shown_costs(record[0]["text"], rows) == {row["cost"]}, record[0]["text"]  # предусловие: показана ровно эта строка
    assert _shown_costs(record[1]["text"], rows) == {row["cost"]}, record[1]["text"]
