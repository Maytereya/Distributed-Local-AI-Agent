"""Класс: выбор строки цены видит вид услуги из данных МИС, а «похожая» строка — не ответ (BUG-2026-10-24-FOREIGN-PRICE-D17,
классы 2 и 4, выключатель `price_select_v2`).

Карточка 24.10: «эхо сердца», «УЗИ сердца» → ЭхоКГ плода; «ведение беременности» (4 варианта из 4) → «1–12 неделя
беременности» (профиль анализов); «первый скрининг» → «Планирование беременности (скрининг)» — нужной строки в списке нет,
а LLM выбирала похожую. Вид услуги в названии строки не виден, он есть в дереве разделов прайса МИС (`priceUnits`).

Инварианты под `price_select_v2 = on`:
- у позиции — раздел прайса МИС (два нижних уровня, без корня и римского номера); справочник недоступен — позиция как была;
- промпты `*_v2` (в двух локациях): плод, беременность, до года — особое условие; «только похоже» — подходящих нет;
- off — тексты позиций и промпты прежние.
"""

from __future__ import annotations

import pathlib

import pytest

from messengers_router.services import _price_select_llm as sel

_ROOT = pathlib.Path(__file__).resolve().parents[1]

UNITS = [
    {"id": 1, "parent": 0, "name": "Дерево прайсов"},
    {"id": 135, "parent": 1, "name": "I Лабораторные исследования"},
    {"id": 142, "parent": 135, "name": "Клинические исследования"},
    {"id": 145, "parent": 1, "name": "II Профили исследований"},
    {"id": 146, "parent": 1, "name": "III Амбулаторно-поликлиническая помощь"},
    {"id": 173, "parent": 146, "name": "УЗИ"},
    {"id": 175, "parent": 173, "name": "Общие ультразвуковые исследования"},
    {"id": 177, "parent": 173, "name": "УЗИ беременных"},
]
ROWS = [
    {"serviceId": 1270, "serviceName": "Эхокардиография", "cost": 3500, "priceUnitId": 175},
    {"serviceId": 9624, "serviceName": "Эхокардиография (Эхо-КГ) одного плода", "cost": 2700, "priceUnitId": 177},
    {"serviceId": 702, "serviceName": "1-12 неделя беременности", "cost": 6895, "priceUnitId": 145},
    {"serviceId": 471, "serviceName": "Определение времени свертываемости крови по Сухареву", "cost": 250, "priceUnitId": 142},
    {"serviceId": 9, "serviceName": "Строка без раздела", "cost": 100},
]


def _texts(monkeypatch, mode: str, units=UNITS) -> list[str]:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_SELECT_V2", mode, raising=False)
    monkeypatch.setattr(cfg, "MR_PRICE_OPTION_MIS_NOTES", "off", raising=False)

    def load():
        if isinstance(units, Exception):
            raise units
        return [dict(u) for u in units]

    from messengers_router.services import _prices_helpers

    monkeypatch.setattr(_prices_helpers.api_price, "load_price_units", load)
    return [option.text for option in sel._options("сколько стоит эхо сердца", ROWS)]


def test_on_shows_the_mis_price_section_of_every_row(monkeypatch) -> None:
    assert _texts(monkeypatch, "on") == [
        "Эхокардиография — 3500 руб. (раздел прайса: УЗИ › Общие ультразвуковые исследования)",
        "Эхокардиография (Эхо-КГ) одного плода — 2700 руб. (раздел прайса: УЗИ › УЗИ беременных)",
        "1-12 неделя беременности — 6895 руб. (раздел прайса: Профили исследований)",
        "Определение времени свертываемости крови по Сухареву — 250 руб. (раздел прайса: Лабораторные исследования › Клинические исследования)",
        "Строка без раздела — 100 руб.",
    ]


def test_off_and_missing_directory_keep_the_old_options(monkeypatch) -> None:
    old = [f"{row['serviceName']} — {row['cost']} руб." for row in ROWS]
    assert _texts(monkeypatch, "off") == old
    assert _texts(monkeypatch, "on", units=OSError("нет среза")) == old


@pytest.mark.parametrize("base", ["choose_price_service", "choose_price_service_qualifier"])
def test_v2_prompts_carry_the_rules_and_live_in_both_places(base) -> None:
    old = (_ROOT / f"messengers_router/prompts/{base}.txt").read_text(encoding="utf-8")
    new = (_ROOT / f"messengers_router/prompts/{base}_v2.txt").read_text(encoding="utf-8")
    assert new == (_ROOT / f"app_data/prompts/mr_{base}_v2.txt").read_text(encoding="utf-8")
    for rule in ("для плода, при беременности, до 1 года", "раздел прайса", "уточнение лучше чужой цены"):
        assert rule in new and rule not in old, rule
    # прежние правила v2 не теряет: всё, что было в прежнем промпте построчно, есть и в новом (кроме изменённых строк)
    changed = {line for line in old.splitlines() if line not in new.splitlines()}
    assert all(line.startswith(("Ты помогаешь", "4.", "5.", "- \"reason\"", "Позиции из прайса")) for line in changed), changed
