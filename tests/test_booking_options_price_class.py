"""Варианты записи на услугу — с ценой строки прайса процедуры (п. 4 топ-5 разбора 95 цен, выключатель `booking_options_price`).

Разбор 95 (деплой 18, класс B): голое название процедуры («УЗИ щитовидки», «УЗИ мошонки с ЦДК») правило ставит в запись, и
пациент видит «Уточните, пожалуйста, на какую услугу вас записать: — … — …» без цен, хотя спрашивал, по сути, что это и
сколько стоит. Инвариант под on: у названия, которое есть строкой прайса процедуры, — её цена (печатает код, принцип 1); у
строки приёма врача цены нет — цену приёма к специальности не называем, пока врач не выбран (решение 25.09); названия без строки
и срез прайса старше недели — без цены; off — названия, как было.
"""

from __future__ import annotations

import datetime

import pytest

from messengers_router import router
from messengers_router.services import _prices_helpers

_ROWS = [
    {"serviceName": "УЗИ органов мошонки + ЦДК сосудов", "cost": 1850.0},
    {"serviceName": "УЗИ органов мошонки", "cost": 1450.0},
    {"serviceName": "Прием (осмотр, консультация) врача-уролога", "cost": 3100.0},
]
_NAMES = ["УЗИ органов мошонки + ЦДК сосудов", "УЗИ органов мошонки", "Прием (осмотр, консультация) врача-уролога", "Услуга не из прайса"]


@pytest.fixture
def catalog(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(_prices_helpers, "_catalog_rows", lambda _region: _ROWS)

    def switch(value: str) -> None:
        monkeypatch.setattr(cfg, "MR_BOOKING_OPTIONS_PRICE", value, raising=False)

    return switch


def test_procedure_rows_carry_their_own_price(catalog):
    catalog("on")
    lines, priced = router._booking_option_lines(_NAMES)

    assert priced
    assert lines.splitlines() == [
        "— УЗИ органов мошонки + ЦДК сосудов — 1 850 руб.",
        "— УЗИ органов мошонки — 1 450 руб.",
        "— Прием (осмотр, консультация) врача-уролога",  # приём врача — без цены (25.09)
        "— Услуга не из прайса",
    ]


@pytest.mark.parametrize("value", ["off", "", "мусор"])
def test_off_is_names_as_before(value, catalog):
    catalog(value)
    lines, priced = router._booking_option_lines(_NAMES)

    assert not priced and lines.splitlines() == [f"— {name}" for name in _NAMES]


def test_stale_price_snapshot_prints_no_price(catalog, monkeypatch):
    from agent_logic_2.nayka_api import snapshots

    catalog("on")
    stale = snapshots.Snapshot(name="price_region_3", rows=[], date=datetime.date(2026, 9, 1), status="stale", age_days=25)
    monkeypatch.setattr(snapshots, "last", lambda prefix: stale if prefix.startswith("price_region") else None)
    lines, priced = router._booking_option_lines(_NAMES)

    assert not priced and "руб." not in lines


def test_only_reception_rows_print_no_price(catalog):
    catalog("on")
    lines, priced = router._booking_option_lines([_NAMES[2]])

    assert not priced and lines == f"— {_NAMES[2]}"
