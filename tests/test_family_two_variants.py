"""Семейство из ДВУХ вариантов тоже требует развилки.

Жалоба администраторов 15.09: пациент спросил фиброколоноскопию и получил
«Фиброколоноскопия левых отделов толстой кишки — 3 000 руб.». В каталоге две:

    7 000   Тотальная фиброколоноскопия (ФКС)
    3 000   Фиброколоноскопия левых отделов толстой кишки

Услугу за 3 000 клиника пациентам НЕ называет вовсе — её проводят, когда
пациент плохо подготовился. В скрипте клиники одна цена: 7 000, с октября 8 000.

Корень: `_is_family_query_candidate` требовал `len(base_names) >= 3`. Порог в три
оставлял неохваченным самый опасный случай — выбор из ДВУХ, где молчаливая
ошибка стоит ровно половину цены. И противоречил решению владельца 2026-07-22:
«при неоднозначности — список вариантов, а НЕ единственная (возможно чужая) цена».
"""

from __future__ import annotations

from agent_logic_2.nayka_api import api_price
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _build_price_family_payload,
    _is_family_query_candidate,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _variants(query: str) -> set[str]:
    payload = _build_price_family_payload(query, ROWS, show_all=False, visible_limit=10)
    if not payload:
        return set()
    return {str(v.get("serviceName") or "") for v in payload.get("family_variants") or []}


def test_two_variant_family_offers_both():
    """Пациент должен увидеть ОБА варианта и выбрать сам."""
    assert _is_family_query_candidate("сколько стоит фиброколоноскопия", ROWS)

    shown = _variants("сколько стоит фиброколоноскопия")
    assert any("тотальная" in s.lower() for s in shown), shown
    assert any("левых отделов" in s.lower() for s in shown), shown


def test_uniquely_named_service_still_answers_directly():
    """Анти-регресс: однозначная услуга не должна превращаться в развилку."""
    assert not _is_family_query_candidate("сколько стоит ферритин", ROWS)
