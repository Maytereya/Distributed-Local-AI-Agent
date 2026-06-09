"""BUG-H sibling: «сколько стоит анализ на <X>» → price_info отдаёт пустые цены.

Корень (офлайн-repro): в retail-ветке `price_info` `retail_query` стартует как
корректно резолвленная услуга (resolve → «ТТГ…»), но затем перебивается
`query_candidate = _extract_price_service_from_query(query_text)` = «анализ ттг»
(framing-слово «анализ»). `_select_patient_price_rows("анализ ттг")` → [] (AND-
семантика: ни одна строка не содержит и «анализ», и «ттг»), а «ттг» → находит.

Инвариант (класс): если retail-выборка по query-кандидату из текста ПУСТА, а по
резолвленной услуге — нет, отдаём резолвленную (не теряем цену из-за framing-шума).
Ловит любой single-lab под «анализ на <X>» (ттг/ферритин/…), не инстанс.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


def _price_names(payload: dict) -> str:
    rows = (payload.get("prices") or []) + (payload.get("family_variants") or [])
    return " | ".join(str(r.get("serviceName") or r.get("name") or "") for r in rows).lower()


@pytest.mark.parametrize(
    "query,expect_fragment",
    [
        ("сколько стоит анализ на ттг", "ттг"),
        ("сколько стоит анализ на ферритин", "ферритин"),
        ("цена анализа на ттг", "ттг"),
    ],
    ids=["ttg_analiz_na", "ferritin_analiz_na", "ttg_cena_analiza"],
)
def test_price_info_analysis_framing_does_not_drop_prices(query, expect_fragment):
    svc = Services()
    payload = run(svc.price_info(query, {}))
    names = _price_names(payload)
    assert names.strip(), f"{query!r}: prices empty (BUG-H sibling not fixed) note={payload.get('note')!r}"
    assert expect_fragment in names, f"{query!r} -> {names[:120]!r}, expected {expect_fragment!r}"


@pytest.mark.parametrize(
    "query,expect_fragment",
    [
        ("сколько стоит ттг", "ттг"),
        ("цена ттг", "ттг"),
        ("сколько стоит ферритин", "ферритин"),
    ],
    ids=["ttg_plain", "ttg_cena", "ferritin_plain"],
)
def test_price_info_plain_lab_query_regression(query, expect_fragment):
    svc = Services()
    payload = run(svc.price_info(query, {}))
    names = _price_names(payload)
    assert names.strip(), f"regression: {query!r} -> empty"
    assert expect_fragment in names, f"regression: {query!r} -> {names[:120]!r}"


def test_service_bundle_info_analysis_framing_does_not_drop_prices():
    """Тот же сиблинг в service_bundle_info: retail_prices не теряются на «анализ на <X>»."""
    svc = Services()
    payload = run(svc.service_bundle_info("сколько стоит анализ на ттг", {}))
    retail = payload.get("retail_prices") or []
    names = " | ".join(str(r.get("serviceName") or r.get("name") or "") for r in retail).lower()
    assert retail, f"service_bundle_info: retail_prices empty (sibling not fixed) note={payload.get('note')!r}"
    assert "ттг" in names, f"service_bundle_info -> {names[:120]!r}"
    # service_name не должен подмениться шумным «анализ ттг»
    assert "анализ ттг" not in str(payload.get("service_name") or "").lower()
