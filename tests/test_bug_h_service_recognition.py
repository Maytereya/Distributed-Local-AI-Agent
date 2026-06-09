"""BUG-H: грубая ошибка распознавания услуги — интент-слово «анализы» как услуга.

Живой прод-баг (после деплоя): «Подскажите, пожалуйста, цены на анализы:
Витамин B1 Витамин B6» → бот выдал «По услуге **Общий анализ мочи**…».

Корень (офлайн-repro): `extract_service_phrase` возвращает голое «Анализы» как
услугу-кандидата; `resolve_price_service_name_from_catalog` резолвит его в
«Общий анализ мочи» и тот ПЕРЕБИВАЕТ зашумлённый, но верный витаминный кандидат.
`_rank_price_rows` на полном тексте при этом находит витамин — т.е. виноват не
матчер, а допуск голого generic-слова в кандидаты.

Инвариант (класс — фундамент для «корзины анализов»): услуга-запрос обязан нести
хотя бы один РАЗЛИЧАЮЩИЙ токен. Голые generic-стволы (анализ/узи/исследование/
орган) НЕ являются услугой и не должны резолвиться в произвольную строку каталога
и перебивать конкретный запрос. Реальные услуги (несут различающий токен:
крови/мочи/витамин/оак/…) — не затрагиваются.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services import Services
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _build_price_catalog_queries,
    _distinctive_service_tokens,
    resolve_price_service_name_from_catalog,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def run(coro):
    return asyncio.run(coro)


def _resolve(q: str) -> str | None:
    return resolve_price_service_name_from_catalog(q, rows=ROWS)


def test_catalog_has_vitamins_sanity():
    """Гард окружения: витамины реально в каталоге (иначе тест бессмысленен)."""
    names = " | ".join(str(r.get("serviceName") or "") for r in ROWS).lower()
    assert "витамин b1" in names and "общий анализ мочи" in names


# --- Корень: голое generic-слово не различающее ---
def test_distinctive_tokens_generic_analiz_is_not_distinctive():
    assert _distinctive_service_tokens("анализы") == set()
    assert _distinctive_service_tokens("анализ") == set()
    assert "мочи" in _distinctive_service_tokens("анализ мочи")
    assert _distinctive_service_tokens("витамин b1")  # non-empty


def test_build_price_catalog_queries_drops_bare_generic_candidate():
    """Голый «Анализы»-кандидат не попадает в варианты поиска по каталогу."""
    cands = _build_price_catalog_queries("Подскажите, пожалуйста, цены на анализы: Витамин B1 Витамин B6")
    for c in cands:
        assert _distinctive_service_tokens(c), f"pure-generic candidate leaked: {c!r}"


# --- Класс-инвариант: интент-шум «анализы/анализ на» + конкретная услуга → услуга ---
@pytest.mark.parametrize(
    "query,expect_fragment",
    [
        ("Подскажите, пожалуйста, цены на анализы: Витамин B1 Витамин B6", "витамин b1"),
        ("цены на анализы: Витамин B1 Витамин B6", "витамин b1"),
        ("анализы на ферритин", "ферритин"),
        ("сколько стоит анализ на ферритин", "ферритин"),
    ],
    ids=["live_bug_full", "live_bug_short", "analizy_na_ferritin", "skolko_analiz_ferritin"],
)
def test_intent_noise_resolves_to_specific_service_not_oam(query, expect_fragment):
    resolved = _resolve(query)
    assert resolved is not None, f"{query!r}: resolved None"
    low = resolved.lower()
    assert expect_fragment in low, f"{query!r} -> {resolved!r}, expected {expect_fragment!r}"
    assert "общий анализ мочи" not in low, f"{query!r} -> ОАМ (gross error not fixed)"


def test_bare_generic_query_does_not_pick_arbitrary_service():
    """Голое «анализы» → не выдаём случайный ОАМ (лучше None → уточнение)."""
    resolved = _resolve("анализы")
    assert resolved is None or "общий анализ мочи" not in (resolved or "").lower()


# --- Регресс: реальные услуги с различающим токеном не затронуты ---
@pytest.mark.parametrize(
    "query,expect_fragment",
    [
        ("общий анализ крови", "крови"),
        ("ОАК", "крови"),  # алиас ОАК → «Общий анализ крови» (BUG-2026-06-02-09)
        ("анализ мочи", "мочи"),
        ("витамин b1", "витамин b1"),
        ("сколько стоит ОАК", "крови"),
    ],
    ids=["oak_full", "oak_abbr", "urine", "vit_b1", "skolko_oak"],
)
def test_real_services_still_resolve(query, expect_fragment):
    resolved = _resolve(query)
    assert resolved is not None, f"regression: {query!r} -> None"
    assert expect_fragment in resolved.lower(), f"regression: {query!r} -> {resolved!r}"


def test_price_info_e2e_vitamin_prefix_not_oam():
    """e2e живого бага: price_info на полную фразу → витамин, не ОАМ."""
    svc = Services()
    payload = run(svc.price_info("Подскажите, пожалуйста, цены на анализы: Витамин B1 Витамин B6", {}))
    prices = payload.get("prices") or []
    names = " | ".join(str(r.get("serviceName") or r.get("name") or "") for r in prices).lower()
    assert "витамин" in names, f"e2e: vitamin missing -> {names[:120]!r}"
    assert "общий анализ мочи" not in names, f"e2e: ОАМ gross error -> {names[:120]!r}"
