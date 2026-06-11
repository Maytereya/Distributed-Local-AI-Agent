"""S1 (Диалог #232): частичный /regions больше НЕ схлопывается на одну «Гагарину 64».

Раньше при частичном живом /regions (неполная parent-иерархия → производный город
не выводится) выживала только «Гагарина 64» (единственная с «Самара» в name), и бот
отдавал один филиал. Теперь единый источник `resolve_samara_branches` делает fallback:
live(здоровый) → last-good снапшот → seed (32 филиала). Worst case — полный
статический список по типу услуги, а не одинокая/неверная Гагарина.

Инвариант (класс): walk-in по анализам/ЭКГ при ЧАСТИЧНОМ живом /regions отдаёт ПОЛНЫЙ
список филиалов соответствующего типа из fallback, не один. Если и fallback пуст (нет
seed/снапшота) — честный degraded (final defense, мой guard). Здоровый live и
конкретный branch-запрос — без изменений.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services import _samara_branches as SB


def run(coro):
    return asyncio.run(coro)


def _samara(idx, addr, **flags):
    f = {"analysis": True}
    f.update(flags)
    return {"id": idx, "name": f"Самара {addr}", "addressForSite": f"г. Самара, {addr}", "city": "Самара", **f}


def _mock_regions(monkeypatch, svc, rows):
    async def fake():
        return rows
    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake)


@pytest.mark.parametrize(
    "query,frag,min_n",
    [("сдать анализы", "анализы", 10), ("где сдать ЭКГ", "ЭКГ", 5)],
    ids=["analysis", "ecg"],
)
def test_partial_regions_falls_back_to_full_list(monkeypatch, query, frag, min_n):
    """Частичный /regions (1 филиал) → полный fallback-список, НЕ одна Гагарина."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64", ecg=True)])
    res = run(svc.address_info(query, {"service_name": frag}))
    addrs = res.get("addresses") or []
    assert len(addrs) >= min_n, f"{query!r}: fallback не дал полный список ({len(addrs)})"
    assert res.get("handoff_reason") != "tech_unavailable"
    assert not (len(addrs) == 1 and "гагарина" in addrs[0].lower()), "схлопнулось на одну Гагарину"


def test_healthy_regions_used_directly(monkeypatch):
    """Здоровый live (≥порога) → используется как есть."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(i, f"ул. Тест {i}") for i in range(12)])
    res = run(svc.address_info("сдать анализы", {"service_name": "анализы"}))
    assert len(res.get("addresses") or []) >= 10
    assert res.get("handoff_reason") != "tech_unavailable"


def test_partial_and_no_fallback_is_degraded(monkeypatch):
    """Final defense: частичный live + нет seed/снапшота → честный degraded, не Гагарина."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64")])
    monkeypatch.setattr(SB, "load_seed", lambda: [])
    monkeypatch.setattr(SB, "load_snapshot", lambda: [])
    res = run(svc.address_info("сдать анализы", {"service_name": "анализы"}))
    assert (res.get("addresses") or []) == []
    assert res.get("handoff_reason") == "tech_unavailable"


def test_specific_branch_query_keeps_branch(monkeypatch):
    """Регресс: конкретный филиал («на Гагарина») → отдаём Гагарину, не теряем."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [_samara(1, "ул.Гагарина, 64"), _samara(2, "пр. Ленина, 5")])
    res = run(svc.address_info("анализы на Гагарина", {"service_name": "анализы", "branch": "Гагарина"}))
    addrs = res.get("addresses") or []
    assert any("гагарина" in a.lower() for a in addrs), f"branch-запрос потерял филиал: {addrs}"


def test_seed_is_healthy_sanity():
    """Гард окружения: seed-снапшот валиден и правдоподобен (≥порога analysis-филиалов)."""
    seed = SB.load_seed()
    assert SB.is_healthy_samara(seed), "seed unhealthy — проверь samara_branches_seed.json"
    assert SB._analysis_count(seed) >= 25
