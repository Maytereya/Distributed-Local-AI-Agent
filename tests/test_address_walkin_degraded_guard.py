"""BUG (Диалог #232): «До какого времени сдать анализы?» → опять один «Гагарина 64».

Диагноз: NLU идентичен рабочему запросу (service_name='анализы'); ответ пришёл из
ЖИВОГО /regions (есть Телефон/График — fallback их не даёт). Значит /regions вернул
ЧАСТИЧНЫЙ ответ (parent-иерархия неполная → BUG-A city-derivation не сработал для
большинства филиалов → уцелел только name-based «Гагарина 64»). Это латентность
апстрима, НЕ регрессия кода. Но бот не должен блефовать одиноким филиалом.

Защитный guard (инвариант, класс): walk-in по анализам/ЭКГ в Самаре всегда даёт
МНОГО филиалов (~31 анализы / ~4 ЭКГ). Если живой /regions (не tech-fail) отдал ≤1
самарский филиал И запрос не про конкретный филиал и не запись — считаем ответ
деградированным: честный tech-ответ, НЕ вводящий в заблуждение единственный
«Гагарина 64». Конкретный branch-запрос и запись к врачу не затрагиваются.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


def _samara(idx, addr):
    # analysis=True — реальная структура /regions: филиал принимает анализы
    # (иначе _filter_regions_by_service_flags отбросит его до regions-success пути).
    return {"id": idx, "addressForSite": f"г. Самара, {addr}", "city": "Самара", "analysis": True}


def _mock_regions(monkeypatch, svc, rows):
    async def fake():
        return rows
    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake)


@pytest.mark.parametrize("rows", [[_samara(1, "ул.Гагарина, 64")], []], ids=["one_branch", "zero_branch"])
def test_walkin_analysis_implausibly_few_branches_degraded(monkeypatch, rows):
    """≤1 самарский филиал для walk-in анализов → degraded, не одинокая Гагарина."""
    svc = Services()
    _mock_regions(monkeypatch, svc, rows)
    res = run(svc.address_info("сдать анализы", {"service_name": "анализы"}))
    assert (res.get("addresses") or []) == [], f"degraded must not list branches: {res.get('addresses')}"
    assert res.get("handoff_reason") == "tech_unavailable", f"expected tech_unavailable, got {res.get('note')!r}"
    assert "гагарина" not in " ".join(res.get("addresses") or []).lower()


def test_walkin_analysis_many_branches_ok(monkeypatch):
    """Здоровый /regions (много филиалов) → guard НЕ срабатывает, отдаём все."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [
        _samara(1, "ул.Гагарина, 64"),
        _samara(2, "пр. Ленина, 5"),
        _samara(3, "ул. Победы, 83"),
    ])
    res = run(svc.address_info("сдать анализы", {"service_name": "анализы"}))
    addrs = res.get("addresses") or []
    assert len(addrs) >= 2, f"healthy result wrongly degraded: {addrs}"
    assert res.get("handoff_reason") != "tech_unavailable"


def test_specific_branch_query_single_is_not_degraded(monkeypatch):
    """Регресс: запрос конкретного филиала («на Гагарина») legitimately 1 → не degraded."""
    svc = Services()
    _mock_regions(monkeypatch, svc, [
        _samara(1, "ул.Гагарина, 64"),
        _samara(2, "пр. Ленина, 5"),
    ])
    res = run(svc.address_info("анализы на Гагарина", {"service_name": "анализы", "branch": "Гагарина"}))
    addrs = res.get("addresses") or []
    assert any("гагарина" in a.lower() for a in addrs), f"specific-branch query lost its branch: {addrs}"
    assert res.get("handoff_reason") != "tech_unavailable"
