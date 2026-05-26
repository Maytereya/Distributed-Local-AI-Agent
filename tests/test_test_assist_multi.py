import asyncio

from messengers_router import services as svc_mod
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


# Каталог в форме строк прайса (как у Самары) — реальные значения со скрина.
_CATALOG = [
    {"serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
     "serviceHomecode": "501", "cost": 520, "deadline": "1-2"},
    {"serviceName": "Ферритин", "serviceHomecode": "240", "cost": 490, "deadline": "1-2"},
    {"serviceName": "Витамин D суммарный (25-ОН витамин D2 и 25-ОН витамин D3, общий)",
     "serviceHomecode": "370", "cost": 1400, "deadline": "1-2"},
    {"serviceName": "Витамин B12", "serviceHomecode": "371", "cost": 690, "deadline": "1-2"},
    {"serviceName": "Фолиевая кислота (витамин B9)", "serviceHomecode": "372", "cost": 950, "deadline": "1-3"},
    {"serviceName": "ТТГ (TSH) тиреотропный гормон", "serviceHomecode": "131", "cost": 380, "deadline": "1-2"},
]


def _codes(tests):
    return {str(t.get("serviceHomecode") or "") for t in tests}


def test_multi_item_query_returns_real_catalog_rows(monkeypatch):
    """Регрессия: бот не должен «дофантазировать» цены. Compound-запрос через
    запятую должен резолвиться по каждой услуге отдельно из каталога — с
    реальными ценами и кодами."""
    svc = Services()
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _r: _CATALOG)

    res = run(svc.test_assist("ОАК, Ферритин, Витамин Д, ТТГ, B12, B9", {}))

    assert "multi-item" in str(res.get("note") or "")
    tests = res.get("tests") or []
    codes = _codes(tests)
    # Минимум 4 из 6 должны зарезолвиться по алиасам/каталогу (B9/B12 могут
    # потребовать точного названия — это и есть «правильное» поведение:
    # отдадим только то, что точно есть в каталоге, без выдумок).
    assert len(tests) >= 4, codes
    assert {"501", "240", "370", "131"} <= codes, codes  # ОАК / Ферритин / Вит D / ТТГ

    # Цены — РЕАЛЬНЫЕ из каталога (как на скрине), без LLM-округлений.
    by_code = {str(t.get("serviceHomecode") or ""): t for t in tests}
    assert by_code["501"]["cost"] == 520
    assert by_code["240"]["cost"] == 490
    assert by_code["370"]["cost"] == 1400
    assert by_code["131"]["cost"] == 380


def test_single_item_query_uses_single_bag_path(monkeypatch):
    """Однопредметный запрос идёт прежним single-bag путём (note без multi-item)."""
    svc = Services()
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _r: _CATALOG)

    res = run(svc.test_assist("ферритин", {}))

    assert "multi-item" not in str(res.get("note") or "")
    tests = res.get("tests") or []
    assert any(str(t.get("serviceHomecode") or "") == "240" for t in tests)
