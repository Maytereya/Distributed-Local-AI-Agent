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


# Каталог чекапов как в Самаре: «чекап» — это ЛИНЕЙКА пакетов, а не одна услуга.
_CHECKUP_CATALOG = [
    {"serviceName": "Ежегодный Чекап", "serviceHomecode": "9001", "cost": 1970, "deadline": "1-2"},
    {"serviceName": "Мужской чекап Базовый", "serviceHomecode": "9002", "cost": 1210, "deadline": "1-2"},
    {"serviceName": "Мужской чекап Стандартный", "serviceHomecode": "9003", "cost": 3150, "deadline": "1-2"},
    {"serviceName": "Мужской чекап Расширенный", "serviceHomecode": "9004", "cost": 7700, "deadline": "1-2"},
    {"serviceName": "Женский чекап Базовый", "serviceHomecode": "9005", "cost": 2860, "deadline": "1-2"},
    {"serviceName": "Женский чекап Расширенный", "serviceHomecode": "9006", "cost": 9900, "deadline": "1-2"},
    # шум — не чекапы, не должны попадать в выдачу
    {"serviceName": "Ферритин", "serviceHomecode": "240", "cost": 490, "deadline": "1-2"},
    {"serviceName": "ТТГ (TSH) тиреотропный гормон", "serviceHomecode": "131", "cost": 380, "deadline": "1-2"},
]


def test_checkup_category_query_returns_whole_family(monkeypatch):
    """Регрессия (Bug #2): голый «чекап» должен вернуть ВСЮ линейку пакетов, а
    «мужской/женский чекап» — сужать по полу. Это контракт широкого запроса,
    который гард в роутере (test_assist_category_kept_broad) обязан сохранить —
    service_name НЕ пиннится в одну каноническую строку."""
    svc = Services()
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _r: _CHECKUP_CATALOG)

    res = run(svc.test_assist("чекап", {}))
    tests = res.get("tests") or []
    assert len(tests) == 6, [t.get("serviceName") for t in tests]
    assert res.get("needs_handoff") is not True

    res_m = run(svc.test_assist("мужской чекап", {}))
    assert len(res_m.get("tests") or []) == 3
    assert all("Мужской" in str(t.get("serviceName") or "") for t in res_m.get("tests") or [])

    res_f = run(svc.test_assist("женский чекап", {}))
    assert len(res_f.get("tests") or []) == 2
    assert all("Женский" in str(t.get("serviceName") or "") for t in res_f.get("tests") or [])


def test_checkup_pinned_service_name_collapses_family(monkeypatch):
    """Анти-регрессия: ДОКУМЕНТИРУЕМ корень бага. Если service_name запиннен в
    «Ежегодный Чекап» (как делал _inject_catalog_candidates до фикса), линейка
    схлопывается в один пакет. Гард в роутере именно поэтому держит запрос
    широким и не пиннит имя."""
    svc = Services()
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _r: _CHECKUP_CATALOG)

    res = run(svc.test_assist("чекап", {"service_name": "Ежегодный Чекап"}))
    tests = res.get("tests") or []
    assert len(tests) == 1
    assert str(tests[0].get("serviceName") or "") == "Ежегодный Чекап"
