import json
from pathlib import Path

from agent_logic_2.nayka_api import api_price


def test_ensure_doctors_source_current_uses_active_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_price, "DOCTORS_DIR", tmp_path)
    monkeypatch.setattr(api_price.api_nayka, "get_active_date_str", lambda: "20260304")

    active = tmp_path / "doctors_20260304.jsonl"
    active.write_text('{"id":1,"fio":"Иванов"}\n', encoding="utf-8")

    resolved = api_price._ensure_doctors_source_current()

    assert resolved == active


def test_ensure_doctors_source_current_refreshes_when_active_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_price, "DOCTORS_DIR", tmp_path)
    monkeypatch.setattr(api_price.api_nayka, "get_active_date_str", lambda: "20260304")

    old_file = tmp_path / "doctors_20260303.jsonl"
    old_file.write_text('{"id":2,"fio":"Петров"}\n', encoding="utf-8")

    active = tmp_path / "doctors_20260304.jsonl"
    calls = {"refresh": 0}

    def fake_get_all_doctors():
        calls["refresh"] += 1
        return [{"id": 1, "fio": "Иванов"}]

    def fake_save_doctors_data(doctors):
        active.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in doctors) + "\n", encoding="utf-8")

    def fake_find_existing():
        if active.exists():
            return active
        return old_file

    monkeypatch.setattr(api_price.api_nayka, "get_all_doctors", fake_get_all_doctors)
    monkeypatch.setattr(api_price.api_nayka, "save_doctors_data", fake_save_doctors_data)
    monkeypatch.setattr(api_price.api_nayka, "find_existing_doctors_file", fake_find_existing)

    resolved = api_price._ensure_doctors_source_current()

    assert calls["refresh"] == 1
    assert resolved == active


def test_load_doctor_prices_builds_today_cache_on_first_access(monkeypatch, tmp_path: Path):
    target = tmp_path / "doctor_prices_20260304.jsonl"
    calls = {"update": 0}

    monkeypatch.setattr(api_price, "doctor_prices_path", lambda date=None: target)

    def fake_update(force: bool = False):
        calls["update"] += 1
        target.write_text('{"doctorId":1,"serviceName":"УЗИ","cost":1000}\n', encoding="utf-8")
        return target

    monkeypatch.setattr(api_price, "update_doctor_prices", fake_update)

    rows = api_price.load_doctor_prices()

    assert calls["update"] == 1
    assert rows and rows[0]["doctorId"] == 1


def test_update_price_units_writes_today_cache(monkeypatch, tmp_path: Path):
    target = tmp_path / "price_units_20260409.jsonl"
    monkeypatch.setattr(api_price, "price_units_path", lambda date=None: target)
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_price,
        "fetch_price_units",
        lambda: [{"id": 158, "name": "Дневной стационар"}],
    )

    out = api_price.update_price_units(force=True)

    assert out == target
    rows = api_price.jsonl_read(target)
    assert rows == [{"id": 158, "name": "Дневной стационар"}]


def test_load_price_units_builds_today_cache_on_first_access(monkeypatch, tmp_path: Path):
    target = tmp_path / "price_units_20260409.jsonl"
    calls = {"update": 0}

    monkeypatch.setattr(api_price, "price_units_path", lambda date=None: target)

    def fake_update(force: bool = False):
        _ = force
        calls["update"] += 1
        target.write_text('{"id":158,"name":"Дневной стационар"}\n', encoding="utf-8")
        return target

    monkeypatch.setattr(api_price, "update_price_units", fake_update)

    rows = api_price.load_price_units()

    assert calls["update"] == 1
    assert rows == [{"id": 158, "name": "Дневной стационар"}]


def test_load_price_units_falls_back_to_latest_cache_when_refresh_fails(monkeypatch, tmp_path: Path):
    stale = tmp_path / "price_units_20260408.jsonl"
    stale.write_text('{"id":311,"name":"Дневной стационар"}\n', encoding="utf-8")

    monkeypatch.setattr(api_price, "PRICE_UNITS_DIR", tmp_path)
    monkeypatch.setattr(api_price, "price_units_path", lambda date=None: tmp_path / "price_units_20260409.jsonl")

    def fake_update(force: bool = False):
        _ = force
        raise RuntimeError("api unavailable")

    monkeypatch.setattr(api_price, "update_price_units", fake_update)

    rows = api_price.load_price_units()

    assert rows == [{"id": 311, "name": "Дневной стационар"}]


def test_update_doctor_prices_uses_company_unit_from_doctor_regions(monkeypatch, tmp_path: Path):
    target = tmp_path / "doctor_prices_20260305.jsonl"
    monkeypatch.setattr(api_price, "doctor_prices_path", lambda date=None: target)
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_price,
        "_load_doctors",
        lambda: [{"id": 1, "fio": "Иванов И.И.", "region_ids": [10], "regions": ["Самара, ул. Тестовая"]}],
    )
    monkeypatch.setattr(
        api_price.api_nayka,
        "site_doctor_regions",
        lambda: [{"worker": 1, "region": 10, "companyUnit": 99}],
    )
    monkeypatch.setattr(api_price.api_nayka, "site_doctors", lambda: [{"id": 1, "fio": "Иванов И.И."}])
    monkeypatch.setattr(
        api_price.api_nayka,
        "site_regions",
        lambda: [{"id": 10, "addressForSite": "Самара, ул. Тестовая"}],
    )

    calls: list[tuple[int, int, int | None]] = []

    def fake_fetch(doctor_id, region_id, company_unit_id=None, is_favorite=False):
        calls.append((int(doctor_id), int(region_id), None if company_unit_id is None else int(company_unit_id)))
        if company_unit_id == 99:
            return [{"serviceName": "УЗИ брюшной полости", "serviceId": 123, "cost": 1500}]
        return []

    monkeypatch.setattr(api_price, "fetch_doctor_prices", fake_fetch)

    out = api_price.update_doctor_prices(force=True)

    assert out == target
    rows = api_price.jsonl_read(target)
    assert len(rows) == 1
    assert rows[0]["doctorId"] == 1
    assert rows[0]["regionId"] == 10
    assert rows[0]["companyUnitId"] == 99
    assert rows[0]["regionName"] == "Самара, ул. Тестовая"
    assert (1, 10, 99) in calls


def test_update_doctor_prices_filters_non_samara_regions(monkeypatch, tmp_path: Path):
    target = tmp_path / "doctor_prices_20260305.jsonl"
    monkeypatch.setattr(api_price, "doctor_prices_path", lambda date=None: target)
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_price,
        "_load_doctors",
        lambda: [{"id": 1, "fio": "Иванов И.И.", "region_ids": [10, 20], "regions": ["Самара", "Оренбург"]}],
    )
    monkeypatch.setattr(
        api_price.api_nayka,
        "site_doctor_regions",
        lambda: [
            {"worker": 1, "region": 10, "companyUnit": 99},
            {"worker": 1, "region": 20, "companyUnit": 77},
        ],
    )
    monkeypatch.setattr(api_price.api_nayka, "site_doctors", lambda: [{"id": 1, "fio": "Иванов И.И."}])
    monkeypatch.setattr(
        api_price.api_nayka,
        "site_regions",
        lambda: [
            {"id": 10, "city": "Самара", "addressForSite": "Самара, Ленина 5"},
            {"id": 20, "city": "Оренбург", "addressForSite": "Оренбург, Чкалова 1"},
        ],
    )

    calls: list[tuple[int, int, int | None]] = []

    def fake_fetch(doctor_id, region_id, company_unit_id=None, is_favorite=False):
        _ = is_favorite
        calls.append((int(doctor_id), int(region_id), None if company_unit_id is None else int(company_unit_id)))
        return [{"serviceName": "Прием", "serviceId": 1, "cost": 1000}]

    monkeypatch.setattr(api_price, "fetch_doctor_prices", fake_fetch)

    api_price.update_doctor_prices(force=True)

    # Проверяем, что запросы в doctorServicePricesByRegion ушли только по Самаре (regionId=10).
    assert calls
    assert all(region_id == 10 for _, region_id, _ in calls)


# --- BUG-2026-09-25-NO-DOCTORS-FOUND-FALSE (30.09) ---------------------------
# /regions бэкенда medserver-egisz НЕ отдаёт поле city: город — в дереве parent
# (Все → область → город → филиалы), у филиала адрес без города. Тесты выше
# кормили сборщик старой структурой с city и были зелёными, а на проде сборщик
# находил 3 «самарских» региона из 33, не делал ни одного запроса и каждый день
# писал пустой doctor_prices → «подходящих врачей не нашёл» в каждом ответе о цене
# приёма. Структура ниже повторяет прод (см. тест BUG-A в test_messenger_services).
_PROD_LIKE_REGIONS = [
    {"id": 1, "parent": None, "name": "Все", "addressForSite": None},
    {"id": 2, "parent": 1, "name": "Самарская область", "addressForSite": None},
    {"id": 3, "parent": 2, "name": "Самара", "addressForSite": None},
    {"id": 8, "parent": 2, "name": "Новокуйбышевск", "addressForSite": None},
    {"id": 101, "parent": 3, "name": "Ленина 5", "companyName": "Наука-Самара", "addressForSite": "пр.Ленина, 5"},
    {"id": 102, "parent": 3, "name": "Гагарина 64 Самара", "companyName": "Наука-Самара", "addressForSite": "ул.Гагарина, 64"},
    {"id": 201, "parent": 8, "name": "Пирогова 4", "companyName": "Наука-Самара", "addressForSite": "ул. Пирогова, 4"},
]


def _prod_like_regions():
    return [dict(r) for r in _PROD_LIKE_REGIONS]


def test_update_doctor_prices_finds_samara_doctors_on_prod_like_regions(monkeypatch, tmp_path: Path):
    target = tmp_path / "doctor_prices_20260930.jsonl"
    monkeypatch.setattr(api_price, "doctor_prices_path", lambda date=None: target)
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_price,
        "_load_doctors",
        lambda: [
            {"id": 1, "fio": "Кардиолог Самары", "region_ids": [101], "regions": ["пр.Ленина, 5"]},
            {"id": 2, "fio": "Врач спутника", "region_ids": [201], "regions": ["ул. Пирогова, 4"]},
        ],
    )
    monkeypatch.setattr(
        api_price.api_nayka,
        "site_doctor_regions",
        lambda: [
            {"worker": 1, "region": 101, "companyUnit": 17},
            {"worker": 2, "region": 201, "companyUnit": 17},
        ],
    )
    monkeypatch.setattr(api_price.api_nayka, "site_doctors", lambda: [])
    monkeypatch.setattr(api_price.api_nayka, "site_regions", _prod_like_regions)

    calls: list[tuple[int, int]] = []

    def fake_fetch(doctor_id, region_id, company_unit_id=None, is_favorite=False):
        _ = company_unit_id, is_favorite
        calls.append((int(doctor_id), int(region_id)))
        return [{"serviceName": "Прием (осмотр, консультация) врача-кардиолога", "cost": 3000}]

    monkeypatch.setattr(api_price, "fetch_doctor_prices", fake_fetch)

    api_price.update_doctor_prices(force=True)

    assert (1, 101) in calls  # самарский филиал без «Самара» в адресе
    assert all(region_id != 201 for _, region_id in calls)  # спутник — свой город
    rows = api_price.jsonl_read(target)
    assert [row["doctorId"] for row in rows] == [1]


def test_doctor_prices_and_router_agree_which_regions_are_samara():
    # Инвариант one_city_rule: сборщик цен и роутер судят о Самаре по ОДНОМУ
    # выведенному городу. Разойдутся — одна из частей бота снова потеряет филиалы.
    from messengers_router.services._regions import _inject_region_cities
    from messengers_router.services._samara_branches import samara_subset

    router_ids = {r["id"] for r in samara_subset(_inject_region_cities(_prod_like_regions()))}

    assert api_price._samara_region_ids(_prod_like_regions()) == router_ids
    assert {101, 102} <= router_ids and 201 not in router_ids


def test_region_city_derivation_has_one_implementation():
    # Вывод города из дерева parent — одна функция на весь код. Копия в сборщике
    # цен и была этим багом: фикс BUG-A (88727d8) починил роутер, копия осталась.
    from agent_logic_2.nayka_api import region_city
    from messengers_router.services import _regions

    assert _regions._inject_region_cities is region_city.inject_region_cities
    assert _regions._derive_region_city is region_city.derive_region_city


# --- empty_snapshot_is_not_success (30.09) ------------------------------------
# Пустой doctor_prices писался с «✅ doctor_prices обновлён (0 цен…)» — и так
# ~16 недель никто не видел, что в ответах о цене приёма пропали врачи. Пустая
# сборка — ошибка в логе, а бот работает на последнем непустом срезе: сбой МИС в
# 08:15 не должен стирать врачей из ответов на сутки.
def _patch_builder_returning_nothing(monkeypatch, target: Path):
    monkeypatch.setattr(api_price, "doctor_prices_path", lambda date=None: target)
    monkeypatch.setattr(api_price, "cleanup_old", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_price,
        "_load_doctors",
        lambda: [{"id": 1, "fio": "Кардиолог Самары", "region_ids": [101], "regions": ["пр.Ленина, 5"]}],
    )
    monkeypatch.setattr(api_price.api_nayka, "site_doctor_regions", lambda: [{"worker": 1, "region": 101, "companyUnit": 17}])
    monkeypatch.setattr(api_price.api_nayka, "site_doctors", lambda: [])
    monkeypatch.setattr(api_price.api_nayka, "site_regions", _prod_like_regions)
    monkeypatch.setattr(api_price, "fetch_doctor_prices", lambda *args, **kwargs: [])  # МИС не отдала ничего


def test_empty_doctor_prices_build_keeps_last_nonempty_snapshot(monkeypatch, tmp_path: Path, caplog):
    previous = tmp_path / "doctor_prices_20260929.jsonl"
    good = {"doctorId": 1, "regionId": 101, "serviceName": "Прием кардиолога", "cost": 3000}
    api_price.jsonl_write(previous, [good])
    target = tmp_path / "doctor_prices_20260930.jsonl"
    _patch_builder_returning_nothing(monkeypatch, target)

    with caplog.at_level("INFO", logger=api_price.log.name):
        api_price.update_doctor_prices(force=True)

    assert api_price.jsonl_read(target) == [good]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors and "ПУСТЫМ" in errors[0].getMessage()
    assert not any("✅" in r.getMessage() for r in caplog.records)


def test_empty_doctor_prices_build_without_previous_is_an_error_not_success(monkeypatch, tmp_path: Path, caplog):
    target = tmp_path / "doctor_prices_20260930.jsonl"
    _patch_builder_returning_nothing(monkeypatch, target)

    with caplog.at_level("INFO", logger=api_price.log.name):
        api_price.update_doctor_prices(force=True)

    assert target.exists() and api_price.jsonl_read(target) == []  # файл есть: иначе каждый запрос пересобирал бы срез
    assert any(r.levelname == "ERROR" and "ПУСТЫМ" in r.getMessage() for r in caplog.records)
    assert not any("✅" in r.getMessage() for r in caplog.records)
