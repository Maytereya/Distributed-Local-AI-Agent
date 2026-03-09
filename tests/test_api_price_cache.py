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
