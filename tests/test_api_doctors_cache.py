import json
from datetime import date

from agent_logic_2.nayka_api import api_nayka


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_find_existing_doctors_file_skips_empty_active(tmp_path, monkeypatch):
    active = tmp_path / "doctors_20260307.jsonl"
    prev = tmp_path / "doctors_20260306.jsonl"
    active.write_text("", encoding="utf-8")
    _write_jsonl(prev, [{"id": 1, "fio": "Трубин Алексей Юрьевич"}])

    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")

    assert api_nayka.find_existing_doctors_file() == prev


def test_get_cached_doctors_data_falls_back_to_last_nonempty_when_refresh_empty(tmp_path, monkeypatch):
    active = tmp_path / "doctors_20260307.jsonl"
    prev = tmp_path / "doctors_20260306.jsonl"
    active.write_text("", encoding="utf-8")
    rows = [{"id": 1, "fio": "Трубин Алексей Юрьевич"}]
    _write_jsonl(prev, rows)

    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])

    assert api_nayka.get_cached_doctors_data() == rows


def test_save_doctors_data_does_not_create_empty_active_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")
    monkeypatch.setattr(api_nayka, "cleanup_old_doctors_files", lambda keep_dates=None: None)

    api_nayka.save_doctors_data([])

    assert not (tmp_path / "doctors_20260307.jsonl").exists()


def test_get_cached_doctors_data_does_not_refresh_when_placeholder_regions_present(tmp_path, monkeypatch):
    active = tmp_path / "doctors_20260307.jsonl"
    rows = [
        {
            "id": 1,
            "fio": "Трубин Алексей Юрьевич",
            "ord": None,
            "unit_links": [],
            "main_units": [],
            "regions": ["ID 8502"],
        }
    ]
    _write_jsonl(active, rows)

    monkeypatch.setattr(api_nayka, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api_nayka, "get_active_date_str", lambda: "20260307")

    calls = {"refresh": 0}

    def fake_get_all_doctors():
        calls["refresh"] += 1
        return []

    monkeypatch.setattr(api_nayka, "get_all_doctors", fake_get_all_doctors)

    result = api_nayka.get_cached_doctors_data()

    assert result == rows
    assert calls["refresh"] == 0


def test_find_doctor_schedule_uses_extended_lookahead_window(monkeypatch):
    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeDate:
        @classmethod
        def today(cls):
            return date(2026, 4, 1)

    seen_schedule_urls: list[str] = []

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/doctors"):
            return _FakeResp([{"id": 1, "fio": "Куршина Марина Владимировна", "ord": 1}])
        if url.endswith("/doctorCompanyUnits"):
            return _FakeResp([{"worker": 1, "specialization": "педиатр"}])
        if url.endswith("/doctorRegions"):
            return _FakeResp([{"worker": 1, "companyUnit": 38, "region": 8882}])
        if "/doctorSchedule?" in url:
            seen_schedule_urls.append(url)
            return _FakeResp([{"id": 501, "curDate": "2026-04-14", "startTime": "09:00", "endTime": "12:00"}])
        if "/doctorScheduleCells?" in url:
            return _FakeResp([{"startTime": "09:30", "free": True}])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(api_nayka, "site_regions", lambda: [{"id": 8882, "name": "Самара", "addressForSite": "г. Самара, пр. Ленина, 5"}])
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)
    monkeypatch.setattr(api_nayka, "SCHEDULE_LOOKAHEAD_DAYS", 14)

    result = api_nayka.find_doctor_schedule("Куршина")

    assert isinstance(result, list)
    assert seen_schedule_urls, "Expected doctorSchedule requests"
    assert any("startDate=2026-04-01" in url for url in seen_schedule_urls)
    assert any("endDate=2026-04-15" in url for url in seen_schedule_urls)
