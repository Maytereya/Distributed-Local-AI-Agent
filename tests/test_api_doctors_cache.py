import json
import logging
from datetime import date

import requests

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

    monkeypatch.setattr(api_nayka, "site_regions", lambda *, realtime=False: [{"id": 8882, "name": "Самара", "addressForSite": "г. Самара, пр. Ленина, 5"}])
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)
    monkeypatch.setattr(api_nayka, "SCHEDULE_LOOKAHEAD_DAYS", 14)

    result = api_nayka.find_doctor_schedule("Куршина")

    assert isinstance(result, list)
    assert seen_schedule_urls, "Expected doctorSchedule requests"
    assert any("startDate=2026-04-01" in url for url in seen_schedule_urls)
    assert any("endDate=2026-04-15" in url for url in seen_schedule_urls)


def test_find_doctor_schedule_all_upstream_calls_are_realtime(monkeypatch):
    # OC-2 invariant (Task 8.1): ALL upstream calls in find_doctor_schedule use the
    # realtime fail-fast session — leading catalog calls (site_regions, /doctors,
    # /doctorCompanyUnits, /doctorRegions) AND per-day schedule + cells fetches.
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

    realtime_by_kind: dict[str, bool] = {}

    def fake_session_get(url: str, **kwargs):
        rt = bool(kwargs.get("realtime", False))
        if url.endswith("/doctors"):
            realtime_by_kind["doctors"] = rt
            return _FakeResp([{"id": 1, "fio": "Куршина Марина Владимировна", "ord": 1}])
        if url.endswith("/doctorCompanyUnits"):
            realtime_by_kind["companyunits"] = rt
            return _FakeResp([{"worker": 1, "specialization": "педиатр"}])
        if url.endswith("/doctorRegions"):
            realtime_by_kind["regions"] = rt
            return _FakeResp([{"worker": 1, "companyUnit": 38, "region": 8882}])
        if "/doctorSchedule?" in url:
            realtime_by_kind["schedule"] = rt
            return _FakeResp([{"id": 501, "curDate": "2026-04-14", "startTime": "09:00", "endTime": "12:00"}])
        if "/doctorScheduleCells?" in url:
            realtime_by_kind["cells"] = rt
            return _FakeResp([{"startTime": "09:30", "free": True}])
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_site_regions(*, realtime=False):
        realtime_by_kind["site_regions"] = realtime
        return [{"id": 8882, "name": "Самара", "addressForSite": "г. Самара, пр. Ленина, 5"}]

    monkeypatch.setattr(api_nayka, "site_regions", fake_site_regions)
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)
    monkeypatch.setattr(api_nayka, "SCHEDULE_LOOKAHEAD_DAYS", 14)

    result = api_nayka.find_doctor_schedule("Куршина")

    assert isinstance(result, list)
    assert realtime_by_kind.get("site_regions") is True
    assert realtime_by_kind.get("doctors") is True
    assert realtime_by_kind.get("companyunits") is True
    assert realtime_by_kind.get("regions") is True
    assert realtime_by_kind.get("schedule") is True
    assert realtime_by_kind.get("cells") is True


def test_is_api_error_message_classification():
    # Сбои обращения к CRM → True (обе ветки начинаются с «Не удалось получить»).
    assert api_nayka.is_api_error_message("Не удалось получить список врачей: timeout")
    assert api_nayka.is_api_error_message("Не удалось получить связи врача: 503")
    # Валидные негативы → False (это НЕ сбой, а корректный исход).
    assert not api_nayka.is_api_error_message(api_nayka._NO_FREE_SLOTS_MESSAGE)
    assert not api_nayka.is_api_error_message("Врач с фамилией (или частью ФИО) 'X' не найден.")
    assert not api_nayka.is_api_error_message("Регион 'Луна' не найден.")
    # Не-строки → False.
    assert not api_nayka.is_api_error_message(None)
    assert not api_nayka.is_api_error_message([])
    # «нет слотов» и «api error» — взаимоисключающие классы.
    assert not api_nayka.is_no_free_slots_message("Не удалось получить список врачей: timeout")


def test_find_doctor_schedule_returns_api_error_string_on_doctors_fetch_failure(monkeypatch):
    """Когда падает запрос /doctors, find_doctor_schedule возвращает строку
    «Не удалось получить список врачей: …», которую is_api_error_message
    распознаёт как сбой источника (а не «врач не найден»)."""

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/doctors"):
            raise requests.ConnectionError("read timed out")
        raise AssertionError(f"Unexpected URL after failure: {url}")

    monkeypatch.setattr(api_nayka, "site_regions", lambda *, realtime=False: [{"id": 8882, "name": "Самара"}])
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)

    result = api_nayka.find_doctor_schedule("Паничева")

    assert isinstance(result, str)
    assert api_nayka.is_api_error_message(result)
    assert not api_nayka.is_no_free_slots_message(result)


def test_find_doctor_schedule_logs_warning_on_schedule_fetch_failure(monkeypatch, caplog):
    """Молчаливый `continue` при сбое /doctorSchedule теперь оставляет
    WARNING в логах — иначе частичный сбой не отличить от «нет приёма»."""

    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/doctors"):
            return _FakeResp([{"id": 1, "fio": "Паничева Анна", "ord": 1}])
        if url.endswith("/doctorCompanyUnits"):
            return _FakeResp([{"worker": 1, "specialization": "терапевт"}])
        if url.endswith("/doctorRegions"):
            return _FakeResp([{"worker": 1, "companyUnit": 38, "region": 8882}])
        if "/doctorSchedule?" in url:
            raise requests.ConnectionError("read timed out")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(api_nayka, "site_regions", lambda *, realtime=False: [{"id": 8882, "name": "Самара"}])
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)

    with caplog.at_level(logging.WARNING, logger="agent_logic_2.nayka_api.api_nayka"):
        result = api_nayka.find_doctor_schedule("Паничева")

    # Врач найден, регион есть, но все /doctorSchedule упали → «нет слотов».
    assert api_nayka.is_no_free_slots_message(result)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("doctorSchedule" in m for m in warnings), warnings
    # И отдельный сигнал, что вывод «нет слотов» мог быть искажён сбоем fetch.
    assert any("no-slots" in m.lower() or "incomplete" in m.lower() for m in warnings), warnings


# ── OC-2 realtime fail-fast profile (Task 8, resilience Фаза1) ─────────────


def test_realtime_session_profile_is_fail_fast():
    # CLASS INVARIANT (OC-2): realtime session = no retries + 8s read; background = total 3, unchanged.
    rt = api_nayka.SESSION_REALTIME.get_adapter("http://x")
    bg = api_nayka.SESSION.get_adapter("http://x")
    assert rt.max_retries.total == 0
    assert bg.max_retries.total == 3            # background MUST stay unchanged
    assert api_nayka.REALTIME_TIMEOUT[1] == 8.0         # realtime read timeout
    assert api_nayka.DEFAULT_TIMEOUT[1] == 20.0         # background read timeout unchanged


def test_session_get_routes_by_realtime_flag(monkeypatch):
    calls = []
    class _Resp: ...
    monkeypatch.setattr(api_nayka.SESSION, "get", lambda url, **kw: calls.append(("bg", url, kw.get("timeout"))) or _Resp())
    monkeypatch.setattr(api_nayka.SESSION_REALTIME, "get", lambda url, **kw: calls.append(("rt", url, kw.get("timeout"))) or _Resp())
    api_nayka._session_get("http://x")
    api_nayka._session_get("http://y", realtime=True)
    assert calls[0][0] == "bg" and calls[0][2] == api_nayka.DEFAULT_TIMEOUT
    assert calls[1][0] == "rt" and calls[1][2] == api_nayka.REALTIME_TIMEOUT


def test_site_regions_threads_realtime_flag(monkeypatch):
    seen = []
    class _Resp:
        def raise_for_status(self): pass
        def json(self): return []
    monkeypatch.setattr(api_nayka, "_session_get", lambda url, *, realtime=False, **kw: seen.append(realtime) or _Resp())
    api_nayka.site_regions(realtime=True)
    api_nayka.site_regions()
    assert seen == [True, False]


def test_result_for_patient_uses_realtime(monkeypatch):
    captured = {}
    class _Resp:
        headers = {"Content-Type": "text/plain"}
        text = "https://naykalab.ru/result/blank.pdf"
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {}
    def fake_get(url, *, realtime=False, **kw):
        captured["realtime"] = realtime
        return _Resp()
    monkeypatch.setattr(api_nayka, "_session_get", fake_get)
    api_nayka.site_result_for_patient(surname="Тестов", year=1990, filial="Бг", number=1)
    assert captured.get("realtime") is True
