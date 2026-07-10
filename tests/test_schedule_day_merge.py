"""Мерж задвоенных дней расписания: две привязки врача к одному адресу.

Прод-проба 10.07 («расписание Трубина»): в /doctorRegions у врача ДВЕ записи
на region 8882 «Ленина 5» через разные companyUnit (17 и 92). Каждая даёт свой
/doctorSchedule, и оба набора дней доклеивались в один блок адреса — пациент
видел «15 июля … 24 июля» дважды. Поведение существовало и до П7-параллелизации.

Инварианты (класс):
1. Дни одного АДРЕСА мержатся по дате: слоты = union (без дублей), окно
   приёма = min(start)..max(end); дни, которых нет у второго подразделения,
   не пропадают. Порядок дат — как пришли из API (first-seen).
2. РАЗНЫЕ адреса не мержатся (по одному блоку на филиал — прежнее поведение).
3. Врач с одной привязкой — байт-в-байт прежний ответ (слоты не
   пересортировываются без нужды).
"""

from __future__ import annotations

from datetime import date

from agent_logic_2.nayka_api import api_nayka


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
        return date(2026, 7, 10)


def _install_two_unit_fixture(monkeypatch):
    """1 врач, ОДИН регион «Ленина 5», ДВЕ привязки (companyUnit 17 и 92)."""

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/doctors"):
            return _FakeResp([{"id": 3109, "fio": "Трубин Алексей Юрьевич", "ord": 1}])
        if url.endswith("/doctorCompanyUnits"):
            return _FakeResp([{"worker": 3109, "specialization": "уролог"}])
        if url.endswith("/doctorRegions"):
            return _FakeResp([
                {"worker": 3109, "companyUnit": 17, "region": 8882},
                {"worker": 3109, "companyUnit": 92, "region": 8882},
            ])
        if "/doctorSchedule?" in url:
            if "companyUnit=17" in url:
                return _FakeResp([
                    {"id": 501, "curDate": "2026-07-15", "startTime": "09:00", "endTime": "12:00"},
                    {"id": 502, "curDate": "2026-07-20", "startTime": "09:00", "endTime": "12:00"},
                ])
            if "companyUnit=92" in url:
                return _FakeResp([
                    # тот же день 15.07: другой слот + один общий; и свой день 22.07
                    {"id": 601, "curDate": "2026-07-15", "startTime": "15:00", "endTime": "19:00"},
                    {"id": 602, "curDate": "2026-07-22", "startTime": "15:00", "endTime": "19:00"},
                ])
            raise AssertionError(f"Unexpected schedule URL: {url}")
        if "/doctorScheduleCells?" in url:
            day_id = int(url.rsplit("=", 1)[1])
            slots = {
                501: [{"startTime": "09:30", "free": True}, {"startTime": "10:00", "free": True}],
                502: [{"startTime": "09:30", "free": True}],
                601: [{"startTime": "15:30", "free": True}, {"startTime": "09:30", "free": True}],
                602: [{"startTime": "15:30", "free": True}],
            }[day_id]
            return _FakeResp(slots)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(
        api_nayka, "site_regions",
        lambda *, realtime=False: [{"id": 8882, "name": "Ленина 5", "addressForSite": "пр.Ленина, 5"}],
    )
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)


def test_same_address_days_merged_no_duplicates(monkeypatch):
    _install_two_unit_fixture(monkeypatch)

    result = api_nayka.find_doctor_schedule("Трубин")

    assert isinstance(result, list) and len(result) == 1
    schedule = result[0]["schedule"]
    assert list(schedule.keys()) == ["пр.Ленина, 5"], "один адрес — один блок"
    rows = schedule["пр.Ленина, 5"]
    dates = [r["date"] for r in rows]
    assert dates == ["2026-07-15", "2026-07-20", "2026-07-22"], f"дни без дублей, first-seen порядок: {dates}"

    by_date = {r["date"]: r for r in rows}
    # 15.07 из ДВУХ подразделений: union слотов (общий 09:30 не задвоен), окно расширено
    merged = by_date["2026-07-15"]
    assert merged["slots"] == ["09:30", "10:00", "15:30"], merged["slots"]
    assert merged["start"] == "09:00" and merged["end"] == "19:00"
    # уникальные дни каждого подразделения не потерялись
    assert by_date["2026-07-20"]["slots"] == ["09:30"]
    assert by_date["2026-07-22"]["slots"] == ["15:30"]


def test_single_unit_schedule_unchanged(monkeypatch):
    """Анти-регресс: врач с одной привязкой — прежний ответ, слоты в порядке API."""

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/doctors"):
            return _FakeResp([{"id": 1, "fio": "Куршина Марина Владимировна", "ord": 1}])
        if url.endswith("/doctorCompanyUnits"):
            return _FakeResp([{"worker": 1, "specialization": "педиатр"}])
        if url.endswith("/doctorRegions"):
            return _FakeResp([{"worker": 1, "companyUnit": 38, "region": 8882}])
        if "/doctorSchedule?" in url:
            return _FakeResp([{"id": 501, "curDate": "2026-07-15", "startTime": "09:00", "endTime": "12:00"}])
        if "/doctorScheduleCells?" in url:
            # намеренно НЕхронологический порядок — мерж не должен пересортировывать
            return _FakeResp([{"startTime": "10:00", "free": True}, {"startTime": "09:30", "free": True}])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(
        api_nayka, "site_regions",
        lambda *, realtime=False: [{"id": 8882, "name": "Ленина 5", "addressForSite": "пр.Ленина, 5"}],
    )
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)

    result = api_nayka.find_doctor_schedule("Куршина")

    rows = result[0]["schedule"]["пр.Ленина, 5"]
    assert len(rows) == 1
    assert rows[0]["slots"] == ["10:00", "09:30"], "без мержа порядок слотов из API не трогаем"
