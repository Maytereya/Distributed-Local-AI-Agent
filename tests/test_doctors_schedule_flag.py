"""Признак «есть расписание на 7 дней» в срезе врачей (ход (г), L-06б).

Сборщик среза уже проверял расписание врача по площадкам (`/doctorSchedule` на 7 дней), но результат выбрасывал — бот
отвечал «расписание не найдено», не отличая «врач есть, онлайн-окон нет» от сбоя. Инвариант: `has_schedule_7d` — True,
только если МИС показала окна; False — только если МИС ответила и окон нет ни на одной площадке; сбой запроса — None
(«нет расписания» по сбою не утверждаем, принцип 3). Журнал: STEP-2026-10-21-G-L06-DOCTORS-BY-SERVICE.
"""

from __future__ import annotations

import pytest
import requests

from agent_logic_2.nayka_api import api_nayka


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


@pytest.fixture
def fake_mis(monkeypatch):
    # врач 1 — окна есть; 2 — МИС ответила пусто; 3 — запрос упал; 4 — МИС сама пометила «расписания нет»
    doctors = [{"id": i, "fio": f"Врач{i}", "ord": i} for i in (1, 2, 3, 4)]
    monkeypatch.setattr(api_nayka, "site_doctors", lambda: doctors)
    monkeypatch.setattr(api_nayka, "site_company_units", lambda: [{"id": 10, "name": "Врач-терапевт"}])
    monkeypatch.setattr(api_nayka, "site_doctor_company_units", lambda: [{"worker": d["id"], "companyUnit": 10, "main": True} for d in doctors])
    monkeypatch.setattr(
        api_nayka,
        "site_doctor_regions",
        lambda: [{"worker": d["id"], "region": 100, "companyUnit": 10, "hasSchedule": d["id"] != 4} for d in doctors],
    )
    monkeypatch.setattr(api_nayka, "site_regions", lambda *a, **k: [{"id": 100, "name": "пр.Ленина, 5"}])

    def session_get(url, **_kw):
        if "doctor=3&" in url:
            raise requests.ConnectionError("МИС не ответила")
        return _Resp([{"id": 1}] if "doctor=1&" in url else [])

    monkeypatch.setattr(api_nayka, "_session_get", session_get)
    return doctors


@pytest.mark.real_mis_loader
def test_schedule_flag_tells_no_windows_from_failed_check(fake_mis):
    built = {d["id"]: d for d in api_nayka.get_all_doctors()}

    assert {i: built[i]["has_schedule_7d"] for i in built} == {1: True, 2: False, 3: None, 4: False}
    # площадки врача без окон остаются в срезе, как было (флаг — дополнение, не фильтр)
    assert all(built[i]["region_ids"] == [100] for i in built)


def test_slice_without_flag_is_not_outdated():
    # срез, собранный до 21.10, пересобирать через VPN не нужно: поле необязательное
    assert api_nayka._doctors_schema_outdated([{"id": 1, "ord": 1, "unit_links": []}]) is False
