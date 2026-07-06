"""TTL-кэш «шапки» find_doctor_schedule: справочники кэшируются, СЛОТЫ — нет.

После П7 (параллельный fanout, e8be283) узким местом стала последовательная
«шапка» каскада: site_regions + /doctors + /doctorCompanyUnits + /doctorRegions
— 4 realtime HTTP на КАЖДЫЙ запрос расписания (~4-5с из 6.4с, замер 2026-07-07).
Справочники меняются ~раз в день (прецедент: doctors_mem_ttl_seconds=300 в
services/core.py), а вот слоты обязаны быть живыми на каждый запрос — владелец:
«не можем показывать слоты даже предыдущей минуты».

Инварианты (класс):
1. ГЛАВНЫЙ: /doctorSchedule и /doctorScheduleCells запрашиваются на КАЖДЫЙ
   вызов (слоты всегда realtime) — кэш шапки на них не распространяется.
2. Справочники в пределах TTL запрашиваются один раз на процесс.
3. Kill-switch: TTL<=0 → кэш выключен, каждый вызов идёт в CRM (прежнее
   поведение).
4. Сбой/пустой ответ шапки НЕ кэшируется (нет отравления кэша на TTL): ошибка
   остаётся честной ошибкой, следующий вызов снова идёт в CRM.
5. Протухание TTL → рефетч.
"""

from __future__ import annotations

import time
from datetime import date

import requests

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
        return date(2026, 7, 7)


def _install_counting_fixture(monkeypatch, *, fail_doctors_first=False):
    """1 врач, 1 филиал, 1 день; счётчик вызовов по типам URL."""

    calls = {"site_regions": 0, "doctors": 0, "companyunits": 0, "doctorregions": 0, "schedule": 0, "cells": 0}
    state = {"doctors_failed_once": False}

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        if url.endswith("/doctors"):
            calls["doctors"] += 1
            if fail_doctors_first and not state["doctors_failed_once"]:
                state["doctors_failed_once"] = True
                raise requests.ConnectionError("boom /doctors")
            return _FakeResp([{"id": 1, "fio": "Куршина Марина Владимировна", "ord": 1}])
        if url.endswith("/doctorCompanyUnits"):
            calls["companyunits"] += 1
            return _FakeResp([{"worker": 1, "specialization": "педиатр"}])
        if url.endswith("/doctorRegions"):
            calls["doctorregions"] += 1
            return _FakeResp([{"worker": 1, "companyUnit": 38, "region": 8882}])
        if "/doctorSchedule?" in url:
            calls["schedule"] += 1
            return _FakeResp([{"id": 501, "curDate": "2026-07-08", "startTime": "09:00", "endTime": "12:00"}])
        if "/doctorScheduleCells?" in url:
            calls["cells"] += 1
            return _FakeResp([{"startTime": "09:30", "free": True}])
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_site_regions(*, realtime=False):
        _ = realtime
        calls["site_regions"] += 1
        return [{"id": 8882, "name": "Самара", "addressForSite": "г. Самара, пр. Ленина, 5"}]

    monkeypatch.setattr(api_nayka, "site_regions", fake_site_regions)
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)
    return calls


def test_slots_fetched_every_call_refs_fetched_once(monkeypatch):
    """ГЛАВНЫЙ инвариант: слоты realtime на каждый вызов, шапка — раз в TTL."""
    monkeypatch.setattr(api_nayka, "SCHEDULE_REFS_TTL_SECONDS", 300.0)
    calls = _install_counting_fixture(monkeypatch)

    r1 = api_nayka.find_doctor_schedule("Куршина")
    r2 = api_nayka.find_doctor_schedule("Куршина")

    assert isinstance(r1, list) and isinstance(r2, list)
    # Слоты — живые оба раза
    assert calls["schedule"] == 2, "слоты /doctorSchedule обязаны запрашиваться на каждый вызов"
    assert calls["cells"] == 2, "слоты /doctorScheduleCells обязаны запрашиваться на каждый вызов"
    # Шапка — один раз
    assert calls["site_regions"] == 1
    assert calls["doctors"] == 1
    assert calls["companyunits"] == 1
    assert calls["doctorregions"] == 1


def test_kill_switch_ttl_zero_disables_cache(monkeypatch):
    monkeypatch.setattr(api_nayka, "SCHEDULE_REFS_TTL_SECONDS", 0.0)
    calls = _install_counting_fixture(monkeypatch)

    api_nayka.find_doctor_schedule("Куршина")
    api_nayka.find_doctor_schedule("Куршина")

    assert calls["site_regions"] == 2
    assert calls["doctors"] == 2
    assert calls["companyunits"] == 2
    assert calls["doctorregions"] == 2


def test_header_fetch_error_is_not_cached(monkeypatch):
    """Сбой /doctors → честная ошибка; повторный вызов снова идёт в CRM и работает."""
    monkeypatch.setattr(api_nayka, "SCHEDULE_REFS_TTL_SECONDS", 300.0)
    calls = _install_counting_fixture(monkeypatch, fail_doctors_first=True)

    r1 = api_nayka.find_doctor_schedule("Куршина")
    assert isinstance(r1, str) and api_nayka.is_api_error_message(r1)

    r2 = api_nayka.find_doctor_schedule("Куршина")
    assert isinstance(r2, list), "после сбоя кэш не должен быть отравлен ошибкой"
    assert calls["doctors"] == 2


def test_ttl_expiry_triggers_refetch(monkeypatch):
    monkeypatch.setattr(api_nayka, "SCHEDULE_REFS_TTL_SECONDS", 0.01)
    calls = _install_counting_fixture(monkeypatch)

    api_nayka.find_doctor_schedule("Куршина")
    time.sleep(0.03)
    api_nayka.find_doctor_schedule("Куршина")

    assert calls["doctors"] == 2
    assert calls["site_regions"] == 2
