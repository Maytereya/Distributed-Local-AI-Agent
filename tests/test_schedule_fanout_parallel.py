"""П7 (Resilience Phase 2): fanout «филиалы × дни» в find_doctor_schedule параллелен.

До П7 все /doctorSchedule (по филиалам) и /doctorScheduleCells (по дням) шли
строго последовательно: врач с 2 филиалами × 12 дней = ~26 HTTP-вызовов цепочкой
(7.2с Трубин при замере baseline-2026-06-26). После П7 независимые fetch'и идут
через ограниченный пул воркеров (не душить medserver).

Инварианты (класс):
1. ПОРЯДОК детерминирован: филиалы — в порядке doctorRegions, дни — в порядке
   ответа /doctorSchedule; сборка по индексам сабмита, НЕ по прибытию ответов
   (медленный день не «уплывает» в конец).
2. Параллельность реальна: медленные независимые fetch'и перекрываются
   (wall-clock < суммы задержек) — иначе П7 не выполняет своей цели.
3. Error-семантика прежняя: сбой /doctorSchedule одного филиала не валит
   остальные; сбой /doctorScheduleCells дня = «нет слотов» этого дня.
"""

from __future__ import annotations

import threading
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
        return date(2026, 7, 6)


_REGIONS = [
    {"id": 100, "name": "Самара", "addressForSite": "г. Самара, пр. Ленина, 5"},
    {"id": 200, "name": "Самара Победы", "addressForSite": "г. Самара, ул. Победы, 83"},
]


def _install_two_region_fixture(monkeypatch, *, slow_day_ids=(), slow_s=0.05, fail_schedule_regions=()):
    """Ставит фикстуру: 1 врач, 2 филиала × 2 дня, управляемые задержки/сбои."""

    inflight = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fake_session_get(url: str, **kwargs):
        _ = kwargs
        with lock:
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
        try:
            if url.endswith("/doctors"):
                return _FakeResp([{"id": 1, "fio": "Трубин Иван Иванович", "ord": 1}])
            if url.endswith("/doctorCompanyUnits"):
                return _FakeResp([{"worker": 1, "specialization": "уролог"}])
            if url.endswith("/doctorRegions"):
                return _FakeResp([
                    {"worker": 1, "companyUnit": 38, "region": 100},
                    {"worker": 1, "companyUnit": 39, "region": 200},
                ])
            if "/doctorSchedule?" in url:
                if "region=100" in url:
                    if 100 in fail_schedule_regions:
                        raise requests.ConnectionError("boom region 100")
                    return _FakeResp([
                        {"id": 501, "curDate": "2026-07-07", "startTime": "09:00", "endTime": "12:00"},
                        {"id": 502, "curDate": "2026-07-08", "startTime": "09:00", "endTime": "12:00"},
                    ])
                if "region=200" in url:
                    if 200 in fail_schedule_regions:
                        raise requests.ConnectionError("boom region 200")
                    return _FakeResp([
                        {"id": 601, "curDate": "2026-07-07", "startTime": "13:00", "endTime": "18:00"},
                        {"id": 602, "curDate": "2026-07-09", "startTime": "13:00", "endTime": "18:00"},
                    ])
                raise AssertionError(f"Unexpected schedule URL: {url}")
            if "/doctorScheduleCells?" in url:
                day_id = int(url.rsplit("=", 1)[1])
                if day_id in slow_day_ids:
                    time.sleep(slow_s)
                return _FakeResp([{"startTime": f"{9 + day_id % 10}:30", "free": True}])
            raise AssertionError(f"Unexpected URL: {url}")
        finally:
            with lock:
                inflight["now"] -= 1

    monkeypatch.setattr(api_nayka, "site_regions", lambda *, realtime=False: _REGIONS)
    monkeypatch.setattr(api_nayka, "_session_get", fake_session_get)
    monkeypatch.setattr(api_nayka, "date", _FakeDate)
    return inflight


def test_fanout_preserves_region_and_day_order_despite_slow_first_day(monkeypatch):
    """Медленный ПЕРВЫЙ день (501) не смещается в конец: порядок = порядок сабмита."""
    _install_two_region_fixture(monkeypatch, slow_day_ids=(501, 601), slow_s=0.05)

    result = api_nayka.find_doctor_schedule("Трубин")

    assert isinstance(result, list) and len(result) == 1
    schedule = result[0]["schedule"]
    # Филиалы в порядке doctorRegions (Ленина → Победы); ключ = addressForSite
    keys = list(schedule.keys())
    assert keys == ["г. Самара, пр. Ленина, 5", "г. Самара, ул. Победы, 83"]
    # Дни каждого филиала в порядке ответа /doctorSchedule, не по прибытию cells
    assert [d["date"] for d in schedule[keys[0]]] == ["2026-07-07", "2026-07-08"]
    assert [d["date"] for d in schedule[keys[1]]] == ["2026-07-07", "2026-07-09"]


def test_fanout_actually_overlaps_independent_fetches(monkeypatch):
    """4 медленных дня × slow_s: параллельный fanout заметно быстрее суммы задержек."""
    slow_s = 0.08
    inflight = _install_two_region_fixture(
        monkeypatch, slow_day_ids=(501, 502, 601, 602), slow_s=slow_s
    )

    t0 = time.monotonic()
    result = api_nayka.find_doctor_schedule("Трубин")
    elapsed = time.monotonic() - t0

    assert isinstance(result, list) and len(result) == 1
    # Последовательно было бы ≥ 4 × slow_s; параллельно (воркеров ≥ 2) — меньше.
    assert elapsed < 4 * slow_s, f"fanout не перекрывается: {elapsed:.3f}s"
    assert inflight["max"] >= 2, "cells-фетчи не шли параллельно"


def test_fanout_bounded_by_worker_limit(monkeypatch):
    """Конкурентность ограничена пулом (не штурмуем medserver без лимита)."""
    inflight = _install_two_region_fixture(
        monkeypatch, slow_day_ids=(501, 502, 601, 602), slow_s=0.03
    )

    api_nayka.find_doctor_schedule("Трубин")

    assert inflight["max"] <= 4, f"конкурентность {inflight['max']} превышает лимит пула"


def test_fanout_one_region_failure_keeps_other_region(monkeypatch):
    """Сбой /doctorSchedule одного филиала не валит расписание другого."""
    _install_two_region_fixture(monkeypatch, fail_schedule_regions=(100,))

    result = api_nayka.find_doctor_schedule("Трубин")

    assert isinstance(result, list) and len(result) == 1
    schedule = result[0]["schedule"]
    assert list(schedule.keys()) == ["г. Самара, ул. Победы, 83"]
    assert [d["date"] for d in schedule["г. Самара, ул. Победы, 83"]] == ["2026-07-07", "2026-07-09"]
