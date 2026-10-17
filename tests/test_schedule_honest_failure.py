"""Расписание по специальности при сбое МИС — честный перевод (ход (д), шаг 4, DATA-3; `mis_honest_degradation`).

Было: `_schedule_by_specialty` опрашивает до 8 врачей; любой сбой (`ScheduleSourceUnavailable`, таймаут, исключение)
становился «у врача ничего нет»: не ответил никто — «расписание не найдено. Уточните фамилию врача или город» (сбой
переложен на пациента); ответила часть — «ближайшее окно» только из ответивших. При сбое кэш расписания отдавал
ответ до 10 минут давности без пометки (занятый слот — свободным).

Стало под `mis_honest_degradation = on` (решения владельца 08.10 «расписание — только живое» и 18.10 «сбой МИС →
оператор»): не ответил хотя бы один врач — перевод `service_error_schedule`, как в пути по фамилии, и
`log_degraded`; устаревший ответ расписания при сбое не отдаётся. Под `off` — как было (кроме `log_degraded` — только лог).

Инвариант класса: сбой источника расписания не выдаётся пациенту за «расписания нет» и не сужает выбор окна молча.

Достижимость: с мая вопрос о расписании по одной специальности планировщик отдаёт списком врачей (`planner.py`:
«расписание <специальности>» → DOCTOR_INFO, без опроса расписаний — честно, не «нет расписания»); путь по специальности
остаётся при названном в контексте враче и вопросе со специальностью. Поэтому проверка — на уровне сервиса
`doctors_schedule_week`, а не через эндпоинт.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from messengers_router import resilience as _R
from messengers_router.policies import handoff_message
from messengers_router.services import Services
from schedule_ttl_cache import AsyncListTTLStaleCache

_PHLEBOLOGISTS = [
    {
        "id": i,
        "fio": fio,
        "ord": i,
        "specialization": "флеболог",
        "regions": ["г. Самара, пр. Ленина, 5"],
        "units": ["Врач-флеболог"],
        "unit_links": [{"company_unit_name": "Врач-флеболог", "main": True, "specialization": "флеболог"}],
        "main_units": ["Врач-флеболог"],
        "main_specializations": ["флеболог"],
    }
    for i, fio in ((1, "Тестов Тест Тестович"), (2, "Петров Пётр Петрович"))
]


def _row(fio: str) -> list[dict]:
    return [{"fio": fio, "regions": ["г. Самара, пр. Ленина, 5"], "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-10-20", "slots": ["09:00"]}]}}]


def _patch(monkeypatch, failing: set[str], honest: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_MIS_HONEST_DEGRADATION", honest, raising=False)

    async def doctors(self):
        return [dict(d) for d in _PHLEBOLOGISTS]

    async def tokens(self):
        return {"г. самара, пр. ленина, 5"}

    async def schedule(self, surname, _region=None):
        if surname in failing:
            raise ConnectionError(f"{surname}: МИС недоступна")
        return _row(next(d["fio"] for d in _PHLEBOLOGISTS if d["fio"].startswith(surname)))

    monkeypatch.setattr(Services, "_ensure_doctors_cache_loaded", doctors)
    monkeypatch.setattr(Services, "_samara_region_tokens", tokens)
    monkeypatch.setattr(Services, "_get_schedule_payload_cached", schedule)


@pytest.mark.parametrize("failing", [{"Тестов", "Петров"}, {"Тестов"}], ids=["all_failed", "part_failed"])
def test_specialty_schedule_failure_hands_off_under_the_switch(failing, monkeypatch, caplog):
    _patch(monkeypatch, failing, "on")

    with caplog.at_level(logging.WARNING, logger="messengers_router.resilience"):
        res = asyncio.run(Services().doctors_schedule_week("расписание флеболога", {}))

    assert res.get("handoff_required") is True
    assert res.get("handoff_message") == handoff_message("service_error_schedule")
    assert res.get("schedule") == []
    assert res.get("degraded") is True and res.get("degraded_mode") == _R.FM_EXCEPTION
    assert "upstream=doctor_schedule" in caplog.text


@pytest.mark.parametrize(("failing", "rows"), [({"Тестов", "Петров"}, 0), ({"Тестов"}, 1)], ids=["all_failed", "part_failed"])
def test_switch_off_keeps_the_old_answer_but_logs(failing, rows, monkeypatch, caplog):
    _patch(monkeypatch, failing, "off")

    with caplog.at_level(logging.WARNING, logger="messengers_router.resilience"):
        res = asyncio.run(Services().doctors_schedule_week("расписание флеболога", {}))

    assert res.get("handoff_required") is not True
    assert len(res.get("schedule") or []) == rows
    assert "upstream=doctor_schedule" in caplog.text  # сбой виден и под off


def test_healthy_schedule_is_untouched_under_the_switch(monkeypatch):
    _patch(monkeypatch, set(), "on")

    res = asyncio.run(Services().doctors_schedule_week("расписание флеболога", {}))

    assert res.get("handoff_required") is not True and res.get("schedule")


@pytest.mark.parametrize(("allow_stale", "stale_served"), [(True, True), (False, False)])
def test_stale_schedule_is_not_served_on_failure_when_only_live_is_allowed(allow_stale, stale_served):
    clock = [1000.0]
    cache = AsyncListTTLStaleCache(fresh_ttl_seconds=30, stale_ttl_seconds=600, negative_ttl_seconds=15, max_keys=10, time_func=lambda: clock[0])

    async def ok():
        return [{"fio": "Тестов"}]

    async def down():
        raise ConnectionError("МИС недоступна")

    async def scenario():
        await cache.get_or_fetch("k", ok)
        clock[0] += 60  # свежесть прошла, окно «устаревшего» — ещё нет
        return await cache.get_or_fetch("k", down, allow_stale=allow_stale)

    if stale_served:
        assert asyncio.run(scenario()) == [{"fio": "Тестов"}]
    else:
        with pytest.raises(ConnectionError):
            asyncio.run(scenario())
