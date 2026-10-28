"""Даты слотов расписания МИС — один формат на входе; ничего не разобрано — перевод (ход (д), шаг 8, DATA-15).

Было: МИС присылает дату дня ISO (`2026-10-07`); другой формат (`07.10.2026`) молча выключал фильтр по дате и сбивал
«ближайшее окно»: пациент просил «на 20 октября» — бот показывал и 7-е, сырой датой.

Стало (решение владельца 18.10 «привести все варианты входящих дат к одному формату»): ответ расписания нормализуется на
входе (`schedule_parse_watch.normalize_schedule` в `Services._fetch_schedule_source`): ISO с временем и без,
`ДД.ММ.ГГГГ`, `ДД.ММ.ГГ` → `ГГГГ-ММ-ДД`. Не разобралась ни одна дата расписания — под `mis_honest_degradation` перевод
`service_error_schedule` (и по фамилии, и по специальности — как сбой источника).

Инвариант класса: дата слота от МИС доходит до выбора окна и рендера в одном формате; непонятное расписание не
выдаётся за «нет расписания».
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from messengers_router import renderer
from messengers_router.policies import handoff_message
from messengers_router.schedule_parse_watch import dates_all_unparsed, normalize_schedule, parse_day
from messengers_router.services import Services


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2026-10-07", date(2026, 10, 7)), ("2026-10-07T00:00:00", date(2026, 10, 7)), ("07.10.2026", date(2026, 10, 7)), ("07.10.26", date(2026, 10, 7)), ("7 октября", None), ("", None)],
)
def test_day_formats(raw, expected):
    assert parse_day(raw) == expected


def _payload(day: str) -> list[dict]:
    return [{"fio": "Трубин Алексей Юрьевич", "regions": ["пр.Ленина, 5"], "schedule": {"пр.Ленина, 5": [{"date": day, "slots": ["09:00", "10:30"]}, {"date": "2026-10-20", "slots": ["13:00"]}]}}]


def test_non_iso_date_is_filtered_like_iso_after_normalization():
    asked = {"date_from": "2026-10-20"}
    iso = renderer.format_doctor_schedule_for_patient({"schedule": normalize_schedule(_payload("2026-10-07"))}, asked)
    dotted = renderer.format_doctor_schedule_for_patient({"schedule": normalize_schedule(_payload("07.10.2026"))}, asked)

    assert dotted == iso
    assert "07.10.2026" not in dotted and "20 октября" in dotted


def test_normalization_does_not_touch_the_source_rows():
    raw = _payload("07.10.2026")
    normalize_schedule(raw)
    assert raw[0]["schedule"]["пр.Ленина, 5"][0]["date"] == "07.10.2026"  # кэш расписания хранит свои копии


def test_all_unparsed_is_detected_only_when_nothing_is_readable():
    assert dates_all_unparsed(_payload("7 октября")) is False  # второй день читается
    nothing = [{"fio": "X", "schedule": {"A": [{"date": "7 октября", "slots": ["09:00"]}]}}]
    assert dates_all_unparsed(nothing) is True
    assert dates_all_unparsed([]) is False


def test_unreadable_schedule_hands_off(monkeypatch):
    svc = Services()

    async def doctors():
        return [{"id": 1, "fio": "Иванов Иван Иванович", "regions": ["г. Самара, пр. Ленина, 5"]}]

    async def schedule(_candidate, _region=None):
        return [{"fio": "Иванов Иван Иванович", "regions": ["г. Самара, пр. Ленина, 5"], "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "в среду", "slots": ["09:00"]}]}}]

    async def tokens():
        return {"г. самара, пр. ленина, 5"}

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", doctors)
    monkeypatch.setattr(svc, "_get_schedule_payload_cached", schedule)
    monkeypatch.setattr(svc, "_samara_region_tokens", tokens)

    res = asyncio.run(svc.doctors_schedule_week("расписание Иванова", {"doctor_name": "Иванов"}))

    assert bool(res.get("handoff_required")) is True
    assert res.get("handoff_message") == handoff_message("service_error_schedule")
