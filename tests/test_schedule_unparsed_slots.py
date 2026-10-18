"""Неразобранные даты и время слотов расписания МИС видны — ход (д), шаг 8а, DATA-15 (17.10).

Было: выбор «ближайшего окна» (`_iter_slot_datetimes`) и рендер расписания (`renderer.format_doctor_schedule_for_patient`)
молча пропускали слот, дату или время которого не разобрали, а неразобранная дата дня выключала фильтр по дате: при
дате «07.10.2026» вместо ISO «ближайшим» становилось более позднее окно, фильтр «на 09.10» показывал другие дни —
и нигде не оставалось следа.

Стало (ответ не меняется — текст «расписание временно недоступно» при «ничего не разобрано» ждёт решения владельца):
каждый пропуск — +1 к полю телеметрии хода `slot_unparsed` и `logger.warning` с образцом один раз на место и форму
значения. Инвариант класса: пропущенный слот расписания оставляет след в строке хода и в логе.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from messengers_router import renderer, schedule_parse_watch, turn_telemetry
from messengers_router.services._doctors_helpers import _iter_slot_datetimes


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(schedule_parse_watch, "_warned", set())
    monkeypatch.setattr(turn_telemetry, "_emit", lambda row: None)


def _in_turn(fn):
    token = turn_telemetry.begin("test", session_id="slots")
    result = fn()
    row = turn_telemetry.finish(token, handoff=False, chars=0) or {}
    return result, row


_SCHEDULE = {
    "Ленина 5": [
        {"date": "07.10.2026", "slots": ["09:00", "10:00"]},  # не ISO — не разбирается
        {"date": "2026-10-20", "slots": ["13:00", "--:--"]},  # второе время не разбирается
    ]
}


def test_nearest_window_counts_what_it_skipped(caplog):
    with caplog.at_level(logging.WARNING, logger=schedule_parse_watch.logger.name):
        slots, row = _in_turn(lambda: _iter_slot_datetimes(_SCHEDULE))

    assert slots == [datetime(2026, 10, 20, 13, 0)]  # как было: разобранное — то же
    assert row.get("slot_unparsed") == 3  # два окна дня «07.10.2026» и «--:--»
    shapes = [r.getMessage() for r in caplog.records]
    assert len(shapes) == 2 and "99.99.9999 99:99" in shapes[0], shapes  # один раз на форму значения


def test_renderer_counts_an_unparsed_day_date_and_time():
    payload = {"schedule": [{"fio": "Трубин Алексей Юрьевич", "regions": ["Ленина 5"], "schedule": _SCHEDULE}]}
    entities = {"time_from": "08:00"}

    before = renderer.format_doctor_schedule_for_patient(payload, entities)  # вне хода — счётчика нет, ответ тот же
    text, row = _in_turn(lambda: renderer.format_doctor_schedule_for_patient(payload, entities))

    assert text == before
    assert row.get("slot_unparsed") == 2  # дата «07.10.2026» и время «--:--»


def test_parsed_schedule_leaves_no_field():
    payload = {"schedule": [{"fio": "Трубин Алексей Юрьевич", "regions": ["Ленина 5"], "schedule": {"Ленина 5": [{"date": "2026-10-20", "slots": ["13:00"]}]}}]}

    _text, row = _in_turn(lambda: renderer.format_doctor_schedule_for_patient(payload, {"time_from": "08:00"}))
    _slots, row2 = _in_turn(lambda: _iter_slot_datetimes(payload["schedule"][0]["schedule"]))

    assert "slot_unparsed" not in row and "slot_unparsed" not in row2
