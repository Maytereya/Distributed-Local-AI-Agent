"""Ежедневное обновление срезов МИС — один цикл в 07:30 по Самаре, повтор каждые 5 минут (решение владельца 18.10).

Было: четыре времени обновления (прайс 00:05 — до ночного обновления баз МИС; врачи 07:45; цены врачей и priceUnits
08:15; памятки 08:20 — после начала рабочего дня) и три «часа» датировки срезов (прайс — часы сервера в UTC, памятки —
Самара, врачи — Самара с границей 06:00; DATA-8): на проде срез прайса, скачанный в 00:05 по Самаре 17.10, лёг под
именем 16-го. Сбой скачивания — до следующего дня.

Стало: «день данных» (`snapshots.data_day`) — самарская дата с границей 07:30 у всех загрузчиков; один цикл
(`daily_refresh.run`): чего нет на день данных — качается сразу, независимые — параллельно, цены врачей — после врачей;
не вышло — повтор через 5 минут, пока все не свежие; всё свежее — сон до следующих 07:30.

Инварианты класса: все справочники называют «сегодня» одной датой; срез, которого нет на день данных, качается в фоне
(в том числе на старте сервиса и когда сегодняшний файл пуст); сбой одного справочника не мешает остальным и
повторяется, пока не скачается; ночью (до 07:30) цикл ничего не качает, если вчерашние срезы на месте.
Тесты — только во временной папке, свои источники (ловушка `CLAUDE.md`: очистка срезов разработчика).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info, daily_refresh, snapshots

SAMARA = ZoneInfo("Europe/Samara")


@pytest.mark.parametrize(
    ("hh", "mm", "expected"),
    [(0, 5, "20261017"), (6, 30, "20261017"), (7, 29, "20261017"), (7, 30, "20261018"), (23, 59, "20261018")],
)
def test_data_day_switches_at_0730_samara(hh, mm, expected):
    assert snapshots.data_day(datetime(2026, 10, 18, hh, mm, tzinfo=SAMARA)) == expected


@pytest.mark.parametrize("hh", [1, 6, 7, 8, 21])
def test_every_loader_names_today_the_same(hh, monkeypatch):
    """Прайс, памятки и врачи — одна дата (DATA-8). 06:00–07:30 врачи раньше уже переключались на «сегодня»."""

    now = datetime(2026, 10, 18, hh, 40, tzinfo=SAMARA)
    for module in (api_price, api_service_info, api_nayka):
        monkeypatch.setattr(module, "_now_samara", lambda now=now: now)

    names = {api_price.today_str(), api_service_info.today_str(), api_nayka.get_active_date_str()}

    assert names == {snapshots.data_day(now)}


# --------------------------------------------------------------------------- цикл на своих источниках во временной папке


class _Mis:
    """Поддельная МИС: `fail` — какие справочники падают (пока не убраны), `empty` — отвечают пустым списком."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.fail: set[str] = set()
        self.empty: set[str] = set()
        self.calls: list[str] = []
        self.started: list[str] = []
        self.gate = asyncio.Event()

    def path(self, name: str) -> Path:
        return self.folder / f"{name}_20261018.jsonl"

    def source(self, name: str, after: str | None = None, *, wait: bool = False) -> daily_refresh.Source:
        async def fetch() -> None:
            self.started.append(name)
            if wait:
                await self.gate.wait()
            self.calls.append(name)
            if name in self.fail:
                raise ConnectionError(f"{name}: МИС недоступна")
            rows = [] if name in self.empty else [{"id": 1}]
            snapshots.write_rows(self.path(name), snapshots.require_rows(rows, name))

        return daily_refresh.Source(name, lambda: self.path(name), fetch, after=after)


def _items(mis: _Mis) -> list[daily_refresh.Source]:
    return [mis.source("doctors"), mis.source("doctor_prices", after="doctors"), mis.source("price_region"), mis.source("service_info")]


def test_missing_snapshots_are_fetched_and_fresh_ones_are_left_alone(tmp_path):
    mis = _Mis(tmp_path)
    snapshots.write_rows(mis.path("service_info"), [{"id": 0}])  # уже свежий
    mis.path("price_region").write_text("", encoding="utf-8")  # сегодняшний пуст — тоже «нет»

    result = asyncio.run(daily_refresh.refresh_missing(_items(mis)))

    assert result == {"doctors": "fresh", "doctor_prices": "fresh", "price_region": "fresh"}
    assert "service_info" not in mis.calls
    assert mis.calls.index("doctors") < mis.calls.index("doctor_prices")  # цены врачей — из среза врачей


def test_independent_sources_run_in_parallel(tmp_path):
    mis = _Mis(tmp_path)
    items = [mis.source("price_region", wait=True), mis.source("service_info", wait=True)]

    async def scenario():
        task = asyncio.create_task(daily_refresh.refresh_missing(items))
        for _ in range(5):
            await asyncio.sleep(0)
        both_started = sorted(mis.started)
        mis.gate.set()
        return both_started, await task

    started, result = asyncio.run(scenario())
    assert started == ["price_region", "service_info"]  # второй начал, не дожидаясь первого
    assert result == {"price_region": "fresh", "service_info": "fresh"}


def test_one_failure_does_not_stop_the_others_and_dependents_wait(tmp_path, caplog):
    mis = _Mis(tmp_path)
    mis.fail = {"doctors"}
    mis.empty = {"price_region"}  # пустой ответ МИС среза не пишет

    with caplog.at_level(logging.WARNING, logger=daily_refresh.log.name):
        result = asyncio.run(daily_refresh.refresh_missing(_items(mis)))

    assert result == {"doctors": "failed", "doctor_prices": "failed", "price_region": "failed", "service_info": "fresh"}
    assert "doctor_prices" not in mis.calls  # зависимость не скачалась — не собираем из вчерашних врачей
    assert not mis.path("price_region").exists()
    assert any("не скачаны: doctor_prices, doctors, price_region" in r.getMessage() for r in caplog.records)


def test_loop_retries_every_five_minutes_until_everything_is_fresh_then_waits_for_0730(tmp_path):
    mis = _Mis(tmp_path)
    mis.fail = {"price_region"}
    pauses: list[float] = []
    now = datetime(2026, 10, 18, 7, 30, 5, tzinfo=SAMARA)

    async def sleep(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 2:
            mis.fail.clear()  # МИС починили
        if len(pauses) == 4:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daily_refresh.run(now=lambda: now, sleep=sleep, items=lambda: _items(mis)))

    assert pauses[:2] == [daily_refresh.RETRY_EVERY_S, daily_refresh.RETRY_EVERY_S]
    assert pauses[2] == pauses[3] == pytest.approx(24 * 3600 - 5)  # всё свежее — до следующих 07:30
    assert mis.calls.count("price_region") == 3 and mis.calls.count("doctors") == 1


def test_at_night_yesterdays_snapshots_are_enough(tmp_path):
    """До 07:30 «день данных» — вчера: вчерашние срезы на месте — цикл ничего не качает и спит до 07:30."""

    mis = _Mis(tmp_path)
    for name in ("doctors", "doctor_prices", "price_region", "service_info"):
        snapshots.write_rows(mis.path(name), [{"id": 1}])
    pauses: list[float] = []
    night = datetime(2026, 10, 19, 3, 0, tzinfo=SAMARA)

    async def sleep(seconds: float) -> None:
        pauses.append(seconds)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daily_refresh.run(now=lambda: night, sleep=sleep, items=lambda: _items(mis)))

    assert mis.calls == []
    assert pauses == [pytest.approx(4.5 * 3600)]


def test_every_old_scheduler_name_starts_the_one_loop(monkeypatch):
    started: list[str] = []
    monkeypatch.setattr(daily_refresh, "_TASK", None)

    async def idle(**_kwargs):
        started.append("run")
        await asyncio.sleep(3600)

    monkeypatch.setattr(daily_refresh, "run", idle)

    async def scenario():
        assert api_nayka.ensure_daily_refresh_started()
        assert api_price.ensure_daily_price_refresh_started()
        assert api_service_info.ensure_daily_service_info_refresh_started()
        await asyncio.sleep(0)
        task = daily_refresh._TASK
        task.cancel()
        return task

    asyncio.run(scenario())
    assert started == ["run"]  # один цикл, не три
    assert daily_refresh.ensure_started() is False  # вне цикла событий — не запускается


def test_real_sources_cover_every_snapshot_the_bot_reads():
    assert [s.name for s in daily_refresh.sources()] == ["doctors", "doctor_prices", "price_region", "price_units", "service_info"]
    assert {s.name: s.after for s in daily_refresh.sources()}["doctor_prices"] == "doctors"
