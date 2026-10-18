"""Ежедневное обновление срезов МИС — одно время, параллельно, проверка, повтор до успеха (решение владельца 18.10).

Было: у каждого справочника свой планировщик и своё время — прайс в 00:05 (до ночного обновления баз МИС), врачи в
07:45, цены врачей и priceUnits в 08:15, памятки и синонимы в 08:20 (уже после начала рабочего дня); сбой — до
следующего дня (а запрос пациента сам качал срез, которого нет).

Стало:
- «день данных» начинается в 07:30 по Самаре (`snapshots.data_day`): регламентные работы МИС идут с 01:00 до ~07:10;
- один цикл: чего нет на «день данных» (файла нет или он пуст) — качается; независимые справочники — параллельно,
  цены врачей — после врачей (собираются из их среза); пустой ответ МИС среза не пишет (`require_rows`);
- не скачалось — повтор каждые `RETRY_EVERY_S` (5 мин), пока все срезы не свежие; всё свежее — сон до следующих 07:30;
- итог каждого прохода — в лог: что скачано, что нет (сбой МИС виден сразу).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info, snapshots

log = logging.getLogger(__name__)

RETRY_EVERY_S = 300
_TASK: asyncio.Task | None = None


@dataclass(frozen=True)
class Source:
    """Справочник: имя, путь среза «на день данных», скачивание (асинхронное; само пишет файл), от кого зависит."""

    name: str
    path: Callable[[], Path]
    fetch: Callable[[], Awaitable[object]]
    after: str | None = None


async def _doctors() -> None:
    await api_nayka._refresh_once()  # врачи и заметки колл-центра; пустой ответ МИС среза не пишет


async def _doctor_prices() -> None:
    await asyncio.to_thread(api_price.update_doctor_prices, True)


async def _price_list() -> None:
    await asyncio.to_thread(api_price.update_price_by_region, api_price.BOT_PRICE_REGION_ID, True)


async def _price_units() -> None:
    await asyncio.to_thread(api_price.update_price_units, True)


async def _service_info() -> None:
    await api_service_info._refresh_service_info_once()  # памятки и добор синонимов


def sources() -> list[Source]:
    """Справочники бота. Пути — те же функции, по которым их читают загрузчики."""

    return [
        Source("doctors", lambda: api_nayka.DATA_DIR / f"doctors_{api_nayka.get_active_date_str()}.jsonl", _doctors),
        Source("doctor_prices", lambda: api_price.doctor_prices_path(), _doctor_prices, after="doctors"),
        Source("price_region", lambda: api_price.price_by_region_path(api_price.BOT_PRICE_REGION_ID), _price_list),
        Source("price_units", lambda: api_price.price_units_path(), _price_units),
        Source("service_info", lambda: api_service_info.service_info_path(), _service_info),
    ]


def missing(items: list[Source] | None = None) -> list[Source]:
    """Справочники, у которых нет непустого среза «на день данных»."""

    return [s for s in (items if items is not None else sources()) if not snapshots.has_rows(s.path())]


async def refresh_missing(items: list[Source] | None = None) -> dict[str, str]:
    """Один проход: скачать недостающие срезы. Возврат: справочник → `fresh` | `failed` (у скачиваемых в проходе)."""

    items = items if items is not None else sources()
    todo = missing(items)
    if not todo:
        return {}
    names = {s.name for s in todo}
    done: dict[str, asyncio.Event] = {s.name: asyncio.Event() for s in todo}
    result: dict[str, str] = {}

    async def run(source: Source) -> None:
        try:
            if source.after in done:
                await done[source.after].wait()
                if result.get(source.after) != "fresh":
                    result[source.name] = "failed"  # зависимость не скачалась — следующий проход
                    return
            try:
                await source.fetch()
            except Exception:
                log.warning("[DAILY REFRESH] %s: скачивание упало", source.name, exc_info=True)
            result[source.name] = "fresh" if snapshots.has_rows(source.path()) else "failed"
        finally:
            done[source.name].set()

    await asyncio.gather(*(run(s) for s in todo))
    failed = sorted(n for n in names if result.get(n) != "fresh")
    if failed:
        log.warning("[DAILY REFRESH] не скачаны: %s — повтор через %s с", ", ".join(failed), RETRY_EVERY_S)
    else:
        log.info("[DAILY REFRESH] скачаны: %s", ", ".join(sorted(names)))
    return result


def seconds_until_next_day(now_samara: datetime) -> float:
    """Сколько ждать до следующего начала «дня данных» (07:30 по Самаре)."""

    hour, minute = snapshots.DATA_DAY_STARTS_AT
    target = now_samara.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_samara >= target:
        target += timedelta(days=1)
    return max(1.0, (target - now_samara).total_seconds())


async def run(
    now: Callable[[], datetime] = api_price._now_samara,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    items: Callable[[], list[Source]] = sources,
) -> None:
    """Вечный цикл: недостающее — скачать; не вышло — повтор через `RETRY_EVERY_S`; всё свежее — до следующих 07:30."""

    while True:
        try:
            await refresh_missing(items())
            pause = min(RETRY_EVERY_S, seconds_until_next_day(now())) if missing(items()) else seconds_until_next_day(now())
        except Exception:
            log.warning("[DAILY REFRESH] проход упал", exc_info=True)
            pause = RETRY_EVERY_S
        await sleep(pause)


def ensure_started() -> bool:
    """Запустить цикл (идемпотентно). Вне работающего цикла событий — False (запустит следующий вызов)."""

    global _TASK
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    if _TASK is None or _TASK.done():
        _TASK = loop.create_task(run())
        log.info("[DAILY REFRESH] цикл обновления срезов запущен: день данных с %02d:%02d по Самаре, повтор %s с", *snapshots.DATA_DAY_STARTS_AT, RETRY_EVERY_S)
    return True
