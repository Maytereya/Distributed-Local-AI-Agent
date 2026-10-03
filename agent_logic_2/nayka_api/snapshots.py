"""Дневные срезы справочников МИС на диске: атомарная запись и выбор непустого среза.

Срез «на сегодня» (прайс региона, priceUnits, serviceInfoAll) скачивается из МИС раз в
день. Модуль закрывает три дыры, через которые бот получал пустые данные или зависал:

- пустой ответ МИС (200 и []) ложился файлом 0 байт, и загрузчик весь день отдавал
  боту пустой прайс или пустые памятки — файл ведь «есть»: так было с прайсом
  региона 1 (02.06) и с doctor_prices (30.09, BUG-2026-09-25-NO-DOCTORS-FOUND-FALSE);
- сбой МИС: пока она не отвечает, каждый запрос пациента заново ждал тайм-аут
  скачивания (60 с с повторами) вместо того, чтобы взять вчерашний срез;
- запись поверх файла: запрос, пришедший в момент записи, читал срез наполовину.

Класс дефектов `empty_snapshot_poisons_day`: пустой или недоступный ответ МИС не
оставляет бота без данных, пока на диске есть последний непустой срез.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

# После неудачной попытки скачать срез «на сегодня» следующая — не раньше чем через
# столько секунд; в промежутке запросы сразу получают последний непустой срез.
RETRY_AFTER_S = 600
_failed_at: dict[str, float] = {}


class EmptySnapshotError(RuntimeError):
    """МИС ответила пустым списком — такой срез не записываем."""


def require_rows(rows: Any, what: str) -> list[Any]:
    """Ответ МИС как список строк; пустой — ошибка, а не срез на весь день.

    :param rows: ответ метода МИС
    :param what: что скачивали — для сообщения
    :return: непустой список
    """

    if not isinstance(rows, list) or not rows:
        raise EmptySnapshotError(f"{what}: МИС вернула пустой ответ, срез не записан")
    return rows


def has_rows(path: Path) -> bool:
    """Файл есть и не пуст."""

    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Пишет JSONL атомарно: читатель видит либо прежний файл, либо новый целиком."""

    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # недописанный временный файл — не мусор на диске
        raise


def latest_nonempty(path: Path) -> Path | None:
    """Самый свежий непустой срез того же справочника: сам `path` или более ранний.

    :param path: срез «на сегодня» (`<имя>_<ГГГГММДД>.jsonl`)
    :return: путь к срезу или None, если непустых нет
    """

    prefix = path.stem.rsplit("_", 1)[0]
    for candidate in sorted(path.parent.glob(f"{prefix}_*{path.suffix}"), reverse=True):
        if candidate.name <= path.name and has_rows(candidate):
            return candidate
    return None


def retry_allowed(key: Path | str) -> bool:
    """Можно ли снова идти в МИС за срезом: прошлый сбой был давно или не был."""

    failed = _failed_at.get(str(key))
    return failed is None or time.monotonic() - failed >= RETRY_AFTER_S


def mark_failed(key: Path | str) -> None:
    """Запомнить сбой скачивания — следующие запросы не ждут МИС `RETRY_AFTER_S` секунд."""

    _failed_at[str(key)] = time.monotonic()


def load_daily(
    path: Path,
    download: Callable[[], Any],
    read: Callable[[Path], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Строки дневного среза.

    Сегодняшний непустой — он. Нет или пуст — скачать (после неудачи — не чаще раза в
    `RETRY_AFTER_S`); не вышло — последний непустой срез. Срезов нет совсем — пусто.

    :param path: срез «на сегодня»
    :param download: скачивает и записывает срез «на сегодня»; пустой ответ — исключение
    :param read: читает файл среза
    :return: строки среза
    """

    if not has_rows(path) and retry_allowed(path):
        try:
            download()
        except Exception as exc:  # МИС недоступна или пуста — вчерашний срез лучше пустого
            log.warning("срез %s не скачан (%s) — беру последний непустой", path.name, exc)
        if not has_rows(path):
            mark_failed(path)
    source = latest_nonempty(path)
    return read(source) if source is not None else []


async def run_daily(hour: int, minute: int, job: Callable[[], Awaitable[Any]], now: Callable[[], datetime]) -> None:
    """Вечный цикл: `job` каждый день в `hour:minute` по часам `now` (самарское время).

    :param hour: час запуска
    :param minute: минута запуска
    :param job: корутина-функция обновления; свои ошибки она гасит сама
    :param now: текущее время в нужном поясе
    """

    while True:
        current = now()
        target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if current >= target:
            target += timedelta(days=1)
        try:
            await asyncio.sleep(max(1.0, (target - current).total_seconds()))
        except Exception:
            pass  # прерванный сон — пересчитать цель на следующем круге
        await job()

