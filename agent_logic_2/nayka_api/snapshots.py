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

Контракт среза (ход (д), решение владельца 08.10): `load_snapshot` → `Snapshot{rows, date, status}`,
статус `fresh` (срез «на сегодня»), `stale` (последний непустой, старше; `age_days` — на сколько дней)
или `unavailable` (непустых срезов нет). Сегодня статус только наблюдается — предупреждение в лог и
поле телеметрии хода через `set_observer`; ответы пациенту не меняются (DATA-6, DATA-9: вчерашний срез
и пустота выглядели как свежие данные).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

FRESH = "fresh"
STALE = "stale"
UNAVAILABLE = "unavailable"

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


def snapshot_date(path: Path) -> date | None:
    """Дата среза из имени файла (`<имя>_<ГГГГММДД>.jsonl`); имя без даты — None."""

    try:
        return datetime.strptime(path.stem.rsplit("_", 1)[-1], "%Y%m%d").date()
    except ValueError:
        return None


@dataclass(frozen=True)
class Snapshot:
    """Строки среза с датой и статусом.

    :param name: справочник — имя файла без даты (`price_region_3`, `price_units`, `service_info`, `doctors`,
        `doctor_prices`)
    :param rows: строки среза; у `unavailable` — пусто; у наблюдения без чтения строк (словарь синонимов строит
        свой индекс сам) — тоже пусто
    :param date: дата прочитанного среза или None
    :param status: `fresh` | `stale` | `unavailable`
    :param age_days: на сколько дней срез старше «сегодня» загрузчика; у `unavailable` — None
    """

    name: str
    rows: list[dict[str, Any]]
    date: date | None
    status: str
    age_days: int | None


_observer: Callable[[Snapshot], None] | None = None
_warned: set[tuple[str, str, date | None]] = set()


def set_observer(observer: Callable[[Snapshot], None] | None) -> None:
    """Кому сообщать о каждом прочитанном срезе (телеметрия хода). Слой МИС не знает о роутере."""

    global _observer
    _observer = observer


def _observe(snapshot: Snapshot) -> None:
    key = (snapshot.name, snapshot.status, snapshot.date)
    if snapshot.status != FRESH and key not in _warned:
        _warned.add(key)  # один раз на срез и статус: загрузчики зовут по нескольку раз за ход
        if snapshot.status == STALE:
            log.warning("срез %s устарел: данные на %s (%s дн.)", snapshot.name, snapshot.date, snapshot.age_days)
        else:
            log.warning("срез %s недоступен: непустых срезов нет", snapshot.name)
    if _observer is not None:
        try:
            _observer(snapshot)
        except Exception:  # наблюдатель не ломает чтение среза
            log.warning("наблюдатель срезов упал", exc_info=True)


def report(path: Path, source: Path | None, rows: list[dict[str, Any]] | None = None) -> Snapshot:
    """Срез, который загрузчик выбрал сам, — со статусом, и сообщить о нём наблюдателю.

    Для загрузчиков со своей логикой выбора файла (врачи, цены врачей, словарь синонимов): `path` — срез «на
    сегодня» загрузчика, `source` — прочитанный (None — непустых нет). Срез не старше «сегодня» — `fresh`
    (врачи с 00:00 до 06:00 по Самаре читают сегодняшний файл при активной вчерашней дате).

    :param path: срез «на сегодня»
    :param source: прочитанный срез или None
    :param rows: прочитанные строки (None — наблюдение без чтения)
    :return: срез
    """

    name = path.stem.rsplit("_", 1)[0]
    if source is None:
        snapshot = Snapshot(name=name, rows=[], date=None, status=UNAVAILABLE, age_days=None)
    else:
        today, taken = snapshot_date(path), snapshot_date(source)
        age = (today - taken).days if today is not None and taken is not None else None
        fresh = source == path or (age is not None and age <= 0)
        snapshot = Snapshot(
            name=name,
            rows=rows if rows is not None else [],  # тот же список, что прочитал загрузчик
            date=taken,
            status=FRESH if fresh else STALE,
            age_days=0 if fresh and age is not None else age,
        )
    _observe(snapshot)
    return snapshot


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
    """Строки дневного среза — `load_snapshot(...).rows` (прежний контракт загрузчиков)."""

    return load_snapshot(path, download, read).rows


def load_snapshot(
    path: Path,
    download: Callable[[], Any],
    read: Callable[[Path], list[dict[str, Any]]],
) -> Snapshot:
    """Дневной срез с датой и статусом.

    Сегодняшний непустой — он (`fresh`). Нет или пуст — скачать (после неудачи — не чаще раза в
    `RETRY_AFTER_S`); не вышло — последний непустой срез (`stale`). Срезов нет совсем — пусто
    (`unavailable`). Давность считается от даты в имени `path` — «сегодня» самого загрузчика.

    :param path: срез «на сегодня»
    :param download: скачивает и записывает срез «на сегодня»; пустой ответ — исключение
    :param read: читает файл среза
    :return: срез
    """

    if not has_rows(path) and retry_allowed(path):
        try:
            download()
        except Exception as exc:  # МИС недоступна или пуста — вчерашний срез лучше пустого
            log.warning("срез %s не скачан (%s) — беру последний непустой", path.name, exc)
        if not has_rows(path):
            mark_failed(path)
    source = latest_nonempty(path)
    return report(path, source, read(source) if source is not None else [])


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

