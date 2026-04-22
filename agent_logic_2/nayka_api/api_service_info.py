"""Кэш `serviceInfoAll` для правил подготовки и описаний услуг.

Модуль скачивает и хранит в JSONL дневной срез Nayka `site/serviceInfoAll`.
Источник используется как API-first слой для подготовки к анализам и
исследованиям, а knowledge fallback остается запасным вариантом.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent_logic_2 import config as c
from agent_logic_2.nayka_api.cache_paths import resolve_cache_data_dir

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


DATE_FMT = "%Y%m%d"
CACHE_TTL_DAYS = 2
DATA_DIR = resolve_cache_data_dir()
SERVICE_INFO_DIR = DATA_DIR / "service_info"
SERVICE_INFO_DIR.mkdir(parents=True, exist_ok=True)

def _service_info_base_url() -> str:
    """
    Возвращает базовый URL для `serviceInfoAll` без дублирования сегмента `/site`.

    :return: базовый URL API без завершающего `/site`
    """

    explicit = str(getattr(c, "nayka_base_url_no_site", "") or "").strip()
    if explicit:
        return explicit.rstrip("/")

    base = str(getattr(c, "nayka_base_url", "") or "").strip().rstrip("/")
    if base.endswith("/site"):
        return base[: -len("/site")]
    return base


BASE_URL = _service_info_base_url()
ENDPOINT_SERVICE_INFO_ALL = f"{BASE_URL}/site/serviceInfoAll"
AUTH = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)
VERIFY_TLS = getattr(c, "nayka_verify_tls", False)
HTTP_TIMEOUT = getattr(c, "nayka_timeout", 60)

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)


def _build_session() -> requests.Session:
    """Создает requests-сессию с retry для GET-запросов к Nayka API."""

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.auth = AUTH
    return session


SESSION = _build_session()
_SERVICE_INFO_REFRESH_TASK: asyncio.Task | None = None


def _now_samara() -> datetime:
    """Возвращает текущее время по Самаре с fallback на UTC+4."""

    try:
        if ZoneInfo is not None:
            return datetime.now(tz=ZoneInfo("Europe/Samara"))
    except Exception:
        pass
    return datetime.utcnow() + timedelta(hours=4)


def today_str() -> str:
    """Возвращает дату в формате YYYYMMDD для имен файлов кэша."""

    return _now_samara().strftime(DATE_FMT)


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Сохраняет список словарей в JSONL-файл."""

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    """Читает JSONL-файл и возвращает список словарей."""

    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception as exc:
                log.warning("Не удалось распарсить строку service_info: %s", exc)
                continue
            if isinstance(item, dict):
                out.append(item)
    return out


def service_info_path(date: str | None = None) -> Path:
    """
    Возвращает путь к файлу service_info за указанную дату.

    :param date: дата в формате YYYYMMDD или None для сегодняшней даты
    :return: путь к JSONL-файлу
    """

    return SERVICE_INFO_DIR / f"service_info_{date or today_str()}.jsonl"


def _cleanup_old() -> None:
    """Удаляет устаревшие service_info файлы старше TTL."""

    cutoff = _now_samara().replace(tzinfo=None) - timedelta(days=CACHE_TTL_DAYS)
    for item in SERVICE_INFO_DIR.glob("service_info_*.jsonl"):
        date_str = item.stem.rsplit("_", 1)[-1]
        try:
            dt = datetime.strptime(date_str, DATE_FMT)
        except Exception:
            continue
        if dt <= cutoff:
            try:
                item.unlink()
            except Exception as exc:
                log.warning("Не удалось удалить старый service_info cache %s: %s", item, exc)


def _has_useful_text(row: dict[str, Any]) -> bool:
    """Проверяет, что в записи есть полезный текст для дальнейшего поиска."""

    for field in ("description", "indication", "preparation"):
        value = str(row.get(field) or "").strip()
        if value:
            return True
    return False


def fetch_service_info_all() -> list[dict[str, Any]]:
    """
    Загружает полный ответ `site/serviceInfoAll` из Nayka API.

    :return: список записей API
    """

    response = SESSION.get(ENDPOINT_SERVICE_INFO_ALL, timeout=HTTP_TIMEOUT, verify=VERIFY_TLS)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def update_service_info(force: bool = False) -> Path:
    """
    Обновляет кэш `serviceInfoAll` за текущий день.

    :param force: при True принудительно пересобирает файл за сегодня
    :return: путь к актуальному JSONL-файлу
    """

    target = service_info_path()
    if target.exists() and not force:
        log.info("✅ serviceInfoAll на сегодня уже собран: %s", target)
        return target

    log.info("⏬ Скачиваем serviceInfoAll...")
    rows = fetch_service_info_all()
    filtered = [row for row in rows if isinstance(row, dict) and _has_useful_text(row)]
    jsonl_write(target, filtered)
    log.info("✅ serviceInfoAll обновлён (%s строк): %s", len(filtered), target)
    _cleanup_old()
    return target


def load_service_info() -> list[dict[str, Any]]:
    """
    Загружает актуальный serviceInfoAll из дневного кэша.

    :return: список записей serviceInfoAll
    """

    target = service_info_path()
    if not target.exists():
        update_service_info()
    return jsonl_read(target)


def _next_service_info_refresh_dt() -> datetime:
    """Возвращает время следующего фонового обновления serviceInfoAll."""

    now = _now_samara()
    target = now.replace(hour=8, minute=20, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return target


async def _refresh_service_info_once() -> None:
    """Один раз обновляет serviceInfoAll в фоне."""

    try:
        await asyncio.to_thread(update_service_info, True)
        log.info("✅ [DAILY REFRESH] serviceInfoAll обновлён")
    except Exception as exc:
        log.warning("⚠️ [DAILY REFRESH] serviceInfoAll refresh failed: %s", exc)


async def _service_info_refresh_loop() -> None:
    """Поддерживает ежедневное обновление serviceInfoAll по расписанию."""

    while True:
        target = _next_service_info_refresh_dt()
        now = _now_samara()
        wait_sec = max(1.0, (target - now).total_seconds())
        try:
            await asyncio.sleep(wait_sec)
        except Exception:
            pass
        await _refresh_service_info_once()


def ensure_daily_service_info_refresh_started() -> bool:
    """
    Запускает фоновый refresh serviceInfoAll в 08:20 по Самаре.

    :return: True, если планировщик запущен или уже активен
    """

    global _SERVICE_INFO_REFRESH_TASK
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    if _SERVICE_INFO_REFRESH_TASK is None or _SERVICE_INFO_REFRESH_TASK.done():
        _SERVICE_INFO_REFRESH_TASK = loop.create_task(_service_info_refresh_loop())
        log.info("▶️ [DAILY REFRESH] Планировщик serviceInfoAll запущен")
        try:
            if not service_info_path().exists():
                loop.create_task(_refresh_service_info_once())
        except Exception:
            pass
    return True


if __name__ == "__main__":
    update_service_info()
