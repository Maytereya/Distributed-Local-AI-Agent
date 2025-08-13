# api_price.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent_logic_2 import config as c

# -----------------------------------------------------------------------------
# Конфиг
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "apidata"

DOCTORS_DIR = DATA_DIR                      # содержит doctors_*.jsonl
PRICES_DIR = DATA_DIR / "doctor_prices"     # кэш по врачам
PRICEALL_DIR = DATA_DIR / "prices"          # кэш priceAll

PRICES_DIR.mkdir(parents=True, exist_ok=True)
PRICEALL_DIR.mkdir(parents=True, exist_ok=True)

DATE_FMT = "%Y%m%d"
CACHE_TTL_DAYS = 2

BASE_URL = c.nayka_base_url.rstrip("/")
AUTH = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)
VERIFY_TLS = getattr(c, "nayka_verify_tls", False)  # по умолчанию как было — False
HTTP_TIMEOUT = getattr(c, "nayka_timeout", 10)

ENDPOINT_PRICE_ALL = f"{BASE_URL}/priceAll"
ENDPOINT_DOCTOR_PRICES_BY_REGION = f"{BASE_URL}/doctorServicePricesByRegion"

# -----------------------------------------------------------------------------
# Логирование
# -----------------------------------------------------------------------------
log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# HTTP-сессия с ретраями
# -----------------------------------------------------------------------------
def _build_session() -> requests.Session:
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

# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def today_str() -> str:
    return datetime.now().strftime(DATE_FMT)

def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def jsonl_read(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                log.error("Не удалось распарсить строку: %s... Ошибка: %s", line[:100], e)
    return out

def latest_file(dir_: Path, pattern: str) -> Path:
    files = sorted(dir_.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Файл {pattern} не найден в {dir_}")
    return files[-1]

def dated_filename(dir_: Path, prefix: str, date: Optional[str] = None) -> Path:
    d = date or today_str()
    return dir_ / f"{prefix}_{d}.jsonl"

def cleanup_old(dir_: Path, prefix: str, days_to_keep: int = CACHE_TTL_DAYS) -> None:
    cutoff = datetime.now() - timedelta(days=days_to_keep)
    for f in dir_.glob(f"{prefix}_*.jsonl"):
        date_str = f.stem.rsplit("_", 1)[-1]
        try:
            dt = datetime.strptime(date_str, DATE_FMT)
        except Exception:
            continue
        if dt <= cutoff:
            log.info("🗑️ Удаляем устаревший кэш: %s", f)
            try:
                f.unlink()
            except Exception as e:
                log.warning("Не удалось удалить %s: %s", f, e)

# -----------------------------------------------------------------------------
# Файлы кэша прайса
# -----------------------------------------------------------------------------
def priceall_path(date: Optional[str] = None) -> Path:
    return dated_filename(PRICEALL_DIR, "price_all", date)

def doctor_prices_path(date: Optional[str] = None) -> Path:
    return dated_filename(PRICES_DIR, "doctor_prices", date)

# -----------------------------------------------------------------------------
# API - доступ
# -----------------------------------------------------------------------------
def fetch_price_all() -> List[Dict[str, Any]]:
    r = SESSION.get(ENDPOINT_PRICE_ALL, timeout=HTTP_TIMEOUT, verify=VERIFY_TLS)
    r.raise_for_status()
    return r.json()

def fetch_doctor_prices(doctor_id: Any, region_id: Any) -> List[Dict[str, Any]]:
    params = {"doctorId": doctor_id, "regionId": region_id}
    try:
        r = SESSION.get(
            ENDPOINT_DOCTOR_PRICES_BY_REGION,
            params=params,
            timeout=HTTP_TIMEOUT,
            verify=VERIFY_TLS,
        )
        if r.status_code == 200:
            return r.json()
        log.error("[HTTP %s] doctorId=%s regionId=%s data=%s",
                  r.status_code, doctor_id, region_id, r.text[:200])
        return []
    except Exception as e:
        log.exception("[EXCEPTION] doctorId=%s regionId=%s error=%s", doctor_id, region_id, e)
        return []

# -----------------------------------------------------------------------------
# Функционал: priceAll
# -----------------------------------------------------------------------------
def update_price_all(force: bool = False) -> Path:
    """Скачивает priceAll в файл за сегодня. Возвращает путь к файлу."""
    fn = priceall_path()
    if fn.exists() and not force:
        log.info("✅ priceAll на сегодня уже скачан: %s", fn)
        return fn

    log.info("⏬ Скачиваем priceAll...")
    data = fetch_price_all()
    jsonl_write(fn, data)
    log.info("✅ priceAll обновлён (%s строк): %s", len(data), fn)
    cleanup_old(PRICEALL_DIR, "price_all", CACHE_TTL_DAYS)
    return fn

def load_price_all() -> List[Dict[str, Any]]:
    fn = priceall_path()
    if not fn.exists():
        update_price_all()
    return jsonl_read(fn)

# -----------------------------------------------------------------------------
# Функционал: doctor_prices
# -----------------------------------------------------------------------------
def _load_doctors() -> List[Dict[str, Any]]:
    doctors_file = latest_file(DOCTORS_DIR, "doctors_*.jsonl")
    return jsonl_read(doctors_file)

def update_doctor_prices(force: bool = False) -> Path:
    """Собирает цены по врачам и регионам на сегодня. Возвращает путь к файлу."""
    fn = doctor_prices_path()
    if fn.exists() and not force:
        log.info("✅ doctor_prices на сегодня уже собран: %s", fn)
        return fn

    log.info("⏬ Собираем doctor_prices...")
    doctors = _load_doctors()
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for doc in doctors:
        doctor_id = doc.get("id")
        region_ids = doc.get("region_ids") or doc.get("regionIds") or []
        if not doctor_id or not region_ids:
            skipped += 1
            log.warning("[WARN] Пропущен врач без регионов: %s", doc.get("fio", doctor_id))
            continue

        for region_id in region_ids:
            if not region_id:
                continue
            services = fetch_doctor_prices(doctor_id, region_id)
            for s in services:
                rows.append({
                    "doctorId": doctor_id,
                    "regionId": region_id,
                    "fio": doc.get("fio"),
                    "serviceName": s.get("serviceName"),
                    "serviceId": s.get("serviceId"),
                    "cost": s.get("cost"),
                    "priceUnitId": s.get("priceUnitId"),
                    "serviceHomecode": s.get("serviceHomecode"),
                    "deadline": s.get("deadline"),
                })

    jsonl_write(fn, rows)
    log.info("✅ doctor_prices обновлён (%s цен, пропущено врачей: %s): %s",
             len(rows), skipped, fn)
    cleanup_old(PRICES_DIR, "doctor_prices", CACHE_TTL_DAYS)
    return fn

def load_doctor_prices() -> List[Dict[str, Any]]:
    fn = doctor_prices_path()
    if not fn.exists():
        update_doctor_prices()
    return jsonl_read(fn)

# -----------------------------------------------------------------------------
# CLI!
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    update_doctor_prices()