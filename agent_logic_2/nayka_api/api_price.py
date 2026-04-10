# api_price.py
from __future__ import annotations

import asyncio
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
from agent_logic_2.nayka_api import api_nayka
from agent_logic_2.nayka_api.cache_paths import resolve_cache_data_dir

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

# -----------------------------------------------------------------------------
# Конфиг
# -----------------------------------------------------------------------------
DATA_DIR = resolve_cache_data_dir()

DOCTORS_DIR = DATA_DIR                      # содержит doctors_*.jsonl
PRICES_DIR = DATA_DIR / "doctor_prices"     # кэш по врачам
PRICEALL_DIR = DATA_DIR / "prices"          # кэш priceAll
PRICE_BY_REGION_DIR = DATA_DIR / "price_by_region"  # кэш priceByRegion/{regionId}
PRICE_UNITS_DIR = DATA_DIR / "price_units"  # кэш справочника priceUnits

PRICES_DIR.mkdir(parents=True, exist_ok=True)
PRICEALL_DIR.mkdir(parents=True, exist_ok=True)
PRICE_BY_REGION_DIR.mkdir(parents=True, exist_ok=True)
PRICE_UNITS_DIR.mkdir(parents=True, exist_ok=True)

DATE_FMT = "%Y%m%d"
CACHE_TTL_DAYS = 2

BASE_URL = c.nayka_base_url.rstrip("/")
AUTH = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)
VERIFY_TLS = getattr(c, "nayka_verify_tls", False)  # по умолчанию как было — False
HTTP_TIMEOUT = getattr(c, "nayka_timeout", 60)

ENDPOINT_PRICE_ALL = f"{BASE_URL}/priceAll"
ENDPOINT_PRICE_BY_REGION = f"{BASE_URL}/priceByRegion"
ENDPOINT_DOCTOR_PRICES_BY_REGION = f"{BASE_URL}/doctorServicePricesByRegion"
ENDPOINT_PRICE_UNITS = f"{BASE_URL}/priceUnits"

CARE_SETTING_POLYCLINIC_ROOT_ID = 146
CARE_SETTING_DAY_HOSPITAL_ROOT_ID = 311
CARE_SETTING_INPATIENT_ROOT_ID = 312

CARE_SETTING_LABEL_BY_ROOT_ID = {
    CARE_SETTING_POLYCLINIC_ROOT_ID: "поликлиника",
    CARE_SETTING_DAY_HOSPITAL_ROOT_ID: "дневной стационар",
    CARE_SETTING_INPATIENT_ROOT_ID: "круглосуточный стационар",
}

CARE_SETTING_ADDRESS_BY_ROOT_ID = {
    CARE_SETTING_POLYCLINIC_ROOT_ID: "г. Самара, пр. Ленина, 5",
    CARE_SETTING_DAY_HOSPITAL_ROOT_ID: "г. Самара, пр. Ленина, 5",
    CARE_SETTING_INPATIENT_ROOT_ID: "г. Самара, ул. Ново-Садовая, 106, кор. 82",
}

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
_PRICE_REFRESH_TASK: asyncio.Task | None = None

# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def today_str() -> str:
    return datetime.now().strftime(DATE_FMT)


def _now_samara() -> datetime:
    try:
        if ZoneInfo is not None:
            return datetime.now(tz=ZoneInfo("Europe/Samara"))
    except Exception:
        pass
    # Фолбэк: Самара = UTC+4
    return datetime.utcnow() + timedelta(hours=4)

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


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def _is_samara_region_row(row: dict[str, Any]) -> bool:
    city = _norm_text(row.get("city"))
    name = _norm_text(row.get("name"))
    addr = _norm_text(row.get("addressForSite"))
    if city == "самара":
        return True
    return "самара" in name or "самара" in addr


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


def price_by_region_path(region_id: Any, date: Optional[str] = None) -> Path:
    region_key = str(region_id).strip()
    if not region_key:
        raise ValueError("region_id is required")
    return dated_filename(PRICE_BY_REGION_DIR, f"price_region_{region_key}", date)


def price_units_path(date: Optional[str] = None) -> Path:
    """
    Возвращает путь к файлу кэша справочника priceUnits за указанную дату.

    :param date: дата в формате YYYYMMDD; если не передана, берется сегодняшняя
    :return: путь до jsonl-файла справочника priceUnits
    """
    return dated_filename(PRICE_UNITS_DIR, "price_units", date)


def _active_doctors_path() -> Path:
    return DOCTORS_DIR / f"doctors_{api_nayka.get_active_date_str()}.jsonl"

# -----------------------------------------------------------------------------
# API - доступ
# -----------------------------------------------------------------------------
def fetch_price_all() -> List[Dict[str, Any]]:
    r = SESSION.get(ENDPOINT_PRICE_ALL, timeout=HTTP_TIMEOUT, verify=VERIFY_TLS)
    r.raise_for_status()
    return r.json()


def fetch_price_by_region(region_id: Any) -> List[Dict[str, Any]]:
    region_key = str(region_id).strip()
    if not region_key:
        raise ValueError("region_id is required")
    r = SESSION.get(f"{ENDPOINT_PRICE_BY_REGION}/{region_key}", timeout=HTTP_TIMEOUT, verify=VERIFY_TLS)
    r.raise_for_status()
    return r.json()


def fetch_price_units() -> List[Dict[str, Any]]:
    """
    Загружает справочник priceUnits из API Науки.

    :return: список словарей со справочными разделами прайса
    """
    r = SESSION.get(ENDPOINT_PRICE_UNITS, timeout=HTTP_TIMEOUT, verify=VERIFY_TLS)
    r.raise_for_status()
    return r.json()

def fetch_doctor_prices(
    doctor_id: Any,
    region_id: Any,
    company_unit_id: Any | None = None,
    is_favorite: bool = False,
) -> List[Dict[str, Any]]:
    params = {"doctorId": doctor_id, "regionId": region_id, "isFavorite": bool(is_favorite)}
    if company_unit_id is not None:
        params["companyUnitId"] = company_unit_id
    try:
        r = SESSION.get(
            ENDPOINT_DOCTOR_PRICES_BY_REGION,
            params=params,
            timeout=HTTP_TIMEOUT,
            verify=VERIFY_TLS,
        )
        if r.status_code == 200:
            return r.json()
        log.error(
            "[HTTP %s] doctorId=%s regionId=%s companyUnitId=%s data=%s",
            r.status_code,
            doctor_id,
            region_id,
            company_unit_id,
            r.text[:200],
        )
        return []
    except Exception as e:
        log.exception(
            "[EXCEPTION] doctorId=%s regionId=%s companyUnitId=%s error=%s",
            doctor_id,
            region_id,
            company_unit_id,
            e,
        )
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


def update_price_by_region(region_id: Any, force: bool = False) -> Path:
    """Скачивает priceByRegion/{region_id} в файл за сегодня. Возвращает путь к файлу."""
    fn = price_by_region_path(region_id)
    if fn.exists() and not force:
        log.info("✅ priceByRegion(%s) на сегодня уже скачан: %s", region_id, fn)
        return fn

    log.info("⏬ Скачиваем priceByRegion(%s)...", region_id)
    data = fetch_price_by_region(region_id)
    jsonl_write(fn, data)
    log.info("✅ priceByRegion(%s) обновлён (%s строк): %s", region_id, len(data), fn)
    cleanup_old(PRICE_BY_REGION_DIR, f"price_region_{str(region_id).strip()}", CACHE_TTL_DAYS)
    return fn


def load_price_by_region(region_id: Any) -> List[Dict[str, Any]]:
    fn = price_by_region_path(region_id)
    if not fn.exists():
        update_price_by_region(region_id)
    return jsonl_read(fn)


def update_price_units(force: bool = False) -> Path:
    """
    Скачивает справочник priceUnits в файл за сегодня.

    :param force: если True, перезаписывает кэш даже при наличии файла за сегодня
    :return: путь к актуальному jsonl-файлу priceUnits
    """
    fn = price_units_path()
    if fn.exists() and not force:
        log.info("✅ priceUnits на сегодня уже скачан: %s", fn)
        return fn

    log.info("⏬ Скачиваем priceUnits...")
    data = fetch_price_units()
    jsonl_write(fn, data)
    log.info("✅ priceUnits обновлён (%s строк): %s", len(data), fn)
    cleanup_old(PRICE_UNITS_DIR, "price_units", CACHE_TTL_DAYS)
    return fn


def load_price_units() -> List[Dict[str, Any]]:
    """
    Загружает из локального кэша справочник priceUnits.

    Если кэш за сегодня отсутствует, сначала скачивает его из API.

    :return: список словарей со справочными разделами прайса
    """
    fn = price_units_path()
    if not fn.exists():
        try:
            update_price_units()
        except Exception as e:
            log.warning("⚠️ Не удалось обновить priceUnits за сегодня: %s", e)
            try:
                fn = latest_file(PRICE_UNITS_DIR, "price_units_*.jsonl")
            except FileNotFoundError:
                raise
    return jsonl_read(fn)


def build_price_units_index(rows: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Строит индекс справочника priceUnits по числовому id.

    :param rows: опциональный список строк priceUnits; если не передан, кэш загружается из файла
    :return: словарь вида {price_unit_id: row}
    """
    src = list(rows) if rows is not None else load_price_units()
    index: Dict[int, Dict[str, Any]] = {}
    for row in src:
        if not isinstance(row, dict):
            continue
        unit_id = _as_int(row.get("id"))
        if unit_id is None:
            continue
        index[unit_id] = row
    return index


def _resolve_price_unit_root_id(price_unit_id: Any, units_index: Dict[int, Dict[str, Any]]) -> int | None:
    """
    Поднимается по parent-цепочке priceUnits до корневого раздела care-setting.

    :param price_unit_id: id раздела прайса у услуги
    :param units_index: индекс справочника priceUnits
    :return: id корневого раздела (146/311/312) либо None
    """
    current_id = _as_int(price_unit_id)
    seen: set[int] = set()
    care_roots = {
        CARE_SETTING_POLYCLINIC_ROOT_ID,
        CARE_SETTING_DAY_HOSPITAL_ROOT_ID,
        CARE_SETTING_INPATIENT_ROOT_ID,
    }

    while current_id is not None and current_id not in seen:
        seen.add(current_id)
        if current_id in care_roots:
            return current_id
        row = units_index.get(current_id)
        if not isinstance(row, dict):
            return None
        current_id = _as_int(row.get("parent"))
    return None


def resolve_price_unit_context(
    price_unit_id: Any,
    *,
    units_index: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Возвращает бизнес-контекст priceUnit: тип оказания услуги и рекомендуемый адрес.

    :param price_unit_id: id раздела прайса услуги
    :param units_index: опциональный заранее построенный индекс priceUnits
    :return: словарь с именем раздела, корневым care-setting и адресом
    """
    idx = units_index or build_price_units_index()
    unit_id = _as_int(price_unit_id)
    unit_row = idx.get(unit_id) if unit_id is not None else None
    root_id = _resolve_price_unit_root_id(unit_id, idx) if unit_id is not None else None
    root_row = idx.get(root_id) if root_id is not None else None

    return {
        "price_unit_id": unit_id,
        "price_unit_name": str((unit_row or {}).get("name") or "").strip(),
        "care_setting_root_id": root_id,
        "care_setting_root_name": str((root_row or {}).get("name") or "").strip(),
        "care_setting_label": str(CARE_SETTING_LABEL_BY_ROOT_ID.get(root_id) or "").strip(),
        "care_setting_address": str(CARE_SETTING_ADDRESS_BY_ROOT_ID.get(root_id) or "").strip(),
    }

# -----------------------------------------------------------------------------
# Функционал: doctor_prices
# -----------------------------------------------------------------------------
def _ensure_doctors_source_current() -> Path:
    """
    Возвращает актуальный doctors_YYYYMMDD.jsonl.

    Если файла за активную дату нет, пересобирает doctor cache через live API
    до первого расчета doctor_prices. Это выравнивает поведение doctor_prices
    с кэшем врачей в мессенджерном сервисе.
    """
    active_path = _active_doctors_path()
    if active_path.exists():
        return active_path

    try:
        doctors = api_nayka.get_all_doctors()
        api_nayka.save_doctors_data(doctors)
    except Exception as e:
        log.warning("Не удалось пересобрать актуальный doctors cache: %s", e)

    if active_path.exists():
        return active_path

    existing = api_nayka.find_existing_doctors_file()
    if existing is not None:
        return existing

    raise FileNotFoundError(f"Актуальный doctors cache не найден в {DOCTORS_DIR}")


def _load_doctors() -> List[Dict[str, Any]]:
    doctors_file = _ensure_doctors_source_current()
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
    calls_total = 0
    calls_with_company_unit = 0

    # 1) Базовые данные из doctors cache (ФИО/регионные названия).
    doc_fio_by_id: Dict[int, str] = {}
    doc_region_name_by_pair: Dict[tuple[int, int], str] = {}
    doc_region_ids_by_doctor: Dict[int, list[int]] = {}
    for doc in doctors:
        did = _as_int(doc.get("id"))
        if did is None:
            continue
        fio = str(doc.get("fio") or "").strip()
        if fio:
            doc_fio_by_id[did] = fio
        region_ids = doc.get("region_ids") or doc.get("regionIds") or []
        region_names = doc.get("regions") or []
        if isinstance(region_ids, list):
            ids = [_as_int(rid) for rid in region_ids]
            doc_region_ids_by_doctor[did] = [rid for rid in ids if rid is not None]
            if isinstance(region_names, list):
                for rid, rname in zip(region_ids, region_names):
                    rid_int = _as_int(rid)
                    if rid_int is None:
                        continue
                    rn = str(rname or "").strip()
                    if rn:
                        doc_region_name_by_pair[(did, rid_int)] = rn

    # 2) Live-источник пар (doctorId, regionId, companyUnitId) из /doctorRegions.
    live_pairs: Dict[int, set[tuple[int, int | None]]] = {}
    try:
        doctor_regions = api_nayka.site_doctor_regions()
    except Exception:
        doctor_regions = []
    for row in doctor_regions if isinstance(doctor_regions, list) else []:
        if not isinstance(row, dict):
            continue
        did = _as_int(row.get("worker"))
        rid = _as_int(row.get("region"))
        cuid = _as_int(row.get("companyUnit"))
        if did is None or rid is None:
            continue
        live_pairs.setdefault(did, set()).add((rid, cuid))

    # 3) Fallback ФИО из /doctors, если doctors cache пуст/неполный.
    try:
        live_doctors = api_nayka.site_doctors()
    except Exception:
        live_doctors = []
    for row in live_doctors if isinstance(live_doctors, list) else []:
        if not isinstance(row, dict):
            continue
        did = _as_int(row.get("id"))
        if did is None or did in doc_fio_by_id:
            continue
        fio = str(row.get("fio") or "").strip()
        if fio:
            doc_fio_by_id[did] = fio

    # 4) Регионные имена fallback из /regions.
    region_name_by_id: Dict[int, str] = {}
    samara_region_ids: set[int] = set()
    try:
        regions = api_nayka.site_regions()
    except Exception:
        regions = []
    for row in regions if isinstance(regions, list) else []:
        if not isinstance(row, dict):
            continue
        rid = _as_int(row.get("id"))
        if rid is None:
            continue
        name = str(row.get("addressForSite") or row.get("name") or "").strip()
        if name:
            region_name_by_id[rid] = name
        if _is_samara_region_row(row):
            samara_region_ids.add(rid)

    # 5) Ограничиваем doctor_prices только самарскими регионами.
    if samara_region_ids:
        filtered_pairs: Dict[int, set[tuple[int, int | None]]] = {}
        for did, pair_set in live_pairs.items():
            kept = {(rid, cuid) for rid, cuid in pair_set if rid in samara_region_ids}
            if kept:
                filtered_pairs[did] = kept
        live_pairs = filtered_pairs

    all_doctor_ids = set(doc_fio_by_id.keys()) | set(doc_region_ids_by_doctor.keys()) | set(live_pairs.keys())
    for doctor_id in sorted(all_doctor_ids):
        fio = doc_fio_by_id.get(doctor_id, "")

        pair_set = live_pairs.get(doctor_id)
        if pair_set:
            region_pairs = sorted(pair_set, key=lambda x: (x[0], x[1] or 0))
        else:
            fallback_region_ids = doc_region_ids_by_doctor.get(doctor_id, [])
            if samara_region_ids:
                fallback_region_ids = [rid for rid in fallback_region_ids if rid in samara_region_ids]
            region_pairs = [(rid, None) for rid in fallback_region_ids if rid is not None]

        if not region_pairs:
            skipped += 1
            continue

        for region_id, company_unit_id in region_pairs:
            calls_total += 1
            if company_unit_id is not None:
                calls_with_company_unit += 1

            services = fetch_doctor_prices(
                doctor_id,
                region_id,
                company_unit_id=company_unit_id,
                is_favorite=False,
            )
            # Защита на случай несовместимого companyUnitId в источнике.
            if not services and company_unit_id is not None:
                services = fetch_doctor_prices(
                    doctor_id,
                    region_id,
                    company_unit_id=None,
                    is_favorite=False,
                )

            if not isinstance(services, list):
                continue
            region_name = (
                doc_region_name_by_pair.get((doctor_id, region_id))
                or region_name_by_id.get(region_id)
            )
            for s in services:
                if not isinstance(s, dict):
                    continue
                rows.append({
                    "doctorId": doctor_id,
                    "regionId": region_id,
                    "companyUnitId": company_unit_id,
                    "regionName": region_name,
                    "fio": fio,
                    "serviceName": s.get("serviceName"),
                    "serviceId": s.get("serviceId"),
                    "cost": s.get("cost"),
                    "priceUnitId": s.get("priceUnitId"),
                    "serviceHomecode": s.get("serviceHomecode"),
                    "deadline": s.get("deadline"),
                })

    jsonl_write(fn, rows)
    log.info(
        "✅ doctor_prices обновлён (%s цен, пропущено врачей: %s, API-вызовов: %s, с companyUnitId: %s): %s",
        len(rows),
        skipped,
        calls_total,
        calls_with_company_unit,
        fn,
    )
    cleanup_old(PRICES_DIR, "doctor_prices", CACHE_TTL_DAYS)
    return fn

def load_doctor_prices() -> List[Dict[str, Any]]:
    fn = doctor_prices_path()
    if not fn.exists():
        update_doctor_prices()
    return jsonl_read(fn)


def _next_doctor_prices_refresh_dt() -> datetime:
    now = _now_samara()
    target = now.replace(hour=8, minute=15, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return target


async def _refresh_doctor_prices_once() -> None:
    """
    Обновляет doctor_prices и связанные справочники прайса в фоновом режиме.

    :return: None
    """

    try:
        await asyncio.to_thread(update_doctor_prices, True)
        log.info("✅ [DAILY REFRESH] doctor_prices обновлен")
    except Exception as e:
        log.warning("⚠️ [DAILY REFRESH] doctor_prices refresh failed: %s", e)
    try:
        await asyncio.to_thread(update_price_units, True)
        log.info("✅ [DAILY REFRESH] priceUnits обновлен")
    except Exception as e:
        log.warning("⚠️ [DAILY REFRESH] priceUnits refresh failed: %s", e)


async def _doctor_prices_refresh_loop() -> None:
    while True:
        target = _next_doctor_prices_refresh_dt()
        now = _now_samara()
        wait_sec = max(1.0, (target - now).total_seconds())
        try:
            await asyncio.sleep(wait_sec)
        except Exception:
            pass
        await _refresh_doctor_prices_once()


def ensure_daily_price_refresh_started() -> bool:
    """Запускает фоновый refresh doctor_prices и priceUnits в 08:15 по Самаре."""
    global _PRICE_REFRESH_TASK
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    if _PRICE_REFRESH_TASK is None or _PRICE_REFRESH_TASK.done():
        _PRICE_REFRESH_TASK = loop.create_task(_doctor_prices_refresh_loop())
        log.info("▶️ [DAILY REFRESH] Планировщик doctor_prices запущен")
        try:
            if not doctor_prices_path().exists() or not price_units_path().exists():
                loop.create_task(_refresh_doctor_prices_once())
        except Exception:
            pass
    return True

# -----------------------------------------------------------------------------
# CLI!
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    update_doctor_prices()
