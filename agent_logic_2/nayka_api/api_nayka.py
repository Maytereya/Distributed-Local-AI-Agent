"""HTTP‑обёртки к CRM «Наука» и сбор агрегированных данных.

Функции:
- site_*: тонкие GET‑вызовы к REST (подразделения, врачи, регионы, связи).
- get_all_doctors(): сборный профиль врача (regions/units/specialization).
- find_doctors_by_keyword(): быстрый поиск по спец‑тям/подразделениям.
- find_doctor_schedule(): расписание по врачам/филиалам с настраиваемым окном поиска.
- Кэш JSONL и ежедневный авто‑рефреш (07:45 Europe/Samara).
Сетевые таймауты/ретраи задаются через переменные окружения.
"""

import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime, date
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Set, Tuple, Union
import requests
import urllib3
import asyncio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


from agent_logic_2 import config as c
from agent_logic_2.nayka_api.cache_paths import resolve_cache_data_dir

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # fallback ниже

# Добавляем корневую директорию в PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

logger = logging.getLogger(__name__)

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)

VERIFY_TLS = os.getenv("NAUKA_VERIFY_TLS", "true").strip().lower() in ("1","true","yes")
CA_BUNDLE = os.getenv("NAUKA_CA_BUNDLE","").strip()
REQ_CONNECT_TIMEOUT = float(os.getenv("NAUKA_TIMEOUT_CONNECT","10"))
REQ_READ_TIMEOUT    = float(os.getenv("NAUKA_TIMEOUT_READ","20"))
DEFAULT_TIMEOUT = (REQ_CONNECT_TIMEOUT, REQ_READ_TIMEOUT)

SESSION = requests.Session()
_retry = Retry(
    total=3, connect=3, read=3, backoff_factor=0.5,
    status_forcelist=[429,500,502,503,504],
    allowed_methods=frozenset(["GET"])
)
adapter = HTTPAdapter(max_retries=_retry)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# --- Realtime fail-fast profile (OC-2) ---
# Patient is waiting on the response (resultForPatient / doctorSchedule / regions-in-request):
# no retry storm, short read timeout, so a sick upstream degrades in seconds, not 45-75s.
# Background cache warming keeps using SESSION (total=3) unchanged.
_retry_realtime = Retry(
    total=0, connect=0, read=0, backoff_factor=0.0,
    # status_forcelist + total=0: a 5xx raises MaxRetryError immediately (no retry storm)
    status_forcelist=[429,500,502,503,504],
    allowed_methods=frozenset(["GET"]),
)
SESSION_REALTIME = requests.Session()
_adapter_realtime = HTTPAdapter(max_retries=_retry_realtime)
SESSION_REALTIME.mount("https://", _adapter_realtime)
SESSION_REALTIME.mount("http://", _adapter_realtime)

REALTIME_READ_TIMEOUT = float(os.getenv("NAUKA_TIMEOUT_READ_REALTIME", "8"))
# Connect timeout intentionally reuses REQ_CONNECT_TIMEOUT (OC-2 specified read=8s only).
# Worst case on a slow-to-connect upstream = connect+read; a per-profile realtime connect
# timeout can be added in Phase 2 if production shows connect stalls.
REALTIME_TIMEOUT = (REQ_CONNECT_TIMEOUT, REALTIME_READ_TIMEOUT)

if CA_BUNDLE:
    VERIFY_ARG = CA_BUNDLE
else:
    VERIFY_ARG = VERIFY_TLS

if VERIFY_ARG is False:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def _session_get(url: str, *, realtime: bool = False, **kwargs):
    kwargs.setdefault("auth", auth)
    kwargs.setdefault("timeout", REALTIME_TIMEOUT if realtime else DEFAULT_TIMEOUT)
    kwargs.setdefault("verify", VERIFY_ARG)
    sess = SESSION_REALTIME if realtime else SESSION
    return sess.get(url, **kwargs)

# Директория для кэширования данных (общая для всех контейнеров через APP_DATA_DIR)
DATA_DIR = resolve_cache_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# регионы, которые исключаем из кэша (Оренбургская область и её потомки)
EXCLUDED_REGION_ROOTS: Set[int] = {19}

# кеш для проверок наличия расписания (doctor_id, company_unit, region_id)
_SCHEDULE_CACHE: Dict[Tuple[int, int, int], bool] = {}
SPECIAL_REGION_NAMES: Dict[int, str] = {
    8502: "Выезд на дом",
}

_ROLE_SPECIALTY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "акушер-гинеколог": ("акушер гинеколог", "гинеколог"),
    "аллерголог": ("аллерголог", "иммунолог"),
    "иммунолог": ("иммунолог", "аллерголог"),
    "кардиолог": ("кардиолог",),
    "эндокринолог": ("эндокринолог",),
    "гинеколог-эндокринолог": ("гинеколог эндокринолог", "гинеколог", "эндокринолог"),
    "гинеколог-маммолог": ("гинеколог маммолог", "гинеколог", "маммолог"),
    "педиатр": ("педиатр",),
    "хирург": ("хирург",),
    "терапевт": ("терапевт",),
    "травматолог": ("травматолог", "ортопед"),
    "проктолог": ("проктолог", "колопроктолог"),
    "колопроктолог": ("колопроктолог", "проктолог"),
    "уролог": ("уролог",),
    "уролог-андролог": ("уролог андролог", "уролог", "андролог"),
    "андролог": ("андролог", "уролог"),
    "онколог": ("онколог",),
    "гинеколог": ("гинеколог",),
    "невролог": ("невролог",),
    "нейрохирург": ("нейрохирург",),
    "нефролог": ("нефролог",),
    "гастроэнтеролог": ("гастроэнтеролог",),
    "гематолог": ("гематолог",),
    "гепатолог": ("гепатолог",),
    "гирудотерапевт": ("гирудотерапевт",),
    "инфекционист": ("инфекционист",),
    "лимфолог": ("лимфолог",),
    "массажист": ("массажист",),
    "пульмонолог": ("пульмонолог",),
    "ревматолог": ("ревматолог",),
    "стоматолог": ("стоматолог",),
    "флеболог": ("флеболог",),
    "фониатр": ("фониатр",),
    "дерматолог": ("дерматолог",),
    "лор": ("лор", "оториноларинг"),
    "оториноларинголог": ("оториноларинголог", "оториноларинг", "лор"),
    "узи": ("узи", "ультразвук"),
}

try:
    SCHEDULE_LOOKAHEAD_DAYS = max(7, int(os.getenv("NAUKA_SCHEDULE_LOOKAHEAD_DAYS", "14")))
except Exception:
    SCHEDULE_LOOKAHEAD_DAYS = 14

# TTL-кэш «шапки» find_doctor_schedule: справочники site_regions + /doctors +
# /doctorCompanyUnits + /doctorRegions меняются ~раз в день (прецедент:
# doctors_mem_ttl_seconds=300 в messengers_router/services/core.py), но
# запрашивались realtime на КАЖДЫЙ запрос расписания — 4 последовательных HTTP,
# ~4-5с из 6.4с (замер 2026-07-07, после П7-параллелизации fanout стали узким
# местом). СЛОТЫ (/doctorSchedule + /doctorScheduleCells) НЕ кэшируются —
# требование владельца: слоты живые на каждый запрос, «даже предыдущей минуты»
# не показываем. 0 (и меньше) = кэш выключен, прежнее поведение.
try:
    SCHEDULE_REFS_TTL_SECONDS = float(os.getenv("NAUKA_SCHEDULE_REFS_TTL_SECONDS", "300"))
except Exception:
    SCHEDULE_REFS_TTL_SECONDS = 300.0

_SCHEDULE_REFS_CACHE: dict = {}
_SCHEDULE_REFS_LOCK = threading.Lock()


def _cached_schedule_ref(key: str, fetch):
    """Отдать справочник из TTL-кэша или сходить в CRM.

    Ошибка fetch пробрасывается и НЕ кэшируется; пустой/falsy результат тоже не
    кэшируется (site_regions при сбое мягко отдаёт [] — нельзя отравить кэш
    пустым списком на весь TTL). Fetch идёт вне лока: конкурентные промахи могут
    сходить в CRM параллельно (не хуже прежнего поведения без кэша).

    :param key: имя справочника (ключ кэша)
    :param fetch: thunk, выполняющий реальный запрос
    :return: результат fetch (возможно, из кэша)
    """

    ttl = SCHEDULE_REFS_TTL_SECONDS
    if ttl <= 0:
        return fetch()
    with _SCHEDULE_REFS_LOCK:
        hit = _SCHEDULE_REFS_CACHE.get(key)
        if hit is not None and (time.monotonic() - hit[0]) < ttl:
            return hit[1]
    value = fetch()
    if value:
        with _SCHEDULE_REFS_LOCK:
            _SCHEDULE_REFS_CACHE[key] = (time.monotonic(), value)
    return value


def _normalise_text(text: str) -> str:
    """
    Нормализует текст для безопасного подстрочного сравнения.

    :param text: исходная строка
    :return: строка в lower-case с заменой "ё" -> "е" и схлопнутыми пробелами
    """
    return " ".join(str(text or "").replace("ё", "е").lower().split())


def _role_terms_for_keyword(keyword: str) -> Tuple[str, ...]:
    """
    Возвращает набор терминов специальности для ролевого поиска.

    :param keyword: ключевое слово пользователя
    :return: кортеж терминов, по которым матчатся подразделения/специализации
    """
    key = _normalise_text(keyword)
    if not key:
        return tuple()
    if key in _ROLE_SPECIALTY_SYNONYMS:
        return _ROLE_SPECIALTY_SYNONYMS[key]
    for canonical, terms in _ROLE_SPECIALTY_SYNONYMS.items():
        if key.startswith(canonical):
            return terms
    return (key,)


def _matches_role_term(text: str, keyword: str) -> bool:
    """
    Проверяет, соответствует ли текст ролевому ключу специальности.

    :param text: текст подразделения или специализации
    :param keyword: ключ поиска (например, "хирург", "узи")
    :return: True, если найдено совпадение по синонимам специальности
    """
    norm = _normalise_text(text)
    if not norm:
        return False
    for term in _role_terms_for_keyword(keyword):
        term_norm = _normalise_text(term)
        if term_norm and term_norm in norm:
            return True
    return False


def _region_display_name(region: Dict[str, Any]) -> str:
    """Берем максимально человекочитаемое имя региона."""
    address = str(region.get("addressForSite") or "").strip()
    name = str(region.get("name") or "").strip()
    return address or name


def _has_schedule(doctor_id: int, company_unit: int, region_id: int, start: str, end: str) -> bool:
    """Проверяет, есть ли у врача активное расписание на площадке в заданный период."""
    key = (doctor_id, company_unit, region_id)
    cached = _SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached

    url = (
        f"{base_url}/doctorSchedule?doctor={doctor_id}&companyUnit={company_unit}&region={region_id}"
        f"&startDate={start}&endDate={end}"
    )
    try:
        resp = _session_get(url)
        resp.raise_for_status()
        data = resp.json() or []
        result = bool(data)
    except (requests.RequestException, ValueError):
        result = False

    _SCHEDULE_CACHE[key] = result
    return result


def _filter_region_entries_with_schedule(
    doctor_id: int,
    entries: List[Dict[str, Any]],
    start_iso: str,
    end_iso: str,
) -> List[Dict[str, Any]]:
    """Возвращает только те doctorRegions, где в ближайшие дни есть расписание."""
    result: List[Dict[str, Any]] = []
    for entry in entries:
        company_unit = entry.get("companyUnit")
        region_id = entry.get("region")
        if not company_unit or not region_id:
            continue
        if _has_schedule(doctor_id, company_unit, region_id, start_iso, end_iso):
            result.append(entry)
    return result


def _now_samara() -> datetime:
    """Текущее время в часовом поясе Самары (Europe/Samara)."""
    try:
        if ZoneInfo is not None:
            return datetime.now(tz=ZoneInfo("Europe/Samara"))
    except Exception:
        pass
    # Фолбэк: считаем, что Самара = UTC+4 без переходов
    return datetime.utcnow() + timedelta(hours=4)


def get_today_str() -> str:
    """Текущая дата (локальная) в формате YYYYMMDD — сохранено для обратной совместимости."""
    return datetime.now().strftime("%Y%m%d")


def get_yesterday_str() -> str:
    """Возвращает вчерашнюю дату в формате YYYYMMDD"""
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def get_active_date_str() -> str:
    """Дата «активного» кэша по Самаре: до 06:00 — вчера, после — сегодня."""
    now = _now_samara()
    if now.hour >= 6:
        return now.strftime("%Y%m%d")
    return (now - timedelta(days=1)).strftime("%Y%m%d")


def _file_has_doctors(file: Path) -> bool:
    """Считаем кэш валидным только если в JSONL есть хотя бы одна непустая строка."""
    try:
        if not file.exists() or file.stat().st_size <= 0:
            return False
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False
    return False


def _doctors_schema_outdated(doctors: List[Dict[str, Any]]) -> bool:
    """
    Проверяет, устарел ли формат кэша doctors_*.jsonl.

    Устаревшим считаем кэш, если:
    - нет поля ord,
    - нет новых main-полей (unit_links/main_units).

    :param doctors: загруженные карточки врачей
    :return: True, если требуется перегенерация кэша
    """
    if not doctors:
        return False
    preview = doctors[:20]
    has_ord = any(isinstance(row, dict) and "ord" in row for row in preview)
    has_main_fields = any(
        isinstance(row, dict) and ("unit_links" in row or "main_units" in row)
        for row in preview
    )
    return (not has_ord) or (not has_main_fields)


def find_existing_doctors_file() -> Union[Path, None]:
    """Находит актуальный непустой файл данных: сначала за активную дату, иначе самый свежий непустой."""
    active = DATA_DIR / f"doctors_{get_active_date_str()}.jsonl"
    if _file_has_doctors(active):
        return active
    files = sorted(DATA_DIR.glob("doctors_*.jsonl"), reverse=True)
    for file in files:
        if _file_has_doctors(file):
            return file
    return None


def get_date_from_filename(file: Path) -> str:
    """Извлекает дату из имени файла"""
    return file.stem.split("_")[-1]


def save_doctors_data(doctors: list):
    """Сохраняет список врачей в формате JSONL — по одному врачу на строку (для активной даты)."""
    if not doctors:
        print("⚠️ Пустой список врачей не сохраняем, чтобы не затереть рабочий кэш")
        return
    # Чистим лишнее, но сохраняем активную и вчерашнюю датy
    cleanup_old_doctors_files()

    date_str = get_active_date_str()
    filename = DATA_DIR / f"doctors_{date_str}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for doc in doctors:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"✅ Врачи сохранены в формате JSONL: {filename}")


def load_doctors_data(file: Path) -> list:
    """Загружает врачей из JSONL файла."""
    doctors_ = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doctors_.append(json.loads(line))
    return doctors_


def cleanup_old_doctors_files(keep_dates: Union[None, set, List[str]] = None):
    """Удаляет все файлы, кроме активной (MSK) и вчерашней дат.
    При передаче keep_dates — сохраняет указанные даты в формате YYYYMMDD.
    """
    if keep_dates is None:
        keep_dates = {get_active_date_str(), (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")}
    else:
        keep_dates = set(keep_dates)
    for file in DATA_DIR.glob("doctors_*.jsonl"):
        file_date = get_date_from_filename(file)
        if file_date not in keep_dates:
            try:
                print(f"🗑️ Удаляем файл с данными о врачах: {file.name}")
                file.unlink()
            except Exception:
                pass


def get_all_doctors() -> List[Dict]:
    """Строит сводные карточки врачей из нескольких эндпоинтов CRM.
    Returns:
        Список словарей:
        {
          id, fio, ord, specialization, regions, region_ids, units,
          unit_links[{company_unit_id, company_unit_name, main, specialization}],
          main_units, main_specializations
        }.
    """
    _SCHEDULE_CACHE.clear()
    # Получаем все данные через API
    doctors = site_doctors()
    units = site_company_units()
    doctor_units = site_doctor_company_units()
    doctor_regions = site_doctor_regions()
    regions = site_regions()

    excluded_region_ids = _collect_region_descendants(regions, EXCLUDED_REGION_ROOTS)

    # Быстрый доступ к названиям регионов и подразделений по id
    regions_dict = {r["id"]: _region_display_name(r) for r in regions}
    units_dict = {u["id"]: u["name"] for u in units}

    # Предподготовка связей
    units_by_doctor: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for link in doctor_units:
        units_by_doctor[link["worker"]].append(link)

    regions_by_doctor: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for link in doctor_regions:
        regions_by_doctor[link["worker"]].append(link)

    start_dt = date.today()
    end_dt = start_dt + timedelta(days=7)
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    result = []
    for doctor in doctors:
        doctor_id = doctor["id"]

        unit_links = units_by_doctor.get(doctor_id, [])
        region_links = regions_by_doctor.get(doctor_id, [])

        # Уберём оренбургские площадки заранее
        valid_region_links = []
        for link in region_links:
            reg_id = link.get("region")
            if not reg_id or reg_id in excluded_region_ids:
                continue
            valid_region_links.append(link)
        if not valid_region_links:
            # Врач присутствует только в исключённых регионах
            continue

        # Все specialization (из unit_links)
        specs = [link.get("specialization", "") or "" for link in unit_links]
        specs = list(dict.fromkeys(filter(None, specs)))

        # Оставляем только те площадки, где есть расписание в ближайшую неделю
        active_region_links = _filter_region_entries_with_schedule(
            doctor_id, valid_region_links, start_iso, end_iso
        )
        selected_region_links = active_region_links or valid_region_links

        seen_region_ids: Set[int] = set()
        region_pairs: List[Tuple[int, str]] = []
        for link in selected_region_links:
            reg_id = link.get("region")
            if not reg_id or reg_id in seen_region_ids:
                continue
            seen_region_ids.add(reg_id)
            reg_name = regions_dict.get(reg_id) or SPECIAL_REGION_NAMES.get(reg_id)
            region_pairs.append((reg_id, reg_name or f"ID {reg_id}"))

        doc_region_ids = [r[0] for r in region_pairs]
        doc_regions = [r[1] for r in region_pairs]
        if not doc_region_ids:
            continue

        # Подразделения только для активных площадок
        unit_ids_from_regions = {
            entry.get("companyUnit") for entry in selected_region_links if entry.get("companyUnit")
        }
        if not unit_ids_from_regions:
            unit_ids_from_regions = {
                entry.get("companyUnit") for entry in valid_region_links if entry.get("companyUnit")
            }
        # Для role-матчинга используем весь валидный самарский контур врача, а не только
        # площадки с ближайшими слотами, иначе можно потерять "main" специализацию.
        unit_ids_from_valid_regions = {
            entry.get("companyUnit") for entry in valid_region_links if entry.get("companyUnit")
        }
        if not unit_ids_from_valid_regions:
            unit_ids_from_valid_regions = {entry.get("companyUnit") for entry in unit_links if entry.get("companyUnit")}

        doc_units: List[str] = []
        for link in unit_links:
            unit_id = link.get("companyUnit")
            if unit_ids_from_regions and unit_id not in unit_ids_from_regions:
                continue
            unit_name = units_dict.get(unit_id)
            if unit_name and unit_name not in doc_units:
                doc_units.append(unit_name)
        if not doc_units:
            # fallback — возьмём все названия подразделений врача
            for link in unit_links:
                unit_name = units_dict.get(link.get("companyUnit"))
                if unit_name and unit_name not in doc_units:
                    doc_units.append(unit_name)
        if not doc_units:
            continue

        unit_links_payload: List[Dict[str, Any]] = []
        seen_link_keys: Set[Tuple[int, bool, str]] = set()
        for link in unit_links:
            unit_id = link.get("companyUnit")
            if unit_id is None:
                continue
            if unit_ids_from_valid_regions and unit_id not in unit_ids_from_valid_regions:
                continue
            unit_name = units_dict.get(unit_id)
            if not unit_name:
                continue
            link_specialization = str(link.get("specialization") or "").strip()
            link_main = bool(link.get("main"))
            dedupe_key = (int(unit_id), link_main, _normalise_text(link_specialization))
            if dedupe_key in seen_link_keys:
                continue
            seen_link_keys.add(dedupe_key)
            unit_links_payload.append(
                {
                    "company_unit_id": int(unit_id),
                    "company_unit_name": unit_name,
                    "main": link_main,
                    "specialization": link_specialization,
                }
            )

        main_units = list(
            dict.fromkeys(
                [
                    str(link.get("company_unit_name") or "").strip()
                    for link in unit_links_payload
                    if bool(link.get("main")) and str(link.get("company_unit_name") or "").strip()
                ]
            )
        )
        main_specializations = list(
            dict.fromkeys(
                [
                    str(link.get("specialization") or "").strip()
                    for link in unit_links_payload
                    if bool(link.get("main")) and str(link.get("specialization") or "").strip()
                ]
            )
        )

        # Каноническое описание врача для кэша:
        # сначала берем specialization из main=true связей (как на сайте),
        # и только при отсутствии main-описания используем legacy первый spec.
        canonical_specialization = main_specializations[0] if main_specializations else (specs[0] if specs else None)

        doctor_data = {
            "id": doctor_id,
            "fio": doctor["fio"],
            "ord": doctor.get("ord"),
            "specialization": canonical_specialization,
            "regions": doc_regions,
            "region_ids": doc_region_ids,
            "units": doc_units,
            "unit_links": unit_links_payload,
            "main_units": main_units,
            "main_specializations": main_specializations,
        }

        result.append(doctor_data)

    return result


def get_cached_doctors_data() -> list:
    """
    Возвращает кэшированные данные о врачах. Активная дата переключается в 06:00 по Москве:
    до 06:00 — используем «вчера», после — «сегодня». При отсутствии файла для активной даты
    — скачиваем заново и сохраняем.
    """
    active = get_active_date_str()
    existing_file = DATA_DIR / f"doctors_{active}.jsonl"
    if _file_has_doctors(existing_file):
        print(f"✅ Нашли кэш за активную дату {active}: {existing_file.name}")
        cached = load_doctors_data(existing_file)
        if not _doctors_schema_outdated(cached):
            return cached
        print(f"ℹ️ Кэш {existing_file.name} в старом формате — перегенерируем через API")
    elif existing_file.exists():
        print(f"⚠️ Кэш за активную дату {active} пустой/битый: {existing_file.name}")
    else:
        print(f"[DEBUG] Кэш за активную дату {active} не найден — обновляем через API!")

    # Для расписаний и doctor-resolution сначала пытаемся получить свежий список врачей.
    try:
        doctors = get_all_doctors()
    except Exception as e:
        print(f"⚠️ Не удалось обновить список врачей через API: {e}")
        doctors = []

    if doctors:
        save_doctors_data(doctors)
        print("✅ Новые данные о врачах успешно загружены")
        return doctors

    fallback_file = find_existing_doctors_file()
    if fallback_file is not None:
        print(f"⚠️ Используем последний непустой кэш врачей: {fallback_file.name}")
        cached = load_doctors_data(fallback_file)
        if not _doctors_schema_outdated(cached):
            return cached
        print(f"ℹ️ Фолбэк-кэш {fallback_file.name} в старом формате — верну как временный запас")
        return cached

    return []

# ==========================
# Ежедневное обновление кэша в 07:45 (Самара)
# ==========================
_refresh_task = None

def _next_refresh_dt() -> datetime:
    now = _now_samara()
    target = now.replace(hour=7, minute=45, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return target

async def _refresh_once():
    try:
        # 1) Врачи
        docs = await asyncio.to_thread(get_all_doctors)
        save_doctors_data(docs)
        # подчистим, оставив активную и вчерашнюю
        cleanup_old_doctors_files()
        print("✅ [DAILY REFRESH] Кэш врачей обновлён")
        # 2) Заметки call-центра (zametka_button_*.json)
        try:
            from agent_logic_2.nayka_api.doctors_cc_info import get_doctors_cc_info as _get_cc
            await asyncio.to_thread(_get_cc, True)  # force=True
            print("✅ [DAILY REFRESH] Кэш заметок КЦ обновлён")
        except Exception as cc_e:
            print(f"⚠️ [DAILY REFRESH] Ошибка обновления заметок КЦ: {cc_e}")
    except Exception as e:
        print(f"⚠️ [DAILY REFRESH] Ошибка обновления кэша: {e}")

async def _daily_refresh_loop():
    while True:
        target = _next_refresh_dt()
        now = _now_samara()
        wait_sec = max(1.0, (target - now).total_seconds())
        try:
            await asyncio.sleep(wait_sec)
        except Exception:
            # если sleep прерван, цикл продолжится и пересчитает target
            pass
        await _refresh_once()

def ensure_daily_refresh_started() -> bool:
    """Запускает фоновую задачу обновления кэша в 07:45 по Самаре (idempotent).
    Возвращает True, если задача запущена (или уже была запущена) внутри запущенного event loop.
    """
    global _refresh_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # нет активного event loop — запустить позже
        return False
    if _refresh_task is None or _refresh_task.done():
        _refresh_task = loop.create_task(_daily_refresh_loop())
        print("▶️ [DAILY REFRESH] Планировщик запущен")
        # Если активного файла нет (например, приложение запущено после 07:45),
        # дергаем немедленное обновление в фоне, чтобы не тратить время на первый запрос.
        try:
            active_file = DATA_DIR / f"doctors_{get_active_date_str()}.jsonl"
            if not active_file.exists():
                loop.create_task(_refresh_once())
        except Exception:
            pass
    return True


def _units_tree() -> Dict[int, Set[int]]:
    """Строит дерево подразделений."""
    units = site_company_units()
    tree = {}
    for unit in units:
        parent_id = unit.get("parentId")
        if parent_id:
            if parent_id not in tree:
                tree[parent_id] = set()
            tree[parent_id].add(unit["id"])
    return tree


def _descendants(unit_ids: Set[int], tree: Dict[int, Set[int]]) -> Set[int]:
    """Находит все дочерние подразделения."""
    result = unit_ids.copy()
    to_process = unit_ids.copy()

    while to_process:
        current = to_process.pop()
        children = tree.get(current, set())
        new_children = children - result
        result.update(new_children)
        to_process.update(new_children)

    return result


def _collect_region_descendants(regions: List[Dict[str, Any]], roots: Set[int]) -> Set[int]:
    """Строит множество id регионов, входящих в указанные корни (с учётом потомков)."""
    if not roots:
        return set()

    tree: Dict[int, Set[int]] = {}
    for region in regions:
        parent = region.get("parentId") or region.get("parent")
        child_id = region.get("id")
        if parent is None or child_id is None:
            continue
        tree.setdefault(parent, set()).add(child_id)

    excluded: Set[int] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        if current in excluded:
            continue
        excluded.add(current)
        stack.extend(tree.get(current, ()))

    return excluded


def find_doctors_by_keyword(keyword: str) -> Union[List[Dict], str]:
    """
    Ищем по:
      • тексту specialization («кардиолог», «ультразвук»)
    Возвращаем краткий список врачей.
    """
    kw = _normalise_text(keyword)
    print(f"\nИщем врачей по ключевому слову: {kw}")

    # Загружаем кэшированные данные
    data = get_cached_doctors_data()

    # Распаковываем данные
    units = site_company_units()
    unit_name_by_id = {u["id"]: u["name"] for u in units}
    links = site_doctor_company_units()
    doctors = {d["id"]: d["fio"] for d in data}

    matched_workers = set()

    # Ролевое совпадение по main=true (новый контракт doctorCompanyUnits.main).
    # Если main-связи есть и они совпали по keyword, считаем это приоритетным попаданием.
    main_role_workers: Set[int] = set()
    for doc in data:
        if not isinstance(doc, dict):
            continue
        doctor_id = doc.get("id")
        if doctor_id is None:
            continue
        doc_links = doc.get("unit_links") or []
        if not isinstance(doc_links, list):
            continue
        has_main_links = False
        for link in doc_links:
            if not isinstance(link, dict):
                continue
            if not bool(link.get("main")):
                continue
            has_main_links = True
            unit_name = str(link.get("company_unit_name") or "")
            link_spec = str(link.get("specialization") or "")
            if _matches_role_term(unit_name, kw) or _matches_role_term(link_spec, kw):
                try:
                    main_role_workers.add(int(doctor_id))
                except Exception:
                    pass
                break
        # Если main-связей нет вовсе — оставляем врача для fallback ниже.
        if has_main_links:
            continue

    for link in links:
        spec = link.get("specialization", "") or ""
        spec_lower = _normalise_text(spec)
        unit = _normalise_text(unit_name_by_id.get(link["companyUnit"], ""))

        is_match = False

        if _matches_role_term(unit, kw):
            is_match = True
        elif _matches_role_term(spec_lower, kw):
            if not any(other in spec_lower for other in [
                "ультразвуковая", "функциональная", "терапевт",
                "в ревматологии", "по ревматологии", "ревматологический"
            ]):
                is_match = True

        if is_match:
            matched_workers.add(link["worker"])

    if main_role_workers:
        matched_workers = main_role_workers

    if not matched_workers:
        return []

    result = []
    for wid in matched_workers:
        base_doc = next((d for d in data if d.get("id") == wid), None)
        if not base_doc:
            continue
        fio = doctors.get(wid)
        if not fio:
            continue
        specs = [
            link.get("specialization", "") or ""
            for link in links
            if link["worker"] == wid
        ]
        specs = list(set(filter(None, specs)))

        doc = {
            "id": wid,
            "fio": fio,
            "specialization": specs[0] if specs else "Специализация не указана",
            "directions": [s for s in specs[1:]],
            "regions": base_doc.get("regions", ["-"]) if base_doc else ["-"],
        }
        result.append(doc)

    return result


def site_company_units():
    """Получить список подразделений."""
    try:
        resp = _session_get(f"{base_url}/companyUnits")
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Ошибка при получении списка подразделений: {e}")
        return []


def site_doctors():
    """Получить список врачей."""
    try:
        resp = _session_get(f"{base_url}/doctors")
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Ошибка при получении списка врачей: {e}")
        return []


def site_doctor_company_units():
    """Получить связи врачей с подразделениями."""
    try:
        resp = _session_get(f"{base_url}/doctorCompanyUnits")
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Ошибка при получении связей врач–подразделение: {e}")
        return []


def site_doctor_regions():
    """Получить связи врачей с регионами."""
    try:
        resp = _session_get(f"{base_url}/doctorRegions")
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Ошибка при получении связей врач–регион: {e}")
        return []


def site_regions(*, realtime: bool = False):
    """Получить список регионов."""
    try:
        resp = _session_get(f"{base_url}/regions", realtime=realtime)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Ошибка при получении списка регионов: {e}")
        return []


def site_result_for_patient(
    *,
    surname: str,
    year: int,
    filial: str,
    number: int,
    lang: str | None = None,
    with_time: bool | None = None,
) -> dict[str, Any]:
    """
    Получить результаты пациента через endpoint /site/resultForPatient.

    Параметры соответствуют форме на сайте:
    - surname: фамилия пациента
    - year: год рождения
    - filial: код/название филиала
    - number: номер (код) анализа
    - lang: язык (опционально)
    - with_time: флаг time (опционально)
    """
    params: dict[str, Any] = {
        "surname": surname,
        "year": int(year),
        "filial": filial,
        "number": int(number),
    }
    if isinstance(lang, str) and lang.strip():
        params["lang"] = lang.strip()
    if with_time is not None:
        params["time"] = bool(with_time)

    try:
        resp = _session_get(f"{base_url}/resultForPatient", params=params, realtime=True)
        resp.raise_for_status()

        content_type = str(resp.headers.get("Content-Type") or "").lower()
        body_text = (resp.text or "").strip()

        # API может возвращать как JSON, так и plain text (например, ссылку на бланк).
        if "application/json" in content_type:
            try:
                payload: Any = resp.json()
            except ValueError:
                # Некорректный JSON при JSON Content-Type: пробуем текст как fallback.
                if not body_text:
                    return {
                        "ok": False,
                        "status_code": resp.status_code,
                        "error": "empty_json_response",
                        "params": params,
                    }
                try:
                    payload = json.loads(body_text)
                except Exception:
                    payload = body_text
        else:
            if not body_text:
                return {
                    "ok": False,
                    "status_code": resp.status_code,
                    "error": "empty_response",
                    "params": params,
                }
            try:
                payload = json.loads(body_text)
            except Exception:
                payload = body_text

        return {
            "ok": True,
            "status_code": resp.status_code,
            "content_type": content_type,
            "data": payload,
        }
    except requests.RequestException as e:
        return {
            "ok": False,
            "status_code": getattr(getattr(e, "response", None), "status_code", None),
            "error": str(e),
            "params": params,
        }


def find_doctor_schedule(
        last_name: str,
        region_name: str = None  # теперь не обязательно
):
    """
    Ищем по:
      • тексту fio (фамилия, имя, отчество)  («Смирнова», «Иванов Иван»)
    Возвращаем краткий список врачей.
    """
    # Лимитер на количество врачей, чтобы не штурмовать API при большом числе совпадений
    MAX_DOCS = int(os.getenv("NAUKA_MAX_SCHEDULE_DOCS", "5"))

    # --- Получаем регионы (шапка: TTL-кэш, при сбое site_regions отдаёт []) ---
    regions = _cached_schedule_ref("site_regions", lambda: site_regions(realtime=True))
    region_map = {r["id"]: _region_display_name(r) for r in regions}
    region_id = None
    if region_name:
        region_id = next((r["id"] for r in regions if region_name.lower() in r["name"].lower()), None)
        if not region_id:
            return f"Регион '{region_name}' не найден."

    # --- Получаем врачей (шапка: TTL-кэш; ошибка не кэшируется) ---
    def _fetch_doctors_ref():
        resp = _session_get(f"{base_url}/doctors", realtime=True)
        resp.raise_for_status()
        return resp.json()

    try:
        doctors = _cached_schedule_ref("doctors", _fetch_doctors_ref)
    except (requests.RequestException, ValueError) as e:
        logger.warning(
            "find_doctor_schedule: /doctors fetch failed last_name=%r error=%s",
            last_name, e,
        )
        return f"Не удалось получить список врачей: {e}"
    doctor_dict = {doc["id"]: doc for doc in doctors}
    matched_doctors = {doc["id"]: doc for doc in doctors if last_name.lower() in doc["fio"].lower()}
    if not matched_doctors:
        return f"Врач с фамилией (или частью ФИО) '{last_name}' не найден."

    # Ограничиваем количество врачей до MAX_DOCS, если их слишком много
    if len(matched_doctors) > MAX_DOCS:
        matched_ids = sorted(matched_doctors.keys(), key=lambda i: doctor_dict[i]["fio"])[:MAX_DOCS]
        matched_doctors = {i: doctor_dict[i] for i in matched_ids}

    # --- Получаем companyUnit и doctorRegions (шапка: TTL-кэш; ошибки не кэшируются) ---
    def _fetch_company_units_ref():
        resp = _session_get(f"{base_url}/doctorCompanyUnits", realtime=True)
        resp.raise_for_status()
        return resp.json()

    def _fetch_doctor_regions_ref():
        resp = _session_get(f"{base_url}/doctorRegions", realtime=True)
        resp.raise_for_status()
        return resp.json()

    try:
        mappings = _cached_schedule_ref("doctor_company_units", _fetch_company_units_ref)
        dr_regions = _cached_schedule_ref("doctor_regions", _fetch_doctor_regions_ref)
    except (requests.RequestException, ValueError) as e:
        logger.warning(
            "find_doctor_schedule: doctorCompanyUnits/doctorRegions fetch failed "
            "last_name=%r error=%s",
            last_name, e,
        )
        return f"Не удалось получить связи врача: {e}"

    start_date = date.today().isoformat()
    end_date = (date.today() + timedelta(days=SCHEDULE_LOOKAHEAD_DAYS)).isoformat()

    # П7 (Resilience Phase 2): fanout «филиалы × дни» распараллелен. Раньше все
    # /doctorSchedule (по филиалам) и /doctorScheduleCells (по дням) шли строго
    # последовательно: 2 филиала × 12 дней = ~26 HTTP-вызовов цепочкой (7.2с
    # Трубин, baseline-2026-06-26). Два фазовых пула с ограничением воркеров
    # (не душить medserver): фаза A — /doctorSchedule по независимым филиалам,
    # фаза B — /doctorScheduleCells по плоскому списку всех дней всех филиалов.
    # Вложенных сабмитов в ОДИН пул нет — иначе дедлок при заполнении воркеров.
    # ПОРЯДОК результатов детерминирован: executor.map возвращает по порядку
    # сабмита (филиалы — порядок doctorRegions, дни — порядок /doctorSchedule),
    # не по прибытию ответов. SESSION шарится между потоками — это уже так для
    # конкурентных find_doctor_schedule из _schedule_by_specialty (gather×8).
    fanout_workers = max(1, int(os.getenv("NAUKA_SCHEDULE_FANOUT_WORKERS", "4")))

    # --- Собираем расписания ---
    result = []
    matched_but_without_slots = False
    # Трекаем сбои fetch внутри цикла: если врач найден и регионы есть, но все
    # слоты собраны при наличии ошибок /doctorSchedule|/doctorScheduleCells —
    # вывод «нет слотов» может быть артефактом сбоя, а не реальностью.
    any_fetch_error = False
    for doctor_id, doctor_obj in matched_doctors.items():
        doctor_mappings = [m for m in mappings if m["worker"] == doctor_id]
        doctor_reg_entries = [r for r in dr_regions if r["worker"] == doctor_id]
        if region_id:
            doctor_reg_entries = [r for r in doctor_reg_entries if r["region"] == region_id]
        if not doctor_reg_entries:
            continue

        # Фаза A: /doctorSchedule по всем филиалам врача параллельно.
        def _fetch_region_days(region_entry: dict[str, Any], *, _doctor_id=doctor_id) -> tuple[list[dict[str, Any]] | None, bool]:
            """Вернуть ``(schedule_days | None, fetch_error)`` для одного филиала."""
            company_unit = region_entry["companyUnit"]
            reg_id = region_entry["region"]
            schedule_url = (
                f"{base_url}/doctorSchedule?doctor={_doctor_id}&companyUnit={company_unit}&region={reg_id}"
                f"&startDate={start_date}&endDate={end_date}"
            )
            try:
                schedule_resp = _session_get(schedule_url, realtime=True)
                schedule_resp.raise_for_status()
                return schedule_resp.json(), False
            except (requests.RequestException, ValueError) as exc:
                logger.warning(
                    "find_doctor_schedule: /doctorSchedule fetch failed, skipping region "
                    "doctor_id=%s region_id=%s url=%s error=%s",
                    _doctor_id, reg_id, schedule_url, exc,
                )
                return None, True

        with ThreadPoolExecutor(max_workers=fanout_workers) as pool:
            region_days = list(pool.map(_fetch_region_days, doctor_reg_entries))

        # Плоский список дней всех филиалов: (индекс филиала, day).
        day_jobs: list[tuple[int, dict[str, Any]]] = []
        for region_idx, (schedule_days, fetch_error) in enumerate(region_days):
            if fetch_error:
                any_fetch_error = True
                continue
            for day in schedule_days or []:
                day_jobs.append((region_idx, day))

        # Фаза B: /doctorScheduleCells по всем дням всех филиалов параллельно.
        def _fetch_day_cells(job: tuple[int, dict[str, Any]], *, _doctor_id=doctor_id) -> tuple[int, dict[str, Any], list[Any], bool]:
            """Вернуть ``(индекс филиала, day, free_slots, fetch_error)`` для одного дня."""
            region_idx, day = job
            cells_url = f"{base_url}/doctorScheduleCells?doctorSchedule={day['id']}"
            try:
                cells_resp = _session_get(cells_url, realtime=True)
                cells_resp.raise_for_status()
                cells = cells_resp.json()
                free_slots = [cell.get("startTime") for cell in cells if isinstance(cell, dict) and cell.get("free")]
                return region_idx, day, free_slots, False
            except (requests.RequestException, ValueError) as exc:
                logger.warning(
                    "find_doctor_schedule: /doctorScheduleCells fetch failed, treating day as no free slots "
                    "doctor_id=%s schedule_day_id=%s url=%s error=%s",
                    _doctor_id, day.get("id"), cells_url, exc,
                )
                return region_idx, day, [], True

        with ThreadPoolExecutor(max_workers=fanout_workers) as pool:
            day_results = list(pool.map(_fetch_day_cells, day_jobs))

        rows_by_region_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for region_idx, day, free_slots, fetch_error in day_results:
            if fetch_error:
                any_fetch_error = True
            if not free_slots:
                continue
            rows_by_region_idx[region_idx].append({
                "date": day.get("curDate"),
                "start": day.get("startTime"),
                "end": day.get("endTime"),
                "slots": free_slots
            })

        # Сборка в исходном порядке doctorRegions (мержим расписание по регионам!)
        schedules_by_region = defaultdict(list)
        region_names = set()
        for region_idx, region_entry in enumerate(doctor_reg_entries):
            region_rows = rows_by_region_idx.get(region_idx) or []
            if region_rows:
                reg_id = region_entry["region"]
                region_name_val = region_map.get(reg_id) or SPECIAL_REGION_NAMES.get(reg_id) or f"[ID {reg_id}]"
                region_names.add(region_name_val)
                schedules_by_region[region_name_val].extend(region_rows)

        # Специализация
        spec = None
        for m in doctor_mappings:
            if m.get("specialization"):
                spec = m["specialization"]
                break

        if not schedules_by_region:
            matched_but_without_slots = True
            continue

        result.append({
            "id": doctor_id,
            "fio": doctor_obj["fio"],
            "ord": doctor_obj.get("ord"),
            "specialization": spec,
            "regions": list(region_names),
            "schedule": dict(schedules_by_region)
        })

    if not result and matched_but_without_slots:
        if any_fetch_error:
            # Врач найден, регионы есть, но слоты так и не собраны ПРИ наличии
            # сбоев fetch. Вывод «нет слотов» здесь может быть артефактом
            # частичного сбоя CRM, а не реальным отсутствием приёма (кейс
            # Паничевой 28.05). Поведение пока не меняем (возвращаем
            # NO_FREE_SLOTS — безопасный handoff-офер), но помечаем в логах как
            # подозрительное: основа для будущей эскалации в api_error.
            logger.warning(
                "find_doctor_schedule: NO_FREE_SLOTS for last_name=%r but some "
                "schedule/cells fetches errored — result may be incomplete "
                "(suspected API failure misreported as no-slots)",
                last_name,
            )
        return _NO_FREE_SLOTS_MESSAGE
    if not result:
        return f"Врач с фамилией '{last_name}' не найден."
    return result


# Каноническое сообщение «врач найден, слотов нет в 14 дней» —
# вынесено в константу, чтобы caller-ы могли отличить этот кейс от
# других строковых ошибок без дублирования текста.
_NO_FREE_SLOTS_MESSAGE = "Врач найден, но свободных слотов нет в ближайшие 2 недели."
_NO_FREE_SLOTS_MARKER = "свободных слотов нет"
# Маркер строковых ответов find_doctor_schedule, означающих СБОЙ обращения
# к CRM (сеть/5xx после исчерпания HTTP-ретраев), а не валидный негатив.
# Обе ветки ошибок («Не удалось получить список врачей: …» и
# «Не удалось получить связи врача: …») начинаются с этой фразы.
_API_ERROR_MARKER = "не удалось получить"


def is_no_free_slots_message(payload: str | None) -> bool:
    """Проверяет, является ли строковый ответ ``find_doctor_schedule``
    маркером «врач есть, свободных слотов нет в 14-дневном окне».

    Используется в `messengers_router.services.core` чтобы отличить
    «нет слотов» от «врач не найден» при нормализации не-list ответов
    в кэшируемый формат.

    :param payload: ответ ``find_doctor_schedule`` или иное строковое
                    сообщение.
    :return: True если это «нет свободных слотов», False во всех
             остальных случаях (включая None / нестрока).
    """
    if not isinstance(payload, str):
        return False
    return _NO_FREE_SLOTS_MARKER in payload.lower()


def is_api_error_message(payload: str | None) -> bool:
    """Проверяет, является ли строковый ответ ``find_doctor_schedule``
    признаком СБОЯ обращения к CRM (сеть/5xx), а не валидным негативом
    («врач не найден» / «регион не найден» / «нет слотов»).

    Нужно, чтобы caller (`messengers_router.services.core`) мог отличить
    недоступность источника (→ stale-fallback из кэша или честный
    handoff на оператора) от ситуации «врача действительно нет»
    (→ сообщение «расписание не найдено»). Раньше оба случая сжимались
    в `[]`, и пациент при сбое API видел вводящее в заблуждение
    «расписание не найдено».

    :param payload: ответ ``find_doctor_schedule`` или иное строковое
                    сообщение.
    :return: True если это ошибка обращения к источнику, иначе False
             (включая None / нестрока / валидные негативы).
    """
    if not isinstance(payload, str):
        return False
    return _API_ERROR_MARKER in payload.lower()


if __name__ == "__main__":
    doctors = find_doctor_schedule(
        last_name="Дразнин"
    )
    regions = site_regions()
    print("Регионы:", regions)

    pprint(doctors, width=150)
