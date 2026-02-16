"""HTTP‑обёртки к CRM «Наука» и сбор агрегированных данных.

Функции:
- site_*: тонкие GET‑вызовы к REST (подразделения, врачи, регионы, связи).
- get_all_doctors(): сборный профиль врача (regions/units/specialization).
- find_doctors_by_keyword(): быстрый поиск по спец‑тям/подразделениям.
- find_doctor_schedule(): расписание на 7 дней по врачам/филиалам.
- Кэш JSONL и ежедневный авто‑рефреш (07:45 Europe/Samara).
Сетевые таймауты/ретраи задаются через переменные окружения.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import timedelta, datetime, date, time
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Set, Tuple, Union
import requests, urllib3
import asyncio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import requests

from agent_logic_2 import config as c

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # fallback ниже

# Добавляем корневую директорию в PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

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

if CA_BUNDLE:
    VERIFY_ARG = CA_BUNDLE
else:
    VERIFY_ARG = VERIFY_TLS

if VERIFY_ARG is False:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def _session_get(url: str, **kwargs):
    kwargs.setdefault("auth", auth)
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    kwargs.setdefault("verify", VERIFY_ARG)
    return SESSION.get(url, **kwargs)

# Директория для кэширования данных
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "apidata"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# регионы, которые исключаем из кэша (Оренбургская область и её потомки)
EXCLUDED_REGION_ROOTS: Set[int] = {19}

# кеш для проверок наличия расписания (doctor_id, company_unit, region_id)
_SCHEDULE_CACHE: Dict[Tuple[int, int, int], bool] = {}


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


def find_existing_doctors_file() -> Union[Path, None]:
    """Находит актуальный файл данных: сначала за активную дату (MSK 06:00), иначе самый свежий."""
    active = DATA_DIR / f"doctors_{get_active_date_str()}.jsonl"
    if active.exists():
        return active
    files = sorted(DATA_DIR.glob("doctors_*.jsonl"), reverse=True)
    return files[0] if files else None


def get_date_from_filename(file: Path) -> str:
    """Извлекает дату из имени файла"""
    return file.stem.split("_")[-1]


def save_doctors_data(doctors: list):
    """Сохраняет список врачей в формате JSONL — по одному врачу на строку (для активной даты)."""
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
        Список словарей: {id, fio, specialization, regions, region_ids, units}.
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
    regions_dict = {r["id"]: r["name"] for r in regions}
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
            reg_name = regions_dict.get(reg_id)
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

        doctor_data = {
            "id": doctor_id,
            "fio": doctor["fio"],
            "specialization": specs[0] if specs else None,
            "regions": doc_regions,
            "region_ids": doc_region_ids,
            "units": doc_units,
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
    if existing_file.exists():
        print(f"✅ Нашли кэш за активную дату {active}: {existing_file.name}")
        return load_doctors_data(existing_file)

    # Кэша на активную дату нет — обновляем
    print(f"[DEBUG] Кэш за активную дату {active} не найден — обновляем через API!")
    doctors = get_all_doctors()
    save_doctors_data(doctors)
    print("✅ Новые данные о врачах успешно загружены")
    return doctors

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
    kw = keyword.lower()
    print(f"\nИщем врачей по ключевому слову: {kw}")

    # Загружаем кэшированные данные
    data = get_cached_doctors_data()

    # Распаковываем данные
    units = site_company_units()
    unit_name_by_id = {u["id"]: u["name"] for u in units}
    links = site_doctor_company_units()
    doctors = {d["id"]: d["fio"] for d in data}

    matched_workers = set()
    for link in links:
        spec = link.get("specialization", "") or ""
        spec_lower = spec.lower()
        unit = unit_name_by_id.get(link["companyUnit"], "").lower()

        is_match = False

        if kw in unit:
            is_match = True
        elif kw in spec_lower:
            if not any(other in spec_lower for other in [
                "ультразвуковая", "функциональная", "терапевт",
                "в ревматологии", "по ревматологии", "ревматологический"
            ]):
                is_match = True

        if is_match:
            matched_workers.add(link["worker"])

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


def site_regions():
    """Получить список регионов."""
    try:
        resp = _session_get(f"{base_url}/regions")
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
        resp = _session_get(f"{base_url}/resultForPatient", params=params)
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

    # --- Получаем регионы ---
    regions = site_regions()
    region_map = {r["id"]: r["name"] for r in regions}
    region_id = None
    if region_name:
        region_id = next((r["id"] for r in regions if region_name.lower() in r["name"].lower()), None)
        if not region_id:
            return f"Регион '{region_name}' не найден."

    # --- Получаем врачей ---
    try:
        doctors_resp = _session_get(f"{base_url}/doctors")
        doctors_resp.raise_for_status()
        doctors = doctors_resp.json()
    except (requests.RequestException, ValueError) as e:
        return f"Не удалось получить список врачей: {e}"
    doctor_dict = {doc["id"]: doc for doc in doctors}
    matched_doctors = {doc["id"]: doc for doc in doctors if last_name.lower() in doc["fio"].lower()}
    if not matched_doctors:
        return f"Врач с фамилией (или частью ФИО) '{last_name}' не найден."

    # Ограничиваем количество врачей до MAX_DOCS, если их слишком много
    if len(matched_doctors) > MAX_DOCS:
        matched_ids = sorted(matched_doctors.keys(), key=lambda i: doctor_dict[i]["fio"])[:MAX_DOCS]
        matched_doctors = {i: doctor_dict[i] for i in matched_ids}

    # --- Получаем companyUnit и doctorRegions ---
    try:
        mappings = _session_get(f"{base_url}/doctorCompanyUnits")
        mappings.raise_for_status()
        mappings = mappings.json()
        dr_regions = _session_get(f"{base_url}/doctorRegions")
        dr_regions.raise_for_status()
        dr_regions = dr_regions.json()
    except (requests.RequestException, ValueError) as e:
        return f"Не удалось получить связи врача: {e}"

    start_date = date.today().isoformat()
    end_date = (date.today() + timedelta(days=7)).isoformat()

    # --- Собираем расписания ---
    result = []
    for doctor_id, doctor_obj in matched_doctors.items():
        doctor_mappings = [m for m in mappings if m["worker"] == doctor_id]
        doctor_reg_entries = [r for r in dr_regions if r["worker"] == doctor_id]
        if region_id:
            doctor_reg_entries = [r for r in doctor_reg_entries if r["region"] == region_id]
        if not doctor_reg_entries:
            continue

        # Мержим расписание по регионам!
        schedules_by_region = defaultdict(list)
        region_names = set()
        for region_entry in doctor_reg_entries:
            company_unit = region_entry["companyUnit"]
            reg_id = region_entry["region"]
            region_name_val = region_map.get(reg_id, f"[ID {reg_id}]")
            region_names.add(region_name_val)
            # --- Запрашиваем расписание ---
            schedule_url = (
                f"{base_url}/doctorSchedule?doctor={doctor_id}&companyUnit={company_unit}&region={reg_id}"
                f"&startDate={start_date}&endDate={end_date}"
            )
            try:
                schedule_resp = _session_get(schedule_url)
                schedule_resp.raise_for_status()
                schedule_days = schedule_resp.json()
            except (requests.RequestException, ValueError):
                continue
            for day in schedule_days:
                # --- Слоты ---
                cells_url = f"{base_url}/doctorScheduleCells?doctorSchedule={day['id']}"
                try:
                    cells_resp = _session_get(cells_url)
                    cells_resp.raise_for_status()
                    cells = cells_resp.json()
                    free_slots = [cell.get("startTime") for cell in cells if isinstance(cell, dict) and cell.get("free")]
                except (requests.RequestException, ValueError):
                    free_slots = []
                schedules_by_region[region_name_val].append({
                    "date": day.get("curDate"),
                    "start": day.get("startTime"),
                    "end": day.get("endTime"),
                    "slots": free_slots
                })

        # Специализация
        spec = None
        for m in doctor_mappings:
            if m.get("specialization"):
                spec = m["specialization"]
                break

        result.append({
            "id": doctor_id,
            "fio": doctor_obj["fio"],
            "specialization": spec,
            "regions": list(region_names),
            "schedule": dict(schedules_by_region)
        })

    if not result:
        return f"Врач с фамилией '{last_name}' не найден."
    return result


if __name__ == "__main__":
    doctors = find_doctor_schedule(
        last_name="Дразнин"
    )
    regions = site_regions()
    print("Регионы:", regions)

    pprint(doctors, width=150)
