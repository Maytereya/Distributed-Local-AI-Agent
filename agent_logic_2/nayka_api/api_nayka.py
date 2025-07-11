import json
import os
import sys
from collections import defaultdict
from datetime import timedelta, datetime, date
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Set

import requests

from agent_logic_2 import config as c

# Добавляем корневую директорию в PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)

# Директория для кэширования данных
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "apidata"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_today_str() -> str:
    """Возвращает текущую дату в формате YYYYMMDD"""
    return datetime.now().strftime("%Y%m%d")


def get_yesterday_str() -> str:
    """Возвращает вчерашнюю дату в формате YYYYMMDD"""
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def find_existing_doctors_file() -> Path or None:
    """Находит самый свежий файл с данными о врачах"""
    pattern = "doctors_*.jsonl"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: f.stem.split("_")[-1])


def get_date_from_filename(file: Path) -> str:
    """Извлекает дату из имени файла"""
    return file.stem.split("_")[-1]


def save_doctors_data(doctors: list):
    """Сохраняет список врачей в формате JSONL — по одному врачу на строку."""
    # Очищаем старые файлы перед сохранением нового
    cleanup_old_doctors_files()

    today = get_today_str()
    filename = DATA_DIR / f"doctors_{today}.jsonl"
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


def cleanup_old_doctors_files():
    """Удаляет все файлы с данными о врачах, кроме сегодняшнего"""
    today = get_today_str()
    pattern = "doctors_*.jsonl"
    for file in DATA_DIR.glob(pattern):
        file_date = get_date_from_filename(file)
        if file_date != today:
            print(f"🗑️ Удаляем файл с данными о врачах: {file.name}")
            file.unlink()


def get_all_doctors() -> List[Dict]:
    """
    Получает список всех врачей с их данными, где regions и region_ids совпадают по позиции.
    """
    # Получаем все данные через API
    doctors = site_doctors()
    units = site_company_units()
    doctor_units = site_doctor_company_units()
    doctor_regions = site_doctor_regions()
    regions = site_regions()

    # Быстрый доступ к названиям регионов по id
    regions_dict = {r["id"]: r["name"] for r in regions}
    units_dict = {u["id"]: u["name"] for u in units}

    result = []
    for doctor in doctors:
        doctor_id = doctor["id"]

        # Все specialization
        specs = [
            link.get("specialization", "") or ""
            for link in doctor_units
            if link["worker"] == doctor_id
        ]
        specs = list(dict.fromkeys(filter(None, specs)))  # Сохраняем порядок, убираем дубли

        # Все подразделения врача
        doc_units = [
            units_dict.get(link["companyUnit"], "")
            for link in doctor_units
            if link["worker"] == doctor_id
        ]
        doc_units = list(dict.fromkeys(filter(None, doc_units)))

        # Готовим пары регионов (id, name) — без дублей, с сохранением порядка
        seen_region_ids = set()
        region_pairs = []
        for link in doctor_regions:
            if link["worker"] == doctor_id:
                reg_id = link["region"]
                if reg_id and reg_id not in seen_region_ids:
                    seen_region_ids.add(reg_id)
                    reg_name = regions_dict.get(reg_id)
                    if reg_name:
                        region_pairs.append((reg_id, reg_name))
                    else:
                        region_pairs.append((reg_id, f"ID {reg_id}"))

        doc_region_ids = [r[0] for r in region_pairs]
        doc_regions = [r[1] for r in region_pairs]

        doctor_data = {
            "id": doctor_id,
            "fio": doctor["fio"],
            "specialization": specs[0] if specs else None,
            "regions": doc_regions,
            "region_ids": doc_region_ids,
            "units": doc_units
        }

        result.append(doctor_data)

    return result


def get_cached_doctors_data() -> list:
    """
    Возвращает кэшированные данные о врачах, если кэш свежий.
    Если кэш устарел или отсутствует — загружает новые данные через API и сохраняет их в кэш.
    """
    cleanup_old_doctors_files()
    today = get_today_str()
    existing_file = find_existing_doctors_file()

    print(f"[DEBUG] Сегодня: {today}")
    print(f"[DEBUG] Найден файл: {existing_file}")

    # Если есть актуальный кэш — используем его
    if existing_file:
        file_date = get_date_from_filename(existing_file)
        print(f"[DEBUG] Дата файла: {file_date}")

        if file_date == today:
            print("✅ Нашли свежие данные о врачах на сегодня (используем кэш)")
            return load_doctors_data(existing_file)
        else:
            print(f"♻️ Данные найдены, но они от {file_date}. Скачиваем новые.")

    # Если кэша нет или он устарел — обновляем через API
    print("[DEBUG] Кэш не найден или устарел — обновляем через API!")
    doctors = get_all_doctors()  # Собирает всё как надо (regions/region_ids и т.д.)
    save_doctors_data(doctors)
    print("✅ Новые данные о врачах успешно загружены")
    return doctors


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


def find_doctors_by_keyword(keyword: str) -> List[Dict] | str:
    """
    Ищем по:
      • тексту specialization    («кардиолог», «ультразвук»)  
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
        specs = [
            link.get("specialization", "") or ""
            for link in links
            if link["worker"] == wid
        ]
        specs = list(set(filter(None, specs)))

        doc = {
            "id": wid,
            "fio": doctors.get(wid, f"[id {wid}]"),
            "specialization": specs[0] if specs else "Специализация не указана",
            "directions": [s for s in specs[1:]],
            "regions": base_doc.get("regions", ["-"]) if base_doc else ["-"],
        }
        result.append(doc)

    return result


def site_company_units():
    """Получить список подразделений."""
    response = requests.get(f"{base_url}/companyUnits", auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка при получении списка подразделений: {response.status_code}")
        return []
    try:
        return response.json()
    except Exception as e:
        print(f"Ошибка при разборе JSON: {e}")
        return []


def site_doctors():
    """Получить список врачей."""
    response = requests.get(f"{base_url}/doctors", auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка при получении списка врачей: {response.status_code}")
        return []
    try:
        return response.json()
    except Exception as e:
        print(f"Ошибка при разборе JSON: {e}")
        return []


def site_doctor_company_units():
    """Получить связи врачей с подразделениями."""
    response = requests.get(f"{base_url}/doctorCompanyUnits", auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка при получении связей: {response.status_code}")
        return []
    try:
        return response.json()
    except Exception as e:
        print(f"Ошибка при разборе JSON: {e}")
        return []


def site_doctor_regions():
    """Получить связи врачей с регионами."""
    response = requests.get(f"{base_url}/doctorRegions", auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка при получении связей: {response.status_code}")
        return []
    try:
        return response.json()
    except Exception as e:
        print(f"Ошибка при разборе JSON: {e}")
        return []


def site_regions():
    """Получить список регионов."""
    response = requests.get(f"{base_url}/regions", auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка при получении списка регионов: {response.status_code}")
        print(f"Ответ: {response.text}")
        return []
    try:
        return response.json()
    except Exception as e:
        print(f"Ошибка при разборе JSON: {e}")
        return []


def find_doctor_schedule(
        last_name: str,
        region_name: str = None  # теперь не обязательно
):
    """
    Ищем по:
      • тексту fio (фамилия, имя, отчество)  («Смирнова», «Иванов Иван»)
    Возвращаем краткий список врачей.
    """
    # --- Получаем регионы ---
    regions = site_regions()
    region_map = {r["id"]: r["name"] for r in regions}
    region_id = None
    if region_name:
        region_id = next((r["id"] for r in regions if region_name.lower() in r["name"].lower()), None)
        if not region_id:
            return f"Регион '{region_name}' не найден."

    # --- Получаем врачей ---
    doctors = requests.get(f"{base_url}/doctors", auth=auth, verify=False).json()
    doctor_dict = {doc["id"]: doc for doc in doctors}
    matched_doctors = {doc["id"]: doc for doc in doctors if last_name.lower() in doc["fio"].lower()}
    if not matched_doctors:
        return f"Врач с фамилией (или частью ФИО) '{last_name}' не найден."

    # --- Получаем companyUnit и doctorRegions ---
    mappings = requests.get(f"{base_url}/doctorCompanyUnits", auth=auth, verify=False).json()
    dr_regions = requests.get(f"{base_url}/doctorRegions", auth=auth, verify=False).json()

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
            schedule_resp = requests.get(schedule_url, auth=auth, verify=False)
            if not schedule_resp.ok:
                continue
            try:
                schedule_days = schedule_resp.json()
            except Exception:
                continue
            for day in schedule_days:
                # --- Слоты ---
                cells_url = f"{base_url}/doctorScheduleCells?doctorSchedule={day['id']}"
                cells_resp = requests.get(cells_url, auth=auth, verify=False)
                if cells_resp.ok:
                    try:
                        cells = cells_resp.json()
                        free_slots = [cell["startTime"] for cell in cells if cell.get("free")]
                    except Exception:
                        free_slots = []
                else:
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
