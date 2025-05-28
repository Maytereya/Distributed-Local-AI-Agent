import json
import os
import sys
from datetime import timedelta, datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Set, Union

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
    """Удаляет старые файлы с данными о врачах"""
    yesterday = get_yesterday_str()
    pattern = "doctors_*.json"
    for file in DATA_DIR.glob(pattern):
        file_date = get_date_from_filename(file)
        if file_date < yesterday:
            print(f"🗑️ Удаляем старый файл с данными о врачах: {file.name}")
            file.unlink()


def get_cached_doctors_data() -> list:
    cleanup_old_doctors_files()
    today = get_today_str()
    existing_file = find_existing_doctors_file()

    print(f"[DEBUG] Сегодня: {today}")
    print(f"[DEBUG] Найден файл: {existing_file}")

    if existing_file:
        file_date = get_date_from_filename(existing_file)
        print(f"[DEBUG] Дата файла: {file_date}")

        if file_date == today:
            print("✅ Нашли свежие данные о врачах на сегодня (используем кэш)")
            return load_doctors_data(existing_file)
        print(f"♻️ Данные найдены, но они от {file_date}. Скачиваем новые.")

    # Собираем все данные с API и формируем список врачей с нужными полями
    print("[DEBUG] Кэш не найден или устарел — обновляем через API!")
    doctors_raw = site_doctors()
    units = site_company_units()
    doctor_units = site_doctor_company_units()
    doctor_regions = site_doctor_regions()
    regions = site_regions()

    units_dict = {u["id"]: u["name"] for u in units}
    regions_dict = {r["id"]: r["name"] for r in regions}

    doctors = []
    for doctor in doctors_raw:
        doctor_id = doctor["id"]
        # Получаем специализации врача
        specs = [
            link.get("specialization", "") or ""
            for link in doctor_units
            if link["worker"] == doctor_id
        ]
        specs = list(set(filter(None, specs)))
        # Получаем подразделения врача
        doc_units = [
            units_dict.get(link["companyUnit"], "")
            for link in doctor_units
            if link["worker"] == doctor_id
        ]
        doc_units = list(set(filter(None, doc_units)))
        # Получаем регионы врача
        doc_regions = [
            regions_dict.get(link["region"], "")
            for link in doctor_regions
            if link["worker"] == doctor_id
        ]
        doc_regions = list(set(filter(None, doc_regions)))
        # Формируем итоговый словарь
        doctor_data = {
            "id": doctor_id,
            "fio": doctor["fio"],
            "specialization": specs[0] if specs else None,
            "regions": doc_regions,
            "units": doc_units
        }
        doctors.append(doctor_data)

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


def find_doctors_by_keyword(keyword: str) -> Union[List[Dict], str]:
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

    # Остальная логика поиска остается без изменений
    matched_workers = set()

    for link in links:
        spec = link.get("specialization", "") or ""
        spec_lower = spec.lower()
        unit = unit_name_by_id.get(link["companyUnit"], "").lower()

        is_match = False

        # Проверяем подразделение
        if kw in unit:
            is_match = True
        # Проверяем специализацию
        elif kw in spec_lower:
            # Проверяем, что это не упоминание в контексте других специальностей
            if not any(other in spec_lower for other in [
                "ультразвуковая", "функциональная", "терапевт",
                "в ревматологии", "по ревматологии", "ревматологический"
            ]):
                is_match = True

        if is_match:
            matched_workers.add(link["worker"])

    if not matched_workers:
        return f"Врачей по ключу «{keyword}» не найдено."

    result = []
    for wid in matched_workers:
        # Получаем все специализации врача
        specs = [
            link.get("specialization", "") or ""
            for link in links
            if link["worker"] == wid
        ]
        # Убираем дубликаты и пустые строки
        specs = list(set(filter(None, specs)))

        if not specs:
            result.append(f"{doctors.get(wid, f'[id {wid}]')} - Специализация не указана")
            continue

        # Формируем вывод
        doctor_info = [f"{doctors.get(wid, f'[id {wid}]')}"]

        # Добавляем основную специализацию
        main_spec = specs[0].split(":")[0].split("-")[0].strip()
        if main_spec:
            doctor_info.append(f"Специализация: {main_spec}")

        # Добавляем направления работы, если есть
        if len(specs) > 1:
            directions = [s.split(":")[0].strip() for s in specs[1:3]]  # Берем максимум 2 направления
            if directions:
                doctor_info.append(f"Направления: {', '.join(directions)}")

        result.append("\n".join(doctor_info))

    return "\n\n".join(result)


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
    print(f"\nИщем врача по фамилии: {last_name}")
    if region_name:
        print(f"В регионе: {region_name}")

    # Загружаем кэшированные данные
    data = get_cached_doctors_data()

    # Распаковываем данные
    units = site_company_units()
    unit_name_by_id = {u["id"]: u["name"] for u in units}
    links = site_doctor_company_units()
    doctors = data
    regions = site_regions()
    doctor_regions = site_doctor_regions()

    # Получаем ID региона, если указан
    region_id = None
    if region_name:
        region_name = region_name.lower()
        for r in regions:
            if region_name in r["name"].lower():
                region_id = r["id"]
                print(f"Найден регион: {r['name']} (ID: {r['id']})")
                break
        if not region_id:
            print(f"Регион '{region_name}' не найден")
            return f"Регион «{region_name}» не найден."

    # Ищем врачей по фамилии
    matched_workers = set()
    last_name = last_name.lower()

    print(f"Ищем врачей с фамилией '{last_name}'...")
    for doctor in doctors:
        fio = doctor["fio"].lower()
        if last_name in fio:
            print(f"Найден врач: {doctor['fio']} (ID: {doctor['id']})")
            matched_workers.add(doctor["id"])

    if not matched_workers:
        print(f"Врачей с фамилией '{last_name}' не найдено")
        return f"Врачей с фамилией «{last_name}» не найдено."

    # Фильтруем по региону, если указан
    if region_id:
        print(f"Фильтруем врачей по региону {region_id}...")
        filtered_workers = set()
        for dr in doctor_regions:
            if dr["worker"] in matched_workers and dr["region"] == region_id:
                filtered_workers.add(dr["worker"])
        if not filtered_workers:
            print(f"В регионе {region_id} нет врачей с фамилией '{last_name}'")
            return f"В регионе «{region_name}» нет врачей с фамилией «{last_name}»."
        matched_workers = filtered_workers

    # Собираем информацию о врачах
    result = []
    for wid in matched_workers:
        # Получаем все специализации врача
        specs = [
            link.get("specialization", "") or ""
            for link in links
            if link["worker"] == wid
        ]
        # Убираем дубликаты и пустые строки
        specs = list(set(filter(None, specs)))

        # Получаем информацию о враче
        doctor = next((d for d in doctors if d["id"] == wid), None)
        if not doctor:
            print(f"Не найдена информация о враче с ID {wid}")
            continue

        if not specs:
            result.append(f"{doctor['fio']} - Специализация не указана")
            continue

        # Формируем вывод
        doctor_info = [f"{doctor['fio']}"]

        # Добавляем основную специализацию
        main_spec = specs[0].split(":")[0].split("-")[0].strip()
        if main_spec:
            doctor_info.append(f"Специализация: {main_spec}")

        # Добавляем направления работы, если есть
        if len(specs) > 1:
            directions = [s.split(":")[0].strip() for s in specs[1:3]]  # Берем максимум 2 направления
            if directions:
                doctor_info.append(f"Направления: {', '.join(directions)}")

        result.append("\n".join(doctor_info))

    return "\n\n".join(result)


def get_all_doctors() -> List[Dict]:
    """
    Получает список всех врачей с их данными.
    Использует кэшированные данные, если они есть.
    
    Returns:
        List[Dict]: Список словарей с данными о врачах
    """
    # Получаем все данные через API
    doctors = site_doctors()
    units = site_company_units()
    doctor_units = site_doctor_company_units()
    doctor_regions = site_doctor_regions()
    regions = site_regions()

    # Создаем словари для быстрого поиска
    units_dict = {u["id"]: u["name"] for u in units}
    regions_dict = {r["id"]: r["name"] for r in regions}

    result = []
    for doctor in doctors:
        doctor_id = doctor["id"]

        # Получаем специализации врача
        specs = [
            link.get("specialization", "") or ""
            for link in doctor_units
            if link["worker"] == doctor_id
        ]
        specs = list(set(filter(None, specs)))

        # Получаем подразделения врача
        doc_units = [
            units_dict.get(link["companyUnit"], "")
            for link in doctor_units
            if link["worker"] == doctor_id
        ]
        doc_units = list(set(filter(None, doc_units)))

        # Получаем регионы врача
        doc_regions = [
            regions_dict.get(link["region"], "")
            for link in doctor_regions
            if link["worker"] == doctor_id
        ]
        doc_regions = list(set(filter(None, doc_regions)))

        # Формируем итоговый словарь
        doctor_data = {
            "id": doctor_id,
            "fio": doctor["fio"],
            "specialization": specs[0] if specs else None,
            "regions": doc_regions,
            "units": doc_units
        }

        result.append(doctor_data)

    return result


if __name__ == "__main__":
    doctors = find_doctor_schedule(
        last_name="Дразнин"
    )

    pprint(doctors, width=150)
