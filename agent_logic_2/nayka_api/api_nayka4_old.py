import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from pprint import pprint
from typing import Dict, List, Set, Union

from agent_logic_2 import config as c
import requests
import urllib3

# Добавляем корневую директорию в PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)


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
      • тексту specialization («кардиолог», «ультразвук»)
    Возвращаем краткий список врачей.
    """
    kw = keyword.lower()
    print(f"\nИщем врачей по ключевому слову: {kw}")

    # 1. Грузим все нужные таблицы
    units = site_company_units()  # дерево подразделений
    unit_name_by_id = {u["id"]: u["name"] for u in units}

    links = site_doctor_company_units()
    doctors = {d["id"]: d["fio"] for d in site_doctors()}

    # 2. Собираем id-врачей по специализации
    matched_workers = set()

    print("\nПроверяем специализации врачей:")
    for link in links:
        spec = link.get("specialization", "") or ""
        spec_lower = spec.lower()
        unit = unit_name_by_id.get(link["companyUnit"], "").lower()

        print(f"\nВрач {doctors.get(link['worker'], 'Неизвестный')}:")
        print(f"Специализация: {spec}")
        print(f"Подразделение: {unit_name_by_id.get(link['companyUnit'], '-')}")

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
            print("✓ Найдено совпадение")
            matched_workers.add(link["worker"])
        else:
            print("✗ Нет совпадения")

    if not matched_workers:
        return f"Врачей по ключу «{keyword}» не найдено."

    result = [
        {
            "id": wid,
            "fio": doctors.get(wid, f"[id {wid}]"),
            "unit": unit_name_by_id.get(
                next(
                    (link["companyUnit"] for link in links
                     if link["worker"] == wid),
                    None
                ),
                "-"
            ),
            "specialization": next(
                (link.get("specialization", "") or "" for link in links
                 if link["worker"] == wid),
                "-"
            )
        }
        for wid in matched_workers
    ]

    print("\nИтоговый результат:")
    for doc in result:
        print(f"\nВрач: {doc['fio']}")
        print(f"Подразделение: {doc['unit']}")
        print(f"Специализация: {doc['specialization']}")

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
    urllib3.disable_warnings()

    # Загружаем регионы
    regions_response = requests.get(f"{base_url}/regions", auth=auth, verify=False)

    if not regions_response.ok:
        print(f"Ошибка при загрузке адресов больниц: {regions_response.status_code}")
        print(regions_response.text)
        return f"Ошибка загрузки адресов больниц с сервера."

    try:
        regions = regions_response.json()
    except Exception as e:
        print("Ошибка при разборе JSON с адресами больниц:", e)
        print("Ответ сервера:", regions_response.text)
        return f"Ошибка при обработке данных об адресах."

    region_map = {r["id"]: r["name"] for r in regions}

    print("Регионы: ")
    print(region_map)

    region_id = None
    if region_name:
        region_id = next((r["id"] for r in regions if region_name.lower() in r["name"].lower()), None)
        if not region_id:
            return f"Регион '{region_name}' не найден."

    # Получаем врачей
    # doctors = requests.get(f"{base_url}/doctors", auth=auth, verify=False).json() # не обрабатывает ошибку доступа

    response = requests.get(f"{base_url}/doctors", auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка запроса: {response.status_code} {response.reason}")
        print(f"Тело ответа: {response.text}")
        return []

    try:
        doctors = response.json()
    except Exception as e:
        print(f"Ошибка парсинга JSON: {e}")
        print(f"Ответ сервера: {response.text}")
        return []

    doctor_dict = {doc["id"]: doc["fio"] for doc in doctors}

    # Фильтрация врачей по фамилии или части ФИО
    matched_doctors = {
        doc_id: fio for doc_id, fio in doctor_dict.items()
        if last_name.lower() in fio.lower()
    }

    if not matched_doctors:
        return f"Врач с фамилией (или частью ФИО) '{last_name}' не найден."

    # Загружаем остальные таблицы
    mappings = requests.get(f"{base_url}/doctorCompanyUnits",
                            auth=auth,
                            verify=False)

    if not mappings.ok:
        print(f"Ошибка {mappings.status_code} при обращении к {base_url}/doctorCompanyUnits")
        return []
    try:
        jsoned_mappings = mappings.json()
    except Exception as e:
        print("Ошибка при разборе JSON:", e)
        print("Ответ сервера:", mappings.text)
        return f"Ошибка при обработке данных о специализациях врачей."

    dr_regions = requests.get(f"{base_url}/doctorRegions",
                              auth=auth,
                              verify=False)

    if not dr_regions.ok:
        print(f"Ошибка {dr_regions.status_code} при обращении к {base_url}/doctorRegions")
        return []
    try:
        jsoned_dr_regions = dr_regions.json()
    except Exception as e:
        print("Ошибка при разборе JSON:", e)
        print("Ответ сервера:", dr_regions.text)
        return f"Ошибка при обработке данных о сопоставлении клиник и врачей."

    start_date = date.today().isoformat()
    end_date = (date.today() + timedelta(days=7)).isoformat()

    result = []

    for doctor_id in matched_doctors:
        doctor_mappings = [m for m in jsoned_mappings if m["worker"] == doctor_id]
        doctor_regions = [r for r in jsoned_dr_regions if r["worker"] == doctor_id]

        if region_id:
            doctor_regions = [r for r in doctor_regions if r["region"] == region_id]

        if not doctor_regions:
            continue

        for region_entry in doctor_regions:
            company_unit = region_entry["companyUnit"]
            reg_id = region_entry["region"]

            schedule_url = (
                f"{base_url}/doctorSchedule?doctor={doctor_id}&companyUnit={company_unit}&region={reg_id}"
                f"&startDate={start_date}&endDate={end_date}"
            )
            # schedule = requests.get(schedule_url, auth=auth, verify=False).json()
            schedule_response = requests.get(schedule_url, auth=auth, verify=False)

            if not schedule_response.ok:
                print(f"+ Ошибка при получении расписания: {schedule_response.status_code}")
                return []
            try:
                schedule = schedule_response.json()
            except Exception as e:
                print("Ошибка при парсинге расписания:", e)
                print("Тело ответа:", schedule_response.text)
                continue

            full_schedule = []

            for day in schedule:

                cells_url = f"{base_url}/doctorScheduleCells?doctorSchedule={day['id']}"

                # response = requests.get(cells_url, auth=auth, verify=False) # Без проверки на ошибки

                response = requests.get(cells_url, auth=auth, verify=False)

                if not response.ok:
                    print(f"⚠️ Ошибка загрузки слотов расписания врача. Код: {response.status_code}, URL: {cells_url}")
                    free_slots = []
                else:
                    try:
                        cells = response.json()
                        free_slots = [cell["startTime"] for cell in cells if cell.get("free")]
                    except Exception as e:
                        print("+ Ошибка при разборе JSON +:", e)
                        print("Ответ сервера:", response.text)
                        free_slots = []

                # try:
                #     cells = response.json()
                #     free_slots = [cell["startTime"] for cell in cells if cell.get("free")]
                #
                # except Exception as e:
                #     free_slots = []
                #     print("+ Ошибка при разборе JSON +:", e)

                full_schedule.append({
                    "date": day["curDate"],
                    "start": day["startTime"],
                    "end": day["endTime"],
                    "slots": free_slots
                })

            specialization = next(
                (m.get("specialization", "") for m in doctor_mappings if m["companyUnit"] == company_unit), ""
            )

            result.append({
                "id": doctor_id,
                "fio": doctor_dict[doctor_id],
                "specialization": specialization,
                "region": region_map.get(reg_id, f"[ID {reg_id}]"),
                "schedule": full_schedule
            })

    # -------------------------------------------------------------
    # Мерджинг карточек врача, если он ведет прием в разных клиниках.
    # ------------------------------------------------------------

    merged = {}

    for doc in result:
        doc_id = doc["id"]
        region = doc["region"]

        if doc_id not in merged:
            merged[doc_id] = {
                "id": doc["id"],
                "fio": doc["fio"],
                "specialization": doc["specialization"],
                "regions": set(),
                "schedules_by_region": defaultdict(list)
            }

        merged[doc_id]["regions"].add(region)
        merged[doc_id]["schedules_by_region"][region].extend(doc["schedule"])

    # Преобразуем в финальный вид
    final_result = []
    for item in merged.values():
        final_result.append({
            "id": item["id"],
            "fio": item["fio"],
            "specialization": item["specialization"],
            "regions": list(item["regions"]),
            "schedule": dict(item["schedules_by_region"])
        })

    return final_result or f"Врач с фамилией '{last_name}' не найден."


if __name__ == "__main__":
    doctors = find_doctor_schedule(
        last_name="Белохвостикова"
    )

    pprint(doctors, width=150)
