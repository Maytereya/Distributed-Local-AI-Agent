import requests
from datetime import date, timedelta
import urllib3
from pprint import pprint
from collections import defaultdict

import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)


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
