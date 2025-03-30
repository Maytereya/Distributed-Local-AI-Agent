import requests
from datetime import date, timedelta
import urllib3
from pprint import pprint
import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)


def find_doctor_schedule(
        last_name: str,
        # specialty: str,
        region_name: str):
    urllib3.disable_warnings()

    # Получаем список регионов
    regions = requests.get(f"{base_url}/regions", auth=auth, verify=False).json()
    region_id = next((r["id"] for r in regions if region_name.lower() in r["name"].lower()), None)

    if not region_id:
        return f"Регион '{region_name}' не найден."

    # Получаем врачей
    doctors = requests.get(f"{base_url}/doctors", auth=auth, verify=False).json()
    doctor_dict = {doc["id"]: doc["fio"] for doc in doctors}

    # Фильтрация по фамилии
    # matched_ids = [doc_id for doc_id, fio in doctor_dict.items() if last_name.lower() in fio.lower()]
    # if not matched_ids:
    #     return f"Врач с фамилией '{last_name}' не найден."

    # Фильтрация врачей по вхождению фамилии (или части ФИО)
    matched_doctors = {
        doc_id: fio for doc_id, fio in doctor_dict.items()
        if last_name.lower() in fio.lower()
    }

    print("matched_doctors: ", matched_doctors)

    if not matched_doctors:
        return f"Врач с фамилией (или частью ФИО) '{last_name}' не найден."

    # Получаем специализации
    mappings = requests.get(f"{base_url}/doctorCompanyUnits", auth=auth, verify=False).json()

    # Получаем doctorRegions
    dr_regions = requests.get(f"{base_url}/doctorRegions", auth=auth, verify=False).json()

    # Даты для расписания
    start_date = date.today().isoformat()
    end_date = (date.today() + timedelta(days=7)).isoformat()

    result = []

    # for doctor_id in matched_ids:
    for doctor_id in matched_doctors:
        doctor_mappings = [m for m in mappings if m["worker"] == doctor_id]

        matched_in_region = [
            m for m in doctor_mappings
            if any(r["worker"] == doctor_id and r["region"] == region_id for r in dr_regions)
        ]

        if not matched_in_region:
            continue

        for item in matched_in_region:
            company_unit = item["companyUnit"]

            schedule_url = (
                f"{base_url}/doctorSchedule?doctor={doctor_id}&companyUnit={company_unit}&region={region_id}"
                f"&startDate={start_date}&endDate={end_date}"
            )
            schedule = requests.get(schedule_url, auth=auth, verify=False).json()

            full_schedule = []
            for day in schedule:
                cells_url = f"{base_url}/doctorScheduleCells?doctorSchedule={day['id']}"
                response = requests.get(cells_url, auth=auth, verify=False)
                try:
                    cells = response.json()
                    free_slots = [cell["startTime"] for cell in cells if cell.get("free")]
                except Exception:
                    free_slots = []

                full_schedule.append({
                    "date": day["curDate"],
                    "start": day["startTime"],
                    "end": day["endTime"],
                    "slots": free_slots
                })

            result.append({
                "id": doctor_id,
                "fio": doctor_dict[doctor_id],
                "specialization": item.get("specialization", ""),
                "region": region_name,
                "schedule": full_schedule
            })

    return result or f"Врач с фамилией '{last_name}', и местом работы '{region_name}' не найден."


if __name__ == "__main__":
    doctors = find_doctor_schedule(
        last_name="Калашни",
        # specialty="Врач ультразвуковой диагностики",
        region_name="Ленина 5"
    )

    pprint(doctors, width=150)
