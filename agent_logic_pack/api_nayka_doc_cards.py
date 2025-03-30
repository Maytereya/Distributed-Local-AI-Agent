import requests
import urllib3

# Конфигурация API
import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)


def print_doctor_cards(
        show_specialization: bool = True,
        filter_by_unit: str = None,
        filter_by_region: str = None
):
    urllib3.disable_warnings()

    # Получаем список врачей
    doctors_url = f"{base_url}/doctors"
    doctors = requests.get(doctors_url, auth=auth, verify=False).json()
    doctor_dict = {doc['id']: doc for doc in doctors}

    # Получаем связи врачей с отделениями
    dc_url = f"{base_url}/doctorCompanyUnits"
    doctor_company = requests.get(dc_url, auth=auth, verify=False).json()

    # Получаем связи врачей с регионами
    region_map = {}
    regions = requests.get(f"{base_url}/doctorRegions", auth=auth, verify=False).json()
    region_names = requests.get(f"{base_url}/regions", auth=auth, verify=False).json()
    for r in region_names:
        region_map[r["id"]] = r["name"]

    # Индекс: врач → регионы
    worker_regions = {}
    for r in regions:
        worker_regions.setdefault(r["worker"], set()).add(region_map.get(r["region"], f"[ID {r['region']}]"))

    # Получаем названия направлений
    units = requests.get(f"{base_url}/companyUnits", auth=auth, verify=False).json()
    unit_map = {u["id"]: u["name"] for u in units}
    name_to_unit_id = {v: k for k, v in unit_map.items()}  # Для фильтрации по названию

    for entry in doctor_company:
        worker_id = entry["worker"]
        fio = doctor_dict.get(worker_id, {}).get("fio", "Неизвестный врач")
        specialization_raw = entry.get("specialization")
        specialization = specialization_raw.strip() if specialization_raw else ""
        doctor_regions = worker_regions.get(worker_id, set())

        unit_id = entry.get("companyUnit")
        unit_name = unit_map.get(unit_id, f"[ID {unit_id}]")

        # Фильтрация по направлению
        if filter_by_unit and unit_name != filter_by_unit:
            continue

        # Фильтрация по региону
        if filter_by_region and filter_by_region not in doctor_regions:
            continue

        print(f"👨‍⚕️ {fio}")
        print(f"🏥 Направление: {unit_name}")
        if doctor_regions:
            print(f"📍 Регион(ы): {', '.join(doctor_regions)}")

        if show_specialization:
            if specialization:
                print("🩺 Специализация:")
                for line in specialization.splitlines():
                    line = line.strip(" -•\t\r\n")
                    if line:
                        print(f"  - {line}")
            else:
                print("🩺 Специализация: — (не указана)")
        print("-" * 60)


if __name__ == "__main__":
    print_doctor_cards(
        show_specialization=False,
        filter_by_unit="Врач ультразвуковой диагностики",  # <- Укажи нужное направление или None
        filter_by_region="Ленина 5"  # <- Укажи нужный регион или None
    )
