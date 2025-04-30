import requests
import urllib3
from pprint import pprint
import pandas as pd
from datetime import datetime
import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)


def find_doctors_with_bad_specializations():
    urllib3.disable_warnings()

    # Загружаем данные
    regions = requests.get(f"{base_url}/doctorRegions", auth=auth, verify=False).json()
    region_names = requests.get(f"{base_url}/regions", auth=auth, verify=False).json()
    doctors = requests.get(f"{base_url}/doctors", auth=auth, verify=False).json()
    mappings = requests.get(f"{base_url}/doctorCompanyUnits", auth=auth, verify=False).json()

    # связи врачей с регионами
    region_map = {}
    for r in region_names:
        region_map[r["id"]] = r["name"]

    # Индекс: врач → регионы
    worker_regions = {}
    for r in regions:
        worker_regions.setdefault(r["worker"], set()).add(region_map.get(r["region"], f"[ID {r['region']}]"))

    # Сопоставление worker -> ФИО
    doctor_dict = {doc["id"]: doc["fio"] for doc in doctors}

    # Ключевые слова, которых, блин, не должно быть
    education_keywords = [
        "диплом", "сертификат", "академия", "удостоверение",
        "повышение квалификации", "приказ", "лицензия", "обучение",
        "курс", "университет", "интернатура", "ординатура",
        "присвоено", "аттестат", "зачислена", "зачислен", "переведен", "переведена", "принят", "принята",
        "свидетельство", "серия", "youtube.com", "св-во", "ФГБУЗ", "ПП", "ПК", "аккредитация", "окончила", "окончил"
    ]

    bad_doctors = []
    # Фильтрация
    for entry in mappings:
        spec = entry.get("specialization")
        text = (spec or "").strip().lower()

        reason = None
        if not text or len(text) < 10:
            reason = "☠️ графа 'специализация' не заполнена"
        elif any(kw in text for kw in education_keywords):
            reason = "😖 содержит постороннюю информацию"

        if reason:
            spec_clean = (spec or "").replace('\n', ' ').replace('\r', ' ')
            bad_doctors.append({
                "ID": entry["worker"],
                "ФИО": doctor_dict.get(entry["worker"], "Неизвестный врач"),
                "Регион": worker_regions.get(entry["worker"], set()),
                "Специализация (фрагмент)": ("..." if len(spec_clean) > 100 else "") + spec_clean[-100:],
                "Предполагаемый тип ошибки": reason
            })

    return bad_doctors


if __name__ == "__main__":
    bad_docs = find_doctors_with_bad_specializations()
    df = pd.DataFrame(bad_docs)
    filename = f"bad_specializations_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"✅ CSV-файл сохранён: {filename}")
    print("=================================")
    print("то же самое на экран:")
    print("=================================")

    pprint(bad_docs, width=160)
