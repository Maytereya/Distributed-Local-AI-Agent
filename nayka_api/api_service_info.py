#

import requests
import json
from pathlib import Path
from pprint import pprint

import config as c  # если нужен auth

url = "https://tc.naykalab.ru:444/H8PdIkzEjteo5ZPvVwt29t4TVjf0XN1K/medserver-test/api/v1/site/serviceInfoAll"
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)  # или None, если доступ без авторизации

output_path = Path("apidata/preparations/filtered_service_info.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

def fetch_and_filter_service_info():
    response = requests.get(url, auth=auth, verify=False)
    if not response.ok:
        print(f"Ошибка при запросе: {response.status_code}")
        print(response.text)
        return

    try:
        data = response.json()
    except Exception as e:
        print("Ошибка при разборе JSON:", e)
        print(response.text)
        return

    # filtered = [
    #     item for item in data
    #     if any(item.get(field) for field in ["description", "indication", "preparation"])
    # ]

    filtered = []
    for item in data:
        values = [item.get(f) for f in ["description", "indication", "preparation"]]
        if any(v and v.strip() for v in values):
            filtered.append(item)
        else:
            # временно покажем записи без полезной информации
            print(f"❌ Пропущено: {item.get('serviceName')}")


    # Запись в jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in filtered:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # Печать на экран
    pprint(filtered, width=160)
    print(f"\n✅ Отфильтровано: {len(filtered)} записей. Сохранено в {output_path}")

if __name__ == "__main__":
    fetch_and_filter_service_info()