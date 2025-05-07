import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import requests
from rapidfuzz import fuzz

from agent_logic_2 import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)

DATA_DIR = Path("apidata/prices")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def get_yesterday_str() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")


def find_existing_price_file(region_id: int) -> Path | None:
    pattern = f"price_{region_id}_*.jsonl"
    files = list(DATA_DIR.glob(pattern))
    if not files:
        return None
    # Берём самый свежий файл по дате в имени
    latest_file = max(files, key=lambda f: f.stem.split("_")[-1])
    return latest_file


def get_date_from_filename(file: Path) -> str:
    return file.stem.split("_")[-1]


def download_price(region_id: int) -> list[dict]:
    url = f"{base_url}/priceByRegion/{region_id}"
    response = requests.get(url, auth=auth, verify=False)
    if not response.ok:
        raise Exception(f"Ошибка запроса к API: {response.status_code} {response.reason}")
    return response.json()


def save_price(region_id: int, data: list[dict]):
    today = get_today_str()
    filename = DATA_DIR / f"price_{region_id}_{today}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Прайс сохранён: {filename}")


def load_price_from_file(file: Path) -> list[dict]:
    with open(file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cleanup_old_prices(region_id: int):
    yesterday = get_yesterday_str()
    pattern = f"price_{region_id}_*.jsonl"
    for file in DATA_DIR.glob(pattern):
        file_date = get_date_from_filename(file)
        if file_date < yesterday:
            print(f"🗑️ Удаляем старый прайс: {file.name}")
            file.unlink()


def get_price(region_id: int) -> list[dict]:
    # Сначала чистим старые прайсы
    cleanup_old_prices(region_id)

    today = get_today_str()
    existing_file = find_existing_price_file(region_id)

    if existing_file:
        file_date = get_date_from_filename(existing_file)
        if file_date == today:
            print(f"✅ Нашли свежий прайс на сегодня для региона {region_id}, используем локальный файл.")
            return load_price_from_file(existing_file)
        else:
            print(f"♻️ Прайс найден, но он от {file_date}. Скачиваем новый.")

    # Либо нет файла, либо он старый
    price_data = download_price(region_id)
    save_price(region_id, price_data)
    print(f"✅ Новый прайс для региона {region_id} успешно загружен.")
    return price_data


# ———————————————————
# Функции поиска услуг по ключевым словам
# Несколько вариантов.
# ———————————————————

def find_services(region_id: int, query: str) -> list[dict]:
    """
    Ищет услуги в прайсе региона по подстроке запроса (без учета регистра).
    Самая базовая версия поиска с использованием оператора in.
    """
    price_list = get_price(region_id)

    query_lower = query.lower()
    matched_services = []

    for service in price_list:
        service_name = service.get("serviceName", "").lower()
        if query_lower in service_name:
            matched_services.append(service)

    return matched_services


def find_services_fuzzy(
        region_id: int,
        queries: list[str],
        synonyms: dict[str, list[str]] | None = None,
        threshold: int = 80
) -> list[dict]:
    """
    Гибкий поиск услуг в прайсе региона:
    - поддержка нескольких ключевых слов
    - разрешение опечаток (порог схожести)
    - учёт синонимов

    :param region_id: ID региона
    :param queries: список поисковых запросов (ключевых слов)
    :param synonyms: словарь синонимов { "ключевое слово": ["синоним1", "синоним2", ...] }
    :param threshold: порог схожести от 0 до 100 (чем ниже, тем больше допускаем ошибку)
    :return: список услуг, подходящих под условия поиска
    """
    price_list = get_price(region_id)

    # Формируем итоговый список поисковых ключей
    all_queries = set()
    for q in queries:
        all_queries.add(q.lower())
        if synonyms and q.lower() in synonyms:
            for syn in synonyms[q.lower()]:
                all_queries.add(syn.lower())

    matched_services = []

    for service in price_list:
        service_name = service.get("serviceName", "").lower()

        for query in all_queries:
            # Фаззи-поиск: если схожесть ≥ threshold, считаем, что подходит
            if fuzz.partial_ratio(query, service_name) >= threshold:
                matched_services.append(service)
                break  # Уже нашли совпадение для этой услуги, переходим к следующей

    return matched_services


def search_services_mode(
        filename: str,
        keywords: List[str],
        mode: str = "any",  # any / all / regex
) -> List[Dict]:
    """Ищет услуги по ключевым словам в указанном файле с использованием различных модов.
    Возможно, функция устарела, использовать не будем
    """
    results = []

    if not os.path.exists(filename):
        print(f"❌ Файл {filename} не найден!")
        return results

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                service = json.loads(line)
                name = service.get("serviceName", "").lower()

                if mode == "any":
                    if any(kw.lower() in name for kw in keywords):
                        results.append(service)
                elif mode == "all":
                    if all(kw.lower() in name for kw in keywords):
                        results.append(service)
                elif mode == "regex":
                    pattern = keywords[0]  # первая строка — паттерн
                    if re.search(pattern, name):
                        results.append(service)
            except Exception as e:
                print(f"Ошибка чтения строки: {e}")
                continue

    return results


# ———————————————————
# Формирование текста каталога для LLM
# ———————————————————

def format_services(services: List[Dict]) -> str:
    """Форматирует услуги для вывода"""
    if not services:
        return "❗ Ничего не найдено."

    lines = []
    for service in services:
        name = service.get("serviceName", "Без названия")
        price = service.get("cost", "Не указана цена")
        lines.append(f"- {name} ({price} руб.)")
    return "\n".join(lines)


# if __name__ == "__main__":
#     region_id = 8805
#     price_list = get_price(region_id)
#
#     for service in price_list:
#         print(service["serviceName"], "-", service["cost"], "₽")

if __name__ == "__main__":
    # 1. Скачиваем прайс на сегодня
    region_id = 8805  # свой регион
    # today_price_file = get_price(region_id)

    # 2. Ищем услуги
    # if today_price_file:

    keywords = ["АЗИ поче", ]  # Смотри как заебато может искать!
    keyword = "УЗИ почек"
    mode = "any"

    found_services_fuz = find_services_fuzzy(region_id=region_id, queries=keywords, )

    found_services = find_services(region_id=region_id, query=keyword, )

    print("\nНайденные услуги:\n")
    print(format_services(found_services))

    print("\nНайденные услуги FUZZ:\n")
    print(format_services(found_services_fuz))
