import re
import json
import requests
from datetime import datetime
from pathlib import Path

from agent_logic_2 import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)

PRICE_DIR = Path("apidata/prices")
PRICE_DIR.mkdir(parents=True, exist_ok=True)

url = f"{base_url}/priceAll"

def get_today_str():
    return datetime.now().strftime("%Y%m%d")

def get_priceall_filename(date=None):
    if date is None:
        date = get_today_str()
    return PRICE_DIR / f"price_all_{date}.jsonl"

def cleanup_old_priceall(days_to_keep=2):
    """Удаляет кэши старше N дней"""
    files = list(PRICE_DIR.glob("price_all_*.jsonl"))
    for f in files:
        date_str = f.stem.rsplit('_', 1)[-1]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            continue
        if (datetime.now() - file_date).days >= days_to_keep:
            print(f"🗑️ Удаляем устаревший кэш: {f}")
            f.unlink()

def update_price_all():
    """Скачивает priceAll если файла на сегодня нет"""
    fn = get_priceall_filename()
    if fn.exists():
        print("✅ priceAll на сегодня уже скачан")
        return
    print("⏬ Скачиваем priceAll...")
    resp = requests.get(url, auth=auth, verify=False)
    data = resp.json()
    with open(fn, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ priceAll обновлен ({len(data)} строк)")
    cleanup_old_priceall(days_to_keep=2)

def load_price_all():
    fn = get_priceall_filename()
    if not fn.exists():
        update_price_all()
    result = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result.append(json.loads(line))
            except Exception as e:
                print(f"[ERROR] Не удалось распарсить строку: {line[:100]}... Ошибка: {e}")
    return result

def find_services(region_id: int, query: str) -> list[dict]:
    """Ищет услуги по региону и подстроке"""
    all_prices = load_price_all()
    q = query.lower()
    return [row for row in all_prices if row["regionId"] == region_id and q in row["serviceName"].lower()]


def get_region_id_by_name_all(region_name: str, price_all: list[dict]) -> int | None:

    def normalize(name: str) -> str:
        name = name.lower()
        name = re.sub(r'\(.*?\)', '', name)
        name = name.replace('клиника', '')
        name = name.replace('ул.', '')
        name = name.replace('д.', '')
        name = re.sub(r'[^а-яa-z0-9 ]+', '', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        return name

    target = normalize(region_name)
    print(f"\n[DEBUG] region_name = '{region_name}' -> target norm = '{target}'")
    region_counter = {}
    for item in price_all:
        rid = item.get('regionId')
        rname = item.get('regionName', '') or item.get('region', '')
        if not rname or not rid:
            continue
        rname_norm = normalize(rname)
        # Если вообще совпадает хоть как-то — печатай для анализа
        if target in rname_norm or rname_norm in target:
            print(f"  [MATCH] region: '{rname}' (norm: '{rname_norm}'), id: {rid}")
            region_counter[rid] = region_counter.get(rid, 0) + 1
    if region_counter:
        print(f"  [OK] Совпавшие region_ids: {region_counter}")
        return max(region_counter, key=region_counter.get)
    print("  [FAIL] Ничего не нашли")
    # Показать все уникальные варианты адресов для ручной сверки
    examples = set()
    for item in price_all:
        rname = item.get('regionName', '') or item.get('region', '')
        if rname:
            examples.add(normalize(rname))
    print(f"  [INFO] Примеры нормализованных регионов в прайсе:\n    " + "\n    ".join(list(examples)[:30]))
    return None