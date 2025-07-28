import json
import requests
from datetime import datetime
from pathlib import Path

from agent_logic_2 import config as c

base_url = c.nayka_base_url
auth = requests.auth.HTTPBasicAuth(c.nayka_login, c.nayka_pass)

API_URL = f"{base_url}/doctorServicePricesByRegion"

# Определяем пути относительно расположения этого скрипта
SCRIPT_DIR = Path(__file__).parent
DOCTORS_DIR = SCRIPT_DIR / "apidata"
PRICES_DIR = SCRIPT_DIR / "apidata" / "doctor_prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

def get_today_str():
    return datetime.now().strftime("%Y%m%d")

def get_prices_filename(date=None):
    if date is None:
        date = get_today_str()
    return PRICES_DIR / f"doctor_prices_{date}.jsonl"

def cleanup_old_prices(days_to_keep=2):
    files = list(PRICES_DIR.glob("doctor_prices_*.jsonl"))
    for f in files:
        date_str = f.stem.rsplit('_', 1)[-1]
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            continue
        if (datetime.now() - file_date).days >= days_to_keep:
            print(f"🗑️ Удаляем устаревший кэш: {f}")
            f.unlink()

def get_latest_doctors_file():
    """Возвращает путь к самому свежему файлу doctors_*.jsonl"""
    files = sorted(DOCTORS_DIR.glob("doctors_*.jsonl"))
    if not files:
        raise FileNotFoundError("Файл doctors_*.jsonl не найден")
    return files[-1]

def load_doctors():
    doctors_file = get_latest_doctors_file()
    with open(doctors_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def fetch_doctor_prices(doctor_id, region_id):
    params = {"doctorId": doctor_id, "regionId": region_id}
    try:
        r = requests.get(API_URL, params=params, auth=auth, timeout=10, verify=False)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[ERROR] doctorId={doctor_id} regionId={region_id} status={r.status_code} data={r.text}")
            return []
    except Exception as e:
        print(f"[EXCEPTION] doctorId={doctor_id} regionId={region_id} error={e}")
        return []

def update_doctor_prices():
    fn = get_prices_filename()
    if fn.exists():
        print("✅ doctor_prices на сегодня уже скачан")
        return
    print("⏬ Собираем doctor_prices...")

    doctors = load_doctors()
    rows = []
    total = 0
    for doc in doctors:
        doctor_id = doc.get("id")
        region_ids = doc.get("region_ids") or doc.get("regionIds") or []
        if not doctor_id or not region_ids:
            print(f"[WARN] Пропущен врач без регионов: {doc.get('fio', doctor_id)}")
            continue
        for region_id in region_ids:
            if not region_id:
                continue
            services = fetch_doctor_prices(doctor_id, region_id)
            for service in services:
                row = {
                    "doctorId": doctor_id,
                    "regionId": region_id,
                    "fio": doc.get("fio"),
                    "serviceName": service.get("serviceName"),
                    "serviceId": service.get("serviceId"),
                    "cost": service.get("cost"),
                    "priceUnitId": service.get("priceUnitId"),
                    "serviceHomecode": service.get("serviceHomecode"),
                    "deadline": service.get("deadline"),
                }
                rows.append(row)
                total += 1
    with open(fn, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ doctor_prices обновлен ({total} цен) — {fn}")
    cleanup_old_prices(days_to_keep=2)

def load_doctor_prices():
    fn = get_prices_filename()
    if not fn.exists():
        update_doctor_prices()
    result = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            try:
                result.append(json.loads(line))
            except Exception as e:
                print(f"[ERROR] Не удалось распарсить строку: {line[:100]}... Ошибка: {e}")
    return result

if __name__ == "__main__":
    update_doctor_prices()