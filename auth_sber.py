import os
import threading
import time
import uuid

import requests

import agent_logic_2.config as c

NGW_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"


def fetch_sber_token(basic_key: str, scope="SALUTE_SPEECH_PERS") -> tuple[str, int]:
    headers = {
        "Authorization": f"Basic {basic_key}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
    }
    verify_path = os.path.abspath(c.SBER_CA) if getattr(c, "SBER_CA", None) else True
    r = requests.post(NGW_URL, headers=headers, data={"scope": scope}, timeout=15, verify=verify_path)
    r.raise_for_status()
    js = r.json()
    # print (js["access_token"])
    return js["access_token"], int(js.get("expires_in", 1800))


def start_token_refresher(basic_key: str = c.salut_authorization):
    """
    Каждые (expires_in - 60) сек обновляет c.SBER_TOKEN в фоне.
    """

    def loop():
        while True:
            try:
                token, ttl = fetch_sber_token(basic_key)
                c.SBER_TOKEN = token
                sleep_for = max(60, ttl - 60)
            except Exception as e:
                # если не вышло — пробуем через минуту
                sleep_for = 60
            time.sleep(sleep_for)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

if __name__ == '__main__':
    t,v = fetch_sber_token(c.salut_authorization)
    print(t)