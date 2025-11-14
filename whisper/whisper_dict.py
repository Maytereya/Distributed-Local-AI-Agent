import gradio as gr
import requests

import agent_logic_2.config as c


def load_prompt():
    try:
        resp = requests.get(f"{c.WHISPER_HTTP_API}/prompt", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        gr.Success("Словарь успешно загружен")
        return data.get("prompt", "")
    except Exception as e:
        gr.Error(f"Ошибка загрузки словаря: {e}")
        return f"Ошибка загрузки словаря: {e}"


def save_prompt(text: str):
    try:
        resp = requests.put(
            f"{c.WHISPER_HTTP_API}/prompt",
            json={"prompt": text},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        gr.Success(f"Сохранено. Длина: {data.get('length', 0)} символов")
        return None
    except Exception as e:
        gr.Error(f"Ошибка сохранения словаря: {e}")
        return None
