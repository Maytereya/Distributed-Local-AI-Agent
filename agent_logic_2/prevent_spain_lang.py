import time
from langdetect import detect
from ollama import Client, Options
import config as c

ollama = Client(c.ollama_url)

llm_model = "llama3.3:70b-instruct-q8_0"
DEFAULT_OPTS = Options(temperature=0.7, top_k=50, top_p=0.9, stop=["<|eot_id|>"])

def safe_generate(prompt: str, retries: int = 2, model: str = llm_model, options: Options = DEFAULT_OPTS) -> str:
    for attempt in range(1, retries + 2):  # Первый + возможные повторы
        try:
            res = ollama.generate(model=model, prompt=prompt, options=options)
            response = res.get("response", "").strip()

            # Проверка 1: пустой или слишком короткий ответ
            if len(response) < 5:
                print(f"⚠️ [{attempt}] Ответ слишком короткий, пробуем снова…")
                continue

            # Проверка 2: язык
            lang = detect(response)
            if lang != "ru":
                print(f"⚠️ [{attempt}] Ответ на '{lang}' вместо 'ru', пробуем снова…")
                continue

            # Всё ок
            return response

        except Exception as e:
            print(f"🚨 Ошибка генерации [{attempt}]: {e}")
            time.sleep(1)

    return "⚠️ Не удалось получить корректный ответ на русском языке."