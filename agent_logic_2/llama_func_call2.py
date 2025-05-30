# РЕАЛИЗОВАН БЫСТРЫЙ ПОИСК ВРАЧЕЙ ПО ФАМИЛИИ
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from ollama import AsyncClient, Options

from agent_logic_2 import config as c
from agent_logic_2.nayka_api.api_nayka4_1 import get_all_doctors
from agent_logic_2.nayka_api.doctors_cc_info import get_doctors_cc_info

# ── Конфигурация ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к данным о врачах
DATA_DIR = os.path.join(os.path.dirname(__file__), "nayka_api", "apidata")
# Настройка Ollama
OLLAMA_MODEL = c.ll_model_big

OLLAMA_OPTIONS = Options(
    temperature=0.3,
    top_k=40,
    top_p=0.9,
    mirostat=1,
    mirostat_tau=5.0,
    mirostat_eta=0.1,
)

ollama_client = AsyncClient(c.ollama_url)

# Расширенный список стоп-слов
STOP_WORDS = {
    # Местоимения
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "меня", "тебя", "его", "её", "нас", "вас", "их",
    "мне", "тебе", "ему", "ей", "нам", "вам", "им",
    "мной", "тобой", "им", "ей", "нами", "вами", "ими",
    "себя", "себе", "собой",
    # Предлоги и союзы
    "у", "в", "на", "с", "к", "о", "об", "от", "и", "или",
    "а", "но", "по", "под", "над", "перед", "за", "через",
    "из", "из-за", "из-под",
    # Служебные слова
    "работает", "работают", "работа", "доктор", "врач",
    "клиника", "принимает", "приём", "запись", "где", "как",
    "есть", "быть", "будет", "будут", "был", "была", "были",
    # Указательные слова
    "этот", "эта", "это", "эти", "тот", "та", "то", "те",
    # Частицы
    "ли", "же", "бы", "ведь", "вот", "даже", "именно",
    # Вопросительные слова
    "кто", "что", "какой", "какая", "какое", "какие",
    "чей", "чья", "чьё", "чьи", "который", "которая",
    "которое", "которые",
}


# ── Утилиты ──────────────────────────────────────────────────────────────────
def with_retries(tries: int = 3):
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(tries):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"{fn.__name__} try {i + 1}/{tries} failed: {e}")
            # Все попытки исчерпаны — кидаем последнюю ошибку
            raise last_exc

        return wrapper

    return decorator


# ── Репозиторий данных врачей ─────────────────────────────────────────────────
class DoctorsRepository:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _today_path(self) -> str:
        date = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.data_dir, f"doctors_{date}.jsonl")

    def _default_path(self) -> str:
        return os.path.join(self.data_dir, "doctors.jsonl")

    def get_path(self) -> str:
        return self._today_path()

    @with_retries(tries=2)
    async def update(self, fetch_fn: Callable[[], List[Dict[str, Any]]]) -> bool:
        today = self._today_path()
        data = fetch_fn()
        if not data:
            logger.error("Fetch doctors returned no data")
            return False
        print(f"Получено {len(data)} врачей")
        print(f"Первый врач: {data[0] if data else 'нет данных'}")
        with open(today, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Written new data file: {today}")
        return True

    def read_all(self) -> List[Dict[str, Any]]:
        path = self.get_path()
        if not os.path.exists(path):
            logger.error(f"Doctors file not found: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def find_by_surname(self, surname: str) -> List[Dict[str, Any]]:
        surname = surname.lower()
        docs = self.read_all()
        found = [d for d in docs if surname in d.get("fio", "").lower()]
        logger.info(f"Found {len(found)} by surname {surname}")
        return found


# ── Извлечение фамилии ───────────────────────────────────────────────────────
async def extract_surname_llm(question: str) -> Optional[str]:
    # 1) Собираем system-prompt из первой версии
    system_base = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — ассистент клиники «Наука». Твоя задача: из вопроса выделить фамилию врача.
Отвечай ровно одной фамилией в именительном падеже без лишних слов.
Если фамилия не найдена — верни 'NONE'.
"""
    user_part = f"\n<|start_header_id|>user<|end_header_id|>\nВопрос: {question}\n<|start_header_id|>assistant<|end_header_id|>"
    prompt = system_base + user_part

    # 2) Делаем вызов LLM
    resp = await ollama_call(prompt)
    text = resp.get("response", "").strip()
    # 3) Если LLM вернул 'NONE', возвращаем None, иначе фамилию
    return None if text.upper() == 'NONE' else text


def is_potential_surname(word: str) -> bool:
    """
    Проверяет, может ли слово быть фамилией.
    Слово считается потенциальной фамилией, если:
    1. Не является стоп-словом
    2. Начинается с заглавной буквы
    3. Содержит только буквы и дефис
    4. Имеет длину не менее 3 символов
    """
    word = word.strip()
    if not word:
        return False

    if word.lower() in STOP_WORDS:
        return False

    if not word[0].isupper():
        return False

    if not all(c.isalpha() or c == "-" for c in word):
        return False

    if len(word) < 3:
        return False

    return True


# ── Поиск похожей фамилии ─────────────────────────────────────────────────────
def find_similar_surname(input_surname: str, doctors: List[Dict[str, Any]], threshold: float = 0.75) -> Optional[str]:
    surnames = {d["fio"].split()[0].lower() for d in doctors}
    best, br = None, 0.0
    for s in surnames:
        r = SequenceMatcher(None, input_surname.lower(), s).ratio()
        if r > br:
            best, br = s, r
    return best.capitalize() if br >= threshold else None


# ── Форматирование ответа ────────────────────────────────────────────────────
def format_doctor(item: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"• ФИО: {item.get('fio', '-')}",
    ]

    # Специализация
    specialization = item.get("specialization") or "-"
    # Обрезаем и чистим
    specialization = specialization.split("\n")[0].replace("-", "").strip() or "-"
    lines.append(f"• Специализация: {specialization}")

    # Регионы
    regions = item.get("regions", ['-'])
    lines.append(f"• Регионы: {', '.join(regions)}")

    # Заметка call-центра
    cc = item.get("callCenterInfo")
    if cc:
        lines.append("─" * 50)
        lines.append("📞 Заметка call-центра:")
        lines.append(str(cc))
        lines.append("─" * 50)

    # Расписание
    schedule = item.get("schedule")
    if isinstance(schedule, dict):
        lines.append("Расписание:")
        for region, days in schedule.items():
            lines.append(f"{region}:")
            for day in days:
                date = day.get("date", "-")
                start = day.get("start", "-")
                end = day.get("end", "-")
                slots = day.get("slots") or []
                lines.append(f"• {date}: {start} – {end}")
                if slots:
                    lines.append(f"  Доступное время: {', '.join(slots)}")

    # Удаляем любые None и приводим всё к str
    cleaned = [str(line) for line in lines if line is not None]
    return "\n".join(cleaned)


# ── Основная логика ───────────────────────────────────────────────────────────
repo = DoctorsRepository(DATA_DIR)


@with_retries(tries=2)
async def ollama_call(prompt: str):  # -> Dict[str, Any]
    call_result = await ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options=OLLAMA_OPTIONS,
        keep_alive=-1,
    )

    return call_result


async def investigate(question: str) -> str:
    print("\n=== Начало обработки вопроса ===")
    print(f"Вопрос: {question}")

    # 1) LLM-парсер выделяет ровно одну фамилию или None
    surname = await extract_surname_llm(question)
    if not surname:
        print("LLM-парсер не смог выделить фамилию.")
        return "Не удалось найти фамилию врача в вопросе."
    print(f"LLM-парсер определил фамилию: {surname}")

    # 2) Проверяем, есть ли в вопросе сразу ФИ или ФИО
    #    Подбираем все слова с заглавной буквы
    words = re.findall(r"[А-ЯЁ][а-яё]+", question)
    full_name = None
    if len(words) > 1:
        # Ищем фамилию в этом списке, и если за ней идёт имя — склеиваем
        try:
            idx = next(i for i, w in enumerate(words) if w.lower() == surname.lower())
            if idx + 1 < len(words):
                full_name = f"{words[idx]} {words[idx + 1]}"
        except StopIteration:
            full_name = None

    # 3) Поиск: если есть full_name — точное совпадение по началу FIO, иначе по фамилии
    if full_name:
        print(f"Ищем по полному имени: {full_name}")
        all_docs = repo.read_all()
        local = [d for d in all_docs if d.get("fio", "").startswith(full_name)]
        # если всё ещё несколько врачей, берём первого
        if len(local) > 1:
            local = [local[0]]
    else:
        print(f"Ищем по фамилии: {surname}")
        local = repo.find_by_surname(surname)

    # 4) Если не нашли ни по одному, обновляем через API и ищем снова
    if not local:
        await repo.update(get_all_doctors)
        if full_name:
            local = [d for d in repo.read_all() if d.get("fio", "").startswith(full_name)]
            if len(local) > 1:
                local = [local[0]]
        else:
            local = repo.find_by_surname(surname)

    # 5) Обогащаем заметками call-центра
    if local:

        print("Загружаем заметки call-центра...")
        cc_info = get_doctors_cc_info()
        for doctor in local:
            doc_id = doctor.get("id")
            for cc in cc_info:
                if cc.get("id") == doc_id:
                    doctor["callCenterInfo"] = cc.get("callCenterInfo", "Нет заметок")
                    break
            else:
                doctor["callCenterInfo"] = "Нет заметок"

    # 6) Если всё ещё пусто — ищем похожую фамилию
    if not local:
        similar = find_similar_surname(surname, repo.read_all())
        if similar:
            print(f"Найдена похожая фамилия: {similar}")
            local = repo.find_by_surname(similar)
            if local:
                return (
                        f"Похоже, опечатка: вы имели в виду '{similar}'?\n\n"
                        + "\n\n".join(format_doctor(d) for d in local)
                )

    # 7) Итоговый вывод
    if not local:
        return f"Врач {surname} не найден в базе данных."

    print("Информация о враче получена!")
    return "\n\n".join(format_doctor(d) for d in local)


async def main():
    """Основная функция."""
    q = input("Введите вопрос: ")
    res = await investigate(q)
    print("\n ======================= ")
    print(f"Ответ модели:\n\n{res}")


if __name__ == "__main__":
    asyncio.run(main())
