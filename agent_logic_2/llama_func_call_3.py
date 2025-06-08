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

from agent_logic_2 import config as c
from ollama import AsyncClient, Options
from nayka_api.api_nayka_4_2 import find_doctors_by_keyword

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
async def extract_search_keyword_llm(question: str) -> tuple:
    """
    Возвращает tuple (тип, значение): ("surname", "Иванов") или ("specialty", "кардиолог"),
    либо ("timetable", "Иванов"), либо ("timetable_specialty", "кардиолог"), либо (None, None)
    """
    system_base = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ты — ассистент клиники «Наука». 
Твоя задача: по вопросу пользователя выделить либо фамилию врача (в именительном падеже), либо специальность (например: "кардиолог", "эндокринолог", "педиатр" и т.п.).
Если в вопросе есть только фамилия — верни: Surname: Иванов
Если в вопросе только специальность — верни: Specialty: кардиолог
Если вопрос про расписание (слова "расписание", "приём", "график работы" и т.п.):
    - если указано ФИО, верни Timetable: Иванов
    - если указана специальность, верни Timetable: Specialty: кардиолог
Если ничего не найдено — верни: NONE
Не добавляй других слов, никаких объяснений, только одну строку ответа!
"""
    user_part = f"\n<|start_header_id|>user<|end_header_id|>\nВопрос: {question}\n<|start_header_id|>assistant<|end_header_id|>"
    prompt = system_base + user_part

    resp = await ollama_call(prompt)
    text = resp.get("response", "").strip()
    if text.upper() == "NONE":
        return None, None
    m1 = re.match(r"^Surname:\s*([А-ЯЁA-Z][а-яёa-z\- ]+)$", text)
    if m1:
        return "surname", m1.group(1).strip()
    m2 = re.match(r"^Specialty:\s*([а-яёa-zA-Z \-]+)$", text)
    if m2:
        return "specialty", m2.group(1).strip()
    m3 = re.match(r"^Timetable:\s*([А-ЯЁA-Z][а-яёa-z\- ]+)$", text)
    if m3:
        return "timetable", m3.group(1).strip()
    m4 = re.match(r"^Timetable:\s*Specialty:\s*([а-яёa-zA-Z \-]+)$", text)
    if m4:
        return "timetable_specialty", m4.group(1).strip()
    # fallback
    return None, text


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

# ── Обогащение ответа заметками колл-центра ────────────────────────────────────────────────────
def enrich_with_cc_info(doctors: list):
    """
    Обогащает каждый dict заметкой call-центра, если есть id.
    """
    try:
        from nayka_api.doctors_cc_info import get_doctors_cc_info
        cc_info = get_doctors_cc_info()
        cc_by_id = {cc.get("id"): cc.get("callCenterInfo", "Нет заметок") for cc in cc_info}
        for doc in doctors:
            doc_id = doc.get("id")
            doc["callCenterInfo"] = cc_by_id.get(doc_id, "Нет заметок")
    except Exception as e:
        print(f"[DEBUG] enrich_with_cc_info error: {e}")
    return doctors


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

def format_doctor_list(docs):
    """
    Форматирует список врачей с нумерацией:
    - Если docs — строка: разбивает на врачей по шаблону ФИО и красиво разделяет.
    - Если docs — список dict: форматирует каждого врача, добавляет call-центр заметки (если есть).
    - Если docs — список строк: просто выводит построчно с разделителем.
    """
    import re
    if not docs:
        return "Врачи по вашей специальности не найдены."

    # Если docs — строка с несколькими врачами внутри
    if isinstance(docs, str):
        pattern = re.compile(r'(?=(?:^|\n)([А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+))')
        indices = [m.start() for m in pattern.finditer(docs)]
        if not indices or len(indices) == 1:
            return docs.strip()  # одна запись или не нашли
        blocks = []
        for i, idx in enumerate(indices):
            next_idx = indices[i + 1] if i + 1 < len(indices) else len(docs)
            blocks.append(f"{i+1}. " + docs[idx:next_idx].strip())
        return "\n\n---\n\n".join(blocks)

    # Если docs — не список, приводи к строке
    if not isinstance(docs, list):
        return str(docs)

    # Если docs — список dict (врачи с enrichment call-центра)
    formatted = []
    for idx, doc in enumerate(docs, 1):
        if isinstance(doc, dict):
            fio = doc.get("fio", "-")
            spec = doc.get("specialization", "-")
            desc = doc.get("description") or doc.get("about") or ""
            cc = doc.get("callCenterInfo")
            block = f"{idx}. {fio}\nСпециализация: {spec}"
            if desc:
                block += f"\n{desc.strip()}"
            if cc and cc != "Нет заметок":
                block += f"\n{'─'*50}\n📞 Заметка call-центра:\n{cc}\n{'─'*50}"
            formatted.append(block.strip())
        else:
            formatted.append(f"{idx}. {str(doc).strip()}")
    return "\n\n---\n\n".join(formatted)


#def format_doctor_schedule(doc):
    # doc — dict одного врача
    fio = doc.get("fio", "-")
    spec = doc.get("specialization", "-")
    regions = doc.get("regions", [])
    schedule = doc.get("schedule", {})
    lines = [
        f"Расписание для {fio} ({spec}):"
    ]
    if schedule:
        for region, days in schedule.items():
            lines.append(f"• {region}:")
            for day in days:
                date = day.get("date", "-")
                start = day.get("start", "-")
                end = day.get("end", "-")
                slots = ", ".join(day.get("slots", []))
                lines.append(f"    {date}: {start}-{end}  Окна: {slots}")
    else:
        lines.append("Расписание не указано.")
    return "\n".join(lines)

# ── Основная логика ───────────────────────────────────────────────────────────
repo = DoctorsRepository(DATA_DIR)


@with_retries(tries=2)
async def ollama_call(prompt: str) -> Dict[str, Any]:
    return await ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options=OLLAMA_OPTIONS,
        keep_alive=-1,
    )



async def investigate(question: str) -> str:
    print("\n=== Начало обработки вопроса ===")
    print(f"Вопрос: {question}")

    key_type, value = await extract_search_keyword_llm(question)
    if not value:
        print("LLM-парсер не смог выделить фамилию, специальность или намерение расписания.")
        return "Не удалось выделить фамилию, специальность или намерение расписания из вашего запроса."

    # Блок: поиск по фамилии
    if key_type == "surname":
        print(f"LLM-парсер определил фамилию: {value}")
        surname = value
        # --- код поиска по фамилии (оставьте без изменений) ---
        words = re.findall(r"[А-ЯЁ][а-яё]+", question)
        full_name = None
        if len(words) > 1:
            try:
                idx = next(i for i, w in enumerate(words) if w.lower() == surname.lower())
                if idx + 1 < len(words):
                    full_name = f"{words[idx]} {words[idx + 1]}"
            except StopIteration:
                full_name = None
        if full_name:
            print(f"Ищем по полному имени: {full_name}")
            all_docs = repo.read_all()
            local = [d for d in all_docs if d.get("fio", "").startswith(full_name)]
            if len(local) > 1:
                local = [local[0]]
        else:
            print(f"Ищем по фамилии: {surname}")
            local = repo.find_by_surname(surname)
        if not local:
            from nayka_api.api_nayka4_1 import get_all_doctors
            await repo.update(get_all_doctors)
            if full_name:
                local = [d for d in repo.read_all() if d.get("fio", "").startswith(full_name)]
                if len(local) > 1:
                    local = [local[0]]
            else:
                local = repo.find_by_surname(surname)
        if local:
            local = enrich_with_cc_info(local)
        if not local:
            similar = find_similar_surname(surname, repo.read_all())
            if similar:
                print(f"Найдена похожая фамилия: {similar}")
                local = repo.find_by_surname(similar)
                if local:
                    local = enrich_with_cc_info(local)
                    doctors_text = "\n\n---\n\n".join(format_doctor(d) for d in local)
                    return f"Похоже, опечатка: вы имели в виду '{similar}'?\n\n{doctors_text}"
        if not local:
            return f"Врач {surname} не найден в базе данных."
        print("Информация о враче получена!")
        return format_doctor_list(local)

    # Блок: поиск по специальности
    if key_type == "specialty":
        print(f"LLM-парсер определил специальность: {value}")
        docs = find_doctors_by_keyword(value)
    # >>>> вот эта строчка — обязательна!
        if docs and isinstance(docs, list) and isinstance(docs[0], dict):
            docs = enrich_with_cc_info(docs)
        if docs:
            return format_doctor_list(docs)
        return f"Врачи по специальности '{value}' не найдены."

    # fallback — если LLM вернул просто текст
    return value

def find_doctors_by_keyword_llm(question: str) -> str:
    return find_doctors_by_keyword(question)

async def main():
    """Основная функция."""
    q = input("Введите вопрос: ")
    res = await investigate(q)
    print("\n ======================= ")
    print(f"Ответ модели:\n\n{res}")


if __name__ == "__main__":
    asyncio.run(main())
