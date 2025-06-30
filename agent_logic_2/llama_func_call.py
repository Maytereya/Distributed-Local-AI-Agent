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
from agent_logic_2.nayka_api.api_nayka import find_doctors_by_keyword, find_doctor_schedule, \
    cleanup_old_doctors_files, get_all_doctors
from nayka_api.api_price_all import update_price_all, load_price_all
from nayka_api.doctors_cc_info import get_doctors_cc_info

# Функция импорта прайса не используется ввиду несостоятельности последнего
# from nayka_api.api_price_by_region_3 import get_price, format_services

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
        # Очищаем старые файлы врачей перед записью нового
        cleanup_old_doctors_files()
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

Если в вопросе встречаются слова: "узи", "узи врач", "узист", "ультразвуковая диагностика", "врач ультразвуковой диагностики", "врач узи", "уз-диагностика" — всегда возвращай Specialty: врач ультразвуковой диагностики.

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
        f"{{NAME}} • ФИО: {item.get('fio', '-')}",
    ]

    # Специализация
    specialization = item.get("specialization") or "-"
    spec_lines = [s.strip().lstrip('-').strip() for s in specialization.splitlines() if s.strip()]
    specialization_full = "\n".join(dict.fromkeys(spec_lines)) if spec_lines else "-"
    lines.append(f"• Специализация:\n{specialization_full}")
    # Адреса работы и region_ids
    regions = item.get("regions", ['-'])
    # region_ids = item.get("region_ids", [])
    lines.append(f"• Адрес/Адреса: {', '.join(regions)}")

    # Загрузим прайс ДО debug print
    # Пока эту функцию отключим, так как не можем получить правильный прайс
    # price_all = load_price_all()  # Загружаем кэш всех прайсов

    # Debug print после загрузки price_all
    # print("[DEBUG] Адреса работы врача:", regions)
    # print("[DEBUG] region_ids врача:", region_ids)
    # print("[DEBUG] regionIds в прайсе:", sorted(set(row['regionId'] for row in price_all)))
    #
    # # Блок — Прайсы по region_id (у врача может быть несколько регионов)
    # price_blocks = []
    # for region, region_id in zip(regions, region_ids):
    #     if not region_id or region == "-":
    #         price_blocks.append(f"У Врача не указан regionId для '{region}'!")
    #         continue
    #
    #
    #     services = [s for s in price_all if s.get("regionId") == region_id]
    #
    #     if not services:
    #         # Fallback: priceByRegion
    #         try:
    #             services = get_price(region_id)
    #             if services:
    #                 price_blocks.append(f"Прайс ({region}, через priceByRegion):\n" + format_services(services[:7]))
    #             else:
    #                 price_blocks.append(f"Для региона {region} прайс не найден даже через priceByRegion.")
    #         except Exception as e:
    #             price_blocks.append(f"Для региона {region} не удалось загрузить priceByRegion: {e}")
    #         continue
    #
    #     price_blocks.append(f"{{PRICE}} • Прайс ({region}):\n" + format_services(services[:7]))

    #     добавляем значение прайса в основной список "lines"
    # lines.append("")
    # lines.append("─" * 10)
    # lines.extend(price_blocks)
    # lines.append("")

    # Заметка call-центра
    cc = item.get("callCenterInfo")
    if cc:
        lines.append("─" * 10)
        lines.append("• 📞Заметка колл-центра:")
        lines.append(str(cc))
        lines.append("─" * 10)
        lines.append("─" * 10)

    # Расписание (если есть)
    schedule = item.get("schedule")
    if isinstance(schedule, dict):
        lines.append("• Расписание:")
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


def format_doctor_schedule(doc):
    fio = doc.get("fio", "-")
    spec = doc.get("specialization", "-")
    spec_lines = [line.strip() for line in spec.split('\n') if line.strip()]
    # Убираем дубли в специализации
    spec_full = "\n".join(dict.fromkeys(spec_lines)) if spec_lines else "-"
    regions = doc.get("regions", [])
    schedule = doc.get("schedule", {})
    cc = doc.get("callCenterInfo")

    lines = []

    # ФИО
    lines.append(f"{fio}:")
    lines.append("")  # Пробел после ФИО

    # Специализация
    if spec_full and spec_full != "-":
        lines.append(spec_full)
        lines.append("")  # Пробел после специализации

    # Call-центр (если есть)
    if cc and cc.strip() and cc != "Нет заметок":
        lines.append("─" * 10)
        lines.append("• 📞Заметка колл-центра:")
        lines.append(str(cc).strip())
        lines.append("─" * 10)
        lines.append("─" * 10)
        lines.append("")  # Пробел после заметки

    # Адреса
    if regions:
        lines.append(f"• Адрес/адреса: {', '.join(regions)}")
        lines.append("")  # Пробел после адреса

    # Расписание
    if schedule:
        lines.append("• Расписание: ")
        for region, days in schedule.items():
            lines.append(f"• По адресу приема {region}:")
            lines.append(f"• По адресу приема {region}:")
            for day in days:
                date = day.get("date", "-")
                start = day.get("start", "-")
                end = day.get("end", "-")
                # убираем секунды
                if start and len(start) >= 5:
                    start = start[:5]
                if end and len(end) >= 5:
                    end = end[:5]
                slots = [slot[:5] for slot in day.get("slots", []) if slot and len(slot) >= 5]
                lines.append(f"    {date}: {start}-{end}  Окна: {', '.join(slots)}")
    else:
        lines.append("Расписание не указано.")

    return "\n".join(lines)



# ── Основная логика ───────────────────────────────────────────────────────────
repo = DoctorsRepository(DATA_DIR)

FORMATTER = "\n\n---\n\n"


@with_retries(tries=2)
async def ollama_call(prompt: str) -> Dict[str, Any]:
    res = await ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options=OLLAMA_OPTIONS,
        keep_alive=-1,
    )
    return res.__dict__


async def investigate(question: str) -> str:
    print("\n=== Начало обработки вопроса ===")
    print(f"Вопрос: {question}")

    key_type, value = await extract_search_keyword_llm(question)
    if not value:
        return (
            "Не удалось выделить фамилию, специальность "
            "или намерение расписания из вашего запроса."
        )
    # Обращаемся к функциям через их словарь. Вроде бы удобнее.
    handlers = {
        "surname": handle_surname_search,
        "specialty": handle_specialty_search,
        "timetable": handle_timetable_search,
    }

    handler = handlers.get(key_type)
    if handler:
        return await handler(value, question)

    # Если LLM вернул просто текст
    return value

async def handle_surname_search(surname: str, question: str) -> str:
    print(f"LLM-парсер определил фамилию: {surname}")

    words = re.findall(r"[А-ЯЁ][а-яё]+", question)
    full_name = extract_full_name(words, surname)

    docs = await search_with_fallback(full_name or surname, bool(full_name))
    if docs:
        docs = enrich_with_cc_info(docs)
        return format_documents(docs)

    similar = find_similar_surname(surname, repo.read_all())
    if similar:
        print(f"Найдена похожая фамилия: {similar}")
        docs = repo.find_by_surname(similar)
        if docs:
            docs = enrich_with_cc_info(docs)
            return (
                f"Похоже, опечатка: вы имели в виду '{similar}'?{FORMATTER}"
                f"{format_documents(docs)}"
            )

    return f"Врач с фамилией '{surname}' не найден в базе данных."

async def search_with_fallback(name: str, is_full_name: bool) -> List[Dict[str, Any]]:
    """
    Попытка найти документы локально, затем обновление репозитория и повторный поиск.
    """
    docs = (
        filter_docs(repo.read_all(), name)
        if is_full_name
        else repo.find_by_surname(name)
    )
    if docs:
        return docs

    await repo.update(get_all_doctors)

    return (
        filter_docs(repo.read_all(), name)
        if is_full_name
        else repo.find_by_surname(name)
    )


def extract_full_name(words: List[str], surname: str) -> Optional[str]:
    """
    Если после фамилии в списке слов есть имя, возвращаем 'Фамилия Имя'.
    """
    surname_lower = surname.lower()
    for i, w in enumerate(words):
        if w.lower() == surname_lower and i + 1 < len(words):
            return f"{words[i]} {words[i + 1]}"
    return None


def filter_docs(docs: List[Dict[str, Any]], full_name: str) -> List[Dict[str, Any]]:
    matches = [d for d in docs if d.get("fio", "").startswith(full_name)]
    return matches[:1] if len(matches) > 1 else matches


def format_documents(docs: List[Dict[str, Any]]) -> str:
    return FORMATTER.join(
        f"{i + 1}. {format_doctor(d)}" for i, d in enumerate(docs)
    )

async def handle_specialty_search(specialty: str, _: str) -> str:
    print(f"LLM-парсер определил специальность: {specialty}")

    docs = find_doctors_by_keyword(specialty)
    if not docs:
        return f"Врачи по специальности '{specialty}' не найдены."

    if isinstance(docs, list):
        docs = enrich_with_cc_info(docs)
    return format_documents(docs)

async def handle_timetable_search(surname: str, _: str) -> str:
    print(f"LLM-парсер определил запрос расписания по фамилии: {surname}")

    docs = find_doctor_schedule(surname)
    if isinstance(docs, str):
        return docs

    return FORMATTER.join(
        f"{i + 1}. {format_doctor_schedule(doc)}"
        for i, doc in enumerate(docs)
    )

def find_doctors_by_keyword_llm(question: str) -> str:
    return find_doctors_by_keyword(question)



def print_unique_priceall_regions():
    price_all = load_price_all()
    regions_in_priceall = set()
    for row in price_all:
        if not isinstance(row, dict):
            continue
        name = row.get("regionName") or row.get("region")
        if name:
            regions_in_priceall.add(name.strip())
    print("[DEBUG] regionName/region из priceAll (первые 20):")
    for idx, r in enumerate(list(regions_in_priceall)[:20]):
        print(f"{idx + 1}. '{r}'")

        print(f"{idx + 1}. '{r}'")


async def main():
    """Основная функция."""
    q = input("Введите вопрос: ")
    update_price_all()
    res = await investigate(q)
    print("\n ======================= ")
    print(f"Ответ модели:\n\n{res}")


if __name__ == "__main__":
    # print_unique_priceall_regions()
    asyncio.run(main())