# ЦЕПЬ ВЫЗОВОВ (вместе с модулем router_preprocessor.py):
# routing(..., think) → get_doc_info_from_api(..., think) →
# doctor_info.investigate(..., think) → extract_search_keyword_llm(..., think) → ollama_call(..., think).

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from converters import html_cleaner

from ollama import AsyncClient

from agent_logic_2.nayka_api.api_nayka import find_doctors_by_keyword, find_doctor_schedule, \
    cleanup_old_doctors_files, get_all_doctors, get_active_date_str
from nayka_api.api_price import load_doctor_prices, update_price_all, load_price_all
# from nayka_api.api_price_all import update_price_all, load_price_all
from nayka_api.doctors_cc_info import get_doctors_cc_info

# Package-relative import to work reliably when this module is imported as part of agent_logic_2
try:
    from . import ollama_settings
except ImportError:
    # Fallback if ollama_settings.py is placed at the project root
    import ollama_settings

try:
    from . import config as c
except ImportError:
    # Fallback if ollama_settings.py is placed at the project root
    import config as c

# ── Конфигурация ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к данным о врачах
DATA_DIR = os.path.join(os.path.dirname(__file__), "nayka_api", "apidata")
# Инициализация Ollama
ollama_client = AsyncClient(c.ollama_url)
# ollama_settings.init_model_name()
ollama_settings.init_options()
model: str = ollama_settings.init_model_name()

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


# ── Нормализация специальности (простая лемматизация множественного к единственному) ──
def normalize_specialty_term(q: str) -> str:
    """
    Очень простая эвристика для приведения множественного числа к единственному для
    названий специальностей: "кардиологи" → "кардиолог", "педиатры" → "педиатр".
    Работает только для однословных форм; если не уверены — возвращаем исходное.
    """
    if not isinstance(q, str) or not q:
        return q
    w = q.strip().lower()
    # частые шумовые слова
    if w in {"врачи", "врач"}:
        return ""
    # базовые правила: ..."и" → удалить, ..."ы" → удалить
    if len(w) > 4 and (w.endswith("и") or w.endswith("ы")):
        return w[:-1]
    return w


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
        # Используем «активную» дату по Москве (до 06:00 — вчера, после — сегодня)
        date = get_active_date_str()
        return os.path.join(self.data_dir, f"doctors_{date}.jsonl")

    def _default_path(self) -> str:
        return os.path.join(self.data_dir, "doctors.jsonl")

    def get_path(self) -> str:
        return self._today_path()

    @with_retries(tries=2)
    async def update(self, fetch_fn: Callable[[], List[Dict[str, Any]]]) -> bool:
        today = self._today_path()
        data = await asyncio.to_thread(fetch_fn)
        if not data:
            logger.error("Fetch doctors returned no data")
            return False
        print(f"Получено {len(data)} врачей")
        print(f"Первый врач: {data[0] if data else 'нет данных'}")
        # Очищаем старые файлы (сохраняем активную и вчерашнюю дату)
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

    async def read_all_async(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.read_all)

    async def find_by_surname_async(self, surname: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.find_by_surname, surname)


# ── Извлечение фамилии ───────────────────────────────────────────────────────
async def extract_search_keyword_llm(question: str, think: bool | None = None) -> tuple:
    """
    Возвращает tuple (тип, значение): ("surname", "Иванов") или ("specialty", "кардиолог"), 
    либо ("timetable", "Иванов"), либо ("timetable_specialty", "кардиолог"), либо (None, None)
    """
    system_base = """
SYSTEM:
Ты — ассистент клиники «Наука». 
Твоя задача: по вопросу пользователя выделить либо фамилию врача (в именительном падеже), либо специальность (например: 
"кардиолог", "эндокринолог", "педиатр", "хирург", "терапевт", "травматолог", "проктолог" и т.п.).
Если в вопросе встречаются слова вида "список <специальность во множественном числе>", 
то определи специальность и верни её в единственном числе, например: Specialty: хирург
Если в вопросе указана специальность во множественном числе (например: "кардиологи", "урологи"), 
верни Specialty в единственном числе: Specialty: кардиолог
Если в вопросе встречаются слова: "узи", "узи врач", "узист", "ультразвуковая диагностика", "врач ультразвуковой диагностики", 
"врач узи", "узи-диагностика" — всегда возвращай Specialty: врач ультразвуковой диагностики.
Если в вопросе есть только фамилия — верни: Surname: Иванов
Если в вопросе только специальность — верни: Specialty: кардиолог
Если вопрос про расписание (слова "расписание", "приём", "график работы", "время работы" и т.п.) и указано ФИО или фамилия, верни Timetable: Иванов
Если ничего не найдено — верни: NONE
Не добавляй других слов, никаких объяснений, только одну строку ответа!
"""
    user_part = f"\nUSER:\nВопрос: {question}\n"
    prompt = system_base + user_part

    resp = await ollama_call(prompt=prompt, llm=ollama_settings.init_model_name(), think=think)
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

# ── Асинхронное обогащение заметками call-центра с кэшем ─────────────────────
_cc_map: Optional[Dict[int, str]] = None
_cc_ts: float = 0.0
CC_TTL: int = 600  # seconds

async def get_cc_map_cached() -> Dict[int, str]:
    import time
    global _cc_map, _cc_ts
    now = time.time()
    if _cc_map is None or (now - _cc_ts) > CC_TTL:
        data = await asyncio.to_thread(get_doctors_cc_info)
        _cc_map = {row.get("id"): row.get("callCenterInfo", "Нет заметок") for row in (data or [])}
        _cc_ts = now
    return _cc_map

async def async_enrich_with_cc_info(doctors: list):
    """
    Асинхронное обогащение каждого dict заметкой call-центра с кешированием.
    """
    try:
        cc_by_id = await get_cc_map_cached()
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
    spec_lines = [s.strip().lstrip('-').strip() for s in specialization.splitlines() if s.strip()]
    specialization_full = "\n".join(dict.fromkeys(spec_lines)) if spec_lines else "-"
    lines.append(f"• Специализация:\n{specialization_full}")
    # Адреса работы и region_ids
    regions = item.get("regions", ['-'])
    # region_ids = item.get("region_ids", [])
    lines.append(f"• Адрес/Адреса: {', '.join(regions)}")

    # Заметка call-центра
    cc = item.get("callCenterInfo")
    if cc:
        lines.append("─" * 10)
        lines.append("• 📞Заметка колл-центра:")
        # Чистим HTML и сущности, чтобы не показывать теги/&#NNNN;
        lines.append(html_cleaner.strip_html(str(cc)))
        # lines.append("─" * 10)

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


def format_doctor_prices(doctor_id, fio, region_map=None):
    """Выводит список услуг с ценами по каждому филиалу для врача."""
    prices = load_doctor_prices()
    doc_prices = [p for p in prices if p.get("doctorId") == doctor_id]
    if not doc_prices:
        return "• Прайс не найден для этого врача."
    blocks = []
    region_groups = {}
    for p in doc_prices:
        reg = p.get("regionId")
        reg_name = (region_map or {}).get(reg) or p.get("regionName") or f"Филиал (ID {reg})"
        region_groups.setdefault(reg_name, []).append(p)
    for region, services in region_groups.items():
        lines = [f"— {region} —"]
        for s in services:
            name = s.get("serviceName", "-")
            cost = s.get("cost", "-")
            lines.append(f"• {name}: {cost} ₽")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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
        # lines.append("─" * 10)
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
                lines.append(f"    {date}: {start}-{end}  * Окна: {', '.join(slots)}")
    else:
        lines.append("Расписание не указано.")

    return "\n".join(lines)


# ── Основная логика ───────────────────────────────────────────────────────────
repo = DoctorsRepository(DATA_DIR)

FORMATTER = "\n\n---\n\n"

# ── Асинхронные обёртки для синхронных I/O/API ────────────────────────────────
async def find_doctors_by_keyword_async(q: str):
    return await asyncio.to_thread(find_doctors_by_keyword, q)

async def find_doctor_schedule_async(surname: str):
    return await asyncio.to_thread(find_doctor_schedule, surname)

async def load_doctor_prices_async():
    return await asyncio.to_thread(load_doctor_prices)


@with_retries(tries=2)
async def ollama_call(prompt: str, llm: str = model, think: bool = None, ) -> Dict[str, Any]:
    if not llm:
        raise ValueError("Model is not specified yet")
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)
    # elif llm:
    #     print("!!! ollama_call llm is: ", llm)
    #     print("!!! ollama_call ollama_settings.OLLAMA_MODEL: ", ollama_settings.OLLAMA_MODEL)
    #     print("!!! ollama_call Think status:", think)
    #     print("!!! options: ", ollama_settings.options_set())

    res = await asyncio.wait_for(
        ollama_client.generate(
            model=llm,
            prompt=prompt,
            options=ollama_settings.options_set(),
            keep_alive=-1,
            think=think,
        ),
        timeout=25
    )

    return res.__dict__


async def investigate(question: str, think: bool = None) -> str:
    print("\n=== Начало обработки вопроса ===")
    print(f"Вопрос: {question}")
    # Быстрый путь: если запрос похож на одиночную фамилию — пропускаем LLM-парсер
    q = (question or "").strip()
    if q and len(q.split()) == 1 and is_potential_surname(q):
        try:
            return await handle_surname_search(q, question)
        except Exception:
            pass

    key_type, value = await extract_search_keyword_llm(question, think=think)
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


def get_latest_doctors_file():
    """Находит актуальный doctors_*.jsonl: сначала за активную дату (MSK 06:00), иначе самый свежий."""
    apidata = Path(DATA_DIR)
    active = get_active_date_str()
    active_file = apidata / f"doctors_{active}.jsonl"
    if active_file.exists():
        return str(active_file)
    all_files = sorted(apidata.glob("doctors_*.jsonl"), reverse=True)
    for f in all_files:
        if f.exists():
            return str(f)
    raise FileNotFoundError("Файл doctors_*.jsonl не найден")


def build_region_map():
    region_map = {}
    try:
        doctors_file = get_latest_doctors_file()
        with open(doctors_file, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                for reg_id, region in zip(doc.get("region_ids", []), doc.get("regions", [])):
                    if reg_id and region:
                        region_map[reg_id] = region
    except Exception as e:
        print(f"[REGION_MAP ERROR] {e}")
    return region_map


_region_map = None


def get_region_map():
    global _region_map
    if _region_map is None:
        _region_map = build_region_map()
    return _region_map

async def build_region_map_async():
    return await asyncio.to_thread(build_region_map)

async def get_region_map_async():
    global _region_map
    if _region_map is None:
        _region_map = await build_region_map_async()
    return _region_map


async def handle_surname_search(surname: str, question: str) -> str:
    print(f"LLM-парсер определил фамилию: {surname}")

    words = re.findall(r"[А-ЯЁ][а-яё]+", question)
    full_name = extract_full_name(words, surname)

    docs = await search_with_fallback(full_name or surname, bool(full_name))
    if docs:
        # Параллельно обогащаем CC и (если нужно) готовим карту регионов/прайс
        need_price = bool(re.search(r"\b(прайс|стоимость|услуги|цены?|цена)\b", question.lower()))
        cc_task = asyncio.create_task(async_enrich_with_cc_info(docs))
        region_task = asyncio.create_task(get_region_map_async()) if need_price else None

        docs = await cc_task
        answer = format_documents(docs)

        if need_price:
            region_map = await region_task
            async def _price_one(doc):
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(format_doctor_prices, doc.get("id"), doc.get("fio"), region_map),
                        timeout=20,
                    )
                except Exception as e:
                    return "• Прайс временно недоступен."

            price_blocks = await asyncio.gather(*[_price_one(d) for d in docs], return_exceptions=False)
            for (doc, price) in zip(docs, price_blocks):
                answer += f"\n\n[Прайс по филиалам для {doc.get('fio')}]\n{price}"
        return answer

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

    # Доп. эвристика: одно слово, но фамилия не найдена — пробуем трактовать как специальность
    if surname and len(surname.split()) == 1:
        spec_try = normalize_specialty_term(surname)
        for variant in filter(None, [spec_try, surname.lower()]):
            try:
                docs = await find_doctors_by_keyword_async(variant)
                if docs:
                    return format_documents(docs)
            except Exception:
                pass

    return f"Врач с фамилией '{surname}' не найден в базе данных."


async def search_with_fallback(name: str, is_full_name: bool) -> List[Dict[str, Any]]:
    """
    Попытка найти документы локально, затем обновление репозитория и повторный поиск.
    """
    docs = (
        filter_docs(await repo.read_all_async(), name)
        if is_full_name
        else await repo.find_by_surname_async(name)
    )
    if docs:
        return docs

    await repo.update(get_all_doctors)

    return (
        filter_docs(await repo.read_all_async(), name)
        if is_full_name
        else await repo.find_by_surname_async(name)
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

    query = normalize_specialty_term(specialty) or specialty
    docs = await find_doctors_by_keyword_async(query)
    if not docs:
        # Попробуем без нормализации как запасной вариант
        if query != specialty:
            docs = await find_doctors_by_keyword_async(specialty)
        if not docs:
            return f"Врачи по специальности '{specialty}' не найдены."
    #
    # Пока отключим обогащение заметками колл-центра списка врачей.
    # if isinstance(docs, list):
    #     docs = enrich_with_cc_info(docs)
    #
    return format_documents(docs)


async def handle_timetable_search(surname: str, _: str) -> str:
    print(f"LLM-парсер определил запрос расписания по фамилии: {surname}")

    docs = await find_doctor_schedule_async(surname)
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


async def main():
    """Основная функция."""
    q = input("Введите вопрос: ")
    if os.getenv("NAUKA_PRICE_WARMUP", "0").strip() in ("1", "true", "yes"):
        try:
            await asyncio.wait_for(asyncio.to_thread(update_price_all), timeout=30)
        except Exception as e:
            print(f"[warmup] update_price_all skipped/failed: {e}")
    res = await investigate(q)
    print("\n" + "= " * 25)
    print(f"Ответ модели:\n\n{res}")


if __name__ == "__main__":
    # print_unique_priceall_regions()
    asyncio.run(main())
