# ЦЕПЬ ВЫЗОВОВ (вместе с модулем router_preprocessor.py):
# routing(..., think) → get_doc_info_from_api(..., think) →
# doctor_info.investigate(..., think) → extract_search_keyword_llm(..., think) → ollama_call(..., think).

"""Интеграция с CRM «Наука» для поиска врачей/расписаний и форматирования ответа.

Содержит:
- Простую нормализацию специальностей и фильтрацию по ФИО.
- Репозиторий врачей с локальным кэшем JSONL и ежедневным обновлением.
- Форматирование карточек врача/расписания, обогащение заметками call‑центра.
Внешние эффекты: сетевые запросы к CRM, чтение/запись в agent_logic_2/nayka_api/apidata.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from converters import html_cleaner
from agent_logic_2.text_fuzzy import fuzzy_match, normalize_text_for_fuzzy
from agent_logic_2.text_constants import (
    STOP_WORDS,
    UZI_QUERY_STOPWORDS,
    PROCEDURE_HINT_REGEX,
    PROCEDURE_QUERY_STOPWORDS,
    PROCEDURE_GENERIC_WORDS,
)
from agent_logic_2.ollama_settings import LLMName
from ollama import AsyncClient

from agent_logic_2.nayka_api.api_nayka import find_doctors_by_keyword, find_doctor_schedule, \
    cleanup_old_doctors_files, get_all_doctors, get_active_date_str
from nayka_api.api_price import load_doctor_prices, update_price_all, load_price_all
# from nayka_api.api_price_all import update_price_all, load_price_all
from nayka_api.doctors_cc_info import get_doctors_cc_info
from schedule_ttl_cache import AsyncListTTLStaleCache

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


def _runtime_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(getattr(c, name))
    except Exception:
        value = int(default)
    value = max(min_value, value)
    value = min(max_value, value)
    return value


def _runtime_bool(name: str, default: bool) -> bool:
    try:
        return bool(getattr(c, name))
    except Exception:
        return bool(default)


def _schedule_cache_key(last_name: str) -> str:
    return re.sub(r"\s+", " ", str(last_name or "").strip()).lower()

# Путь к данным о врачах
DATA_DIR = os.path.join(os.path.dirname(__file__), "nayka_api", "apidata")
# Инициализация Ollama
ollama_client = AsyncClient(c.ollama_url)
# Таймаут обращения к ollama
timeout: int = 300
# ollama_settings.init_model_name()
ollama_settings.init_options()

# model: str = ollama_settings.init_model_name()

NEGATIVE_CONTEXT_WORDS = {"кроме", "исключая", "исключением"}

# Шаблон для детекта УЗИ-запросов
UZI_RE = re.compile(r"\b(узи|узист|ультразвук\w*|ультразвуков\w*)\b", re.IGNORECASE)
UZI_ALT_RE = re.compile(
    r"\b(уздг|дуплекс\w*|триплекс\w*|допплер\w*|сканирован\w*)\b",
    re.IGNORECASE,
)
PROCEDURE_HINT_RE = re.compile(PROCEDURE_HINT_REGEX, re.IGNORECASE)

# Обобщённые слова, которые не несут смысла для типа УЗИ
UZI_GENERIC_WORDS = {
    "узи", "ультразвуковое", "ультразвуковая", "ультразвуковой", "исследование", "диагностика",
    "комплексное", "комплексная", "обследование",
}

UZI_EXCLUDE_KEYWORDS = {
    "под контрол", "пункц", "биопс", "инъекц", "операц", "дренирован", "лапароцентез",
}

_UZI_PROCEDURES_CACHE: list[Dict[str, Any]] | None = None

# Суффиксы для грубой нормализации русских слов (минимальная "лемматизация")
_RU_SUFFIXES: tuple[str, ...] = (
    "иями", "ями", "ами", "ями", "ыми", "ими",
    "иях", "ях", "ах", "ях",
    "ого", "его", "ому", "ему", "ыми", "ими",
    "ыми", "ими", "ыми", "ими",
    "ый", "ий", "ой", "ая", "яя", "ое", "ее",
    "ов", "ев", "ам", "ям", "ом", "ем",
    "ах", "ях", "ою", "ею", "ью",
    "а", "я", "ы", "и", "е", "о", "у", "ю", "ь", "й",
)


def _normalize_ru_token(token: str) -> str:
    t = token or ""
    for suf in _RU_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[:-len(suf)]
    return t


def _token_match(a: str, b: str) -> bool:
    if a == b:
        return True
    min_len = min(len(a), len(b))
    threshold = 0.75 if min_len <= 4 else 0.86
    return fuzzy_match(a, b, threshold=threshold)


def _is_procedure_like_query(text: str) -> bool:
    if not text:
        return False
    return bool(UZI_RE.search(text) or PROCEDURE_HINT_RE.search(text))


async def normalize_procedure_query_llm(query: str, think: bool | None = None) -> list[str]:
    """Нормализует запрос по процедуре в 1–4 канонических формулировки."""
    if not c.LLM_PROCEDURE_NORMALIZATION:
        logger.info("LLM procedure normalization disabled; using raw query")
        return [query]

    system = (
        "SYSTEM:\n"
        "Ты нормализуешь медицинскую процедуру/исследование.\n"
        "Верни только JSON вида {\"variants\":[...]}.\n"
        "variants — 1-4 коротких формулировки процедуры без врачей, адресов и пояснений.\n"
        "Если не уверен — верни исходную формулировку.\n"
    )
    prompt = system + f"\nUSER:\n{query}\n"
    try:
        resp = await ollama_call(prompt=prompt, llm=LLMName.get(), think=think)
        text = (resp.get("response") or "").strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return [query]
        data = json.loads(m.group(0))
        variants = data.get("variants")
        if not isinstance(variants, list):
            return [query]
        out = []
        for v in variants:
            v = str(v).strip()
            if v and v not in out:
                out.append(v)
        final_variants = out[:3] or [query]
        logger.info("LLM procedure variants: %s", final_variants)
        return final_variants
    except Exception:
        logger.info("LLM procedure normalization failed; using raw query")
        return [query]


def _find_docs_by_specialization_variants(variants: list[str], uzi_only: bool = False) -> list[Dict[str, Any]]:
    docs = []
    if not variants:
        return docs

    token_groups = []
    for v in variants:
        tokens = _uzi_tokens(v) if uzi_only else _procedure_tokens(v)
        if tokens:
            token_groups.append(tokens)
    if not token_groups:
        return docs

    for doc in repo.read_all():
        fio = (doc.get("fio") or "").strip()
        if not fio:
            continue
        spec = doc.get("specialization") or ""
        if not spec:
            continue
        if uzi_only and not (UZI_RE.search(spec) or UZI_ALT_RE.search(spec)):
            continue
        lines = [ln.strip(" \t•-") for ln in spec.splitlines() if ln.strip()]
        if uzi_only:
            i = 0
            while i < len(lines):
                line = lines[i]
                if not _uzi_line_is_candidate(line):
                    i += 1
                    continue
                block_tokens = _uzi_tokens(line)
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if _uzi_line_is_candidate(nxt):
                        break
                    block_tokens.extend(_uzi_tokens(nxt))
                    j += 1
                if any(all(any(_token_match(t, lt) for lt in block_tokens) for t in group) for group in token_groups):
                    docs.append(doc)
                    break
                i = j
        else:
            for line in lines:
                line_tokens = _procedure_tokens(line)
                if any(all(any(_token_match(t, lt) for lt in line_tokens) for t in group) for group in token_groups):
                    docs.append(doc)
                    break
    return docs

_PROCEDURE_CATALOG_CACHE: list[Dict[str, Any]] | None = None


def _uzi_tokens(text: str, drop_generic: bool = True) -> list[str]:
    norm = normalize_text_for_fuzzy(text)
    if not norm:
        return []
    tokens = [t for t in norm.split() if len(t) >= 3]
    if drop_generic:
        tokens = [t for t in tokens if t not in UZI_GENERIC_WORDS]
    return [_normalize_ru_token(t) for t in tokens]


def _procedure_tokens(text: str, drop_generic: bool = True) -> list[str]:
    norm = normalize_text_for_fuzzy(text)
    if not norm:
        return []
    tokens = [t for t in norm.split() if len(t) >= 3]
    if drop_generic:
        tokens = [t for t in tokens if t not in PROCEDURE_GENERIC_WORDS]
    return [_normalize_ru_token(t) for t in tokens]


def _uzi_line_is_candidate(line: str) -> bool:
    if not line or not (UZI_RE.search(line) or UZI_ALT_RE.search(line)):
        return False
    norm = normalize_text_for_fuzzy(line)
    if any(bad in norm for bad in UZI_EXCLUDE_KEYWORDS):
        return False
    return True


def extract_uzi_query_phrase(text: str) -> str:
    """Достаёт из запроса пользовательскую часть про УЗИ."""
    if not isinstance(text, str) or not text:
        return ""
    m = UZI_RE.search(text)
    src = text[m.start():] if m else text
    norm = normalize_text_for_fuzzy(src)
    if not norm:
        return ""
    tokens = [t for t in norm.split() if t and t not in UZI_QUERY_STOPWORDS]
    return " ".join(tokens)


def extract_procedure_query_phrase(text: str) -> str:
    """Достаёт из запроса пользовательскую часть про процедуру."""
    if not isinstance(text, str) or not text:
        return ""
    norm = normalize_text_for_fuzzy(text)
    if not norm:
        return ""
    tokens = [t for t in norm.split() if t and t not in PROCEDURE_QUERY_STOPWORDS]
    return " ".join(tokens)


def _build_uzi_catalog() -> list[Dict[str, Any]]:
    """Собирает каталог реальных УЗИ процедур (raw/norm/tokens)."""
    global _UZI_PROCEDURES_CACHE
    if _UZI_PROCEDURES_CACHE is not None:
        return _UZI_PROCEDURES_CACHE

    phrases: set[str] = set()

    # Из кэша прайса врача (берём самый свежий файл, без сетевых запросов)
    try:
        prices_dir = Path(DATA_DIR) / "doctor_prices"
        files = sorted(prices_dir.glob("doctor_prices_*.jsonl"))
        if files:
            with files[-1].open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    name = (row.get("serviceName") or "").strip()
                    if name and (UZI_RE.search(name) or UZI_ALT_RE.search(name)):
                        phrases.add(name)
    except Exception:
        pass

    # Из специализаций врачей (реальные формулировки)
    try:
        for doc in repo.read_all():
            spec = (doc.get("specialization") or "").splitlines()
            for line in spec:
                line = line.strip(" \t•-")
                if _uzi_line_is_candidate(line):
                    phrases.add(line)
    except Exception:
        pass

    catalog: list[Dict[str, Any]] = []
    for raw in sorted(phrases):
        norm = normalize_text_for_fuzzy(raw)
        tokens = _uzi_tokens(norm)
        if not tokens:
            continue
        catalog.append({
            "raw": raw,
            "norm": norm,
            "tokens": tokens,
            "key": " ".join(tokens),
        })

    _UZI_PROCEDURES_CACHE = catalog
    return catalog


def _build_procedure_catalog() -> list[Dict[str, Any]]:
    """Собирает каталог реальных процедур из doctor_prices (raw/tokens/doc_ids)."""
    global _PROCEDURE_CATALOG_CACHE
    if _PROCEDURE_CATALOG_CACHE is not None:
        return _PROCEDURE_CATALOG_CACHE

    catalog_map: dict[str, Dict[str, Any]] = {}

    try:
        prices = load_doctor_prices()
        for row in prices:
            if not isinstance(row, dict):
                continue
            name = (row.get("serviceName") or "").strip()
            doc_id = row.get("doctorId")
            if not name or not doc_id:
                continue
            tokens = _procedure_tokens(name)
            if not tokens:
                continue
            key = " ".join(tokens)
            entry = catalog_map.get(key)
            if not entry:
                entry = {"key": key, "tokens": tokens, "doc_ids": set(), "raws": set()}
                catalog_map[key] = entry
            entry["raws"].add(name)
            entry["doc_ids"].add(doc_id)
    except Exception:
        pass

    catalog: list[Dict[str, Any]] = []
    for entry in catalog_map.values():
        catalog.append({
            "key": entry["key"],
            "tokens": entry["tokens"],
            "doc_ids": entry["doc_ids"],
            "raws": entry["raws"],
        })

    _PROCEDURE_CATALOG_CACHE = catalog
    return catalog


def _uzi_match_groups(query: str) -> tuple[list[Dict[str, Any]], list[str]]:
    """Возвращает (matched_catalog_entries, query_tokens)."""
    phrase = extract_uzi_query_phrase(query)
    if not phrase:
        return [], []
    query_norm = normalize_text_for_fuzzy(phrase)
    query_tokens = _uzi_tokens(query_norm)
    if not query_tokens:
        return [], []

    catalog = _build_uzi_catalog()
    matched = [c for c in catalog if all(t in c["tokens"] for t in query_tokens)]
    if not matched:
        matched = [
            c for c in catalog
            if all(any(_token_match(t, ct) for ct in c["tokens"]) for t in query_tokens)
        ]
    if not matched:
        query_key = " ".join(query_tokens)
        matched = [c for c in catalog if fuzzy_match(query_key, c["key"], threshold=0.88)]
    return matched, query_tokens


def _procedure_match_groups(query: str) -> tuple[list[Dict[str, Any]], list[str]]:
    """Возвращает (matched_catalog_entries, query_tokens)."""
    phrase = extract_procedure_query_phrase(query)
    if not phrase:
        return [], []
    query_norm = normalize_text_for_fuzzy(phrase)
    query_tokens = _procedure_tokens(query_norm)
    if not query_tokens:
        return [], []

    catalog = _build_procedure_catalog()
    matched = [c for c in catalog if all(t in c["tokens"] for t in query_tokens)]
    if not matched:
        matched = [
            c for c in catalog
            if all(any(_token_match(t, ct) for ct in c["tokens"]) for t in query_tokens)
        ]
    if not matched:
        query_key = " ".join(query_tokens)
        matched = [c for c in catalog if fuzzy_match(query_key, c["key"], threshold=0.88)]
    return matched, query_tokens


def find_uzi_doctors_by_specialty(query: str) -> List[Dict[str, Any]]:
    """Находит врачей по УЗИ‑процедуре строго по полю specialization."""
    matched_catalog, query_tokens = _uzi_match_groups(query)
    token_groups = [c["tokens"] for c in matched_catalog] if matched_catalog else (
        [query_tokens] if query_tokens else []
    )
    matched: list[Dict[str, Any]] = []
    for doc in repo.read_all():
        fio = (doc.get("fio") or "").strip()
        if not fio:
            continue
        spec = doc.get("specialization") or ""
        if not spec or not (UZI_RE.search(spec) or UZI_ALT_RE.search(spec)):
            continue
        lines = [ln.strip(" \t•-") for ln in spec.splitlines() if ln.strip()]
        candidates = [ln for ln in lines if _uzi_line_is_candidate(ln)]
        if not candidates:
            continue
        if not token_groups:
            matched.append(doc)
            continue
        for line in candidates:
            line_tokens = _uzi_tokens(line)
            if any(all(any(_token_match(t, lt) for lt in line_tokens) for t in group) for group in token_groups):
                matched.append(doc)
                break
    return matched


def find_doctors_by_procedure(query: str) -> List[Dict[str, Any]]:
    """Находит врачей по процедуре (specialization + doctor_prices по флагу)."""
    phrase = extract_procedure_query_phrase(query)
    if not phrase:
        return []
    query_norm = normalize_text_for_fuzzy(phrase)
    query_tokens = _procedure_tokens(query_norm)
    if not query_tokens:
        return []

    matched_catalog: list[Dict[str, Any]] = []
    if c.USE_DOCTOR_PRICES_FOR_PROCEDURES:
        matched_catalog, _ = _procedure_match_groups(query)

    token_groups = [c_entry["tokens"] for c_entry in matched_catalog] if matched_catalog else [query_tokens]
    doc_ids: set[int] = set()
    for entry in matched_catalog:
        doc_ids.update(entry.get("doc_ids") or set())

    docs_by_id = {d.get("id"): d for d in repo.read_all() if (d.get("fio") or "").strip()}

    # Доп. фильтр: упоминание процедуры в специализации врача
    for doc in docs_by_id.values():
        spec = doc.get("specialization") or ""
        if not spec:
            continue
        lines = [ln.strip(" \t•-") for ln in spec.splitlines() if ln.strip()]
        for line in lines:
            line_tokens = _procedure_tokens(line)
            if any(all(any(_token_match(t, lt) for lt in line_tokens) for t in group) for group in token_groups):
                doc_ids.add(doc.get("id"))
                break

    return [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]


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
    """Локальный кэш данных о врачах (JSONL), с переключением активной даты.
    Читает/пишет файлы doctors_YYYYMMDD.jsonl, предоставляет быстрый доступ:
    read_all(), find_by_surname(), update(fetch_fn).
    """

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
Если в вопросе встречаются слова: "лор", "лор-врач", "оториноларинголог" — всегда возвращай Specialty: оториноларинголог.
Если в вопросе встречаются слова: "онкогинеколог", "онколог-гинеколог", "онкологический гинеколог" — всегда возвращай Specialty: онкогинеколог.
Если в вопросе есть только фамилия — верни: Surname: Иванов
Если в вопросе только специальность — верни: Specialty: кардиолог
Если вопрос про расписание (слова "расписание", "приём", "график работы", "время работы" и т.п.) и указано ФИО или фамилия, верни Timetable: Иванов
Если ничего не найдено — верни: NONE
Не добавляй других слов, никаких объяснений, только одну строку ответа!
"""
    user_part = f"\nUSER:\nВопрос: {question}\n"
    prompt = system_base + user_part

    resp = await ollama_call(prompt=prompt,
                             think=think)
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
    """
    Ищет в кеше наиболее похожую фамилию (по SequenceMatcher).
    Возвращает нормализованную фамилию, если схожесть ≥ threshold, иначе None.
    """
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
    """Кэширует заметки call‑центра по id врача с TTL, снижая нагрузку на API."""
    global _cc_map, _cc_ts
    now = time.time()
    if _cc_map is None or (now - _cc_ts) > CC_TTL or (_cc_map is not None and len(_cc_map) == 0):
        data = await asyncio.to_thread(get_doctors_cc_info)
        if not data:
            data = await asyncio.to_thread(get_doctors_cc_info, True)
        _cc_map = {row.get("id"): row.get("callCenterInfo", "Нет заметок") for row in (data or [])}
        _cc_ts = now
    return _cc_map


async def async_enrich_with_cc_info(doctors: list):
    """
    Асинхронное обогащение каждого dict заметкой call-центра с кешированием.
    """
    try:
        cc_by_id = await get_cc_map_cached()
        total = len(doctors)
        with_notes = 0
        for doc in doctors:
            doc_id = doc.get("id")
            note = cc_by_id.get(doc_id)
            if note:
                with_notes += 1
            doc["callCenterInfo"] = note or "Нет заметок"
        logger.info(
            "cc_enrich: docs=%d notes=%d missing=%d map=%d",
            total,
            with_notes,
            total - with_notes,
            len(cc_by_id),
        )
    except Exception as e:
        print(f"[DEBUG] enrich_with_cc_info error: {e}")
    return doctors


async def find_doctors_by_cc_notes_fallback_async(specialty: str, threshold: float = 0.86) -> List[Dict[str, Any]]:
    """
    Ищет врачей по заметкам call-центра с нестрогим совпадением.
    Используется как fallback, когда поиск по specialization/units ничего не дал.
    """
    if not specialty or not isinstance(specialty, str):
        return []

    query_norm = normalize_text_for_fuzzy(specialty)
    query_join = query_norm.replace(" ", "")
    if len(query_join) < 4:
        return []

    try:
        cc_by_id = await get_cc_map_cached()
    except Exception:
        return []

    docs = await repo.read_all_async()
    if not docs:
        return []

    matched: Dict[int, Dict[str, Any]] = {}
    for d in docs:
        did = d.get("id")
        if did is None:
            continue
        cc_text = cc_by_id.get(did)
        if not cc_text and isinstance(did, (int, str)):
            try:
                cc_text = cc_by_id.get(int(did))
            except (TypeError, ValueError):
                cc_text = None
        if not cc_text:
            continue

        cc_plain = html_cleaner.strip_html(str(cc_text)).replace("\xa0", " ")
        note_norm = normalize_text_for_fuzzy(cc_plain)
        if not note_norm:
            continue

        note_join = note_norm.replace(" ", "")
        if _has_negative_specialty_mention(note_norm, specialty, threshold):
            continue
        if query_norm in note_norm or (query_join and query_join in note_join):
            dd = dict(d)
            dd["callCenterInfo"] = cc_text
            try:
                key = int(did)
            except (TypeError, ValueError):
                key = did
            matched[key] = dd
            continue

        tokens = note_norm.split()
        if any(fuzzy_match(specialty, t, threshold) for t in tokens):
            dd = dict(d)
            dd["callCenterInfo"] = cc_text
            try:
                key = int(did)
            except (TypeError, ValueError):
                key = did
            matched[key] = dd
            continue

        if len(tokens) > 1:
            for i in range(len(tokens) - 1):
                joined = tokens[i] + tokens[i + 1]
                if fuzzy_match(specialty, joined, threshold):
                    dd = dict(d)
                    dd["callCenterInfo"] = cc_text
                    try:
                        key = int(did)
                    except (TypeError, ValueError):
                        key = did
                    matched[key] = dd
                    break

    return list(matched.values())


# ── Форматирование ответа ────────────────────────────────────────────────────
def format_doctor(item: Dict[str, Any]) -> str:
    """Форматирует карточку врача: ФИО, спец-ть, адреса, заметка КЦ (очищенный HTML)."""
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
    """Формирует блок расписания: под каждую локацию выводит даты, окна и дополнительные комментарии. На вход принимает расписание в формате API."""
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

_SCHEDULE_CACHE = AsyncListTTLStaleCache(
    fresh_ttl_seconds=_runtime_int("MR_SCHEDULE_FRESH_TTL_SECONDS", 30, min_value=1, max_value=300),
    stale_ttl_seconds=_runtime_int("MR_SCHEDULE_STALE_TTL_SECONDS", 600, min_value=1, max_value=3600),
    negative_ttl_seconds=_runtime_int("MR_SCHEDULE_NEGATIVE_TTL_SECONDS", 15, min_value=1, max_value=120),
    max_keys=_runtime_int("MR_SCHEDULE_CACHE_MAX_KEYS", 1000, min_value=50, max_value=10000),
    logger=logger,
    log_events=_runtime_bool("MR_SCHEDULE_CACHE_LOG_EVENTS", False),
    name="schedule_cache",
    time_func=lambda: time.time(),
)


# ── Асинхронные обёртки для синхронных I/O/API ────────────────────────────────
async def find_doctors_by_keyword_async(q: str):
    return await asyncio.to_thread(find_doctors_by_keyword, q)


async def _find_doctor_schedule_source_async(surname: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            return await asyncio.to_thread(find_doctor_schedule, surname)
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(0.12)
                continue
    if last_exc is not None:
        raise last_exc
    return []


async def find_doctor_schedule_async(surname: str):
    key = _schedule_cache_key(surname)
    if not key:
        return []
    return await _SCHEDULE_CACHE.get_or_fetch(
        key,
        lambda: _find_doctor_schedule_source_async(surname),
        key_details={"last_name": key},
    )


async def load_doctor_prices_async():
    return await asyncio.to_thread(load_doctor_prices)


@with_retries(tries=2)
async def ollama_call(prompt: str, llm: str = LLMName.get(), think: bool = None, ) -> Dict[str, Any]:
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
            think=think,
        ),
        timeout=timeout,
    )

    return res.__dict__


async def investigate(question: str, think: bool = None) -> str:
    """Определяет тип запроса (фамилия/спец-ть/расписание) и возвращает отформатированный ответ.
    Args:
        question: Исходный текст пользователя.
        think: Флаг reasoning для LLM.
    Returns:
        Готовый текстовый ответ (карточки, список, расписание или пояснение).
    """
    print("\n=== Начало обработки вопроса ===")
    print(f"Вопрос: {question}")
    # Быстрый путь: если запрос похож на одиночную фамилию — пропускаем LLM-парсер
    q = (question or "").strip()
    if q and len(q.split()) == 1 and is_potential_surname(q):
        try:
            return await handle_surname_search(q, question)
        except Exception:
            pass

    # Эвристика: если в вопросе есть потенциальная фамилия, и она есть в базе — обрабатываем как поиск по фамилии.
    timetable_keywords = ("распис", "график", "прием", "приём", "schedule")
    has_timetable_intent = any(kw in q.lower() for kw in timetable_keywords)

    if not has_timetable_intent:
        tokens = re.findall(r"[А-ЯЁа-яё\-]+", q)
        for raw_word in tokens:
            candidate = raw_word if raw_word[:1].isupper() else raw_word.capitalize()
            if not is_potential_surname(candidate):
                continue
            try:
                docs = await repo.find_by_surname_async(candidate)
            except Exception:
                docs = []
            if docs:
                try:
                    return await handle_surname_search(candidate, question)
                except Exception:
                    break

    # УЗИ-запрос: ищем по специализации и реальным формулировкам процедур
    if UZI_RE.search(q):
        try:
            variants = await normalize_procedure_query_llm(q, think=think)
            docs = _find_docs_by_specialization_variants(variants, uzi_only=True)
            if not docs:
                docs = find_uzi_doctors_by_specialty(q)
            if docs:
                docs = await async_enrich_with_cc_info(docs)
                return format_documents(docs)
        except Exception:
            pass

    # Процедурный запрос: ищем по процедурам в прайсах/специализации
    procedure_like = _is_procedure_like_query(q)
    logger.info(
        "procedure_like=%s hint_match=%s",
        procedure_like,
        bool(PROCEDURE_HINT_RE.search(q)),
    )
    if procedure_like:
        try:
            variants = await normalize_procedure_query_llm(q, think=think)
            docs = _find_docs_by_specialization_variants(variants, uzi_only=False)
            if not docs:
                for v in variants:
                    docs = find_doctors_by_procedure(v)
                    if docs:
                        break
            if docs:
                docs = await async_enrich_with_cc_info(docs)
                return format_documents(docs)
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
    """Получает карту регионов из кеша или строит её заново."""
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
    """Ищет врача по фамилии/ФИО и формирует карточки; при необходимости добавляет прайс.
    Аргументы:
        surname: Определённая фамилия (или ФИО).
        question: Исходный запрос (для эвристик, например «цены»).
    Возвращает:
        Отформатированный список карточек (1..N) или сообщение об отсутствии данных.
    """
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
                        timeout=50,
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
    """Фильтрует список врачей по фамилии."""
    matches = [d for d in docs if d.get("fio", "").startswith(full_name)]
    return matches[:1] if len(matches) > 1 else matches


def format_documents(docs: List[Dict[str, Any]]) -> str:
    """Форматирует список врачей в строку: нумерует и форматирует каждого врача."""
    return FORMATTER.join(
        f"{i + 1}. {format_doctor(d)}" for i, d in enumerate(docs)
    )


def _has_negative_specialty_mention(text: str, term: str, threshold: float = 0.86) -> bool:
    """Проверяет, что термин упомянут в негативном контексте (например: «кроме ...»)."""
    norm = normalize_text_for_fuzzy(text)
    tokens = norm.split()
    if not tokens:
        return False

    def has_negative_window(idx: int) -> bool:
        window = tokens[max(0, idx - 3):idx]
        return any(w in NEGATIVE_CONTEXT_WORDS for w in window)

    for i, t in enumerate(tokens):
        if fuzzy_match(term, t, threshold) and has_negative_window(i):
            return True

    for i in range(len(tokens) - 1):
        joined = tokens[i] + tokens[i + 1]
        if fuzzy_match(term, joined, threshold) and has_negative_window(i):
            return True

    return False


async def handle_specialty_search(specialty: str, _: str) -> str:
    """Ищет врачей по специальности и возвращает компактный список карточек."""
    print(f"LLM-парсер определил специальность: {specialty}")

    query = normalize_specialty_term(specialty) or specialty
    docs = await find_doctors_by_keyword_async(query)
    if docs:
        filtered = [
            d for d in docs
            if not _has_negative_specialty_mention(d.get("specialization", ""), query)
        ]
        if filtered:
            docs = filtered
        else:
            docs = []
    if not docs:
        # Попробуем без нормализации как запасной вариант
        if query != specialty:
            docs = await find_doctors_by_keyword_async(specialty)
            if docs:
                filtered = [
                    d for d in docs
                    if not _has_negative_specialty_mention(d.get("specialization", ""), specialty)
                ]
                if filtered:
                    docs = filtered
                else:
                    docs = []
        if not docs:
            docs = await find_doctors_by_cc_notes_fallback_async(query)
            if not docs and query != specialty:
                docs = await find_doctors_by_cc_notes_fallback_async(specialty)
            if not docs:
                return f"Врачи по специальности '{specialty}' не найдены."
    #
    # Пока отключим обогащение заметками колл-центра списка врачей.
    # if isinstance(docs, list):
    #     docs = enrich_with_cc_info(docs)
    #
    return format_documents(docs)


async def handle_timetable_search(surname: str, _: str) -> str:
    """Ищет расписание врача(ей) по фамилии и возвращает пронумерованный список."""
    print(f"LLM-парсер определил запрос расписания по фамилии: {surname}")

    docs = await find_doctor_schedule_async(surname)
    if isinstance(docs, str):
        return docs

    return FORMATTER.join(
        f"{i + 1}. {format_doctor_schedule(doc)}"
        for i, doc in enumerate(docs)
    )


def find_doctors_by_keyword_llm(question: str) -> str:
    """Ищет врачей по ключевому слову."""
    return find_doctors_by_keyword(question)


def print_unique_priceall_regions():
    """Выводит уникальные регионы из priceAll."""
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
