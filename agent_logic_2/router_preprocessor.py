"""
Маршрутизация пользовательских запросов: сегментация, классификация, вызов модулей.

Пайплайн:
1) split_into_segments() — переформулировка и разбиение текста.
2) classify() — присвоение label‑ов (API_INFO/APPOINTMENT/SCRIPTS/NEWS/UNDEFINED).
3) process_segments() — логика FILTER/эвристик/быстрых путей и сбор ответов.
4) routing() — асинхронный генератор ответов (стриминг), финальный постпроцесс.

Особенности:
- «SAFE_LIST» детерминированный формат для списков врачей (без участия LLM).
- «<NO_POSTPROC>» позволяет миновать финальную постобработку.
- Фильтры: ARRIVING/CHILDREN/DMS из явных меток, ключевых слов и возрастных паттернов.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, AsyncGenerator, TypeAlias, Literal, Optional

from ollama import AsyncClient

import agent_logic_2.ollama_settings as ollama_settings
from agent_logic_2 import llama_func_call as doctor_info, config as c
from agent_logic_2.gigachat import async_gigachat_logic as gigachat
from agent_logic_2.llama_func_call import repo
from agent_logic_2.nayka_api.api_nayka import ensure_daily_refresh_started
from agent_logic_2.nayka_api.doctors_cc_info import get_doctors_cc_info
from agent_logic_2.prompts import load_prompt
from agent_logic_1 import meilisearch_client as meilisearch
from converters import html_cleaner
from agent_logic_2.text_fuzzy import fuzzy_match, normalize_text_for_fuzzy

#  Инициализация logging для понимания логики роутера
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# =========================
# КОНСТАНТЫ И НАСТРОЙКИ
# =========================

# Label‑ы и порядок в LABEL_PRIORITY:
LABEL_PRIORITY = load_prompt("LABEL_PRIORITY", False).split(",")
ALLOWED = set(LABEL_PRIORITY + ["UNDEFINED"])

# Какой индекс MEILISEARCH открывать для конкретного лейбла:
INDEX_BY_LABEL: dict[str, str] = {
    "NEWS": "news",
    "SCRIPTS": "main_index",
}

# Константы форматирования
LABEL_DOC = load_prompt("LABEL_DOC", False)
RAW_MODE_MARKER = "<NO_POSTPROC>"
EXAMPLES = load_prompt("EXAMPLES", False)
KNOWLEDGE_MARKER = "{KNOWLEDGE_SNIPPET}"
ITEM_MARKER = "⊢ID:"
SEGMENT_SEPARATOR = "\n\n— — —\n\n"

# Регулярные выражения
NOTE_WORD_RE = re.compile(r"(замет\w*|кнопк\w*|кнопоч\w*)", re.IGNORECASE)
MANAGER_WORD_RE = re.compile(r"\bменеджер\w*", re.IGNORECASE)
SCRIPT_WORD_RE = re.compile(r"\bскрипт\w*", re.IGNORECASE)
NOTE_SPLIT_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
FILTER_RE = re.compile(r'^\s*FILTER:\s*(.+)$', re.IGNORECASE)

# Стоп-слова для заметок
NOTE_STOPWORDS = {
    "список", "списке", "врач", "врачи", "врачей", "у", "кого", "есть",
    "со", "словом", "слово", "какие", "каких", "какая", "каком", "в каких",
    "по", "про", "покажи", "покажите", "выведи", "выведите", "выдай", "выдайте",
    "найди", "найдите", "найти", "на", "в", "и", "или", "что", "всех",
    "заметка", "заметки", "заметках", "заметке", "заметок",
    "кнопка", "кнопки", "кнопке", "кнопку", "кнопок",
    "кнопочка", "кнопочки", "кнопочке", "кнопочку", "кнопочек",
    "присутствует", "присутствуют", "встречается", "встречаются", "содержится", "содержатся",
    "информация", "информацию", "об", "о", "обо"
}

NOTE_TRIGGER_WORDS = ("заметка", "кнопка")

# Возрастные паттерны
AGE_PATTERNS: tuple[str, ...] = (
    r"\b[сc]\s*(\d{1,2})\s*(?:-?[а-я]{1,3})?\s*лет\b",
    r"возраст[а-я\s]*?(?:[сc]\s*)?(\d{1,2})\s*(?:-?[а-я]{1,3})?\s*лет",
    r"\b[сc]\s*возраста\s*(\d{1,2})\s*(?:-?[а-я]{1,3})?\s*лет\b",
)

# Ключевые слова для фильтров
KEYWORDS_TRUE = {
    'dms': [
        'по дмс', 'принимает по дмс', 'принимают по дмс', 'работает по дмс', 'работают по дмс', 'есть дмс', 'дмс да'
    ],
    'children': [
        'работает с детьми', 'работают с детьми', 'принимает детей', 'принимают детей', 'детей принимает', 'детям',
        'с детьми', 'дети', 'детский', 'детская', 'детские', 'детские врачи', 'детский врач', 'с детьми'
    ],
    'arriving': [
        'приходящий', 'приходящие', 'разовый', 'совмещает'
    ],
}

KEYWORDS_FALSE = {
    'dms': [
        'не по дмс', 'не принимает по дмс', 'не принимают по дмс', 'без дмс', 'дмс нет', 'нет дмс'
    ],
    'children': [
        'не работает с детьми', 'не работают с детьми', 'без детей', 'детей не принимает', 'детей не принимают',
        'только взросл'
    ],
    'arriving': [
        'не приходящий', 'не приходящие', 'неприходящий', 'неприходящие'
    ],
}

# Константы для парсинга фильтров
DEF_TRUE = {"yes", "да", "true", "1"}
DEF_FALSE = {"no", "нет", "false", "0"}


# =========================
# УТИЛИТАРНЫЕ КЛАССЫ
# =========================

class CCNotesProcessor:
    """Обработка заметок call-центра"""

    @staticmethod
    def normalize_text(cc: str | None) -> str:
        """Очищает HTML, заменяет NBSP на пробел и приводит к lower()."""
        if not cc:
            return ""
        return html_cleaner.strip_html(cc).replace("\xa0", " ").lower()

    @staticmethod
    def extract_age(cc: str) -> str:
        """Пытается вытащить возраст из заметки КЦ и вернуть нормализованную строку вида "с N лет"."""
        if not isinstance(cc, str) or not cc:
            return "не указан"

        s = CCNotesProcessor.normalize_text(cc)

        # Возрастные формулировки
        for pat in AGE_PATTERNS + (r"принимает\s*(пациентов\s*)?[сc]\s*(\d{1,2})\s*(?:-?и)?\s*лет",):
            m = re.search(pat, s)
            if m:
                num = (m.group(2) or m.group(1)) if (m.lastindex or 0) >= 2 else m.group(1)
                return f"с {num} лет"

        # Дополнительные паттерны
        patterns = [
            (r"\bв\s*возрасте\s*(\d{1,2})\s*(?:-?и)?\s*лет\b", "с {0} лет"),
            (r"\bв\s*возрасте\s*(\d{1,2})\s*\+", "с {0} лет"),
            (r"\b[сc]\s*возраста\s*(\d{1,2})\s*(?:-?и)?\s*лет\b", "с {0} лет"),
            (r"принимает\s*(пациентов\s*)?[сc]\s*(\d{1,2})\s*(?:-?и)?\s*лет", "с {1} лет"),
        ]

        for pattern, template in patterns:
            m = re.search(pattern, s)
            if m:
                num = m.group(2) if (m.lastindex or 0) >= 2 else m.group(1)
                return template.format(num)

        # Специальные случаи
        if "совершеннолет" in s:
            return "с 18 лет"
        if "0+" in s or re.search(r"\b[сc]\s*0\s*лет\b", s) or re.search(r"\b0\s*\+", s):
            return "с 0 лет"
        if "только взросл" in s or "взросл" in s or re.search(r"\b[сc]\s*18\s*лет\b", s):
            return "с 18 лет"

        return "не указан"

    @staticmethod
    def get_flag(cc: str, key: str) -> Optional[bool]:
        """Определяет флаг из заметки call-центра."""
        if not cc:
            return None

        s = CCNotesProcessor.normalize_text(cc)

        if key == 'arriving':
            if 'не приход' in s or 'неприход' in s:
                return False
            if 'приходящ' in s:
                return True
            return None

        if key == 'children':
            # Явные отрицания/только взрослые/совершеннолетние
            if (
                re.search(r"\b[сc]\s*18\s*лет\b", s)
                or re.search(r"принимает\s*[сc]\s*18", s)
                or 'только взросл' in s
                or 'взросл' in s
                or 'совершеннолет' in s
            ):
                return False
            # Явные указания работы с детьми
            if 'дет' in s and ('работ' in s or 'принимает' in s):
                return True
            # Возрастной признак
            for pat in AGE_PATTERNS:
                m = re.search(pat, s)
                if m:
                    try:
                        n = int(m.group(1))
                        return n < 18
                    except Exception:
                        continue
            return None

        if key == 'dms':
            if re.search(
                    r"(?:не\s*(?:принима(ет|ют)|работа(ет|ют))\s*по\s*дмс|не\s*по\s*дмс|без\s*дмс|дмс\s*(?:[:\-—]\s*)?нет)",
                    s):
                return False
            if re.search(r"(?:по\s*дмс|дмс\s*(?:[:\-—]\s*)?да)", s):
                return True
            return None

        return None

    @staticmethod
    def extract_note_keywords(segment: str) -> List[str]:
        """Выделяет ключевые слова/фразы из запроса про заметки."""
        if not isinstance(segment, str) or not segment.strip():
            return []

        # Берём только хвост после первого «заметк*»
        m = NOTE_WORD_RE.search(segment)
        tail = segment[m.end():] if m else segment

        # Извлекаем кавычённые фразы
        quoted: List[str] = []

        def _drop(m):
            for g in m.groups():
                if g:
                    q = g.strip().lower()
                    if q:
                        quoted.append(q)
            return " "

        cleaned = re.sub(r"\"([^\"]+)\"|'([^']+)'|«([^»]+)»", _drop, tail)

        # Токенизация и фильтрация
        tokens = [tok for tok in NOTE_SPLIT_RE.split(cleaned.lower()) if tok]

        seen = set()
        keywords: List[str] = []

        def _push(term: str):
            if not term or term in seen:
                return
            seen.add(term)
            keywords.append(term)

        for tok in tokens:
            if tok in NOTE_STOPWORDS or len(tok) < 2:
                continue
            if any(fuzzy_match(tok, w, 0.84) for w in NOTE_TRIGGER_WORDS):
                continue
            _push(tok)

        for phrase in quoted:
            _push(phrase)

        return keywords


class SegmentProcessor:
    """Обработка сегментов текста"""

    @staticmethod
    def check_pattern_match(segment: str, text: str, pattern: re.Pattern) -> Tuple[bool, bool]:
        """Универсальная проверка паттернов в сегменте и тексте."""
        hit_in_segment = bool(pattern.search(segment))
        hit_in_full = bool(pattern.search(text)) if text else False
        return hit_in_segment, hit_in_full

    @staticmethod
    def extract_specialty_terms(segment: str, docs: List[Dict[str, Any]]) -> List[str]:
        """Пытается извлечь возможные ключевые слова специальности из сегмента."""
        if not isinstance(segment, str) or not segment.strip():
            return []

        seg = segment.lower()
        seg = re.sub(r"[,.!?;:()\[\]{}]", " ", seg)
        tokens = [t for t in re.split(r"\s+", seg) if t]

        # Уберём частые служебные слова и слова фильтров
        stop = set(getattr(doctor_info, 'STOP_WORDS', set())) | {
            'дмс', 'страховка', 'страховой', 'страховая', 'по', 'с', 'без',
            'дети', 'детям', 'детей', 'взрослые', 'взрослый', 'приходящий', 'приходящие', 'неприходящий',
            'принимает', 'работает', 'где', 'кто', 'список', 'нужен', 'ищу'
        }
        tokens = [t for t in tokens if t not in stop and len(t) >= 3]

        # Морфологическая нормализация
        norm_tokens: List[str] = []
        for t in tokens:
            try:
                if hasattr(doctor_info, 'normalize_specialty_term'):
                    nt = doctor_info.normalize_specialty_term(t) or t
                else:
                    nt = t[:-1] if (len(t) > 4 and (t.endswith('и') or t.endswith('ы'))) else t
            except Exception:
                nt = t
            norm_tokens.append(nt)

        # Базовые синонимы
        synonyms = {
            'лор': ['отоларинголог', 'оториноларинголог', 'лор-врач'],
            'узи': ['ультразвуков'],
            'узист': ['ультразвуков'],
        }
        expanded_tokens: List[str] = []
        for t in norm_tokens:
            expanded_tokens.append(t)
            vals = synonyms.get(t)
            if isinstance(vals, list):
                expanded_tokens.extend(vals)
            elif isinstance(vals, str):
                expanded_tokens.append(vals)

        if not expanded_tokens:
            return []

        # Построим текст для поиска по каждому врачу
        texts = []
        for d in docs:
            spec = (d.get('specialization') or '').lower()
            units = ", ".join(d.get('units') or [])
            texts.append(spec + " " + units.lower())

        # Оставляем только те термины, которые где-то реально встречаются
        valid_terms = []
        for t in expanded_tokens:
            if any(t in txt for txt in texts):
                valid_terms.append(t)

        # Уберём дубликаты, сохранив порядок
        seen = set()
        uniq_terms = []
        for t in valid_terms:
            if t not in seen:
                seen.add(t)
                uniq_terms.append(t)
        return uniq_terms


class FilterProcessor:
    """Обработка фильтров"""

    @staticmethod
    def parse_filters(expr: str) -> Dict[str, bool]:
        """Парсит строку фильтров и возвращает словарь с фильтрами."""
        pairs = [p.strip() for p in expr.split(';') if p.strip()]
        out: Dict[str, bool] = {}
        for p in pairs:
            if '=' not in p:
                continue
            k, v = [t.strip().lower() for t in p.split('=', 1)]
            if v in DEF_TRUE:
                out[k] = True
            elif v in DEF_FALSE:
                out[k] = False
        return out

    @staticmethod
    def keyword_to_filter(segment: str, original: str | None = None) -> Dict[str, Any] | None:
        """Грубое извлечение фильтров из естественных формулировок."""
        if not isinstance(segment, str) or not segment.strip():
            return None

        s = segment.lower()
        out: Dict[str, Any] = {}

        # DMS
        if re.search(r"\b(не\s*(принима(ет|ют)|работа(ет|ют)|по)\s*по\s*дмс|без\s*дмс|дмс\s*нет|нет\s*дмс)\b", s):
            out['dms'] = False
        elif re.search(r"\b(принима(ет|ют)\s*по\s*дмс|работа(ет|ют)\s*по\s*дмс|по\s*дмс|дмс\s*да)\b", s):
            out['dms'] = True

        # CHILDREN
        if re.search(r"\b(не\s*работа(ет|ют)\s*с\s*детьми|без\s*детей|детей\s*не\s*принима(ет|ют)|только\s*взросл)\b",
                     s):
            out['children'] = False
        elif any(k in s for k in KEYWORDS_TRUE['children']):
            out['children'] = True

        # ARRIVING
        if any(k in s for k in KEYWORDS_FALSE['arriving']):
            out['arriving'] = False
        elif any(k in s for k in KEYWORDS_TRUE['arriving']):
            out['arriving'] = True

        # NOTE keywords
        note_terms: List[str] = []
        sources: List[str] = []
        if isinstance(segment, str):
            sources.append(segment)
        if original and original not in sources:
            sources.append(original)

        for src in sources:
            if not src or not NOTE_WORD_RE.search(src):
                continue
            for kw in CCNotesProcessor.extract_note_keywords(src):
                if kw not in note_terms:
                    note_terms.append(kw)

        if note_terms:
            out.setdefault('notes', note_terms)

        # Дополнительное правило: "с N лет" → children=True (если N < 18)
        m = re.search(r"\bс\s*(\d{1,2})\s*(?:-?[а-я]{1,3})?\s*лет\b", s)
        if m:
            try:
                n = int(m.group(1))
                if n < 18:
                    out.setdefault('children', True)
                else:
                    out.setdefault('children', False)
            except Exception:
                pass

        return out or None


# =========================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И ИНИЦИАЛИЗАЦИЯ
# =========================

# LLM‑клиент для классификации входящих запросов
ollama = AsyncClient(c.ollama_url)

# CC‑info thin process cache (по id врача)
_CC_INFO_MAP: Dict[int, str] | None = None


def _get_cc_info_map() -> Dict[int, str]:
    """Получает информацию о заметках call-центра по врачам."""
    global _CC_INFO_MAP
    if _CC_INFO_MAP is None:
        try:
            data = get_doctors_cc_info()
            _CC_INFO_MAP = {
                int(row.get('id', 0)): row.get('callCenterInfo', '')
                for row in data if isinstance(row, dict) and row.get('id') is not None
            }
        except Exception:
            _CC_INFO_MAP = {}
    return _CC_INFO_MAP


# --- ensure doctors repo is warmed up (file may be absent on first FILTER run) ---
async def _ensure_doctors_repo_loaded() -> None:
    """Обеспечивает загрузку кэша врачей (JSONL) и его обновление из API при необходимости."""
    try:
        # стараемся запустить планировщик ежедневного обновления (внутри активного event loop)
        try:
            ensure_daily_refresh_started()
        except Exception:
            pass
        # try fast-path
        data = repo.read_all()
        if not data:
            raise RuntimeError("doctors repo is empty")
    except Exception:
        # Fallback: обновляем репозиторий напрямую из API без участия LLM
        try:
            await repo.update(doctor_info.get_all_doctors)
        except Exception:
            pass


# =========================
# УТИЛИТАРНЫЕ ФУНКЦИИ ДЛЯ ОБРАБОТКИ СЕГМЕНТОВ
# =========================

async def _try_doctor_fallback(segment: str, sess: SessionType, think: bool | None, context: str = "") -> Optional[str]:
    """Универсальная функция для fallback к doctor_info."""
    try:
        if await is_possible_surname_or_specialty(segment):
            logger.debug(f"[{context}] doctor_info fallback")
            response, _ = await get_doc_info_from_api(segment, session=sess, think=think)
            return RAW_MODE_MARKER + "\n" + response
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("_try_doctor_fallback ОСТАНОВЛЕН")
        # Отмена сверху => закроется HTTP-стрим клиента => Ollama прекращает генерацию
        raise
    except Exception as e:
        logger.exception(f"[{context}] fallback check error: %s", e)
    return None


async def _handle_note_search(segment: str, text: str) -> Optional[str]:
    """Обработка поиска по заметкам."""
    note_hit_seg, note_hit_full = SegmentProcessor.check_pattern_match(segment, text, NOTE_WORD_RE)
    if not (note_hit_seg or note_hit_full):
        def _fuzzy_hit(src: str) -> bool:
            if not src:
                return False
            norm = normalize_text_for_fuzzy(src)
            for tok in norm.split():
                if any(fuzzy_match(tok, w, 0.84) for w in NOTE_TRIGGER_WORDS):
                    return True
            return False

        note_hit_seg = _fuzzy_hit(segment)
        note_hit_full = _fuzzy_hit(text)
    if note_hit_seg or note_hit_full:
        search_source = segment if note_hit_seg else text
        logger.debug("CC NOTES search (source=%s)", "segment" if note_hit_seg else "full")
        return await _search_in_cc_notes(search_source, fallback_segment=text, match_mode="all")
    return None


async def _handle_manager_search(segment: str, text: str) -> Optional[str]:
    """Обработка поиска по менеджерам."""
    manager_hit_seg, manager_hit_full = SegmentProcessor.check_pattern_match(segment, text, MANAGER_WORD_RE)
    if manager_hit_seg or manager_hit_full:
        search_source = segment if manager_hit_seg else text
        logger.debug("MANAGER search (source=%s)", "segment" if manager_hit_seg else "full")
        response, _ = await instructions_search(search_source, index="main_index", manager_mode=True)
        return response
    return None


async def _handle_filter_search(segment: str, text: str) -> Optional[str]:
    """Обработка фильтров."""
    # Проверяем явные фильтры
    m = FILTER_RE.match(segment)
    if m:
        flt = FilterProcessor.parse_filters(m.group(1))
        logger.debug("FILTER explicit -> %s", flt)
        return await _filter_doctors_via_cc_info(flt, segment=segment, fallback_segment=text)

    # Проверяем ключевые слова
    kw_filters = FilterProcessor.keyword_to_filter(segment, text)
    if kw_filters:
        logger.debug("FILTER kw -> %s", kw_filters)
        return await _filter_doctors_via_cc_info(kw_filters, segment=segment, fallback_segment=text)

    return None


# Удалены дублированные определения - используются из утилитарных классов


# Удалена - заменена на CCNotesProcessor.extract_age()


def _bool_to_ru(v: Optional[bool]) -> str:
    return "Да" if v is True else ("Нет" if v is False else "—")


# Удалена - заменена на CCNotesProcessor.extract_note_keywords()


def _compact_line(d: dict) -> str:
    """Форматирует строку компактного формата для отображения информации о враче."""
    fio = (d.get("fio") or "").strip()
    spec = d.get("units") or d.get("specialization") or ""
    if isinstance(spec, list):
        spec = ", ".join(spec[:2])
    addr_list = d.get("regions") or []
    addr = ", ".join(addr_list[:2])

    # исходная заметка КЦ (как есть)
    cc_text = d.get("callCenterInfo") or ""

    # нормализованные флаги через единую функцию
    arriving_flag = CCNotesProcessor.get_flag(cc_text, "arriving")
    dms_flag = CCNotesProcessor.get_flag(cc_text, "dms")
    children_flag = CCNotesProcessor.get_flag(cc_text, "children")

    age = CCNotesProcessor.extract_age(cc_text)

    return (
        f"{ITEM_MARKER}{d.get('id')} — {fio} — {spec} — {addr} — "
        f"ДМС: {_bool_to_ru(dms_flag)} — "
        f"Приходящий: {_bool_to_ru(arriving_flag)} — "
        f"Дети: {_bool_to_ru(children_flag)} — "
        f"Возраст: {age}"
    )


def _format_compact(matched: List[Dict[str, Any]]) -> str:
    """Форматирует список врачей в строку компактного формата для отображения информации о врачах."""
    lines = [_compact_line(d) for d in matched]
    n = len(lines)
    header = f"[SAFE_LIST]\nN={n}\nORDER=PRESERVE\nFORMAT=ECHO_ALL\n"
    return header + "\n".join(lines)


def _extract_safe_list_meta(s: str) -> Tuple[int, List[str] | None]:
    """Returns (N, ids) if [SAFE_LIST] is present, else (0, None)."""
    if "[SAFE_LIST]" not in s:
        return 0, None
    try:
        head, *_ = s.split("\n\n", 1)
        m = re.search(r"\bN=(\d+)", head)
        n = int(m.group(1)) if m else 0
        ids = re.findall(rf"{re.escape(ITEM_MARKER)}(\d+)", s)
        return n, ids
    except Exception:
        return 0, None


# =========================
# SAFE_LIST deterministic helpers
# =========================
def _parse_safe_compact_line(line: str) -> Dict[str, str]:
    """Парсит строку формата
    ⊢ID:123 — Иванов И.И. — Врач терапевт — Ленина 5 — ДМС: Да — Приходящий: Нет — Дети: Да — Возраст: с 18 лет
    и возвращает словарь полей. Не бросает исключений.
    """
    out = {"id": "", "fio": "", "spec": "", "addr": "", "dms": "—", "arriving": "—", "children": "—",
           "age": "не указан"}
    if not isinstance(line, str) or ITEM_MARKER not in line:
        return out
    try:
        head = line.split(ITEM_MARKER, 1)[1].strip()
        parts = [p.strip() for p in head.split(" — ")]
        if parts:
            out["id"] = parts[0]
        if len(parts) > 1:
            out["fio"] = parts[1]
        if len(parts) > 2:
            out["spec"] = parts[2]
        if len(parts) > 3:
            out["addr"] = parts[3]
        # Остальные метки в произвольном порядке
        tail = " — ".join(parts[4:]) if len(parts) > 4 else ""
        tl = tail.lower()
        # ДМС
        m = re.search(r"дмс:\s*(да|нет)", tl)
        if m:
            out["dms"] = "Да" if m.group(1) == "да" else "Нет"
        # Приходящий: учитываем ТОЛЬКО явные метки "да"/"нет", без эвристик
        # Если метки нет — оставляем значение по умолчанию "—" (неизвестно)
        m = re.search(r"\bприходящий:\s*(да|нет)\b", tl)
        if m:
            out["arriving"] = "Да" if m.group(1) == "да" else "Нет"
        # Дети
        m = re.search(r"дети:\s*(да|нет)", tl)
        if m:
            out["children"] = "Да" if m.group(1) == "да" else "Нет"
        # Возраст
        m = re.search(r"возраст:\s*([^—\n]+)", tail, flags=re.IGNORECASE)
        if m:
            out["age"] = m.group(1).strip()
    except Exception:
        pass
    return out


def _shorten_specialty(text: str, limit_words: int = 10) -> str:
    """Аккуратно укорачивает строку спец-сти до <= limit_words слов."""
    if not isinstance(text, str) or not text.strip():
        return "не указано"
    # иногда приходит перечисление через запятую — берём первый фрагмент
    first = text.split(";")[0].split(".")[0]
    words = [w for w in re.split(r"\s+|,\s*", first) if w]
    if len(words) <= limit_words:
        return first.strip()
    return " ".join(words[:limit_words]).strip()


def _render_numbered_list_from_safe(compact_payload: str) -> str:
    """Строит итоговый пронумерованный список в формате final_answer (без участия LLM)."""
    query_text = None
    m = re.search(r'CC_QUERY="([^"]+)"', compact_payload)
    if m:
        query_text = m.group(1).replace("\n", " ").strip()

    lines = [ln for ln in compact_payload.splitlines() if ln.strip().startswith(ITEM_MARKER)]
    blocks: List[str] = []
    for i, ln in enumerate(lines, 1):
        row = _parse_safe_compact_line(ln)
        fio = row.get("fio") or "не указано"
        spec_short = _shorten_specialty(row.get("spec") or "")
        addr = row.get("addr") or "не указан"
        arriving = row.get("arriving") or "—"
        dms = row.get("dms") or "—"
        age = row.get("age") or "не указан"
        block = (
            f"{i}. **ФИО врача:** {fio}\n"
            f"**Специализация кратко:** {spec_short}\n"
            f"**Адрес/адреса работы:** {addr}\n"
            f"**Приходящий:** {('Да!' if arriving == 'Да' else ('Нет!' if arriving == 'Нет' else 'не указан'))}\n"
            f"**Возраст пациентов:** {age}\n"
            f"**ДМС:** {('Да' if dms == 'Да' else ('Нет' if dms == 'Нет' else 'не указан'))}\n"
            "\n---\n"
        )
        blocks.append(block)
    if not blocks:
        return "Релевантной информации не найдено"
    body = "\n".join(blocks) + "\n— Конец списка —"
    if query_text:
        return f"В заметках по запросу «{query_text}» найдены следующие врачи:\n\n{body}"
    return body


# Удалены - заменены на FilterProcessor


def _safe_format(template: str, **kwargs) -> str:
    """
    Безопасно подставляет {placeholders} в шаблон, где также встречаются JSON-скобки.
    Экранирует все фигурные скобки, кроме известных плейсхолдеров из kwargs.
    """
    if not isinstance(template, str):
        return template
    # временно защищаем целевые плейсхолдеры
    protected: dict[str, str] = {}
    for k in kwargs.keys():
        protected[k] = f"<<__{k.upper()}__>>"
        template = template.replace("{" + k + "}", protected[k])
    # экранируем все остальные скобки
    template = template.replace("{", "{{").replace("}", "}}")
    # возвращаем плейсхолдеры и форматируем
    for k, marker in protected.items():
        template = template.replace(marker, "{" + k + "}")
    return template.format(**kwargs)


def _strip_service_markers(text: str) -> str:
    """Удаляет служебные маркеры из выходного текста перед отправкой пользователю."""
    if not isinstance(text, str):
        return text
    cleaned = text.replace(KNOWLEDGE_MARKER, "")
    return cleaned.replace("{{KNOWLEDGE_SNIPPET}}", "")


# --- robust JSON extraction without PCRE recursion ---

def _extract_json_object(text: str):
    """
    Достаёт первый валидный JSON-объект { ... } из текста.
    Учитывает строки и экранирование, не бросает исключений.
    """
    if not text:
        return None

    s = text
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if esc:
            esc = False
            continue

        if ch == "\\":
            esc = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        # мягкая попытка почистить лишние переводы строк/пробелы
                        cleaned = " ".join(chunk.split())
                        try:
                            return json.loads(cleaned)
                        except Exception:
                            return None
    return None


# ----------------------------------------------
# Функция обработки истории запросов
# и ответов на них
# ----------------------------------------------

def _history_reveal(user: str, sess: Dict[str, Any]) -> str:
    """
    Processes the session history to generate a formatted string for revealing the
    interaction history. This function excludes the last user query in the history if it
    matches the current user to avoid duplication and uses a specific format for
    presentation.

    :param user: The identifier for the current user.
    :type user: str
    :param sess: The session dictionary containing interaction history. The expected key
        in the dictionary is "history" with a list of dialogue turns.
    :type sess: Dict[str, Any]
    :return: A formatted string representing the interaction history, where each turn
        is labeled as either "User:" or "Assistant:" followed by the content.
    :rtype: str
    """
    history = sess.get("history", [])

    # Отбрасываем последний пользовательский запрос, чтобы он не дублировался в истории, передаваемой в запрос
    if history and history[-1].get("user") == user:
        history = history[:-1]

    # Печатаем скорректированный history с помощью генератора
    generator = "\n".join(
        f"{'User:' if 'user' in turn else 'Assistant:'} "
        f"{turn.get('user') or turn.get('bot') or turn.get('assistant')}"
        # нет уверенности, что вариант извлечения assistant нужен, но пока оставим.
        for turn in history
    )
    print("_history_reveal output: ", generator)
    return generator


# ----------------------------------------------
# Функции обработки входящих сообщений
# ----------------------------------------------

def classificator_prompt(text_user: str, sess: Dict[str, Any]) -> str:
    """Форматирует строку prompt для классификации входящего сообщения пользователя."""
    today = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    template = load_prompt("classificator_prompt", False)

    return _safe_format(
        template,
        today=today,
        user=text_user,
        sess=_history_reveal(text_user, sess),  # обрабатывается история диалога с пользователем
        LABEL_DOC=LABEL_DOC,
        EXAMPLES=EXAMPLES,
    )


def split_prompt(text_user: str, sess: Dict[str, Any]) -> str:
    """Форматирует строку prompt для сплита входящего сообщения пользователя."""
    template = load_prompt("split_prompt", False)

    return _safe_format(
        template,
        user=text_user,
        sess=_history_reveal(text_user, sess),  # обрабатывается история диалога с пользователем
    )


async def split_into_segments(text: str, sess: Dict[str, Any], think: bool | None = None) -> List[str]:
    """
    Важно! Функция переформулирует запрос!
    Это первая функция в каскаде обработки входящего сообщения пользователя.

    :param think: Включать ли reasoning у поддерживающих моделей
    :param text: Входящий сырой запрос.
    :param sess:
    :return: Возвращает список запросов от пользователя.
    """
    think = ollama_settings.resolve_think(think)
    # print("!!!THINK:", think)
    ollama_settings.init_model_name()
    # print("split_into_segments OLLAMA_MODEL:", ollama_settings.OLLAMA_MODEL)
    # print("split_into_segments options: ", ollama_settings.options_set())
    if not ollama_settings.OLLAMA_MODEL:
        raise ValueError("split_into_segments OLLAMA_MODEL cannot be empty")

    # Защищаемся от долгого ответа модели: ограничиваем время ожидания
    try:
        res = await asyncio.wait_for(
            ollama.generate(
                model=ollama_settings.OLLAMA_MODEL,
                prompt=split_prompt(text, sess),  # Добавляется sess
                options=ollama_settings.options_set(),
                format="json",
                keep_alive=-1,
                think=think,
            ),
            timeout=20,
        )
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("split_into_segments ОСТАНОВЛЕН")
        # Отмена сверху => закроется HTTP-стрим клиента => Ollama прекращает генерацию
        raise

    except Exception as e:
        print(f"\n⚠️ split_into_segments timeout/error: {e}")
        return [text]

    try:
        obj = _extract_json_object(res.get("response"))
        segments = []
        if isinstance(obj, dict):
            raw = obj.get("segments")
            if isinstance(raw, list):
                segments = [str(s).strip() for s in raw if str(s).strip()]
        # Fallback: если модель не выполнила контракт — считаем весь текст одним сегментом
        if not segments:
            segments = [text]

        print("\nProcessing results:")
        print(f"• Segments count: {len(segments)}")
        print(f"• Final segments: {segments}")
        return segments

    except Exception as e:
        print(f"\n⚠️ Unexpected error in split_into_segments: {e}")
        print("⚠️ Returned original text as single segment")
        return [text]


async def classify(text: str, sess: Dict[str, Any], think: bool | None = None) -> List[str]:
    """
    Классификатор сегментов входящего запроса пользователя,
    второй этап обработки входящего сообщения пользователя.

    :param think: Включать ли reasoning у поддерживающих моделей
    :param text: Фрагмент (выделенный предыдущей функцией) входящего текста для анализа и маркировки.
    :param sess:
    :return:
    """

    think = ollama_settings.resolve_think(think)

    if not ollama_settings.OLLAMA_MODEL:
        raise ValueError("classify OLLAMA_MODEL cannot be empty")

    try:
        res = await asyncio.wait_for(
            ollama.generate(
                model=ollama_settings.OLLAMA_MODEL,
                prompt=classificator_prompt(text, sess),
                options=ollama_settings.options_set(),
                format="json",
                keep_alive=-1,
                think=think,
            ),
            timeout=20,
        )
    except (asyncio.CancelledError, GeneratorExit):
        logger.info("classify ОСТАНОВЛЕН")
        # Отмена сверху => закроется HTTP-стрим клиента => Ollama прекращает генерацию
        raise

    except Exception as e:
        print(f"\n⚠️ classify timeout/error: {e}")
        return ["UNDEFINED"]
    try:
        obj = _extract_json_object(res.get("response"))
        labels_raw: list[str] = []
        if isinstance(obj, dict):
            raw = obj.get("labels")
            if isinstance(raw, list):
                labels_raw = raw
        labels = [str(label).upper() for label in labels_raw if str(label).upper() in ALLOWED]

        return labels or ["UNDEFINED"]
    except Exception as e:
        print(f"\n Classificator error: {e}")
        return ["UNDEFINED"]


async def final_answering(primary_request: str,
                          collected_info: str,
                          think: bool | None = None,
                          ai_feed: Literal["local", "cloud"] = "local",
                          ):  # Пока неясно что за тип данных будет возвращаться
    """Форматирует строку prompt для генерации ответа на входящее сообщение пользователя."""
    template = load_prompt("final_answer", False)
    template_cloud = load_prompt("final_answer_giga", False)
    prompt = _safe_format(
        template,
        primary_request=primary_request,
        collected_info=collected_info,
    )
    cloud_prompt = _safe_format(
        template_cloud,
        collected_info=collected_info,
    )

    if not ollama_settings.OLLAMA_MODEL:
        raise ValueError("final_answering OLLAMA_MODEL cannot be empty")
    think = ollama_settings.resolve_think(think)
    partial = ""  # накопитель
    if ai_feed == "local":
        try:
            logger.info("final_answering LOCAL branch has activated/активировано подключение к локальной LLM")
            stream = await ollama.generate(
                model=ollama_settings.OLLAMA_MODEL,
                prompt=prompt,
                options=ollama_settings.options_set(),
                keep_alive=-1,
                stream=True,
                think=think,
            )
            async for chunk in stream:
                partial += chunk["response"]
                partial = _strip_service_markers(partial)
                yield partial

        except (asyncio.CancelledError, GeneratorExit):
            logger.info("final_answering ollama ОСТАНОВЛЕН")
            # Отмена сверху => закроется HTTP-стрим клиента => Ollama прекращает генерацию
            raise

    if ai_feed == "cloud":
        try:
            logger.info("final_answering cloud branch has activated/активировано подключение к облачной LLM")
            stream = gigachat.gigachad_echo_async(
                system=cloud_prompt,
                prompt=primary_request,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                partial += delta
                partial = _strip_service_markers(partial)
                yield partial

        except (asyncio.CancelledError, GeneratorExit):
            logger.info("final_answering GigaChat ОСТАНОВЛЕН")
            # Отмена сверху => закроется HTTP-стрим клиента => Ollama прекращает генерацию
            raise


# ──────────────────────────────────────────────────────
# Подключаем doctor_info из llama_func_call
# ──────────────────────────────────────────────────────

async def get_doc_info_from_api(question: str, think: bool | None = None, **_, ) -> Tuple[str, bool]:
    """Получает информацию из API Мед.центра."""
    think = ollama_settings.resolve_think(think)
    try:
        result = await doctor_info.investigate(question, think=think)
        return result, False

    except (asyncio.CancelledError, GeneratorExit):
        print("get_doc_info_from_api ОСТАНОВЛЕН")
        # Отмена сверху => закроется HTTP-стрим клиента => Ollama прекращает генерацию
        raise


# ------------------------------------------------------
# Подключаем заглушку функции записи пациента
# ------------------------------------------------------
async def appointment_stub(_text: str, think: bool | None = None, **__) -> Tuple[str, bool]:
    """Заглушка функции записи пациента."""
    think = ollama_settings.resolve_think(think)
    return "Модуль записи к врачу скоро появится. ", False


# ──────────────────────────────────────────────────────
# Search with MEILISEARCH function
# (может использовать LLM переформулировку)
# ──────────────────────────────────────────────────────
async def instructions_search(_text: str,
                              think: bool | None = None,
                              index: str = "main_index",
                              raw: bool = False,
                              manager_mode: bool = False,
                              **__) -> Tuple[
    str, bool]:
    """
    Для поиска нужной информации в главном индексе или коллекции используется переформулировка запроса пользователя
    Пока неясно, следует ли ее делать.
    :param index:
    :param think:
    :param _text:
    :param __:
    :return: Кортеж: результат поиска и стоп - паттерн для PENDING
    """
    # Временно отключу переформулировку!
    # extracted_keyword = await formulate.extract_keyword(_text, extract_type="sentence")

    collected_info = await asyncio.to_thread(meilisearch.search_meili, index_name=index, query=_text)

    # Очистка HTML перед подстановкой в prompt
    clean_info = html_cleaner.strip_html(collected_info)
    if manager_mode:
        clean_info = (
            clean_info
            .replace("\r\n_\r\n", "\n\n")
            .replace("\r\n_\n", "\n\n")
            .replace("\n_\r\n", "\n\n")
            .replace("\n_\n", "\n\n")
            .replace("\n_", "\n")
            .replace("_\n", "\n")
        )
    if raw:
        return RAW_MODE_MARKER + "\n" + clean_info, False
    # Добавление маркера для лучшего распознавания LLM
    marker = "[MANAGER_INFO]\n" if manager_mode else ""
    marked_info = KNOWLEDGE_MARKER + "\n" + marker + clean_info
    return marked_info, False


# ------- поиск в новостном индексе---------------


def _fmt_news(hit: dict) -> str:
    vf = hit.get("valid_from") or ""
    vt = hit.get("valid_to") or ""
    title = hit.get("title") or "(без заголовка)"
    body = hit.get("content") or hit.get("body") or ""
    # короткий фрагмент:
    snippet = body.strip()
    if len(snippet) > 5000:
        snippet = snippet[:5000].rstrip() + "… [новость сокращена до 5.000 символов]"
    return f"[{vf} — {vt}] {title}\n{snippet}"


async def news_search(text: str, think: bool | None = None, index: str = "news", **__) -> Tuple[str, bool]:
    """
    Возвращает список активных на сейчас новостей/акций из индекса news.
    keyword берётся из исходного сегмента (без переформулировки).
    """
    # ключевое слово — сам сегмент | переформулировка не подключена!
    keyword = (text or "").strip() or None
    # extracted_keyword = await formulate.extract_keyword(keyword, extract_type="sentence")
    logger.info(f"Работает поиск по новостям, фраза запроса: {keyword}")
    # logger.info(f"Работает поиск по новостям, экстрагированное ключевое слово: {extracted_keyword}")
    logger.info(
        f"Индекс поиска: {index}, timestamp: {int(datetime.now(timezone.utc).timestamp())}"
    )
    hits = await asyncio.to_thread(
        meilisearch.search_news_active,
        index_name=index,
        keyword=keyword,
        now_ts=int(datetime.now(timezone.utc).timestamp()),
        limit=20,
        sort=["from_ts:desc"],
    )

    if not hits:
        return KNOWLEDGE_MARKER + "\nСейчас нет активных новостей/акций по заданным критериям.", False

    lines = [_fmt_news(h) for h in hits]
    # Помечаем как «сырое содержимое» — минуется final_answering для теста
    # payload = "<NO_POSTPROC>\n" + "\n\n---\n\n".join(lines)
    payload = "\n\n---\n\n".join(lines)
    marked_payload = KNOWLEDGE_MARKER + "\n" + payload
    # logger.info(payload)
    return marked_payload, False


# ────────────────────────────────────────────────
# Основная логика роутера / Router main logic
# ────────────────────────────────────────────────

# Ярлыки для вызова функций обработки данных после роутинга
# дополнительная секция констант для работы логики (не может быть обозначена вверху модуля, так как содержит в себе
# объявление функций для вызова


# Нужно промежуточное извлечение, так как строка не может содержать вызова функции
MODULES_str: str = load_prompt("MODULES", False)

# Шаблон: "КЛЮЧ": ИМЯ_ФУНКЦИИ
pattern = r'"(?P<key>[^"]+)":\s*(?P<name>\w+)'
pairs = re.findall(pattern, MODULES_str)

# Берём функции из globals() (или из MODULE) и тут уже обращение к функциям
MODULES: Dict[str, Any] = {}
for key, name in pairs:
    func = globals().get(name)
    if callable(func):
        MODULES[key] = func
    else:
        print(f"[router_preprocessor] ⚠️ MODULES: функция '{name}' для ключа '{key}' не найдена")

# Дополнительные обозначения типов для понимания вывода
# Данные по переменной sess:
SessionType: TypeAlias = Dict[str, Any]
RoutingResult: TypeAlias = Tuple[str, SessionType]


async def handle_pending_module(text: str, sess: SessionType, think: bool | None = None) -> RoutingResult | None:
    """Обрабатывает задержки выполнения модулей."""
    if (pending_module := sess.get("pending")) and pending_module in MODULES:
        kwargs = {"session": sess, "think": think}
        idx_name = INDEX_BY_LABEL.get(pending_module)
        if idx_name:
            kwargs["index"] = idx_name
        # ToDo: возможно сюда так же стоит поставить перехватчик ошибки - команды СТОП от пользователя
        # но это не точно, поскольку этот обработчик есть у вызываемых функций.
        response, continue_pending = await MODULES[pending_module](text, **kwargs)
        sess["pending"] = pending_module if continue_pending else None
        response = _strip_service_markers(response)
        sess["history"].extend([
            {"user": text},
            {"bot": response}
        ])

        return response, sess
    return None


async def is_possible_surname_or_specialty(segment: str) -> bool:
    """
    Возвращает True если сегмент похож на фамилию или спец-ность врача
    (использует repo.read_all() для ФИО и specialties)
    """
    text = segment.strip()
    if not text or len(text.split()) > 3:
        return False
    # ensure repo is ready (creates data file on first use)
    await _ensure_doctors_repo_loaded()
    docs = repo.read_all()
    # Проверка ФИО и специальности
    tl = text.lower()
    for d in docs:
        if tl in d.get('fio', '').lower().split():
            return True
        if tl == (d.get('specialization') or '').lower():
            return True
    return False


# Удалены - заменены на FilterProcessor


def _norm_text(s: str) -> str:
    """Приводит строку к lower()."""
    return (s or "").lower()


# Удалена - заменена на SegmentProcessor.extract_specialty_terms()


async def _search_in_cc_notes(
        keywords: str,
        fallback_segment: str | None = None,
        match_mode: Literal["all", "any"] = "all",
) -> str:
    """Ищет врачей по ключевым словам/фразам в заметках call‑центра."""
    await _ensure_doctors_repo_loaded()
    try:
        cc_by_id = _get_cc_info_map()
    except Exception as e:
        return f"Релевантной информации не найдено (ошибка загрузки заметок: {e})."

    docs = repo.read_all()

    # Извлекаем ключевые слова/фразы
    base_text = keywords or ""
    if not base_text and fallback_segment:
        base_text = fallback_segment

    search_terms = CCNotesProcessor.extract_note_keywords(base_text)

    if not search_terms:
        return "Не удалось извлечь ключевые слова для поиска."

    query_repr = " ".join(search_terms)
    matched: List[Dict[str, Any]] = []

    logger.debug("[CC_NOTES] terms=%s | mode=%s", search_terms, match_mode)

    fuzzy_threshold = 0.86

    for d in docs:
        did = d.get('id')
        if did is None:
            continue
        cc_text = cc_by_id.get(int(did), '')
        if not cc_text:
            continue

        # Очищаем HTML, NBSP → пробел, приводим к lower
        clean_cc = CCNotesProcessor.normalize_text(cc_text)
        clean_cc_fuzzy = normalize_text_for_fuzzy(clean_cc)
        note_tokens = clean_cc_fuzzy.split()
        note_join = clean_cc_fuzzy.replace(" ", "")
        if not clean_cc_fuzzy:
            continue

        def _term_matches(term: str) -> bool:
            term_norm = normalize_text_for_fuzzy(term)
            if not term_norm:
                return False
            if term_norm in clean_cc_fuzzy:
                return True
            term_join = term_norm.replace(" ", "")
            if term_join and term_join in note_join:
                return True
            for tok in note_tokens:
                if fuzzy_match(term_norm, tok, fuzzy_threshold):
                    return True
            for i in range(len(note_tokens) - 1):
                joined = note_tokens[i] + note_tokens[i + 1]
                if fuzzy_match(term_norm, joined, fuzzy_threshold):
                    return True
            return False

        # Проверяем наличие терминов согласно match_mode
        if match_mode == "all":
            has_match = all(_term_matches(term) for term in search_terms)
        else:
            has_match = any(_term_matches(term) for term in search_terms)

        if has_match:
            dd = dict(d)
            dd['callCenterInfo'] = cc_text
            matched.append(dd)

    if not matched:
        return f"Врачи с упоминанием «{query_repr}» в заметках не найдены."

    # Используем существующий формат SAFE_LIST
    full_text = doctor_info.format_documents(matched)
    compact = _format_compact(matched)
    compact_with_query = compact.replace("[SAFE_LIST]", f"[SAFE_LIST]\nCC_QUERY=\"{query_repr}\"")

    # Добавляем заголовок с запросом
    header = f"Запрос \"{query_repr}\" встречается у следующих врачей:\n"
    return header + compact_with_query + "\n\n[RAW_FULL]\n" + full_text


async def _filter_doctors_via_cc_info(
        filters: Dict[str, Any],
        segment: str | None = None,
        fallback_segment: str | None = None,
) -> str:
    """Возвращает отформатированный список врачей, удовлетворяющих фильтрам."""
    await _ensure_doctors_repo_loaded()
    try:
        cc_by_id = _get_cc_info_map()
    except Exception as e:
        return f"Релевантной информации не найдено (ошибка загрузки заметок колл-центра: {e})."

    docs = repo.read_all()

    # Попробуем аккуратно вытащить ключевые слова специальности из сегмента
    spec_terms: List[str] = []
    if segment:
        try:
            spec_terms = SegmentProcessor.extract_specialty_terms(segment, docs)
        except Exception:
            spec_terms = []

    # Fallback: если не нашли в текущем сегменте, попробуем во всём тексте запроса
    if not spec_terms and fallback_segment and fallback_segment != segment:
        try:
            spec_terms = SegmentProcessor.extract_specialty_terms(fallback_segment, docs)
        except Exception:
            spec_terms = []

    require_spec = bool(spec_terms)
    matched: List[Dict[str, Any]] = []

    for d in docs:
        did = d.get('id')
        if did is None:
            continue
        cc_text = cc_by_id.get(int(did), '')
        ok = True

        # 1) Специальность/направление (если распознан термин спец-сти в сегменте)
        if require_spec:
            units_text = _norm_text(" ".join(d.get('units') or []))
            spec_text = _norm_text(d.get('specialization') or '')

            def _match_in_units(term: str) -> bool:
                return term in units_text

            def _match_in_spec(term: str) -> bool:
                if not spec_text:
                    return False
                # требуем соседство с "врач" в пределах 25 символов в любую сторону
                pattern = rf"(врач[^\n\r\-,:;]{{0,25}}{re.escape(term)})|({re.escape(term)}[^\n\r\-,:;]{{0,25}}врач)"
                return re.search(pattern, spec_text) is not None

            if not (any(_match_in_units(t) for t in spec_terms) or any(_match_in_spec(t) for t in spec_terms)):
                ok = False

        # 2) Флаги FILTER (ДМС/дети/приходящий)
        for k, desired in filters.items():
            key = k.lower()
            if key.startswith('arriv') or key.startswith('приход'):
                key = 'arriving'
            elif key.startswith('child') or key.startswith('дет'):
                key = 'children'
            elif key in ('dms', 'дмс'):
                key = 'dms'
            elif key.startswith('notes'):
                key = 'notes'
            else:
                continue

            if key == 'notes':
                keywords = desired if isinstance(desired, list) else [desired]
                note_text = CCNotesProcessor.normalize_text(cc_text or '')
                if not keywords or not all(k in note_text for k in keywords):
                    ok = False
                    break
                continue

            val = CCNotesProcessor.get_flag(cc_text, key)
            if val is None or val != desired:
                ok = False
                break

        if ok:
            dd = dict(d)
            dd['callCenterInfo'] = cc_text or 'Нет заметок'
            matched.append(dd)

    if not matched:
        return "Релевантной информации не найдено"

    full_text = doctor_info.format_documents(matched)
    compact = _format_compact(matched)
    note_terms = filters.get('notes') if isinstance(filters, dict) else None
    header = ""
    if isinstance(note_terms, list) and note_terms:
        query_phrase = " ".join(note_terms)
        header = f"Запрос \"{query_phrase}\" встречается у следующих врачей:\n"
    return header + compact + "\n\n[RAW_FULL]\n" + full_text


async def process_segments(text: str, sess: SessionType, think: bool | None = None) -> str:
    """
    Процессинг системы для принятия решения об использовании документальной базы.
    - Происходит в ходе сплита (билдинг сегментов) запроса на основе шаблонов.
    - Упрощенное разбиение на сегменты
    - Обработка поиска по заметкам КЦ
    - Обработка поиска по менеджерам
    - Обработка фильтров
    - Обработка искомой информации из документальной базы
    """
    segments = await split_into_segments(text, sess, think)
    logger.debug("segments=%d: %s", len(segments), segments)
    responses: List[str] = []

    for idx, segment in enumerate(segments, 1):
        logger.debug("[seg#%d] raw='%s'", idx, segment)

        # 1. Поиск по заметкам КЦ
        if response := await _handle_note_search(segment, text):
            responses.append(response)
            continue

        # 2. Поиск по менеджерам
        if response := await _handle_manager_search(segment, text):
            responses.append(response)
            continue

        # 3. Обработка фильтров
        if response := await _handle_filter_search(segment, text):
            responses.append(response)
            continue

        # 4. Early fallback к doctor_info
        if response := await _try_doctor_fallback(segment, sess, think, f"seg#{idx} EARLY"):
            responses.append(response)
            continue

        # 5. Классификация и обработка через модули
        labels = await classify(segment, sess, think)
        main_labels = [lbl for lbl in labels if lbl in LABEL_PRIORITY]
        logger.debug("[seg#%d] labels=%s | main=%s", idx, labels, main_labels)

        # 5.5. Патч: принудительная активация meilisearch при наличии слова "скрипт"
        script_hit_seg, script_hit_full = SegmentProcessor.check_pattern_match(segment, text, SCRIPT_WORD_RE)
        if (script_hit_seg or script_hit_full) and "SCRIPTS" not in main_labels:
            logger.debug("[seg#%d] скрипт обнаружен → принудительно добавляем SCRIPTS", idx)
            main_labels.insert(0, "SCRIPTS")  # Добавляем в начало для приоритета

        # 6. Late fallback к doctor_info
        if response := await _try_doctor_fallback(segment, sess, think, f"seg#{idx} LATE"):
            responses.append(response)
            continue

        # 7. Если нет основных лейблов - fallback к doctor_info
        if not main_labels:
            logger.debug("[seg#%d] no main_labels → doctor_info fallback", idx)
            response, _ = await get_doc_info_from_api(segment, session=sess, think=think)
            responses.append(RAW_MODE_MARKER + "\n" + response)
            continue

        # 8. Обработка через модули
        for label in main_labels:
            kwargs: Dict[str, Any] = {"session": sess, "think": think}
            idx_name = INDEX_BY_LABEL.get(label)
            if idx_name:
                kwargs["index"] = idx_name
            logger.info("[seg#%d] call MODULE label=%s index=%s", idx, label, idx_name)

            response, continue_pending = await MODULES[label](segment, **kwargs)

            logger.debug("[seg#%d] module=%s responded, size=%d", idx, label, len(response) if response else 0)
            if response:
                responses.append(response)

            if continue_pending:
                sess["pending"] = label
                logger.debug("[seg#%d] pending set to %s", idx, label)
                break

    final_answ = SEGMENT_SEPARATOR.join(responses)
    logger.debug("final response size=%d", len(final_answ))
    return final_answ


#  Модуль - пример для сохранения единообразия:
# async def _call_module(label: str, text: str, sess: SessionType, think: bool | None):
#     return await MODULES[label](text, session=sess, think=think)
#

async def routing(text: str,
                  sess: SessionType | None = None,
                  extra_processing: Literal["direct", "processed"] = "processed",
                  think: bool | None = None,
                  ai_feed: Literal["local", "cloud"] = "local",
                  ) -> AsyncGenerator[RoutingResult, None]:
    """
    Обработка входящего запроса пользователя идет в следующем направлении:
    1. split_into_segments - ПЕРЕФОРМУЛИРОВКА и разбивка на смысловые сегменты.
    2.  <- process_segments <- classify маркировка сегментов запроса пользователя.
    3. handle_pending_module и process_segments - заключительный шаг обработки модулями извлечения информации
    и передача сообщения в данную функцию.

    Here is variant of routing() that *yields* (partial_answer, session) pairs,
        so that the outer UI can stream them.

    :param ai_feed: Что подключаем к итоговой обработке текста: облачную LLM или локальную
    :param think: Включает Reasoning у поддерживающей его модели
    :param extra_processing: определяется, будет ли использоваться на выходе
    постобработка входящих данных с помощью функции final_answering либо же
    данные из БД будут выводиться напрямую
    :param text: Сообщение пользователя
    :param sess: словарь состояния сессии, хранит pending-модуль и историю
    :return: ответ, обновлённая сессия
    """

    # 1. Инициализируем состояние сессии обработки входящего текстового блока

    sess = sess or {}
    sess.setdefault("pending", None)
    sess.setdefault("history", [])

    # Handle pending module if exists
    # TODO: Понять зачем вообще это тут вызывается
    if pending_result := await handle_pending_module(text, sess, think=think):
        yield pending_result

    # Process text segments
    result = await process_segments(text, sess, think=think, )

    # ==== SAFE_LIST path (детерминированный рендер без изменения final_answer) ====
    N, ids = _extract_safe_list_meta(result)
    if N and ids:
        compact, raw_full = (result.split("\n\n[RAW_FULL]\n", 1) + [""])[:2] if "\n\n[RAW_FULL]\n" in result else (
            result, "")
        # Рендерим сами (без LLM), чтобы не терять позиции и не зависеть от final_answer
        formatted = _render_numbered_list_from_safe(compact)
        sess["history"].extend([{"user": text}, {"bot": formatted}])
        yield formatted, sess
        return

    # Update history
    sess["history"].extend([
        {"user": text},
        {"bot": result}
    ])

    # ⬇️ новый быстрый выход для «сырых» (готовых) результатов
    if isinstance(result, str) and (RAW_MODE_MARKER in result):
        # отдаём как есть, без постпроцесса (final_answering)
        cleaned = result.replace(RAW_MODE_MARKER, "").lstrip("\n\r ")
        cleaned = _strip_service_markers(cleaned)
        yield cleaned, sess
        return

    if extra_processing == "processed":
        async for partial in final_answering(text, result, think=think, ai_feed=ai_feed):
            yield partial, sess
            yield partial, sess  # Stream final response V1 with processing by final_answering func.
    else:
        yield _strip_service_markers(result), sess  # Stream final response V2 without handling by final_answering func.


async def process_routing_request(query: str, think: bool | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Запускает маршрутизацию и собирает все части ответа из async-генератора,
    возвращая финальную строку и итоговую сессию. Нужно чисто для тестирования данного модуля
    """
    final_response: str = ""
    final_session: dict[str, Any] = {}

    # routing возвращает AsyncGenerator[(partial_response, session), None]
    async for partial, sess in routing(query.strip(), think=think, ):
        # на каждой итерации приходят (partial, sess)
        final_response = partial  # перезаписываем — в итоге останется последний
        final_session = sess

    return final_response, final_session


async def main():
    async for partial, sess in routing("Смирнова", think=False, ):
        print(partial)  # или обновлять UI


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]).strip() or input("Введите запрос: ").strip()
    final_resp, final_sess = asyncio.run(process_routing_request(query, think=False))
    print(final_resp)
    # print(classificator_prompt("-сообщение пользователя-",
    #                            {"history": [{"user": "-содержимое памяти-",
    #                                          "bot": "_невнятное сообщение ассистента_", }, ]}))
    # print()
    # print("ALLOWED: ", ALLOWED)
    # print()
    # print("MODULES: ", MODULES)
