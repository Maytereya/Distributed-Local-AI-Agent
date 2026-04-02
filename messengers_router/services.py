"""Сервисный слой интеграций для роутера пациентов.

Содержит вызовы внешних источников (Nayka API, price, meili), кэш врачей,
поиск расписания/цен/адресов и fallback-контракты для handoff при сбоях.

Ответственность модуля:
1) Доступ к внешним данным и их нормализация к стабильному внутреннему формату.
2) Локальный кэш/дедупликация/ограничение объема данных для рендера.
3) Прозрачный graceful degradation (note/reason/handoff flags) при сбоях.

Модуль не должен принимать state-machine решения по диалогу.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from urllib.parse import quote_from_bytes
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info
from converters import html_cleaner
from schedule_ttl_cache import AsyncListTTLStaleCache

from .doctor_name_port import (
    extract_doctor_name_candidate,
    resolve_schedule_surname,
    surname_variants,
)
from .service_phrase import extract_service_phrase
from .runtime_config import config as c

logger = logging.getLogger(__name__)

_ADDRESS_HINT_RE = re.compile(
    r"\b(ул\.?|улица|пр\.?|проспект|пр-?т|тракт|б-р|бульвар|шоссе|пер\.?|переулок|наб\.?|площадь|дом|д\.|корп\.?|к\.|пом\.?)\b",
    re.I,
)
_SCHEDULE_QUERY_RE = re.compile(r"\b(расписани\w*|график|когда\b.*\bпринима\w*|принима\w*)\b", re.I)
_FIO_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]{2,}")
_PHONE_EXTRACT_RE = re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d")
_PRICE_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
_PRICE_HOMECODE_DOTTED_RE = re.compile(r"\b\d+(?:\.\d+){1,6}\b")
_PRICE_HOMECODE_NUM_RE = re.compile(r"\b\d{4,}\b")
_DOCTOR_PRICE_HINT_RE = re.compile(r"\b(?:у|врач\w*|доктор\w*)\s+[а-яё\-]{3,}\b", re.I)
_PRICE_REQUEST_RE = re.compile(r"\b(стоим\w*|цен\w*|сколько)\b", re.I)
_PRICE_CONSULT_HINT_RE = re.compile(r"\b(при[её]м\w*|консультаци\w*)\b", re.I)
_PRICE_SERVICE_PREFIX_RE = re.compile(
    r"^\s*(?:а\s+)?(?:сколько\s+стоит|сколько\s+будет\s+стоить|цена|стоимость)\s+",
    re.I,
)
_PRICE_DOCTOR_SUFFIX_RE = re.compile(
    r"\bу\s+[а-яё\-]{3,}(?:\s+[а-яё\-]{2,}){0,2}\b.*$",
    re.I,
)
_NONBOOKABLE_POINTS_PATH = Path(__file__).resolve().parent / "data" / "nonbookable_points.json"
_NEAREST_HINT_RE = re.compile(r"\b(ближайш\w*|сам\w*\s+ранн\w*|раньше|поскорее|свободн\w*\s+окн\w*)\b", re.I)
_UZI_QUERY_RE = re.compile(r"\b(узи|узист|ультразвук\w*|ультразвуков\w*)\b", re.I)
_UZI_LINE_RE = re.compile(r"\b(узи|ультразвук\w*|ультразвуков\w*)\b", re.I)
_CITY_PREFIX_RE = re.compile(r"\b(?:г|город)\.?\s*([а-яёa-z\-]+)\b", re.I)
_UZI_FALSE_POSITIVE_RE = re.compile(
    r"\b(под\s+контролем\s+узи|во\s+время\s+консультативн\w*\s+при(е|ё)м\w*|"
    r"в\s+рамках\s+при(е|ё)м\w*|интерпретац\w*|разъяснен\w*)\b",
    re.I,
)
_UZI_ROLE_HINT_RE = re.compile(
    r"\b(узист\w*|врач\w*\s+узи|врач\w*\s+ультразвуков\w*\s+диагностик\w*|"
    r"каки\w*\s+узист\w*|каки\w*\s+врач\w*\s+узи)\b",
    re.I,
)
_UZI_PROCEDURE_HINT_RE = re.compile(
    r"\b(сдела\w*|дела\w*|провест\w*|процедур\w*|исследован\w*|"
    r"брюшн\w*|щитовид\w*|мал\w*\s+таз\w*|молочн\w*|почек|печен\w*|сердц\w*|сосуд\w*)\b",
    re.I,
)
_SPECIALTY_ROLE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "акушер-гинеколог": ("акушер гинеколог", "гинеколог"),
    "аллерголог": ("аллерголог", "иммунолог"),
    "иммунолог": ("иммунолог", "аллерголог"),
    "кардиолог": ("кардиолог",),
    "эндокринолог": ("эндокринолог",),
    "гинеколог-эндокринолог": ("гинеколог эндокринолог", "эндокринолог", "гинеколог"),
    "гинеколог-маммолог": ("гинеколог маммолог", "гинеколог", "маммолог"),
    "педиатр": ("педиатр",),
    "хирург": ("хирург",),
    "пластический хирург": ("пластический хирург", "пластическ"),
    "терапевт": ("терапевт",),
    "травматолог": ("травматолог",),
    "травматолог-ортопед": ("травматолог ортопед", "травматолог", "ортопед"),
    "проктолог": ("проктолог", "колопроктолог"),
    "колопроктолог": ("колопроктолог", "проктолог"),
    "уролог": ("уролог",),
    "уролог-андролог": ("уролог андролог", "уролог", "андролог"),
    "андролог": ("андролог", "уролог"),
    "онколог": ("онколог",),
    "гинеколог": ("гинеколог",),
    "невролог": ("невролог",),
    "нейрохирург": ("нейрохирург",),
    "нефролог": ("нефролог",),
    "гастроэнтеролог": ("гастроэнтеролог",),
    "гематолог": ("гематолог",),
    "гепатолог": ("гепатолог",),
    "гирудотерапевт": ("гирудотерапевт",),
    "дерматолог": ("дерматолог", "дерматовенеролог"),
    "дерматовенеролог": ("дерматовенеролог", "дерматолог"),
    "инфекционист": ("инфекционист",),
    "эндоскопист": ("эндоскопист", "эндоскоп"),
    "эндоскопия": ("эндоскопист", "эндоскоп"),
    "лор": ("лор", "оториноларинг"),
    "оториноларинголог": ("оториноларинголог", "оториноларинг", "лор"),
    "лимфолог": ("лимфолог",),
    "массажист": ("массажист",),
    "мануальный терапевт": ("мануальный терапевт", "мануальн"),
    "пульмонолог": ("пульмонолог",),
    "ревматолог": ("ревматолог",),
    "стоматолог": ("стоматолог",),
    "стоматолог-ортопед": ("стоматолог ортопед", "стоматолог", "ортопед"),
    "флеболог": ("флеболог",),
    "фониатр": ("фониатр",),
    "физиотерапевт": ("физиотерапевт", "физиотерап"),
    "функциональная диагностика": ("функциональная диагностика", "функциональн"),
    "анестезиолог": ("анестезиолог", "реаниматолог"),
    "реаниматолог": ("реаниматолог", "анестезиолог"),
    "узи": ("узи", "ультразвук"),
}
_SPECIALTY_PRIORITY_SURNAMES: dict[str, tuple[str, ...]] = {
    # Бизнес-приоритет списка хирургов в выдаче.
    "хирург": ("тюрин", "джарар", "алимназаров", "губский"),
}
_SERVICE_FILTER_STOPWORDS = {
    "хочу",
    "нужно",
    "надо",
    "можно",
    "сделать",
    "пройти",
    "провести",
    "выполняет",
    "выполняют",
    "делает",
    "делают",
    "какой",
    "какие",
    "врач",
    "врачи",
    "доктор",
    "доктора",
    "процедура",
    "процедуры",
    "услуга",
    "услуги",
    "исследование",
    "исследования",
}
_SERVICE_QUERY_SIGNAL_RE = re.compile(
    r"\b(услуг\w*|процедур\w*|исследован\w*|анализ\w*|сда[тч]\w*|"
    r"сдела\w*|провед\w*|провод\w*|выполня\w*|дела\w*|удали\w*|удалени\w*|"
    r"узи|экг|мрт|кт|фгдс|фкс|эндоскоп\w*|гастроскоп\w*|"
    r"кольпоскоп\w*|колоноскоп\w*|рентген\w*|холтер\w*)\b",
    re.I,
)
_ENDOSCOPY_SERVICE_RE = re.compile(
    r"\b(эндоскоп\w*|фгдс|фдгс|фгс|егдс|эгдс|фкс|гастроскоп\w*|колоноскоп\w*|"
    r"ректороманоскоп\w*|эзофагогастродуоденоскоп\w*)\b",
    re.I,
)
_PROCEDURE_BRANCH_LOOKUP_RE = re.compile(
    r"\b(где|сдела\w*|пройти|провест\w*|выполня\w*|дела\w*|можно|пройти\s+диагностик\w*)\b",
    re.I,
)
_SPECIALTY_CANONICAL = (
    "акушер-гинеколог",
    "аллерголог",
    "иммунолог",
    "гастроэнтеролог",
    "гематолог",
    "гепатолог",
    "гирудотерапевт",
    "гинеколог-маммолог",
    "гинеколог-эндокринолог",
    "эндокринолог",
    "офтальмолог",
    "дерматолог",
    "дерматовенеролог",
    "эндоскопист",
    "эндоскопия",
    "кардиолог",
    "колопроктолог",
    "лимфолог",
    "массажист",
    "мануальный терапевт",
    "невролог",
    "нейрохирург",
    "нефролог",
    "проктолог",
    "травматолог",
    "травматолог-ортопед",
    "ревматолог",
    "пульмонолог",
    "гинеколог",
    "терапевт",
    "педиатр",
    "уролог",
    "уролог-андролог",
    "андролог",
    "онколог",
    "инфекционист",
    "хирург",
    "пластический хирург",
    "ортопед",
    "стоматолог",
    "стоматолог-ортопед",
    "флеболог",
    "фониатр",
    "физиотерапевт",
    "функциональная диагностика",
    "анестезиолог",
    "реаниматолог",
    "лор",
    "оториноларинголог",
)
_SCHEDULE_SPECIALTY_TOKENS = set(_SPECIALTY_CANONICAL) | {"узи", "узист", "экг", "мрт", "кт", "фгдс", "фкс"}
_SPECIALTY_RE = re.compile(
    r"\b(" + "|".join(re.escape(x) for x in sorted(_SPECIALTY_CANONICAL, key=len, reverse=True)) + r")\w*\b",
    re.I,
)
# Fallback-карта для процедур, где API не отдает надежный branch-level match.
_STATIC_PROCEDURE_BRANCH_OVERRIDES: dict[str, tuple[str, ...]] = {
    "флюорограф": ("г. Самара, пр. Ленина, 5",),
}
# Важно: для /priceByRegion нужен city-level regionId (Самара = 3),
# а для /doctorServicePricesByRegion используются branch-level regionId из doctorRegions.
SAMARA_PRICE_REGION_ID = 3
try:
    DOCTORS_TOP_N = max(1, int(c.MR_DOCTORS_TOP_N))
except Exception:
    DOCTORS_TOP_N = 4


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
_PRICE_QUERY_STOPWORDS = {
    "сколько",
    "стоит",
    "стоимость",
    "цена",
    "цена",
    "на",
    "в",
    "по",
    "у",
    "для",
    "и",
    "или",
    "услуга",
    "услуги",
    "процедура",
    "процедуры",
    "анализ",
    "анализы",
    "врач",
    "врача",
    "доктор",
    "доктора",
    "самара",
    "подскажите",
    "скажите",
    "пожалуйста",
    "как",
    "его",
    "ее",
    "её",
    "пройти",
    "сдать",
    "сделать",
    "узнать",
    "мне",
    "нужно",
    "надо",
    "хочу",
    "можно",
}
_PRICE_QUERY_CANONICAL_TOKENS = {
    "алт": "алат",
    "алат": "алат",
    "alat": "алат",
    "ast": "асат",
    "аст": "асат",
    "асат": "асат",
    "asat": "асат",
}
_PREPARE_QUERY_STOPWORDS = {
    "как",
    "подготовиться",
    "подготовится",
    "подготовка",
    "подготовке",
    "подготовки",
    "к",
    "для",
    "перед",
    "процедурой",
    "процедуре",
    "процедуры",
    "процедуру",
    "исследованием",
    "исследованию",
    "исследования",
    "исследование",
    "анализом",
    "подскажите",
    "скажите",
    "пожалуйста",
    "мне",
    "нужно",
    "надо",
    "можно",
    "ли",
    "что",
    "чтобы",
    "когда",
    "будет",
}


def _is_schedule_no_slots_text(payload: Any) -> bool:
    """
    Определяет текстовый ответ Nayka API, когда врач найден, но свободных слотов нет.

    :param payload: ответ из find_doctor_schedule
    :return: True, если это кейс отсутствия свободных слотов, а не отсутствия врача
    """

    if not isinstance(payload, str):
        return False
    norm = _normalise_input(payload).replace("ё", "е")
    return "свободных слотов нет" in norm
_PREPARE_LEADIN_RE = re.compile(
    r"^\s*(?:подскажите[, ]+)?(?:как\s+)?подготов(?:иться|ится|ка)\s*(?:к|для)?\s+",
    re.I,
)
_PREPARE_ENTITY_RE = re.compile(
    r"(?:подготов(?:иться|ится|ка)\s*(?:к|для)\s+)(?P<entity>.+)$",
    re.I,
)
_PREPARE_SYNONYM_HINTS: dict[str, tuple[str, ...]] = {
    "фгдс": ("гастроскопия", "фиброгастродуоденоскопия"),
    "гастроскоп": ("фгдс",),
    "эгдс": ("фгдс",),
    "фгс": ("фгдс",),
    "ректороманоскоп": ("ректоскопия",),
    "колоноскоп": ("колоноскопия",),
    "кольпоскоп": ("кольпоскопия",),
    "вульвоскоп": ("вульвоскопия",),
}


def _normalise_input(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _normalise_prepare_text(text: str) -> str:
    norm = _normalise_input(text).replace("ё", "е")
    norm = re.sub(r"[\"'«»!?.,;:()]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _dedupe_queries(queries: list[str], *, max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        q = str(raw or "").strip()
        if not q:
            continue
        key = _normalise_prepare_text(q)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max_items:
            break
    return out


def _extract_prepare_entity_phrase(text: str) -> str:
    norm = _normalise_prepare_text(text)
    if not norm:
        return ""
    m = _PREPARE_ENTITY_RE.search(norm)
    if m:
        return str(m.group("entity") or "").strip(" ?!.,")
    stripped = _PREPARE_LEADIN_RE.sub("", norm, count=1).strip(" ?!.,")
    return stripped


def _prepare_keyword_query(text: str) -> str:
    norm = _normalise_prepare_text(text)
    if not norm:
        return ""
    tokens = re.findall(r"[a-zа-я0-9]{2,}", norm)
    if not tokens:
        return ""
    picked = [t for t in tokens if t not in _PREPARE_QUERY_STOPWORDS]
    if not picked:
        return ""
    return " ".join(picked[:6]).strip()


def _prepare_synonym_queries(text: str) -> list[str]:
    norm = _normalise_prepare_text(text)
    if not norm:
        return []
    out: list[str] = []
    for hint, variants in _PREPARE_SYNONYM_HINTS.items():
        if hint not in norm:
            continue
        for var in variants:
            out.append(var)
            out.append(f"подготовка к {var}")
    return _dedupe_queries(out, max_items=6)


def _prepare_query_variants(raw_query: str, entity_query: str = "") -> list[str]:
    variants: list[str] = []
    for src in (raw_query, entity_query):
        src_clean = str(src or "").strip()
        if not src_clean:
            continue
        variants.append(src_clean)

        entity_phrase = _extract_prepare_entity_phrase(src_clean)
        if entity_phrase:
            variants.append(f"подготовка к {entity_phrase}")
            variants.append(entity_phrase)
            keyword_query = _prepare_keyword_query(entity_phrase)
            if keyword_query and keyword_query != entity_phrase:
                variants.append(keyword_query)
                variants.append(f"подготовка к {keyword_query}")
        else:
            stripped = _PREPARE_LEADIN_RE.sub("", _normalise_prepare_text(src_clean), count=1).strip()
            if stripped:
                variants.append(stripped)
                variants.append(f"подготовка к {stripped}")

        variants.extend(_prepare_synonym_queries(" ".join(x for x in (src_clean, entity_phrase) if x)))
    return _dedupe_queries(variants)


_PREPARE_SERVICE_INFO_SYNONYMS: dict[str, tuple[str, ...]] = {
    "холестерин": (
        "холестерин",
        "общий холестерин",
        "анализ крови на холестерин",
        "липидный профиль",
        "липидограмма",
    ),
    "фгдс": ("фгдс", "фдгс", "фгс", "гастроскопия"),
    "фкс": ("фкс", "колоноскопия"),
}
_PREPARE_SERVICE_INFO_GENERIC_TOKENS = {
    "подготовка",
    "исследование",
    "исследованию",
    "исследования",
    "процедура",
    "процедуре",
    "процедуры",
    "процедуру",
    "анализ",
    "анализа",
    "анализу",
    "анализом",
    "анализы",
    "кровь",
    "крови",
    "подскажите",
    "скажите",
}


def _prepare_service_info_queries(raw_query: str, entity_query: str = "") -> list[str]:
    """
    Расширяет prepare-запрос для поиска по serviceInfoAll.

    :param raw_query: исходный текст пользователя
    :param entity_query: ранее извлеченная услуга/анализ
    :return: список нормализованных поисковых вариантов
    """

    variants = _prepare_query_variants(raw_query, entity_query)
    expanded = list(variants)
    for item in variants:
        norm = _normalise_prepare_text(item)
        for hint, synonyms in _PREPARE_SERVICE_INFO_SYNONYMS.items():
            if hint not in norm:
                continue
            expanded.extend(synonyms)
            expanded.extend(f"подготовка к {syn}" for syn in synonyms)
    return _dedupe_queries(expanded, max_items=16)


def _prepare_service_info_core_tokens(text: str) -> set[str]:
    """
    Возвращает смысловые токены prepare-запроса для матчинга serviceInfoAll.

    Убирает общие слова вроде "подготовка" и "анализ", чтобы выбор записи
    опирался на саму услугу/процедуру, а не на шаблонный текст поля.

    :param text: текст запроса или варианта запроса
    :return: множество смысловых токенов
    """

    norm = _normalise_prepare_text(text)
    if not norm:
        return set()
    tokens = _doc_tokens(norm)
    return {t for t in tokens if t not in _PREPARE_SERVICE_INFO_GENERIC_TOKENS}


def _service_info_row_score(queries: list[str], row: dict[str, Any]) -> tuple[int, int]:
    """
    Считает релевантность строки serviceInfoAll для prepare-запроса.

    :param queries: подготовленные варианты запроса
    :param row: строка из serviceInfoAll
    :return: кортеж score для сортировки по убыванию
    """

    service_name = _normalise_input(str(row.get("serviceName") or "")).replace("ё", "е")
    preparation = str(row.get("preparation") or "").strip()
    if not service_name or not preparation:
        return (0, 0)

    best = 0
    service_tokens = _doc_tokens(service_name)
    for query in queries:
        query_norm = _normalise_prepare_text(query)
        if not query_norm:
            continue
        query_core_tokens = _prepare_service_info_core_tokens(query_norm)
        if not query_core_tokens:
            continue

        if service_name == query_norm:
            best = max(best, 12)
        elif query_norm in service_name or service_name in query_norm:
            best = max(best, 9)

        overlap = len(query_core_tokens & service_tokens)
        if overlap:
            best = max(best, overlap * 3 + 4)

    return (best, len(preparation))


def _choose_service_info_preparation(
    rows: list[dict[str, Any]],
    queries: list[str],
) -> str | None:
    """
    Выбирает лучший текст подготовки из serviceInfoAll.

    :param rows: записи serviceInfoAll
    :param queries: варианты запроса пользователя
    :return: текст preparation или None
    """

    ranked: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = _service_info_row_score(queries, row)
        if score[0] <= 0:
            continue
        ranked.append((score, row))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][1]
    preparation = str(best.get("preparation") or "").strip()
    return preparation or None


def _is_samara_city_value(value: str | None) -> bool:
    city = _extract_city_token(value)
    return city == "самара"


def _is_non_samara_city_value(value: str | None) -> bool:
    city = _extract_city_token(value)
    return bool(city and city != "самара")


def _extract_city_token(value: str | None) -> str | None:
    if not value:
        return None
    norm = _normalise_input(value).replace("ё", "е")
    if not norm:
        return None
    if _ADDRESS_HINT_RE.search(norm):
        return None
    norm = re.sub(r"[^a-zа-я0-9\-]+", " ", norm).strip()
    if not norm or any(ch.isdigit() for ch in norm):
        return None
    parts = norm.split()
    if not parts:
        return None
    if parts[0] in {"г", "город"}:
        parts = parts[1:]
    if len(parts) != 1:
        return None
    city = parts[0].strip()
    return city or None


def _normalize_region_text(value: str) -> str:
    norm = _normalise_input(value).replace("ё", "е")
    return re.sub(r"\s+", " ", norm).strip()


def _compact_region_text(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", _normalize_region_text(value))


def _is_explicit_non_samara_region(value: str) -> bool:
    norm = _normalize_region_text(value)
    if not norm:
        return False
    if "самара" in norm:
        return False
    # Частый кейс в данных: оренбургские площадки не должны попадать в самарский контур.
    if "оренбург" in norm:
        return True
    for city in _CITY_PREFIX_RE.findall(norm):
        city_norm = _normalize_region_text(city)
        if city_norm and city_norm != "самара":
            return True
    return False


def _region_matches_samara_tokens(region: str, samara_tokens: set[str]) -> bool:
    if not samara_tokens:
        return False
    region_norm = _normalize_region_text(region)
    if not region_norm:
        return False
    if region_norm in samara_tokens:
        return True
    region_compact = _compact_region_text(region_norm)
    for token in samara_tokens:
        if region_norm in token or token in region_norm:
            return True
        token_compact = _compact_region_text(token)
        if region_compact and token_compact and (region_compact in token_compact or token_compact in region_compact):
            return True
    return False


def _has_explicit_non_samara_regions(values: list[str]) -> bool:
    for value in values:
        if _is_explicit_non_samara_region(value):
            return True
    return False


def _extract_specialty_from_text(text: str) -> str:
    if _UZI_QUERY_RE.search(text or ""):
        return "узи"
    if _ENDOSCOPY_SERVICE_RE.search(text or ""):
        return "эндоскопист"
    m = _SPECIALTY_RE.search(text or "")
    if not m:
        return ""
    return str(m.group(1) or "").strip().lower().replace("ё", "е")


def _procedure_query_role_specialty(text: str) -> str:
    """
    Возвращает ролевую специальность для процедурного запроса.

    Пример:
    - "фгдс", "эндоскопия", "колоноскопия" -> "эндоскопист"

    :param text: текст запроса или service_name
    :return: каноническая специальность или пустая строка
    """
    if _ENDOSCOPY_SERVICE_RE.search(text or ""):
        return "эндоскопист"
    return ""


def _looks_like_schedule_specialty_token(value: str) -> bool:
    """
    Проверяет, является ли токен названием специальности/исследования,
    а не фамилией врача.

    :param value: кандидат на фамилию
    :return: True, если это specialty-like токен
    """
    norm = _normalise_input(value).replace("ё", "е")
    if not norm:
        return False
    return norm in _SCHEDULE_SPECIALTY_TOKENS


def _has_nearest_hint(text: str) -> bool:
    return bool(_NEAREST_HINT_RE.search(text or ""))


def _is_uzi_query_text(text: str) -> bool:
    return bool(_UZI_QUERY_RE.search(text or ""))


def _split_spec_lines(spec_text: str) -> list[str]:
    return [ln.strip(" \t•-") for ln in str(spec_text or "").splitlines() if ln.strip()]


def _matches_uzi_doctor_profile(doc: dict[str, Any]) -> bool:
    spec_text = str(doc.get("specialization") or "")
    if not spec_text:
        return False

    units_text = " ".join(str(x or "") for x in (doc.get("units") or []))
    units_norm = _normalise_input(units_text)
    if "ультразвук" in units_norm or re.search(r"\bузи\b", units_norm):
        return True

    for raw_line in _split_spec_lines(spec_text):
        line = _normalise_input(raw_line)
        if not line or not _UZI_LINE_RE.search(line):
            continue
        if _UZI_FALSE_POSITIVE_RE.search(line):
            continue
        if "врач ультразвуковой диагностики" in line or "ультразвуков" in line:
            return True
        if line.startswith("узи "):
            return True
    return False


def _specialty_terms(specialty: str) -> tuple[str, ...]:
    """
    Возвращает нормализованные термины специальности для role-матчинга.

    :param specialty: каноническая специальность (например, "хирург", "лор", "узи")
    :return: кортеж терминов/синонимов для подстрочного поиска
    """
    spec_norm = _normalise_input(specialty).replace("ё", "е")
    if not spec_norm:
        return tuple()
    terms = _SPECIALTY_ROLE_SYNONYMS.get(spec_norm, (spec_norm,))
    out: list[str] = []
    for term in terms:
        term_norm = _normalise_input(term).replace("ё", "е")
        if term_norm and term_norm not in out:
            out.append(term_norm)
    return tuple(out)


def _matches_specialty_terms(text: str, specialty: str) -> bool:
    """
    Проверяет совпадение текста с role-терминами специальности.

    :param text: произвольный текст (подразделение/специализация)
    :param specialty: искомая специальность
    :return: True, если найдено совпадение по одному из терминов
    """
    norm = _normalise_input(text).replace("ё", "е")
    if not norm:
        return False
    tokens = re.findall(r"[a-zа-я0-9]+", norm)
    if not tokens:
        return False

    for term in _specialty_terms(specialty):
        t = _normalise_input(term).replace("ё", "е")
        if not t:
            continue
        term_tokens = re.findall(r"[a-zа-я0-9]+", t)
        if len(term_tokens) > 1:
            if all(any(tok == part or tok.startswith(part) for tok in tokens) for part in term_tokens):
                return True
            continue
        # Короткие термины должны совпадать целиком (например, "узи", "лор"),
        # иначе получаем ложные срабатывания по подстрокам.
        if len(t) <= 4:
            if any(tok == t for tok in tokens):
                return True
            continue

        # Для длинных терминов допускаем:
        # - точное совпадение токена ("эндокринолог")
        # - префиксное совпадение ("ультразвук" -> "ультразвуковой").
        if any(tok == t or tok.startswith(t) for tok in tokens):
            return True
    return False


def _collect_role_unit_names(doc: dict[str, Any], *, main_value: bool) -> list[str]:
    """
    Возвращает список названий подразделений врача по признаку main.

    :param doc: карточка врача из doctors.jsonl
    :param main_value: True для main=true, False для main=false fallback
    :return: уникализированный список unit names
    """
    out: list[str] = []
    unit_links = doc.get("unit_links") or []
    if isinstance(unit_links, list):
        for raw_link in unit_links:
            if not isinstance(raw_link, dict):
                continue
            if bool(raw_link.get("main")) != main_value:
                continue
            unit_name = str(raw_link.get("company_unit_name") or "").strip()
            if unit_name and unit_name not in out:
                out.append(unit_name)
    if main_value:
        # Поддержка старого формата кэша, где main-характеристика уже агрегирована в main_units.
        for raw in (doc.get("main_units") or []):
            unit_name = str(raw or "").strip()
            if unit_name and unit_name not in out:
                out.append(unit_name)
    if not main_value:
        # Для legacy-кэшей без unit_links/main берем units как fallback-связи.
        main_units_norm = {
            _normalise_input(str(x or ""))
            for x in (doc.get("main_units") or [])
            if str(x or "").strip()
        }
        for raw in (doc.get("units") or []):
            unit_name = str(raw or "").strip()
            if main_units_norm and _normalise_input(unit_name) in main_units_norm:
                continue
            if unit_name and unit_name not in out:
                out.append(unit_name)
    return out


def _doctor_role_specialty_match_level(doc: dict[str, Any], specialty: str) -> int:
    """
    Матчинг ролевого запроса по специальности с приоритетом main-полей.

    Уровни:
    - 2: найдено совпадение в unit name с main=true
    - 1: найдено совпадение в unit name с main=false / legacy units
    - 0: совпадений нет

    :param doc: карточка врача
    :param specialty: каноническая специальность (например, "хирург")
    :return: целочисленный приоритет совпадения
    """
    spec_norm = _normalise_input(specialty).replace("ё", "е")
    if not spec_norm:
        return 0

    main_true_units = _collect_role_unit_names(doc, main_value=True)
    if any(_matches_specialty_terms(unit_name, spec_norm) for unit_name in main_true_units):
        return 2

    main_false_units = _collect_role_unit_names(doc, main_value=False)
    if any(_matches_specialty_terms(unit_name, spec_norm) for unit_name in main_false_units):
        return 1

    return 0


def _doctor_main_payload(doc: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    """
    Извлекает main-поля врача из нового и старого формата кэша.

    :param doc: карточка врача из doctors.jsonl
    :return: (main_units, main_specializations, has_any_main_links)
    """
    main_units: list[str] = []
    for raw in (doc.get("main_units") or []):
        value = str(raw or "").strip()
        if value and value not in main_units:
            main_units.append(value)

    main_specs: list[str] = []
    for raw in (doc.get("main_specializations") or []):
        value = str(raw or "").strip()
        if value and value not in main_specs:
            main_specs.append(value)

    has_main_links = False
    unit_links = doc.get("unit_links") or []
    if isinstance(unit_links, list):
        for raw_link in unit_links:
            if not isinstance(raw_link, dict):
                continue
            if not bool(raw_link.get("main")):
                continue
            has_main_links = True
            unit_name = str(raw_link.get("company_unit_name") or "").strip()
            link_spec = str(raw_link.get("specialization") or "").strip()
            if unit_name and unit_name not in main_units:
                main_units.append(unit_name)
            if link_spec and link_spec not in main_specs:
                main_specs.append(link_spec)

    if main_units or main_specs:
        has_main_links = True
    return main_units, main_specs, has_main_links


def _iter_unit_link_specs(doc: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """
    Возвращает список описаний из unit_links в виде (unit_name, specialization, main_flag).

    :param doc: карточка врача
    :return: список описаний по связям подразделений
    """
    out: list[tuple[str, str, bool]] = []
    unit_links = doc.get("unit_links") or []
    if not isinstance(unit_links, list):
        return out
    for raw_link in unit_links:
        if not isinstance(raw_link, dict):
            continue
        unit_name = str(raw_link.get("company_unit_name") or "").strip()
        link_spec = str(raw_link.get("specialization") or "").strip()
        if not link_spec:
            continue
        out.append((unit_name, link_spec, bool(raw_link.get("main"))))
    return out


def _specialization_matches_specialty(unit_name: str, link_spec: str, specialty: str) -> bool:
    """
    Проверяет, относится ли specialization-блок к нужной специальности.

    :param unit_name: имя подразделения врача (company_unit_name)
    :param link_spec: текст specialization для связи
    :param specialty: целевая специальность
    :return: True, если specialization релевантен специальности
    """
    spec_norm = _normalise_input(specialty).replace("ё", "е")
    if not spec_norm:
        return False
    # В ролевом режиме (по специальности) опираемся именно на unit_name.
    # Иначе длинный текст specialization может содержать "чужие" термины
    # и подмешивать нерелевантные блоки описания.
    return _matches_specialty_terms(unit_name, spec_norm)


def _pick_display_specialization(
    doc: dict[str, Any],
    *,
    preferred_specialty: str = "",
    preferred_service: str = "",
) -> str:
    """
    Выбирает описание врача для UI без «перепутанных» блоков специализации.

    Приоритет:
    1) main=true + совпадение с запрошенной специальностью/услугой
    2) main=false + совпадение с запрошенной специальностью/услугой
    3) любой main=true specialization
    4) любой main=false specialization
    5) top-level specialization из кэша

    :param doc: карточка врача
    :param preferred_specialty: специальность из запроса (если есть)
    :param preferred_service: услуга/процедура из запроса (если есть)
    :return: выбранный текст specialization
    """
    spec_norm = _normalise_input(preferred_specialty).replace("ё", "е")
    service_norm = _normalise_input(preferred_service)
    link_specs = _iter_unit_link_specs(doc)

    main_true_matched: list[str] = []
    main_false_matched: list[str] = []
    main_true_any: list[str] = []
    main_false_any: list[str] = []

    for unit_name, link_spec, is_main in link_specs:
        if is_main:
            if link_spec not in main_true_any:
                main_true_any.append(link_spec)
        else:
            if link_spec not in main_false_any:
                main_false_any.append(link_spec)

        is_match = False
        if spec_norm and _specialization_matches_specialty(unit_name, link_spec, spec_norm):
            is_match = True
        if not is_match and service_norm:
            # Для процедурных запросов match по тексту specialization.
            if _doctor_matches_service({"specialization": link_spec, "unit_links": [], "main_specializations": []}, service_norm):
                is_match = True

        if is_match:
            if is_main:
                if link_spec not in main_true_matched:
                    main_true_matched.append(link_spec)
            else:
                if link_spec not in main_false_matched:
                    main_false_matched.append(link_spec)

    if main_true_matched:
        return main_true_matched[0]
    if main_false_matched:
        return main_false_matched[0]

    _, main_specs, _ = _doctor_main_payload(doc)
    if main_specs:
        return main_specs[0]
    if main_true_any:
        return main_true_any[0]
    if main_false_any:
        return main_false_any[0]
    return str(doc.get("specialization") or "")


def _is_role_specialty_query(query_text: str, specialty: str) -> bool:
    """
    Определяет, является ли запрос ролевым (по специальности), а не процедурным.

    Логика:
    - Для большинства специальностей считаем запрос ролевым.
    - Для УЗИ различаем:
      - role: "какие узисты", "врач узи";
      - procedure: "узи брюшной полости", "сделать узи ...".

    :param query_text: текст запроса пользователя
    :param specialty: распознанная специальность
    :return: True для ролевого сценария, False для процедурного
    """
    spec_norm = _normalise_input(specialty).replace("ё", "е")
    query_norm = _normalise_input(query_text).replace("ё", "е")
    if not spec_norm:
        return False
    if spec_norm != "узи":
        return True
    if _UZI_ROLE_HINT_RE.search(query_norm) and not _UZI_PROCEDURE_HINT_RE.search(query_norm):
        return True
    return False


def _doctor_matches_specialty(doc: dict[str, Any], specialty: str, query_text: str) -> bool:
    """
    Проверяет соответствие врача специальности с учетом нового поля main.

    Правило:
    - role-запрос (например, "какие хирурги"): сначала матчим по main=true;
      если main-связей у врача нет, используем fallback по старому текстовому профилю.
    - процедурный запрос (например, "узи брюшной полости"): используем старый путь
      по specialization/эвристикам процедуры.

    :param doc: карточка врача
    :param specialty: искомая специальность
    :param query_text: исходный запрос пользователя
    :return: True, если врач подходит под фильтр
    """
    spec_norm = _normalise_input(specialty).replace("ё", "е")
    if not spec_norm:
        return False

    role_query = _is_role_specialty_query(query_text, spec_norm)
    if spec_norm == "узи" and not role_query:
        return _matches_uzi_doctor_profile(doc)

    if role_query:
        return _doctor_role_specialty_match_level(doc, spec_norm) > 0

    fio = _normalise_input(str(doc.get("fio", "")))
    spec_text = _normalise_input(str(doc.get("specialization", "")))
    regions = " ".join([_normalise_input(str(x)) for x in (doc.get("regions") or []) if str(x).strip()])
    units = " ".join([_normalise_input(str(x)) for x in (doc.get("units") or []) if str(x).strip()])
    hay = " | ".join([fio, spec_text, regions, units])
    if spec_norm == "узи":
        return _matches_uzi_doctor_profile(doc)
    if spec_norm in hay:
        return True
    return _matches_specialty_terms(hay, spec_norm)


def _stem_service_token(token: str) -> str:
    """
    Упрощенный стемминг русских слов для match процедур (без NLP-библиотек).

    :param token: токен услуги
    :return: укороченный вариант токена
    """
    t = str(token or "").strip().lower().replace("ё", "е")
    if len(t) < 5:
        return t
    endings = (
        "иями",
        "ями",
        "ами",
        "иями",
        "ией",
        "ия",
        "ие",
        "ию",
        "ии",
        "ой",
        "ей",
        "ом",
        "ем",
        "ах",
        "ях",
        "ам",
        "ям",
        "ый",
        "ий",
        "ая",
        "ое",
        "ые",
        "ую",
        "ого",
        "ему",
        "ым",
        "им",
        "у",
        "а",
        "я",
    )
    for suffix in endings:
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)]
    return t


def _service_tokens(service_name: str) -> list[str]:
    """
    Нормализует service_name в информативные токены.

    :param service_name: название услуги/процедуры
    :return: список токенов для поиска в специализации врача
    """
    raw_tokens = re.findall(r"[a-zа-яё0-9]{2,}", _normalise_input(service_name))
    out: list[str] = []
    for tok in raw_tokens:
        t = tok.lower().replace("ё", "е")
        if t in _SERVICE_FILTER_STOPWORDS:
            continue
        if len(t) < 3:
            continue
        if t not in out:
            out.append(t)
    return out


def _doctor_matches_service(doc: dict[str, Any], service_name: str) -> bool:
    """
    Проверяет, выполняет ли врач конкретную процедуру/услугу.

    В этом фильтре используем только профильные текстовые поля врача
    (specialization и link-level specialization), т.к. запрос процедурный.

    :param doc: карточка врача
    :param service_name: название услуги от NLU/эвристики
    :return: True, если в профиле врача найдено совпадение по услуге
    """
    tokens = _service_tokens(service_name)
    if not tokens:
        return False

    parts: list[str] = [str(doc.get("specialization") or "")]
    for raw in (doc.get("main_specializations") or []):
        parts.append(str(raw or ""))
    for raw_link in (doc.get("unit_links") or []):
        if isinstance(raw_link, dict):
            parts.append(str(raw_link.get("specialization") or ""))
    hay = _normalise_input(" ".join(parts)).replace("ё", "е")
    if not hay:
        return False

    matched = 0
    for tok in tokens:
        stem = _stem_service_token(tok)
        if tok in hay or (stem and stem in hay):
            matched += 1

    if len(tokens) == 1:
        return matched >= 1
    return matched >= min(len(tokens), 2)


def _iter_slot_datetimes(schedule: dict[str, Any]) -> list[datetime]:
    out: list[datetime] = []
    if not isinstance(schedule, dict):
        return out
    for days in schedule.values():
        if not isinstance(days, list):
            continue
        for day in days:
            if not isinstance(day, dict):
                continue
            day_date = str(day.get("date") or day.get("curDate") or "").strip()
            if not day_date:
                continue
            slots = day.get("slots") or []
            if not isinstance(slots, list):
                continue
            for slot in slots:
                t = str(slot or "").strip()
                if len(t) < 5:
                    continue
                try:
                    out.append(datetime.fromisoformat(f"{day_date}T{t[:5]}:00"))
                except Exception:
                    continue
    return out


def _get_first_present(d: dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _as_int(val: Any) -> int | None:
    try:
        return int(val)
    except Exception:
        return None


def _fio_tokens(text: str) -> list[str]:
    return [t.lower() for t in _FIO_TOKEN_RE.findall(str(text or ""))]


def _doctor_matches_fio(fio: str, doctor_query: str, resolved_surname: str | None = None) -> bool:
    """
    Проверка фамилии/ФИО ТОЛЬКО по fio врача.
    Не ищем по specialization, чтобы "Ким" не матчился на "хроническим".
    """
    tokens = _fio_tokens(fio)
    if not tokens:
        return False

    candidates: list[str] = []
    if resolved_surname:
        candidates.extend([v for v in surname_variants(resolved_surname) if v])
    else:
        q_tokens = _fio_tokens(doctor_query)
        if q_tokens:
            candidates.extend([v for v in surname_variants(q_tokens[0]) if v])

    normalized = sorted({ _normalise_input(x) for x in candidates if len(_normalise_input(x)) >= 2 }, key=len, reverse=True)
    if not normalized:
        return False

    for token in tokens:
        for c in normalized:
            if token.startswith(c):
                return True
    return False


def _compact_specialization(
    text: str,
    max_lines: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    Сжимает только дубли строк в специализации.
    Ограничения по строкам/символам отключены по умолчанию (полный текст),
    но могут быть включены параметрами max_lines/max_chars.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if not ln:
            continue
        key = re.sub(r"\s+", " ", ln).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
        if isinstance(max_lines, int) and max_lines > 0 and len(out) >= max_lines:
            break

    compact = "\n".join(out).strip()
    if isinstance(max_chars, int) and max_chars > 0 and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


def _dedupe_doctors_by_fio(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in docs:
        fio = _normalise_input(str(d.get("fio") or ""))
        if not fio or fio in seen:
            continue
        seen.add(fio)
        out.append(d)
    return out


def _extract_result_query_fields(entities: dict[str, Any], query: str) -> dict[str, Any]:
    surname = _get_first_present(entities, ["surname", "result_surname"])
    filial = _get_first_present(entities, ["filial", "result_filial"])
    year_raw = entities.get("year")
    number_raw = entities.get("number")

    if number_raw is None:
        number_raw = entities.get("order_id")

    year = _as_int(year_raw)
    number = _as_int(number_raw)
    return {
        "surname": str(surname or "").strip(),
        "year": year,
        "filial": str(filial or "").strip(),
        "number": number,
        "lang": _get_first_present(entities, ["lang", "result_lang"]) or "ru",
    }


def _cp1251_urlencode(value: str) -> str:
    """
    Кодирование параметров под контракт ссылки naykalab/getanaliz:
    Windows-1251 + URL-encode.
    """
    raw = str(value or "").strip().encode("cp1251", errors="replace")
    return quote_from_bytes(raw, safe="")


def _build_public_result_link(fields: dict[str, Any]) -> str | None:
    surname = str(fields.get("surname") or "").strip()
    filial = str(fields.get("filial") or "").strip()
    year = _as_int(fields.get("year"))
    number = _as_int(fields.get("number"))
    if not surname or not filial or year is None or number is None:
        return None
    return (
        "https://naykalab.ru/getanaliz.php"
        f"?fam={_cp1251_urlencode(surname)}"
        f"&year={year}"
        f"&nom={_cp1251_urlencode(filial)}"
        f"&nom2={number}"
        "&fast=1"
    )


def _region_display_name(region: dict[str, Any]) -> str:
    """
    Берем человекочитаемый адрес из live /regions.
    Приоритет: addressForSite -> name.
    """
    addr = str(region.get("addressForSite") or "").strip()
    name = str(region.get("name") or "").strip()
    return addr or name


def _looks_like_real_address(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if re.fullmatch(r"ID\s+\d+", s, flags=re.I):
        return False
    if _ADDRESS_HINT_RE.search(s):
        return True
    if re.search(r"\d", s):
        return True
    return False


def _service_query_matches(service_q: str, service_name: str) -> bool:
    sq = _normalise_input(service_q)
    sn = _normalise_input(service_name)
    if not sq or not sn:
        return False
    if sq in sn:
        return True
    if "анализ" in sq or "лаборатор" in sq:
        analysis_tokens = (
            "анализ",
            "лаборатор",
            "биоматериал",
            "взятие",
            "кров",
            "моч",
            "мазок",
            "сыворот",
            "плазм",
        )
        return any(tok in sn for tok in analysis_tokens)
    if sq == "экг":
        return "экг" in sn or "электрокардиограм" in sn
    return False


def _is_procedure_branch_lookup_query(query_text: str, service_q: str) -> bool:
    """
    Определяет, что пользователь ищет филиал под конкретную процедуру.

    :param query_text: исходный текст запроса
    :param service_q: нормализованная процедура/услуга
    :return: True, если это адресный lookup по процедуре
    """
    if not str(service_q or "").strip():
        return False
    q = _normalise_input(query_text or "")
    if not q:
        return False
    return bool(_PROCEDURE_BRANCH_LOOKUP_RE.search(q))


def _soft_address_match(left: str, right: str) -> bool:
    """
    Мягко сопоставляет два адреса из разных источников (API/кэш).

    :param left: адрес из первого источника
    :param right: адрес из второго источника
    :return: True, если строки похожи и описывают один филиал
    """
    l = _normalise_input(left)
    r = _normalise_input(right)
    if not l or not r:
        return False
    if l == r or l in r or r in l:
        return True
    lc = re.sub(r"[^a-zа-я0-9]+", "", l)
    rc = re.sub(r"[^a-zа-я0-9]+", "", r)
    if not lc or not rc:
        return False
    return lc == rc or lc in rc or rc in lc


def _static_procedure_addresses(service_q: str) -> list[str]:
    """
    Возвращает статические адреса для процедур с известными API-пробелами.

    :param service_q: нормализованное имя процедуры
    :return: список адресов филиалов
    """
    sq = _normalise_input(service_q or "")
    if not sq:
        return []
    out: list[str] = []
    for needle, addresses in _STATIC_PROCEDURE_BRANCH_OVERRIDES.items():
        if needle in sq:
            for addr in addresses:
                if addr not in out:
                    out.append(addr)
    return out


def _addresses_to_branch_payload(
    addresses: list[str],
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Преобразует список адресов в унифицированный branch payload.

    Если адрес находится в live `/regions`, дополняем id/phone/work_time.
    Иначе возвращаем минимальную карточку адреса.

    :param addresses: адреса филиалов
    :param regions: live-список филиалов
    :return: список branches для `address_info`
    """
    out: list[dict[str, Any]] = []
    for addr in addresses:
        best_region: dict[str, Any] | None = None
        for region in regions:
            if not isinstance(region, dict):
                continue
            disp = _region_display_name(region)
            if not disp:
                continue
            if _soft_address_match(addr, disp):
                best_region = region
                break

        out.append(
            {
                "id": _as_int(best_region.get("id")) if isinstance(best_region, dict) else None,
                "address": _region_display_name(best_region) if isinstance(best_region, dict) else addr,
                "city": str(best_region.get("city") or "").strip() if isinstance(best_region, dict) else "Самара",
                "phone": _extract_region_phone(best_region) if isinstance(best_region, dict) else "",
                "work_time": _extract_region_work_time(best_region) if isinstance(best_region, dict) else "",
            }
        )
    return out


def _extract_homecode_query(text: str) -> str:
    s = _normalise_input(text)
    m = _PRICE_HOMECODE_DOTTED_RE.search(s)
    if m:
        return str(m.group(0)).strip()
    m = _PRICE_HOMECODE_NUM_RE.search(s)
    if m:
        return str(m.group(0)).strip()
    return ""


def _price_query_tokens(text: str) -> list[str]:
    s = _normalise_input(text).replace("ё", "е")
    out: list[str] = []
    for t in _PRICE_TOKEN_RE.findall(s):
        token = str(t or "").strip().lower().replace("ё", "е")
        token = _PRICE_QUERY_CANONICAL_TOKENS.get(token, token)
        if len(token) < 2:
            continue
        if token in _PRICE_QUERY_STOPWORDS:
            continue
        out.append(token)
        # "прием" и "консультация" считаем взаимозаменяемыми для ранжирования цен.
        if token.startswith("прием") and "консультац" not in out:
            out.append("консультац")
        elif token.startswith("консультац") and "прием" not in out:
            out.append("прием")
    return out


def _extract_price_service_from_query(query: str) -> str | None:
    raw = str(query or "").strip()
    if not raw:
        return None
    if _PRICE_CONSULT_HINT_RE.search(raw):
        return "прием"
    q = _normalise_input(raw)
    q = _PRICE_DOCTOR_SUFFIX_RE.sub("", q).strip(" ?!.,;:")
    q = _PRICE_SERVICE_PREFIX_RE.sub("", q).strip(" ?!.,;:")
    if not q:
        return None
    if q in {"цена", "стоимость"}:
        return None
    words = [w for w in q.split() if w]
    if not words:
        return None
    filtered_words = [w for w in words if w not in _PRICE_QUERY_STOPWORDS]
    if filtered_words:
        words = filtered_words
    # Ограничиваем длину candidate, чтобы не тянуть в ranking целый диалог.
    return " ".join(words[:8])


def _dedupe_price_queries(queries: list[str], *, max_items: int = 8) -> list[str]:
    """
    Дедуплицирует варианты price-запроса перед поиском по каталогу услуг.

    :param queries: список сырых вариантов запроса
    :param max_items: максимальное количество вариантов
    :return: очищенный список уникальных запросов
    """

    out: list[str] = []
    seen: set[str] = set()
    for raw in queries:
        value = str(raw or "").strip()
        if not value:
            continue
        key = _normalise_input(value).replace("ё", "е")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _build_price_catalog_queries(query_text: str, *, current_service_name: str = "") -> list[str]:
    """
    Собирает варианты запроса для поиска услуги в price-каталоге.

    :param query_text: исходный текст пользователя
    :param current_service_name: уже известная услуга из state
    :return: список поисковых вариантов от самых полезных к запасным
    """

    raw = str(query_text or "").strip()
    queries: list[str] = []
    if current_service_name:
        queries.append(current_service_name)
    extracted = _extract_price_service_from_query(raw)
    if extracted:
        queries.append(extracted)
    phrase = extract_service_phrase(raw)
    if phrase:
        queries.append(phrase)
    compact = " ".join(_price_query_tokens(raw)).strip()
    if compact:
        queries.append(compact)
    if raw:
        queries.append(raw)
    return _dedupe_price_queries(queries)


def resolve_price_service_name_from_catalog(
    query_text: str,
    *,
    current_service_name: str = "",
    rows: list[dict[str, Any]] | None = None,
) -> str | None:
    """
    Приземляет пользовательский price-запрос на реальную услугу из price-каталога.

    Используется как узкий catalog-grounded слой для `PRICE`, чтобы не
    перечислять лабораторные анализы и процедуры в regex/anchors.

    :param query_text: исходный текст пользователя
    :param current_service_name: услуга из текущего state, если уже есть
    :param rows: опционально заранее загруженные строки priceByRegion
    :return: каноническое название услуги из каталога либо None
    """

    queries = _build_price_catalog_queries(query_text, current_service_name=current_service_name)
    if not queries:
        return None

    catalog_rows = rows
    if catalog_rows is None:
        try:
            loaded = api_price.load_price_by_region(SAMARA_PRICE_REGION_ID)
        except Exception:
            return None
        catalog_rows = [row for row in loaded if isinstance(row, dict)]
    else:
        catalog_rows = [row for row in rows if isinstance(row, dict)]

    if not catalog_rows:
        return None

    best_row: dict[str, Any] | None = None
    best_score = 0
    best_matched = 0
    for query in queries:
        ranked = _rank_price_rows(catalog_rows, query, limit=3)
        if not ranked:
            continue
        row = ranked[0]
        score, matched = _price_row_score(
            row,
            query=_normalise_input(query).replace("ё", "е"),
            tokens=_price_query_tokens(query),
            homecode_query=_extract_homecode_query(query),
        )
        if score <= 0:
            continue
        if score > best_score or (score == best_score and matched > best_matched):
            best_row = row
            best_score = score
            best_matched = matched

    if not best_row:
        return None
    if best_score < 100:
        return None
    return str(best_row.get("serviceName") or best_row.get("name") or "").strip() or None


def _is_city_only_reply(query: str) -> bool:
    """
    Проверяет, что реплика состоит только из города без новой услуги.

    :param query: текст текущей реплики
    :return: True, если пользователь просто ответил названием города
    """

    city = _extract_city_token(query)
    if not city:
        return False
    tokens = [t for t in re.findall(r"[a-zа-яё]+", _normalise_input(query)) if t]
    tokens = [t for t in tokens if t not in {"г", "город"}]
    return len(tokens) == 1 and tokens[0] == city


def _price_row_score(row: dict[str, Any], *, query: str, tokens: list[str], homecode_query: str) -> tuple[int, int]:
    name = _normalise_input(str(row.get("serviceName") or row.get("name") or "")).replace("ё", "е")
    homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
    if not name:
        return 0, 0

    score = 0
    if homecode_query:
        if homecode == homecode_query:
            score += 260
        elif homecode_query in homecode:
            score += 180

    if query:
        if query == name:
            score += 220
        elif query in name:
            score += 150

    matched = 0
    if tokens:
        for tok in tokens:
            if tok in name:
                matched += 1
        score += matched * 25
        if matched == len(tokens):
            score += 80
        elif matched >= max(2, len(tokens) - 1):
            score += 40

    # Слегка понижаем заведомо нерелевантный общий тариф.
    if "выезд на дом" in name and not any(tok in name for tok in tokens):
        score -= 30

    return score, matched


def _rank_price_rows(rows: list[dict[str, Any]], query_text: str, *, limit: int = 10) -> list[dict[str, Any]]:
    query = _normalise_input(query_text).replace("ё", "е")
    tokens = _price_query_tokens(query_text)
    homecode_query = _extract_homecode_query(query_text)

    scored: list[tuple[int, int, int, int, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score, matched = _price_row_score(row, query=query, tokens=tokens, homecode_query=homecode_query)
        if score <= 0:
            continue
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        name_gap = abs(len(name) - len(query)) if query else len(name)
        cost = _as_int(row.get("cost")) or 0
        scored.append((score, matched, -name_gap, -cost, row))

    if not scored:
        if query:
            fallback = [r for r in rows if isinstance(r, dict) and query in _normalise_input(str(r.get("serviceName") or ""))]
            if fallback:
                return fallback[:limit]
        if homecode_query:
            fallback = [
                r
                for r in rows
                if isinstance(r, dict)
                and homecode_query in _normalise_input(str(r.get("serviceHomecode") or r.get("homecode") or ""))
            ]
            if fallback:
                return fallback[:limit]
        return []

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, _, _, row in scored:
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        code = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
        key = (name, code)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _doctor_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    try:
        ord_value = int(doc.get("ord"))
    except Exception:
        ord_value = 10**9
    fio = _normalise_input(str(doc.get("fio") or ""))
    return ord_value, fio


def _specialty_priority_rank(doc: dict[str, Any], specialty: str) -> int:
    """
    Возвращает приоритет врача внутри специальности по бизнес-списку фамилий.

    :param doc: карточка врача
    :param specialty: специальность запроса
    :return: индекс приоритета (0..N-1), либо большой ранг если врач не в приоритете
    """
    spec_norm = _normalise_input(specialty).replace("ё", "е")
    priorities = _SPECIALTY_PRIORITY_SURNAMES.get(spec_norm)
    if not priorities:
        return 10**6

    fio_norm = _normalise_input(str(doc.get("fio") or "")).replace("ё", "е")
    if not fio_norm:
        return 10**6
    fio_tokens = [token for token in re.findall(r"[a-zа-я0-9]+", fio_norm) if token]
    if not fio_tokens:
        return 10**6
    surname = fio_tokens[0]

    for idx, wanted in enumerate(priorities):
        if surname.startswith(wanted):
            return idx
    return 10**6


def _coerce_top_n(value: Any, *, default: int = DOCTORS_TOP_N) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(1, min(parsed, 20))


def _schedule_regions_with_free_slots(schedule: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not isinstance(schedule, dict):
        return out
    for region_name, days in schedule.items():
        if not isinstance(days, list):
            continue
        has_free = False
        for day in days:
            if not isinstance(day, dict):
                continue
            slots = day.get("slots")
            if not isinstance(slots, list):
                continue
            if any(str(s or "").strip() for s in slots):
                has_free = True
                break
        if has_free:
            region_clean = str(region_name or "").strip()
            if region_clean and region_clean not in out:
                out.append(region_clean)
    return out


def _extract_region_phone(region: dict[str, Any]) -> str:
    phone_keys = ("phone", "phoneForSite", "phones", "phoneNumbers", "tel", "telephone")
    for k in phone_keys:
        v = region.get(k)
        if isinstance(v, str):
            nums = _PHONE_EXTRACT_RE.findall(v)
            if nums:
                return ", ".join(dict.fromkeys(n.strip() for n in nums))
            if v.strip():
                return v.strip()
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                if isinstance(item, str):
                    nums = _PHONE_EXTRACT_RE.findall(item)
                    parts.extend(nums or [item.strip()])
                elif isinstance(item, dict):
                    val = str(item.get("phone") or item.get("value") or "").strip()
                    if val:
                        parts.append(val)
            clean = [p for p in parts if p]
            if clean:
                return ", ".join(dict.fromkeys(clean))
    return ""


def _extract_region_work_time(region: dict[str, Any]) -> str:
    work_keys = (
        "workTime",
        "work_time",
        "worktime",
        "workHours",
        "work_hours",
        "schedule",
        "scheduleForSite",
        "openingHours",
        "hours",
        "mode",
    )
    for k in work_keys:
        v = region.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if str(x).strip()]
            if parts:
                return "; ".join(parts)
        if isinstance(v, dict):
            parts = []
            for kk, vv in v.items():
                txt = str(vv).strip()
                if txt:
                    parts.append(f"{kk}: {txt}")
            if parts:
                return "; ".join(parts)
    return ""


def _norm_city(s: str) -> str:
    t = _normalise_input(s or "")
    t = t.replace("ё", "е")
    t = re.sub(r"^г\.?\s*", "", t)
    return t.strip()


@lru_cache(maxsize=1)
def _load_nonbookable_points() -> dict[str, list[dict[str, Any]]]:
    if not _NONBOOKABLE_POINTS_PATH.exists():
        return {}
    try:
        raw = json.loads(_NONBOOKABLE_POINTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for city, rows in raw.items():
        if not isinstance(city, str) or not isinstance(rows, list):
            continue
        key = _norm_city(city)
        if not key:
            continue
        norm_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            norm_rows.append(
                {
                    "address": addr,
                    "phone": str(row.get("phone") or "").strip(),
                    "work_time": str(row.get("work_time") or "").strip(),
                    "has_analysis": bool(row.get("has_analysis", True)),
                    "has_ekg": bool(row.get("has_ekg", False)),
                    "city": str(row.get("city") or city).strip(),
                }
            )
        if norm_rows:
            out[key] = norm_rows
    return out


def _nonbookable_needs(service_q: str) -> tuple[bool, bool]:
    s = _normalise_input(service_q or "")
    if not s:
        return False, False
    need_analysis = bool(re.search(r"\b(анализ\w*|лаборатор\w*|биоматериал)\b", s))
    need_ekg = bool(re.search(r"\b(экг|электрокардиограм\w*)\b", s))
    return need_analysis, need_ekg


def _static_nonbookable_branches(city: str, service_q: str) -> list[dict[str, Any]]:
    data = _load_nonbookable_points()
    city_key = _norm_city(city)
    if not city_key:
        return []
    rows = data.get(city_key) or []
    if not rows:
        return []
    need_analysis, need_ekg = _nonbookable_needs(service_q)
    out: list[dict[str, Any]] = []
    for row in rows:
        if need_analysis and not bool(row.get("has_analysis", True)):
            continue
        if need_ekg and not bool(row.get("has_ekg", False)):
            continue
        out.append(dict(row))
    return out


def _filter_regions_by_service_flags(regions: list[dict[str, Any]], service_q: str) -> list[dict[str, Any]]:
    sq = _normalise_input(service_q or "")
    if not sq:
        return list(regions)

    need_analysis, need_ekg = _nonbookable_needs(sq)
    need_uzi = bool(_UZI_QUERY_RE.search(sq))
    need_doctor = False
    if not (need_analysis or need_ekg or need_uzi):
        need_doctor = (
            "прием" in sq
            or "приём" in sq
            or "консультац" in sq
            or "осмотр" in sq
            or bool(_extract_specialty_from_text(sq))
        )

    if not (need_analysis or need_ekg or need_uzi or need_doctor):
        return list(regions)

    out: list[dict[str, Any]] = []
    for row in regions:
        if not isinstance(row, dict):
            continue
        if need_analysis and not bool(row.get("analysis")):
            continue
        if need_ekg and not bool(row.get("ecg")):
            continue
        if need_uzi and not bool(row.get("usi")):
            continue
        if need_doctor and not bool(row.get("doctorService")):
            continue
        out.append(row)
    return out


def _service_fallback(
    *,
    note: str,
    handoff_message: str,
    entities: dict[str, Any],
    reason: str = "service_error",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": note,
        "handoff_required": True,
        "handoff_reason": reason,
        "handoff_message": handoff_message,
        "entities_used": entities,
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _prepare_subject_hint(query: str, entities: dict[str, Any]) -> str:
    """
    Возвращает краткое название исследования/процедуры для fallback-ответа по подготовке.

    :param query: текст текущего запроса пользователя
    :param entities: текущие сущности роутера
    :return: короткая фраза с названием анализа или процедуры
    """

    entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
    for raw in (entity_query, query):
        phrase = _extract_prepare_entity_phrase(str(raw or "").strip())
        if phrase:
            return phrase
    return str(entity_query or query or "исследованию").strip()


def _prepare_clarify_response(query: str, entities: dict[str, Any], *, note: str) -> dict[str, Any]:
    """
    Возвращает безопасный non-handoff fallback для PREPARE, если точной инструкции не найдено.

    :param query: текст запроса пользователя
    :param entities: текущие сущности роутера
    :param note: диагностическая пометка источника
    :return: payload PREPARE без handoff_required
    """

    subject = _prepare_subject_hint(query, entities)
    return {
        "prepare": (
            f"Пока не удалось автоматически найти точные правила подготовки к «{subject}». "
            "Уточните полное название анализа или процедуры, и я попробую ещё раз."
        ),
        "note": note,
        "entities_used": entities,
    }


def _test_assist_clarify_response(entities: dict[str, Any], *, note: str) -> dict[str, Any]:
    """
    Возвращает безопасный non-handoff fallback для подбора анализов.

    :param entities: текущие сущности роутера
    :param note: диагностическая пометка источника
    :return: payload TEST_ASSIST без handoff_required
    """

    return {
        "tests": [],
        "promos": [],
        "message": (
            "Уточните, пожалуйста, какие симптомы, жалобы или цель обследования вас интересуют, "
            "и я помогу подобрать анализы."
        ),
        "note": note,
        "entities_used": entities,
    }


def _tax_doc_guidance_response(entities: dict[str, Any], *, note: str) -> dict[str, Any]:
    """
    Возвращает детерминированную ссылку на оформление справки для налогового вычета.

    :param entities: текущие сущности роутера
    :param note: диагностическая пометка источника
    :return: payload OTHER/doc без handoff_required
    """

    return {
        "content": "Заказ справки на налоговый вычет осуществляется на сайте https://naykalab.ru/spravka-nalogoviy-vichet",
        "note": note,
        "entities_used": entities,
    }


def _is_meili_error_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return "Ошибка поисковой системы" in text or "Meilisearch" in text


def _is_meili_no_matches_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    norm = _normalise_input(text)
    if not norm:
        return True
    return "совпадений не найдено" in norm


_DOC_RELEVANCE_STOPWORDS = {
    "как",
    "что",
    "где",
    "когда",
    "нужно",
    "нужна",
    "нужен",
    "нужны",
    "получить",
    "получения",
    "подскажите",
    "пожалуйста",
    "добрый",
    "день",
    "здравствуйте",
    "мне",
    "для",
    "по",
    "про",
    "это",
    "этого",
    "требуется",
    "делаете",
    "сколько",
    "стоимость",
    "стоимости",
}


def _doc_tokens(text: str) -> set[str]:
    norm = _normalise_input(text).replace("ё", "е")
    out: set[str] = set()
    for token in re.findall(r"[a-zа-я0-9]{3,}", norm):
        if token.isdigit() or token in _DOC_RELEVANCE_STOPWORDS:
            continue
        out.add(token)
    return out


def _is_main_index_relevant(query: str, content: str, *, doc_kind: str) -> bool:
    content_norm = _normalise_input(content).replace("ё", "е")
    query_norm = _normalise_input(query).replace("ё", "е")
    if not content_norm:
        return False

    if doc_kind == "tax":
        tax_anchors = (
            "налог",
            "вычет",
            "фнс",
            "налогов",
            "оплате медицинских услуг",
        )
        if not any(anchor in content_norm for anchor in tax_anchors):
            return False
    else:
        if "договор" in query_norm and "договор" not in content_norm:
            return False
        if "амбулатор" in query_norm and not ("амбулатор" in content_norm or "карт" in content_norm):
            return False
        if ("соревн" in query_norm or "допуск" in query_norm) and not (
            "соревн" in content_norm or "допуск" in content_norm
        ):
            return False
        if "справк" in query_norm and not any(x in query_norm for x in ("налог", "вычет", "фнс")) and not (
            "справк" in content_norm or "допуск" in content_norm
        ):
            return False

    q_tokens = _doc_tokens(query_norm)
    if not q_tokens:
        return True
    content_tokens = _doc_tokens(content_norm)
    return bool(q_tokens & content_tokens)


def _is_prepare_relevant(query: str, content: str) -> bool:
    query_norm = _normalise_input(query).replace("ё", "е")
    content_norm = _normalise_input(content).replace("ё", "е")
    if not content_norm:
        return False
    if not any(x in content_norm for x in ("подготов", "натощак", "перед процедур", "перед исследован")):
        return False

    anchor_groups = (
        ("фгдс", "фдгс", "фгс", "гастроскоп"),
        ("кольпоскоп",),
        ("вульвоскоп",),
        ("биопс",),
        ("узи",),
        ("анализ",),
        ("кров",),
        ("моч",),
        ("сперм",),
        ("холестерин", "липид", "липидограмма"),
    )
    query_groups = [group for group in anchor_groups if any(anchor in query_norm for anchor in group)]
    if query_groups and not all(any(anchor in content_norm for anchor in group) for group in query_groups):
        return False

    q_tokens = _doc_tokens(query_norm)
    c_tokens = _doc_tokens(content_norm)
    if q_tokens and not (q_tokens & c_tokens):
        return False
    return True


_PREPARE_GENERIC_HEADINGS = {
    "подготовка к исследованию",
    "подготовка к анализу",
    "подготовка к процедуре",
    "подготовка к обследованию",
}
_PREPARE_ACTIONABLE_HINTS = (
    "натощак",
    "за ",
    "час",
    "день",
    "сутк",
    "исключ",
    "воздерж",
    "перед",
    "утром",
    "вечером",
    "не ",
    "нельзя",
    "можно",
    "нужно",
    "рекоменду",
)
_PREPARE_TARGET_HINTS = (
    "фгдс",
    "фдгс",
    "фгс",
    "гастроскоп",
    "кольпоскоп",
    "вульвоскоп",
    "биопс",
    "пайпел",
    "узи",
    "анализ",
    "кров",
    "моч",
    "мазок",
    "холестерин",
    "липид",
    "пцр",
)
_PREPARE_CONTENT_TARGET_ANCHORS = (
    "кров",
    "моч",
    "биопс",
    "фгдс",
    "фдгс",
    "фгс",
    "гастроскоп",
    "кольпоскоп",
    "вульвоскоп",
    "мазок",
    "пцр",
    "липид",
    "холестерин",
    "эндоскоп",
)


def _is_prepare_content_actionable(content: str) -> bool:
    """
    Отсекает слишком общий/шаблонный текст подготовки.

    :param content: кандидатный текст подготовки
    :return: True, если текст выглядит содержательным
    """

    norm = _normalise_input(content).replace("ё", "е")
    if not norm:
        return False
    if norm in _PREPARE_GENERIC_HEADINGS:
        return False

    tokens = re.findall(r"[a-zа-я0-9]{3,}", norm)
    if len(tokens) <= 5 and ("подготовк" in norm and ("исследован" in norm or "анализ" in norm or "процедур" in norm)):
        return False

    if any(hint in norm for hint in _PREPARE_ACTIONABLE_HINTS):
        return True
    return len(tokens) >= 20


def _is_prepare_service_info_usable(query: str, content: str) -> bool:
    """
    Решает, можно ли принимать API-first результат `serviceInfoAll` без fallback.

    :param query: исходный пользовательский запрос
    :param content: текст подготовки из API-кэша
    :return: True, если ответ достаточно качественный и релевантный
    """

    if not _is_prepare_content_actionable(content):
        return False

    query_norm = _normalise_input(query).replace("ё", "е")
    has_specific_target = any(anchor in query_norm for anchor in _PREPARE_TARGET_HINTS)
    if has_specific_target and not _is_prepare_relevant(query, content):
        content_norm = _normalise_input(content).replace("ё", "е")
        if not any(anchor in content_norm for anchor in _PREPARE_CONTENT_TARGET_ANCHORS):
            return False
    return True


_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT = "В моей базе данных информации недостаточно, перевожу на оператора."


@dataclass
class Services:
    """
    Сервисный слой для patient/messenger router.

    - doctors_info: использует ФАЙЛОВЫЙ кэш врачей (JSONL) + in-memory кэш.
    - doctors_schedule_week: realtime с коротким TTL-кэшем и stale-fallback при сбоях API.
    """

    # in-memory кэш врачей
    # Приходится использовать идентификатор полей "field" и его свойство default_factory=list из @dataclass, так как list
    # относится к изменяемым типам данных. В противном случае переменная спика будет
    # рандомно перезаписываться в неожиданных местах (база).
    _doctors_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    # Тут неизменяемый тип str, не усложняем:
    _doctors_cache_path: Optional[str] = field(default=None, init=False)
    _doctors_cache_loaded_at: float = field(default=0.0, init=False)
    _regions_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _regions_cache_loaded_at: float = field(default=0.0, init=False)
    _procedure_rows_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _procedure_rows_loaded_at: float = field(default=0.0, init=False)
    _schedule_cache_client: AsyncListTTLStaleCache = field(init=False)

    # блокировка, чтобы несколько запросов параллельно не перегенерировали кэш
    _doctors_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _regions_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _procedure_rows_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # TTL in-memory кэша (латентность, сек, определят свежесть кэша)
    doctors_mem_ttl_seconds: int = 300
    regions_mem_ttl_seconds: int = 300
    procedure_rows_mem_ttl_seconds: int = 300
    schedule_fresh_ttl_seconds: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_FRESH_TTL_SECONDS",
            30,
            min_value=1,
            max_value=300,
        )
    )
    schedule_stale_ttl_seconds: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_STALE_TTL_SECONDS",
            600,
            min_value=1,
            max_value=3600,
        )
    )
    schedule_negative_ttl_seconds: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_NEGATIVE_TTL_SECONDS",
            15,
            min_value=1,
            max_value=120,
        )
    )
    schedule_cache_max_keys: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_CACHE_MAX_KEYS",
            1000,
            min_value=50,
            max_value=10000,
        )
    )
    schedule_cache_log_events: bool = field(
        default_factory=lambda: _runtime_bool("MR_SCHEDULE_CACHE_LOG_EVENTS", False)
    )

    def __post_init__(self) -> None:
        self._schedule_cache_client = AsyncListTTLStaleCache(
            fresh_ttl_seconds=self.schedule_fresh_ttl_seconds,
            stale_ttl_seconds=self.schedule_stale_ttl_seconds,
            negative_ttl_seconds=self.schedule_negative_ttl_seconds,
            max_keys=self.schedule_cache_max_keys,
            logger=logger,
            log_events=self.schedule_cache_log_events,
            name="schedule_cache",
            time_func=lambda: time.time(),
        )

    # -----------------------------
    # Low-level helpers (async)
    # -----------------------------

    def ensure_background_refresh_started(self) -> None:
        """Запускает фоновые refresh-задачи кэшей (idempotent)."""
        try:
            api_price.ensure_daily_price_refresh_started()
        except Exception:
            pass
        try:
            api_service_info.ensure_daily_service_info_refresh_started()
        except Exception:
            pass

    @staticmethod
    def _schedule_cache_key(last_name: str, region_name: str | None = None) -> tuple[str, str]:
        return _normalise_input(last_name), _normalise_input(region_name or "")

    async def _fetch_schedule_source(self, last_name: str, region_name: str | None = None) -> Any:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                if region_name:
                    try:
                        return await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name, region_name)
                    except TypeError:
                        return await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name)
                return await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.12)
                    continue
        if last_exc is not None:
            raise last_exc
        return []

    async def _get_schedule_payload_cached(self, last_name: str, region_name: str | None = None) -> Any:
        key = self._schedule_cache_key(last_name, region_name)
        return await self._schedule_cache_client.get_or_fetch(
            key,
            lambda: self._fetch_schedule_source(last_name, region_name),
            key_details={"last_name": key[0], "region": key[1]},
        )

    async def _ensure_doctors_cache_loaded(self) -> list[dict[str, Any]]:
        """
        1) Пытаемся получить свежий doctor cache через api_nayka.get_cached_doctors_data()
        2) При неуспехе откатываемся на последний непустой файл
        3) Держим in-memory-кэш поверх файлового кэша
        """
        now = time.time()

        # быстрый путь: ещё не протухло
        if self._doctors_cache and (now - self._doctors_cache_loaded_at) < self.doctors_mem_ttl_seconds:
            return self._doctors_cache

        async with self._doctors_lock:
            # повторная проверка под локом
            now = time.time()
            if self._doctors_cache and (now - self._doctors_cache_loaded_at) < self.doctors_mem_ttl_seconds:
                return self._doctors_cache

            try:
                doctors_loaded = await asyncio.to_thread(api_nayka.get_cached_doctors_data)
                file_path = await asyncio.to_thread(api_nayka.find_existing_doctors_file)
            except Exception:
                return self._doctors_cache or []

            self._doctors_cache = doctors_loaded
            self._doctors_cache_path = str(file_path) if file_path is not None else None
            self._doctors_cache_loaded_at = time.time()
            return self._doctors_cache

    async def _ensure_regions_loaded(self) -> list[dict[str, Any]]:
        """
        Live-список регионов/филиалов из Nayka API (/regions) с коротким in-memory TTL.
        """
        now = time.time()
        if self._regions_cache and (now - self._regions_cache_loaded_at) < self.regions_mem_ttl_seconds:
            return self._regions_cache

        async with self._regions_lock:
            now = time.time()
            if self._regions_cache and (now - self._regions_cache_loaded_at) < self.regions_mem_ttl_seconds:
                return self._regions_cache

            try:
                regions = await asyncio.to_thread(api_nayka.site_regions)
                if not isinstance(regions, list):
                    regions = []
            except Exception:
                regions = []

            self._regions_cache = regions
            self._regions_cache_loaded_at = time.time()
            return self._regions_cache

    async def _samara_region_tokens(self) -> set[str]:
        regions = await self._ensure_regions_loaded()
        tokens: set[str] = set()
        for r in regions:
            if not isinstance(r, dict):
                continue
            city = str(r.get("city") or "").strip()
            name = str(r.get("name") or "").strip()
            addr = str(r.get("addressForSite") or "").strip()
            if not (
                _is_samara_city_value(city)
                or "самара" in _normalise_input(name)
                or "самара" in _normalise_input(addr)
            ):
                continue
            for raw in (name, addr):
                n = _normalise_input(raw)
                if n:
                    tokens.add(n)
        return tokens

    async def _ensure_procedure_rows_loaded(self) -> list[dict[str, Any]]:
        """
        Готовит in-memory индекс процедур по филиалам из doctor_prices.

        Источник: `api_price.load_doctor_prices()` (кэш по branch-level regionId).
        Для защиты от мусора оставляем только самарские строки с реальными адресами.

        :return: строки вида {"serviceName": str, "regionName": str}
        """
        now = time.time()
        if self._procedure_rows_cache and (now - self._procedure_rows_loaded_at) < self.procedure_rows_mem_ttl_seconds:
            return self._procedure_rows_cache

        async with self._procedure_rows_lock:
            now = time.time()
            if self._procedure_rows_cache and (now - self._procedure_rows_loaded_at) < self.procedure_rows_mem_ttl_seconds:
                return self._procedure_rows_cache

            try:
                rows = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception:
                rows = []
            if not isinstance(rows, list):
                rows = []

            samara_tokens = await self._samara_region_tokens()
            filtered: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                service_name = str(row.get("serviceName") or "").strip()
                region_name = str(row.get("regionName") or "").strip()
                if not service_name or not region_name:
                    continue
                if not _looks_like_real_address(region_name):
                    continue
                if _is_explicit_non_samara_region(region_name):
                    continue
                if samara_tokens and not _region_matches_samara_tokens(region_name, samara_tokens):
                    continue
                filtered.append({"serviceName": service_name, "regionName": region_name})

            self._procedure_rows_cache = filtered
            self._procedure_rows_loaded_at = time.time()
            return self._procedure_rows_cache

    async def _procedure_branches_from_index(
        self,
        service_q: str,
        regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Находит филиалы для процедуры по индексу `doctor_prices`.

        :param service_q: нормализованная процедура
        :param regions: live-список филиалов /regions (для phone/work_time)
        :return: список branches в формате address_info
        """
        role_specialty = _procedure_query_role_specialty(service_q)
        if role_specialty:
            doctors = await self._ensure_doctors_cache_loaded()
            samara_tokens = await self._samara_region_tokens()
            role_addresses: list[str] = []
            for doc in doctors:
                if not isinstance(doc, dict):
                    continue
                if _doctor_role_specialty_match_level(doc, role_specialty) <= 0:
                    continue
                regions_src = [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()]
                if _has_explicit_non_samara_regions(regions_src):
                    continue
                for addr in regions_src:
                    if not _looks_like_real_address(addr):
                        continue
                    if samara_tokens and not _region_matches_samara_tokens(addr, samara_tokens):
                        continue
                    if addr not in role_addresses:
                        role_addresses.append(addr)
            if role_addresses:
                return _addresses_to_branch_payload(role_addresses, regions)

        rows = await self._ensure_procedure_rows_loaded()
        ranked = _rank_price_rows(rows, service_q, limit=200)

        addresses: list[str] = []
        for row in ranked:
            addr = str(row.get("regionName") or "").strip()
            if not addr or not _looks_like_real_address(addr):
                continue
            if addr not in addresses:
                addresses.append(addr)

        if not addresses:
            addresses = _static_procedure_addresses(service_q)
        return _addresses_to_branch_payload(addresses, regions)

    async def _schedule_by_specialty(
        self,
        specialty: str,
        entities: dict[str, Any],
        *,
        nearest_only: bool,
        query_text: str = "",
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Ищет расписание по специальности и возвращает найденные строки вместе с причиной пустой выдачи.

        :param specialty: каноническая специальность
        :param entities: текущие сущности диалога
        :param nearest_only: вернуть только ближайшего врача при наличии слотов
        :param query_text: исходный текст запроса пользователя
        :return: кортеж (список строк расписания, причина пустой выдачи или None)
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return [], None
        spec = _normalise_input(specialty)
        if not spec:
            return [], None

        samara_tokens = await self._samara_region_tokens()
        role_query = _is_role_specialty_query(query_text or specialty, spec)
        role_levels = {
            id(d): _doctor_role_specialty_match_level(d, spec)
            for d in doctors
        } if role_query else {}
        candidates = sorted(
            [
            d for d in doctors
            if (
                (role_levels.get(id(d), 0) > 0)
                if role_query
                else _doctor_matches_specialty(d, spec, query_text or specialty)
            )
            and not _has_explicit_non_samara_regions([str(x) for x in (d.get("regions") or []) if str(x).strip()])
            and (
                not samara_tokens
                or any(
                    _region_matches_samara_tokens(str(x), samara_tokens)
                    for x in (d.get("regions") or [])
                    if str(x).strip()
                )
            )
            ],
            key=(
                (lambda d: (
                    -role_levels.get(id(d), 0),
                    _specialty_priority_rank(d, spec),
                    *_doctor_sort_key(d),
                ))
                if role_query
                else _doctor_sort_key
            ),
        )[:8]
        if not candidates:
            return [], None

        out_rows: list[dict[str, Any]] = []
        matched_but_without_slots = False
        for doc in candidates:
            fio = str(doc.get("fio") or "").strip()
            if not fio:
                continue
            surname = fio.split()[0]
            try:
                data = await self._get_schedule_payload_cached(surname)
            except Exception:
                continue
            if _is_schedule_no_slots_text(data):
                matched_but_without_slots = True
                continue
            if not isinstance(data, list) or not data:
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                row_fio = str(row.get("fio") or "").strip()
                if row_fio and _normalise_input(row_fio) != _normalise_input(fio):
                    continue
                item = dict(row)
                display_spec = _pick_display_specialization(
                    doc,
                    preferred_specialty=spec,
                )
                item["specialization"] = _compact_specialization(display_spec)
                slots = _iter_slot_datetimes(item.get("schedule") or {})
                if slots:
                    item["_nearest_slot"] = min(slots)
                out_rows.append(item)
                break

        if not out_rows:
            if matched_but_without_slots:
                return [], "no_free_slots_2_weeks"
            return [], None
        with_slots = [x for x in out_rows if isinstance(x.get("_nearest_slot"), datetime)]
        if with_slots:
            with_slots.sort(key=lambda x: x["_nearest_slot"])  # type: ignore[index]
            chosen = with_slots[:1] if nearest_only else with_slots[:3]
        else:
            chosen = out_rows[:1] if nearest_only else out_rows[:3]

        for item in chosen:
            item.pop("_nearest_slot", None)
        return chosen, None

    async def _doctor_availability_snapshot(self, fio: str, *, samara_tokens: set[str]) -> dict[str, Any]:
        fio_clean = str(fio or "").strip()
        surname = fio_clean.split()[0] if fio_clean else ""
        if not surname:
            return {
                "available": False,
                "nearest_slot": "",
                "regions_with_slots": [],
                "note": "availability_missing_surname",
            }

        try:
            data = await self._get_schedule_payload_cached(surname)
        except Exception:
            return {
                "available": False,
                "nearest_slot": "",
                "regions_with_slots": [],
                "note": "availability_source_unavailable",
            }

        if not isinstance(data, list) or not data:
            return {
                "available": False,
                "nearest_slot": "",
                "regions_with_slots": [],
                "note": "availability_empty",
            }

        target_norm = _normalise_input(fio_clean)
        chosen: dict[str, Any] | None = None
        for row in data:
            if not isinstance(row, dict):
                continue
            row_fio = str(row.get("fio") or "").strip()
            if not row_fio:
                continue
            row_norm = _normalise_input(row_fio)
            if target_norm and row_norm == target_norm:
                chosen = row
                break
            if _doctor_matches_fio(row_fio, fio_clean, resolved_surname=surname):
                chosen = row
                break
        if chosen is None:
            chosen = next((row for row in data if isinstance(row, dict)), None)
        if not isinstance(chosen, dict):
            return {
                "available": False,
                "nearest_slot": "",
                "regions_with_slots": [],
                "note": "availability_unmatched",
            }

        schedule_raw = chosen.get("schedule")
        schedule: dict[str, Any] = {}
        if isinstance(schedule_raw, dict):
            if samara_tokens:
                for region_name, days in schedule_raw.items():
                    region = str(region_name or "").strip()
                    if not region:
                        continue
                    if _region_matches_samara_tokens(region, samara_tokens):
                        schedule[region] = days
            else:
                schedule = {str(k): v for k, v in schedule_raw.items()}

        slots = _iter_slot_datetimes(schedule)
        nearest_slot = min(slots).isoformat(timespec="minutes") if slots else ""
        return {
            "available": bool(slots),
            "nearest_slot": nearest_slot,
            "regions_with_slots": _schedule_regions_with_free_slots(schedule),
            "note": "availability_checked",
        }

    async def service_bundle_info(
        self,
        query: str,
        entities: dict[str, Any],
        *,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        top_limit = _coerce_top_n(top_n, default=DOCTORS_TOP_N)
        entity_service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        query_text = str(query or "").strip()
        if entity_service_name and _is_city_only_reply(query_text):
            query_service_name = None
        else:
            query_service_name = resolve_price_service_name_from_catalog(
                query_text,
                current_service_name=entity_service_name,
            ) or _extract_price_service_from_query(query_text)
        service_name = query_service_name or entity_service_name or query_text
        needle = _normalise_input(service_name)

        out: dict[str, Any] = {
            "service_name": service_name,
            "retail_prices": [],
            "doctors": [],
            "prepare": "",
            "top_n_applied": top_limit,
            "note": "service_bundle_info",
            "entities_used": {
                **entities,
                "service_name_effective": service_name,
            },
        }
        if not needle:
            out["note"] = "service_bundle_info: no service query"
            return out

        # 1) Retail price by city-level regionId (Самара = 3).
        try:
            retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
            out["retail_prices"] = _rank_price_rows(
                [p for p in retail_rows if isinstance(p, dict)],
                service_name,
                limit=5,
            )
        except Exception:
            out["retail_prices"] = []
            out["note"] = "service_bundle_info: retail source unavailable"

        # 2) Top-N doctors by ord among doctors that have the matched service in doctor prices.
        samara_tokens = await self._samara_region_tokens()
        doctors = await self._ensure_doctors_cache_loaded()
        by_id: dict[int, dict[str, Any]] = {}
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            doc_id = _as_int(doc.get("id"))
            if doc_id is None:
                continue
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            if _has_explicit_non_samara_regions(raw_regions):
                continue
            if samara_tokens and raw_regions and not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
            by_id[doc_id] = doc

        matched_price_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
        try:
            doctor_prices = await asyncio.to_thread(api_price.load_doctor_prices)
        except Exception:
            doctor_prices = []
        query_norm = _normalise_input(service_name).replace("ё", "е")
        query_tokens = _price_query_tokens(service_name)
        homecode_query = _extract_homecode_query(service_name)
        for row in doctor_prices:
            if not isinstance(row, dict):
                continue
            doctor_id = _as_int(row.get("doctorId"))
            if doctor_id is None or doctor_id not in by_id:
                continue
            score, matched = _price_row_score(
                row,
                query=query_norm,
                tokens=query_tokens,
                homecode_query=homecode_query,
            )
            if score <= 0:
                continue
            cost = _as_int(row.get("cost")) or 0
            matched_price_rows.append((score, matched, -cost, doctor_id, row))

        if not matched_price_rows and query_norm:
            # Мягкий fallback на substring, если ranker не дал совпадений.
            for row in doctor_prices:
                if not isinstance(row, dict):
                    continue
                doctor_id = _as_int(row.get("doctorId"))
                if doctor_id is None or doctor_id not in by_id:
                    continue
                service_row_name = _normalise_input(str(row.get("serviceName") or ""))
                if query_norm and query_norm in service_row_name:
                    cost = _as_int(row.get("cost")) or 0
                    matched_price_rows.append((1, 1, -cost, doctor_id, row))

        matched_price_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        best_row_by_doctor: dict[int, dict[str, Any]] = {}
        for _, _, _, doctor_id, row in matched_price_rows:
            if doctor_id not in best_row_by_doctor:
                best_row_by_doctor[doctor_id] = row

        doctor_cards = sorted(
            [by_id[doctor_id] for doctor_id in best_row_by_doctor if doctor_id in by_id],
            key=_doctor_sort_key,
        )[:top_limit]

        out_doctors: list[dict[str, Any]] = []
        for doc in doctor_cards:
            doctor_id = _as_int(doc.get("id"))
            if doctor_id is None:
                continue
            price_row = best_row_by_doctor.get(doctor_id, {})
            availability = await self._doctor_availability_snapshot(
                str(doc.get("fio") or ""),
                samara_tokens=samara_tokens,
            )
            out_doctors.append(
                {
                    "id": doctor_id,
                    "fio": str(doc.get("fio") or "").strip(),
                    "ord": _as_int(doc.get("ord")),
                    "specialization": _compact_specialization(str(doc.get("specialization") or "")),
                    "regions": [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()],
                    "service_price": _as_int(price_row.get("cost")),
                    "available": bool(availability.get("available")),
                    "nearest_slot": str(availability.get("nearest_slot") or ""),
                    "regions_with_slots": list(availability.get("regions_with_slots") or []),
                    "availability_note": str(availability.get("note") or ""),
                }
            )
        out["doctors"] = out_doctors

        # 3) Preparation guidance by service/test name.
        prepare_payload = await self.test_prepare(service_name, {"service_name": service_name})
        if isinstance(prepare_payload, dict) and not prepare_payload.get("handoff_required"):
            out["prepare"] = str(prepare_payload.get("prepare") or "").strip()

        return out

    # -----------------------------
    # NAUKA API used by router
    # -----------------------------

    async def resolve_doctor_name(self, raw_text_or_name: str) -> str | None:
        """
        Валидация кандидата фамилии/ФИО по актуальному кэшу врачей.
        Возвращает каноническую фамилию только если удалось сопоставить с кэшем.
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return None
        value = str(raw_text_or_name or "").strip()
        if not value:
            return None
        resolved = resolve_schedule_surname(value, doctors)
        if resolved:
            return resolved

        candidate = extract_doctor_name_candidate(value, prefer_schedule=True)
        if candidate and _normalise_input(candidate) != _normalise_input(value):
            return resolve_schedule_surname(candidate, doctors)
        return None

    async def _resolve_doctor_id_from_name(self, raw_text_or_name: str) -> tuple[int | None, str | None]:
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return None, None

        raw = str(raw_text_or_name or "").strip()
        if not raw:
            return None, None

        resolved_surname = resolve_schedule_surname(raw, doctors)
        samara_tokens = await self._samara_region_tokens()
        matched: list[dict[str, Any]] = []
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            fio = str(doc.get("fio") or "").strip()
            if not fio:
                continue
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            if _has_explicit_non_samara_regions(raw_regions):
                continue
            if samara_tokens and raw_regions and not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
            if _doctor_matches_fio(fio, raw, resolved_surname):
                matched.append(doc)

        if not matched:
            return None, None
        matched = sorted(matched, key=_doctor_sort_key)
        first = matched[0]
        return _as_int(first.get("id")), str(first.get("fio") or "").strip() or None

    async def doctors_info(self, query: str, entities: dict[str, Any], output_max: int | None = None) -> dict[str, Any]:
        """
        Возвращает список врачей из кэша (без real-time API).
        Фильтрация делается программно:
        - по фамилии/ФИО
        - по специализации
        - по региону/филиалу (по строкам regions/units если есть)
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return _service_fallback(
                note="doctors_info source unavailable",
                handoff_message="Сейчас не удалось получить список врачей автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"doctors": []},
            )

        q = _normalise_input(query)
        doctor_raw = _get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
        fio_q = _normalise_input(doctor_raw)
        spec_q = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
        if not spec_q:
            spec_q = _extract_specialty_from_text(query)
        service_q = _normalise_input(_get_first_present(entities, ["service_name", "test_name"]) or "")
        if not service_q:
            # Доверяем LLM в первую очередь, но если сущность не извлечена —
            # мягко подхватываем процедурную фразу только при явном сигнале услуг/процедур.
            if _SERVICE_QUERY_SIGNAL_RE.search(_normalise_input(query)):
                extracted_service = extract_service_phrase(query)
                if extracted_service:
                    service_q = _normalise_input(extracted_service)
        region_q = _normalise_input(_get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")
        resolved_surname = resolve_schedule_surname(doctor_raw, doctors) if doctor_raw else None

        query_candidate = extract_doctor_name_candidate(query, prefer_schedule=True) if query else None
        query_resolved = resolve_schedule_surname(query_candidate, doctors) if query_candidate else None
        if not resolved_surname:
            resolved_surname = query_resolved

        # При поиске по специальности игнорируем "залипший" doctor_name из прошлого контекста.
        if spec_q and not query_resolved:
            fio_q = ""
            resolved_surname = None

        samara_tokens = await self._samara_region_tokens()
        role_query = bool(spec_q and _is_role_specialty_query(query, spec_q))
        role_levels = {
            id(d): _doctor_role_specialty_match_level(d, spec_q)
            for d in doctors
        } if role_query else {}

        # если из entities пусто — попробуем хотя бы query как ключ
        # (но аккуратно: не хотим показывать всех врачей по любому вопросу)
        keyword = ""
        if fio_q:
            keyword = fio_q
        elif spec_q:
            keyword = spec_q
        elif region_q:
            keyword = region_q
        else:
            keyword = q

        keyword = keyword.strip()

        def match_doc(doc: dict[str, Any], ) -> bool:
            fio = _normalise_input(str(doc.get("fio", "")))
            spec_text = _normalise_input(str(doc.get("specialization", "")))
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            regions = " ".join([_normalise_input(x) for x in raw_regions])
            units = " ".join([_normalise_input(str(x)) for x in (doc.get("units") or [])])

            hay = " | ".join([fio, spec_text, regions, units])
            # Даже без live /regions не допускаем в выдачу явно не-самарские площадки.
            if _has_explicit_non_samara_regions(raw_regions):
                return False
            if samara_tokens:
                if not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                    return False
            if fio_q:
                if not _doctor_matches_fio(fio, fio_q, resolved_surname):
                    return False
            if spec_q:
                if role_query:
                    if role_levels.get(id(doc), 0) <= 0:
                        return False
                elif not _doctor_matches_specialty(doc, spec_q, query):
                    return False
            if service_q:
                if not _doctor_matches_service(doc, service_q):
                    return False
            if region_q and region_q not in hay:
                return False

            # если ничего конкретного не задано — используем keyword, но требуем хотя бы 3 символа
            if not (fio_q or spec_q or region_q or service_q):
                if len(keyword) < 3:
                    return False
                return keyword in hay

            return True

        filtered = [d for d in doctors if match_doc(d)]
        filtered = _dedupe_doctors_by_fio(filtered)
        if role_query:
            filtered = sorted(
                filtered,
                key=lambda d: (
                    -role_levels.get(id(d), 0),
                    _specialty_priority_rank(d, spec_q),
                    *_doctor_sort_key(d),
                ),
            )
        else:
            filtered = sorted(filtered, key=_doctor_sort_key)

        limit = _coerce_top_n(output_max, default=DOCTORS_TOP_N)
        if resolved_surname:
            # при явной фамилии врача не раздуваем выдачу.
            limit = min(limit, 3)

        # ограничим размер, чтобы не отправлять сотни карточек в LLM
        # (далее LLM/рендерер красиво завернёт)
        # Определить сколько тут карточек нужно в выводе обычно
        filtered = filtered[:limit]
        compact: list[dict[str, Any]] = []
        for d in filtered:
            row = dict(d)
            display_spec = _pick_display_specialization(
                row,
                preferred_specialty=spec_q,
                preferred_service=service_q,
            )
            row["specialization"] = _compact_specialization(display_spec)
            compact.append(row)

        return {
            "doctors": compact,
            "note": "doctors_info: from cached registry (jsonl)",
            "cache_file": self._doctors_cache_path,
            "entities_used": {
                "doctor_query": fio_q,
                "doctor_resolved": resolved_surname,
                "specialty_query": spec_q,
                "service_query": service_q,
                "region_query": region_q,
                "output_limit": limit,
            },
        }

    async def doctors_schedule_week(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Реалтайм расписание на неделю.

        Используем high-level функцию api_nayka.find_doctor_schedule(last_name, region_name=None)
        Она сама ходит в:
        - /doctors, /regions, /doctorRegions, /doctorCompanyUnits
        - /doctorSchedule + /doctorScheduleCells

        Возвращаем как есть (агрегированный список).
        """
        raw_name = _get_first_present(
            entities,
            ["last_name", "doctor_last_name", "doctor", "doctor_name", "fio"],
        )
        specialty = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
        query_specialty = _extract_specialty_from_text(query)
        query_procedure_specialty = _procedure_query_role_specialty(query or "")
        if query_specialty:
            specialty = query_specialty
        elif query_procedure_specialty:
            specialty = query_procedure_specialty
        if raw_name and _looks_like_schedule_specialty_token(str(raw_name)):
            raw_name = ""

        doctors = await self._ensure_doctors_cache_loaded()
        raw_for_match = str(raw_name or "").strip()
        # Если прилетело полное ФИО, для расписания берем фамилию (1-е слово),
        # иначе fuzzy-резолвер может схватить отчество и вернуть ложные матчи.
        if " " in raw_for_match:
            first = raw_for_match.split()[0].strip()
            raw_for_match = first or raw_for_match

        query_doctor_candidate = extract_doctor_name_candidate(str(query or ""), prefer_schedule=True) if query else None
        if query_doctor_candidate and _looks_like_schedule_specialty_token(str(query_doctor_candidate)):
            query_doctor_candidate = None
        query_name = resolve_schedule_surname(str(query_doctor_candidate), doctors) if query_doctor_candidate else None
        last_name = resolve_schedule_surname(raw_for_match, doctors)
        # Если в текущей реплике явно фигурирует другая фамилия по расписанию,
        # приоритет отдаем ей (сброс от залипшего doctor_name из state).
        has_schedule_signal = bool(_SCHEDULE_QUERY_RE.search(str(query or "")))
        if query_name and (
            not last_name
            or has_schedule_signal
            or _normalise_input(str(query_name)) != _normalise_input(str(last_name))
        ):
            last_name = query_name
        elif not last_name and query and query != raw_name:
            # Для запросов вида "расписание онколог/узи" не пытаемся
            # резолвить фамилию из всей фразы: это ведет к ложным doctor_name.
            if not specialty:
                last_name = query_name or resolve_schedule_surname(query, doctors)

        query_has_specialty_signal = bool(query_specialty or query_procedure_specialty)
        if query_has_specialty_signal and specialty and not query_name:
            # Текущая реплика явно про специальность/процедуру (например, ФГДС),
            # поэтому не используем "залипшую" фамилию из прошлых сообщений.
            last_name = None

        if not last_name and specialty:
            schedule_by_spec, schedule_unavailable_reason = await self._schedule_by_specialty(
                specialty,
                entities,
                nearest_only=_has_nearest_hint(query),
                query_text=query,
            )
            return {
                "schedule": schedule_by_spec,
                "note": "doctors_schedule_week: by specialty",
                "schedule_unavailable_reason": schedule_unavailable_reason,
                "entities_used": {"specialty": specialty, "raw_name": raw_name},
            }

        if not last_name:
            return {
                "schedule": [],
                "note": "doctors_schedule_week: missing doctor last name",
                "entities_used": entities,
            }

        # необязательный фильтр региона/филиала/города
        region_name = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city", "branch_name"])
        if region_name and _is_non_samara_city_value(region_name):
            return _service_fallback(
                note=f"doctors_schedule_week unsupported city: {region_name}",
                handoff_message="Сейчас могу помочь только по Самаре. Соединяю с оператором.",
                entities=entities,
                reason="city_not_supported",
                extra={"schedule": []},
            )
        if region_name and _is_samara_city_value(region_name):
            # Для города Самара не применяем region-фильтр в Nayka API:
            # endpoint ожидает branch-level region name, а city-value дает пустой/строковый ответ.
            region_name = None

        # Пробуем несколько вариантов фамилии (родительный падеж -> именительный).
        data = None
        schedule_unavailable_reason: str | None = None
        candidates = surname_variants(str(last_name))
        if not candidates:
            candidates = [str(last_name)]
        if query_name:
            for qv in surname_variants(str(query_name)):
                if qv not in candidates:
                    candidates.append(qv)

        try:
            for candidate in candidates:
                data = await self._get_schedule_payload_cached(candidate, region_name)
                if isinstance(data, list) and data:
                    last_name = candidate
                    break
                if _is_schedule_no_slots_text(data):
                    schedule_unavailable_reason = "no_free_slots_2_weeks"
                # fallback: если регионный фильтр дал пусто, пробуем без региона
                if region_name:
                    data = await self._get_schedule_payload_cached(candidate, None)
                    if isinstance(data, list) and data:
                        last_name = candidate
                        break
                    if _is_schedule_no_slots_text(data):
                        schedule_unavailable_reason = "no_free_slots_2_weeks"
        except Exception:
            return _service_fallback(
                note="doctors_schedule_week unavailable",
                handoff_message="Сейчас не удалось получить расписание автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"schedule": []},
            )

        # Защита от чрезмерно длинных/дублирующихся specialization блоков в realtime API.
        if isinstance(data, list):
            compact_data: list[dict[str, Any]] = []
            samara_tokens = await self._samara_region_tokens()
            doctor_by_fio = {
                _normalise_input(str(d.get("fio") or "")): d
                for d in doctors
                if isinstance(d, dict) and str(d.get("fio") or "").strip()
            }
            for row in data:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                row_fio_key = _normalise_input(str(item.get("fio") or ""))
                cache_doc = doctor_by_fio.get(row_fio_key)
                if cache_doc:
                    display_spec = _pick_display_specialization(
                        cache_doc,
                        preferred_specialty=specialty,
                    )
                else:
                    display_spec = str(item.get("specialization") or "")
                item["specialization"] = _compact_specialization(display_spec)
                regions_src = [str(x) for x in (item.get("regions") or []) if str(x).strip()]
                if _has_explicit_non_samara_regions(regions_src):
                    continue
                if samara_tokens:
                    if regions_src and not any(_region_matches_samara_tokens(x, samara_tokens) for x in regions_src):
                        continue
                    sched = item.get("schedule")
                    if isinstance(sched, dict) and sched:
                        sched_filtered: dict[str, Any] = {}
                        for k, v in sched.items():
                            if _region_matches_samara_tokens(str(k), samara_tokens):
                                sched_filtered[k] = v
                        if sched_filtered:
                            item["schedule"] = sched_filtered
                        elif regions_src:
                            continue
                compact_data.append(item)
            data = compact_data

        return {
            "schedule": data or [],
            "note": "doctors_schedule_week: realtime from Nayka API",
            "schedule_unavailable_reason": schedule_unavailable_reason,
            "entities_used": {"last_name": last_name, "raw_name": raw_name, "region_name": region_name},
        }

    # Остальные методы пока как заглушки
    async def main_index_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {
                "content": "",
                "note": "main_index_info: no query",
                "entities_used": entities,
            }
        doc_kind = str(entities.get("doc_request_kind") or "").strip().lower()
        if doc_kind not in {"tax", "generic"}:
            norm_q = _normalise_input(q)
            doc_kind = "tax" if any(k in norm_q for k in ("налог", "вычет", "фнс")) else "generic"

        if doc_kind == "tax":
            return _tax_doc_guidance_response(entities, note="main_index_info: tax direct link")

        normalized_q = _normalise_input(q)
        fallback_queries: list[str] = []
        if doc_kind == "tax" and any(k in normalized_q for k in ("налог", "фнс", "вычет", "справк")):
            fallback_queries = [
                "справка для налоговой",
                "налоговый вычет",
                "справка об оплате медицинских услуг",
            ]

        queries = [q]
        for fq in fallback_queries:
            if _normalise_input(fq) != normalized_q:
                queries.append(fq)

        cleaned = ""
        relevant_hit = False
        try:
            for qq in queries:
                raw = await asyncio.to_thread(
                    meilisearch.search_meili,
                    "main_index",
                    qq,
                    output_mode="content_only",
                    max_chars=12000,
                )
                cleaned = html_cleaner.strip_html(raw).strip()
                if _is_meili_error_text(cleaned):
                    continue
                if _is_meili_no_matches_text(cleaned):
                    continue
                if _is_main_index_relevant(qq, cleaned, doc_kind=doc_kind):
                    relevant_hit = True
                    break
        except Exception:
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback unavailable")
            return _service_fallback(
                note="main_index_info source unavailable",
                handoff_message="Сейчас не удалось найти информацию автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"content": ""},
            )

        if _is_meili_error_text(cleaned):
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback error")
            return _service_fallback(
                note="main_index_info source unavailable",
                handoff_message="Сейчас не удалось найти информацию автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"content": ""},
            )

        if _is_meili_no_matches_text(cleaned):
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback no matches")
            return _service_fallback(
                note="main_index_info: no matches",
                handoff_message=_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT,
                entities=entities,
                reason="knowledge_not_found",
                extra={"content": ""},
            )

        if not relevant_hit:
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback weak relevance")
            return _service_fallback(
                note=f"main_index_info: weak relevance ({doc_kind})",
                handoff_message=_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT,
                entities=entities,
                reason="knowledge_not_found",
                extra={"content": ""},
            )

        return {
            "content": cleaned,
            "note": f"main_index_info: main_index ({doc_kind})",
            "entities_used": entities,
        }

    async def appointment_help(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        if query:
            try:
                raw = await asyncio.to_thread(
                    meilisearch.search_meili,
                    "main_index",
                    query,
                    output_mode="content_only",
                    max_chars=12000,
                )
                cleaned = html_cleaner.strip_html(raw)
            except Exception:
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message="Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            if _is_meili_error_text(cleaned):
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message="Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            return {"instructions": cleaned, "entities_used": entities}
        return {
            "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
            "entities_used": entities,
        }

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        test_name = _get_first_present(entities, ["test_name", "service_name"]) or query
        needle = _normalise_input(test_name)
        if not needle:
            return _test_assist_clarify_response(entities, note="test_assist: no test query")

        try:
            price_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
        except Exception:
            return _test_assist_clarify_response(entities, note="test_assist source unavailable")
        matches = _rank_price_rows([p for p in price_rows if isinstance(p, dict)], test_name, limit=10)

        if not matches:
            return _test_assist_clarify_response(entities, note=f"test_assist: no matches ({SAMARA_PRICE_REGION_ID})")

        return {
            "tests": matches,
            "promos": [],
            "note": f"test_assist: priceByRegion({SAMARA_PRICE_REGION_ID})",
            "entities_used": entities,
        }

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        raw_query = str(query or "").strip()
        entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
        # Для нового вопроса берем текст пользователя, чтобы не залипала старая услуга из контекста.
        q = raw_query or entity_query
        if not q:
            return {"prepare": "", "note": "no query", "entities_used": entities}

        api_cached_prepare = await self._prepare_from_analysis_api_cache(q, entities)
        if api_cached_prepare:
            api_cached_cleaned = html_cleaner.strip_html(api_cached_prepare).strip()
            if not _is_prepare_service_info_usable(q, api_cached_cleaned):
                api_cached_prepare = None
            else:
                api_cached_prepare = api_cached_cleaned
        if api_cached_prepare:
            return {
                "prepare": api_cached_prepare,
                "note": "prepare: serviceInfoAll",
                "entities_used": entities,
            }

        variants = _prepare_query_variants(q, entity_query)
        if not variants:
            variants = [q]

        saw_no_matches = False
        saw_non_empty = False
        saw_service_error = False
        for candidate in variants:
            try:
                raw = await asyncio.to_thread(
                    meilisearch.search_meili,
                    "main_index",
                    candidate,
                    output_mode="content_only",
                    max_chars=12000,
                )
                cleaned = html_cleaner.strip_html(raw).strip()
            except Exception:
                saw_service_error = True
                continue

            if _is_meili_error_text(cleaned):
                saw_service_error = True
                continue

            if _is_meili_no_matches_text(cleaned):
                saw_no_matches = True
                continue

            if not cleaned:
                continue

            saw_non_empty = True
            if _is_prepare_relevant(candidate, cleaned) or _is_prepare_relevant(q, cleaned):
                return {"prepare": cleaned, "entities_used": entities}

        if saw_non_empty:
            return _prepare_clarify_response(q, entities, note="prepare: weak relevance")

        if saw_no_matches or not saw_service_error:
            return _prepare_clarify_response(q, entities, note="prepare: no matches")

        return _prepare_clarify_response(q, entities, note="prepare source unavailable")

    async def _prepare_from_analysis_api_cache(self, query: str, entities: dict[str, Any]) -> str | None:
        """
        Ищет подготовку к анализу в кэше `serviceInfoAll`.

        :param query: текст запроса пользователя
        :param entities: текущие сущности роутера
        :return: текст поля `preparation` или None
        """

        entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
        queries = _prepare_service_info_queries(query, entity_query)
        if not queries:
            return None

        try:
            rows = await asyncio.to_thread(api_service_info.load_service_info)
        except Exception:
            return None

        return _choose_service_info_preparation(rows, queries)

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        def _result_fallback(note: str, message: str = "Сейчас не удалось получить результаты автоматически. Соединяю с оператором.") -> dict[str, Any]:
            return _service_fallback(
                note=note,
                handoff_message=message,
                entities=entities,
                reason="test_result_fallback",
                extra={"ready": False},
            )

        fields = _extract_result_query_fields(entities, query)
        missing = [k for k in ("surname", "year", "filial", "number") if not fields.get(k)]
        if missing:
            return {
                "ready": False,
                "note": "missing_result_fields",
                "missing_fields": missing,
                "entities_used": entities,
            }

        try:
            api_resp = await asyncio.to_thread(
                api_nayka.site_result_for_patient,
                surname=fields["surname"],
                year=int(fields["year"]),
                filial=fields["filial"],
                number=int(fields["number"]),
                lang=fields["lang"],
                with_time=None,
            )
        except Exception as e:
            return _result_fallback(f"resultForPatient failed: {e}")

        if not isinstance(api_resp, dict) or not api_resp.get("ok"):
            return _result_fallback(f"resultForPatient error: {api_resp}")

        payload = api_resp.get("data")
        if not payload:
            return {
                "ready": False,
                "note": "result_not_found_or_not_ready",
                "result_payload": payload,
                "result_preview": "По указанным данным результаты пока не найдены или еще не готовы.",
                "entities_used": entities,
            }

        link = _build_public_result_link(fields)
        if not link:
            return _result_fallback(
                "result_link_build_failed",
                "Сейчас не удалось сформировать ссылку на результат автоматически. Соединяю с оператором.",
            )

        return {
            "ready": True,
            "note": "result_link_constructed",
            "result_payload": payload,
            "result_preview": "Ссылка на результат сформирована.",
            "result_links": [link],
            "entities_used": entities,
        }

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        async def _load_with_retry(fn: Any, *args: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    return await asyncio.to_thread(fn, *args)
                except Exception as exc:
                    last_exc = exc
                    if attempt == 0:
                        await asyncio.sleep(0.12)
                        continue
            if last_exc is not None:
                raise last_exc
            return []

        doctor_id = _as_int(entities.get("doctor_id"))
        doctor_name = _get_first_present(entities, ["doctor_name", "doctor", "fio", "last_name", "doctor_last_name"]) or ""
        resolved_doctor_fio: str | None = None
        if not doctor_id and doctor_name:
            doctor_id, resolved_doctor_fio = await self._resolve_doctor_id_from_name(doctor_name)
        if not doctor_id and query and _DOCTOR_PRICE_HINT_RE.search(str(query or "")):
            # Fallback для фраз вида "сколько стоит ... у Белохвостиковой":
            # извлекаем врача из полного текста запроса, даже если classifier не выделил doctor_name.
            doctor_id, q_resolved_fio = await self._resolve_doctor_id_from_name(str(query))
            if doctor_id and q_resolved_fio:
                resolved_doctor_fio = q_resolved_fio
                doctor_name = q_resolved_fio
        entity_service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        query_text = str(query or "").strip()
        doctor_price_query = bool(
            doctor_id
            and query_text
            and _DOCTOR_PRICE_HINT_RE.search(query_text)
            and _PRICE_REQUEST_RE.search(query_text)
        )
        current_service_name_for_resolution = "" if doctor_price_query else entity_service_name
        if entity_service_name and _is_city_only_reply(query_text):
            query_service_name = None
        else:
            query_service_name = resolve_price_service_name_from_catalog(
                query_text,
                current_service_name=current_service_name_for_resolution,
            ) or _extract_price_service_from_query(query_text)
        # Для явного нового price-запроса не тянем старую услугу из entities.
        if query_service_name:
            service_name = query_service_name
        elif query_text and _PRICE_REQUEST_RE.search(query_text):
            service_name = query_text
        else:
            service_name = entity_service_name or query_text
        # Если вопрос явно doctor-specific и сформулирован как новый price-запрос,
        # не тянем "залипшую" услугу из прошлого контекста.
        if doctor_price_query and not query_service_name:
            service_name = query_text
        needle = _normalise_input(service_name)

        if doctor_id:
            # doctor prices: branch-level regionId из /doctorServicePricesByRegion cache
            try:
                prices = await _load_with_retry(api_price.load_doctor_prices)
            except Exception:
                return _service_fallback(
                    note="price_info source unavailable",
                    handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"prices": []},
                )
            doc_prices = [p for p in prices if _as_int(p.get("doctorId")) == doctor_id]
            if needle:
                doc_prices = _rank_price_rows(doc_prices, service_name, limit=10)
            if not needle:
                doc_prices = sorted(
                    [p for p in doc_prices if isinstance(p, dict)],
                    key=lambda p: (_normalise_input(str(p.get("serviceName") or "")), _as_int(p.get("cost")) or 0),
                )
            return {
                "prices": doc_prices[:10],
                "note": "price_info: doctorServicePricesByRegion (branch-level regionId)",
                "entities_used": {
                    **entities,
                    "doctor_id_resolved": doctor_id,
                    "doctor_name_resolved": resolved_doctor_fio or doctor_name or "",
                    "service_name_effective": service_name,
                },
            }

        # retail prices: city-level regionId в /priceByRegion/{cityRegionId}
        try:
            price_rows = await _load_with_retry(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
        except Exception:
            return _service_fallback(
                note="price_info source unavailable",
                handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prices": []},
            )
        if not needle:
            return {"prices": [], "note": "no service query", "entities_used": entities}
        matches = _rank_price_rows([p for p in price_rows if isinstance(p, dict)], service_name, limit=10)
        return {
            "prices": matches,
            "note": f"price_info: priceByRegion({SAMARA_PRICE_REGION_ID})",
            "entities_used": {
                **entities,
                "service_name_effective": service_name,
            },
        }

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        try:
            regions = await self._ensure_regions_loaded()
        except Exception:
            regions = []
        # Работаем только по Самаре.
        regions = [
            r for r in regions
            if isinstance(r, dict)
            and (
                _is_samara_city_value(str(r.get("city") or ""))
                or "самара" in _normalise_input(str(r.get("name") or ""))
                or "самара" in _normalise_input(str(r.get("addressForSite") or ""))
            )
        ]
        appointment_mode = bool(entities.get("__appointment_mode"))
        branch = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city"]) or ""
        if not branch:
            raw_query = str(query or "").strip()
            if raw_query and (_looks_like_real_address(raw_query) or _ADDRESS_HINT_RE.search(raw_query)):
                branch = raw_query
        branch_q = _normalise_input(branch)
        service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        if not service_name:
            extracted = extract_service_phrase(query or "")
            if extracted:
                service_name = extracted
        service_q = _normalise_input(service_name)
        city_for_static = _get_first_present(entities, ["city"])
        if city_for_static and _is_non_samara_city_value(city_for_static):
            return _service_fallback(
                note=f"address_info unsupported city: {city_for_static}",
                handoff_message="Сейчас могу помочь только по Самаре. Соединяю с оператором.",
                entities=entities,
                reason="city_not_supported",
                extra={"addresses": [], "branches": []},
            )

        allowed_doctor_addresses_norm: set[str] = set()
        if appointment_mode:
            try:
                doctors = await self._ensure_doctors_cache_loaded()
            except Exception:
                doctors = []
            for d in doctors:
                if not isinstance(d, dict):
                    continue
                for addr in (d.get("regions") or d.get("addresses") or []):
                    a = str(addr).strip()
                    if not a or not _looks_like_real_address(a):
                        continue
                    allowed_doctor_addresses_norm.add(_normalise_input(a))

        if service_q and _is_procedure_branch_lookup_query(query, service_q):
            procedure_branches = await self._procedure_branches_from_index(service_q, regions)
            if procedure_branches:
                if branch_q:
                    procedure_branches = [
                        b
                        for b in procedure_branches
                        if branch_q in _normalise_input(str(b.get("address") or ""))
                    ]
                if procedure_branches:
                    return {
                        "addresses": [str(b.get("address") or "").strip() for b in procedure_branches if str(b.get("address") or "").strip()],
                        "branches": procedure_branches,
                        "note": "address_info: procedure->branches (doctor_prices index)",
                        "entities_used": entities,
                    }

        if service_q:
            regions = _filter_regions_by_service_flags(regions, service_q)

        addresses: list[str] = []
        branches: list[dict[str, Any]] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            rid = _as_int(r.get("id"))
            disp = _region_display_name(r)
            if not disp:
                continue
            # в выдачу пациенту пускаем только реальные адреса филиалов
            if not _looks_like_real_address(disp):
                continue
            hay = " | ".join(
                [
                    _normalise_input(disp),
                    _normalise_input(str(r.get("name") or "")),
                    _normalise_input(str(r.get("city") or "")),
                ]
            )
            if branch_q and branch_q not in hay:
                continue
            addresses.append(disp)
            branches.append(
                {
                    "id": rid,
                    "address": disp,
                    "city": str(r.get("city") or "").strip(),
                    "phone": _extract_region_phone(r),
                    "work_time": _extract_region_work_time(r),
                }
            )

        uniq = sorted(set(addresses))
        if uniq:
            by_addr: dict[str, dict[str, Any]] = {}
            for b in branches:
                addr = str(b.get("address") or "").strip()
                if not addr:
                    continue
                prev = by_addr.get(addr)
                if prev is None:
                    by_addr[addr] = b
                    continue
                prev_score = int(bool(prev.get("phone"))) + int(bool(prev.get("work_time")))
                cur_score = int(bool(b.get("phone"))) + int(bool(b.get("work_time")))
                if cur_score > prev_score:
                    by_addr[addr] = b

            if appointment_mode and allowed_doctor_addresses_norm:
                def _is_doctor_capable(addr: str) -> bool:
                    n = _normalise_input(addr)
                    for x in allowed_doctor_addresses_norm:
                        if n == x or n in x or x in n:
                            return True
                    return False

                filtered_addrs = [a for a in uniq if _is_doctor_capable(a)]
                if filtered_addrs:
                    uniq = filtered_addrs
                    by_addr = {k: v for k, v in by_addr.items() if _is_doctor_capable(k)}

            note = "address_info: live regions API"
            if service_q:
                note += " + filtered by service flags"
            if appointment_mode:
                note += " + filtered by doctor-capable branches"
            return {
                "addresses": uniq,
                "branches": list(by_addr.values()),
                "note": note,
                "entities_used": entities,
            }

        # fallback: старый путь через кэш врачей
        try:
            doctors = await self._ensure_doctors_cache_loaded()
        except Exception:
            doctors = []
        fallback: list[str] = []
        for d in doctors:
            for addr in (d.get("regions") or d.get("addresses") or []):
                a = str(addr).strip()
                if not a:
                    continue
                if not _looks_like_real_address(a):
                    continue
                if branch_q and branch_q not in _normalise_input(a):
                    continue
                fallback.append(a)
        return {
            "addresses": sorted(set(fallback)),
            "branches": [{"address": a, "phone": "", "work_time": ""} for a in sorted(set(fallback))],
            "note": "address_info: doctors cache fallback",
            "entities_used": entities,
        }

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        try:
            hits = await asyncio.to_thread(
                meilisearch.search_news_active,
                index_name="news",
                keyword=query or None,
                limit=10,
                sort=["from_ts:desc"],
            )
        except Exception:
            # Для новостей деградация источника не критична: возвращаем пустой ответ
            # без принудительного handoff.
            return {
                "news": [],
                "note": "news source unavailable",
                "entities_used": entities,
            }
        return {"news": hits, "entities_used": entities}

    def get_branches(self) -> list[dict[str, str]]:
        """
        Возвращает справочник филиалов.
        Формат:
          [{"id":"branch_1","name":"Филиал на Проспекте Ленина","aliases":"Ленина, Ленинская,Проспект Ленина 5"}]
        Пока заглушка.
        """
        regions = self._regions_cache or []
        if not regions:
            try:
                data = api_nayka.site_regions()
                if isinstance(data, list):
                    regions = data
                    self._regions_cache = data
                    self._regions_cache_loaded_at = time.time()
            except Exception:
                regions = []
        out: list[dict[str, str]] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            if not (
                _is_samara_city_value(str(r.get("city") or ""))
                or "самара" in _normalise_input(str(r.get("name") or ""))
                or "самара" in _normalise_input(str(r.get("addressForSite") or ""))
            ):
                continue
            rid = r.get("id")
            disp = _region_display_name(r)
            if not disp:
                continue
            if not _looks_like_real_address(disp):
                continue
            bid = f"branch_{rid}" if rid is not None else f"branch_{len(out) + 1}"
            aliases = ", ".join(
                [
                    _normalise_input(disp),
                    _normalise_input(str(r.get("name") or "")),
                    _normalise_input(str(r.get("city") or "")),
                ]
            )
            out.append({"id": bid, "name": disp, "aliases": aliases})

        if out:
            # убираем дубли по имени
            uniq_by_name: dict[str, dict[str, str]] = {}
            for b in out:
                uniq_by_name.setdefault(b["name"], b)
            return list(uniq_by_name.values())

        return []


if __name__ == "__main__":
    async def main():
        s = Services()
        print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
        print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

    asyncio.run(main())
