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
from difflib import get_close_matches
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
from .llm_doesnt_work_fallback import build_prepare_fallback_answer
from .llm_runtime import generate_text
from .prompt_registry import load_prompt_text
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
    r"^\s*(?:а\s+)?(?:сколько\s+стоит|сколько\s+будет\s+стоить|"
    r"каков(?:а|о|ы)?\s+стоимость|каков(?:а|о|ы)?\s+цена|"
    r"кака(?:я|ое|ие)\s+стоимость|кака(?:я|ое|ие)\s+цена|"
    r"цена|стоимость)\s+",
    re.I,
)
_PRICE_DOCTOR_SUFFIX_RE = re.compile(
    r"\bу\s+[а-яё\-]{3,}(?:\s+[а-яё\-]{2,}){0,2}\b.*$",
    re.I,
)
_PRICE_PREPARE_HINT_RE = re.compile(
    r"\b(подготов\w*|натощак|перед\s+(анализ\w*|исследован\w*|процедур\w*))\b",
    re.I,
)
_PRICE_CONSULT_EXCLUDE_RE = re.compile(
    r"\b(подготов\w*|узи|анализ\w*|пакет\w*|комплекс\w*|программ\w*)\b",
    re.I,
)
_LAB_SERVICE_HINT_RE = re.compile(
    r"\b("
    r"анализ\w*|лаборатор\w*|кров\w*|моч\w*|кал\w*|мазок\w*|соскоб\w*|"
    r"пцр|антител\w*|антиген\w*|гормон\w*|биохим\w*|коагул\w*|гемостаз\w*|"
    r"глюкоз\w*|холестерин\w*|липид\w*|оак|оам|бакпосев\w*|чекап\w*|панел\w*|профил\w*"
    r")\b",
    re.I,
)
_DOCTOR_SERVICE_HINT_RE = re.compile(
    r"\b("
    r"при[её]м\w*|консультац\w*|осмотр\w*|врач\w*|доктор\w*|"
    r"операц\w*|хирург\w*|узи|ультразвук\w*|эндоскоп\w*|фгдс|фкс|гастроскоп\w*|"
    r"колоноскоп\w*|рентген\w*|мрт|кт|флюорограф\w*|экг"
    r")\b",
    re.I,
)
_LAB_DEADLINE_HINT_RE = re.compile(r"\b\d+\s*(?:-\s*\d+)?\s*(?:дн|дней|нед|час)\b", re.I)
_PRICE_CITO_QUERY_RE = re.compile(r"\b(cito|сроч\w*|экспресс\w*)\b", re.I)
_PRICE_CAPILLARY_QUERY_RE = re.compile(r"\b(капилляр\w*|из\s+пальца|палец)\b", re.I)
_PRICE_CHILD_QUERY_RE = re.compile(r"\b(дет\w*|ребен\w*|ребён\w*)\b", re.I)
_PRICE_REPEAT_QUERY_RE = re.compile(r"\b(повторн\w*|повтор)\b", re.I)
_PRICE_KMN_QUERY_RE = re.compile(r"\b(к\.?\s*м\.?\s*н\.?|кандидат\w*\s+медицин\w*\s+наук)\b", re.I)
_PRICE_HOME_QUERY_RE = re.compile(r"\b(на\s+дому|домой|выезд\w*\s+на\s+дом)\b", re.I)
_PRICE_PACKAGE_QUERY_RE = re.compile(r"\b(совместно|комплекс\w*|пакет\w*|программ\w*|combo|комбо|с\s+узи)\b", re.I)
_PRICE_GENETIC_QUERY_RE = re.compile(r"\b(ген\w*|мутац\w*|полиморф\w*|генет\w*|vdr)\b", re.I)
_PRICE_CITO_ROW_RE = re.compile(r"\b(cito|сроч\w*|экспресс\w*)\b", re.I)
_PRICE_CAPILLARY_ROW_RE = re.compile(r"\bкапилляр\w*\b", re.I)
_PRICE_CHILD_ROW_RE = re.compile(r"\b(дет\w*|ребен\w*|ребён\w*)\b", re.I)
_PRICE_REPEAT_ROW_RE = re.compile(r"\b(повторн\w*|повтор)\b", re.I)
_PRICE_KMN_ROW_RE = re.compile(r"\b(к\.?\s*м\.?\s*н\.?|кандидат\w*\s+медицин\w*\s+наук)\b", re.I)
_PRICE_HOME_ROW_RE = re.compile(r"\b(на\s+дому|домой|выезд\w*\s+на\s+дом)\b", re.I)
_PRICE_PACKAGE_ROW_RE = re.compile(r"\b(совместно|комплекс\w*|пакет\w*|программ\w*|combo|комбо|регулярн\w*)\b", re.I)
_PRICE_GENETIC_ROW_RE = re.compile(r"\b(ген\w*|мутац\w*|полиморф\w*|генет\w*|vdr)\b", re.I)
_PRICE_SHOW_ALL_RE = re.compile(
    r"^\s*(?:все|всё|покажи\s+все|показать\s+все|все\s+варианты|все\s+услуги|все\s+анализы)\s*[!.,?]*\s*$",
    re.I,
)
_PRICE_COMPOUND_LAB_FRAGMENT_RE = re.compile(
    r"\b(?:сдать\s+кровь\s+на|кровь\s+на|анализ(?:ы)?\s+на|сдать\s+анализ(?:ы)?\s+на)\s+([a-zа-яё0-9\-/ ]{2,80})",
    re.I,
)
_PRICE_DIAGNOSTIC_NO_DOCTOR_RE = re.compile(
    r"\b(экг|флюорограф\w*|маммограф\w*|рентген\w*|мрт|кт)\b",
    re.I,
)
_PRICE_PROCEDURE_LIKE_RE = re.compile(
    r"\b("
    r"при[её]м\w*|консультац\w*|осмотр\w*|узи|ультразвук\w*|эндоскоп\w*|фгдс|фкс|гастроскоп\w*|"
    r"колоноскоп\w*|рентген\w*|мрт|кт|флюорограф\w*|маммограф\w*|экг|операц\w*|удалени\w*|"
    r"массаж\w*|пломб\w*|зуб\w*|подтяжк\w*|хирург\w*|травматолог\w*|ортопед\w*|стоматолог\w*|"
    r"анестези\w*|имплант\w*|протез\w*|сустав\w*|лечени\w*|профилактик\w*|"
    r"биопс\w*|резекц\w*|склерозир\w*|препарат\w*"
    r")\b",
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
_CATALOG_DOCTOR_STOPWORDS = {
    "запишите",
    "записать",
    "записаться",
    "расписание",
    "расписание",
    "врач",
    "доктор",
    "специалист",
    "прием",
    "приеме",
    "приём",
    "приёме",
    "да",
    "нет",
    "пожалуйста",
    "будьте",
    "добры",
}
_CATALOG_SERVICE_LEADIN_RE = re.compile(
    r"^\s*(?:(?:пожалуйста|будьте\s+добры|подскажите|скажите|мне)\s+)?"
    r"(?:(?:запишите|записать|записаться|можно|хочу|нужно|надо)\s+)?"
    r"(?:(?:на|к)\s+)?",
    re.I,
)
_CATALOG_SERVICE_TRAILING_TIME_RE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{1,2}:\d{2})\b.*$",
    re.I,
)
_CATALOG_WORD_RE = re.compile(r"[a-zа-яё0-9\-]+", re.I)
_CATALOG_SERVICE_SIGNAL_RE = re.compile(
    r"\b(услуг\w*|процедур\w*|исследован\w*|анализ\w*|сда[тч]\w*|"
    r"узи|экг|холтер|мрт|кт|фгдс|фкс|эндоскоп\w*|гастроскоп\w*|кольпоскоп\w*|"
    r"колоноскоп\w*|рентген\w*|флюорограф\w*|биопс\w*|пункц\w*|"
    r"при[её]м\w*|консультац\w*|операц\w*|липид\w*|холестерин\w*|оак|оам)\b",
    re.I,
)
_CATALOG_SERVICE_STOPWORDS = _SERVICE_FILTER_STOPWORDS | {
    "мне",
    "бы",
    "пож",
    "пожалуйста",
    "будьте",
    "добры",
    "здравствуйте",
    "добрый",
    "день",
    "запишите",
    "записаться",
    "запись",
    "к",
    "на",
    "в",
    "во",
    "по",
    "из",
    "могу",
    "можете",
    "покажите",
    "подскажите",
    "скажите",
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


def _runtime_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    try:
        value = float(getattr(c, name))
    except Exception:
        value = float(default)
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
    "какой",
    "какая",
    "какое",
    "какие",
    "какую",
    "какого",
    "какому",
    "каким",
    "каких",
    "каков",
    "какова",
    "каково",
    "каковы",
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
    "ли",
    "здравствуйте",
    "добрый",
    "день",
    "данный",
    "данную",
    "данных",
    "пакет",
    "рассчитывается",
    "выходным",
    "выходные",
    "выходной",
    "будни",
    "будний",
    "дешевле",
    "будет",
    "чем",
    "это",
    "возможно",
    "сейчас",
    "сегодня",
    "подскажите",
    "скажите",
    "пожалуйста",
}
_PRICE_QUERY_CANONICAL_TOKENS = {
    "алт": "алат",
    "алат": "алат",
    "alat": "алат",
    "ast": "асат",
    "аст": "асат",
    "асат": "асат",
    "asat": "асат",
    "общего": "общий",
    "общем": "общий",
    "общую": "общий",
    "общая": "общий",
    "общей": "общий",
    "анализа": "анализ",
    "анализу": "анализ",
    "анализом": "анализ",
    "анализе": "анализ",
    "витамина": "витамин",
    "витамину": "витамин",
    "витамине": "витамин",
    "лпвп": "лпвп",
    "лпнп": "лпнп",
    "лпонп": "лпнп",
}
_PRICE_SHORT_TOKEN_WHITELIST = {
    "оак",
    "оам",
    "алт",
    "аст",
    "т3",
    "т4",
    "пса",
    "лпнп",
    "лпвп",
    "срб",
    "мно",
    "пцр",
    "вич",
    "rw",
    "рв",
}
_PRICE_GENERIC_SERVICE_TOKENS = {
    "анализ",
    "анализы",
    "анализов",
    "прием",
    "приемы",
    "консультация",
    "консультации",
    "осмотр",
    "услуга",
    "услуги",
    "процедура",
    "процедуры",
    "пакет",
    "комплекс",
}
_PRICE_GENERIC_FAMILY_ROOT_TOKENS = {
    "удален",
    "подтяжк",
    "операц",
    "пластик",
}
_PRICE_QUERY_SERVICE_NOISE_TOKENS = {
    "г",
    "город",
    "адрес",
    "филиал",
    "обследование",
    "осбледование",
    "нам",
    "там",
    "тут",
}
_PRICE_SERVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "оак": ("общий анализ крови",),
    "общий анализ крови": ("общий анализ крови",),
    "оам": ("общий анализ мочи",),
    "общий анализ мочи": ("общий анализ мочи",),
    "липидограмма": ("липидограмма", "липидный профиль", "кровь на холестерин"),
    "липидный профиль": ("липидограмма", "липидный профиль", "кровь на холестерин"),
    "холестерин": ("кровь на холестерин", "липидограмма", "липидный профиль"),
    "лпнп": ("лпнп", "липопротеиды низкой плотности"),
    "лпвп": ("лпвп", "липопротеиды высокой плотности"),
    "копрология": ("копрологическое исследование", "копрология"),
}
_VITAMIN_CODE_MAP = {
    "а": "a",
    "a": "a",
    "в": "b",
    "b": "b",
    "с": "c",
    "c": "c",
    "д": "d",
    "d": "d",
    "е": "e",
    "e": "e",
    "к": "k",
    "k": "k",
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
    "правила",
    "памятка",
    "инструкция",
    "условия",
    "сдать",
    "сдавать",
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


def _schedule_payload_matches_doctor(data: Any, doctor_name: str) -> bool:
    """
    Проверяет, что payload расписания действительно относится к нужному врачу.

    Защищает от ответов API, где по фамилии может вернуться "общий" список
    других врачей (ложный позитив на первом непустом list).

    :param data: ответ find_doctor_schedule
    :param doctor_name: ожидаемая фамилия/ФИО
    :return: True, если в payload есть совпадающий врач
    """

    target = str(doctor_name or "").strip()
    if not target or not isinstance(data, list):
        return False
    for row in data:
        if not isinstance(row, dict):
            continue
        row_fio = str(row.get("fio") or "").strip()
        if not row_fio:
            continue
        if _doctor_matches_fio(row_fio, target, resolved_surname=target):
            return True
    return False
_PREPARE_LEADIN_RE = re.compile(
    r"^\s*(?:(?:здравствуй(?:те)?|добрый\s+день)[, ]+)?"
    r"(?:(?:подскажите|скажите)[, ]+)?"
    r"(?:(?:пожалуйста)[, ]+)?"
    r"(?:(?:как|каким\s+образом|каковы?)\s+)?"
    r"(?:(?:правила|памятка|инструкц(?:ия|ии)|условия)\s+)?"
    r"(?:подготов(?:иться|ится|ка|ки)|готовиться|сдать|сдавать)\s*"
    r"(?:(?:к|для|перед|по)\s+)?",
    re.I,
)
_PREPARE_ENTITY_RE = re.compile(
    r"(?:"
    r"(?:правила|памятка|инструкц(?:ия|ии)|условия)\s+подготов(?:ки|ка)?\s*(?:к|для|перед|по)?\s+|"
    r"подготов(?:иться|ится|ка|ки)\s*(?:к|для|перед|по)?\s+|"
    r"готовиться\s*(?:к|для|перед|по)?\s+|"
    r"(?:как\s+)?(?:сдать|сдавать)\s+"
    r")(?P<entity>.+)$",
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


def _normalise_catalog_text(s: str) -> str:
    norm = _normalise_input(s).replace("ё", "е")
    norm = re.sub(r"[^a-zа-я0-9\- ]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _dedupe_str(items: list[str], *, max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw or "").strip()
        if not value:
            continue
        key = _normalise_catalog_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _doctor_catalog_query_candidates(raw_text_or_name: str) -> list[str]:
    raw = str(raw_text_or_name or "").strip()
    if not raw:
        return []
    candidate = extract_doctor_name_candidate(raw, prefer_schedule=True)
    probes = _dedupe_str([candidate or "", raw], max_items=4)
    out: list[str] = []
    for probe in probes:
        norm = _normalise_catalog_text(probe)
        if not norm:
            continue
        tokens = [
            token
            for token in _CATALOG_WORD_RE.findall(norm)
            if len(token) >= 3 and token not in _CATALOG_DOCTOR_STOPWORDS
        ]
        if not tokens:
            continue
        out.append(tokens[0])
        if len(tokens) > 1:
            out.append(" ".join(tokens[:2]))
    return _dedupe_str(out, max_items=6)


def _service_catalog_query_candidates(raw_text_or_name: str, *, current_service_name: str = "") -> list[str]:
    raw = str(raw_text_or_name or "").strip()
    if not raw and not current_service_name:
        return []
    service_phrase = extract_service_phrase(raw) if raw else None
    stripped = _CATALOG_SERVICE_LEADIN_RE.sub("", raw).strip()
    stripped = _CATALOG_SERVICE_TRAILING_TIME_RE.sub("", stripped).strip(" ,.;:-")
    base_candidates = [service_phrase or "", stripped, raw, str(current_service_name or "")]
    candidates = _dedupe_str(base_candidates, max_items=8)

    out: list[str] = []
    for item in candidates:
        norm = _normalise_catalog_text(item)
        if not norm:
            continue
        has_service_signal = bool(service_phrase and item == service_phrase) or bool(_CATALOG_SERVICE_SIGNAL_RE.search(item))
        tokens = [token for token in _CATALOG_WORD_RE.findall(norm) if len(token) >= 2]
        core_tokens = [token for token in tokens if token not in _CATALOG_SERVICE_STOPWORDS]
        if not has_service_signal and not core_tokens:
            continue
        if core_tokens:
            out.append(" ".join(core_tokens[:8]))
        if has_service_signal:
            out.append(item)
    return _dedupe_str(out, max_items=8)


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
    stripped = re.sub(r"^(?:мне\s+)?(?:нужно|надо|хочу)\s+", "", stripped, count=1, flags=re.I).strip(" ?!.,")
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


_PREPARE_SERVICE_INFO_SYNONYMS: dict[str, tuple[str, ...]] = {}
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
_PREPARE_RELEVANCE_VALIDATOR_FALLBACK_PROMPT = (
    "Ты валидатор релевантности ответа по подготовке к анализу/процедуре.\n"
    "Проверь, соответствует ли КАНДИДАТ запросу пациента.\n"
    "Требования:\n"
    "1) Используй только смысл запроса и кандидата.\n"
    "2) Если тема не совпадает или ответ слишком общий — IRRELEVANT.\n"
    "3) Верни строго JSON без markdown.\n"
    "Формат JSON:\n"
    "{\"verdict\":\"RELEVANT|IRRELEVANT\",\"confidence\":0.0,\"reason\":\"кратко\"}\n\n"
    "ЗАПРОС:\n<<USER_QUERY>>\n\n"
    "ИСТОЧНИК:\n<<SOURCE_KIND>>\n\n"
    "СЕРВИС:\n<<SERVICE_TITLE>>\n\n"
    "КАНДИДАТ:\n<<CANDIDATE_TEXT>>\n"
)
_PREPARE_RELEVANCE_VERDICT_RELEVANT = "RELEVANT"
_PREPARE_RELEVANCE_VERDICT_IRRELEVANT = "IRRELEVANT"


@dataclass
class _PrepareCandidate:
    source: str
    text: str
    query_variant: str
    service_title: str = ""
    score: float = 0.0
    margin: float = 0.0
    note: str = ""


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
    has_empty_stomach_phrase = re.search(r"голод\w*\s+желуд", norm) is not None
    out: set[str] = set()
    for token in tokens:
        if token in _PREPARE_SERVICE_INFO_GENERIC_TOKENS:
            continue
        # Формы "подготов..." не несут предметного смысла и размывают match.
        if token.startswith("подготов"):
            continue
        # "сдают/сдать/сдача" — служебные слова для формулировки вопроса.
        if token.startswith("сда"):
            continue
        # Фразу "на голодный желудок" приводим к каноничному "натощак".
        if has_empty_stomach_phrase and (token.startswith("голод") or token.startswith("желуд")):
            continue
        out.add(token)
    if has_empty_stomach_phrase:
        out.add("натощак")
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Извлекает первый JSON-объект из произвольного текстового ответа модели.

    :param text: raw-ответ LLM
    :return: dict или None
    """

    s = str(text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start : idx + 1]
                try:
                    obj = json.loads(chunk)
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _prepare_term_roots(text: str) -> set[str]:
    """
    Возвращает корни смысловых токенов для query/content matching.

    :param text: исходный текст
    :return: множество токенов-корней
    """

    roots: set[str] = set()
    for token in _prepare_service_info_core_tokens(text):
        stem = _stem_service_token(token)
        root = str(stem or token).strip()
        if len(root) >= 3:
            roots.add(root)
    return roots


def _prepare_roots_match(query_root: str, candidate_roots: set[str]) -> bool:
    if query_root in candidate_roots:
        return True
    query_root = str(query_root or "").strip()
    if len(query_root) < 5:
        return False
    for cand in candidate_roots:
        cand = str(cand or "").strip()
        if len(cand) < 5:
            continue
        if query_root.startswith(cand) or cand.startswith(query_root):
            return True
        if len(query_root) >= 7 and len(cand) >= 7 and query_root[:6] == cand[:6]:
            return True
    return False


def _prepare_roots_coverage(query_roots: set[str], candidate_roots: set[str]) -> float:
    """
    Покрытие корней запроса в кандидате (0..1).

    :param query_roots: корни запроса
    :param candidate_roots: корни кандидата
    :return: доля покрытых корней
    """

    if not query_roots or not candidate_roots:
        return 0.0
    matched = 0
    for q in query_roots:
        if _prepare_roots_match(q, candidate_roots):
            matched += 1
    return float(matched) / float(len(query_roots))


def _prepare_fast_relevance_score(query: str, content: str, *, title: str = "") -> float:
    """
    Быстрый score релевантности без доменных словарей.

    :param query: запрос пациента
    :param content: текст подготовки
    :param title: заголовок услуги/сервиса (если есть)
    :return: score 0..1
    """

    query_roots = _prepare_term_roots(query)
    if not query_roots:
        query_norm = _normalise_input(query).replace("ё", "е")
        content_norm = _normalise_input(content).replace("ё", "е")
        generic_prepare_query = any(x in query_norm for x in ("подготов", "анализ", "исслед", "натощак"))
        if generic_prepare_query and _is_prepare_content_actionable(content):
            # Generic query без таргета: разрешаем умеренный score для fallback по main_index.
            return 0.40 if content_norm else 0.0
        return 0.0

    content_roots = _prepare_term_roots(content)
    title_roots = _prepare_term_roots(title)
    if not content_roots and not title_roots:
        return 0.0

    body_cov = _prepare_roots_coverage(query_roots, content_roots)
    title_cov = _prepare_roots_coverage(query_roots, title_roots)
    actionable = 1.0 if _is_prepare_content_actionable(content) else 0.0

    score = 0.62 * body_cov + 0.28 * title_cov + 0.10 * actionable
    if actionable < 1.0:
        score -= 0.08

    query_norm = _normalise_input(query).replace("ё", "е")
    content_norm = _normalise_input(content).replace("ё", "е")
    title_norm = _normalise_input(title).replace("ё", "е")
    if query_norm and len(query_norm) >= 6:
        if query_norm in content_norm:
            score += 0.05
        if title_norm and query_norm in title_norm:
            score += 0.08

    return max(0.0, min(1.0, score))


def _prepare_relevance_thresholds() -> tuple[float, float, float]:
    """
    Возвращает пороги релевантности для fast gate.

    :return: (low_threshold, high_threshold, margin_threshold)
    """

    # Quality-first defaults: шире серая зона, чтобы чаще подключать LLM-валидатор.
    low = _runtime_float("MR_PREPARE_RELEVANCE_LOW_THRESHOLD", 0.28, min_value=0.05, max_value=0.95)
    high = _runtime_float("MR_PREPARE_RELEVANCE_HIGH_THRESHOLD", 0.78, min_value=0.10, max_value=0.99)
    margin = _runtime_float("MR_PREPARE_RELEVANCE_MARGIN_THRESHOLD", 0.18, min_value=0.01, max_value=0.60)
    if low >= high:
        low = max(0.05, high - 0.10)
    return low, high, margin


def _prepare_relevance_gate(score: float, margin: float) -> str:
    """
    Решает fast gate для кандидата подготовки.

    :param score: fast relevance score
    :param margin: разрыв между top1 и top2
    :return: "accept" | "llm" | "reject"
    """

    low, high, margin_threshold = _prepare_relevance_thresholds()
    value = max(0.0, min(1.0, float(score)))
    gap = max(0.0, float(margin))
    if value < low:
        return "reject"
    if value >= high and gap >= margin_threshold:
        return "accept"
    return "llm"


def _prepare_relevance_prompt(
    query: str,
    candidate_text: str,
    *,
    source_kind: str,
    service_title: str = "",
) -> str:
    """
    Формирует prompt для LLM-валидации релевантности prepare-кандидата.

    :param query: исходный запрос пациента
    :param candidate_text: кандидатный текст ответа
    :param source_kind: источник кандидата (serviceInfoAll/main_index)
    :param service_title: serviceName для API-кандидата
    :return: prompt string
    """

    try:
        tmpl = load_prompt_text("prepare_relevance_validator")
    except Exception:
        tmpl = _PREPARE_RELEVANCE_VALIDATOR_FALLBACK_PROMPT
    return (
        str(tmpl or "")
        .replace("<<USER_QUERY>>", str(query or "").strip())
        .replace("<<SOURCE_KIND>>", str(source_kind or "").strip())
        .replace("<<SERVICE_TITLE>>", str(service_title or "").strip())
        .replace("<<CANDIDATE_TEXT>>", str(candidate_text or "").strip())
        .strip()
    )


def _parse_prepare_relevance_validator(raw: str) -> tuple[bool, float, str]:
    """
    Парсит JSON-ответ LLM-валидатора релевантности.

    :param raw: raw-ответ LLM
    :return: (is_relevant, confidence, reason)
    """

    obj = _extract_json_object(raw)
    if not isinstance(obj, dict):
        return False, 0.0, "llm_non_json"

    verdict = str(obj.get("verdict") or "").strip().upper()
    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(obj.get("reason") or "").strip()
    return verdict == _PREPARE_RELEVANCE_VERDICT_RELEVANT, confidence, reason


def _dedupe_prepare_candidates(candidates: list[_PrepareCandidate], *, limit: int = 12) -> list[_PrepareCandidate]:
    """
    Удаляет дубли prepare-кандидатов по тексту и сортирует по score.

    :param candidates: список кандидатов
    :param limit: максимальное число кандидатов
    :return: отсортированный дедуплицированный список
    """

    ranked = sorted(
        candidates,
        key=lambda item: (item.score, len(str(item.text or ""))),
        reverse=True,
    )
    out: list[_PrepareCandidate] = []
    seen: set[str] = set()
    for item in ranked:
        key = _normalise_prepare_text(str(item.text or ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(1, limit):
            break
    return out


def _service_info_row_score(queries: list[str], row: dict[str, Any]) -> tuple[float, str]:
    """
    Считает релевантность строки serviceInfoAll для prepare-запроса.

    :param queries: подготовленные варианты запроса
    :param row: строка из serviceInfoAll
    :return: (score, лучшая query-вариация)
    """

    service_name = _normalise_input(str(row.get("serviceName") or "")).replace("ё", "е")
    preparation = str(row.get("preparation") or "").strip()
    if not service_name or not preparation:
        return 0.0, ""

    best = 0.0
    best_query = ""
    for query in queries:
        query_norm = _normalise_prepare_text(query)
        if not query_norm:
            continue
        score = _prepare_fast_relevance_score(query_norm, preparation, title=service_name)
        if score > best:
            best = score
            best_query = query_norm

    return best, best_query


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

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score, _ = _service_info_row_score(queries, row)
        if score <= 0.0:
            continue
        preparation_len = len(str(row.get("preparation") or "").strip())
        ranked.append((score, preparation_len, row))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = ranked[0][2]
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
    """
    Извлекает каноническую специальность из пользовательского текста.

    Сначала пытается найти составную специальность через role-синонимы
    (`травматолог ортопед`, `уролог андролог` и т.п.), затем использует
    legacy-regex fallback.

    :param text: исходный текст пользователя
    :return: каноническая специальность или пустая строка
    """

    if _UZI_QUERY_RE.search(text or ""):
        return "узи"
    if _ENDOSCOPY_SERVICE_RE.search(text or ""):
        return "эндоскопист"
    for specialty in sorted(_SPECIALTY_CANONICAL, key=len, reverse=True):
        if _matches_specialty_terms(text, specialty):
            return specialty
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


def _specialty_norm(value: str) -> str:
    return _normalise_input(value).replace("ё", "е")


def _specialty_equivalent(left: str, right: str) -> bool:
    """
    Проверяет эквивалентность двух обозначений специальности.

    Пример: "лор" ~= "оториноларинголог".

    :param left: первая специальность
    :param right: вторая специальность
    :return: True, если обозначения эквивалентны
    """

    left_norm = _specialty_norm(left)
    right_norm = _specialty_norm(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_terms = {left_norm, *_specialty_terms(left_norm)}
    right_terms = {right_norm, *_specialty_terms(right_norm)}
    return bool(left_terms & right_terms)


def _extract_specialties_from_text(text: str) -> tuple[str, ...]:
    """
    Извлекает все распознанные специальности из произвольного текста.

    :param text: исходный текст
    :return: кортеж нормализованных специальностей
    """

    norm = _specialty_norm(text)
    if not norm:
        return tuple()
    found: list[str] = []
    for raw in _SPECIALTY_CANONICAL:
        spec = _specialty_norm(raw)
        if spec and _matches_specialty_terms(norm, spec) and spec not in found:
            found.append(spec)
    return tuple(found)


def _is_direct_specialty_text_match(text: str, specialty: str) -> bool:
    """
    Строго проверяет соответствие текста конкретной специальности.

    Важно для прямых запросов по врачу:
    - "терапевт" не должен матчиться на "гирудотерапевт";
    - гибриды вида "кардиолог-ревматолог" не должны попадать в чистый запрос
      "кардиолог".

    :param text: текст для проверки (serviceName/unit_name)
    :param specialty: целевая специальность
    :return: True для чистого соответствия специальности
    """

    target = _specialty_norm(specialty)
    if not target:
        return False
    found = _extract_specialties_from_text(text)
    if not found:
        return False
    if not any(_specialty_equivalent(spec, target) for spec in found):
        return False
    for spec in found:
        if not _specialty_equivalent(spec, target):
            return False
    return True


def _doctor_matches_primary_specialty(doc: dict[str, Any], specialty: str) -> bool:
    """
    Проверяет, что врач относится к специальности именно по primary/main профилю.

    :param doc: карточка врача
    :param specialty: целевая специальность
    :return: True, если есть релевантный main-unit для этой специальности
    """

    main_units = _collect_role_unit_names(doc, main_value=True)
    if main_units:
        return any(_is_direct_specialty_text_match(unit_name, specialty) for unit_name in main_units)
    # Legacy fallback: если main-структуры нет, используем любой unit-level match.
    return _doctor_role_specialty_match_level(doc, specialty) >= 1


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

    if resolved_surname:
        target = _normalise_input(resolved_surname)
        return bool(target and tokens and _normalise_input(tokens[0]) == target)

    candidates: list[str] = []
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


def _specialty_label_for_doctor(doc: dict[str, Any], *, preferred_specialty: str = "") -> str:
    """
    Возвращает короткую человекочитаемую метку основной специальности врача.

    :param doc: карточка врача
    :param preferred_specialty: специальность из запроса, если она есть
    :return: короткая метка специальности для patient-facing ответа
    """

    preferred = _normalise_input(preferred_specialty).replace("ё", "е")
    if preferred and _doctor_matches_primary_specialty(doc, preferred):
        return preferred_specialty.strip().capitalize()

    for unit_name in _collect_role_unit_names(doc, main_value=True) + _collect_role_unit_names(doc, main_value=False):
        found = _extract_specialties_from_text(unit_name)
        if found:
            return found[0].capitalize()

    display_spec = _pick_display_specialization(doc, preferred_specialty=preferred_specialty)
    for candidate in _split_spec_lines(display_spec):
        found = _extract_specialties_from_text(candidate)
        if found:
            return found[0].capitalize()

    return ""


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
            # В branch payload подмешиваем только реальные адреса филиалов.
            # Иначе generic live-region вроде "Самара" может перехватить точный
            # адрес из priceUnits и испортить patient-facing рендер.
            if not _looks_like_real_address(disp):
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


def _normalise_price_token(token: str) -> str:
    """
    Приводит price-токен к каноническому виду для устойчивого ранжирования.

    :param token: исходный токен из запроса или строки прайса
    :return: нормализованный токен
    """

    norm = str(token or "").strip().lower().replace("ё", "е")
    if not norm:
        return ""
    if norm.startswith("ультразвук"):
        return "узи"
    return _PRICE_QUERY_CANONICAL_TOKENS.get(norm, norm)


def _extract_vitamin_designator(text: str) -> str:
    """
    Извлекает буквенно-цифровой код витамина из запроса или строки прайса.

    Примеры:
    - `витамин Д` -> `d`
    - `витамин B12` -> `b12`

    :param text: исходный текст
    :return: канонический код витамина или пустая строка
    """

    raw = _normalise_input(text).replace("ё", "е")
    m = re.search(r"\bвитамин\w*\s+([a-zа-я]\d{0,2})\b", raw, re.I)
    if not m:
        return ""
    token = str(m.group(1) or "").strip().lower()
    if not token:
        return ""
    head = _VITAMIN_CODE_MAP.get(token[0], token[0])
    return f"{head}{token[1:]}"


def _augment_price_tokens(tokens: list[str], *, raw_text: str = "") -> list[str]:
    """
    Добавляет безопасные alias-токены к price-выражению.

    Сейчас нужен в первую очередь для кейса `УЗИ` -> `ультразвуковое исследование`,
    чтобы взрослая строка не проигрывала детской только из-за буквального матча.

    :param tokens: уже выделенные и нормализованные токены
    :param raw_text: исходный текст, из которого токены были получены
    :return: список токенов с alias-дополнениями без дублей
    """

    out: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        norm = _normalise_price_token(value)
        if not norm or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for token in tokens:
        _push(token)

    raw = _normalise_input(raw_text).replace("ё", "е")
    if raw and _UZI_LINE_RE.search(raw):
        _push("узи")
    vitamin_code = _extract_vitamin_designator(raw)
    if vitamin_code:
        _push(f"витамин_{vitamin_code}")
    return out


def _price_query_tokens(text: str) -> list[str]:
    s = _normalise_input(text).replace("ё", "е")
    out: list[str] = []
    for t in _PRICE_TOKEN_RE.findall(s):
        token = _normalise_price_token(str(t or ""))
        if len(token) < 2:
            continue
        if len(token) < 3 and token not in _PRICE_SHORT_TOKEN_WHITELIST and not token.isdigit():
            continue
        if token in _PRICE_QUERY_STOPWORDS:
            continue
        out.append(token)
        # "прием" и "консультация" считаем взаимозаменяемыми для ранжирования цен.
        if token.startswith("прием") and "консультац" not in out:
            out.append("консультац")
        elif token.startswith("консультац") and "прием" not in out:
            out.append("прием")
    return _augment_price_tokens(out, raw_text=s)


def _price_alias_candidates(query_text: str) -> list[str]:
    raw = _normalise_input(str(query_text or "")).replace("ё", "е")
    if not raw:
        return []
    out: list[str] = []
    direct = raw.strip(" ?!.,;:")
    if direct:
        out.append(direct)
    extracted = _extract_price_service_from_query(raw)
    if extracted:
        out.append(_normalise_input(extracted).replace("ё", "е"))
    compact = " ".join(_price_query_tokens(raw)).strip()
    if compact:
        out.append(compact)
    dedup: list[str] = []
    seen: set[str] = set()
    for cand in out:
        key = str(cand or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return dedup[:5]


def _resolve_price_alias_from_catalog(query_text: str, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    for candidate in _price_alias_candidates(query_text):
        variants = _PRICE_SERVICE_ALIASES.get(candidate, ())
        if not variants:
            continue
        for variant in variants:
            ranked = _rank_price_rows(rows, variant, limit=1)
            if not ranked:
                continue
            resolved = str(ranked[0].get("serviceName") or ranked[0].get("name") or "").strip()
            if resolved:
                return resolved
    return None


def _extract_price_service_from_query(query: str) -> str | None:
    raw = str(query or "").strip()
    if not raw:
        return None
    if _PRICE_CONSULT_HINT_RE.search(raw):
        specialty = _extract_specialty_from_text(raw)
        if specialty:
            return f"прием {specialty}"
        return None
    q = _normalise_input(raw)
    q = _PRICE_DOCTOR_SUFFIX_RE.sub("", q).strip(" ?!.,;:")
    q = _PRICE_SERVICE_PREFIX_RE.sub("", q).strip(" ?!.,;:")
    if not q:
        return None
    if q in {"цена", "стоимость"}:
        return None
    words = [w for w in _PRICE_TOKEN_RE.findall(q) if w]
    if not words:
        return None
    filtered_words = [w for w in words if w not in _PRICE_QUERY_STOPWORDS]
    if filtered_words:
        words = filtered_words
    significant_words = [w for w in words if w not in _PRICE_GENERIC_SERVICE_TOKENS]
    if not significant_words:
        return None
    if len(significant_words) == 1 and significant_words[0].startswith("анализ"):
        return None
    if len(words) >= 6 and not any(_LAB_SERVICE_HINT_RE.search(w) or _DOCTOR_SERVICE_HINT_RE.search(w) for w in words):
        return None
    # Ограничиваем длину candidate, чтобы не тянуть в ranking целый диалог.
    return " ".join(words[:8])


def _is_generic_uzi_price_request(query_text: str) -> bool:
    """
    Определяет, что пользователь спрашивает цену только по общему термину УЗИ.

    В таком сценарии нельзя надежно выбирать первую попавшуюся услугу из каталога,
    потому что в прайсе десятки видов УЗИ. Нужен дополнительный clarify.

    :param query_text: исходный текст запроса пользователя
    :return: True, если требуется уточнение конкретного вида УЗИ
    """

    raw = str(query_text or "").strip()
    if not raw or not _PRICE_REQUEST_RE.search(raw):
        return False

    extracted = str(_extract_price_service_from_query(raw) or "").strip()
    norm = _normalise_input(extracted).replace("ё", "е")
    return norm in {"узи", "ультразвук", "ультразвуковое исследование"}


def _is_consultation_service_query(value: str) -> bool:
    """
    Проверяет, что service_name относится к приему/консультации врача.

    :param value: строка услуги
    :return: True для консультационных услуг
    """

    norm = _normalise_input(str(value or ""))
    return bool(norm and _PRICE_CONSULT_HINT_RE.search(norm))


def _detect_service_kind(
    service_name: str,
    *,
    query_text: str = "",
    top_retail: dict[str, Any] | None = None,
    is_consult_query: bool = False,
) -> str:
    """
    Определяет тип услуги для PRICE-bundle:
    - `lab` для лабораторных анализов;
    - `doctor` для врачебных услуг/процедур/диагностики.

    :param service_name: итоговое название услуги
    :param query_text: исходный запрос пользователя
    :param top_retail: верхняя retail-строка (если есть)
    :param is_consult_query: заранее вычисленный признак консультации
    :return: `lab` | `doctor`
    """

    norm_name = _normalise_input(str(service_name or "")).replace("ё", "е")
    norm_query = _normalise_input(str(query_text or "")).replace("ё", "е")

    if is_consult_query:
        return "doctor"

    if norm_name and _DOCTOR_SERVICE_HINT_RE.search(norm_name):
        return "doctor"

    if norm_name and _LAB_SERVICE_HINT_RE.search(norm_name):
        return "lab"

    if isinstance(top_retail, dict):
        homecode = _normalise_input(
            str(top_retail.get("serviceHomecode") or top_retail.get("homecode") or "")
        )
        deadline = _normalise_input(str(top_retail.get("deadline") or ""))
        if homecode.isdigit() and len(homecode) >= 4 and not _DOCTOR_SERVICE_HINT_RE.search(norm_name):
            return "lab"
        if deadline and _LAB_DEADLINE_HINT_RE.search(deadline) and not _DOCTOR_SERVICE_HINT_RE.search(norm_name):
            return "lab"

    if norm_query and _LAB_SERVICE_HINT_RE.search(norm_query) and not _DOCTOR_SERVICE_HINT_RE.search(norm_query):
        return "lab"

    return "doctor"


def _is_clean_consultation_row_name(value: str) -> bool:
    """
    Проверяет, что строка прайса похожа именно на услугу приема/консультации,
    а не на пакет/подготовку с вкраплением слова "прием".

    :param value: имя услуги из прайса
    :return: True для чистого консультационного тарифа
    """

    norm = _normalise_input(str(value or ""))
    if not norm or not _PRICE_CONSULT_HINT_RE.search(norm):
        return False
    return _PRICE_CONSULT_EXCLUDE_RE.search(norm) is None


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


def _should_prefer_current_price_query_over_context(query_text: str, current_service_name: str) -> bool:
    """
    Определяет, что новый `PRICE`-запрос содержит собственную услугу и должен
    иметь приоритет над услугой из прошлого контекста.

    Правило нужно против stale-context кейсов вида:
    1. PREPARE по одному анализу,
    2. затем новый явный вопрос о цене по другому анализу.

    В таких репликах старый `current_service_name` полезен только как fallback,
    но не должен доминировать над текущим текстом пользователя.

    :param query_text: текущая реплика пользователя
    :param current_service_name: услуга из state предыдущего шага
    :return: True, если текущий текст должен иметь приоритет над контекстом
    """

    raw = str(query_text or "").strip()
    current = str(current_service_name or "").strip()
    if not raw or not current:
        return False
    if not _PRICE_REQUEST_RE.search(raw):
        return False
    if _is_city_only_reply(raw):
        return False

    current_norm = _normalise_input(current).replace("ё", "е")
    current_tokens = set(_price_query_tokens(current))
    query_variants: list[str] = []

    extracted = _extract_price_service_from_query(raw)
    if extracted:
        query_variants.append(extracted)

    phrase = extract_service_phrase(raw)
    if phrase:
        query_variants.append(phrase)

    compact = " ".join(_price_query_tokens(raw)).strip()
    if compact:
        query_variants.append(compact)

    for candidate in query_variants:
        cand_norm = _normalise_input(candidate).replace("ё", "е")
        if not cand_norm or cand_norm == current_norm:
            continue
        cand_tokens = set(_price_query_tokens(candidate))
        if len(cand_tokens) >= 2 and not cand_tokens.issubset(current_tokens):
            return True
        if (
            len(cand_norm.split()) >= 2
            and cand_norm not in current_norm
            and current_norm not in cand_norm
        ):
            return True

    return False


def _service_name_matches_specialty(service_name: str, specialty: str) -> bool:
    """
    Проверяет, что имя услуги относится к нужной специальности.

    :param service_name: строка услуги из каталога/контекста
    :param specialty: каноническая специальность
    :return: True, если в названии услуги есть термин специальности
    """

    return _is_direct_specialty_text_match(service_name, specialty)


def _is_prepare_requested_in_price_query(query_text: str) -> bool:
    """
    Проверяет, просит ли пользователь именно подготовку в PRICE-реплике.

    :param query_text: текст запроса пользователя
    :return: True, если запрошены правила подготовки
    """

    return bool(_PRICE_PREPARE_HINT_RE.search(str(query_text or "")))


def _is_strong_doctor_price_match(
    *,
    query_norm: str,
    query_tokens: list[str],
    row_name_norm: str,
    matched_tokens: int,
    target_homecode: str,
    row_homecode: str,
) -> bool:
    """
    Решает, достаточно ли сильное соответствие doctor_price-строки услуге.

    Для процедурных запросов блокирует «случайные» совпадения по одному слову
    (например, `желудка`), из-за которых в хирургии всплывают УЗИ-врачи.

    :param query_norm: нормализованная целевая услуга
    :param query_tokens: токены целевой услуги
    :param row_name_norm: нормализованное имя строки doctor_price
    :param matched_tokens: число совпавших токенов из score-функции
    :param target_homecode: homecode целевой услуги из retail
    :param row_homecode: homecode строки doctor_price
    :return: True, если строка релевантна целевой услуге
    """

    if target_homecode and row_homecode and target_homecode == row_homecode:
        return True
    if not query_norm or not row_name_norm:
        return False
    if query_norm == row_name_norm or query_norm in row_name_norm:
        return True
    if len(row_name_norm) >= 12 and row_name_norm in query_norm:
        return True

    token_count = len([t for t in query_tokens if t])
    if token_count <= 1:
        return matched_tokens >= 1
    if token_count == 2:
        return matched_tokens >= 2
    return matched_tokens >= max(2, token_count - 1)


def _has_specific_price_tokens(text: str) -> bool:
    tokens = _price_query_tokens(text)
    if not tokens:
        return False
    return any(tok not in _PRICE_GENERIC_SERVICE_TOKENS for tok in tokens)


def _build_price_catalog_queries(query_text: str, *, current_service_name: str = "") -> list[str]:
    """
    Собирает варианты запроса для поиска услуги в price-каталоге.

    :param query_text: исходный текст пользователя
    :param current_service_name: уже известная услуга из state
    :return: список поисковых вариантов от самых полезных к запасным
    """

    raw = str(query_text or "").strip()
    queries: list[str] = []
    consult_specialty = ""
    if raw and _PRICE_CONSULT_HINT_RE.search(raw):
        consult_specialty = _extract_specialty_from_text(raw)

    # При запросах "стоимость приема <специальность>" не даем stale-контексту
    # другой специальности доминировать над текущим запросом.
    if current_service_name and (
        not consult_specialty or _service_name_matches_specialty(current_service_name, consult_specialty)
    ):
        queries.append(current_service_name)
    extracted = _extract_price_service_from_query(raw)
    if extracted:
        queries.append(extracted)
    phrase = extract_service_phrase(raw)
    if phrase:
        queries.append(phrase)
    compact = " ".join(_price_query_tokens(raw)).strip()
    if compact and _has_specific_price_tokens(compact):
        queries.append(compact)
    if raw and _has_specific_price_tokens(raw):
        queries.append(raw)
    return _dedupe_price_queries(queries)


def _resolve_best_price_row_from_queries(
    queries: list[str],
    catalog_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int, int]:
    """
    Находит лучшую строку price-каталога по набору query-вариантов.

    :param queries: список вариантов пользовательского запроса
    :param catalog_rows: строки priceByRegion
    :return:
        - лучшая строка каталога либо None,
        - её итоговый score,
        - количество совпавших токенов
    """

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
    return best_row, best_score, best_matched


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

    alias_hit = _resolve_price_alias_from_catalog(query_text, catalog_rows)
    if alias_hit:
        return alias_hit

    prefer_query_over_context = _should_prefer_current_price_query_over_context(query_text, current_service_name)
    if prefer_query_over_context:
        text_only_queries = _build_price_catalog_queries(query_text, current_service_name="")
        if text_only_queries:
            best_row, best_score, _ = _resolve_best_price_row_from_queries(text_only_queries, catalog_rows)
            if best_row and best_score >= 100:
                return str(best_row.get("serviceName") or best_row.get("name") or "").strip() or None

    queries = _build_price_catalog_queries(query_text, current_service_name=current_service_name)
    if not queries:
        return None

    best_row, best_score, _ = _resolve_best_price_row_from_queries(queries, catalog_rows)

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


def _query_nonbase_price_flags(query_text: str) -> set[str]:
    """
    Выделяет из запроса модификаторы, которые пациент запросил явно.

    :param query_text: исходный текст пользователя
    :return: набор флагов модификаторов
    """

    query = _normalise_input(str(query_text or ""))
    flags = _query_price_variant_flags(query_text)
    if _PRICE_REPEAT_QUERY_RE.search(query):
        flags.add("repeat")
    if _PRICE_KMN_QUERY_RE.search(query):
        flags.add("kmn")
    if _PRICE_HOME_QUERY_RE.search(query):
        flags.add("home")
    if _PRICE_PACKAGE_QUERY_RE.search(query):
        flags.add("package")
    if _PRICE_GENETIC_QUERY_RE.search(query):
        flags.add("genetic")
    return flags


def _row_nonbase_price_flags(row: dict[str, Any]) -> set[str]:
    """
    Выделяет модификаторы из конкретной строки прайса.

    :param row: строка прайса
    :return: набор флагов модификаторов
    """

    name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
    flags = _lab_price_variant_flags(row)
    if _PRICE_REPEAT_ROW_RE.search(name):
        flags.add("repeat")
    if _PRICE_KMN_ROW_RE.search(name):
        flags.add("kmn")
    if _PRICE_HOME_ROW_RE.search(name):
        flags.add("home")
    if _PRICE_PACKAGE_ROW_RE.search(name):
        flags.add("package")
    if _PRICE_GENETIC_ROW_RE.search(name):
        flags.add("genetic")
    return flags


def _price_row_modifier_penalty(row: dict[str, Any], query_text: str) -> int:
    """
    Считает штраф для специальных модификаторов, которые пользователь не просил.

    Нужен, чтобы generic-запросы не выбирали детские, Cito, повторные,
    `к.м.н.` и пакетные строки раньше базовых тарифов.

    :param row: строка прайса
    :param query_text: исходный пользовательский запрос
    :return: отрицательный штраф или 0
    """

    requested = _query_nonbase_price_flags(query_text)
    row_flags = _row_nonbase_price_flags(row)
    if not row_flags:
        return 0

    penalties = {
        "child": -100,
        "cito": -35,
        "capillary": -30,
        "repeat": -28,
        "kmn": -40,
        "home": -45,
        "package": -40,
        "genetic": -120,
    }
    penalty = 0
    for flag in row_flags:
        if flag in requested:
            continue
        penalty += penalties.get(flag, 0)
    return penalty


def _score_price_rows(
    rows: list[dict[str, Any]],
    query_text: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Возвращает отсортированные candidate-строки прайса вместе с debug score.

    :param rows: строки прайса
    :param query_text: текст пользовательского запроса
    :param limit: максимум уникальных строк на выходе
    :return: список словарей вида `{"row": ..., "score": ..., "matched": ...}`
    """

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
        if not _is_price_match_strong(
            score=score,
            matched=matched,
            query=query,
            tokens=tokens,
            homecode_query=homecode_query,
        ):
            continue
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        name_gap = abs(len(name) - len(query)) if query else len(name)
        cost = _as_int(row.get("cost")) or 0
        scored.append((score, matched, -name_gap, -cost, row))

    if not scored:
        fallback_rows: list[dict[str, Any]] = []
        if query:
            fallback_rows = [
                r for r in rows if isinstance(r, dict) and query in _normalise_input(str(r.get("serviceName") or ""))
            ]
        elif homecode_query:
            fallback_rows = [
                r
                for r in rows
                if isinstance(r, dict)
                and homecode_query in _normalise_input(str(r.get("serviceHomecode") or r.get("homecode") or ""))
            ]
        return [
            {"row": row, "score": 0, "matched": 0}
            for row in fallback_rows[:limit]
        ]

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for score, matched, _, _, row in scored:
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        code = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
        key = (name, code)
        if key in seen:
            continue
        seen.add(key)
        out.append({"row": row, "score": score, "matched": matched})
        if len(out) >= limit:
            break
    return out


def _price_row_score(row: dict[str, Any], *, query: str, tokens: list[str], homecode_query: str) -> tuple[int, int]:
    name = _normalise_input(str(row.get("serviceName") or row.get("name") or "")).replace("ё", "е")
    homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
    if not name:
        return 0, 0
    row_tokens = _augment_price_tokens(
        [_normalise_price_token(tok) for tok in _PRICE_TOKEN_RE.findall(name) if tok],
        raw_text=name,
    )
    row_tokens_set = set(row_tokens)

    # Жесткий фильтр для консультационных price-запросов по специальности:
    # "стоимость приема уролога" не должен матчиться на фониатра/терапевта.
    query_specialty = _extract_specialty_from_text(query)
    if _PRICE_CONSULT_HINT_RE.search(query):
        if not _is_clean_consultation_row_name(name):
            return 0, 0
        if query_specialty and not _service_name_matches_specialty(name, query_specialty):
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
            if tok in row_tokens_set:
                matched += 1
                continue
            if "_" in tok:
                continue
            # Для длинных токенов допускаем умеренно мягкий префиксный матч.
            if len(tok) >= 5 and any(
                len(rt) >= 4 and (rt.startswith(tok[:4]) or tok.startswith(rt[:4]))
                for rt in row_tokens
            ):
                matched += 1
        score += matched * 25
        if matched == len(tokens):
            score += 80
        elif matched >= max(2, len(tokens) - 1):
            score += 40

    # Слегка понижаем заведомо нерелевантный общий тариф.
    if "выезд на дом" in name and not any(tok in name for tok in tokens):
        score -= 30
    if "узи" in tokens and "ультразвук" in name and not _PRICE_CHILD_QUERY_RE.search(query):
        score += 160
    score += _price_row_modifier_penalty(row, query)

    return score, matched


def _is_price_match_strong(*, score: int, matched: int, query: str, tokens: list[str], homecode_query: str) -> bool:
    if homecode_query and score >= 180:
        return True
    token_count = len([t for t in tokens if t])
    if token_count == 0:
        return bool(query and score >= 170)
    if token_count == 1:
        tok = tokens[0]
        if len(tok) <= 3:
            return matched >= 1 and score >= 220
        return matched >= 1 and score >= 100
    if token_count == 2:
        return matched >= 2 or score >= 220
    return matched >= max(2, token_count - 1) or score >= 260


def _rank_price_rows(rows: list[dict[str, Any]], query_text: str, *, limit: int = 10) -> list[dict[str, Any]]:
    return [
        item["row"]
        for item in _score_price_rows(rows, query_text, limit=limit)
        if isinstance(item, dict) and isinstance(item.get("row"), dict)
    ]


def _lab_price_variant_flags(row: dict[str, Any]) -> set[str]:
    """
    Выделяет специальные модификаторы лабораторной строки прайса.

    Нужен для patient-facing выдачи, чтобы без явного запроса не подмешивать
    срочные, капиллярные и детские варианты в базовый список цен.

    :param row: строка прайса
    :return: набор флагов варианта (`cito`, `capillary`, `child`)
    """

    name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
    homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
    flags: set[str] = set()
    if _PRICE_CITO_ROW_RE.search(name):
        flags.add("cito")
    if _PRICE_CAPILLARY_ROW_RE.search(name) or homecode.endswith("к"):
        flags.add("capillary")
    if _PRICE_CHILD_ROW_RE.search(name):
        flags.add("child")
    return flags


def _query_price_variant_flags(query_text: str) -> set[str]:
    """
    Извлекает из запроса признаки явно запрошенного модификатора анализа.

    :param query_text: исходный запрос пользователя
    :return: набор флагов (`cito`, `capillary`, `child`)
    """

    query = _normalise_input(str(query_text or ""))
    flags: set[str] = set()
    if _PRICE_CITO_QUERY_RE.search(query):
        flags.add("cito")
    if _PRICE_CAPILLARY_QUERY_RE.search(query):
        flags.add("capillary")
    if _PRICE_CHILD_QUERY_RE.search(query):
        flags.add("child")
    return flags


def _is_lab_price_query_for_catalog(query_text: str) -> bool:
    """
    Определяет, что ценовой запрос относится к лабораторным анализам.

    :param query_text: текст пользовательского запроса
    :return: True для лабораторного price-запроса
    """

    query = _normalise_input(str(query_text or ""))
    if not query:
        return False
    return bool(_LAB_SERVICE_HINT_RE.search(query) and not _DOCTOR_SERVICE_HINT_RE.search(query))


def _select_patient_price_rows(
    rows: list[dict[str, Any]],
    query_text: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Возвращает patient-facing список цен с мягкой развилкой для лабораторных tie-case.

    Если у лабораторного запроса несколько базовых услуг с одинаковой высокой
    релевантностью, показываем их все, но без специальных модификаторов
    (`Cito`, капиллярная кровь, детские версии), если они не были явно
    запрошены пользователем.

    :param rows: строки прайса
    :param query_text: текст, по которому ранжируем выдачу
    :param limit: максимальное количество строк
    :return: список строк для ответа пациенту
    """

    ranked = _rank_price_rows(rows, query_text, limit=max(limit, 20))
    if not ranked:
        return []
    if not _is_lab_price_query_for_catalog(query_text):
        return ranked[:limit]

    requested_flags = _query_price_variant_flags(query_text)
    filtered_ranked: list[dict[str, Any]] = []
    for row in ranked:
        row_flags = _lab_price_variant_flags(row)
        if row_flags and not row_flags.issubset(requested_flags):
            continue
        filtered_ranked.append(row)
    if filtered_ranked:
        ranked = filtered_ranked

    query = _normalise_input(query_text).replace("ё", "е")
    tokens = _price_query_tokens(query_text)
    homecode_query = _extract_homecode_query(query_text)
    top_score, top_matched = _price_row_score(
        ranked[0],
        query=query,
        tokens=tokens,
        homecode_query=homecode_query,
    )
    tied_rows: list[dict[str, Any]] = []
    for row in ranked:
        score, matched = _price_row_score(
            row,
            query=query,
            tokens=tokens,
            homecode_query=homecode_query,
        )
        if score != top_score or matched != top_matched:
            break
        tied_rows.append(row)

    if len(tied_rows) < 2:
        return ranked[:limit]

    filtered: list[dict[str, Any]] = []
    for row in tied_rows:
        row_flags = _lab_price_variant_flags(row)
        if row_flags and not row_flags.issubset(requested_flags):
            continue
        filtered.append(row)

    if filtered:
        return filtered[:limit]
    return tied_rows[:limit]


def _normalise_family_variant_name(value: str) -> str:
    """
    Приводит название услуги к "базовому" виду для family-кластеризации.

    :param value: имя услуги из прайса
    :return: нормализованное имя без модификаторов
    """

    norm = _normalise_input(str(value or "")).replace("ё", "е")
    norm = _PRICE_CITO_ROW_RE.sub(" ", norm)
    norm = _PRICE_CAPILLARY_ROW_RE.sub(" ", norm)
    norm = _PRICE_CHILD_ROW_RE.sub(" ", norm)
    norm = _PRICE_REPEAT_ROW_RE.sub(" ", norm)
    norm = _PRICE_KMN_ROW_RE.sub(" ", norm)
    norm = _PRICE_HOME_ROW_RE.sub(" ", norm)
    norm = _PRICE_PACKAGE_ROW_RE.sub(" ", norm)
    norm = re.sub(r"[\[\]()]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _expand_family_rows_by_root_token(
    rows: list[dict[str, Any]],
    *,
    family_query: str,
    root_token: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Расширяет семейную выдачу по одному корневому токену запроса.

    Используется для generic family-query, когда обычный ranker видит только
    один широкий комплекс, но в каталоге есть целое семейство строк с тем же
    корнем (`витамин*`, `гепатит*`, `зуб*`).

    :param rows: полный список строк прайса
    :param family_query: нормализованный family-query пользователя
    :param root_token: значимый корневой токен запроса
    :param limit: максимальный размер расширенной выдачи
    :return: список строк прайса, сгруппированных вокруг root-token
    """

    token = _normalise_price_token(root_token)
    if len(token) < 4:
        return []
    needle = token[:4]
    candidate_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_name = str(row.get("serviceName") or row.get("name") or "").strip()
        if not raw_name:
            continue
        row_tokens = _augment_price_tokens(
            [_normalise_price_token(tok) for tok in _PRICE_TOKEN_RE.findall(raw_name)],
            raw_text=raw_name,
        )
        if not any(
            len(rt) >= 4 and (rt.startswith(needle) or needle.startswith(rt[:4]))
            for rt in row_tokens
        ):
            continue
        dedup_key = (
            _normalise_input(raw_name),
            str(row.get("serviceHomecode") or row.get("homecode") or "").strip(),
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        candidate_rows.append(row)

    if len(candidate_rows) < 2:
        return []

    ranked = _rank_price_rows(candidate_rows, token, limit=max(limit, 20))
    if not ranked:
        return []

    if _is_lab_price_query_for_catalog(family_query):
        requested_flags = _query_price_variant_flags(family_query)
        filtered_ranked = [
            row
            for row in ranked
            if not _lab_price_variant_flags(row) or _lab_price_variant_flags(row).issubset(requested_flags)
        ]
        if filtered_ranked:
            ranked = filtered_ranked

    return ranked[:limit]


def _family_query_root_tokens(query_text: str) -> list[str]:
    """
    Выделяет из price-запроса информативные токены для family-кластеризации.

    В отличие от ручного списка корней, функция опирается на уже очищенные
    токены запроса и подходит для новых семейств услуг без пополнения regex.

    :param query_text: исходный пользовательский запрос
    :return: список значимых токенов, отсортированных по длине
    """

    family_query = str(_extract_price_service_from_query(query_text) or query_text).strip()
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _price_query_tokens(family_query):
        norm = _normalise_price_token(token)
        if (
            not norm
            or norm in _PRICE_GENERIC_SERVICE_TOKENS
            or len(norm) < 4
            or norm.isdigit()
            or "_" in norm
        ):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        tokens.append(norm)
    if len(tokens) > 1:
        narrowed = [
            tok
            for tok in tokens
            if not any(tok.startswith(stem) for stem in _PRICE_GENERIC_FAMILY_ROOT_TOKENS)
        ]
        if narrowed:
            tokens = narrowed
    tokens.sort(key=len, reverse=True)
    return tokens


def _dedupe_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Удаляет дубли строк прайса по названию и homecode.

    :param rows: список строк прайса
    :return: список уникальных строк в исходном порядке
    """

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            _normalise_input(str(row.get("serviceName") or row.get("name") or "")),
            _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _select_family_variant_rows(
    rows: list[dict[str, Any]],
    family_query: str,
    *,
    ranking_query: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Отбирает patient-facing строки для family-режима без жесткой привязки к одному кейсу.

    Для лабораторных запросов сохраняет уже существующую фильтрацию модификаторов,
    а для остальных услуг использует обычный price-ranking.

    :param rows: candidate-строки прайса
    :param family_query: очищенный family-query пользователя
    :param ranking_query: текст, по которому ранжируем family-кандидаты
    :param limit: максимум строк на выходе
    :return: список вариантов для family-выдачи
    """

    unique_rows = _dedupe_price_rows(rows)
    if not unique_rows:
        return []
    effective_query = str(ranking_query or family_query).strip() or family_query
    if _is_lab_price_query_for_catalog(family_query):
        ranked = _rank_price_rows(unique_rows, effective_query, limit=max(limit, 20))
        requested_flags = _query_price_variant_flags(family_query)
        filtered_ranked: list[dict[str, Any]] = []
        for row in ranked:
            row_flags = _lab_price_variant_flags(row)
            if row_flags and not row_flags.issubset(requested_flags):
                continue
            filtered_ranked.append(row)
        if filtered_ranked:
            ranked = filtered_ranked
        return ranked[:limit]
    ranked = _rank_price_rows(unique_rows, effective_query, limit=max(limit, 20))
    family_norm = _normalise_input(family_query).replace("ё", "е")
    if _is_uzi_query_text(family_query) and _UZI_PROCEDURE_HINT_RE.search(family_norm):
        filtered_ranked = [
            row
            for row in ranked
            if not _is_lab_like_service_name(str(row.get("serviceName") or row.get("name") or ""))
        ]
        if filtered_ranked:
            ranked = filtered_ranked
    return ranked[:limit]


def _family_variant_base_names(rows: list[dict[str, Any]]) -> set[str]:
    """
    Выделяет множество нормализованных "базовых" названий family-вариантов.

    :param rows: candidate-строки family-выдачи
    :return: множество нормализованных названий без служебных модификаторов
    """

    base_names = {
        _normalise_family_variant_name(str(row.get("serviceName") or row.get("name") or ""))
        for row in rows
        if isinstance(row, dict)
    }
    base_names.discard("")
    return base_names


def _build_family_candidate_rows(
    query_text: str,
    rows: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Собирает family-кандидаты по структуре каталога, а не по списку корневых слов.

    Логика сначала берет близкие top-scored строки, а затем при необходимости
    расширяет кластер по информативным токенам запроса и выбирает набор с
    наибольшим разнообразием базовых вариантов.

    :param query_text: исходный пользовательский запрос
    :param rows: полный список строк прайса
    :param limit: максимум family-вариантов на выходе
    :return: candidate-строки family-кластера
    """

    family_query = str(_extract_price_service_from_query(query_text) or query_text).strip()
    if not family_query:
        return []

    scored = _score_price_rows(rows, family_query, limit=40)
    if not scored:
        return []

    top_score = int(scored[0].get("score") or 0)
    top_matched = int(scored[0].get("matched") or 0)
    direct_rows: list[dict[str, Any]] = []
    for item in scored:
        score = int(item.get("score") or 0)
        matched = int(item.get("matched") or 0)
        if score < max(0, top_score - 25):
            break
        if matched < max(1, top_matched - 1):
            continue
        row = item.get("row")
        if isinstance(row, dict):
            direct_rows.append(row)

    best_rows = _select_family_variant_rows(direct_rows, family_query, limit=limit)
    best_base_names = _family_variant_base_names(best_rows)

    for token in _family_query_root_tokens(query_text):
        expanded_rows = _expand_family_rows_by_root_token(
            rows,
            family_query=family_query,
            root_token=token,
            limit=max(limit, 40),
        )
        if len(expanded_rows) < 2:
            continue
        candidate_rows = _select_family_variant_rows(
            direct_rows + expanded_rows,
            family_query,
            ranking_query=token,
            limit=limit,
        )
        candidate_base_names = _family_variant_base_names(candidate_rows)
        if len(candidate_base_names) > len(best_base_names) or (
            len(candidate_base_names) == len(best_base_names) and len(candidate_rows) > len(best_rows)
        ):
            best_rows = candidate_rows
            best_base_names = candidate_base_names

    return best_rows


def _is_family_query_candidate(query_text: str, rows: list[dict[str, Any]]) -> bool:
    """
    Определяет, нужен ли для price-запроса режим выдачи семейства вариантов.

    :param query_text: исходный запрос пользователя
    :param rows: строки прайса
    :return: True, если лучше показать набор вариантов вместо single best match
    """

    if not query_text or not rows:
        return False
    if _is_generic_uzi_price_request(query_text):
        return False
    if _is_consultation_service_query(query_text):
        return False
    if _extract_vitamin_designator(query_text):
        return False
    if _query_nonbase_price_flags(query_text):
        return False

    family_query = str(_extract_price_service_from_query(query_text) or query_text).strip()
    tokens = [tok for tok in _price_query_tokens(family_query) if tok not in _PRICE_GENERIC_SERVICE_TOKENS]
    if not tokens or len(tokens) > 4:
        return False

    candidate_rows = _build_family_candidate_rows(query_text, rows, limit=20)
    if len(candidate_rows) < 2:
        return False

    base_names = _family_variant_base_names(candidate_rows)
    return len(base_names) >= 3


def _build_price_family_payload(
    query_text: str,
    rows: list[dict[str, Any]],
    *,
    show_all: bool = False,
    visible_limit: int = 10,
) -> dict[str, Any] | None:
    """
    Формирует payload для family-query режима price-поиска.

    :param query_text: исходный price-запрос
    :param rows: строки прайса
    :param show_all: нужно ли показать все найденные варианты
    :param visible_limit: лимит строк в первом ответе
    :return: payload family-query либо None
    """

    family_query = str(_extract_price_service_from_query(query_text) or query_text).strip()
    if not _is_family_query_candidate(query_text, rows):
        return None

    ranked = _build_family_candidate_rows(query_text, rows, limit=50)
    if len(ranked) < 2:
        return None

    ranked = _annotate_price_rows_with_care_context(ranked)

    service_name = family_query
    visible_count = len(ranked) if show_all else min(len(ranked), visible_limit)
    remaining_count = max(0, len(ranked) - visible_count)
    show_all_hint = ""
    if remaining_count > 0 and not show_all:
        show_all_hint = (
            f"По вашему запросу найдено еще {remaining_count} вариантов. "
            'Чтобы показать их, напишите: "все".'
        )

    return {
        "service_name": service_name,
        "service_kind": "family_query",
        "family_variants": ranked,
        "showing_all": show_all,
        "visible_limit": visible_limit,
        "remaining_count": remaining_count,
        "show_all_hint": show_all_hint,
        "note": "price_family_query",
    }


def _price_family_payload_from_context(entities: dict[str, Any], *, show_all: bool = False) -> dict[str, Any] | None:
    """
    Восстанавливает family-query payload из сохраненного сессионного контекста.

    :param entities: текущие сущности/state
    :param show_all: нужно ли отдать все варианты
    :return: восстановленный payload либо None
    """

    raw_ctx = entities.get("_price_family_context")
    if not isinstance(raw_ctx, dict):
        return None
    variants = raw_ctx.get("family_variants")
    if not isinstance(variants, list) or not variants:
        return None
    service_name = str(raw_ctx.get("service_name") or "").strip()
    visible_limit = int(raw_ctx.get("visible_limit") or 10)
    visible_count = len(variants) if show_all else min(len(variants), visible_limit)
    remaining_count = max(0, len(variants) - visible_count)
    show_all_hint = ""
    if remaining_count > 0 and not show_all:
        show_all_hint = (
            f"По вашему запросу найдено еще {remaining_count} вариантов. "
            'Чтобы показать их, напишите: "все".'
        )
    return {
        "service_name": service_name,
        "service_kind": "family_query",
        "family_variants": variants,
        "showing_all": show_all,
        "visible_limit": visible_limit,
        "remaining_count": remaining_count,
        "show_all_hint": show_all_hint,
        "note": "price_family_context",
    }


def _annotate_price_rows_with_care_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Обогащает строки прайса контекстом оказания услуги из справочника priceUnits.

    :param rows: найденные строки прайса
    :return: копии строк с полями care-setting/address, если контекст удалось определить
    """
    if not rows:
        return []
    try:
        units_index = api_price.build_price_units_index(api_price.load_price_units())
    except Exception:
        return [row for row in rows if isinstance(row, dict)]

    enriched: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        annotated = dict(row)
        annotated.update(
            api_price.resolve_price_unit_context(
                row.get("priceUnitId"),
                units_index=units_index,
            )
        )
        enriched.append(annotated)
    return enriched


def _care_setting_addresses_from_price_rows(rows: list[dict[str, Any]]) -> list[str]:
    """
    Извлекает уникальные адреса care-setting из уже обогащенных строк прайса.

    :param rows: строки прайса с полем care_setting_address
    :return: список уникальных адресов в порядке появления
    """
    addresses: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        address = str(row.get("care_setting_address") or "").strip()
        if address and address not in addresses:
            addresses.append(address)
    return addresses


def _select_address_price_rows(
    rows: list[dict[str, Any]],
    query_text: str,
    *,
    limit: int = 10,
    family_limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Подбирает строки прайса для адресного ответа по услуге.

    Сначала берет обычные top-match строки, затем при необходимости расширяет
    выборку family-кандидатами. Расширение применяется только если добавляет
    новые care-setting адреса, а не просто раздувает список однотипных строк.

    :param rows: строки прайса региона
    :param query_text: текст услуги / исходный запрос пользователя
    :param limit: лимит обычного top-match отбора
    :param family_limit: лимит family-расширения
    :return: обогащенные строки прайса с care-setting полями
    """

    ranked_rows = _select_patient_price_rows(rows, query_text, limit=limit)
    if not ranked_rows:
        return []

    ranked_annotated = _annotate_price_rows_with_care_context(ranked_rows)
    ranked_addresses = _care_setting_addresses_from_price_rows(ranked_annotated)

    family_rows = _build_family_candidate_rows(query_text, rows, limit=family_limit)
    if len(family_rows) < 2:
        return ranked_annotated

    merged_rows = _dedupe_price_rows(ranked_rows + family_rows)
    merged_annotated = _annotate_price_rows_with_care_context(merged_rows)
    merged_addresses = _care_setting_addresses_from_price_rows(merged_annotated)
    if len(merged_addresses) > len(ranked_addresses):
        return merged_annotated
    return ranked_annotated


def _is_price_show_all_request(query_text: str) -> bool:
    """
    Проверяет короткий follow-up пациента с просьбой показать все варианты.

    :param query_text: текст реплики
    :return: True для `все` / `покажи все` / `все варианты`
    """

    return bool(_PRICE_SHOW_ALL_RE.fullmatch(str(query_text or "").strip()))


def _is_lab_like_service_name(value: str) -> bool:
    """
    Определяет, похожа ли строка прайса на лабораторный анализ.

    :param value: название услуги
    :return: True для lab-like строки
    """

    norm = _normalise_input(str(value or "")).replace("ё", "е")
    if not norm:
        return False
    if _PRICE_PROCEDURE_LIKE_RE.search(norm):
        return False
    return True


def _classify_catalog_service_kind(
    service_name: str,
    *,
    query_text: str,
    retail_rows: list[dict[str, Any]],
    has_exact_doctor_link: bool,
    is_consult_query: bool = False,
) -> str:
    """
    Классифицирует тип услуги по matched catalog rows, а не только по regex запроса.

    :param service_name: эффективное имя услуги
    :param query_text: исходный пользовательский запрос
    :param retail_rows: релевантные retail-строки
    :param has_exact_doctor_link: найден ли надежный exact-link в doctor_prices
    :param is_consult_query: является ли запрос консультационным
    :return: `lab`, `doctor_consult`, `procedure_with_doctor`, `diagnostic_no_doctor` или `ambiguous`
    """

    if is_consult_query:
        return "doctor_consult"

    top_name = str(
        (retail_rows[0].get("serviceName") or retail_rows[0].get("name") or service_name)
        if retail_rows else service_name
    ).strip()
    top_norm = _normalise_input(top_name).replace("ё", "е")
    service_norm = _normalise_input(service_name).replace("ё", "е")
    query_norm = _normalise_input(query_text).replace("ё", "е")
    top_rows = retail_rows[:3] if retail_rows else []
    lab_signal = bool(
        _LAB_SERVICE_HINT_RE.search(service_norm)
        or _LAB_SERVICE_HINT_RE.search(query_norm)
        or any(
            _LAB_SERVICE_HINT_RE.search(_normalise_input(str(row.get("serviceName") or row.get("name") or "")))
            or str(row.get("deadline") or "").strip()
            for row in top_rows
            if isinstance(row, dict)
        )
    )

    if _PRICE_DIAGNOSTIC_NO_DOCTOR_RE.search(top_norm) or _PRICE_DIAGNOSTIC_NO_DOCTOR_RE.search(service_norm):
        return "diagnostic_no_doctor"

    if has_exact_doctor_link:
        return "procedure_with_doctor"

    if top_rows and all(
        _is_lab_like_service_name(str(row.get("serviceName") or row.get("name") or ""))
        for row in top_rows
    ) and lab_signal:
        return "lab"

    if _is_lab_like_service_name(service_name) and not _PRICE_PROCEDURE_LIKE_RE.search(query_norm) and lab_signal:
        return "lab"

    if _PRICE_PROCEDURE_LIKE_RE.search(top_norm) or _PRICE_PROCEDURE_LIKE_RE.search(service_norm):
        return "procedure_with_doctor" if has_exact_doctor_link else "ambiguous"

    return "ambiguous"


def _has_reliable_doctor_service_link(
    matched_rows: list[tuple[int, int, int, int, dict[str, Any]]],
    query_norm: str,
) -> bool:
    """
    Проверяет, что doctor-price linkage достаточно надежный для показа врачей.

    Exact homecode остаётся самым сильным сигналом, но для старых кэшей иногда
    нет homecode на retail-строке. Тогда допускаем показ врачей только если
    строки doctor_prices почти буквально совпадают с целевой услугой.

    :param matched_rows: уже отфильтрованные matched doctor rows
    :param query_norm: нормализованное имя целевой услуги
    :return: True, если linkage можно считать надежным
    """

    if not matched_rows or not query_norm:
        return False
    strong_hits = 0
    for _, matched, _, _, row in matched_rows[:3]:
        row_name = _normalise_input(str(row.get("serviceName") or row.get("name") or "")).replace("ё", "е")
        if not row_name:
            continue
        if query_norm == row_name or query_norm in row_name or row_name in query_norm:
            strong_hits += 1
            continue
        if matched >= max(2, len(_price_query_tokens(query_norm)) - 1):
            strong_hits += 1
    return strong_hits >= 1


def _build_price_kind_ambiguous_prompt(
    query_text: str,
    retail_rows: list[dict[str, Any]],
    *,
    has_exact_doctor_link: bool,
) -> str:
    """
    Собирает prompt для LLM fallback по ambiguous PRICE-кейсам.

    :param query_text: исходный пользовательский запрос
    :param retail_rows: top retail rows
    :param has_exact_doctor_link: найден ли надежный doctor linkage
    :return: готовый prompt
    """

    tmpl = load_prompt_text("price_kind_ambiguous")
    sample_rows = []
    for row in retail_rows[:5]:
        if not isinstance(row, dict):
            continue
        sample_rows.append(
            {
                "serviceName": str(row.get("serviceName") or row.get("name") or "").strip(),
                "serviceHomecode": str(row.get("serviceHomecode") or row.get("homecode") or "").strip(),
                "deadline": str(row.get("deadline") or "").strip(),
                "cost": _as_int(row.get("cost")),
            }
        )
    return (
        tmpl.replace("<<USER_QUERY>>", str(query_text or "").strip())
        .replace("<<TOP_ROWS>>", json.dumps(sample_rows, ensure_ascii=False))
        .replace("<<HAS_EXACT_DOCTOR_LINK>>", json.dumps(bool(has_exact_doctor_link), ensure_ascii=False))
    ).strip()


def _parse_price_kind_ambiguous_result(raw: str) -> str | None:
    """
    Разбирает ответ LLM fallback для ambiguous PRICE-классификации.

    :param raw: сырой текст модели
    :return: допустимый kind либо None
    """

    try:
        data = json.loads(str(raw or "").strip())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    kind = str(data.get("kind") or "").strip()
    if kind in {
        "lab",
        "doctor_consult",
        "procedure_with_doctor",
        "diagnostic_no_doctor",
        "family_query",
        "operator",
    }:
        return kind
    return None


async def _resolve_ambiguous_price_kind_with_llm(
    query_text: str,
    retail_rows: list[dict[str, Any]],
    *,
    has_exact_doctor_link: bool,
    runtime_llm_mode: str = "",
) -> str:
    """
    Запускает LLM fallback только для ambiguous PRICE-кейсов.

    :param query_text: исходный пользовательский запрос
    :param retail_rows: top retail rows
    :param has_exact_doctor_link: найден ли надежный doctor linkage
    :param runtime_llm_mode: текущий llm_mode (`strict|hybrid|rich`)
    :return: выбранный kind либо `ambiguous`
    """

    mode = str(runtime_llm_mode or "").strip().lower()
    if mode not in {"hybrid", "rich"}:
        return "ambiguous"
    prompt = _build_price_kind_ambiguous_prompt(
        query_text,
        retail_rows,
        has_exact_doctor_link=has_exact_doctor_link,
    )
    if not prompt:
        return "ambiguous"
    try:
        raw = await generate_text(
            prompt,
            timeout_s=20,
            queue_timeout_ms=4000,
            fmt="json",
            think=False,
        )
    except Exception:
        return "ambiguous"
    kind = _parse_price_kind_ambiguous_result(raw)
    if not kind:
        return "ambiguous"
    if kind == "procedure_with_doctor" and not has_exact_doctor_link:
        return "ambiguous"
    return kind

def _should_prefer_retail_query_candidate(query_candidate: str, service_name: str) -> bool:
    """
    Решает, когда для retail-поиска лучше взять текст из текущего запроса,
    а не уже резолвленную услугу.

    Это нужно для лабораторных случаев, где catalog-grounding может приземлить
    запрос в специальный вариант (`Cito`, капиллярная кровь), а пациент спросил
    про базовый анализ без уточняющих модификаторов.

    :param query_candidate: очищенная услуга из текущего запроса
    :param service_name: каноническая услуга после grounding
    :return: True, если для retail-ranking полезнее текущий текст запроса
    """

    query_candidate_norm = _normalise_input(str(query_candidate or ""))
    service_name_norm = _normalise_input(str(service_name or ""))
    if not query_candidate_norm:
        return False
    if not service_name_norm:
        return True
    if query_candidate_norm == service_name_norm:
        return False
    if query_candidate_norm in service_name_norm and len(query_candidate_norm) < len(service_name_norm):
        return True

    candidate_tokens = [tok for tok in _price_query_tokens(query_candidate_norm) if tok not in _PRICE_GENERIC_SERVICE_TOKENS]
    service_tokens = [tok for tok in _price_query_tokens(service_name_norm) if tok not in _PRICE_GENERIC_SERVICE_TOKENS]
    if candidate_tokens and len(service_tokens) > len(candidate_tokens):
        candidate_covers_service_base = True
        for candidate_token in candidate_tokens:
            if not any(
                service_token == candidate_token
                or (
                    len(candidate_token) >= 4
                    and len(service_token) >= 4
                    and (
                        service_token.startswith(candidate_token[:4])
                        or candidate_token.startswith(service_token[:4])
                    )
                )
                for service_token in service_tokens
            ):
                candidate_covers_service_base = False
                break
        if candidate_covers_service_base:
            return True

    service_flags = _lab_price_variant_flags({"serviceName": service_name})
    if not service_flags:
        return False

    query_flags = _query_price_variant_flags(query_candidate)
    return not service_flags.issubset(query_flags)


def _meaningful_price_service_tokens(value: str) -> list[str]:
    """
    Возвращает информативные токены service_name без общих служебных слов.

    :param value: строка услуги
    :return: список нормализованных токенов
    """

    return [
        tok
        for tok in _price_query_tokens(value)
        if tok not in _PRICE_GENERIC_SERVICE_TOKENS
    ]


def _is_price_service_noise_token(token: str) -> bool:
    """
    Проверяет, что токен не добавляет предметной специфики к услуге.

    :param token: нормализованный токен услуги
    :return: True для шумового токена
    """

    norm = _normalise_price_token(token)
    if not norm:
        return True
    if norm.isdigit():
        return True
    return norm in _PRICE_QUERY_SERVICE_NOISE_TOKENS


def _select_effective_price_service_name(entity_service_name: str, query_service_name: str) -> str:
    """
    Выбирает итоговое имя услуги между извлеченной entity и candidate из query.

    Правило защищает от деградации, когда нижний слой повторно извлекает услугу
    из полного текста и получает более шумную строку с адресом, вторым интентом
    или служебными словами. При этом новый query-candidate все еще может
    победить, если он действительно задает другую или более точную услугу.

    :param entity_service_name: service_name, уже выделенный NLU/grounding слоем
    :param query_service_name: service_name, извлеченный из полного query
    :return: наиболее надежное имя услуги для дальнейшей обработки
    """

    entity = str(entity_service_name or "").strip()
    query = str(query_service_name or "").strip()
    if not entity:
        return query
    if not query:
        return entity

    entity_norm = _normalise_input(entity).replace("ё", "е")
    query_norm = _normalise_input(query).replace("ё", "е")
    if not query_norm or entity_norm == query_norm:
        return entity

    entity_tokens = _meaningful_price_service_tokens(entity)
    query_tokens = _meaningful_price_service_tokens(query)
    if not entity_tokens:
        return query
    if not query_tokens:
        return entity

    entity_set = set(entity_tokens)
    query_set = set(query_tokens)
    if not entity_set.intersection(query_set):
        return query
    if query_set.issubset(entity_set):
        return entity

    query_extra = [tok for tok in query_tokens if tok not in entity_set]
    if not query_extra:
        return entity

    if entity_set.issubset(query_set):
        if any(_is_price_service_noise_token(tok) for tok in query_extra):
            return entity
        entity_kind = _detect_service_kind(entity, query_text=entity)
        query_kind = _detect_service_kind(query, query_text=query)
        if entity_kind != query_kind:
            return entity
        return query

    return query


def _compound_price_secondary_lab_service(
    query_text: str,
    *,
    primary_service_name: str,
    retail_rows: list[dict[str, Any]],
) -> str | None:
    """
    Пытается выделить вторую лабораторную услугу из mixed PRICE-запроса.

    Используем только консервативные сигналы:
    - явные конструкции вида `кровь на ...` / `анализ на ...`;
    - короткие alias из справочника `_PRICE_SERVICE_ALIASES`.

    :param query_text: исходный запрос пользователя
    :param primary_service_name: уже выбранная primary-услуга
    :param retail_rows: строки retail-прайса для catalog-grounding
    :return: каноническое имя второй лабораторной услуги либо None
    """

    query_norm = _normalise_input(str(query_text or "")).replace("ё", "е")
    primary_norm = _normalise_input(str(primary_service_name or "")).replace("ё", "е")
    if not query_norm or not primary_norm or " и " not in f" {query_norm} ":
        return None

    fragments: list[str] = []
    fragment_from_lab_phrase = False
    match = _PRICE_COMPOUND_LAB_FRAGMENT_RE.search(query_norm)
    if match:
        fragment = str(match.group(1) or "").strip(" -")
        if fragment:
            fragments.append(fragment)
            fragment_from_lab_phrase = True

    for alias in sorted(_PRICE_SERVICE_ALIASES.keys(), key=len, reverse=True):
        alias_norm = _normalise_input(alias).replace("ё", "е")
        if not alias_norm or alias_norm in primary_norm or alias_norm not in query_norm:
            continue
        fragments.append(alias)

    seen: set[str] = set()
    for idx, fragment in enumerate(fragments):
        key = _normalise_input(fragment).replace("ё", "е")
        if not key or key in seen:
            continue
        seen.add(key)
        candidate = resolve_price_service_name_from_catalog(fragment, rows=retail_rows)
        if not candidate:
            variants = _PRICE_SERVICE_ALIASES.get(key, ())
            candidate = str(variants[0] or "").strip() if variants else ""
        candidate_norm = _normalise_input(candidate).replace("ё", "е")
        if not candidate_norm or candidate_norm == primary_norm:
            continue
        top_rows = _select_patient_price_rows(retail_rows, candidate, limit=3)
        if not top_rows:
            continue
        if idx == 0 and fragment_from_lab_phrase:
            return candidate
        kind = _classify_catalog_service_kind(
            candidate,
            query_text=candidate,
            retail_rows=top_rows,
            has_exact_doctor_link=False,
            is_consult_query=False,
        )
        if kind == "lab":
            return candidate
    return None


def _build_compound_price_clarify_payload(
    *,
    query_text: str,
    entities: dict[str, Any],
    primary_service_name: str,
    retail_rows: list[dict[str, Any]],
    primary_retail_prices: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Строит clarify-payload для mixed PRICE-запроса с двумя услугами.

    Мы не пытаемся сразу оркестрировать сложный multi-service ответ. Вместо
    этого честно просим выбрать, какую услугу разобрать первой, если видим
    надежную primary-услугу и отдельную вторую лабораторную цель.

    :param query_text: исходный запрос пользователя
    :param entities: текущие сущности роутера
    :param primary_service_name: уже выбранная основная услуга
    :param retail_rows: все строки retail-прайса региона
    :param primary_retail_prices: top retail rows для основной услуги
    :return: payload для clarify либо None
    """

    secondary_labels = {
        str(label or "").strip().upper()
        for label in (entities.get("secondary_intents") or [])
        if str(label or "").strip()
    }
    if "TEST_ASSIST" not in secondary_labels:
        return None

    secondary_service = _compound_price_secondary_lab_service(
        query_text,
        primary_service_name=primary_service_name,
        retail_rows=retail_rows,
    )
    if not secondary_service:
        return None

    primary_kind = _classify_catalog_service_kind(
        primary_service_name,
        query_text=primary_service_name,
        retail_rows=primary_retail_prices,
        has_exact_doctor_link=False,
        is_consult_query=_is_consultation_service_query(primary_service_name),
    )
    if primary_kind == "lab":
        return None

    clarify_text = (
        "Вижу в запросе две услуги:\n"
        f"1. {primary_service_name}\n"
        f"2. {secondary_service}\n\n"
        "Чтобы не смешать цену и доступность по разным услугам, лучше проверить их по очереди.\n"
        f"Если хотите, сначала покажу по {primary_service_name}. "
        f"Также можно сразу написать: «{secondary_service}»."
    )
    return {
        "service_name": primary_service_name,
        "retail_prices": primary_retail_prices,
        "doctors": [],
        "prepare": "",
        "show_prepare": False,
        "service_kind": "compound_clarify",
        "clarify_text": clarify_text,
        "compound_price_services": [primary_service_name, secondary_service],
        "compound_price_default_service": primary_service_name,
        "note": "service_bundle_info: compound_price_clarify",
    }


def match_compound_price_service_option(user_text: str, options: list[str]) -> str | None:
    """
    Сопоставляет короткий follow-up пользователя с одной из услуг compound PRICE.

    :param user_text: текущая реплика пользователя
    :param options: допустимые услуги из pending compound flow
    :return: выбранная услуга либо None
    """

    reply_norm = _normalise_input(str(user_text or "")).replace("ё", "е")
    if not reply_norm:
        return None
    reply_tokens = set(_meaningful_price_service_tokens(reply_norm))

    alias_hits: set[str] = set()
    for alias, variants in _PRICE_SERVICE_ALIASES.items():
        alias_norm = _normalise_input(alias).replace("ё", "е")
        if alias_norm and alias_norm in reply_norm:
            alias_hits.add(alias_norm)
            for variant in variants:
                variant_norm = _normalise_input(variant).replace("ё", "е")
                if variant_norm:
                    alias_hits.add(variant_norm)

    for option in options:
        option_text = str(option or "").strip()
        option_norm = _normalise_input(option_text).replace("ё", "е")
        if not option_norm:
            continue
        if reply_norm == option_norm or reply_norm in option_norm or option_norm in reply_norm:
            return option_text
        option_tokens = set(_meaningful_price_service_tokens(option_text))
        if reply_tokens and option_tokens and (reply_tokens.issubset(option_tokens) or option_tokens.issubset(reply_tokens)):
            return option_text
        if alias_hits and any(alias in option_norm for alias in alias_hits):
            return option_text
    return None


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
    candidates: list[str] = []
    for raw in (query, entity_query):
        phrase = _extract_prepare_entity_phrase(str(raw or "").strip())
        if phrase:
            candidates.append(str(phrase).strip())
    if candidates:
        # Предпочитаем более полную форму из запроса пользователя.
        return max(candidates, key=lambda x: len(str(x or "").strip()))
    return str(query or entity_query or "исследованию").strip()


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
    content_norm = _normalise_input(content).replace("ё", "е")
    if not content_norm:
        return False
    if not any(x in content_norm for x in ("подготов", "натощак", "перед", "за ")):
        return False

    score = _prepare_fast_relevance_score(query, content)
    low, _, _ = _prepare_relevance_thresholds()
    dynamic_cutoff = max(0.20, low * 0.85)
    if score < dynamic_cutoff:
        return False

    query_roots = _prepare_term_roots(query)
    if query_roots:
        content_roots = _prepare_term_roots(content)
        if _prepare_roots_coverage(query_roots, content_roots) <= 0.0:
            return False
    if not _is_prepare_content_actionable(content):
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
    "необходим",
    "рекоменду",
)
_PREPARE_STRONG_HINTS = (
    "натощак",
    "за час",
    "за сутки",
    "утром",
    "вечером",
    "воздерж",
    "исключ",
    "не кур",
    "не употреб",
    "пить воду",
)
_PREPARE_LLM_WRAP_NO_RELEVANT = "NO_RELEVANT_CONTENT"
_PREPARE_LLM_WRAP_FALLBACK_PROMPT = (
    "Ты ассистент клиники. Сократи ответ по подготовке к исследованию.\n"
    "Используй только факты из блока ИСТОЧНИК, ничего не выдумывай.\n"
    "Оставь только то, что релевантно запросу пациента.\n"
    "Формат ответа:\n"
    "- краткая вводная (1 предложение);\n"
    "- 2-6 пунктов с конкретными шагами.\n"
    "Если релевантной информации нет, верни строго: NO_RELEVANT_CONTENT.\n\n"
    "ЗАПРОС ПАЦИЕНТА:\n<<USER_QUERY>>\n\n"
    "ИСТОЧНИК:\n<<SOURCE_TEXT>>\n"
)
_PREPARE_CODE_FENCE_START_RE = re.compile(r"^\s*```(?:\w+)?\s*", re.I)
_PREPARE_CODE_FENCE_END_RE = re.compile(r"\s*```\s*$", re.I)


def _prepare_wrap_prompt(query: str, source_text: str) -> str:
    """
    Формирует prompt для LLM-компактора ответа PREPARE.

    :param query: исходный запрос пользователя
    :param source_text: сырой текст подготовки из источника
    :return: итоговый prompt
    """

    try:
        tmpl = load_prompt_text("messenger_final_answer")
    except Exception:
        tmpl = ""
    if not str(tmpl or "").strip():
        tmpl = _PREPARE_LLM_WRAP_FALLBACK_PROMPT
    return (
        str(tmpl or "")
        .replace("<<USER_QUERY>>", str(query or "").strip())
        .replace("<<SOURCE_TEXT>>", str(source_text or "").strip())
        .strip()
    )


def _prepare_wrap_clean(text: str) -> str:
    """
    Нормализует ответ LLM после компактирования.

    :param text: raw-ответ модели
    :return: очищенный текст
    """

    out = str(text or "").strip()
    if not out:
        return ""
    out = _PREPARE_CODE_FENCE_START_RE.sub("", out, count=1)
    out = _PREPARE_CODE_FENCE_END_RE.sub("", out, count=1)
    out = html_cleaner.strip_html(out).strip()
    out = re.sub(r"^\s*(?:ответ|краткий ответ|результат)\s*:\s*", "", out, flags=re.I)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _prepare_query_has_specific_target(query: str) -> bool:
    """
    Проверяет, есть ли в запросе пациента конкретный объект подготовки.

    :param query: текст запроса
    :return: True при наличии таргета (например, ФГДС/биопсия/холестерин)
    """

    return bool(_prepare_term_roots(query))


def _is_prepare_wrap_output_usable(query: str, source_text: str, wrapped: str) -> bool:
    """
    Валидация ответа LLM-компактора, чтобы не ухудшить качество PREPARE.

    :param query: исходный запрос пользователя
    :param source_text: исходный текст подготовки
    :param wrapped: компактный ответ LLM
    :return: True, если результат можно отдавать пациенту
    """

    wrapped_text = str(wrapped or "").strip()
    if not wrapped_text:
        return False
    if _PREPARE_LLM_WRAP_NO_RELEVANT.lower() in wrapped_text.lower():
        return False
    if len(wrapped_text) > max(2200, len(source_text) + 250):
        return False
    if not _is_prepare_content_actionable(wrapped_text):
        return False
    if _prepare_query_has_specific_target(query) and not _is_prepare_relevant(query, wrapped_text):
        return False

    source_tokens = _doc_tokens(source_text)
    wrapped_tokens = _doc_tokens(wrapped_text)
    if source_tokens and wrapped_tokens and not (source_tokens & wrapped_tokens):
        return False

    if len(source_text) >= 900 and len(wrapped_text) >= int(len(source_text) * 0.95):
        return False
    return True


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

    if any(hint in norm for hint in _PREPARE_ACTIONABLE_HINTS):
        return True
    if len(tokens) <= 5 and ("подготовк" in norm and ("исследован" in norm or "анализ" in norm or "процедур" in norm)):
        return False
    return len(tokens) >= 20


def _has_prepare_strong_hints(content: str) -> bool:
    norm = _normalise_input(content).replace("ё", "е")
    if not norm:
        return False
    return any(h in norm for h in _PREPARE_STRONG_HINTS)


def _is_prepare_service_info_usable(query: str, content: str, *, title: str = "") -> bool:
    """
    Решает, можно ли принимать API-first результат `serviceInfoAll` без fallback.

    :param query: исходный пользовательский запрос
    :param content: текст подготовки из API-кэша
    :return: True, если ответ достаточно качественный и релевантный
    """

    if not _is_prepare_content_actionable(content):
        return False

    score = _prepare_fast_relevance_score(query, content, title=title)
    low, _, _ = _prepare_relevance_thresholds()
    if score < low and title and _prepare_term_roots(query):
        title_cov = _prepare_roots_coverage(_prepare_term_roots(query), _prepare_term_roots(title))
        if title_cov >= 0.99 and _has_prepare_strong_hints(content):
            return True
    return score >= low


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
    _service_catalog_rows_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _service_catalog_rows_loaded_at: float = field(default=0.0, init=False)
    _service_catalog_last_error: str = field(default="", init=False)
    _service_catalog_last_source_counts: dict[str, int] = field(default_factory=dict, init=False)
    _schedule_cache_client: AsyncListTTLStaleCache = field(init=False)

    # блокировка, чтобы несколько запросов параллельно не перегенерировали кэш
    _doctors_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _regions_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _procedure_rows_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _service_catalog_rows_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # TTL in-memory кэша (латентность, сек, определят свежесть кэша)
    doctors_mem_ttl_seconds: int = 300
    regions_mem_ttl_seconds: int = 300
    procedure_rows_mem_ttl_seconds: int = 300
    service_catalog_mem_ttl_seconds: int = 300
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

    async def _ensure_service_catalog_rows_loaded(self) -> list[dict[str, Any]]:
        """
        Готовит объединенный каталог услуг клиники для exact/fuzzy матчинга.

        Источники:
        - city retail прайс `priceByRegion` (анализы и общие услуги);
        - doctor prices (doctorServicePricesByRegion) для услуг, которые бывают
          только в doctor-строках.
        """

        now = time.time()
        if self._service_catalog_rows_loaded_at and (now - self._service_catalog_rows_loaded_at) < self.service_catalog_mem_ttl_seconds:
            return self._service_catalog_rows_cache

        async with self._service_catalog_rows_lock:
            now = time.time()
            if self._service_catalog_rows_loaded_at and (now - self._service_catalog_rows_loaded_at) < self.service_catalog_mem_ttl_seconds:
                return self._service_catalog_rows_cache

            merged_rows: list[dict[str, Any]] = []
            source_errors: list[str] = []
            try:
                retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
            except Exception as exc:
                source_errors.append(f"price_by_region:{type(exc).__name__}")
                retail_rows = []
            try:
                doctor_rows = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception as exc:
                source_errors.append(f"doctor_prices:{type(exc).__name__}")
                doctor_rows = []

            for src in (retail_rows, doctor_rows):
                if not isinstance(src, list):
                    continue
                for row in src:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("serviceName") or row.get("name") or "").strip()
                    if len(name) < 3:
                        continue
                    merged_rows.append(
                        {
                            "serviceName": name,
                            "name": name,
                            "serviceHomecode": str(row.get("serviceHomecode") or row.get("homecode") or "").strip(),
                            "cost": _as_int(row.get("cost")) or 0,
                        }
                    )

            deduped: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for row in merged_rows:
                name_norm = _normalise_catalog_text(str(row.get("serviceName") or row.get("name") or ""))
                if not name_norm:
                    continue
                code_norm = _normalise_input(str(row.get("serviceHomecode") or "")).replace("ё", "е")
                key = (name_norm, code_norm)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)

            self._service_catalog_rows_cache = deduped
            self._service_catalog_rows_loaded_at = time.time()
            self._service_catalog_last_error = ";".join(source_errors)
            self._service_catalog_last_source_counts = {
                "retail_rows": len(retail_rows) if isinstance(retail_rows, list) else 0,
                "doctor_rows": len(doctor_rows) if isinstance(doctor_rows, list) else 0,
                "catalog_rows": len(deduped),
            }
            return self._service_catalog_rows_cache

    async def get_catalog_health(self) -> dict[str, Any]:
        """
        Централизованный health-check каталога для роутера.

        Возвращает сводный статус каталогов, которые используются для
        service/doctor grounding в мессенджерном контуре.
        """

        service_rows = await self._ensure_service_catalog_rows_loaded()
        doctors = await self._ensure_doctors_cache_loaded()
        service_ok = len(service_rows) > 0
        doctors_ok = len(doctors) > 0
        ok = service_ok and doctors_ok

        reasons: list[str] = []
        if not service_ok:
            reasons.append(self._service_catalog_last_error or "service_catalog_empty")
        if not doctors_ok:
            reasons.append("doctors_catalog_empty")

        return {
            "ok": ok,
            "status": "ok" if ok else "degraded",
            "service_catalog_ok": service_ok,
            "doctors_catalog_ok": doctors_ok,
            "reason": ";".join(reasons),
            "checked_at": int(time.time()),
            "source_counts": dict(self._service_catalog_last_source_counts or {}),
        }

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, Any]:
        """
        Матчит врача по каталогу doctors-cache:
        1) exact (resolve_schedule_surname)
        2) fuzzy (difflib по фамилии)
        """

        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return {"status": "miss", "query": "", "canonical": ""}

        queries = _doctor_catalog_query_candidates(raw_text_or_name)
        if not queries:
            return {"status": "miss", "query": "", "canonical": ""}

        for query in queries:
            exact = resolve_schedule_surname(query, doctors)
            if exact:
                return {
                    "status": "exact",
                    "query": query,
                    "canonical": str(exact).strip(),
                }

        surname_map: dict[str, str] = {}
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            fio = str(doc.get("fio") or "").strip()
            if not fio:
                continue
            surname = str(fio.split()[0] or "").strip()
            norm = _normalise_catalog_text(surname)
            if norm and norm not in surname_map:
                surname_map[norm] = surname
        surname_keys = list(surname_map.keys())
        if not surname_keys:
            return {"status": "miss", "query": "", "canonical": ""}

        for query in queries:
            norm = _normalise_catalog_text(query)
            if len(norm) < 4:
                continue
            hit = get_close_matches(norm, surname_keys, n=1, cutoff=0.84)
            if not hit:
                continue
            canonical = surname_map.get(hit[0], "").strip()
            if canonical and _normalise_catalog_text(canonical) != norm:
                return {
                    "status": "fuzzy",
                    "query": query,
                    "canonical": canonical,
                    "matched_key": hit[0],
                }
        return {"status": "miss", "query": queries[0], "canonical": ""}

    async def match_catalog_service(
        self,
        raw_text_or_name: str,
        *,
        current_service_name: str = "",
    ) -> dict[str, Any]:
        """
        Матчит услугу по объединенному каталогу услуг клиники:
        1) exact через resolver price-catalog
        2) fuzzy через difflib по нормализованным названиям услуг
        """

        queries = _service_catalog_query_candidates(
            raw_text_or_name,
            current_service_name=current_service_name,
        )
        if not queries:
            return {"status": "miss", "query": "", "canonical": ""}

        catalog_rows = await self._ensure_service_catalog_rows_loaded()
        if not catalog_rows:
            return {
                "status": "unavailable",
                "query": queries[0],
                "canonical": "",
                "reason": self._service_catalog_last_error or "service_catalog_empty",
            }

        for query in queries:
            exact = resolve_price_service_name_from_catalog(
                query,
                current_service_name="",
                rows=catalog_rows,
            )
            if exact:
                return {
                    "status": "exact",
                    "query": query,
                    "canonical": str(exact).strip(),
                }

        name_map: dict[str, str] = {}
        for row in catalog_rows:
            name = str(row.get("serviceName") or row.get("name") or "").strip()
            norm = _normalise_catalog_text(name)
            if norm and norm not in name_map:
                name_map[norm] = name
        name_keys = list(name_map.keys())
        if not name_keys:
            return {"status": "miss", "query": "", "canonical": ""}

        for query in queries:
            norm = _normalise_catalog_text(query)
            if len(norm) < 4:
                continue
            hit = get_close_matches(norm, name_keys, n=1, cutoff=0.86)
            if not hit:
                continue
            canonical = str(name_map.get(hit[0]) or "").strip()
            if canonical and _normalise_catalog_text(canonical) != norm:
                return {
                    "status": "fuzzy",
                    "query": query,
                    "canonical": canonical,
                    "matched_key": hit[0],
                }

        return {"status": "miss", "query": queries[0], "canonical": ""}

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
        if _is_price_show_all_request(query_text):
            family_payload = _price_family_payload_from_context(entities, show_all=True)
            if family_payload:
                family_payload["entities_used"] = entities
                return family_payload
        if _is_generic_uzi_price_request(query_text):
            return {
                "service_name": "УЗИ",
                "retail_prices": [],
                "doctors": [],
                "prepare": "",
                "show_prepare": False,
                "top_n_applied": top_limit,
                "clarify_text": (
                    "Введите конкретное название процедуры, например: "
                    "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
                ),
                "note": "service_bundle_info: generic_uzi_clarify",
                "entities_used": entities,
            }
        try:
            retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
        except Exception:
            retail_rows = []
        retail_rows = [p for p in retail_rows if isinstance(p, dict)]

        family_payload = _build_price_family_payload(
            query_text,
            retail_rows,
            show_all=False,
            visible_limit=10,
        )
        if family_payload:
            family_payload["top_n_applied"] = top_limit
            family_payload["entities_used"] = entities
            return family_payload
        if entity_service_name and _is_city_only_reply(query_text):
            query_service_name = None
        else:
            query_service_name = resolve_price_service_name_from_catalog(
                query_text,
                current_service_name=entity_service_name,
            ) or _extract_price_service_from_query(query_text)
        service_name = _select_effective_price_service_name(
            entity_service_name,
            query_service_name,
        )
        needle = _normalise_input(service_name)

        out: dict[str, Any] = {
            "service_name": service_name,
            "retail_prices": [],
            "doctors": [],
            "prepare": "",
            "show_prepare": False,
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
        retail_query = service_name
        retail_prefers_query_candidate = False
        if query_text and not _is_city_only_reply(query_text):
            query_candidate = _extract_price_service_from_query(query_text)
            retail_prefers_query_candidate = _should_prefer_retail_query_candidate(
                query_candidate or "",
                service_name,
            )
            if retail_prefers_query_candidate:
                retail_query = query_candidate or service_name
        try:
            out["retail_prices"] = _select_patient_price_rows(
                retail_rows,
                retail_query,
                limit=5,
            )
            out["retail_prices"] = _annotate_price_rows_with_care_context(out["retail_prices"])
        except Exception:
            out["retail_prices"] = []
            out["note"] = "service_bundle_info: retail source unavailable"
        if retail_prefers_query_candidate and retail_query:
            out["service_name"] = retail_query

        compound_payload = _build_compound_price_clarify_payload(
            query_text=query_text,
            entities=entities,
            primary_service_name=service_name,
            retail_rows=retail_rows,
            primary_retail_prices=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
        )
        if compound_payload:
            compound_payload["top_n_applied"] = top_limit
            compound_payload["entities_used"] = {
                **entities,
                "service_name_effective": service_name,
            }
            return compound_payload

        # 2) Top-N doctors by ord among doctors that have the matched service in doctor prices.
        top_retail = out["retail_prices"][0] if isinstance(out.get("retail_prices"), list) and out["retail_prices"] else {}
        target_homecode = _normalise_input(
            str(top_retail.get("serviceHomecode") or top_retail.get("homecode") or "")
        )
        is_consult_query = _is_consultation_service_query(service_name)
        preliminary_kind = _classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=False,
            is_consult_query=is_consult_query,
        )
        query_norm = _normalise_input(service_name).replace("ё", "е")
        query_tokens = _price_query_tokens(service_name)
        homecode_query = _extract_homecode_query(service_name)
        matched_price_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
        exact_link_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
        samara_tokens: set[str] = set()
        by_id: dict[int, dict[str, Any]] = {}
        doctor_prices: list[dict[str, Any]] = []
        service_kind = preliminary_kind
        if preliminary_kind not in {"lab", "diagnostic_no_doctor"}:
            samara_tokens = await self._samara_region_tokens()
            doctors = await self._ensure_doctors_cache_loaded()
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

            try:
                doctor_prices = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception:
                doctor_prices = []

            for row in doctor_prices:
                if not isinstance(row, dict):
                    continue
                doctor_id = _as_int(row.get("doctorId"))
                if doctor_id is None or doctor_id not in by_id:
                    continue
                row_name_norm = _normalise_input(str(row.get("serviceName") or row.get("name") or "")).replace("ё", "е")
                row_homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
                score, matched = _price_row_score(
                    row,
                    query=query_norm,
                    tokens=query_tokens,
                    homecode_query=homecode_query,
                )
                if not is_consult_query and target_homecode and row_homecode and target_homecode == row_homecode:
                    score = max(score, 260)
                    matched = max(matched, 1)
                if score <= 0:
                    continue
                if not is_consult_query and not _is_strong_doctor_price_match(
                    query_norm=query_norm,
                    query_tokens=query_tokens,
                    row_name_norm=row_name_norm,
                    matched_tokens=matched,
                    target_homecode=target_homecode,
                    row_homecode=row_homecode,
                ):
                    continue
                cost = _as_int(row.get("cost")) or 0
                item = (score, matched, -cost, doctor_id, row)
                matched_price_rows.append(item)
                if target_homecode and row_homecode and target_homecode == row_homecode:
                    exact_link_rows.append(item)

            has_reliable_doctor_link = bool(exact_link_rows) or _has_reliable_doctor_service_link(
                matched_price_rows,
                query_norm,
            )
            service_kind = _classify_catalog_service_kind(
                service_name,
                query_text=query_text,
                retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
                has_exact_doctor_link=has_reliable_doctor_link,
                is_consult_query=is_consult_query,
            )
        if service_kind == "ambiguous":
            service_kind = await _resolve_ambiguous_price_kind_with_llm(
                query_text,
                out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
                has_exact_doctor_link=bool(exact_link_rows) or _has_reliable_doctor_service_link(
                    matched_price_rows,
                    query_norm,
                ),
                runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
            )
        if service_kind == "operator":
            return _service_fallback(
                note="service_bundle_info ambiguous operator fallback",
                handoff_message="Сейчас по этой услуге безопаснее уточнить у оператора. Соединяю с оператором.",
                entities=entities,
                reason="ambiguous_price_service",
                extra={
                    "retail_prices": out.get("retail_prices") or [],
                    "service_name": service_name,
                },
            )
        if service_kind == "family_query":
            family_payload = _build_price_family_payload(
                query_text,
                retail_rows,
                show_all=False,
                visible_limit=10,
            )
            if family_payload:
                family_payload["entities_used"] = entities
                family_payload["top_n_applied"] = top_limit
                return family_payload
        out["service_kind"] = service_kind

        if service_kind in {"doctor_consult", "procedure_with_doctor"}:
            candidate_rows = (
                exact_link_rows
                if service_kind == "procedure_with_doctor" and exact_link_rows
                else matched_price_rows
            )
            allow_soft_substring_fallback = service_kind == "doctor_consult" and len(query_tokens) <= 1
            if not candidate_rows and query_norm and allow_soft_substring_fallback:
                for row in doctor_prices:
                    if not isinstance(row, dict):
                        continue
                    doctor_id = _as_int(row.get("doctorId"))
                    if doctor_id is None or doctor_id not in by_id:
                        continue
                    service_row_name = _normalise_input(str(row.get("serviceName") or ""))
                    if query_norm and query_norm in service_row_name:
                        cost = _as_int(row.get("cost")) or 0
                        candidate_rows.append((1, 1, -cost, doctor_id, row))

            candidate_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
            best_row_by_doctor: dict[int, dict[str, Any]] = {}
            for _, _, _, doctor_id, row in candidate_rows:
                if doctor_id not in best_row_by_doctor:
                    best_row_by_doctor[doctor_id] = row

            doctor_cards = sorted(
                [by_id[doctor_id] for doctor_id in best_row_by_doctor if doctor_id in by_id],
                key=_doctor_sort_key,
            )[:top_limit]

            out_doctors: list[dict[str, Any]] = []
            query_specialty = _extract_specialty_from_text(query_text) or _extract_specialty_from_text(service_name)
            for doc in doctor_cards:
                doctor_id = _as_int(doc.get("id"))
                if doctor_id is None:
                    continue
                if service_kind == "doctor_consult" and query_specialty and not _doctor_matches_primary_specialty(doc, query_specialty):
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
                        "specialization": _compact_specialization(
                            _pick_display_specialization(
                                doc,
                                preferred_specialty=query_specialty,
                                preferred_service=service_name,
                            )
                        ),
                        "specialty_label": _specialty_label_for_doctor(
                            doc,
                            preferred_specialty=query_specialty,
                        ),
                        "regions": [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()],
                        "service_price": _as_int(price_row.get("cost")),
                        "available": bool(availability.get("available")),
                        "nearest_slot": str(availability.get("nearest_slot") or ""),
                        "regions_with_slots": list(availability.get("regions_with_slots") or []),
                        "availability_note": str(availability.get("note") or ""),
                    }
                )
            out["doctors"] = out_doctors
        else:
            out["doctors"] = []
            out["note"] = (
                f"{out['note']}; " if str(out.get("note") or "").strip() else ""
            ) + f"service_bundle_info: {service_kind or 'no_doctors'}"

        # 3) Preparation guidance by service/test name.
        # В PRICE показываем подготовку только по явному запросу пациента.
        # Иначе блок шумит и мешает основной задаче (цена/врач/расписание).
        show_prepare = _is_prepare_requested_in_price_query(query_text)
        out["show_prepare"] = show_prepare
        if show_prepare and not is_consult_query:
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
                if isinstance(data, list) and data and _schedule_payload_matches_doctor(data, candidate):
                    last_name = candidate
                    break
                if _is_schedule_no_slots_text(data):
                    schedule_unavailable_reason = "no_free_slots_2_weeks"
                # fallback: если регионный фильтр дал пусто, пробуем без региона
                if region_name:
                    data = await self._get_schedule_payload_cached(candidate, None)
                    if isinstance(data, list) and data and _schedule_payload_matches_doctor(data, candidate):
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
                if last_name:
                    row_fio = str(item.get("fio") or "").strip()
                    if row_fio and not _doctor_matches_fio(row_fio, str(last_name), resolved_surname=str(last_name)):
                        continue
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

    async def _prepare_llm_validate_candidate(
        self,
        query: str,
        candidate: _PrepareCandidate,
    ) -> tuple[bool, float, str]:
        """
        LLM-валидация релевантности для кандидата PREPARE в серой зоне score.

        :param query: запрос пользователя
        :param candidate: кандидат из API/Meili
        :return: (релевантно, confidence, reason)
        """

        if not _runtime_bool("MR_PREPARE_RELEVANCE_LLM_ENABLED", True):
            return False, 0.0, "llm_disabled"

        prompt = _prepare_relevance_prompt(
            query,
            candidate.text,
            source_kind=candidate.source,
            service_title=candidate.service_title,
        )
        if not prompt:
            return False, 0.0, "empty_prompt"

        timeout_s = _runtime_int(
            "MR_PREPARE_RELEVANCE_LLM_TIMEOUT_S",
            15,
            min_value=1,
            max_value=60,
        )
        queue_timeout_ms = _runtime_int(
            "MR_PREPARE_RELEVANCE_LLM_QUEUE_TIMEOUT_MS",
            3000,
            min_value=200,
            max_value=20000,
        )
        try:
            raw = await generate_text(
                prompt,
                timeout_s=timeout_s,
                queue_timeout_ms=queue_timeout_ms,
                fmt="json",
                think=False,
            )
        except Exception as e:
            logger.info("prepare relevance llm skipped: %s", e.__class__.__name__)
            return False, 0.0, "llm_unavailable"

        return _parse_prepare_relevance_validator(str(raw or ""))

    async def _pick_prepare_candidate(
        self,
        query: str,
        candidates: list[_PrepareCandidate],
    ) -> _PrepareCandidate | None:
        """
        Выбирает лучший кандидат PREPARE по fast-score + LLM в серой зоне.

        :param query: запрос пользователя
        :param candidates: кандидаты из источников
        :return: лучший релевантный кандидат или None
        """

        ranked = _dedupe_prepare_candidates(candidates, limit=12)
        if not ranked:
            return None

        max_llm_checks = _runtime_int(
            "MR_PREPARE_RELEVANCE_LLM_MAX_CHECKS",
            2,
            min_value=1,
            max_value=8,
        )
        llm_checks = 0
        query_roots = _prepare_term_roots(query)
        for idx, cand in enumerate(ranked):
            next_score = ranked[idx + 1].score if idx + 1 < len(ranked) else 0.0
            margin = max(0.0, float(cand.score) - float(next_score))
            cand.margin = margin
            gate = _prepare_relevance_gate(cand.score, cand.margin)
            if gate == "accept":
                cand.note = (cand.note + "; " if cand.note else "") + "prepare_fast_gate=accept"
                return cand
            if gate == "reject":
                cand.note = (cand.note + "; " if cand.note else "") + "prepare_fast_gate=reject"
                continue

            if cand.source == "serviceInfoAll" and query_roots:
                title_roots = _prepare_term_roots(cand.service_title)
                title_cov = _prepare_roots_coverage(query_roots, title_roots)
                if title_cov >= 0.99 and _has_prepare_strong_hints(cand.text):
                    cand.note = (
                        (cand.note + "; " if cand.note else "")
                        + "prepare_fast_gate=accept_service_title_anchor"
                    )
                    return cand

            if llm_checks >= max_llm_checks:
                cand.note = (cand.note + "; " if cand.note else "") + "prepare_llm_skipped=max_checks"
                continue

            llm_checks += 1
            ok, confidence, reason = await self._prepare_llm_validate_candidate(query, cand)
            cand.note = (
                (cand.note + "; " if cand.note else "")
                + f"prepare_llm={_PREPARE_RELEVANCE_VERDICT_RELEVANT if ok else _PREPARE_RELEVANCE_VERDICT_IRRELEVANT}"
                + f"({confidence:.2f})"
                + (f":{reason}" if reason else "")
            )
            if ok:
                return cand

            # Quality-first override for Meili mixed-docs:
            # если LLM отверг из-за "смешанности", но у кандидата высокий fast-score,
            # полное покрытие корней запроса и actionable-текст, пропускаем в wrapper.
            if cand.source == "main_index" and query_roots:
                body_roots = _prepare_term_roots(cand.text)
                body_cov = _prepare_roots_coverage(query_roots, body_roots)
                override_score = _runtime_float(
                    "MR_PREPARE_MAIN_INDEX_OVERRIDE_SCORE",
                    0.70,
                    min_value=0.30,
                    max_value=0.95,
                )
                if (
                    cand.score >= override_score
                    and body_cov >= 0.99
                    and _is_prepare_content_actionable(cand.text)
                ):
                    cand.note = (
                        (cand.note + "; " if cand.note else "")
                        + "prepare_llm_reject_override_main_index"
                    )
                    return cand

            if reason in {"llm_unavailable", "llm_non_json", "llm_disabled", "empty_prompt"}:
                fallback_score = _runtime_float(
                    "MR_PREPARE_RELEVANCE_LLM_UNAVAILABLE_ACCEPT_SCORE",
                    0.40,
                    min_value=0.10,
                    max_value=0.95,
                )
                if cand.score >= fallback_score:
                    cand.note = (cand.note + "; " if cand.note else "") + "prepare_llm_fallback_fast_accept"
                    return cand
        return None

    async def _prepare_candidates_from_analysis_api_cache(
        self,
        query: str,
        entities: dict[str, Any],
    ) -> list[_PrepareCandidate]:
        """
        Возвращает отсортированные prepare-кандидаты из serviceInfoAll.

        :param query: исходный запрос пользователя
        :param entities: сущности роутера
        :return: список кандидатов
        """

        entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
        if not _prepare_term_roots(" ".join(x for x in (query, entity_query) if x)):
            return []
        queries = _prepare_service_info_queries(query, entity_query)
        if not queries:
            return []

        try:
            rows = await asyncio.to_thread(api_service_info.load_service_info)
        except Exception:
            return []

        candidates: list[_PrepareCandidate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            service_name = str(row.get("serviceName") or "").strip()
            preparation = str(row.get("preparation") or "").strip()
            if not service_name or not preparation:
                continue

            best_score = 0.0
            best_query = ""
            for query_variant in queries:
                score = _prepare_fast_relevance_score(query_variant, preparation, title=service_name)
                if score > best_score:
                    best_score = score
                    best_query = query_variant

            if best_score <= 0.0:
                continue

            candidates.append(
                _PrepareCandidate(
                    source="serviceInfoAll",
                    text=preparation,
                    query_variant=best_query or query,
                    service_title=service_name,
                    score=best_score,
                    note="prepare: serviceInfoAll candidate",
                )
            )

        return _dedupe_prepare_candidates(candidates, limit=16)

    async def _maybe_compact_prepare_text(self, query: str, source_text: str) -> tuple[str, str, str]:
        """
        Компактирует длинный PREPARE-текст через LLM с безопасным fallback.

        :param query: исходный запрос пациента
        :param source_text: текст подготовки из источника
        :return: (итоговый текст, статус wrap, причина/диагностика)
        """

        text = str(source_text or "").strip()
        if not text:
            return "", "empty_source", "no_source_text"
        if not _runtime_bool("MR_PREPARE_LLM_WRAP_ENABLED", True):
            return text, "disabled", "llm_wrap_disabled"

        min_chars = _runtime_int(
            "MR_PREPARE_LLM_WRAP_MIN_CHARS",
            700,
            min_value=120,
            max_value=12000,
        )
        if len(text) < min_chars:
            return text, "short_source", "below_min_chars"

        source_max_chars = _runtime_int(
            "MR_PREPARE_LLM_WRAP_SOURCE_MAX_CHARS",
            9000,
            min_value=500,
            max_value=30000,
        )
        source_for_prompt = text[:source_max_chars].strip()

        def _fallback_or_source(reason: str) -> tuple[str, str, str]:
            compacted = build_prepare_fallback_answer(
                query,
                source_for_prompt,
                max_chars=_runtime_int(
                    "MR_PREPARE_FALLBACK_MAX_CHARS",
                    1600,
                    min_value=400,
                    max_value=4000,
                ),
                max_points=_runtime_int(
                    "MR_PREPARE_FALLBACK_MAX_POINTS",
                    7,
                    min_value=3,
                    max_value=10,
                ),
            )
            if compacted and len(compacted) < len(text):
                return compacted, "fallback_compact", reason
            if compacted:
                return text, "fallback_not_shorter", reason
            return text, "fallback_failed", reason

        prompt = _prepare_wrap_prompt(query, source_for_prompt)
        if not prompt:
            return _fallback_or_source("empty_prompt")

        timeout_s = _runtime_int(
            "MR_PREPARE_LLM_WRAP_TIMEOUT_S",
            30,
            min_value=3,
            max_value=90,
        )
        queue_timeout_ms = _runtime_int(
            "MR_PREPARE_LLM_WRAP_QUEUE_TIMEOUT_MS",
            6000,
            min_value=300,
            max_value=30000,
        )
        try:
            raw = await generate_text(
                prompt,
                timeout_s=timeout_s,
                queue_timeout_ms=queue_timeout_ms,
                think=False,
            )
        except Exception as e:
            logger.info("prepare llm wrap skipped: %s", e.__class__.__name__)
            return _fallback_or_source(f"llm_wrap_error:{e.__class__.__name__}")

        wrapped = _prepare_wrap_clean(str(raw or ""))
        if not _is_prepare_wrap_output_usable(query, source_for_prompt, wrapped):
            return _fallback_or_source("llm_wrap_invalid_output")
        return wrapped, "llm_wrapped", "ok"

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        raw_query = str(query or "").strip()
        entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
        # Для нового вопроса берем текст пользователя, чтобы не залипала старая услуга из контекста.
        q = raw_query or entity_query
        if not q:
            return {"prepare": "", "note": "no query", "entities_used": entities}

        api_candidates = await self._prepare_candidates_from_analysis_api_cache(q, entities)
        api_best = await self._pick_prepare_candidate(q, api_candidates)
        if api_best:
            api_cached_cleaned = html_cleaner.strip_html(api_best.text).strip()
            if not _is_prepare_service_info_usable(q, api_cached_cleaned, title=api_best.service_title):
                api_best = None
            else:
                api_best.text = api_cached_cleaned
        if api_best:
            compacted, wrap_status, wrap_reason = await self._maybe_compact_prepare_text(q, api_best.text)
            return {
                "prepare": compacted,
                "note": "prepare: serviceInfoAll",
                "entities_used": entities,
                "prepare_wrap_status": wrap_status,
                "prepare_wrap_reason": wrap_reason,
            }

        variants = _prepare_query_variants(q, entity_query)
        if not variants:
            variants = [q]

        meili_candidates: list[_PrepareCandidate] = []
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
            score = max(
                _prepare_fast_relevance_score(q, cleaned),
                _prepare_fast_relevance_score(candidate, cleaned),
            )
            if score <= 0.0:
                continue
            meili_candidates.append(
                _PrepareCandidate(
                    source="main_index",
                    text=cleaned,
                    query_variant=candidate,
                    score=score,
                    note="prepare: main_index candidate",
                )
            )

        meili_best = await self._pick_prepare_candidate(q, meili_candidates)
        if meili_best:
            compacted, wrap_status, wrap_reason = await self._maybe_compact_prepare_text(q, meili_best.text)
            return {
                "prepare": compacted,
                "note": "prepare: main_index",
                "entities_used": entities,
                "prepare_wrap_status": wrap_status,
                "prepare_wrap_reason": wrap_reason,
            }

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

        candidates = await self._prepare_candidates_from_analysis_api_cache(query, entities)
        best = await self._pick_prepare_candidate(query, candidates)
        if not best:
            return None
        return str(best.text or "").strip() or None

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
        has_price_request = bool(query_text and _PRICE_REQUEST_RE.search(query_text))
        if _is_price_show_all_request(query_text):
            family_payload = _price_family_payload_from_context(entities, show_all=True)
            if family_payload:
                family_payload["entities_used"] = entities
                return family_payload
        if _is_generic_uzi_price_request(query_text):
            return {
                "prices": [],
                "clarify_text": (
                    "Введите конкретное название процедуры, например: "
                    "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
                ),
                "note": "price_info: generic_uzi_clarify",
                "entities_used": entities,
            }
        doctor_query_specialty = _extract_specialty_from_text(query_text) if (doctor_id and has_price_request) else ""

        if doctor_id:
            # Для doctor-specific PRICE не приземляемся в городский retail-catalog:
            # иначе вопрос "у Иванова" может маппиться в случайную услугу по всему прайсу.
            query_service_name = _extract_price_service_from_query(query_text) if query_text else None
            if not query_service_name:
                query_service_name = entity_service_name
            # "сколько стоит прием у <врач>" без специальности:
            # принудительно удерживаем консультационный контекст вместо stale service_name.
            if has_price_request and _PRICE_CONSULT_HINT_RE.search(query_text) and not doctor_query_specialty:
                if not _is_consultation_service_query(str(query_service_name or "")):
                    query_service_name = "консультация"
            service_name = query_service_name or ""
            # Если это doctor-specific price без явной услуги, оставляем query как fallback
            # (для редких строк doctor_price, где нет стандартных маркеров).
            if not service_name and has_price_request and _DOCTOR_PRICE_HINT_RE.search(query_text):
                service_name = query_text
        else:
            if entity_service_name and _is_city_only_reply(query_text):
                query_service_name = None
            else:
                query_service_name = resolve_price_service_name_from_catalog(
                    query_text,
                    current_service_name=entity_service_name,
                ) or _extract_price_service_from_query(query_text)
            service_name = _select_effective_price_service_name(
                entity_service_name,
                query_service_name,
            )
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
                ranked = _rank_price_rows(doc_prices, service_name, limit=10)
                if (
                    not ranked
                    and has_price_request
                    and _PRICE_CONSULT_HINT_RE.search(query_text)
                ):
                    ranked = [
                        p for p in doc_prices
                        if _is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                    ]
                doc_prices = ranked
            if not needle:
                if has_price_request and _PRICE_CONSULT_HINT_RE.search(query_text):
                    consult_rows = [
                        p for p in doc_prices
                        if _is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                    ]
                    if consult_rows:
                        doc_prices = consult_rows
                doc_prices = sorted(
                    [p for p in doc_prices if isinstance(p, dict)],
                    key=lambda p: (_normalise_input(str(p.get("serviceName") or "")), _as_int(p.get("cost")) or 0),
                )
            doc_prices = _annotate_price_rows_with_care_context(doc_prices[:10])
            return {
                "prices": doc_prices,
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
        family_payload = _build_price_family_payload(
            query_text,
            [p for p in price_rows if isinstance(p, dict)],
            show_all=False,
            visible_limit=10,
        )
        if family_payload and not doctor_id:
            family_payload["entities_used"] = {
                **entities,
                "service_name_effective": str(family_payload.get("service_name") or "").strip(),
            }
            return family_payload
        if not needle:
            return {"prices": [], "note": "no service query", "entities_used": entities}
        retail_query = service_name
        if query_text and not _is_city_only_reply(query_text):
            query_candidate = _extract_price_service_from_query(query_text)
            if _should_prefer_retail_query_candidate(query_candidate or "", service_name):
                retail_query = query_candidate or service_name
        matches = _select_patient_price_rows(
            [p for p in price_rows if isinstance(p, dict)],
            retail_query,
            limit=10,
        )
        matches = _annotate_price_rows_with_care_context(matches)
        return {
            "prices": matches,
            "service_kind": "lab" if _classify_catalog_service_kind(
                service_name,
                query_text=query_text,
                retail_rows=matches,
                has_exact_doctor_link=False,
                is_consult_query=False,
            ) == "lab" else "",
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

        if service_q and (appointment_mode or _is_procedure_branch_lookup_query(query, service_q)):
            try:
                retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
            except Exception:
                retail_rows = []
            if isinstance(retail_rows, list) and retail_rows:
                care_query = str(service_name or query or "").strip()
                retail_matches = _select_address_price_rows(
                    [row for row in retail_rows if isinstance(row, dict)],
                    care_query,
                    limit=10,
                    family_limit=50,
                )
                care_addresses = _care_setting_addresses_from_price_rows(retail_matches)
                if branch_q:
                    care_addresses = [
                        addr for addr in care_addresses
                        if branch_q in _normalise_input(addr)
                    ]
                if care_addresses:
                    return {
                        "addresses": care_addresses,
                        "branches": _addresses_to_branch_payload(care_addresses, regions),
                        "note": "address_info: priceUnits care-setting",
                        "entities_used": entities,
                    }

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


from .services.doctors import (
    _doctor_availability_snapshot as _doctor_availability_snapshot_impl,
    _resolve_doctor_id_from_name as _resolve_doctor_id_from_name_impl,
    _schedule_by_specialty as _schedule_by_specialty_impl,
    doctors_info as _doctors_info_impl,
    doctors_schedule_week as _doctors_schedule_week_impl,
    match_catalog_doctor as _match_catalog_doctor_impl,
    resolve_doctor_name as _resolve_doctor_name_impl,
)

Services.match_catalog_doctor = _match_catalog_doctor_impl
Services._schedule_by_specialty = _schedule_by_specialty_impl
Services._doctor_availability_snapshot = _doctor_availability_snapshot_impl
Services.resolve_doctor_name = _resolve_doctor_name_impl
Services._resolve_doctor_id_from_name = _resolve_doctor_id_from_name_impl
Services.doctors_info = _doctors_info_impl
Services.doctors_schedule_week = _doctors_schedule_week_impl


if __name__ == "__main__":
    async def main():
        s = Services()
        print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
        print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

    asyncio.run(main())
