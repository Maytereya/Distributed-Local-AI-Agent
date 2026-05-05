"""Price helpers (Stage 20 cluster 5).

Pure helpers for price-catalog matching, family/OAK variants, compound
price clarification and multi-service resolution. Moved out of
``services_legacy.py`` behind a re-export shim.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent_logic_2.nayka_api import api_price

from ..prompt_registry import load_prompt_text
from ..russian_nlu import normalize_ru
from ..service_phrase import extract_service_phrase
from ._addresses_helpers import _extract_homecode_query
from ._common import _as_int, _normalise_input
from ._doctors_helpers import (
    _UZI_LINE_RE,
    _UZI_PROCEDURE_HINT_RE,
    _classify_catalog_service_kind,
    _extract_specialty_from_text,
    _is_clean_consultation_row_name,
    _is_consultation_service_query,
    _is_lab_like_service_name,
    _is_uzi_query_text,
    _meaningful_price_service_tokens,
    _service_name_allows_specialty,
    _service_name_matches_specialty,
)
from .. import llm_runtime as llm_runtime_mod
from ._regions import _extract_city_token


# ---------------------------------------------------------------------------
# Regex patterns and constant sets (order preserved from services_legacy.py)
# ---------------------------------------------------------------------------

_PRICE_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
_PRICE_REQUEST_RE = re.compile(r"\b(стоим\w*|цен\w*|сколько)\b", re.I)
_PRICE_CONSULT_HINT_RE = re.compile(r"\b(при[её]м\w*|консультаци\w*)\b", re.I)
_DOCTOR_PRICE_HINT_RE = re.compile(r"\b(?:у|врач\w*|доктор\w*)\s+[а-яё\-]{3,}\b", re.I)
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
# Важно: для /priceByRegion нужен city-level regionId (Самара = 3),
# а для /doctorServicePricesByRegion используются branch-level regionId из doctorRegions.
SAMARA_PRICE_REGION_ID = 3
_PRICE_QUERY_STOPWORDS = {
    "спасибо",
    "благодарю",
    "благодарствую",
    "спс",
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
    "подскажи",
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


# ---------------------------------------------------------------------------
# Helpers (order preserved from services_legacy.py for diff readability)
# ---------------------------------------------------------------------------


def _normalise_price_token(token: str) -> str:
    """
    Приводит price-токен к каноническому виду для устойчивого ранжирования.

    :param token: исходный токен из запроса или строки прайса
    :return: нормализованный токен
    """

    norm = normalize_ru(token)
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

    raw = _normalise_input(text)
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

    raw = _normalise_input(raw_text)
    if raw and _UZI_LINE_RE.search(raw):
        _push("узи")
    vitamin_code = _extract_vitamin_designator(raw)
    if vitamin_code:
        _push(f"витамин_{vitamin_code}")
    return out


def _price_query_tokens(text: str) -> list[str]:
    s = _normalise_input(text)
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
    raw = _normalise_input(str(query_text or ""))
    if not raw:
        return []
    out: list[str] = []
    direct = raw.strip(" ?!.,;:")
    if direct:
        out.append(direct)
    extracted = _extract_price_service_from_query(raw)
    if extracted:
        out.append(_normalise_input(extracted))
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
    norm = _normalise_input(extracted)
    return norm in {"узи", "ультразвук", "ультразвуковое исследование"}


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
        key = _normalise_input(value)
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

    current_norm = _normalise_input(current)
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
        cand_norm = _normalise_input(candidate)
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
            query=_normalise_input(query),
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

    query = _normalise_input(query_text)
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
    name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
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
        if query_specialty and not _service_name_allows_specialty(name, query_specialty):
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
            # Минимальный shared-префикс — 5 символов: 4-буквенный префикс
            # пропускал слишком общие греческие корни ("эндо" → эндокринолог /
            # эндомизий / эндотрахеальный при запросе "эндоскопия"). Для
            # медицинских терминов 5 символов разделяют семьи слов адекватно
            # (маммо-, флюоро-, эндос-, рентг-, гастр- и т.д.).
            if len(tok) >= 5 and any(
                len(rt) >= 5 and (rt.startswith(tok[:5]) or tok.startswith(rt[:5]))
                for rt in row_tokens
            ):
                matched += 1
        score += matched * 25
        if matched == len(tokens):
            score += 80
        elif matched >= max(2, len(tokens) - 1):
            score += 40

    # Бонус за «головной» матч.
    # Когда пациент пишет короткую аббревиатуру («ТТГ», «АЛТ», «АСТ»,
    # «СОЭ» и т.п.), он почти всегда имеет в виду базовую услугу,
    # каноническое имя которой НАЧИНАЕТСЯ с этой аббревиатуры —
    # «ТТГ (TSH) тиреотропный гормон». А «Антитела к рецепторам ТТГ»
    # содержит ту же подстроку, но это субспециальное исследование.
    # Без этого бонуса score обоих кандидатов одинаков, и финальный
    # tie-breaker `-name_gap` (длина названия) случайно выбирает тот,
    # у кого имя короче — что для «Антитела к рецепторам ТТГ» (25 символов)
    # < «ТТГ (TSH) тиреотропный гормон» (28 символов). Жалоба
    # заказчика 2026-05-05: бот по запросу «ТТГ» отдаёт «Антитела к
    # рецепторам ТТГ», а простой ТТГ за 380₽ не показывает.
    #
    # Фикс: если первое значимое слово серviceName совпадает с
    # query или с одним из её tokens — даём +90, чтобы перекрыть
    # tie-breaker по name_gap.
    if tokens and row_tokens:
        head_token = row_tokens[0]
        query_first_token = tokens[0]
        if head_token == query or head_token == query_first_token:
            score += 90

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

    query = _normalise_input(query_text)
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

    norm = _normalise_input(str(value or ""))
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
    family_norm = _normalise_input(family_query)
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

    direct_filtered = _select_family_variant_rows(direct_rows, family_query, limit=limit)
    best_rows = list(direct_filtered)
    best_base_names = _family_variant_base_names(best_rows)

    def _row_key(row: dict[str, Any]) -> tuple[str, str]:
        return (
            _normalise_input(str(row.get("serviceName") or row.get("name") or "")),
            _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or "")),
        )

    for token in _family_query_root_tokens(query_text):
        expanded_rows = _expand_family_rows_by_root_token(
            rows,
            family_query=family_query,
            root_token=token,
            limit=max(limit, 40),
        )
        if len(expanded_rows) < 2:
            continue
        # Ранжируем expansion по одному root-токену (сохраняет поведение для
        # запросов типа «анализы на витамины», где полный запрос не матчится
        # ни на одну отдельную витаминную строку сильно).
        expanded_ranked = _select_family_variant_rows(
            expanded_rows,
            family_query,
            ranking_query=token,
            limit=max(limit, 40),
        )
        # Держим уже отфильтрованные direct_rows (сильные матчи по полному
        # запросу) в голове кандидатного списка, чтобы широкая expansion по
        # одному root-токену не вытесняла их (кейс «общий анализ крови» —
        # expansion по «кровь» не должен прятать ОАК-варианты).
        merged: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for row in list(direct_filtered) + list(expanded_ranked):
            key = _row_key(row)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(row)
            if len(merged) >= limit:
                break
        candidate_rows = merged
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


_OAK_ALIAS_RE = re.compile(r"\bоак\b", re.I)
_OAK_PHRASE_RE = re.compile(r"общ\w*\s+анализ\w*\s+кров\w*", re.I)
_OAK_CANONICAL_BASE_RE = re.compile(r"^\s*общий\s+анализ\s+крови\b", re.I)
_OAK_HOMEVISIT_RE = re.compile(r"\bна\s+дом\b|выезд\s+на\s+дом", re.I)


def _is_oak_base_query(query_text: str) -> bool:
    """Проверяет, что запрос адресован именно к базовому ОАК.

    Явно запрошенные модификаторы (cito, капиллярная, детский) не считаем
    базовым ОАК — там стандартный family/ranking путь даст нужный вариант.

    :param query_text: исходный запрос пользователя
    :return: True, если речь именно про обычный общий анализ крови
    """

    raw = _normalise_input(str(query_text or ""))
    if not raw:
        return False
    if _query_price_variant_flags(raw):
        return False
    if _OAK_ALIAS_RE.search(raw):
        return True
    return bool(_OAK_PHRASE_RE.search(raw))


def _select_oak_canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Возвращает канонические ОАК-строки без служебных модификаторов.

    Канонические — те, что начинаются с «Общий анализ крови», без cito,
    капиллярной крови, детских вариантов и выездов на дом. На реальном
    прайсе это ровно две позиции (390 и 490 руб).

    :param rows: строки прайса
    :return: канонические ОАК-строки, отсортированные по стоимости
    """

    canonical: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        if not name or not _OAK_CANONICAL_BASE_RE.search(name):
            continue
        if _lab_price_variant_flags(row):
            continue
        if _OAK_HOMEVISIT_RE.search(name):
            continue
        canonical.append(row)

    canonical.sort(key=lambda r: _as_int(r.get("cost")) or 0)
    return canonical


def _build_oak_canonical_payload(
    query_text: str,
    rows: list[dict[str, Any]],
    *,
    show_all: bool = False,
    visible_limit: int = 2,
) -> dict[str, Any] | None:
    """Формирует payload для ОАК: по умолчанию две базовые позиции, остальное — по «все».

    :param query_text: исходный запрос пользователя
    :param rows: строки прайса
    :param show_all: разворачивать ли весь список
    :param visible_limit: сколько позиций показывать в первом ответе
    :return: family-query-совместимый payload либо None, если канонических <2
    """

    canonical = _select_oak_canonical_rows(rows)
    if len(canonical) < 2:
        return None

    canonical_keys = {
        (
            _normalise_input(str(row.get("serviceName") or row.get("name") or "")),
            _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or "")),
        )
        for row in canonical
    }

    extras: list[dict[str, Any]] = []
    extra_source = _rank_price_rows(rows, "общий анализ крови", limit=60)
    for row in extra_source:
        if not isinstance(row, dict):
            continue
        key = (
            _normalise_input(str(row.get("serviceName") or row.get("name") or "")),
            _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or "")),
        )
        if key in canonical_keys:
            continue
        extras.append(row)

    variants = _annotate_price_rows_with_care_context(canonical + extras)
    visible_count = len(variants) if show_all else min(len(variants), visible_limit)
    remaining_count = max(0, len(variants) - visible_count)
    show_all_hint = ""
    if remaining_count > 0 and not show_all:
        show_all_hint = (
            f"По вашему запросу найдено еще {remaining_count} вариантов. "
            'Чтобы показать их, напишите: "все".'
        )

    return {
        "service_name": "общий анализ крови",
        "service_kind": "family_query",
        "family_variants": variants,
        "showing_all": show_all,
        "visible_limit": visible_limit,
        "remaining_count": remaining_count,
        "show_all_hint": show_all_hint,
        "note": "price_oak_canonical",
    }


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

    if _is_oak_base_query(query_text):
        oak_payload = _build_oak_canonical_payload(
            query_text,
            rows,
            show_all=show_all,
            visible_limit=2,
        )
        if oak_payload is not None:
            return oak_payload

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

    # Адреса оказания услуги резолвятся внутри resolve_price_unit_context:
    # сначала по priceUnit-override'ам (PRICE_UNIT_ADDRESSES), затем fallback
    # на care-setting root-level хардкод. doctor_prices больше не используется
    # как источник адресов — regionName в нём указывает филиал приёма врача,
    # а не место проведения процедуры (подтверждено разработчиком Наяки).
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

    Дедуп по нормализованному ключу (без учёта пробелов и регистра), чтобы
    одинаковые адреса с разной пунктуацией ("г. Самара, ул. Ново-Садовая…"
    vs "г.Самара, ул.Ново-Садовая…") схлопывались в один пункт выдачи.

    :param rows: строки прайса с полем care_setting_address
    :return: список уникальных адресов в порядке появления
    """
    addresses: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        candidate_str = str(candidate or "").strip()
        if not candidate_str:
            return
        key = _normalise_input(candidate_str)
        if not key or key in seen:
            return
        seen.add(key)
        addresses.append(candidate_str)

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_addresses = row.get("care_setting_addresses")
        if isinstance(row_addresses, list) and row_addresses:
            for candidate in row_addresses:
                _add(candidate)
            continue
        address = str(row.get("care_setting_address") or "").strip()
        if not address:
            continue
        # Допускаем старый форматированный вид "A; B" от внешних источников.
        for part in address.split(";"):
            _add(part)
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


_MULTI_PRICE_SPLIT_RE = re.compile(r"\s*(?:,|;|\bи\b|\+|/)\s*", re.I)
_MULTI_PRICE_SERVICE_HINT_RE = re.compile(
    r"[a-zа-яё]{3,}",
    re.I,
)


def _split_price_query_items(query_text: str) -> list[str]:
    """Делит мульти-услуговый price-запрос на отдельные фрагменты-услуги.

    Консервативный сплиттер: режем по `,` / `;` / ` и ` / `+` / `/`, отбрасываем
    служебные стоп-слова («стоимость», «цена» и т.п.) из каждого фрагмента.

    :param query_text: исходный текст пользователя
    :return: список непустых фрагментов услуг; пустой список, если делить нечего
    """

    raw = str(query_text or "").strip()
    if not raw:
        return []

    head = re.sub(
        r"^\s*(?:стоимость|цена|сколько\s+стоит|прайс)\s+",
        "",
        raw,
        flags=re.I,
    ).strip()
    if not head:
        head = raw

    fragments: list[str] = []
    for chunk in _MULTI_PRICE_SPLIT_RE.split(head):
        frag = chunk.strip(" ?!.,;:-–—")
        if not frag:
            continue
        if not _MULTI_PRICE_SERVICE_HINT_RE.search(frag):
            continue
        if _normalise_input(frag) in _PRICE_QUERY_STOPWORDS:
            continue
        fragments.append(frag)

    seen: set[str] = set()
    unique: list[str] = []
    for frag in fragments:
        key = _normalise_input(frag)
        if key in seen:
            continue
        seen.add(key)
        unique.append(frag)
    return unique


def _resolve_multi_price_items(
    query_text: str,
    retail_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Разрешает фрагменты мульти-услугового запроса в реальные услуги каталога.

    Возвращает список словарей `{"service_name", "prices"}` только если ≥2
    фрагментов успешно приземлились на каталог. Иначе — пустой список
    (fallback на стандартный single-service путь).

    :param query_text: исходный запрос пользователя
    :param retail_rows: строки retail-прайса региона
    :return: список разрешённых услуг с top-рядами цен
    """

    fragments = _split_price_query_items(query_text)
    if len(fragments) < 2:
        return []

    resolved: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for frag in fragments:
        service_name = resolve_price_service_name_from_catalog(frag, rows=retail_rows)
        if not service_name:
            extracted = _extract_price_service_from_query(frag) or frag
            alias_variants = _PRICE_SERVICE_ALIASES.get(_normalise_input(extracted), ())
            service_name = str(alias_variants[0] or "").strip() if alias_variants else ""
        if not service_name:
            continue
        key = _normalise_input(service_name)
        if key in seen_names:
            continue
        top_rows = _select_patient_price_rows(retail_rows, service_name, limit=2)
        if not top_rows:
            continue
        seen_names.add(key)
        resolved.append(
            {
                "service_name": service_name,
                "prices": _annotate_price_rows_with_care_context(top_rows),
            }
        )

    if len(resolved) < 2:
        return []
    return resolved


def _build_multi_price_payload(
    query_text: str,
    retail_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Формирует payload для мульти-услугового price-запроса.

    Каждая услуга отрисовывается как отдельная family-подгруппа через
    `care_setting_label`, чтобы рендер умел группировать список без
    дополнительного рендер-кейса.

    :param query_text: исходный запрос пользователя
    :param retail_rows: строки retail-прайса региона
    :return: family-query-совместимый payload либо None
    """

    items = _resolve_multi_price_items(query_text, retail_rows)
    if not items:
        return None

    variants: list[dict[str, Any]] = []
    for item in items:
        label = f"По услуге «{item['service_name']}»"
        for row in item["prices"]:
            annotated = dict(row)
            annotated["care_setting_label"] = label
            annotated["care_setting_address"] = ""
            variants.append(annotated)

    if len(variants) < 2:
        return None

    service_name = ", ".join(str(item["service_name"]) for item in items)
    visible_limit = len(variants)
    return {
        "service_name": service_name,
        "service_kind": "family_query",
        "family_variants": variants,
        "showing_all": True,
        "visible_limit": visible_limit,
        "remaining_count": 0,
        "show_all_hint": "",
        "note": "price_multi_service",
    }


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

    query_norm = _normalise_input(str(query_text or ""))
    primary_norm = _normalise_input(str(primary_service_name or ""))
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
        alias_norm = _normalise_input(alias)
        if not alias_norm or alias_norm in primary_norm or alias_norm not in query_norm:
            continue
        fragments.append(alias)

    seen: set[str] = set()
    for idx, fragment in enumerate(fragments):
        key = _normalise_input(fragment)
        if not key or key in seen:
            continue
        seen.add(key)
        candidate = resolve_price_service_name_from_catalog(fragment, rows=retail_rows)
        if not candidate:
            variants = _PRICE_SERVICE_ALIASES.get(key, ())
            candidate = str(variants[0] or "").strip() if variants else ""
        candidate_norm = _normalise_input(candidate)
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

    reply_norm = _normalise_input(str(user_text or ""))
    if not reply_norm:
        return None
    reply_tokens = set(_meaningful_price_service_tokens(reply_norm))

    alias_hits: set[str] = set()
    for alias, variants in _PRICE_SERVICE_ALIASES.items():
        alias_norm = _normalise_input(alias)
        if alias_norm and alias_norm in reply_norm:
            alias_hits.add(alias_norm)
            for variant in variants:
                variant_norm = _normalise_input(variant)
                if variant_norm:
                    alias_hits.add(variant_norm)

    for option in options:
        option_text = str(option or "").strip()
        option_norm = _normalise_input(option_text)
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
        raw = await llm_runtime_mod.generate_text(
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
