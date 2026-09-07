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
from ._biomaterial import (
    mis_synonym_service,
    service_accepts_biomaterial,
    split_biomaterial_tail,
)
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
_PRICE_CITO_QUERY_RE = re.compile(r"\b(cito|сроч\w*)\b", re.I)
_PRICE_CAPILLARY_QUERY_RE = re.compile(r"\b(капилляр\w*|из\s+пальца|палец)\b", re.I)
_PRICE_CHILD_QUERY_RE = re.compile(r"\b(детск\w*|детям|детей|ребен\w*|ребён\w*)\b", re.I)
_PRICE_REPEAT_QUERY_RE = re.compile(r"\b(повторн\w*|повтор)\b", re.I)
_PRICE_KMN_QUERY_RE = re.compile(r"\b(к\.?\s*м\.?\s*н\.?|кандидат\w*\s+медицин\w*\s+наук)\b", re.I)
_PRICE_HOME_QUERY_RE = re.compile(r"\b(на\s+дому|домой|выезд\w*\s+на\s+дом)\b", re.I)
_PRICE_PACKAGE_QUERY_RE = re.compile(r"\b(совместно|комплекс\w*|пакет\w*|программ\w*|combo|комбо|с\s+узи)\b", re.I)
_PRICE_GENETIC_QUERY_RE = re.compile(r"\b(ген(?![ти])\w*|мутац\w*|полиморф\w*|vdr)\b", re.I)
_PRICE_CITO_ROW_RE = re.compile(r"\b(cito|сроч\w*)\b", re.I)
_PRICE_CAPILLARY_ROW_RE = re.compile(r"\bкапилляр\w*\b", re.I)
_PRICE_CHILD_ROW_RE = re.compile(r"\b(детск\w*|детям|детей|ребен\w*|ребён\w*)\b", re.I)
_PRICE_REPEAT_ROW_RE = re.compile(r"\b(повторн\w*|повтор)\b", re.I)
_PRICE_KMN_ROW_RE = re.compile(r"\b(к\.?\s*м\.?\s*н\.?|кандидат\w*\s+медицин\w*\s+наук)\b", re.I)
_PRICE_HOME_ROW_RE = re.compile(r"\b(на\s+дому|домой|выезд\w*\s+на\s+дом)\b", re.I)
_PRICE_PACKAGE_ROW_RE = re.compile(r"\b(совместно|комплекс\w*|пакет\w*|программ\w*|combo|комбо|регулярн\w*)\b", re.I)
_PRICE_GENETIC_ROW_RE = re.compile(r"\b(ген(?![ти])\w*|мутац\w*|полиморф\w*|vdr)\b", re.I)
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
    # Падежные формы «стоимость» — без них не отбрасывалось «стоимостью»
    # / «стоимости» в формулировках вроде «о стоимости АЛТ».
    "стоимости",
    "стоимостью",
    "стоимостей",
    "цена",
    # Падежные формы «цена» — без них «подскажите цену АЛТ» оставлял
    # «цену» в токенах, ранкер искал строку с обоими «цену» и «алт»
    # и не находил. Жалоба заказчика 2026-05-05.
    "цену",
    "цены",
    "ценой",
    "цене",
    "ценах",
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
    # Недостающие словоформы уже перечисленных выше разговорных слов
    # («нужно/надо», «возможно», «стоит»). Без них гард неудовлетворимого
    # уточнения принимал «нужен»/«возможна»/«стоить» за значащее уточнение и
    # блокировал живые запросы («мне НУЖЕН анализ на ферритин»). Тот же принцип,
    # что у падежных форм «цена»/«стоимость» выше.
    "нужен",
    "нужна",
    "нужны",
    "возможен",
    "возможна",
    "возможны",
    "стоить",
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

# Курируемые синонимы «слово пациента → термин каталога», применяются к ТЕКСТУ запроса
# ПЕРЕД лексическим матчем (сохраняя контекст-уточнения вроде «из вены»). Узкий
# fast-path для частых расхождений формулировок; общий случай (любой синоним/аббревиатура/
# опечатка) — LLM-нормализация (вариант B). Затрагивает ТОЛЬКО запросы с этими словами,
# остальные не меняются. ReDoS-safe (литерал + ограниченный `\w*`).
#   • «забор крови/материала» → «взятие …» (каталог использует «Взятие …»). BUG-2026-06-23-01.
#   • «электромиография» → аббревиатура каталога «ЭМГ» (иначе цеплялось чужое «Электро…
#     коагуляция» по общему префиксу «электро»). BUG-2026-06-23-01.
_SERVICE_SYNONYM_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bзабор\w*", re.IGNORECASE), "взятие"),
    (re.compile(r"\bэлектромиограф\w*", re.IGNORECASE), "ЭМГ"),
)


def _apply_service_synonyms(query_text: str) -> str:
    """Нормализует пациентские синонимы к терминам каталога перед лексическим матчем."""
    out = str(query_text or "")
    for pattern, repl in _SERVICE_SYNONYM_SUBS:
        out = pattern.sub(repl, out)
    return out
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
# Homoglyph folding (Cyrillic→Latin) — для сравнения lab-кодов
#
# Применяется ТОЛЬКО к токенам, содержащим цифру (признак lab-кода),
# как ДОПОЛНИТЕЛЬНЫЙ кандидат рядом с оригиналом. Это позволяет
# «Са-125» → «Ca-125» совпасть с «CA - 125» без риска, что обычные
# русские слова («сахар», «рак») ложно матчатся на латинские коды.
# Символы-разделители (пробел, дефис) нормализуются в пустую строку
# для нечувствительности к форматированию пробелов («CA - 125» ~ «CA-125»).
# ---------------------------------------------------------------------------

_PRICE_CYR_TO_LAT: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "к": "k", "в": "b", "н": "h",
    "м": "m", "т": "t",
}


def _fold_homoglyph_token(token: str) -> str | None:
    """Fold Cyrillic homoglyphs to Latin for a single token (only if it contains a digit).

    Returns folded string if the token contains a digit AND has at least one
    homoglyph substitution, otherwise returns None (no extra candidate needed).
    Only applied to short alphanumeric tokens to avoid corrupting Russian words.

    :param token: нормализованный (lowercase) токен
    :return: сложенный кандидат или None
    """
    if not token or not any(ch.isdigit() for ch in token):
        return None
    folded = "".join(_PRICE_CYR_TO_LAT.get(ch, ch) for ch in token)
    return folded if folded != token else None


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


# BUG-D: turnaround-фразировка («срок готовности», «сколько делается», «когда будет
# готов», «за сколько дней», «как долго делается») — ШУМ вокруг названия анализа.
# Эти токены ломают лексический матч по каталогу (запрос «срок готовности ОАК» не
# извлекал test_name → весь текст шёл в матчер → 0 совпадений). Снимаем фразу перед
# матчингом, чтобы анализ нашёлся и его `deadline` дошёл до пациента. НЕ путать с
# TEST_RESULT («результаты готовы») и PREPARE («как подготовиться»).
_READINESS_TIME_RE = re.compile(
    r"\b("
    r"сроки?\s+готовности(?:\s+(?:выполнения|изготовления|анализа|услуги))?"
    r"|сроки?\s+выполнения"
    r"|когда\s+(?:будет\s+)?готов(?:а|о|ы)?"
    r"|за\s+сколько\s+(?:дней|времени)?\s*(?:дела\w+|готов\w+)?"
    r"|через\s+сколько\s+(?:дней\s+)?(?:будет\s+)?готов\w*"
    r"|сколько\s+(?:по\s+времени\s+)?(?:дней\s+)?(?:дела\w+|готов\w+|ждать)"
    r"|как\s+(?:быстро|долго|скоро)\s+(?:дела\w+|готов\w+)"
    r")\b",
    re.IGNORECASE,
)

_READINESS_LEADING_FILLER = {
    "анализ", "анализа", "анализы", "анализов", "на", "для", "по",
    "услуга", "услуги", "услугу",
}


def strip_readiness_phrasing(text: str) -> tuple[str, bool]:
    """Снимает turnaround-фразировку («срок готовности»/«сколько делается») с запроса.

    Такие слова — ШУМ вокруг названия анализа: они ломают лексический матч по
    каталогу. Возвращает ``(очищенное_название, is_readiness)``. При
    ``is_readiness=True`` найденный анализ несёт ``deadline``, который renderer
    транслирует пациенту (промпт уже инструктирует показывать «сроки готовности …
    буквальные значения из Данные»). Обычные запросы возвращаются без изменений.

    :param text: исходный запрос пользователя
    :return: (текст без turnaround-фразы, был ли это запрос про срок готовности)
    """
    raw = str(text or "")
    if not _READINESS_TIME_RE.search(raw):
        return raw, False
    cleaned = _READINESS_TIME_RE.sub(" ", raw)
    tokens = cleaned.split()
    _EDGE = "?!.,:;»«\"'()-"
    while tokens and tokens[0].strip(_EDGE).lower() in _READINESS_LEADING_FILLER:
        tokens.pop(0)
    return " ".join(tokens).strip(" " + _EDGE), True


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
        # Homoglyph-folded EXTRA candidate: for tokens containing a digit (lab codes
        # like «са» from «са-125»), also add the Latin-folded version so that
        # Cyrillic-typed «Са-125» can match catalog Latin «CA - 125» via the folded
        # token «ca». Applied only to digit-bearing tokens to avoid corrupting normal
        # Russian words. The original token is kept; folded is an additive extra.
        folded = _fold_homoglyph_token(token)
        if folded and folded not in out:
            out.append(folded)
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
    # Также пробуем каждый токен по отдельности, если он есть в карте
    # `_PRICE_SERVICE_ALIASES`. Это нужно для запросов типа
    # «оак срочно», «оам срочно», «лпвп натощак», где аббревиатура
    # стоит рядом с модификатором: full-string match не сработает,
    # а токенный — да. Жалоба заказчика 2026-05-05: «ОАК срочно»
    # возвращал None.
    for tok in _price_query_tokens(raw):
        if tok in _PRICE_SERVICE_ALIASES:
            out.append(tok)
    dedup: list[str] = []
    seen: set[str] = set()
    for cand in out:
        key = str(cand or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return dedup[:8]


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


# ---------------------------------------------------------------------------
# Ответ ДАТОЙ/ВРЕМЕНЕМ — не запрос услуги
#
# Класс дефекта `datetime_answer_as_service` (SEVERE, дезинформация ценой).
# На шаге APPOINTMENT бот спрашивает «На какую дату и время вам удобно?».
# Ответ пациента доходил до `_extract_price_service_from_query`, тот отдавал
# его как КАНДИДАТА В УСЛУГИ («6.09.2026» → «6 09 2026»), а `_rank_price_rows`
# матчил цифры на `serviceHomecode` каталога:
#     «6.09.2026» → homecode 2026 → «Индейка (F284)»          →    850 руб.
#     «10:30»     → «Абонемент на 10 сеансов массажа»         → 21 250 руб.
# Пациенту называлась цена ЧУЖОЙ услуги вместо цены приёма врача
# (прод, диалог #1109: «записать на приём к ортопед ... стоимость 850 руб»
# при реальной цене приёма 2700 руб).
#
# Гард СТРУКТУРНЫЙ, не словарный: у даты и времени есть форма («д.м.гггг»,
# «чч:мм»), которой нет у кода услуги («Код 5437» — целое число), поэтому
# homecode-запросы он не задевает. Срабатывает, только если КРОМЕ даты/времени
# в реплике не осталось ни одного значащего слова: «сколько стоит приём
# завтра» гардом не глушится.
# ---------------------------------------------------------------------------

_DATETIME_SHAPED_RE = re.compile(
    r"(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"                    # ISO: 2026-09-06
    r"|\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"     # 6.09.2026, 6/9, 06-09-26
    r"|\d{1,2}[:.]\d{2}"                         # 10:30, 14.00
    r")"
)
_MONTH_WORD_RE = re.compile(
    r"\b(?:январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]\w*|июн\w*|июл\w*|"
    r"август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\b"
)
_RELATIVE_DAY_RE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|вчера|утром|утра|днем|вечером|вечера|"
    r"ночью|понедельник\w*|вторник\w*|сред[ауы]|четверг\w*|пятниц\w*|"
    r"суббот\w*|воскресен\w*)\b"
)
# Словами о часах/минутах — только точные формы: `час\w*` поймал бы «часть».
_CLOCK_WORD_RE = re.compile(r"\b(?:часов|часа|часу|час|минут\w*|мин)\b")
# Служебная обвязка даты: предлоги и слова-указатели. Триггером НЕ являются.
_DATETIME_FILLER_RE = re.compile(
    r"\b(?:в|во|на|к|ко|с|со|до|про|около|числа|числу|число|время|дата|дату)\b"
)

_DATETIME_TRIGGER_RES = (
    _DATETIME_SHAPED_RE,
    _MONTH_WORD_RE,
    _RELATIVE_DAY_RE,
    _CLOCK_WORD_RE,
)


def _is_datetime_only_query(text: str) -> bool:
    """
    Определяет, что реплика состоит только из даты/времени.

    :param text: сырой текст реплики пациента
    :return: True — реплику нельзя трактовать как название услуги
    """

    s = _normalise_input(text).strip()
    if not s:
        return False
    if not any(rx.search(s) for rx in _DATETIME_TRIGGER_RES):
        return False
    rest = s
    for rx in _DATETIME_TRIGGER_RES:
        rest = rx.sub(" ", rest)
    rest = _DATETIME_FILLER_RE.sub(" ", rest)
    # Голый остаток из цифр («20 ноября 2026» → «20 2026») услугу не уточняет.
    return not re.findall(r"[a-zа-я]{3,}", rest)


def _price_raw_query_fallback(query_text: str) -> str:
    """
    Сырой текст запроса как fallback-имя услуги.

    Отдаёт пустую строку для реплик-дат/времени: иначе `or query_text` ниже
    возвращает то самое, что отсёк `_extract_price_service_from_query`, и
    дата снова уезжает в матчинг по каталогу (см. `datetime_answer_as_service`).

    :param query_text: исходный пользовательский запрос
    :return: текст для family-матчинга либо пустая строка
    """

    return "" if _is_datetime_only_query(query_text) else str(query_text or "")


def _extract_price_service_from_query(query: str) -> str | None:
    raw = str(query or "").strip()
    if not raw:
        return None
    # Ответ датой/временем на вопрос о записи — не название услуги.
    if _is_datetime_only_query(raw):
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


# BUG-H: «анализ/анализы/анализов» как framing-слово price-запроса («цены на
# анализы: …», «анализы на …»). В отличие от «УЗИ»/«орган» оно почти никогда не
# различает конкретную услугу, зато как лишний токен роняет score нужной услуги
# ниже порога. Снимаем его ОТДЕЛЬНЫМ кандидатом (аддитивно — услуги с «анализ» в
# названии сохраняют исходный вариант).
_ANALYSIS_FRAMING_RE = re.compile(r"\bанализ\w*\b", re.IGNORECASE)


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
        framing_stripped = re.sub(r"\s+", " ", _ANALYSIS_FRAMING_RE.sub(" ", extracted)).strip()
        if framing_stripped and framing_stripped != extracted:
            queries.append(framing_stripped)
    phrase = extract_service_phrase(raw)
    if phrase:
        queries.append(phrase)
    compact = " ".join(_price_query_tokens(raw)).strip()
    if compact and _has_specific_price_tokens(compact):
        queries.append(compact)
    if raw and _has_specific_price_tokens(raw):
        queries.append(raw)
    # BUG-H: голое generic/интент-слово («анализы», «узи») — не услуга. Кандидат
    # без единого РАЗЛИЧАЮЩЕГО токена резолвится в произвольную строку каталога
    # («анализы» → «Общий анализ мочи») и перебивает конкретный запрос. Отсекаем
    # такие кандидаты; если различающих не осталось → пусто → честное уточнение.
    distinctive_queries = [q for q in queries if _distinctive_service_tokens(q)]
    return _dedupe_price_queries(distinctive_queries)


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


def _alias_overrides_explicit_primary(
    alias_hit: str,
    current_service_name: str,
    query_text: str,
) -> bool:
    """
    Проверяет, перебивает ли alias-хит явную primary-услугу в compound-запросе.

    Catalog-alias резолвер возвращает первую попавшуюся услугу по алиасу и может
    схватить вторичную лабораторную позицию из запроса вида
    «<primary> и сдать кровь на <lab>», перекрыв уже выбранную primary-услугу.
    Если primary (``current_service_name``) явно задана, отличается от алиаса и её
    токены присутствуют в этой реплике — это compound, и alias не должен делать
    short-circuit: пусть scored-matcher ниже корректно ранжирует primary из текста.
    Для честного topic-switch («а сколько стоит ЛПНП?») и synonym-запросов primary
    в реплике отсутствует, поэтому alias сохраняется.

    :param alias_hit: услуга, к которой привёл alias-резолвер
    :param current_service_name: уже выбранная primary-услуга из state
    :param query_text: исходный текст пользователя
    :return: True, если alias нужно подавить (compound с присутствующей primary)
    """

    current_norm = _normalise_input(current_service_name)
    if not current_norm or current_norm == _normalise_input(alias_hit):
        return False
    primary_tokens = _price_query_tokens(current_service_name)
    if not primary_tokens:
        return False
    query_tokens = set(_price_query_tokens(query_text))
    return all(token in query_tokens for token in primary_tokens)


# Generic-стволы УЗИ/процедур, которые НЕ различают конкретную услугу. Для
# проверки доверия скорер-матчу важны именно РАЗЛИЧАЮЩИЕ токены (орган/анализ),
# а не общее «узи»/«исследование».
_GENERIC_SERVICE_PREFIXES = ("узи", "ультразвук", "исследов", "орган", "анализ")


def _distinctive_service_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for t in _price_query_tokens(text):
        if len(t) < 3:
            continue
        if any(t.startswith(g) or g.startswith(t) for g in _GENERIC_SERVICE_PREFIXES):
            continue
        out.add(t)
    return out


def _token_covered_by(token: str, pool: set[str] | frozenset[str]) -> bool:
    """True, если токен присутствует в наборе точно или префиксом (4 символа).

    Префиксное сравнение — то же правило, которым `_scorer_match_has_distinctive_overlap`
    сопоставляет словоформы («кардиолога» ↔ «кардиолог»); вынесено, чтобы гарды
    судили о покрытии одинаково.
    """
    if token in pool:
        return True
    if len(token) < 4:
        return False
    prefix = token[:4]
    return any(len(candidate) >= 4 and candidate[:4] == prefix for candidate in pool)


def _scorer_match_has_distinctive_overlap(query_text: str, canonical: str) -> bool:
    """True, если у скорер-кандидата есть пересечение РАЗЛИЧАЮЩИХ токенов с запросом.

    Защита от ложной подмены услуги (BUG-2026-06-02-07): «УЗИ органов пищеварения»
    скорер тянет на «УЗИ органов мошонки» по общему «узи органов», игнорируя
    различающее «пищеварения». Если ни один различающий токен запроса не встречается
    в каноне — это подмена, скорер-результату доверять нельзя. Alias-путь сюда не
    попадает (он отрабатывает раньше), поэтому аббревиатуры (ОАК и т.п.) не задеты.
    """
    q = _distinctive_service_tokens(query_text)
    if not q:
        return True  # нет различающих токенов — судить не о чем, не блокируем
    canon_tokens = set(re.findall(r"[a-zа-яё0-9]+", _normalise_input(canonical)))
    return any(_token_covered_by(qt, canon_tokens) for qt in q)


# ---------------------------------------------------------------------------
# Неудовлетворимое уточнение (класс `unsatisfiable_qualifier_dropped`)
#
# Прод 13.08: «А возможна ли онлайн консультация?» → бот отвечал про обычный
# приём. Проверено офлайн — уточнение отбрасывалось у 5 из 10 фраз: «онлайн»,
# «по ОМС», «по телефону», «удалённо», «по видеосвязи». Опаснее всех ОМС:
# пациент спрашивает про полис, получает платную цену.
#
# `_scorer_match_has_distinctive_overlap` не спасает — ему достаточно ОДНОГО
# совпавшего токена («консультация»); отсутствие «онлайн» он не проверяет.
#
# Механизм CATALOG-DERIVED (как у family-дискриминатора, без списков в коде):
# словарь допустимых токенов = сам прайс. Если различающий токен запроса не
# покрыт найденной услугой И не встречается НИ В ОДНОЙ строке каталога — такое
# свойство клиника вообще не предоставляет, подставлять услугу без него нельзя.
# Токен, каталогу известный, гард НЕ трогает: различать варианты внутри каталога
# — работа family-дискриминатора и scorer-overlap, не эта.
# ---------------------------------------------------------------------------

_CATALOG_VOCAB_CACHE: dict[tuple[int, str, str], frozenset[str]] = {}
# Минимальный размер среза, на котором утверждение «такого слова в прайсе нет»
# осмысленно. Живой прайс Самары — ~3700 строк; синтетические тестовые срезы —
# единицы строк, там гард обязан молчать.
_CATALOG_VOCAB_MIN_ROWS = 100


def _catalog_token_vocabulary(rows: list[dict[str, Any]]) -> frozenset[str]:
    """Множество всех токенов названий услуг каталога.

    Кэшируется по «отпечатку» среза (размер + первое/последнее имя): каталог за
    ход не меняется, а строить словарь по 3600+ строкам на каждый из 14 вызовов
    резолвера было бы дорого.

    :param rows: строки priceByRegion
    :return: замороженное множество токенов
    """

    if not rows:
        return frozenset()
    key = (
        len(rows),
        str(rows[0].get("serviceName") or rows[0].get("name") or "")[:48],
        str(rows[-1].get("serviceName") or rows[-1].get("name") or "")[:48],
    )
    cached = _CATALOG_VOCAB_CACHE.get(key)
    if cached is not None:
        return cached
    vocab: set[str] = set()
    for row in rows:
        name = str(row.get("serviceName") or row.get("name") or "")
        if name:
            # ТОТ ЖЕ токенайзер, что и на стороне запроса: иначе несимметричные
            # производные (гомоглиф-фолдинг кодов «с91»→«c91», синонимы
            # «прием»↔«консультац») выглядели бы как «каталогу неизвестно» и гард
            # срабатывал бы на самих названиях каталога.
            vocab.update(_price_query_tokens(name))
            vocab.update(re.findall(r"[a-zа-яё0-9]+", _normalise_input(name)))
        # Homecode — тоже «слово каталога»: «Код 5437» это валидный запрос
        # услуги по коду, а не неудовлетворимое уточнение.
        homecode = str(row.get("serviceHomecode") or row.get("homecode") or "")
        if homecode:
            vocab.update(re.findall(r"[a-zа-яё0-9]+", _normalise_input(homecode)))
    frozen = frozenset(vocab)
    if len(_CATALOG_VOCAB_CACHE) >= 4:
        _CATALOG_VOCAB_CACHE.clear()
    _CATALOG_VOCAB_CACHE[key] = frozen
    return frozen


def query_has_unsatisfiable_qualifier(
    query_text: str,
    rows: list[dict[str, Any]] | None = None,
) -> bool:
    """Публичный гард: в реплике есть уточнение, которого каталог не знает вовсе.

    Нужен на стыках, где нет «найденной услуги», но есть fallback на сырую фразу
    из запроса: без него отказ single-резолвера обходился
    `_extract_price_service_from_query` и пациент всё равно получал цену услуги
    без запрошенного свойства («приём терапевта по ОМС» → платный приём).

    :param query_text: исходный текст пользователя
    :param rows: строки priceByRegion (если не переданы — грузим сами)
    :return: True, если уточнение неудовлетворимо каталогом
    """

    catalog_rows = rows
    if catalog_rows is None:
        try:
            loaded = api_price.load_price_by_region(SAMARA_PRICE_REGION_ID)
        except Exception:
            return False
        catalog_rows = [row for row in loaded if isinstance(row, dict)]
    return _match_drops_unsatisfiable_qualifier(query_text, "", catalog_rows)


def _match_drops_unsatisfiable_qualifier(
    query_text: str,
    canonical: str,
    rows: list[dict[str, Any]],
) -> bool:
    """True, если запрос несёт уточнение, которого каталог не может удовлетворить.

    :param query_text: вариант запроса, на котором состоялось совпадение
    :param canonical: найденное название услуги
    :param rows: строки priceByRegion (источник словаря допустимых токенов)
    :return: True — совпадение засчитывать нельзя (уточнение потеряно)
    """

    # Гард утверждает «клиника такого свойства НЕ оказывает — во всём прайсе
    # этого слова нет». Утверждение опирается на ПОЛНЫЙ прайс: на синтетическом
    # срезе из пары строк словарь — не словарь клиники, и любой обычный запрос
    # выглядел бы как неудовлетворимый. Ниже порога молчим.
    if len(rows) < _CATALOG_VOCAB_MIN_ROWS:
        return False
    tokens = _distinctive_service_tokens(query_text)
    if not tokens:
        return False
    # Запрос услуги ПО КОДУ («Код 5437», «услуга 5437») резолвится по homecode,
    # а не по названию: обрамляющие слова тут не уточнение услуги.
    if _extract_homecode_query(query_text):
        return False
    canon_tokens = set(re.findall(r"[a-zа-яё0-9]+", _normalise_input(canonical)))
    vocab = _catalog_token_vocabulary(rows)
    if not vocab:
        return False
    for token in tokens:
        # Против НАЙДЕННОЙ услуги сравниваем с префиксным допуском — там речь о
        # словоформах одного слова («кардиолога» ↔ «кардиолог»).
        if _token_covered_by(token, canon_tokens):
            continue
        # Против ВСЕГО каталога — только точное совпадение. Префиксный допуск
        # здесь ложно «оправдывает» уточнение чужим словом с тем же началом:
        # «УДАЛённая консультация» ← «УДАЛение», «ВИДЕосвязи» ← «ВИДЕокольпоскопия»,
        # «ТЕЛЕфону» ← «ТЕЛЕмедицина». Токенайзер по обе стороны один и тот же,
        # поэтому словоформы каталога уже нормализованы симметрично.
        if token in vocab:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Различающий токен СЕМЕЙСТВА анализов (гепатит С/В/А, и т.п.)
#
# Класс дефекта `family_discriminator_dropped` (SEVERE, дезинформация ценой):
# 1-символьная буква-дискриминатор («гепатит С» → «с») отбрасывалась
# `_price_query_tokens` (len<2), запрос схлопывался в ствол → приземлялся на
# ПЕРВУЮ строку семейства («Гепатит В - HBsAg» 350₽) — пациент получал цену
# ЧУЖОГО гепатита. Единый гард на двух catalog-aware стыках (single-резолвер +
# family-builder) вместо правки глобального токенизатора: узкий триггер,
# минимальный blast radius.
#
# Механизм CATALOG-DERIVED — без списка болезней в коде: «семейство» = ствол
# ≥6 симв., у которого в самом прайсе ≥2 РАЗНЫХ буквенных варианта в позиции
# сразу после ствола (Гепатит А/В/С/D/E). Работает для любого такого семейства.
# ---------------------------------------------------------------------------

_FAMILY_LETTER_FOLD = {
    "а": "a", "в": "b", "с": "c", "д": "d", "е": "e", "к": "k", "ц": "c", "г": "g",
}

# Запрос: «<ствол ≥6 симв> <ОДИНОЧНАЯ буква>», где буква — ПОСЛЕДНИЙ символ.
# Требование «последний» отсекает предлоги: «гепатит В крови» (В=предлог, за ним
# «крови») и «гепатит С суммарные» (за С — различающий токен, и так матчится
# обычным путём) → оба НЕ дискриминатор. Ровно ОДНА буква (без цифр) держит scope
# на буквенных семействах (гепатит А/В/С/D/Е); кодовые семейства (аллерген f1,
# витамин B12, тиреоид т3) сюда НЕ попадают — это отдельный трек, иные механизмы.
_FAMILY_QUERY_DISCRIMINATOR_RE = re.compile(r"\b([а-яёa-z]{6,})\s+([а-яёa-z])\s*$", re.I)


def _fold_family_letter(token: str) -> str:
    """Сворачивает буквенный дискриминатор к канону (кириллица→латиница)."""
    low = str(token or "").strip().lower()
    if not low:
        return ""
    return f"{_FAMILY_LETTER_FOLD.get(low[0], low[0])}{low[1:]}"


def _family_query_discriminator(query_text: str) -> tuple[str, str] | None:
    """``(stem6, folded_letter)`` если запрос = «<ствол≥6> <буква>» (буква —
    последний токен), иначе ``None``. Витамины идут своим (composite-token)
    путём — их пропускаем, чтобы не задвоить логику.

    :param query_text: исходный price-запрос
    :return: (6-символьный корень ствола, свёрнутая буква) либо None
    """
    if _extract_vitamin_designator(query_text):
        return None
    m = _FAMILY_QUERY_DISCRIMINATOR_RE.search(_normalise_input(query_text))
    if not m:
        return None
    return m.group(1)[:6], _fold_family_letter(m.group(2))


def _row_family_letter(name: str, stem6: str) -> str | None:
    """Буквенный дискриминатор строки прайса сразу после ствола ``stem6``.

    :param name: название услуги из каталога
    :param stem6: 6-символьный корень ствола семейства
    :return: свёрнутая буква варианта либо None, если строка не «<ствол> <буква>»
    """
    if not stem6:
        return None
    m = re.search(
        rf"\b{re.escape(stem6)}[а-яёa-z]*\s+[–—-]?\s*([а-яёa-z]\d{{0,2}})\b",
        _normalise_input(name),
    )
    return _fold_family_letter(m.group(1)) if m else None


def _family_discriminator_context(
    query_text: str, rows: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """``(stem6, letter)`` только если запрос несёт букву-дискриминатор И ствол —
    реальное семейство каталога (≥2 РАЗНЫХ буквы в позиции после ствола). Иначе
    ``None`` (обычный путь). Никакого хардкода болезней — только структура прайса.

    :param query_text: исходный price-запрос
    :param rows: строки прайса (для проверки, что ствол — семейство)
    :return: (корень ствола, искомая буква) либо None
    """
    disc = _family_query_discriminator(query_text)
    if disc is None:
        return None
    stem6, letter = disc
    letters: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_letter = _row_family_letter(str(row.get("serviceName") or row.get("name") or ""), stem6)
        if row_letter:
            letters.add(row_letter)
            if len(letters) >= 2:
                return stem6, letter
    return None


def restrict_rows_to_family_letter(
    query_text: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Оставляет из ``rows`` ТОЛЬКО строки буквы-дискриминатора запроса.

    Нужна в single/lab-ветке ``price_info`` для МАЛЫХ семейств (гепатит А/D/E —
    <3 вариантов, family-mode не триггерит), чтобы в ответ на «гепатит А» не
    всплыла чужая буква (В/С). Если запрос без дискриминатора — возвращает
    ``rows`` как есть. Если буквы нет в каталоге — вернёт пусто (→ честное
    уточнение, а не чужая цена). Catalog-derived, см. ``_family_discriminator_context``.

    :param query_text: исходный price-запрос
    :param rows: строки-кандидаты прайса
    :return: строки нужной буквы, исходный список (нет дискриминатора) либо пусто
    """
    ctx = _family_discriminator_context(query_text, rows)
    if ctx is None:
        return rows
    stem6, letter = ctx
    return [
        row
        for row in rows
        if _row_family_letter(str(row.get("serviceName") or row.get("name") or ""), stem6) == letter
    ]


def resolve_price_service_name_from_catalog(
    query_text: str,
    *,
    current_service_name: str = "",
    rows: list[dict[str, Any]] | None = None,
) -> str | None:
    """
    Приземляет пользовательский price-запрос на реальную услугу из price-каталога.

    Используется как узкий catalog-grounded слой для `PRICE`, чтобы не
    перечислять лабораторные анализы и процедуры в regex/anchors. ЕДИНАЯ точка
    (14 вызовов из 7 модулей: цены, запись, подготовка, поиск врача), поэтому
    слои МИС подключены здесь, а не в каждом вызывающем.

    Порядок слоёв: синоним МИС (высшее доверие) → лексический матч по каталогу →
    снятие биоматериала и повторный матч. LLM-нормализатор работает ПОСЛЕ, у
    вызывающего.

    :param query_text: исходный текст пользователя
    :param current_service_name: услуга из текущего state, если уже есть
    :param rows: опционально заранее загруженные строки priceByRegion
    :return: каноническое название услуги из каталога либо None
    """

    query_text = _apply_service_synonyms(query_text)
    # Слой 1 — `serviceSynonyms` МИС: прямое «слово пациента → услуга», высшее
    # доверие. Поле заполняет клиника (на срезе 14.08 пусто) — механика
    # подключена заранее и включается сама по мере наполнения справочника.
    synonym_hit = mis_synonym_service(query_text)
    if synonym_hit and _catalog_has_service_name(synonym_hit, rows):
        return synonym_hit
    core = _resolve_price_service_core(
        query_text, current_service_name=current_service_name, rows=rows
    )
    # Слой 3 — биоматериал: снять и перепроверить (см. `_biomaterial`).
    return _resolve_with_biomaterial_stripped(
        query_text,
        core,
        current_service_name=current_service_name,
        rows=rows,
    )


def _catalog_has_service_name(service_name: str, rows: list[dict[str, Any]] | None) -> bool:
    """Есть ли такая услуга в прайсе (синоним МИС может указывать на услугу вне среза)."""

    target = _normalise_input(service_name)
    if not target:
        return False
    catalog_rows = rows
    if catalog_rows is None:
        try:
            loaded = api_price.load_price_by_region(SAMARA_PRICE_REGION_ID)
        except Exception:
            return False
        catalog_rows = [row for row in loaded if isinstance(row, dict)]
    return any(
        _normalise_input(str(row.get("serviceName") or row.get("name") or "")) == target
        for row in catalog_rows
        if isinstance(row, dict)
    )


def _resolve_with_biomaterial_stripped(
    query_text: str,
    core_result: str | None,
    *,
    current_service_name: str = "",
    rows: list[dict[str, Any]] | None = None,
) -> str | None:
    """Снимает хвост-биоматериал и предпочитает результат, ПОДТВЕРЖДЁННЫЙ МИС.

    Прод 10.08: «мазок на микрофлору с поверхности задней стенки глотки» не
    матчился вовсе, а «сколько стоит мазок на микрофлору с задней стенки глотки»
    приземлялся на «Обработка задней стенки глотки … СО2 лазера» — упоминание
    материала уводило в ЧУЖУЮ услугу.

    Результат со снятым материалом принимается ТОЛЬКО если справочник МИС
    подтверждает: найденная услуга этот материал принимает. Биоматериал при этом
    НЕ идентифицирует услугу (один соскоб подходит к 44 услугам, «кровь из вены»
    — к 1103) — он только снимается и служит проверкой.

    :param query_text: исходный запрос
    :param core_result: результат обычного лексического матча (может быть None)
    :param current_service_name: услуга из state
    :param rows: строки прайса
    :return: уточнённое название услуги либо исходный результат
    """

    head, tail = split_biomaterial_tail(query_text)
    if not tail or not head:
        return core_result
    stripped = _resolve_price_service_core(
        head, current_service_name=current_service_name, rows=rows
    )
    if not stripped or stripped == core_result:
        return core_result
    # Ключевая проверка: услуга без материала должна ПРИНИМАТЬ названный материал.
    if service_accepts_biomaterial(stripped, tail) is not True:
        return core_result
    # Исходный матч сохраняем, только если он тоже принимает этот материал —
    # иначе это подмена по локативному совпадению (лазерная обработка глотки).
    if core_result and service_accepts_biomaterial(core_result, tail) is True:
        return core_result
    return stripped


def _resolve_price_service_core(
    query_text: str,
    *,
    current_service_name: str = "",
    rows: list[dict[str, Any]] | None = None,
) -> str | None:
    """Лексический матч по прайсу — исходное тело резолвера (слой 2).

    :param query_text: текст запроса (уже с применёнными синонимами)
    :param current_service_name: услуга из текущего state
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

    # Гард различающего токена семейства (family_discriminator_dropped): если
    # запрос указывает семейство+букву (гепатит С), single-резолвер НЕ вправе
    # вернуть строку с ДРУГОЙ буквой (Гепатит В) — это дезинформация ценой.
    # При несовпадении отдаём None: сработает family-mode (список только С) или
    # уточнение. `_letter_ok` пропускает всё, если дискриминатора нет.
    family_disc = _family_discriminator_context(query_text, catalog_rows)

    def _letter_ok(name: str | None) -> bool:
        if not family_disc or not name:
            return True
        stem6, want_letter = family_disc
        return _row_family_letter(name, stem6) == want_letter

    alias_hit = _resolve_price_alias_from_catalog(query_text, catalog_rows)
    if alias_hit and not _alias_overrides_explicit_primary(
        alias_hit, current_service_name, query_text
    ) and _letter_ok(alias_hit):
        return alias_hit

    prefer_query_over_context = _should_prefer_current_price_query_over_context(query_text, current_service_name)
    if prefer_query_over_context:
        text_only_queries = _build_price_catalog_queries(query_text, current_service_name="")
        if text_only_queries:
            best_row, best_score, _ = _resolve_best_price_row_from_queries(
                text_only_queries, catalog_rows
            )
            if best_row and best_score >= 100:
                canonical = str(best_row.get("serviceName") or best_row.get("name") or "").strip()
                if (
                    canonical
                    and _scorer_match_has_distinctive_overlap(query_text, canonical)
                    and _letter_ok(canonical)
                    and not _match_drops_unsatisfiable_qualifier(query_text, canonical, catalog_rows)
                ):
                    return canonical

    queries = _build_price_catalog_queries(query_text, current_service_name=current_service_name)
    if not queries:
        return None

    best_row, best_score, _ = _resolve_best_price_row_from_queries(queries, catalog_rows)

    if not best_row:
        return None
    if best_score < 100:
        return None
    canonical = str(best_row.get("serviceName") or best_row.get("name") or "").strip()
    if canonical and not _scorer_match_has_distinctive_overlap(query_text, canonical):
        return None
    if not _letter_ok(canonical):
        return None
    # Гард неудовлетворимого уточнения (класс `unsatisfiable_qualifier_dropped`).
    # Судим по СЫРОМУ тексту: варианты запроса как раз и отбрасывают уточнение
    # («приём кардиолога онлайн» → вариант «прием кардиолог»), поэтому по ним
    # потерю не увидеть. Разговорную обвязку («мне нужен», «подскажите», «хочу
    # узнать») снимает `_PRICE_QUERY_STOPWORDS` внутри токенайзера.
    if canonical and _match_drops_unsatisfiable_qualifier(query_text, canonical, catalog_rows):
        return None
    return canonical or None


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

    tokens = _price_query_tokens(query_text)
    # M4: score against the cleaned token-join, not the raw normalised text. A
    # leading stopword («стоимость ттг», «где сдать соэ») otherwise defeats the
    # exact/substring/head bonuses and pushes short abbreviations under the
    # strong-match threshold → empty result. Fall back to raw for all-stopword
    # queries so behaviour is unchanged when there is nothing to clean.
    query = " ".join(tokens) or _normalise_input(query_text)
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


def _common_prefix_len(a: str, b: str) -> int:
    """Длина общего префикса двух строк."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# Синонимы для токенов СТРОКИ прайса (расширяем row-set, НЕ query-токены —
# иначе раздувается token_count и завышается порог _is_price_match_strong).
# Пациент пишет «антитела на корь», а в прайсе — «Вирус кори Ig M/Ig G»:
# - «ig»/«иммуноглобулин» в названии == «антитела» в запросе;
# - «кори» (род. падеж в названии) == «корь» (как пишет пациент).
_PRICE_ROW_TOKEN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ig": ("антитела", "антитело", "иммуноглобулин"),
    "иммуноглобулин": ("антитела", "антитело"),
    "кори": ("корь",),
    "корь": ("кори",),
}


def _expand_row_token_synonyms(row_tokens_set: set[str]) -> set[str]:
    """Дополняет набор токенов строки прайса безопасными синонимами.

    Расширяется ТОЛЬКО row-set (для точечного membership-матча query-токенов),
    чтобы не раздувать ``len(tokens)`` запроса в ``_is_price_match_strong``.

    :param row_tokens_set: токены названия услуги
    :return: набор с добавленными синонимами
    """
    extra: set[str] = set()
    for tok in row_tokens_set:
        extra.update(_PRICE_ROW_TOKEN_SYNONYMS.get(tok, ()))
    return row_tokens_set | extra


def _price_row_score(row: dict[str, Any], *, query: str, tokens: list[str], homecode_query: str) -> tuple[int, int]:
    name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
    homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
    if not name:
        return 0, 0
    row_tokens = _augment_price_tokens(
        [_normalise_price_token(tok) for tok in _PRICE_TOKEN_RE.findall(name) if tok],
        raw_text=name,
    )
    row_tokens_set = _expand_row_token_synonyms(set(row_tokens))

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
            # Префиксный матч для длинных токенов с защитой от ложных
            # совпадений по общему медицинскому корню.
            #
            # Раньше было достаточно 5-символьного префикса. Это давало
            # ложные срабатывания между разными процедурами с одним
            # греческим корнем — заказчик 2026-05-05: запрос
            # «кольпоскопия» (диагностика) находил «кольпорафию»,
            # «кольпоперинеорафию», «реконструкцию тазового дна
            # +кольпорафию» — это совсем другие хирургические
            # вмешательства, общий корень «кольпо-» (влагалище)
            # не означает взаимозаменяемости.
            #
            # Новое правило: матч засчитывается только если хотя бы
            # один из токенов «доезжает» до общего префикса с разницей
            # не более 2 символов в хвосте. Это пропускает варианты
            # «кольпоскопия / кольпоскопическая», «маммограф /
            # маммография», «эндоскоп / эндоскопия», но отсекает
            # «кольпоскопия / кольпорафия» (расхождение 6 vs 5
            # символов).
            if len(tok) < 5:
                continue
            for rt in row_tokens:
                if len(rt) < 5:
                    continue
                shared = _common_prefix_len(tok, rt)
                if shared < 5:
                    continue
                diverge_in_query = len(tok) - shared
                diverge_in_row = len(rt) - shared
                if diverge_in_query <= 2 or diverge_in_row <= 2:
                    matched += 1
                    break
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

    family_query = str(
        _extract_price_service_from_query(query_text)
        or _price_raw_query_fallback(query_text)
    ).strip()
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

    family_query = str(
        _extract_price_service_from_query(query_text)
        or _price_raw_query_fallback(query_text)
    ).strip()
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

    # Фильтр различающего токена семейства: если запрос несёт букву (гепатит С),
    # в семействе оставляем ТОЛЬКО строки этой буквы — иначе пациент видит чужие
    # варианты (гепатит В) в ответе на запрос про С. Пусто (буквы нет в каталоге)
    # → оставляем как есть (generic-фолбэк, не хуже прежнего).
    family_disc = _family_discriminator_context(query_text, rows)
    if family_disc is not None:
        stem6, letter = family_disc
        filtered = [
            row
            for row in best_rows
            if _row_family_letter(str(row.get("serviceName") or row.get("name") or ""), stem6) == letter
        ]
        if filtered:
            best_rows = filtered

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

    family_query = str(
        _extract_price_service_from_query(query_text)
        or _price_raw_query_fallback(query_text)
    ).strip()
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

    family_query = str(
        _extract_price_service_from_query(query_text)
        or _price_raw_query_fallback(query_text)
    ).strip()
    if not _is_family_query_candidate(query_text, rows):
        return None

    ranked = _build_family_candidate_rows(query_text, rows, limit=50)
    if len(ranked) < 2:
        return None

    # Гард неудовлетворимого уточнения — ВТОРОЙ стык (как у family-дискриминатора
    # гепатитов). Single-резолвер уже отказывает «приём терапевта по ОМС», но без
    # этой проверки запрос проваливался в family-режим и получал список ЧУЖИХ
    # платных приёмов (хирург, мануальный терапевт) — дезинформация вместо
    # честного «не нашёл». «Найденное» здесь — весь предложенный список: если
    # уточнение не удовлетворено НИ ОДНИМ вариантом и вообще не встречается в
    # каталоге, показывать список нельзя.
    offered_names = " ".join(
        str(row.get("serviceName") or row.get("name") or "") for row in ranked
    )
    if _match_drops_unsatisfiable_qualifier(query_text, offered_names, rows):
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


# П6 «корзина» (BUG-B): + `\n` и `•` — реальные списки пациентов идут переносами
# строк/маркерами, а не запятыми. Дефис НЕ разделитель (В12-дефицит, АТ-ТПО).
_MULTI_PRICE_SPLIT_RE = re.compile(r"\s*(?:,|;|\bи\b|\+|/|\n|•)\s*", re.I)
_MULTI_PRICE_SERVICE_HINT_RE = re.compile(
    r"[a-zа-яё]{3,}",
    re.I,
)
# Шумовые фрагменты списка («Сколько будет стоить?», «итого») — отбрасываются
# МОЛЧА: это не услуга и не «нераспознанная позиция» (иначе пациент увидел бы
# «Не распознал: Сколько будет стоить» — бессмыслица).
_MULTI_PRICE_NOISE_RE = re.compile(
    r"^(?:сколько(?:\s+будет)?\s+сто[ии]\w*|стоимост\w*|цен\w*|прайс\w*|итого|всего|посчитай\w*)[\s?!.]*$",
    re.I,
)
# Слова-шум для токен-рана при сегментации по каталогу (S2): цена/вежливость/
# служебные. Пропускаются МОЛЧА (не услуги и не «нераспознанные») и РАЗРЫВАЮТ
# окна n-грамм (услуга не может содержать «пожалуйста» внутри).
_MULTI_PRICE_NOISE_WORDS = frozenset(
    {
        "сколько", "будет", "стоит", "стоить", "стоимость", "цена", "цены", "цен",
        "прайс", "итого", "всего", "посчитай", "посчитайте", "подскажите",
        "пожалуйста", "скажите", "здравствуйте", "добрый", "день", "вечер", "утро",
        "спасибо", "нужно", "надо", "хочу", "можно", "меня", "мне", "нам", "себя",
        "после", "перед", "возле", "около", "завтра", "сегодня", "послезавтра",
    }
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
        # П6: шумовой хвост списка («Сколько будет стоить?») — не услуга; молча мимо.
        if _MULTI_PRICE_NOISE_RE.match(frag):
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


def _segment_items_by_catalog(
    query_text: str,
    retail_rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """S2 «корзины» (BUG-B): сегментирует сплошной токен-ран ПО КАТАЛОГУ.

    Реальные списки пациентов часто идут одной строкой CAPS без разделителей
    («ОАК ОАМ ОБЩИЙ БЕЛОК ГЛЮКОЗА …»). Где кончается одна услуга и начинается
    другая, решают ДАННЫЕ: жадный longest-match n-грамм (окно 3→1) через тот же
    `resolve_price_service_name_from_catalog` — лексикон = сам прайс («ОБЩИЙ
    БЕЛОК» — 2 токена, «ОАК» — 1). Нераспознанные токены копятся отдельно
    (S3 — честный «не распознал»), шумовые слова цены отбрасываются молча.

    :param query_text: исходный текст (сплошной список)
    :param retail_rows: строки retail-прайса региона
    :return: (фрагменты-услуги в порядке появления, нераспознанные токены)
    """

    head = re.sub(
        r"^\s*(?:стоимость|цена|сколько\s+стоит|прайс)\s+",
        "",
        str(query_text or "").strip(),
        flags=re.I,
    )
    tokens = [t for t in re.findall(r"[0-9A-Za-zА-Яа-яЁё\-]+", head) if t][:40]
    if len(tokens) < 4:
        return [], []

    def _is_noise(tok: str) -> bool:
        low = tok.lower()
        return low in _MULTI_PRICE_NOISE_WORDS or len(low) < 2

    def _tokens_covered(window_tokens: list[str], canonical: str) -> bool:
        """Каждый содержательный токен находится в каноне (префикс-4)."""
        canon_words = [w for w in _normalise_input(canonical).split() if w]
        for tok in window_tokens:
            norm = _normalise_input(tok)
            if len(norm) < 3:
                continue
            pref = norm[:4]
            if not any(w.startswith(pref) or norm.startswith(w[:4]) for w in canon_words):
                return False
        return True

    # Анти-шинковка одиночной многословной услуги («cito общий анализ крови»,
    # «узи брюшной полости»): если ВЕСЬ контент-ран резолвится как ОДНА услуга
    # И каждый его токен покрыт каноном — это не список (single-путь разберётся).
    # Покрытие обязательно: мега-список тоже фаззи-резолвится «во что-то одно»
    # по сильному токену, но канон не покрывает остальные позиции.
    content_tokens = [t for t in tokens if not _is_noise(t)]
    if len(content_tokens) < 4:
        return [], []
    whole_canonical = resolve_price_service_name_from_catalog(
        " ".join(content_tokens), rows=retail_rows
    )
    if whole_canonical and _tokens_covered(content_tokens, whole_canonical):
        return [], []

    recognized: list[str] = []
    unrecognized: list[str] = []
    i = 0
    while i < len(tokens):
        if _is_noise(tokens[i]):
            i += 1
            continue
        matched = False
        for window in (3, 2, 1):
            if i + window > len(tokens):
                continue
            window_tokens = tokens[i : i + window]
            # Услуга не может содержать шум-слово внутри — окно разрывается.
            if window > 1 and any(_is_noise(t) for t in window_tokens):
                continue
            candidate = " ".join(window_tokens)
            canonical = resolve_price_service_name_from_catalog(candidate, rows=retail_rows)
            if not canonical:
                continue
            # Курируемый алиас (ОАК/ОАМ/…) точен по построению — гарды ниже минует
            # (канон алиаса не содержит букв аббревиатуры, overlap их зарезал бы).
            is_alias = _normalise_input(candidate) in _PRICE_SERVICE_ALIASES
            if not is_alias:
                # Гард подмены (класс BUG-2026-06-02-07): различающий токен запроса
                # обязан присутствовать в каноне («Витамин B1» ≠ «Витамин B12»).
                if not _scorer_match_has_distinctive_overlap(candidate, canonical):
                    continue
                # Анти-глотание соседей: у многословного окна КАЖДЫЙ токен обязан
                # быть в каноне («ОАК ОАМ ОБЩИЙ» → одна услуга — запрещено).
                if window > 1 and not _tokens_covered(window_tokens, canonical):
                    continue
            recognized.append(candidate)
            i += window
            matched = True
            break
        if not matched:
            unrecognized.append(tokens[i])
            i += 1
    return recognized, unrecognized


def _resolve_multi_price_items(
    query_text: str,
    retail_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Разрешает фрагменты мульти-услугового запроса в реальные услуги каталога.

    Возвращает `(items, unrecognized)`: items — список `{"service_name", "prices"}`
    только если ≥2 фрагментов приземлились на каталог (иначе пусто — fallback на
    single-service путь); unrecognized — позиции списка, которые каталогом не
    распознались (S3: честно перечисляются пациенту, не замалчиваются).

    :param query_text: исходный запрос пользователя
    :param retail_rows: строки retail-прайса региона
    :return: (разрешённые услуги с top-рядами цен, нераспознанные позиции)
    """

    fragments = _split_price_query_items(query_text)
    unrecognized: list[str] = []
    if len(fragments) < 2:
        # S2: разделителей нет — пробуем сегментацию сплошного списка по каталогу.
        seg_fragments, seg_unrecognized = _segment_items_by_catalog(query_text, retail_rows)
        if len(seg_fragments) >= 2:
            fragments = seg_fragments
            unrecognized = seg_unrecognized
        else:
            return [], []

    resolved: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for frag in fragments:
        service_name = resolve_price_service_name_from_catalog(frag, rows=retail_rows)
        if not service_name:
            extracted = _extract_price_service_from_query(frag) or frag
            alias_variants = _PRICE_SERVICE_ALIASES.get(_normalise_input(extracted), ())
            service_name = str(alias_variants[0] or "").strip() if alias_variants else ""
        if not service_name:
            unrecognized.append(frag)
            continue
        key = _normalise_input(service_name)
        if key in seen_names:
            continue
        top_rows = _select_patient_price_rows(retail_rows, service_name, limit=2)
        if not top_rows:
            unrecognized.append(frag)
            continue
        seen_names.add(key)
        resolved.append(
            {
                "service_name": service_name,
                "prices": _annotate_price_rows_with_care_context(top_rows),
            }
        )

    if len(resolved) < 2:
        return [], []
    return resolved, unrecognized


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

    items, unrecognized = _resolve_multi_price_items(query_text, retail_rows)
    if not items:
        return None

    # Гард неудовлетворимого уточнения — ТРЕТИЙ стык. Мульти-путь режет реплику
    # на фрагменты, и уточнение теряется между ними: «приём терапевта по ОМС»
    # распадался на «приём терапевта» + мусор и выдавал список платных приёмов
    # (хирург, мануальный терапевт). Каждый фрагмент по отдельности гард в
    # single-резолвере пропускает — проверяем реплику целиком против того, что
    # реально предлагаем.
    #
    # Нераспознанную позицию засчитываем как «предложенное» ТОЛЬКО если она —
    # самостоятельный пункт списка: тогда S3 честно перечислит её пациенту
    # («Не распознал: КВАНТОВЫЙ АНАЛИЗ АУРЫ»). Если же слово живёт ВНУТРИ
    # фразы, из которой мы вытащили услуги («приём терапевта по ОМС» — один
    # пункт), то оно молча поглощено, и список показывать нельзя.
    top_level = {_normalise_input(part) for part in _split_price_query_items(query_text)}
    honest_unrecognized = [
        str(part) for part in unrecognized if _normalise_input(str(part)) in top_level
    ]
    offered_names = " ".join(
        [str(item.get("service_name") or "") for item in items] + honest_unrecognized
    )
    if _match_drops_unsatisfiable_qualifier(query_text, offered_names, retail_rows):
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
    payload: dict[str, Any] = {
        "service_name": service_name,
        "service_kind": "family_query",
        "family_variants": variants,
        "showing_all": True,
        "visible_limit": visible_limit,
        "remaining_count": 0,
        "show_all_hint": "",
        "note": "price_multi_service",
    }
    if unrecognized:
        # S3 «корзины» (BUG-B): нераспознанные позиции честно перечисляем —
        # молчание о них дезинформировало бы полнотой ответа.
        shown = ", ".join(dict.fromkeys(str(u).strip() for u in unrecognized if str(u).strip()))
        if shown:
            payload["unrecognized_note"] = (
                f"Не распознал: {shown}. Уточните эти названия — помогу по ним отдельно."
            )
    return payload


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
