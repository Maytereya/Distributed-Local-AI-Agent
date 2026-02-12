"""Общий helper для распознавания и fuzzy-матчинга фамилии врача.

Используется в мессенджерном роутере и может переиспользоваться в старом
контуре оператора. Не содержит форматирования ответов и call-center заметок.
"""

from __future__ import annotations

import re
from typing import Any

from agent_logic_2.text_constants import STOP_WORDS
from agent_logic_2.text_fuzzy import fuzzy_ratio

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]{3,}")
_SCHEDULE_DOCTOR_AFTER_RE = re.compile(
    r"(?:расписани\w*|график)\s+"
    r"(?:доктор\w*\s+|врач\w*\s+|уролог\w*\s+|кардиолог\w*\s+|невролог\w*\s+|"
    r"терапевт\w*\s+|гинеколог\w*\s+|педиатр\w*\s+|лор\w*\s+|хирург\w*\s+)?"
    r"([А-ЯЁа-яё\-]{3,})",
    re.I,
)
_SCHEDULE_DOCTOR_BEFORE_RE = re.compile(r"\b([А-ЯЁа-яё\-]{3,})\s+(?:расписани\w*|график)\b", re.I)
_SCHEDULE_WHEN_AFTER_RE = re.compile(
    r"(?:когда\s+принима\w*|принима\w*\s+когда)\s+"
    r"(?:доктор\w*\s+|врач\w*\s+|уролог\w*\s+|кардиолог\w*\s+|невролог\w*\s+|"
    r"терапевт\w*\s+|гинеколог\w*\s+|педиатр\w*\s+|лор\w*\s+|хирург\w*\s+)?"
    r"([А-ЯЁа-яё\-]{3,})",
    re.I,
)
_SCHEDULE_WHEN_BEFORE_RE = re.compile(
    r"\b([А-ЯЁа-яё\-]{3,})\s+(?:когда\s+принима\w*|принима\w*)\b",
    re.I,
)
_SCHEDULE_SLOTS_U_RE = re.compile(
    r"(?:свободн\w*\s+)?(?:окн\w*|слот\w*)\D{0,25}\bу\s+([А-ЯЁа-яё\-]{3,})\b",
    re.I,
)
_DOCTOR_APPOINTMENT_RE = re.compile(
    r"\bк\s+(?:доктор\w*\s+|врач\w*\s+)?([А-ЯЁа-яё\-]{3,})\b",
    re.I,
)
_DOCTOR_ABOUT_RE = re.compile(
    r"\b(?:о|про)\s+(?:доктор\w*|врач\w*)\s+([А-ЯЁа-яё\-]{3,})\b",
    re.I,
)
_DOCTOR_NEED_RE = re.compile(
    r"\bнуж\w*\s+([А-ЯЁа-яё\-]{3,})\b",
    re.I,
)
_DOCTOR_FIO_RE = re.compile(r"\b([А-ЯЁа-яё\-]{3,})\s+([А-ЯЁа-яё\-]{3,})\b")
_EXTRA_STOPWORD_STEMS = (
    "уролог",
    "кардиолог",
    "невролог",
    "терапевт",
    "гинеколог",
    "педиатр",
    "хирург",
    "лор",
    "эндокринолог",
    "врач",
    "доктор",
    "расписан",
    "график",
    "принима",
    "инф",
    "нуж",
)
_EXTRA_STOPWORDS = {
    "расписание",
    "расписания",
    "график",
    "врач",
    "врача",
    "доктор",
    "доктора",
    "специалист",
    "специалиста",
    "на",
    "по",
    "в",
    "и",
    "или",
    "неделю",
    "ближайшую",
    "ближайшей",
    "покажи",
    "покажите",
    "подскажи",
    "подскажите",
    "какие",
    "какой",
    "какая",
    "какое",
    "какую",
    "есть",
    "свободные",
    "свободное",
    "свободный",
    "слот",
    "слоты",
    "слота",
    "окно",
    "окна",
    "окну",
    "у",
    "скажи",
    "скажите",
    "передумал",
    "передумала",
    "давайте",
    "когда",
    "принимает",
    "принимают",
    "прием",
    "приёма",
    "приема",
    "есть",
    "показывает",
    "покажи",
    "покажите",
    "уролог",
    "кардиолог",
    "невролог",
    "терапевт",
    "гинеколог",
    "педиатр",
    "хирург",
    "лор",
    "эндокринолог",
    "инфа",
    "инфо",
    "информация",
    "нужен",
    "нужна",
    "нужно",
    "нужны",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
    "январь",
    "февраль",
    "март",
    "апрель",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
}


def _clean_token(token: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-яЁё\-]", "", token or "").strip()


def _normalize_token(token: str) -> str | None:
    cleaned = _clean_token(token)
    if not cleaned or len(cleaned) < 3:
        return None
    low = cleaned.lower()
    if low in _EXTRA_STOPWORDS:
        return None
    if any(low.startswith(stem) for stem in _EXTRA_STOPWORD_STEMS):
        return None
    return cleaned.capitalize()


def surname_variants(value: str) -> list[str]:
    """
    Возвращает варианты фамилии с учетом частых русских падежных форм.
    Пример: "Дразнина" -> ["Дразнина", "Дразнин"].
    """
    s = _clean_token(value)
    if not s:
        return []
    variants: list[str] = [s]
    low = s.lower()
    if low.endswith("а") and len(s) >= 4:
        variants.append(s[:-1])
    if low.endswith("я") and len(s) >= 4:
        variants.append(s[:-1] + "й")
    if low.endswith("ой") and len(s) >= 5:
        variants.append(s[:-2] + "ая")
    if low.endswith("ей") and len(s) >= 5:
        variants.append(s[:-2] + "ая")

    seen: set[str] = set()
    uniq: list[str] = []
    for item in variants:
        k = item.lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(item)
    return uniq


def _extract_surname_from_schedule_phrase(text: str) -> str | None:
    m = _SCHEDULE_DOCTOR_AFTER_RE.search(text)
    if m:
        candidate = _normalize_token(m.group(1))
        if candidate:
            return candidate
    m = _SCHEDULE_DOCTOR_BEFORE_RE.search(text)
    if m:
        candidate = _normalize_token(m.group(1))
        if candidate:
            return candidate
    m = _SCHEDULE_WHEN_AFTER_RE.search(text)
    if m:
        candidate = _normalize_token(m.group(1))
        if candidate:
            return candidate
    m = _SCHEDULE_WHEN_BEFORE_RE.search(text)
    if m:
        candidate = _normalize_token(m.group(1))
        if candidate:
            return candidate
    m = _SCHEDULE_SLOTS_U_RE.search(text)
    if m:
        candidate = _normalize_token(m.group(1))
        if candidate:
            return candidate
    return None


def _extract_name_from_appointment_phrase(text: str) -> str | None:
    m = _DOCTOR_APPOINTMENT_RE.search(text or "")
    if not m:
        m = _DOCTOR_ABOUT_RE.search(text or "")
    if not m:
        m = _DOCTOR_NEED_RE.search(text or "")
    if not m:
        return None
    return _normalize_token(m.group(1))


def extract_surname_candidate(text: str) -> str | None:
    """
    Пытается извлечь фамилию из произвольной фразы пользователя.
    Поддерживает форматы: "расписание Дразнина", "Дразнин расписание".
    """
    if not isinstance(text, str):
        return None
    schedule_candidate = _extract_surname_from_schedule_phrase(text)
    if schedule_candidate:
        return schedule_candidate

    words = _WORD_RE.findall(text)
    if not words:
        return None

    stop = {w.lower() for w in STOP_WORDS} | _EXTRA_STOPWORDS
    filtered = [w for w in words if w.lower() not in stop]
    if filtered:
        # Берем первый осмысленный токен (а не последний),
        # чтобы фразы "когда принимает Дразнин уролог" не превращались в "уролог".
        for token in filtered:
            normalized = _normalize_token(token)
            if normalized:
                return normalized
        return None
    fallback = _normalize_token(words[-1])
    if not fallback or fallback.lower() in stop:
        return None
    return fallback


def extract_doctor_name_candidate(text: str, *, prefer_schedule: bool = False) -> str | None:
    """
    Унифицированное извлечение имени/фамилии врача из пользовательской фразы.
    Для schedule-сценариев используем `prefer_schedule=True`.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    if prefer_schedule:
        return extract_surname_candidate(text)

    by_appointment = _extract_name_from_appointment_phrase(text)
    if by_appointment:
        return by_appointment

    by_schedule = extract_surname_candidate(text)
    if by_schedule:
        return by_schedule

    fio_match = _DOCTOR_FIO_RE.search(text)
    if fio_match:
        n1 = _normalize_token(fio_match.group(1))
        n2 = _normalize_token(fio_match.group(2))
        if n1 and n2:
            return f"{n1} {n2}"

    return None


def _doctor_surnames(doctors: list[dict[str, Any]]) -> dict[str, str]:
    """
    Возвращает map normalized_surname -> canonical_surname из списка врачей.
    """
    out: dict[str, str] = {}
    for doc in doctors or []:
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            continue
        surname = _clean_token(fio.split()[0])
        if not surname:
            continue
        key = surname.lower().replace("ё", "е")
        if key not in out:
            out[key] = surname
    return out


def find_best_surname_match(
    input_surname: str,
    doctors: list[dict[str, Any]],
    threshold: float = 0.75,
) -> str | None:
    """
    Ищет наиболее близкую фамилию врача в кэше (с учетом опечаток/падежа/регистра).
    """
    candidates = surname_variants(input_surname)
    if not candidates:
        return None

    surnames_map = _doctor_surnames(doctors)
    if not surnames_map:
        return candidates[0].capitalize()

    for candidate in candidates:
        key = candidate.lower().replace("ё", "е")
        exact = surnames_map.get(key)
        if exact:
            return exact

    best: str | None = None
    best_ratio = 0.0
    for candidate in candidates:
        for normalized, canonical in surnames_map.items():
            score = fuzzy_ratio(candidate, normalized)
            if score > best_ratio:
                best_ratio = score
                best = canonical
    if best and best_ratio >= threshold:
        return best
    return candidates[0].capitalize()


def resolve_schedule_surname(raw_text: str, doctors: list[dict[str, Any]], threshold: float = 0.75) -> str | None:
    """
    Возвращает нормализованную фамилию для запроса расписания.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    surnames_map = _doctor_surnames(doctors)
    stop = {w.lower() for w in STOP_WORDS} | _EXTRA_STOPWORDS
    words = _WORD_RE.findall(raw_text)
    token_candidates: list[str] = []
    for token in words:
        low = token.lower()
        if low in stop:
            continue
        normalized = _normalize_token(token)
        if normalized:
            token_candidates.append(normalized)

    # 1) exact match по любому токену/его падежным вариантам
    if surnames_map and token_candidates:
        for token in token_candidates:
            for variant in surname_variants(token):
                exact = surnames_map.get(variant.lower().replace("ё", "е"))
                if exact:
                    return exact

    # 2) fuzzy best по токенам
    if surnames_map and token_candidates:
        best: str | None = None
        best_ratio = 0.0
        for token in token_candidates:
            for variant in surname_variants(token):
                for normalized, canonical in surnames_map.items():
                    score = fuzzy_ratio(variant, normalized)
                    if score > best_ratio:
                        best_ratio = score
                        best = canonical
        if best and best_ratio >= threshold:
            return best
        return None

    # 3) fallback без кэша — только если кандидат действительно есть
    candidate = extract_surname_candidate(raw_text)
    if not candidate:
        return None
    return find_best_surname_match(candidate, doctors, threshold=threshold)
