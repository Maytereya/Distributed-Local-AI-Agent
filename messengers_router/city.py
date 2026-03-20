"""Утилиты распознавания города и базовой адресной эвристики.

Модуль загружает справочник городов, нормализует словоформы и помогает
отделять упоминание города от уличного адреса в коротких ответах пациента.
Ответственность модуля: устойчиво извлекать город из пользовательской реплики.
"""

from __future__ import annotations

from difflib import get_close_matches
import re
from functools import lru_cache
from pathlib import Path

ADDRESS_WORD_RE = re.compile(
    r"\b(ул\.?|улиц[аеы]|пр\.?|проспект|пр-?т|шоссе|бульвар|пер\.?|переулок|наб\.?|набережн|пл\.?|площадь)\b",
    re.I,
)
CITY_HINT_RE = re.compile(r"\b(в|из|по)\s+([А-ЯЁа-яё\-]{3,})\b")
_CITY_SWITCH_RE = re.compile(r"\bне\s+в\s+([a-zа-яё\-]{3,})\b.*?\bа\s+в\s+([a-zа-яё\-]{3,})\b", re.I)
_CITY_SWITCH_SOFT_RE = re.compile(r"\bне\s+в\s+([a-zа-яё\-]{3,})\b[\s,;:.!\-]+в\s+([a-zа-яё\-]{3,})\b", re.I)
_CITY_FUZZY_CUTOFF = 0.82

_CITIES_PATH = Path(__file__).resolve().parent / "data" / "cities.txt"


def norm_city_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("ё", "е")
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"[^a-zа-яё0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_address(text: str) -> bool:
    return bool(ADDRESS_WORD_RE.search(text) or re.search(r"\d", text))


@lru_cache
def load_city_list() -> list[str]:
    if not _CITIES_PATH.exists():
        return []
    cities: list[str] = []
    with _CITIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            cities.append(name)
    return cities


@lru_cache
def city_variants_map() -> dict[str, str]:
    cities = load_city_list()
    if not cities:
        return {}

    norm_to_name: dict[str, str] = {}
    for c in cities:
        norm_to_name[norm_city_text(c)] = c

    def _canonical_name(norm: str) -> str | None:
        if norm in norm_to_name:
            return norm_to_name[norm]
        if norm.endswith("ы"):
            cand = norm[:-1] + "а"
            if cand in norm_to_name:
                return norm_to_name[cand]
        if norm.endswith("и"):
            cand = norm[:-1] + "я"
            if cand in norm_to_name:
                return norm_to_name[cand]
            cand = norm[:-1] + "ь"
            if cand in norm_to_name:
                return norm_to_name[cand]
        if norm.endswith("е"):
            cand = norm[:-1] + "а"
            if cand in norm_to_name:
                return norm_to_name[cand]
            cand = norm[:-1] + "я"
            if cand in norm_to_name:
                return norm_to_name[cand]
        return None

    def _generate_variants(norm: str) -> set[str]:
        variants = {norm}
        words = norm.split()
        if not words:
            return variants
        base = words[-1]
        last_variants = {base}
        if base.endswith("а"):
            last_variants.update({base[:-1] + "ы", base[:-1] + "е", base[:-1] + "у"})
        elif base.endswith("я"):
            last_variants.update({base[:-1] + "и", base[:-1] + "е", base[:-1] + "ю"})
        elif base.endswith("ь"):
            last_variants.update({base[:-1] + "и", base[:-1] + "е"})
        elif base.endswith("й"):
            last_variants.update({base[:-1] + "я", base[:-1] + "е"})
        else:
            if base[-1] in "бвгджзклмнпрстфхцчшщ":
                last_variants.update({base + "а", base + "е"})

        for v in last_variants:
            if len(words) > 1:
                variants.add(" ".join(words[:-1] + [v]))
            else:
                variants.add(v)
        return variants

    mapping: dict[str, str] = {}
    for c in cities:
        norm = norm_city_text(c)
        canonical = _canonical_name(norm) or c
        for v in _generate_variants(norm):
            if v not in mapping:
                mapping[v] = canonical
    return mapping


@lru_cache
def city_regex() -> re.Pattern:
    mapping = city_variants_map()
    if not mapping:
        return CITY_HINT_RE
    escaped = sorted({re.escape(k) for k in mapping.keys() if k}, key=len, reverse=True)
    return re.compile(rf"\b(в|из|по)\s+({"|".join(escaped)})\b", re.I)


@lru_cache
def _city_variant_keys() -> tuple[str, ...]:
    mapping = city_variants_map()
    return tuple(sorted([k for k in mapping.keys() if k], key=len))


def fuzzy_match_city(text: str, cutoff: float = _CITY_FUZZY_CUTOFF) -> str | None:
    mapping = city_variants_map()
    if not mapping:
        return None

    t_norm = norm_city_text(text)
    if not t_norm:
        return None
    if t_norm in mapping:
        return mapping[t_norm]

    candidates: list[str] = [t_norm]
    m = CITY_HINT_RE.search(t_norm)
    if m:
        candidates.append(norm_city_text(m.group(2)))
    tokens = [tok for tok in t_norm.split() if tok and not re.search(r"\d", tok)]
    for size in (1, 2, 3):
        if len(tokens) < size:
            continue
        for i in range(len(tokens) - size + 1):
            gram = " ".join(tokens[i : i + size]).strip()
            if gram and gram not in candidates:
                candidates.append(gram)

    keys = _city_variant_keys()
    for candidate in candidates:
        if not candidate or len(candidate) < 3:
            continue
        best = get_close_matches(candidate, keys, n=1, cutoff=cutoff)
        if best:
            return mapping.get(best[0])
    return None


def match_city(text: str) -> str | None:
    cities = load_city_list()
    if not cities:
        m = CITY_HINT_RE.search(text)
        return m.group(2) if m else None

    mapping = city_variants_map()
    t_norm = norm_city_text(text)

    # В конструкциях вида "не в Самаре, а в Сызрани" предпочитаем целевой город после "а в ...".
    for rx in (_CITY_SWITCH_RE, _CITY_SWITCH_SOFT_RE):
        sw = rx.search(t_norm)
        if not sw:
            continue
        target_raw = norm_city_text(sw.group(2))
        if target_raw in mapping:
            return mapping[target_raw]
        target_fuzzy = fuzzy_match_city(target_raw)
        if target_fuzzy:
            return target_fuzzy

    m = city_regex().search(t_norm)
    if m:
        key = norm_city_text(m.group(2))
        return mapping.get(key, m.group(2))

    # short reply fallback: exact match to city variants
    if t_norm and t_norm in mapping:
        return mapping[t_norm]
    # mixed phrase fallback: ищем известный город как отдельный фрагмент внутри фразы
    # ("Анализы Самара", "нужен филиал в Самаре", "Самара анализы").
    for key in sorted((k for k in mapping.keys() if k), key=len, reverse=True):
        if re.search(rf"(?<![a-zа-яё0-9]){re.escape(key)}(?![a-zа-яё0-9])", t_norm):
            return mapping[key]
    # typo-tolerant fallback: only canonical names from known city list
    return fuzzy_match_city(text)
