"""Shared specialty parsing primitives.

This module is intentionally lightweight so both messenger runtime and
FreeTalk can reuse the same specialty extraction logic without importing
heavy router/service modules.
"""

from __future__ import annotations

import re

from .russian_nlu import normalize_ru

UZI_QUERY_RE = re.compile(r"\b(узи|узист|ультразвук\w*|ультразвуков\w*)\b", re.I)
ENDOSCOPY_SERVICE_RE = re.compile(
    r"\b(эндоскоп\w*|фгдс|фдгс|фгс|егдс|эгдс|фкс|гастроскоп\w*|колоноскоп\w*|"
    r"ректороманоскоп\w*|эзофагогастродуоденоскоп\w*)\b",
    re.I,
)

SPECIALTY_ROLE_SYNONYMS: dict[str, tuple[str, ...]] = {
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

SPECIALTY_CANONICAL = (
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

SPECIALTY_RE = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in sorted(SPECIALTY_CANONICAL, key=len, reverse=True)) + r")\w*\b",
    re.I,
)


def _normalise_input(text: str | None) -> str:
    return re.sub(r"\s+", " ", normalize_ru(text))


def specialty_terms(specialty: str) -> tuple[str, ...]:
    spec_norm = _normalise_input(specialty)
    if not spec_norm:
        return tuple()
    terms = SPECIALTY_ROLE_SYNONYMS.get(spec_norm, (spec_norm,))
    out: list[str] = []
    for term in terms:
        term_norm = _normalise_input(term)
        if term_norm and term_norm not in out:
            out.append(term_norm)
    return tuple(out)


def _matches_canonical_specialty_label(text: str, specialty: str) -> bool:
    norm = _normalise_input(text)
    spec_norm = _normalise_input(specialty)
    if not norm or not spec_norm:
        return False
    tokens = re.findall(r"[a-zа-я0-9]+", norm)
    if not tokens:
        return False
    label_tokens = re.findall(r"[a-zа-я0-9]+", spec_norm)
    if not label_tokens:
        return False
    if len(label_tokens) > 1:
        return all(any(token == part or token.startswith(part) for token in tokens) for part in label_tokens)
    term = label_tokens[0]
    if len(term) <= 4:
        return any(token == term for token in tokens)
    return any(token == term or token.startswith(term) for token in tokens)


def matches_specialty_terms(text: str, specialty: str) -> bool:
    norm = _normalise_input(text)
    if not norm:
        return False
    tokens = re.findall(r"[a-zа-я0-9]+", norm)
    if not tokens:
        return False

    for term in specialty_terms(specialty):
        term_tokens = re.findall(r"[a-zа-я0-9]+", term)
        if len(term_tokens) > 1:
            if all(any(token == part or token.startswith(part) for token in tokens) for part in term_tokens):
                return True
            continue
        if len(term) <= 4:
            if any(token == term for token in tokens):
                return True
            continue
        if any(token == term or token.startswith(term) for token in tokens):
            return True
    return False


def specialty_equivalent(left: str, right: str) -> bool:
    left_norm = _normalise_input(left)
    right_norm = _normalise_input(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_terms = {left_norm, *specialty_terms(left_norm)}
    right_terms = {right_norm, *specialty_terms(right_norm)}
    return bool(left_terms & right_terms)


def extract_specialties_from_text(text: str) -> tuple[str, ...]:
    norm = _normalise_input(text)
    if not norm:
        return tuple()
    found: list[str] = []
    for raw in SPECIALTY_CANONICAL:
        spec = _normalise_input(raw)
        if spec and matches_specialty_terms(norm, spec) and spec not in found:
            found.append(spec)
    return tuple(found)


def extract_specialty_from_text(text: str) -> str:
    probe = str(text or "")
    if UZI_QUERY_RE.search(probe):
        return "узи"
    if ENDOSCOPY_SERVICE_RE.search(probe):
        return "эндоскопист"
    for specialty in sorted(SPECIALTY_CANONICAL, key=len, reverse=True):
        if _matches_canonical_specialty_label(probe, specialty):
            return specialty
    match = SPECIALTY_RE.search(probe)
    if not match:
        return ""
    return normalize_ru(match.group(1))


__all__ = [
    "ENDOSCOPY_SERVICE_RE",
    "SPECIALTY_CANONICAL",
    "SPECIALTY_RE",
    "SPECIALTY_ROLE_SYNONYMS",
    "UZI_QUERY_RE",
    "extract_specialties_from_text",
    "extract_specialty_from_text",
    "matches_specialty_terms",
    "specialty_equivalent",
    "specialty_terms",
]
