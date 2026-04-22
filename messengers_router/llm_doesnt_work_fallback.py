"""Детерминированный fallback для PREPARE, когда LLM-обертка недоступна.

Модуль сознательно не использует LLM и внешние сервисы: только фильтрацию и
компоновку текста по эвристикам, чтобы ответ оставался коротким и релевантным.
"""

from __future__ import annotations

import re

from .russian_nlu import normalize_ru

_WORD_RE = re.compile(r"[a-zа-я0-9]{2,}", re.I)
_PREPARE_TARGET_RE = re.compile(
    r"подготов(?:иться|ка)\s*(?:к|для)\s+(?P<target>.+?)(?:[?.!,]|$)",
    re.I,
)
_PREPARE_ACTION_HINTS = (
    "натощак",
    "за ",
    "час",
    "день",
    "сутк",
    "нельзя",
    "не ",
    "исключ",
    "воздерж",
    "перед",
    "утром",
    "вечером",
    "сдать",
    "сдавать",
    "процедур",
    "исследован",
)
_PREPARE_BLOCK_HINTS = (
    "стоим",
    "цена",
    "руб",
    "итого",
    "врач",
    "доктор",
    "адрес",
    "филиал",
    "телефон",
    "запис",
    "консультац",
    "акци",
)
_PREPARE_STOPWORDS = {
    "как",
    "к",
    "для",
    "подготовиться",
    "подготовка",
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
    "по",
    "с",
    "на",
    "и",
    "или",
}
_PREPARE_TOPIC_ROOTS = {
    "фгдс",
    "гастроскоп",
    "колоноскоп",
    "ректороманоскоп",
    "кольпоскоп",
    "вульвоскоп",
    "биопс",
    "пайпел",
    "спирал",
    "холестерин",
    "липид",
    "гормон",
    "урогенитал",
    "мазок",
    "зппп",
    "иппп",
    "цервик",
}
_PREPARE_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "фгдс": ("гастроскоп", "эгдс", "фгс", "эндоскоп"),
    "гастроскоп": ("фгдс", "эгдс", "эндоскоп"),
    "эгдс": ("фгдс", "гастроскоп", "эндоскоп"),
    "кольпоскоп": ("вульвоскоп",),
    "вульвоскоп": ("кольпоскоп",),
    "холестерин": ("липид", "липидограмм"),
    "биопс": ("пайпел",),
    "пайпел": ("биопс",),
}
_RU_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ими",
    "ыми",
    "ией",
    "ией",
    "ией",
    "ией",
    "ия",
    "ие",
    "ий",
    "ой",
    "ей",
    "ые",
    "ое",
    "ая",
    "яя",
    "ов",
    "ев",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "ам",
    "ям",
    "у",
    "ю",
    "а",
    "я",
    "ы",
    "и",
    "е",
    "о",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_ru(text))


def _stem(token: str) -> str:
    t = normalize_ru(token)
    if len(t) < 4:
        return t
    for suffix in _RU_SUFFIXES:
        if t.endswith(suffix) and (len(t) - len(suffix)) >= 4:
            return t[: -len(suffix)]
    return t


def _query_roots(query: str) -> set[str]:
    norm = _norm(query)
    if not norm:
        return set()
    roots: set[str] = set()
    for token in _WORD_RE.findall(norm):
        if token in _PREPARE_STOPWORDS:
            continue
        stem = _stem(token)
        if len(stem) >= 3:
            roots.add(stem)
    for hint, variants in _PREPARE_QUERY_SYNONYMS.items():
        if hint in norm:
            roots.add(_stem(hint))
            for variant in variants:
                roots.add(_stem(variant))
    return roots


def _extract_target_phrase(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    m = _PREPARE_TARGET_RE.search(text)
    if not m:
        return ""
    target = str(m.group("target") or "").strip(" \t\n\r.,:;!?")
    target = re.sub(r"\s+", " ", target)
    return target


def _split_segments(source_text: str) -> list[str]:
    raw = str(source_text or "")
    if not raw.strip():
        return []

    # Сначала делим по строкам/буллетам, затем дополнительно по завершенным фразам.
    pre = raw.replace("•", "\n").replace("—", "-")
    lines = [ln.strip(" \t-") for ln in pre.splitlines() if ln.strip()]
    if not lines:
        lines = [raw]

    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        chunks = re.split(r"(?<=[.!?;])\s+", line)
        for chunk in chunks:
            text = re.sub(r"\s+", " ", chunk).strip(" \t-")
            if len(text) < 8:
                continue
            key = _norm(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def _segment_roots(segment: str) -> set[str]:
    roots: set[str] = set()
    for token in _WORD_RE.findall(_norm(segment)):
        stem = _stem(token)
        if len(stem) >= 3:
            roots.add(stem)
    return roots


def _is_blocked_segment(norm_segment: str) -> bool:
    return any(h in norm_segment for h in _PREPARE_BLOCK_HINTS)


def _segment_score(segment: str, query_roots: set[str]) -> float:
    norm = _norm(segment)
    if not norm or _is_blocked_segment(norm):
        return -1000.0

    roots = _segment_roots(segment)
    common = len(query_roots & roots)
    actionable = any(h in norm for h in _PREPARE_ACTION_HINTS)
    has_time = bool(re.search(r"\b\d{1,2}\s*(?:час|дн|сут|нед|мин)", norm))
    has_empty_stomach = "натощак" in norm

    foreign_topics = 0
    for root in _PREPARE_TOPIC_ROOTS:
        stem = _stem(root)
        if stem in roots and stem not in query_roots:
            foreign_topics += 1

    score = 0.0
    score += float(common) * 2.2
    if actionable:
        score += 1.6
    if has_time:
        score += 1.0
    if has_empty_stomach:
        score += 1.0
    if foreign_topics:
        score -= float(foreign_topics) * 1.2
    return score


def _pick_segments(query: str, source_text: str, *, max_points: int) -> list[str]:
    segments = _split_segments(source_text)
    if not segments:
        return []

    roots = _query_roots(query)
    ranked: list[tuple[float, int, str]] = []
    for idx, seg in enumerate(segments):
        score = _segment_score(seg, roots)
        if score <= 0.4:
            continue
        ranked.append((score, idx, seg))

    if not ranked:
        for idx, seg in enumerate(segments):
            norm = _norm(seg)
            if _is_blocked_segment(norm):
                continue
            if any(h in norm for h in _PREPARE_ACTION_HINTS):
                ranked.append((0.6, idx, seg))

    if not ranked:
        return []

    ranked.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    top = ranked[: max(1, max_points)]
    top.sort(key=lambda row: row[1])
    return [row[2] for row in top]


def _format_answer(query: str, segments: list[str], *, max_chars: int) -> str:
    if not segments:
        return ""

    target = _extract_target_phrase(query)
    if target:
        intro = f"Подготовка к {target}:"
    else:
        intro = "Краткая подготовка по вашему запросу:"

    bullets = [f"- {re.sub(r'\\s+', ' ', seg).strip()}" for seg in segments if str(seg).strip()]
    if not bullets:
        return ""

    # Ограничиваем длину без грубого обрезания середины инструкции.
    while bullets and len(intro) + 1 + sum(len(b) + 1 for b in bullets) > max_chars:
        bullets.pop()
    if not bullets:
        return intro

    text = intro + "\n" + "\n".join(bullets)
    if len(text) <= max_chars:
        return text

    # Финальный страховочный trim последней строки.
    allowed = max_chars - (len(intro) + 2)
    if allowed <= 20:
        return intro
    body = "\n".join(bullets)
    body = body[:allowed].rstrip()
    return intro + "\n" + body


def build_prepare_fallback_answer(
    query: str,
    source_text: str,
    *,
    max_chars: int = 1600,
    max_points: int = 7,
) -> str:
    """
    Строит короткий детерминированный PREPARE-ответ без участия LLM.

    :param query: исходный запрос пользователя
    :param source_text: сырой текст из выбранного источника
    :param max_chars: верхняя граница длины ответа
    :param max_points: максимум пунктов в итоговом ответе
    :return: компактный ответ; пустая строка, если собрать его не удалось
    """

    source = str(source_text or "").strip()
    if not source:
        return ""
    points = max(2, min(10, int(max_points)))
    limit = max(400, min(4000, int(max_chars)))

    chosen = _pick_segments(query, source, max_points=points)
    answer = _format_answer(query, chosen, max_chars=limit)
    if answer.strip():
        return answer.strip()
    return ""

