"""Helpers для обработки запросов о подготовке к анализам/процедурам (Stage 20, cluster 2).

Содержит константы, регулярные выражения и функции для поиска,
ранжирования и валидации ответов по подготовке пациентов.

Переносится из ``services_legacy.py`` в рамках Stage 20 с сохранением
обратной совместимости через re-export shim в legacy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from converters import html_cleaner

from ..prompt_registry import load_prompt_text
from ._common import (
    _doc_tokens,
    _extract_json_object,
    _get_first_present,
    _normalise_input,
)


def _get_legacy():
    """Лениво импортирует services_legacy, чтобы избежать цикла импортов.

    Используется для получения patchable-ссылок на функции, которые тесты
    монкипатчат через svc_mod (services/__init__.py → services_legacy).
    """
    from .. import services_legacy as legacy  # noqa: PLC0415

    return legacy


def _get_stem_service_token():
    """Temporary: moves to _doctors_helpers in Cluster 6."""
    return _get_legacy()._stem_service_token


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PRICE_PREPARE_HINT_RE = re.compile(
    r"\b(подготов\w*|натощак|перед\s+(анализ\w*|исследован\w*|процедур\w*))\b",
    re.I,
)

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

_PREPARE_CODE_FENCE_START_RE = re.compile(r"^\s*```(?:\w+)?\s*", re.I)
_PREPARE_CODE_FENCE_END_RE = re.compile(r"\s*```\s*$", re.I)

# ---------------------------------------------------------------------------
# Set / dict constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# String constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _normalise_prepare_text(text: str) -> str:
    norm = _normalise_input(text)
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


def _prepare_term_roots(text: str) -> set[str]:
    """
    Возвращает корни смысловых токенов для query/content matching.

    :param text: исходный текст
    :return: множество токенов-корней
    """

    _stem_service_token = _get_stem_service_token()
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
        query_norm = _normalise_input(query)
        content_norm = _normalise_input(content)
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

    query_norm = _normalise_input(query)
    content_norm = _normalise_input(content)
    title_norm = _normalise_input(title)
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
    # Use lazy lookup so monkeypatching _runtime_float via svc_mod works in tests.
    _runtime_float = _get_legacy()._runtime_float
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

    service_name = _normalise_input(str(row.get("serviceName") or ""))
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


def _is_prepare_requested_in_price_query(query_text: str) -> bool:
    """
    Проверяет, просит ли пользователь именно подготовку в PRICE-реплике.

    :param query_text: текст запроса пользователя
    :return: True, если запрошены правила подготовки
    """

    return bool(_PRICE_PREPARE_HINT_RE.search(str(query_text or "")))


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


def _is_prepare_relevant(query: str, content: str) -> bool:
    content_norm = _normalise_input(content)
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

    norm = _normalise_input(content)
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
    norm = _normalise_input(content)
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
