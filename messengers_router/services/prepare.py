"""Модуль домена подготовки к анализам/процедурам.

Содержит миграцию PREPARE-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.nayka_api import api_service_info
from converters import html_cleaner

from .. import llm_runtime as llm_runtime_mod
from ..llm_doesnt_work_fallback import build_prepare_fallback_answer
from . import _common as _common_mod
from ._common import (
    _get_first_present,
    _is_meili_error_text,
    _is_meili_no_matches_text,
)
from ._prepare import (
    _PREPARE_RELEVANCE_VERDICT_IRRELEVANT,
    _PREPARE_RELEVANCE_VERDICT_RELEVANT,
    _PrepareCandidate,
    _dedupe_prepare_candidates,
    _has_prepare_strong_hints,
    _is_prepare_content_actionable,
    _is_prepare_service_info_usable,
    _is_prepare_wrap_output_usable,
    _parse_prepare_relevance_validator,
    _prepare_clarify_response,
    _prepare_fast_relevance_score,
    _prepare_query_variants,
    _prepare_relevance_gate,
    _prepare_relevance_prompt,
    _prepare_roots_coverage,
    _prepare_service_info_queries,
    _prepare_term_roots,
    _prepare_wrap_clean,
    _prepare_wrap_prompt,
)

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migrated Services methods (kept in legacy's declaration order).
# ---------------------------------------------------------------------------


async def _prepare_llm_validate_candidate(
    self: "Services",
    query: str,
    candidate: "Any",
) -> tuple[bool, float, str]:
    """
    LLM-валидация релевантности для кандидата PREPARE в серой зоне score.

    :param query: запрос пользователя
    :param candidate: кандидат из API/Meili
    :return: (релевантно, confidence, reason)
    """

    if not _common_mod._runtime_bool("MR_PREPARE_RELEVANCE_LLM_ENABLED", True):
        return False, 0.0, "llm_disabled"

    prompt = _prepare_relevance_prompt(
        query,
        candidate.text,
        source_kind=candidate.source,
        service_title=candidate.service_title,
    )
    if not prompt:
        return False, 0.0, "empty_prompt"

    timeout_s = _common_mod._runtime_int(
        "MR_PREPARE_RELEVANCE_LLM_TIMEOUT_S",
        15,
        min_value=1,
        max_value=60,
    )
    queue_timeout_ms = _common_mod._runtime_int(
        "MR_PREPARE_RELEVANCE_LLM_QUEUE_TIMEOUT_MS",
        3000,
        min_value=200,
        max_value=20000,
    )
    try:
        raw = await llm_runtime_mod.generate_text(
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
    self: "Services",
    query: str,
    candidates: list["Any"],
) -> "Any | None":
    """
    Выбирает лучший кандидат PREPARE по fast-score + LLM в серой зоне.

    :param query: запрос пользователя
    :param candidates: кандидаты из источников
    :return: лучший релевантный кандидат или None
    """

    ranked = _dedupe_prepare_candidates(candidates, limit=12)
    if not ranked:
        return None

    max_llm_checks = _common_mod._runtime_int(
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
            override_score = _common_mod._runtime_float(
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
            fallback_score = _common_mod._runtime_float(
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
    self: "Services",
    query: str,
    entities: dict[str, Any],
) -> list["Any"]:
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

    candidates: list[Any] = []
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


async def _maybe_compact_prepare_text(
    self: "Services",
    query: str,
    source_text: str,
) -> tuple[str, str, str]:
    """
    Компактирует длинный PREPARE-текст через LLM с безопасным fallback.

    :param query: исходный запрос пациента
    :param source_text: текст подготовки из источника
    :return: (итоговый текст, статус wrap, причина/диагностика)
    """

    text = str(source_text or "").strip()
    if not text:
        return "", "empty_source", "no_source_text"
    if not _common_mod._runtime_bool("MR_PREPARE_LLM_WRAP_ENABLED", True):
        return text, "disabled", "llm_wrap_disabled"

    min_chars = _common_mod._runtime_int(
        "MR_PREPARE_LLM_WRAP_MIN_CHARS",
        700,
        min_value=120,
        max_value=12000,
    )
    if len(text) < min_chars:
        return text, "short_source", "below_min_chars"

    source_max_chars = _common_mod._runtime_int(
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
            max_chars=_common_mod._runtime_int(
                "MR_PREPARE_FALLBACK_MAX_CHARS",
                1600,
                min_value=400,
                max_value=4000,
            ),
            max_points=_common_mod._runtime_int(
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

    timeout_s = _common_mod._runtime_int(
        "MR_PREPARE_LLM_WRAP_TIMEOUT_S",
        30,
        min_value=3,
        max_value=90,
    )
    queue_timeout_ms = _common_mod._runtime_int(
        "MR_PREPARE_LLM_WRAP_QUEUE_TIMEOUT_MS",
        6000,
        min_value=300,
        max_value=30000,
    )
    try:
        raw = await llm_runtime_mod.generate_text(
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


# Голый запрос подготовки: «правила подготовки», «как подготовиться»,
# «подготовка», «нужна ли подготовка». Без явного указания предмета
# подготовки.
_GENERIC_PREPARE_QUERY_RE = re.compile(
    r"^\s*(?:"
    r"правил\w*\s+подготовк\w*|"
    r"подготовк\w*(?:\s+к\s+анализ\w*|\s+к\s+процедур\w*)?|"
    r"как\s+(?:готовит\w*|подготовит\w*)|"
    r"нужн\w*\s+ли\s+подготовк\w*"
    r")\s*[.!?]?\s*$",
    re.I,
)

# Признак того, что в запросе есть конкретный предмет подготовки —
# название анализа, процедуры, специальности или общеупотребимая
# лабораторная аббревиатура.
_PREPARE_SUBJECT_HINT_RE = re.compile(
    r"\b(?:"
    # лабораторные аббревиатуры
    r"оак|оам|алт|аст|алат|асат|ттг|сое|соэ|мно|пти|ггт|"
    r"лдг|кфк|hba1c|hbsag|hcv|пцр|ифа|"
    # общие лабораторные термины
    r"анализ\w*|кров\w*|моч[аеи]\b|кал\b|биохим\w*|гормон\w*|"
    r"копроло\w*|спирометри\w*|узи|ультразвук\w*|кт\b|мрт\b|"
    r"рентген\w*|биопси\w*|гистолог\w*|ферритин\w*|глюкоз\w*|"
    r"холестерин\w*|инсулин\w*|витамин\w*|"
    # процедуры
    r"эндоскоп\w*|фгдс|фкс|колоноскоп\w*|кольпоскоп\w*|"
    r"вульвоскоп\w*|маммограф\w*|флюорограф\w*|урофлоуметр\w*|"
    r"экг|эхокардиограф\w*|эхокг|холтер\w*|"
    # популярные тесты
    r"гемоглобин\w*|тромбоцит\w*|лейкоцит\w*|эритроцит\w*|"
    # ключевые слова специальностей (узкий набор для signal-only)
    r"гинеколог\w*|уролог\w*|стоматолог\w*|кардиолог\w*"
    r")\b",
    re.I,
)


def _is_generic_prepare_query(text: str) -> bool:
    """Возвращает True для голых запросов про подготовку без предмета."""
    return bool(_GENERIC_PREPARE_QUERY_RE.match(str(text or "").strip()))


def _has_specific_subject_in_query(text: str) -> bool:
    """Возвращает True, если в запросе явно упомянут анализ/процедура."""
    return bool(_PREPARE_SUBJECT_HINT_RE.search(str(text or "")))


# Биоматериал по тексту — для отсечения конфликтующей stale-сущности.
# «общий анализ крови» (ОАК) и «общий анализ мочи» (ОАМ) — разные биоматериалы.
_PREPARE_BIOMATERIAL_RES: dict[str, re.Pattern[str]] = {
    "blood": re.compile(r"\bкров\w*", re.I),
    "urine": re.compile(r"\bмоч[аеиую]\w*", re.I),
    "feces": re.compile(r"\bкал\b|\bкопрол\w*", re.I),
}


def _prepare_biomaterial(text: str) -> str | None:
    """Определяет биоматериал (blood/urine/feces) по тексту или None."""
    raw = str(text or "")
    for material, pattern in _PREPARE_BIOMATERIAL_RES.items():
        if pattern.search(raw):
            return material
    return None


async def test_prepare(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    raw_query = str(query or "").strip()
    entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
    # Если пациент в этом ходе сам назвал биоматериал, а stale-сущность из
    # прошлого хода — про другой биоматериал (кейс «сдать кровь» при stale
    # «общий анализ мочи»), не тащим её: иначе в варианты/clarify подмешивается
    # чужой анализ и пациент видит подготовку «к общий анализ мочи».
    raw_material = _prepare_biomaterial(raw_query)
    stale_material = _prepare_biomaterial(entity_query)
    if entity_query and raw_material and stale_material and raw_material != stale_material:
        entities = {k: v for k, v in entities.items() if k not in ("test_name", "service_name")}
        entity_query = ""
    # Для нового вопроса берем текст пользователя, чтобы не залипала старая услуга из контекста.
    q = raw_query or entity_query
    if not q:
        return {"prepare": "", "note": "no query", "entities_used": entities}

    # Защита от голого запроса «правила подготовки» / «как
    # подготовиться» без указания анализа или процедуры. Раньше
    # бот в этом случае подтягивал stale `service_name`/`test_name`
    # из state предыдущей PRICE-турны и выдавал правила к чужому
    # анализу — заказчик 2026-05-05 видел, как после диалога про
    # ТТГ запрос «правила подготовки» возвращал инструкцию по
    # отбору эпителия из мочеиспускательного канала. Теперь —
    # явный clarify, чтобы пациент сам назвал предмет подготовки.
    if _is_generic_prepare_query(raw_query) and not _has_specific_subject_in_query(raw_query):
        return _prepare_clarify_response(
            raw_query,
            entities,
            note="prepare: generic query, asking for specific subject",
        )

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

    meili_candidates: list[Any] = []
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


async def _prepare_from_analysis_api_cache(
    self: "Services",
    query: str,
    entities: dict[str, Any],
) -> str | None:
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
