"""Модуль домена подготовки к анализам/процедурам.

Содержит миграцию PREPARE-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, Sequence

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.nayka_api import api_service_info
from converters import html_cleaner

from .. import llm_runtime as llm_runtime_mod
from ..llm_doesnt_work_fallback import build_prepare_fallback_answer, join_wrapped_lines
from . import _common as _common_mod
from ._biomaterial import mis_synonym_candidates
from ._common import (
    _get_first_present,
    _normalise_catalog_text,
)
from ._prepare_select_llm import SOURCE_MIS, PrepareMemo, kb_patient_memos, select_prepare_memos
from ._prepare import (
    _PREPARE_RELEVANCE_VERDICT_IRRELEVANT,
    _PREPARE_RELEVANCE_VERDICT_RELEVANT,
    _PrepareCandidate,
    _dedupe_prepare_candidates,
    _has_prepare_strong_hints,
    _is_generic_prepare_heading,
    _is_prepare_content_actionable,
    _is_prepare_service_info_usable,
    _is_prepare_wrap_output_usable,
    _parse_prepare_relevance_validator,
    _prepare_clarify_response,
    _prepare_no_memo_response,
    _prepare_fast_relevance_score,
    _prepare_names_subject,
    _prepare_relevance_gate,
    _prepare_relevance_prompt,
    _prepare_roots_coverage,
    _prepare_service_info_queries,
    _prepare_subject_phrase,
    _prepare_term_roots,
    _prepare_wrap_clean,
    _prepare_wrap_prompt,
)

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)

# Сколько памяток МИС (лучших по совпадению слов) показывать LLM при выборе.
_PREPARE_MIS_CANDIDATES = 10

# Поля документа базы знаний, в которых лежит текст, — в порядке приоритета (как у
# `meilisearch_client.search_meili`).
_DOCUMENT_TEXT_FIELDS = ("content", "html", "csv", "description", "indication", "preparation", "body")
_DOCUMENT_MAX_CHARS = 12000


def _document_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, dict):
        return " ".join(str(v).strip() for v in value.values() if str(v).strip())
    return str(value).strip()


def _search_main_index_documents(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Найденные документы базы знаний клиники — каждый отдельно, с заголовком.

    `search_meili(..., output_mode="content_only")` склеивает найденное в один текст без
    границ. Склейку получает LLM-обёртка — она выбирает строки о процедуре пациента. Но
    запасной путь без LLM резал пункты из всей склейки, и подготовка к спирали получила
    строки памятки «ФКС + ФГДС» со слабительным и «Фортрансом»
    (BUG-2026-10-01-PREPARE-FALLBACK-MIXES-DOCS). Путям без LLM — один документ отсюда.
    Клиент Meilisearch — общий (`agent_logic_1.meilisearch_client`), его не меняем.

    :param query: поисковый запрос
    :param limit: сколько документов вернуть
    :return: [{"id", "title", "content"}]; ошибка поиска — исключение
    """

    client = meilisearch.get_meilisearch_client()
    meilisearch.ensure_default_indexes_once()
    hits = client.index("main_index").search(query, {"limit": limit}).get("hits", [])
    documents: list[dict[str, str]] = []
    for hit in hits if isinstance(hits, list) else []:
        if not isinstance(hit, dict):
            continue
        content = next((_document_text(hit.get(key)) for key in _DOCUMENT_TEXT_FIELDS if _document_text(hit.get(key))), "")
        if not content:
            continue
        documents.append(
            {
                "id": str(hit.get("id") or ""),
                "title": _document_text(hit.get("title")),
                "content": content[:_DOCUMENT_MAX_CHARS],
            }
        )
    return documents


def _best_single_document(query: str, variant: str) -> str:
    """Текст одного документа базы знаний, лучше всех подходящего к вопросу.

    Для путей без LLM: запасной ответ, короткий текст, выключенная обёртка. При равной
    оценке — порядок поиска. Ничего подходящего — пустая строка (тогда — склейка, как
    раньше: лучше прежний ответ, чем никакого).
    """

    try:
        documents = _search_main_index_documents(variant)
    except Exception:
        return ""
    best_text, best_score = "", 0.0
    for document in documents:
        text = html_cleaner.strip_html(document["content"]).strip()
        title = document.get("title") or ""
        score = max(
            _prepare_fast_relevance_score(query, text, title=title),
            _prepare_fast_relevance_score(variant, text, title=title),
        )
        if text and score > best_score:
            best_text, best_score = text, score
    return best_text


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
    Памятки МИС (serviceInfoAll) — кандидаты для выбора LLM: годные, лучшие первыми.

    :param query: исходный запрос пользователя
    :param entities: сущности роутера
    :return: список кандидатов
    """

    entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
    # Вариант без предмета («Сдаче крови», «Анализу» — такое кладёт в service_name
    # извлечение для записи) даёт 0.40 любой памятке с конкретными указаниями и
    # вытесняет из кандидатов настоящую: «Ферритин» (0.38) не доходил до LLM (стенд 02.10).
    queries = [variant for variant in _prepare_service_info_queries(query, entity_query) if _prepare_term_roots(variant)]
    if not queries:
        return []

    try:
        rows = await asyncio.to_thread(api_service_info.load_service_info)
    except Exception:
        return []

    # Услуги, которые клиника завела синонимом слова пациента («сахар» → «Глюкоза»), —
    # кандидаты первыми: в тексте их памятки слова пациента может не быть вовсе.
    subject = _prepare_subject_phrase(query)
    synonym_titles = {
        _normalise_catalog_text(name)
        for text in (query, entity_query, subject)
        if text
        for name in mis_synonym_candidates(text)
    }
    # Название карточки называет предмет вопроса — опора выбора без LLM (`_choose_without_llm`).
    subject_roots = _prepare_term_roots(subject)
    candidates: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        service_name = str(row.get("serviceName") or "").strip()
        preparation = str(row.get("preparation") or "").strip()
        if not service_name or not preparation:
            continue

        by_synonym = _normalise_catalog_text(service_name) in synonym_titles
        best_score = 1.0 if by_synonym else 0.0
        best_query = ""
        for query_variant in queries:
            score = _prepare_fast_relevance_score(query_variant, preparation, title=service_name)
            if score > best_score:
                best_score = score
                best_query = query_variant

        if best_score <= 0.0:
            continue
        # Синоним — слово самой клиники: предмет назван, нужна лишь содержательная памятка.
        # Проверка по словам вопроса отбросила бы «Глюкозу» на «сахар» — слова «сахар» в
        # её памятке нет.
        text = html_cleaner.strip_html(preparation).strip()
        if by_synonym:
            if not _is_prepare_content_actionable(text):
                continue
        elif not _is_prepare_service_info_usable(query, text, title=service_name):
            continue

        candidates.append(
            _PrepareCandidate(
                source="serviceInfoAll",
                text=preparation,
                query_variant=best_query or query,
                service_title=service_name,
                score=best_score,
                note="prepare: serviceInfoAll candidate",
                named=by_synonym or _prepare_names_subject(subject_roots, _prepare_term_roots(service_name)),
            )
        )

    return _dedupe_prepare_candidates(candidates, limit=16)


async def _maybe_compact_prepare_text(
    self: "Services",
    query: str,
    source_text: str,
    *,
    single_source: str = "",
) -> tuple[str, str, str]:
    """
    Компактирует длинный PREPARE-текст через LLM с безопасным fallback.

    :param query: исходный запрос пациента
    :param source_text: текст подготовки из источника (у базы знаний — склейка документов)
    :param single_source: один документ о процедуре пациента; им отвечают все пути без
        LLM, склейку видит только LLM-обёртка (BUG-2026-10-01-PREPARE-FALLBACK-MIXES-DOCS)
    :return: (итоговый текст, статус wrap, причина/диагностика)
    """

    text = str(source_text or "").strip()
    if not text:
        return "", "empty_source", "no_source_text"
    plain = str(single_source or "").strip() or text
    if not _common_mod._runtime_bool("MR_PREPARE_LLM_WRAP_ENABLED", True):
        return plain, "disabled", "llm_wrap_disabled"

    min_chars = _common_mod._runtime_int(
        "MR_PREPARE_LLM_WRAP_MIN_CHARS",
        700,
        min_value=120,
        max_value=12000,
    )
    if len(text) < min_chars:
        return plain, "short_source", "below_min_chars"

    source_max_chars = _common_mod._runtime_int(
        "MR_PREPARE_LLM_WRAP_SOURCE_MAX_CHARS",
        9000,
        min_value=500,
        max_value=30000,
    )
    source_for_prompt = text[:source_max_chars].strip()
    plain_for_fallback = plain[:source_max_chars].strip()

    def _fallback_or_source(reason: str) -> tuple[str, str, str]:
        compacted = build_prepare_fallback_answer(
            query,
            plain_for_fallback,
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
        if compacted and len(compacted) < len(plain):
            return compacted, "fallback_compact", reason
        if compacted:
            return plain, "fallback_not_shorter", reason
        return plain, "fallback_failed", reason

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


# Общий вопрос «во сколько / когда прийти сдать кровь/мочу/анализ» — это вопрос
# про ВРЕМЯ и МЕСТО сдачи, а не про правила подготовки конкретного анализа.
_LAB_VISIT_TIMING_RE = re.compile(
    r"(?:во\s*сколько|к\s*скольк\w*|до\s*скольк\w*|когда|в\s+какое\s+время|в\s+котор\w*\s+час\w*)",
    re.I,
)
_LAB_COLLECTION_ACTION_RE = re.compile(
    r"\b(сда(?:ть|вать|ча|чи|ю|ём|ем)\w*|прий[тд]\w*|приход\w*|подойти|подъехать|принима\w*)\b",
    re.I,
)


def _is_lab_visit_timing_query(text: str) -> bool:
    """True для общих вопросов «во сколько/когда прийти сдать кровь/мочу/анализ».

    Требует одновременно: сигнал времени/визита, действие сдачи/прихода и
    биоматериал. Конкретные «подготовка к ОАК/ТТГ» (без «во сколько/когда»)
    сюда НЕ попадают и идут обычным prepare-путём.
    """
    raw = str(text or "")
    if not (_LAB_VISIT_TIMING_RE.search(raw) and _LAB_COLLECTION_ACTION_RE.search(raw)):
        return False
    return bool(_prepare_biomaterial(raw))


# Фиксированная памятка о сдаче крови (предоставлена клиникой) — добавляется
# к ответу на вопросы вида «во сколько сдать кровь».
_BLOOD_COLLECTION_GUIDANCE = (
    "Кровь сдается без записи, в порядке живой очереди, строго натощак, после "
    "ночного голодания. Пить можно простую, не газированную воду. Обратиться "
    "можно в любое отделение нашей клиники. Для оформления договора при себе "
    "иметь паспорт."
)


async def _lab_collection_branches_answer(self: "Services", query: str = "") -> str:
    """Ответ про время/место сдачи: памятка (для крови) + самарские филиалы с графиком.

    Отдаём готовый текст без LLM и без выдумывания медфактов: памятка по крови —
    фиксированная, адреса/часы — из ``/regions``.
    """
    from ._addresses_helpers import _looks_like_real_address
    from ._regions import (
        _extract_region_work_time,
        _is_samara_city_value,
        _region_display_name,
    )

    norm = _common_mod._normalise_input
    try:
        regions = await self._ensure_regions_loaded()
    except Exception:
        regions = []

    lines: list[str] = []
    seen: set[str] = set()
    for r in regions:
        if not isinstance(r, dict):
            continue
        # Только Самара (та же логика, что в address_info).
        if not (
            _is_samara_city_value(str(r.get("city") or ""))
            or "самара" in norm(str(r.get("name") or ""))
            or "самара" in norm(str(r.get("addressForSite") or ""))
        ):
            continue
        disp = _region_display_name(r)
        if not disp or not _looks_like_real_address(disp):
            continue
        key = disp.strip()
        if key in seen:
            continue
        seen.add(key)
        work_time = _extract_region_work_time(r)
        lines.append(f"— {disp}" + (f" (график: {work_time})" if work_time else ""))

    parts: list[str] = []
    if _prepare_biomaterial(query) == "blood":
        parts.append(_BLOOD_COLLECTION_GUIDANCE)
    if lines:
        lines.sort()
        parts.append("Адреса филиалов в Самаре и часы их работы:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _several_memos_answer(memos: Sequence[PrepareMemo]) -> str:
    """Несколько памяток в одном ответе — каждая целиком под своим названием.

    Склейка без границ уже давала пациенту подготовку к чужой процедуре
    (BUG-2026-10-01-PREPARE-FALLBACK-MIXES-DOCS), поэтому каждая памятка — отдельным
    блоком; длинную сжимаем без LLM и только из её собственного текста.
    """

    max_chars = _common_mod._runtime_int("MR_PREPARE_FALLBACK_MAX_CHARS", 1600, min_value=400, max_value=4000)
    max_points = _common_mod._runtime_int("MR_PREPARE_FALLBACK_MAX_POINTS", 7, min_value=3, max_value=10)
    parts = ["Подходят несколько памяток клиники — посмотрите ту, что про ваш анализ."]
    for memo in memos:
        heading = (memo.title if memo.title.startswith("«") else f"«{memo.title}»") + ":"
        lines = [line for line in join_wrapped_lines(memo.text) if not _is_generic_prepare_heading(line)]
        full = "\n".join(lines)
        if len(full) <= max_chars:
            parts.append(f"{heading}\n{full}")
            continue
        short = build_prepare_fallback_answer(
            memo.title, memo.text, max_chars=max_chars, max_points=max_points, intro=heading
        )
        parts.append(short or f"{heading}\n{full[:max_chars].rstrip()}…")
    return "\n\n".join(parts)


async def test_prepare(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    raw_query = str(query or "").strip()
    # «Во сколько/когда прийти сдать кровь» — вопрос про время/место сдачи,
    # а не про подготовку. Отдаём адреса филиалов с графиком работы вместо
    # clarify-петли «к какому анализу нужна подготовка?».
    if _is_lab_visit_timing_query(raw_query):
        branches_text = await _lab_collection_branches_answer(self, raw_query)
        if branches_text:
            return {
                "prepare": branches_text,
                "note": "prepare: lab-collection branches+hours",
                "entities_used": entities,
            }
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

    # Памятку выбирает LLM: предмет памятки — процедура пациента, а не то, что в ней
    # упомянуто (BUG-2026-10-01-PREPARE-MENTION-AS-SUBJECT). Кандидаты — памятки МИС и,
    # в переходный период, памятки пациенту из базы знаний; скрипты администраторов в
    # подготовку не попадают (решения владельца 02.10).
    mis_candidates = await self._prepare_candidates_from_analysis_api_cache(q, entities)
    memos = [
        PrepareMemo(
            source=SOURCE_MIS,
            title=cand.service_title,
            text=html_cleaner.strip_html(cand.text).strip(),
            score=cand.score,
            named=cand.named,
        )
        for cand in mis_candidates[:_PREPARE_MIS_CANDIDATES]
    ]
    memos.extend(await asyncio.to_thread(kb_patient_memos))
    chosen = await select_prepare_memos(
        q, memos, runtime_llm_mode=str(entities.get("__runtime_llm_mode") or "")
    )
    if len(chosen) == 1:
        memo = chosen[0]
        compacted, wrap_status, wrap_reason = await self._maybe_compact_prepare_text(
            q, memo.text, single_source=memo.text
        )
        return {
            "prepare": compacted,
            "note": f"prepare: {memo.source}",
            "prepare_source_title": memo.title,
            "entities_used": entities,
            "prepare_wrap_status": wrap_status,
            "prepare_wrap_reason": wrap_reason,
        }
    if chosen:
        # Без LLM при ничьей — обе памятки (решение владельца 03.10), каждая под своим
        # названием и только своим текстом; LLM-обёртку не зовём — она только что не ответила.
        return {
            "prepare": _several_memos_answer(chosen),
            "note": "prepare: several memos without llm",
            "prepare_source_title": " | ".join(memo.title for memo in chosen),
            "entities_used": entities,
            "prepare_wrap_status": "several_memos",
            "prepare_wrap_reason": "tie_without_llm",
        }

    # Конкретные правила подготовки не нашлись. Если вопрос про сдачу КРОВИ —
    # отдаём общую памятку забора крови (фиксированная, предоставлена клиникой):
    # эти правила применимы к ЛЮБОЙ сдаче крови (живая очередь, строго натощак,
    # паспорт). Для мочи/кала и пр. общей памятки нет — предлагаем оператора.
    # Универсально: триггер — биоматериал «кровь» в тексте запроса.
    if _prepare_biomaterial(q) == "blood" or raw_material == "blood":
        return {
            "prepare": _BLOOD_COLLECTION_GUIDANCE,
            "note": "prepare: blood-collection general guidance fallback",
            "entities_used": entities,
        }

    note = "prepare: no memo about the subject" if memos else "prepare: no matches"
    return _prepare_no_memo_response(q, entities, note=note)
