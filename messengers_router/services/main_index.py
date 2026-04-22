"""Модуль домена полнотекстового поиска по главной базе знаний.

Содержит миграцию ``main_index_info`` из ``services_legacy`` (Stage 21).
Метод обращается к meili-индексу ``main_index`` и формирует
унифицированный ответ с деградацией на tax-fallback и handoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from agent_logic_1 import meilisearch_client as meilisearch
from converters import html_cleaner

from ..policies import handoff_message
from ..policies import handoff_message as _handoff_message
from ._common import (
    _is_main_index_relevant,
    _is_meili_error_text,
    _is_meili_no_matches_text,
    _normalise_input,
    _service_fallback,
    _tax_doc_guidance_response,
)

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)

_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT = _handoff_message("knowledge_not_found")


async def main_index_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Полнотекстовый поиск по ``main_index`` с graceful degradation.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности (включая ``doc_request_kind``)
    :return: словарь с полем ``content`` либо handoff payload при сбое/отсутствии матчей
    """

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
            handoff_message=handoff_message("service_error_doctor_info"),
            entities=entities,
            extra={"content": ""},
        )

    if _is_meili_error_text(cleaned):
        if doc_kind == "tax":
            return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback error")
        return _service_fallback(
            note="main_index_info source unavailable",
            handoff_message=handoff_message("service_error_doctor_info"),
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
