"""Модуль домена полнотекстового поиска по главной базе знаний.

Содержит миграцию ``main_index_info`` из ``services_legacy`` (Stage 21).
Метод обращается к meili-индексу ``main_index`` и формирует
унифицированный ответ с деградацией на tax-fallback и handoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ..policies import handoff_message as _handoff_message

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)

_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT = _handoff_message("knowledge_not_found")


def _legacy_module():
    """Лениво импортирует legacy-модуль, чтобы не создать цикл импортов.

    :return: модуль ``messengers_router.services_legacy``
    """

    from . import core as legacy

    return legacy


async def main_index_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Полнотекстовый поиск по ``main_index`` с graceful degradation.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности (включая ``doc_request_kind``)
    :return: словарь с полем ``content`` либо handoff payload при сбое/отсутствии матчей
    """

    legacy = _legacy_module()

    q = str(query or "").strip()
    if not q:
        return {
            "content": "",
            "note": "main_index_info: no query",
            "entities_used": entities,
        }
    doc_kind = str(entities.get("doc_request_kind") or "").strip().lower()
    if doc_kind not in {"tax", "generic"}:
        norm_q = legacy._normalise_input(q)
        doc_kind = "tax" if any(k in norm_q for k in ("налог", "вычет", "фнс")) else "generic"

    if doc_kind == "tax":
        return legacy._tax_doc_guidance_response(entities, note="main_index_info: tax direct link")

    normalized_q = legacy._normalise_input(q)
    fallback_queries: list[str] = []
    if doc_kind == "tax" and any(k in normalized_q for k in ("налог", "фнс", "вычет", "справк")):
        fallback_queries = [
            "справка для налоговой",
            "налоговый вычет",
            "справка об оплате медицинских услуг",
        ]

    queries = [q]
    for fq in fallback_queries:
        if legacy._normalise_input(fq) != normalized_q:
            queries.append(fq)

    cleaned = ""
    relevant_hit = False
    try:
        for qq in queries:
            raw = await asyncio.to_thread(
                legacy.meilisearch.search_meili,
                "main_index",
                qq,
                output_mode="content_only",
                max_chars=12000,
            )
            cleaned = legacy.html_cleaner.strip_html(raw).strip()
            if legacy._is_meili_error_text(cleaned):
                continue
            if legacy._is_meili_no_matches_text(cleaned):
                continue
            if legacy._is_main_index_relevant(qq, cleaned, doc_kind=doc_kind):
                relevant_hit = True
                break
    except Exception:
        if doc_kind == "tax":
            return legacy._tax_doc_guidance_response(entities, note="main_index_info: tax fallback unavailable")
        return legacy._service_fallback(
            note="main_index_info source unavailable",
            handoff_message=legacy.handoff_message("service_error_doctor_info"),
            entities=entities,
            extra={"content": ""},
        )

    if legacy._is_meili_error_text(cleaned):
        if doc_kind == "tax":
            return legacy._tax_doc_guidance_response(entities, note="main_index_info: tax fallback error")
        return legacy._service_fallback(
            note="main_index_info source unavailable",
            handoff_message=legacy.handoff_message("service_error_doctor_info"),
            entities=entities,
            extra={"content": ""},
        )

    if legacy._is_meili_no_matches_text(cleaned):
        if doc_kind == "tax":
            return legacy._tax_doc_guidance_response(entities, note="main_index_info: tax fallback no matches")
        return legacy._service_fallback(
            note="main_index_info: no matches",
            handoff_message=_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT,
            entities=entities,
            reason="knowledge_not_found",
            extra={"content": ""},
        )

    if not relevant_hit:
        if doc_kind == "tax":
            return legacy._tax_doc_guidance_response(entities, note="main_index_info: tax fallback weak relevance")
        return legacy._service_fallback(
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
