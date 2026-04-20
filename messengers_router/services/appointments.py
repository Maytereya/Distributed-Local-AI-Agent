"""Модуль домена записи на приём.

Здесь живёт логика записи, переноса и отмены записи.
На шаге 6.1 модуль был создан как безопасная заглушка для будущей миграции;
на Stage 21 сюда перенесён первый реальный метод ``appointment_help``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from agent_logic_1 import meilisearch_client as meilisearch
from converters import html_cleaner

from ..policies import handoff_message
from ._common import _is_meili_error_text, _service_fallback

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)


async def appointment_help(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    if query:
        try:
            raw = await asyncio.to_thread(
                meilisearch.search_meili,
                "main_index",
                query,
                output_mode="content_only",
                max_chars=12000,
            )
            cleaned = html_cleaner.strip_html(raw)
        except Exception:
            return _service_fallback(
                note="appointment_help source unavailable",
                handoff_message=handoff_message("service_error_appointments"),
                entities=entities,
                extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
            )
        if _is_meili_error_text(cleaned):
            return _service_fallback(
                note="appointment_help source unavailable",
                handoff_message=handoff_message("service_error_appointments"),
                entities=entities,
                extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
            )
        return {"instructions": cleaned, "entities_used": entities}
    return {
        "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
        "entities_used": entities,
    }
