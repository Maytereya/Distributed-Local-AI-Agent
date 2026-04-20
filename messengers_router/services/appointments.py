"""Модуль домена записи на приём.

Здесь живёт логика записи, переноса и отмены записи.
На шаге 6.1 модуль был создан как безопасная заглушка для будущей миграции;
на Stage 21 сюда перенесён первый реальный метод ``appointment_help``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)


def _legacy_module():
    """Лениво импортирует legacy-модуль, чтобы не создать цикл импортов.

    :return: модуль ``messengers_router.services_legacy``
    """

    from . import core as legacy

    return legacy


async def appointment_help(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    legacy = _legacy_module()

    if query:
        try:
            raw = await asyncio.to_thread(
                legacy.meilisearch.search_meili,
                "main_index",
                query,
                output_mode="content_only",
                max_chars=12000,
            )
            cleaned = legacy.html_cleaner.strip_html(raw)
        except Exception:
            return legacy._service_fallback(
                note="appointment_help source unavailable",
                handoff_message=legacy.handoff_message("service_error_appointments"),
                entities=entities,
                extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
            )
        if legacy._is_meili_error_text(cleaned):
            return legacy._service_fallback(
                note="appointment_help source unavailable",
                handoff_message=legacy.handoff_message("service_error_appointments"),
                entities=entities,
                extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
            )
        return {"instructions": cleaned, "entities_used": entities}
    return {
        "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
        "entities_used": entities,
    }
