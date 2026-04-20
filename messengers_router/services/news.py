"""Модуль домена новостей клиники.

Содержит миграцию ``news_info`` из ``services_legacy`` (Stage 21).
Метод обращается к meili-индексу ``news`` и возвращает активные новости;
деградация источника считается некритичной — возвращается пустой список
без принудительного handoff.
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


async def news_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Получает активные новости из meili-индекса ``news``.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя (используется как keyword)
    :param entities: извлечённые NLU-сущности
    :return: словарь с полем ``news`` (список) и ``entities_used``
    """

    legacy = _legacy_module()

    try:
        hits = await asyncio.to_thread(
            legacy.meilisearch.search_news_active,
            index_name="news",
            keyword=query or None,
            limit=10,
            sort=["from_ts:desc"],
        )
    except Exception:
        # Для новостей деградация источника не критична: возвращаем пустой ответ
        # без принудительного handoff.
        return {
            "news": [],
            "note": "news source unavailable",
            "entities_used": entities,
        }
    return {"news": hits, "entities_used": entities}
