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

from agent_logic_1 import meilisearch_client as meilisearch

if TYPE_CHECKING:
    from .core import Services


logger = logging.getLogger(__name__)


async def news_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Получает активные новости из meili-индекса ``news``.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя (используется как keyword)
    :param entities: извлечённые NLU-сущности
    :return: словарь с полем ``news`` (список) и ``entities_used``
    """

    try:
        hits = await asyncio.to_thread(
            meilisearch.search_news_active,
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
