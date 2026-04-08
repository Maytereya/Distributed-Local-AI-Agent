"""Port to legacy clinic-data services for FreeTalk."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Awaitable, Callable

from .legacy import import_legacy_alias

import_legacy_alias("messengers_router")


ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


@lru_cache(maxsize=1)
def _services_instance() -> Any:
    endpoint_mod = importlib.import_module("localragagent.messengers_router.endpoint")
    return endpoint_mod.get_services()


class LegacyServicesPort:
    def __init__(self) -> None:
        self._services = _services_instance()

    async def get_catalog_health(self) -> dict[str, Any]:
        return await self._services.get_catalog_health()

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, Any]:
        return await self._services.match_catalog_service(raw_text_or_name, current_service_name=current_service_name)

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, Any]:
        return await self._services.match_catalog_doctor(raw_text_or_name)

    async def service_bundle_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.service_bundle_info(query, entities)

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.price_info(query, entities)

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.test_prepare(query, entities)

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.test_assist(query, entities)

    async def doctors_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.doctors_info(query, entities)

    async def doctors_schedule_week(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.doctors_schedule_week(query, entities)

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.address_info(query, entities)

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.test_result_status(query, entities)

    async def main_index_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.main_index_info(query, entities)

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return await self._services.news_info(query, entities)

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, ToolHandler]:
        handlers: dict[str, ToolHandler] = {
            "service_bundle_info": self.service_bundle_info,
            "price_info": self.price_info,
            "test_prepare": self.test_prepare,
            "test_assist": self.test_assist,
            "doctors_info": self.doctors_info,
            "doctors_schedule_week": self.doctors_schedule_week,
            "address_info": self.address_info,
            "test_result_status": self.test_result_status,
        }
        if include_meili_tools:
            handlers["main_index_info"] = self.main_index_info
            handlers["news_info"] = self.news_info
        return handlers

