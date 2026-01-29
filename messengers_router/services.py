from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Services:
    """
    Сервисный слой: сюда подключим реальные данные (API/meili/репозитории).
    """

    async def doctors_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"doctors": [], "note": "stub doctors_info", "entities_used": entities}

    async def doctors_schedule_week(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"schedule": [], "note": "stub doctors_schedule_week", "entities_used": entities}

    async def appointment_help(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.", "entities_used": entities}

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"tests": [], "promos": [], "note": "stub test_assist", "entities_used": entities}

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"prepare": "", "note": "stub test_prepare", "entities_used": entities}

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"ready": False, "note": "stub test_result_status", "entities_used": entities}

    async def test_result_pdf(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"pdf": None, "note": "stub test_result_pdf", "entities_used": entities}

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"prices": [], "note": "stub price_info", "entities_used": entities}

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"addresses": [], "note": "stub address_info", "entities_used": entities}

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"news": [], "note": "stub news_info", "entities_used": entities}

    def get_branches(self) -> list[dict[str, str]]:
        """
        Возвращает справочник филиалов.
        Формат:
          [{"id":"branch_1","name":"Филиал на Пушкина","aliases":"пушкина,пушкинская,пушкина 10"}]
        Пока заглушка. Позже подключишь реальные данные.
        """
        return [
            {"id": "branch_pushkina", "name": "Филиал на Пушкина", "aliases": "пушкина,пушкинская,ул пушкина"},
            {"id": "branch_lenina", "name": "Филиал на Ленина", "aliases": "ленина,ул ленина,ленина 10"},
        ]