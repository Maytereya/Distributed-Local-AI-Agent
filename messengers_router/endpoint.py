from __future__ import annotations

import json
from typing import Any, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .memory import MemoryStore
from .router import patient_routing_stream
from .services import Services
from .mess_types import ResponseEnvelope as InternalEnvelope

router = APIRouter()

memory = MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)
services = Services()


# ---------------------------------------------------------------------
# OpenAPI / Swagger схемы (Pydantic)
# ---------------------------------------------------------------------

class MessengerGenerateRequest(BaseModel):
    session_id: str = Field(
        default="anon",
        description="ID диалога/чата. Используйте один и тот же session_id для продолжения контекста.",
        examples=["s_test", "tg_123456789", "wa_7f2c9a"],
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Текст сообщения пользователя (пациента).",
        examples=["покажи расписание уролога", "как записаться на УЗИ", "у меня кровотечение"],
    )


class Attachment(BaseModel):
    type: Literal["pdf", "file", "url"] = Field(..., description="Тип вложения.")
    name: Optional[str] = Field(default=None, description="Имя файла/описание.")
    url: str = Field(..., description="URL для скачивания/просмотра.")


class ResponseEnvelopeOut(BaseModel):
    """
    Итоговый ответ (для debug endpoint без стрима).
    """
    text: str = Field(default="", description="Итоговый текст ответа (уже склеенный).")
    attachments: list[Attachment] = Field(default_factory=list, description="Вложения (если есть).")
    handoff: bool = Field(default=False, description="Нужно передать диалог оператору.")
    state_update: dict[str, Any] = Field(default_factory=dict, description="Опционально: обновления состояния.")


class ResponseEnvelopeLine(BaseModel):
    """
    ОДНА строка NDJSON (для streaming endpoint).
    """
    text: str = Field(default="", description="Фрагмент текста (delta). Клиент должен склеивать.")
    attachments: list[Attachment] = Field(default_factory=list, description="Вложения (если есть).")
    handoff: bool = Field(default=False, description="Сигнал передать оператору (обычно отдельной строкой в конце).")
    state_update: dict[str, Any] = Field(default_factory=dict, description="Опционально: обновления состояния клиента.")


# ---------------------------------------------------------------------
# Streaming endpoint (NDJSON)
# ---------------------------------------------------------------------

@router.post(
    "/api/messenger-generate",
    summary="Генерация ответа для мессенджеров (NDJSON stream)",
    description=(
        "Возвращает поток **NDJSON** (application/x-ndjson): каждая строка — JSON объект.\n\n"
        "Клиент (интегратор Telegram/WhatsApp) должен:\n"
        "1) читать строки по мере поступления\n"
        "2) склеивать `text` в итоговый ответ\n"
        "3) обработать `attachments`\n"
        "4) если получен `handoff=true` — передать диалог оператору\n"
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "NDJSON streaming response (каждая строка — ResponseEnvelopeLine).",
            "content": {
                "application/x-ndjson": {
                    "examples": {
                        "clarification_then_handoff": {
                            "summary": "Уточнение + сигнал handoff",
                            "value": (
                                '{"text":"Уточните, пожалуйста, на какую дату нужно расписание?","attachments":[],"handoff":false,"state_update":{}}\n'
                                '{"text":"","attachments":[],"handoff":true,"state_update":{}}\n'
                            ),
                        },
                        "pdf_attachment": {
                            "summary": "Отдача PDF вложения",
                            "value": (
                                '{"text":"Ваши результаты готовы.","attachments":[],"handoff":false,"state_update":{}}\n'
                                '{"text":"","attachments":[{"type":"pdf","name":"Результаты анализов.pdf","url":"https://.../result.pdf"}],"handoff":false,"state_update":{}}\n'
                            ),
                        },
                    }
                }
            },
        },
        422: {"description": "Validation error: неверный JSON или отсутствуют обязательные поля."},
    },
)
async def messenger_generate(payload: MessengerGenerateRequest):
    session_id = payload.session_id or "anon"
    text = payload.text.strip()

    state = await memory.aget(session_id)
    try:
        memory.append_turn(state, "user", text)

        async def event_stream():
            async for env in patient_routing_stream(text, state, services, memory):
                env: InternalEnvelope # Чтобы IDE понимала что за тип данных.
                obj = ResponseEnvelopeLine(
                    text=env.text or "",
                    attachments=env.attachments or [],
                    handoff=bool(env.handoff),
                    state_update=getattr(env, "state_update", {}) or {},
                ).model_dump()
                yield (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")
    finally:
        await memory.aset(state)


# ---------------------------------------------------------------------
# Debug endpoint (НЕ стримит) — удобно для Swagger / ручных тестов
# ---------------------------------------------------------------------

@router.post(
    "/api/messenger-generate-once",
    summary="Debug: генерация ответа одним JSON (без стрима)",
    description=(
        "Удобно для Swagger-понимания/ручных тестов.\n\n"
        "Внутри читает NDJSON-стрим и:\n"
        "- склеивает весь `text` в одну строку\n"
        "- собирает `attachments`\n"
        "- `handoff=true`, если в стриме был сигнал handoff\n"
    ),
    response_model=ResponseEnvelopeOut,
    responses={422: {"description": "Validation error: неверный JSON или отсутствуют обязательные поля."}},
)
async def messenger_generate_once(payload: MessengerGenerateRequest):
    session_id = payload.session_id or "anon"
    text = payload.text.strip()

    state = await memory.aget(session_id)
    try:
        memory.append_turn(state, "user", text)

        parts: list[str] = []
        attachments: list[Attachment] = []
        handoff = False
        state_update: dict[str, Any] = {}

        async for env in patient_routing_stream(text, state, services, memory):
            if env.text:
                parts.append(env.text)

            if env.attachments:
                for a in env.attachments:
                    attachments.append(Attachment.model_validate(a)) #Валидация типа, годная конкретно для Pydantic

            if env.handoff:
                handoff = True

            if getattr(env, "state_update", None):
                state_update = getattr(env, "state_update") or state_update

        out = ResponseEnvelopeOut(
            text="".join(parts).strip(),
            attachments=attachments,
            handoff=handoff,
            state_update=state_update,
        )
        # JSONResponse чтобы Swagger красиво показал тело
        return JSONResponse(content=out.model_dump())
    finally:
        await memory.aset(state)