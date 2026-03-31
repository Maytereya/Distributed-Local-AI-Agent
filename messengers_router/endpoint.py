"""HTTP endpoint-слой мессенджерного контура.

Определяет streaming и debug-once точки входа, читает/сохраняет session state
и преобразует внутренние envelope-ответы в публичный JSON/NDJSON контракт.
Ответственность модуля: только API-контракт и жизненный цикл сессии на уровне HTTP.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .llm_mode_policy import normalize_runtime_options
from .memory import MemoryStore
from .router import patient_routing_stream
from .services import Services
from .policies import handoff_message

router = APIRouter()


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    return MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)


@lru_cache(maxsize=1)
def get_services() -> Services:
    services = Services()
    try:
        services.ensure_background_refresh_started()
    except Exception:
        pass
    return services


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
    debug: bool = Field(
        default=False,
        description="Debug-режим. Для /api/messenger-generate игнорируется (не добавляет debug в стрим). "
                    "Для /api/messenger-generate-once возвращает диагностику в state_update.",
        examples=[False, True],
    )
    llm_mode: Literal["strict", "hybrid", "rich"] = Field(
        default="hybrid",
        description=(
            "Режим участия LLM: strict — legacy/fallback NLU, "
            "hybrid — llm_primary NLU, rich — llm_primary NLU + self-check/refine."
        ),
        examples=["hybrid", "rich"],
    )
    self_check: bool = Field(
        default=False,
        description="Дополнительная self-check проверка ответа (используется только в rich-режиме).",
        examples=[False, True],
    )
    self_check_max_retries: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Сколько раз перегенерировать ответ после неуспешной self-check.",
        examples=[1],
    )
    queue_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Максимальное ожидание в очереди LLM (мс).",
        examples=[30000],
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
    state_update: dict[str, Any] = Field(default_factory=dict, description="Диагностика/обновления состояния (только debug-once).")


class ResponseEnvelopeLine(BaseModel):
    """
    ОДНА строка NDJSON (для streaming endpoint).
    """
    text: str = Field(default="", description="Фрагмент текста (delta). Клиент должен склеивать.")
    attachments: list[Attachment] = Field(default_factory=list, description="Вложения (если есть).")
    handoff: bool = Field(default=False, description="Сигнал передать оператору (обычно отдельной строкой в конце).")
    state_update: dict[str, Any] = Field(default_factory=dict, description="(В стриме не используется).")


# ---------------------------------------------------------------------
# Streaming endpoint (NDJSON)
# Debug-флаг принимаем, но намеренно игнорируем (чтобы не ломать интеграции).
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
        "4) если получен `handoff=true` — передать диалог оператору\n\n"
        "Примечание: параметр `debug` в этом endpoint игнорируется (debug-данные в стрим не добавляются). "
        "Для диагностики используйте /api/messenger-generate-once с debug=true."
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
                        "result_link": {
                            "summary": "Отдача ссылки на результат в тексте",
                            "value": (
                                '{"text":"Ваши результаты готовы.","attachments":[],"handoff":false,"state_update":{}}\n'
                                '{"text":"Ссылка на результат: https://naykalab.ru/getanaliz.php?fam=...","attachments":[],"handoff":false,"state_update":{}}\n'
                            ),
                        },
                    }
                }
            },
        },
        422: {"description": "Validation error: неверный JSON или отсутствуют обязательные поля."},
    },
)
async def messenger_generate(
    payload: MessengerGenerateRequest,
    memory: MemoryStore = Depends(get_memory_store),
    services: Services = Depends(get_services),
):
    session_id = payload.session_id or "anon"

    # Защищаемся от строк, содержащих внезапные символьные суррогаты.
    text = payload.text.strip()
    text = text.encode("utf-8", "ignore").decode("utf-8")

    runtime_options = normalize_runtime_options(
        llm_mode=payload.llm_mode,
        self_check=payload.self_check,
        self_check_max_retries=payload.self_check_max_retries,
        queue_timeout_ms=payload.queue_timeout_ms,
    )

    state = await memory.aget(session_id)
    memory.append_turn(state, "user", text)

    async def event_stream():
        # debug=False — осознанно
        assistant_parts: list[str] = []
        try:
            async for env in patient_routing_stream(
                text,
                state,
                services,
                memory,
                debug=False,
                runtime_options=runtime_options,
            ):
                if env.text:
                    assistant_parts.append(str(env.text))
                obj = {
                    "text": env.text or "",
                    "attachments": env.attachments or [],
                    "handoff": bool(env.handoff),
                    "state_update": {},  # в стриме не используем
                }
                yield (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        except Exception:
            fallback = {
                "text": handoff_message("service_error"),
                "attachments": [],
                "handoff": True,
                "state_update": {},
            }
            yield (json.dumps(fallback, ensure_ascii=False) + "\n").encode("utf-8")
        finally:
            assistant_text = "".join(assistant_parts).strip()
            if assistant_text:
                memory.append_turn(state, "assistant", assistant_text)
            await memory.aset(state)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------
# Debug endpoint (НЕ стримит) — удобно для Swagger / ручных тестов
# Включает debug-метаданные в state_update (если payload.debug=true).
# ---------------------------------------------------------------------

@router.post(
    "/api/messenger-generate-once",
    summary="Debug: генерация ответа одним JSON (без стрима)",
    description=(
        "Удобно для Swagger/ручных тестов.\n\n"
        "Внутри читает поток и:\n"
        "- склеивает весь `text` в одну строку\n"
        "- собирает `attachments`\n"
        "- `handoff=true`, если был сигнал handoff\n\n"
        "Если `debug=true`, дополнительно вернёт диагностику в `state_update` "
        "(`decision/plan/evidence/pending/history_tail/nlu_trace`)."
    ),
    response_model=ResponseEnvelopeOut,
    responses={422: {"description": "Validation error: неверный JSON или отсутствуют обязательные поля."}},
)
async def messenger_generate_once(
    payload: MessengerGenerateRequest,
    memory: MemoryStore = Depends(get_memory_store),
    services: Services = Depends(get_services),
):
    session_id = payload.session_id or "anon"
    text = payload.text.strip()
    text = text.encode("utf-8", "ignore").decode("utf-8")

    runtime_options = normalize_runtime_options(
        llm_mode=payload.llm_mode,
        self_check=payload.self_check,
        self_check_max_retries=payload.self_check_max_retries,
        queue_timeout_ms=payload.queue_timeout_ms,
    )

    state = await memory.aget(session_id)
    try:
        memory.append_turn(state, "user", text)

        parts: list[str] = []
        attachments: list[Attachment] = []
        handoff = False
        state_update: dict[str, Any] = {}

        try:
            async for env in patient_routing_stream(
                text,
                state,
                services,
                memory,
                debug=payload.debug,
                runtime_options=runtime_options,
            ):
                if env.text:
                    parts.append(env.text)

                if env.attachments:
                    for a in env.attachments:
                        attachments.append(Attachment.model_validate(a))

                if env.handoff:
                    handoff = True

                # если debug включён — patient_routing_stream отдаст meta через state_update (обычно первым env)
                if payload.debug and getattr(env, "state_update", None):
                    su = env.state_update or {}
                    # берём целиком (это будет {"debug": {...}})
                    if su:
                        state_update = su
        except Exception as e:
            parts = [handoff_message("service_error")]
            attachments = []
            handoff = True
            if payload.debug:
                state_update = {"debug": {"endpoint_error": str(e)}}

        out = ResponseEnvelopeOut(
            text="".join(parts).strip(),
            attachments=attachments,
            handoff=handoff,
            state_update=state_update,
        )
        if out.text:
            memory.append_turn(state, "assistant", out.text)
        # JSONResponse чтобы Swagger красиво показал тело
        return JSONResponse(content=out.model_dump())
    finally:
        await memory.aset(state)
