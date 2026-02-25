from __future__ import annotations
import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Dict, Literal, Optional
from fastapi import FastAPI, Header, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from messengers_router.endpoint import router as messenger_router
import agent_logic_2.config as c

# ------------------------------------------------------------------------------
# API setup
# ------------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
SessionType = Dict[str, Any]

app = FastAPI(
    title="Neiry Agent API",
    version="0.2",
    description="""
API для работы с ИИ-агентом клиники.
### Группы эндпоинтов
- **for-messengers** — протокол NDJSON для Telegram / WhatsApp / Web chat
- **for-call-center** — SSE-протокол для операторов и админских интерфейсов
⚠️ Протоколы отличаются намеренно.
""",
)

# роуты мессенджеров
app.include_router(
    messenger_router,
    tags=["for-messengers"],
)

EXPECTED_API_KEY = c.AGENT_API_KEY.strip()

# ------------------------------------------------------------------------------
# Security
# ------------------------------------------------------------------------------
def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if not EXPECTED_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: AGENT_API_KEY is empty",
        )
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class AgentRequest(BaseModel):
    """
    Запрос к агенту для операторского интерфейса (call-center).
    """
    text: str = Field(
        ...,
        min_length=1,
        description="Текст сообщения оператора или клиента",
        example="Покажи расписание уролога",
    )
    session: Optional[SessionType] = Field(
        default=None,
        description=(
            "Состояние диалога для операторских интерфейсов. "
            "Используется только в call-center API. "
            "В мессенджерных endpoint'ах память хранится на сервере "
            "и определяется по session_id."
        ),
        example={},
    )
    extra_processing: Literal["direct", "processed"] = Field(
        default="processed",
        description="Режим дополнительной обработки ответа",
    )
    think: bool = Field(
        default=False,
        description="Включить think-режим модели (если поддерживается)",
    )
    ai_feed: Literal["local", "cloud"] = Field(
        default="local",
        description="Источник LLM",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Произвольные метаданные клиента",
    )

# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
SSE_MEDIA_TYPE = "text/event-stream"

SSE_EVENT_META = "meta"
SSE_EVENT_CHUNK = "chunk"
SSE_EVENT_DONE = "done"
SSE_EVENT_ERROR = "error"


def format_sse_event(event: str, data: Any) -> str:
    """
    Формирует SSE-сообщение.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _agent_sse_stream(
    *,
    routing: Any,
    request: Request,
    text: str,
    sess: SessionType,
    extra_processing: Literal["direct", "processed"],
    think: bool,
    ai_feed: Literal["local", "cloud"],
    trace_id: str,
) -> AsyncGenerator[str, None]:
    """
    Единая точка генерации SSE-событий для streaming-endpoint'а колл-центра.
    """
    final_sess: SessionType = sess

    yield format_sse_event(SSE_EVENT_META, {"trace_id": trace_id})

    try:
        async for partial, final_sess in routing(
            text,
            sess=sess,
            extra_processing=extra_processing,
            think=think,
            ai_feed=ai_feed,
        ):
            if await request.is_disconnected():
                return
            yield format_sse_event(SSE_EVENT_CHUNK, {"text": partial})

        yield format_sse_event(SSE_EVENT_DONE, {"session": final_sess})
    except asyncio.CancelledError:
        return
    except Exception as e:
        yield format_sse_event(
            SSE_EVENT_ERROR,
            {"message": str(e), "trace_id": trace_id},
        )

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.post(
    "/v1/agent/stream",
    tags=["for-call-center"],
    summary="SSE-стрим генерации ответа (для операторов)",
    description="""
Потоковый SSE endpoint для **колл-центра / админских интерфейсов**.

### Протокол
- Content-Type: `text/event-stream`
- События:
  - `meta` — служебная информация (trace_id)
  - `chunk` — фрагмент текста ответа
  - `done` — завершение ответа + обновлённая session
  - `error` — ошибка

### Отличия от мессенджеров
- SSE вместо NDJSON
- Нет `handoff`
- Ответ ориентирован на операторов колл-центра
""",
    responses={
        200: {
            "description": "SSE stream started",
        },
        401: {
            "description": "Invalid API key",
        },
        500: {
            "description": "Internal server error",
        },
    },
)
async def stream_agent(
    payload: AgentRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Основной streaming-endpoint для операторов.
    """
    verify_api_key(x_api_key)

    # lazy import — чтобы не инициализировать логику без надобности
    from agent_logic_2.router_preprocessor import routing

    trace_id = str(uuid.uuid4())
    initial_sess: SessionType = payload.session or {}
    text = payload.text

    gen = _agent_sse_stream(
        routing=routing,
        request=request,
        text=text,
        sess=initial_sess,
        extra_processing=payload.extra_processing,
        think=payload.think,
        ai_feed=payload.ai_feed,
        trace_id=trace_id,
    )
    return StreamingResponse(gen, media_type=SSE_MEDIA_TYPE)
