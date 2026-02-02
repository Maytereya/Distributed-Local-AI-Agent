from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from messengers_router.endpoint import router as messenger_router  # <-- путь к генератору ответов
# для мессенджеров
import agent_logic_2.config as c
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

SessionType = Dict[str, Any]

app = FastAPI(title="Neiry Agent API", version="0.1")
app.include_router(messenger_router, tags=["for-messengers"])

API_KEY = c.AGENT_API_KEY.strip()


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: AGENT_API_KEY is empty")
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class AgentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session: Optional[SessionType] = None  # клиент хранит session_state

    extra_processing: Literal["direct", "processed"] = "processed"
    think: bool = False   # ← ВАЖНО
    ai_feed: Literal["local", "cloud"] = "local"

    meta: Dict[str, Any] = Field(default_factory=dict)


def sse_event(event: str, data: Any) -> str:
    # SSE-формат: event + data(JSON)
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/agent/stream", tags=["for-call-center"])
async def stream_agent(payload: AgentRequest, request: Request, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    from agent_logic_2.router_preprocessor import routing  # lazy import чтобы не спамить в логи, если он не дергается намеренно
    trace_id = str(uuid.uuid4())
    sess: SessionType = payload.session or {}

    async def gen() -> AsyncGenerator[str, None]:
        final_sess: SessionType = sess

        yield sse_event("meta", {"trace_id": trace_id})

        try:
            async for partial, final_sess in routing(
                payload.text,
                sess=sess,
                extra_processing=payload.extra_processing,
                think=payload.think,
                ai_feed=payload.ai_feed,
            ):
                if await request.is_disconnected():
                    return
                yield sse_event("chunk", {"text": partial})

            yield sse_event("done", {"session": final_sess})

        except asyncio.CancelledError:
            return
        except Exception as e:
            yield sse_event("error", {"message": str(e), "trace_id": trace_id})

    return StreamingResponse(gen(), media_type="text/event-stream")
