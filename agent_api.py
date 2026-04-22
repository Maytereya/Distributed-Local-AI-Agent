from __future__ import annotations
import asyncio
import importlib
import json
from pathlib import Path
import sys
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
- **for-freetalk** — JSON debug endpoint для свободного многотурового диалога и eval
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


class FreeTalkRequest(BaseModel):
    """
    Запрос к FT-режиму.
    """

    session_id: str = Field(
        default="anon",
        description="ID диалога. Используйте один и тот же session_id для продолжения контекста.",
        examples=["ft_eval_case_001", "gr_ft_ab12cd34"],
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Текст сообщения пользователя.",
        examples=[
            "Покажи расписание Дразнина",
            "А на Ленина утром?",
            "Проверь результат анализа",
        ],
    )
    debug: bool = Field(
        default=False,
        description="Вернуть debug-поля: dialog_state, session entity memory, tool payload, history tail.",
        examples=[False, True],
    )


class FreeTalkSourceFragment(BaseModel):
    text: str = Field(default="", description="Фрагмент ответа.")
    source: str = Field(default="", description="Источник фрагмента: clinic_data/general_knowledge/web_search.")


class FreeTalkResponse(BaseModel):
    """
    Ответ FT debug endpoint.
    """

    text: str = Field(default="", description="Итоговый текст ответа.")
    session_id: str = Field(default="", description="Session id, использованный для этого запроса.")
    next_session_id: str = Field(
        default="",
        description="Session id, который нужно использовать для следующего шага диалога.",
    )
    source: str = Field(default="", description="Источник ответа: clinic_data/general_knowledge/mixed/system.")
    tool_name: str = Field(default="", description="Имя вызванного tool-а, если был tool_call.")
    reply_kind: Literal["clarify", "final", "session_rotate", "system"] = Field(
        default="final",
        description="Тип ответа: уточнение, финальный ответ, ротация сессии или системное сообщение.",
    )
    source_fragments: list[FreeTalkSourceFragment] = Field(
        default_factory=list,
        description="Трассировка источников ответа.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Диагностика FT: dialog_state, session_entity_memory, tool_payload, history_tail.",
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


def _freetalk_dialog_state_is_active(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    intent = str(payload.get("intent") or "").strip().lower()
    return bool(
        (intent and intent != "unknown")
        or payload.get("entities")
        or payload.get("missing_slots")
        or str(payload.get("clarify_type") or "").strip()
        or payload.get("tool_plan")
        or str(payload.get("confirmation_target") or "").strip()
        or str(payload.get("phase") or "").strip()
    )


def _infer_freetalk_reply_kind(
    *,
    source: str,
    session_id: str,
    next_session_id: str,
    dialog_state: dict[str, Any],
) -> Literal["clarify", "final", "session_rotate", "system"]:
    source_name = str(source or "").strip().lower()
    if source_name == "system":
        return "system"
    if next_session_id and next_session_id != session_id:
        return "session_rotate"
    if _freetalk_dialog_state_is_active(dialog_state):
        return "clarify"
    return "final"


def _import_freetalk_runner() -> Any:
    module_name = "localragagent.freetalk.runner"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "") or "") != "localragagent":
            raise
        src_dir = Path(__file__).resolve().parent / "src"
        if src_dir.exists():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
        return importlib.import_module(module_name)


async def _run_freetalk_once(
    *,
    text: str,
    session_id: str,
    debug: bool,
) -> FreeTalkResponse:
    free_talk_runner = _import_freetalk_runner()
    agent = free_talk_runner._agent()
    used_session_id = free_talk_runner.ensure_session_id(session_id)
    reply = await agent.chat(str(text or "").strip(), used_session_id)
    next_session_id = str(reply.next_session_id or "").strip() or used_session_id

    dialog_state_payload: dict[str, Any] = {}
    try:
        dialog_state = await agent._load_dialog_state(next_session_id)
        dialog_state_payload = agent._dialog_state_payload(dialog_state)
    except Exception:
        dialog_state_payload = {}

    debug_payload: dict[str, Any] = {}
    if debug:
        try:
            session_entity_memory = await agent._load_session_entity_memory(next_session_id)
        except Exception:
            session_entity_memory = {}
        try:
            context = await agent.memory.load_context(
                next_session_id,
                history_tail_turns=agent.config.history_tail_turns,
            )
            history_tail = list(context.turns or [])
            summary = str(context.summary or "")
        except Exception:
            history_tail = []
            summary = ""
        debug_payload = {
            "dialog_state": dialog_state_payload,
            "session_entity_memory": session_entity_memory,
            "tool_payload": dict(reply.tool_payload or {}),
            "history_tail": history_tail,
            "summary": summary,
        }

    return FreeTalkResponse(
        text=str(reply.text or "").strip(),
        session_id=used_session_id,
        next_session_id=next_session_id,
        source=str(reply.source or ""),
        tool_name=str(reply.tool_name or ""),
        reply_kind=_infer_freetalk_reply_kind(
            source=str(reply.source or ""),
            session_id=used_session_id,
            next_session_id=next_session_id,
            dialog_state=dialog_state_payload,
        ),
        source_fragments=[
            FreeTalkSourceFragment(
                text=str(fragment.get("text") or ""),
                source=str(fragment.get("source") or ""),
            )
            for fragment in (reply.source_fragments or [])
            if isinstance(fragment, dict)
        ],
        debug=debug_payload,
    )


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


@app.post(
    "/v1/freetalk/generate-once",
    tags=["for-freetalk"],
    summary="FT: один шаг свободного диалога одним JSON",
    description="""
JSON endpoint для **Free-talk-Ai**.

Назначение:
- автоматизированный eval FT,
- отладка многоходового поведения,
- внешние интеграции, которым не нужен streaming.

### Контракт
- хранит диалог по `session_id`;
- возвращает `next_session_id` на случай ротации сессии;
- в `reply_kind` различает `clarify/final/session_rotate/system`;
- при `debug=true` добавляет `dialog_state`, `session_entity_memory`, `tool_payload`, `history_tail`.

Подходит для проверки свойств:
- multiturn,
- clarification,
- self-error-correction,
- memory reuse,
- flexibility follow-up.
""",
    response_model=FreeTalkResponse,
    responses={
        200: {"description": "FT response envelope"},
        401: {"description": "Invalid API key"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def freetalk_generate_once(
    payload: FreeTalkRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Debug endpoint FT-режима.
    """

    verify_api_key(x_api_key)
    response = await _run_freetalk_once(
        text=payload.text,
        session_id=payload.session_id,
        debug=payload.debug,
    )
    return response
