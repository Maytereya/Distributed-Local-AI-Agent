"""Единый runtime для вызовов локальной LLM с ограничением конкуренции.

Ответственность модуля: оборачивать вызовы Ollama, контролировать очередь/таймауты
и отдавать безопасные helper-функции для text/json/stream генерации.
"""

from __future__ import annotations

import asyncio
import json as _json
import os as _os
import sys as _sys
import time as _time
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path as _Path
from typing import Any, AsyncGenerator

import httpx
from ollama import AsyncClient, RequestError, ResponseError

from agent_logic_2 import ollama_settings
from agent_logic_2.ollama_settings import LLMName

from . import turn_telemetry
from .runtime_config import config as c

_DEFAULT_MAX_CONCURRENCY = 5
try:
    _MAX_CONCURRENCY = int(str(c.MR_LLM_MAX_CONCURRENCY).strip() or _DEFAULT_MAX_CONCURRENCY)
except Exception:
    _MAX_CONCURRENCY = _DEFAULT_MAX_CONCURRENCY
_SEMAPHORE = asyncio.Semaphore(max(1, _MAX_CONCURRENCY))

# Параметры коротких решений (валидаторы, различители: ответ — одно слово). Явно на месте
# вызова, а не из общих настроек Gradio: одинаковый вопрос — одинаковое решение, ответ не
# растекается (ревью DLA, L-03/LLM-2, 08.10).
DECISION_OPTIONS: dict[str, Any] = {"temperature": 0, "num_predict": 24}
_OLLAMA_CLIENT = AsyncClient(c.ollama_url)


# --- DEBUG: файловый лог промптов и ответов ---
# Включается env-переменной MR_LLM_LOG_PROMPTS=1.
# Пишет одну JSON-line на каждое событие в /tmp/llm_prompts.jsonl
# События: prompt_request (до вызова), prompt_response (после успеха),
#          prompt_timeout (на TimeoutError), prompt_error (на любое исключение).
# Цель: захватить реальный промпт даже при зависании LLM, сравнить
# отличие между быстрыми ("Расписание хирург") и медленными
# ("Расписание уролог") запросами.
_PROMPT_LOG_PATH = _Path(_os.getenv("MR_LLM_LOG_PATH", "/tmp/llm_prompts.jsonl"))
# Маркер-файл: если /tmp/llm_prompts.enabled существует → пишем prompt-лог.
# Так не нужно перезапускать сервер — touch/rm включает/выключает на лету.
_PROMPT_LOG_MARKER = _Path("/tmp/llm_prompts.enabled")


def _prompt_log_enabled() -> bool:
    if str(_os.getenv("MR_LLM_LOG_PROMPTS", "")).strip() in {"1", "true", "True"}:
        return True
    try:
        return _PROMPT_LOG_MARKER.exists()
    except Exception:
        return False


def _log_prompt_event(event: str, payload: dict[str, Any]) -> None:
    if not _prompt_log_enabled():
        return
    record = {"ts": _time.time(), "event": event, **payload}
    try:
        _PROMPT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROMPT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    except Exception:
        # Не падаем, если лог-файл недоступен — debug-фича.
        pass


class LLMQueueTimeoutError(RuntimeError):
    pass


# Отказ LLM, а не дефект кода: ollama недоступна (`ConnectionError` — так клиент ollama
# переводит `httpx.ConnectError`), очередь к LLM не дождалась слота, сервер ответил
# ошибкой, обрыв/тайм-аут чтения потока. Ловящий обязан ответить честно (оператор), а
# не уронить ход. `TimeoutError` сюда не входит — у бюджета ответа своя ветка, а
# `CancelledError`/`BaseException` не ловятся никогда (стенд 09.10, RENDER-LLM-ERROR).
LLM_UNAVAILABLE_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    LLMQueueTimeoutError,
    ResponseError,
    RequestError,
    httpx.TransportError,
)


def _caller_site() -> str:
    """Место вызова для телеметрии хода — модуль, позвавший runtime (`classifier`, `renderer`…)."""

    frame = _sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
    if frame is None:
        return "unknown"
    return str(frame.f_globals.get("__name__") or "unknown").rsplit(".", 1)[-1]


def _outcome(exc: BaseException) -> str:
    if isinstance(exc, LLMQueueTimeoutError):
        return "queue_timeout"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    return f"error:{type(exc).__name__}"


def _extract_response_text(res: Any) -> str:
    if isinstance(res, dict):
        raw = res.get("response")
    else:
        raw = getattr(res, "response", None)
        if raw is None and hasattr(res, "model_dump"):
            try:
                dumped = res.model_dump()
                if isinstance(dumped, dict):
                    raw = dumped.get("response")
            except Exception:
                raw = None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return raw
    return ""


def _extract_usage(res: Any) -> dict[str, int]:
    if isinstance(res, dict):
        dumped = res
    elif hasattr(res, "model_dump"):
        try:
            raw = res.model_dump()
            dumped = raw if isinstance(raw, dict) else {}
        except Exception:
            dumped = {}
    else:
        dumped = {}

    usage: dict[str, int] = {}
    for key in ("prompt_eval_count", "eval_count", "total_duration", "prompt_eval_duration", "eval_duration"):
        value = dumped.get(key)
        try:
            parsed = int(value)
        except Exception:
            continue
        usage[key] = parsed
    return usage


def _generation_options(override: dict[str, Any] | None) -> Any:
    """Общие параметры генерации и правка поверх них на один вызов.

    Выбор строки прайса просит temperature 0: одинаковый вопрос — одинаковые строки.
    Без правки уходит тот же объект `Options`, что и раньше.
    """

    base = ollama_settings.options_set()
    if not override:
        return base
    return type(base)(**{**base.model_dump(exclude_none=True), **override})


async def generate_text_with_usage(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    llm: str | None = None,
    think: bool | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[str, dict[str, int]]:
    model = llm or LLMName.get()
    resolved_think = ollama_settings.resolve_think(think)
    request_id = _uuid.uuid4().hex[:12] if _prompt_log_enabled() else ""
    started_at = _time.monotonic()
    if _prompt_log_enabled():
        _log_prompt_event(
            "prompt_request",
            {
                "id": request_id,
                "model": model,
                "fmt": fmt,
                "timeout_s": int(timeout_s),
                "prompt_len": len(prompt),
                "prompt": prompt,
            },
        )
    site = _caller_site()
    queue_ms = 0
    try:
        async with llm_slot(queue_timeout_ms):
            queue_ms = int((_time.monotonic() - started_at) * 1000)
            res = await asyncio.wait_for(
                _OLLAMA_CLIENT.generate(
                    model=model,
                    prompt=prompt,
                    options=_generation_options(options),
                    format=fmt,
                    keep_alive=-1,
                    think=resolved_think,
                ),
                timeout=max(5, int(timeout_s)),
            )
    except BaseException as exc:
        turn_telemetry.note_llm(site, _outcome(exc), int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms)
        if isinstance(exc, asyncio.TimeoutError):
            if _prompt_log_enabled():
                _log_prompt_event(
                    "prompt_timeout",
                    {"id": request_id, "duration_s": round(_time.monotonic() - started_at, 3)},
                )
        elif isinstance(exc, Exception) and _prompt_log_enabled():
            _log_prompt_event(
                "prompt_error",
                {
                    "id": request_id,
                    "duration_s": round(_time.monotonic() - started_at, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                },
            )
        raise
    text = _extract_response_text(res)
    usage = _extract_usage(res)
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage)
    if _prompt_log_enabled():
        _log_prompt_event(
            "prompt_response",
            {
                "id": request_id,
                "duration_s": round(_time.monotonic() - started_at, 3),
                "response_len": len(text),
                "response": text,
                "usage": usage,
            },
        )
    return text, usage


async def generate_structured(
    *,
    site: str,
    system: str | None,
    user: str,
    schema: dict[str, Any] | str | None,
    options: dict[str, Any],
    timeout_s: float,
    queue_timeout_ms: int,
    inherit_options: bool = False,
    llm: str | None = None,
) -> tuple[str, dict[str, int]]:
    """Короткое решение по закрытому списку (ход (в) рефакторинга, `choose()`).

    Инструкция — в роли system, данные хода (текст пациента, варианты) — в роли user (SEC-3:
    реплика не становится частью инструкции); форма ответа — JSON Schema; параметры генерации —
    явные параметры места вызова, без общих настроек Gradio и операторского эндпоинта (LLM-2:
    одинаковый вопрос — одинаковое решение); место вызова — явное, а не имя модуля.

    Роли передаются полем `system` у /api/generate: сервер собирает из `system` и `prompt` те же
    сообщения [system, user], что и /api/chat, а подделки LLM (стенд, гейт, тесты) реализуют один
    метод `generate`.

    :param site: место вызова — в телеметрию хода (`choose_<задача>`, как имя промпта)
    :param system: инструкция; None — прежний промпт одной строкой (перенос клиента без смены
        промпта: запрос к модели тот же, что был до переноса)
    :param user: данные хода
    :param schema: JSON Schema ответа (dict); "json" или None — у прежнего промпта
    :param options: параметры генерации места вызова — целиком
    :param inherit_options: поверх общих настроек, как было до переноса (только прежний промпт)
    :return: (сырой текст ответа модели, счётчики токенов)
    :raises: `LLM_UNAVAILABLE_ERRORS` и `asyncio.TimeoutError` — их ловит вызывающий
    """

    request: dict[str, Any] = {
        "model": llm or LLMName.get(),
        "prompt": user,
        "options": _generation_options(options) if inherit_options else dict(options),
        "format": schema,
        "keep_alive": -1,
        "think": ollama_settings.resolve_think(False),
    }
    if system is not None:
        request["system"] = system
    if _prompt_log_enabled():
        _log_prompt_event("prompt_request", {"site": site, "fmt": schema, "system": system, "prompt": user})
    started_at = _time.monotonic()
    queue_ms = 0
    try:
        async with llm_slot(queue_timeout_ms):
            queue_ms = int((_time.monotonic() - started_at) * 1000)
            res = await asyncio.wait_for(_OLLAMA_CLIENT.generate(**request), timeout=max(1.0, float(timeout_s)))
    except BaseException as exc:
        turn_telemetry.note_llm(site, _outcome(exc), int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms)
        if isinstance(exc, Exception) and _prompt_log_enabled():
            _log_prompt_event("prompt_error", {"site": site, "error_type": type(exc).__name__})
        raise
    text = _extract_response_text(res)
    usage = _extract_usage(res)
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage)
    if _prompt_log_enabled():
        _log_prompt_event("prompt_response", {"site": site, "response": text, "usage": usage})
    return text, usage


@asynccontextmanager
async def llm_slot(queue_timeout_ms: int) -> AsyncGenerator[None, None]:
    timeout_s = max(1.0, float(queue_timeout_ms) / 1000.0)
    try:
        await asyncio.wait_for(_SEMAPHORE.acquire(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise LLMQueueTimeoutError("llm_queue_timeout") from e
    try:
        yield
    finally:
        _SEMAPHORE.release()


async def generate_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    llm: str | None = None,
    think: bool | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    text, _usage = await generate_text_with_usage(
        prompt,
        timeout_s=timeout_s,
        queue_timeout_ms=queue_timeout_ms,
        fmt=fmt,
        llm=llm,
        think=think,
        options=options,
    )
    return text


async def generate_stream_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    llm: str | None = None,
    think: bool | None = None,
) -> AsyncGenerator[str, None]:
    model = llm or LLMName.get()
    resolved_think = ollama_settings.resolve_think(think)
    site = _caller_site()
    started_at = _time.monotonic()
    queue_ms = 0
    usage: dict[str, int] = {}
    try:
        async with llm_slot(queue_timeout_ms):
            queue_ms = int((_time.monotonic() - started_at) * 1000)
            stream = await asyncio.wait_for(
                _OLLAMA_CLIENT.generate(
                    model=model,
                    prompt=prompt,
                    options=ollama_settings.options_set(),
                    stream=True,
                    keep_alive=-1,
                    think=resolved_think,
                ),
                timeout=max(5, int(timeout_s)),
            )
            async for chunk in stream:
                text = _extract_response_text(chunk)
                if getattr(chunk, "done", False):
                    usage = _extract_usage(chunk)
                if text:
                    yield text
    except BaseException as exc:
        # GeneratorExit — потребитель закрыл поток (бюджет рендера): это не сбой модели.
        outcome = "closed" if isinstance(exc, GeneratorExit) else _outcome(exc)
        turn_telemetry.note_llm(site, outcome, int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms)
        raise
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage)
