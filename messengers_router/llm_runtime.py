"""Единый runtime для вызовов локальной LLM с ограничением конкуренции.

Ответственность модуля: оборачивать вызовы Ollama, контролировать очередь/таймауты
и отдавать безопасные helper-функции для text/json/stream генерации.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno as _errno
import json as _json
import os as _os
import sys as _sys
import time as _time
import uuid as _uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any, AsyncGenerator

import httpx
from ollama import AsyncClient, RequestError, ResponseError

from agent_logic_2 import ollama_settings
from agent_logic_2.ollama_settings import LLMName

from . import turn_telemetry
from .runtime_config import config as c

# Запасное, если настройка не читается; = умолчанию `[OLLAMA] llm_max_concurrency` (2 — как на проде).
_DEFAULT_MAX_CONCURRENCY = 2
try:
    _MAX_CONCURRENCY = int(str(c.MR_LLM_MAX_CONCURRENCY).strip() or _DEFAULT_MAX_CONCURRENCY)
except Exception:
    _MAX_CONCURRENCY = _DEFAULT_MAX_CONCURRENCY
_SEMAPHORE = asyncio.Semaphore(max(1, _MAX_CONCURRENCY))

# Параметры коротких решений (валидаторы, различители: ответ — одно слово). Явно на месте
# вызова, а не из общих настроек Gradio: одинаковый вопрос — одинаковое решение, ответ не
# растекается (ревью DLA, L-03/LLM-2, 08.10).
DECISION_OPTIONS: dict[str, Any] = {"temperature": 0, "num_predict": 24}

# Сколько ждать установки соединения с LLM: сеть или VPN не отвечает — честный сбой за 2 с, а не
# весь бюджет хода (DEBT-2026-10-13-LLM-CONNECT-DIAG). Чтение не ограничено: длину генерации держит
# `asyncio.wait_for` места вызова. Не настройка: это свойство сети до LLM, а не политика ответа.
_CONNECT_TIMEOUT_S = 2.0


def _new_client() -> AsyncClient:
    return AsyncClient(c.ollama_url, timeout=httpx.Timeout(None, connect=_CONNECT_TIMEOUT_S))


_OLLAMA_CLIENT = _new_client()


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


class LLMBudgetExhaustedError(RuntimeError):
    """Бюджет LLM-решений хода исчерпан (L-11) — вызова не было."""


class LLMCircuitOpenError(RuntimeError):
    """Предохранитель LLM-решений разомкнут: модель подряд не отвечала (L-11) — вызова не было."""


# Отказ LLM, а не дефект кода: ollama недоступна (`ConnectionError` — так клиент ollama
# переводит `httpx.ConnectError`; тайм-аут соединения — `httpx.ConnectTimeout`, его ollama не
# переводит), очередь к LLM не дождалась слота, сервер ответил ошибкой, обрыв/тайм-аут чтения
# потока. Ловящий обязан ответить честно (оператор), а не уронить ход. `TimeoutError` сюда не
# входит — у бюджета ответа своя ветка, а `CancelledError`/`BaseException` не ловятся никогда
# (стенд 09.10, RENDER-LLM-ERROR).
LLM_UNAVAILABLE_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    LLMQueueTimeoutError,
    LLMBudgetExhaustedError,
    LLMCircuitOpenError,
    ResponseError,
    RequestError,
    httpx.TransportError,
)

# Сбой самой генерации — то, что считает предохранитель. Таймаут очереди (семафор) сюда не входит.
_GENERATION_FAILURES: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
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


def error_detail(exc: BaseException) -> str | None:
    """Корневая причина сбоя LLM для телеметрии и лога: errno сокета (`EHOSTDOWN`, `ECONNREFUSED`…)
    или имя класса корня цепочки.

    ollama 0.5.3 переводит `httpx.ConnectError` в `ConnectionError(...) from None`: тип один на любую
    причину (VPN, хост, порт), а errno остаётся только в цепочке `__cause__` / `__context__` — под
    `OSError` без errno («All connection attempts failed»). None — у сбоя нет цепочки или исход
    объясняет сам (таймаут, очередь, отмена).
    """

    if isinstance(exc, (asyncio.TimeoutError, asyncio.CancelledError, LLMQueueTimeoutError, GeneratorExit)):
        return None
    seen: set[int] = set()
    current: BaseException | None = exc
    root: BaseException | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = current.errno if isinstance(current, OSError) else None
        if isinstance(code, int):
            return _errno.errorcode.get(code, str(code))
        root = current
        current = current.__cause__ or current.__context__
    return type(root).__name__ if root is not None and root is not exc else None


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
        err = error_detail(exc)
        turn_telemetry.note_llm(site, _outcome(exc), int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, err=err)
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
                    "err": err,
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


# --- Бюджет хода и предохранитель LLM-решений (ход (в), L-11) --------------------------------
#
# Касаются только структурных решений (`generate_structured`, модуль `choose()`): текст ответа,
# классификатор и прежние слои идут как раньше. Что ответить пациенту, когда решения нет, решает
# вызывающий (его политика сбоя), поэтому оба механизма значат «вызова не было», а не ответ.


@dataclass
class _TurnBudget:
    """Сколько решений и секунд выбора потрачено в этом ходе."""

    calls: int = 0
    spent_s: float = 0.0


_TURN_BUDGET: contextvars.ContextVar[_TurnBudget | None] = contextvars.ContextVar("mr_llm_turn_budget", default=None)


def begin_turn_budget() -> contextvars.Token:
    """Открыть бюджет решений хода — в эндпоинте рядом с телеметрией; токен — в `end_turn_budget`.

    Вне хода (скрипты, тесты сервисов) бюджета нет: ограничение — свойство хода пациента.
    """

    return _TURN_BUDGET.set(_TurnBudget())


def end_turn_budget(token: contextvars.Token) -> None:
    try:
        _TURN_BUDGET.reset(token)
    except ValueError:  # закрыт в другом контексте (поток ответа оборван) — бюджет уйдёт вместе с ним
        pass


def turn_decisions_spent() -> int:
    """Сколько решений хода уже потрачено (вне хода — 0): вызывающий узнаёт, тратил ли лимит его вызов (сбой бюджета и
    разомкнутый предохранитель лимит не тратят — их возвращать нечего)."""

    budget = _TURN_BUDGET.get()
    return budget.calls if budget is not None else 0


def refund_turn_decision() -> bool:
    """Вернуть ходу одно решение (секунды не возвращаются): ответ на висящий вопрос, не закрывший ход, лимит решений не
    тратит — обычному пути остаются оба (решение владельца Q5 18.10, `pending_answer`). Вне хода или нечего вернуть — False."""

    budget = _TURN_BUDGET.get()
    if budget is None or budget.calls <= 0:
        return False
    budget.calls -= 1
    return True


def _decision_limits() -> tuple[int, float, int, float]:
    """(решений на ход, секунд на ход, сбоев до размыкания, пауза предохранителя); 0 — без ограничения."""

    try:
        return (
            int(getattr(c, "MR_LLM_DECISION_TURN_CALLS", 3) or 0),
            float(getattr(c, "MR_LLM_DECISION_TURN_SECONDS", 20) or 0),
            int(getattr(c, "MR_LLM_DECISION_BREAKER_FAILURES", 3) or 0),
            float(getattr(c, "MR_LLM_DECISION_BREAKER_OPEN_S", 30) or 0),
        )
    except (TypeError, ValueError):
        return 3, 20.0, 3, 30.0


@dataclass
class _Breaker:
    """N сбоев генерации подряд → пауза без вызовов; после паузы — одна проба.

    Таймаут очереди к LLM (семафор) не считается: занятость общей LLM соседями (колл-центр,
    FreeTalk) — сигнал в телеметрию, а не признак, что модель лежит. Поэтому честному
    предохранителю нужен семафор, равный числу слотов сервера: иначе ожидание в очереди самой
    ollama выглядит как долгая генерация.
    """

    failures: int = 0
    open_until: float = 0.0

    def allows(self, now: float) -> bool:
        return now >= self.open_until

    def success(self) -> None:
        self.failures = 0

    def failure(self, now: float, threshold: int, open_s: float) -> None:
        if threshold <= 0:
            return
        self.failures += 1
        if self.failures >= threshold:
            self.open_until = now + open_s
            self.failures = threshold - 1  # проба после паузы: её сбой снова размыкает, успех — замыкает


_BREAKER = _Breaker()


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
    shadow: bool = False,
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
    :param shadow: решение тени (L-15b, шаг 4; решение владельца 13.10, Q1) — параллельно ходу, но вне
        его бюджета и предохранителя: бюджет не проверяется и не тратится, предохранитель только
        читается (разомкнут — вызова нет), ни сбой, ни успех тени его не двигают; в телеметрии хода —
        `shadow: true`. Так тень не меняет ни одного ответа хода
    :return: (сырой текст ответа модели, счётчики токенов)
    :raises: `LLM_UNAVAILABLE_ERRORS` и `asyncio.TimeoutError` — их ловит вызывающий; бюджет хода
        и предохранитель (L-11) — без вызова модели
    """

    max_calls, max_turn_s, breaker_failures, breaker_open_s = _decision_limits()
    if not _BREAKER.allows(_time.monotonic()):
        turn_telemetry.note_llm(site, "circuit_open", 0, shadow=shadow)
        raise LLMCircuitOpenError(site)
    timeout = max(1.0, float(timeout_s))
    budget = None if shadow else _TURN_BUDGET.get()
    if budget is not None:
        left_s = max_turn_s - budget.spent_s if max_turn_s else timeout
        if (max_calls and budget.calls >= max_calls) or left_s < 1.0:
            turn_telemetry.note_llm(site, "budget", 0)
            raise LLMBudgetExhaustedError(site)
        budget.calls += 1
        timeout = min(timeout, left_s)
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
            res = await asyncio.wait_for(_OLLAMA_CLIENT.generate(**request), timeout=timeout)
    except BaseException as exc:
        err = error_detail(exc)
        turn_telemetry.note_llm(site, _outcome(exc), int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, err=err, shadow=shadow)
        if isinstance(exc, _GENERATION_FAILURES) and not shadow:
            _BREAKER.failure(_time.monotonic(), breaker_failures, breaker_open_s)
        if isinstance(exc, Exception) and _prompt_log_enabled():
            _log_prompt_event("prompt_error", {"site": site, "error_type": type(exc).__name__, "err": err})
        raise
    finally:
        if budget is not None:
            budget.spent_s += _time.monotonic() - started_at
    if not shadow:
        _BREAKER.success()
    text = _extract_response_text(res)
    usage = _extract_usage(res)
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage, shadow=shadow)
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
        turn_telemetry.note_llm(site, outcome, int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, err=error_detail(exc))
        raise
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage)
