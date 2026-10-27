"""Единый runtime для вызовов локальной LLM с ограничением конкуренции.

Ответственность модуля: оборачивать вызовы Ollama, контролировать очередь/таймауты
и отдавать безопасные helper-функции для text/json/stream генерации.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno as _errno
import heapq as _heapq
import itertools as _itertools
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


class _SlotPool:
    """Слоты LLM (= `OLLAMA_NUM_PARALLEL` сервера): семафор, у которого очередь — по ключу, а не по приходу.

    `acquire()` — как у `asyncio.Semaphore`: ключ — время прихода, очередь FIFO (без дедлайна хода — как было).
    `acquire_by(key)` — ключ задаёт вызывающий: под дедлайном хода — время начала хода, слот получает ход, начатый раньше
    (BUG-2026-10-27-LLM-QUEUE-GIVES-UP: начатые ходы добегают, классификатор нового хода ждёт). Освободившийся слот
    передаётся ждущему с меньшим ключом сразу, мимо счётчика: пришедший позже не перехватит его у очереди.
    """

    def __init__(self, slots: int):
        self._free = max(1, int(slots))
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = _itertools.count()

    def locked(self) -> bool:
        return self._free <= 0

    async def acquire(self) -> bool:
        await self.acquire_by(_time.monotonic())
        return True

    async def acquire_by(self, key: float) -> None:
        while self._waiters and self._waiters[0][2].done():  # ушедшие по тайм-ауту — не очередь
            _heapq.heappop(self._waiters)
        if self._free > 0 and not self._waiters:
            self._free -= 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        _heapq.heappush(self._waiters, (float(key), next(self._seq), fut))
        try:
            await fut
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release()  # слот уже передан, а ждущий ушёл (тайм-аут в тот же миг) — передать дальше
            raise

    def release(self) -> None:
        while self._waiters:
            _key, _seq, fut = _heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._free += 1


_SEMAPHORE: Any = _SlotPool(_MAX_CONCURRENCY)

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


class LLMDeadlineError(LLMQueueTimeoutError):
    """Дедлайн хода (`llm_turn_deadline_s`) не оставил вызову времени: слот не освободился до «остаток − p95 задачи».

    Наследник отказа очереди: политики сбоя мест вызова те же; в журнале хода — исход `deadline`, а ответ хода — честный
    оффер оператора (`router`), не правило."""


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
    if isinstance(exc, LLMDeadlineError):
        return "deadline"
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
    deadline_s: float | None = None,
) -> tuple[str, dict[str, int]]:
    """`deadline_s` — потолок всего вызова, очередь и генерация вместе (предохранитель классификатора); исход сбоя —
    прежний `queue_timeout` или `timeout`. None — как было: очередь и генерация ограничены каждая своим."""

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
    cut = False
    if deadline_s is not None:
        queue_timeout_ms = min(queue_timeout_ms, int(deadline_s * 1000))
    try:
        async with llm_slot(queue_timeout_ms, reserve_s=_reserve_s(site, max(5, int(timeout_s))), ceiling_s=deadline_s):
            queue_ms = int((_time.monotonic() - started_at) * 1000)
            generation_s: float = max(5, int(timeout_s))
            if deadline_s is not None:
                generation_s = min(generation_s, max(0.0, deadline_s - (_time.monotonic() - started_at)))
            generation_s, cut = _deadline_cap(generation_s)
            res = await asyncio.wait_for(
                _OLLAMA_CLIENT.generate(
                    model=model,
                    prompt=prompt,
                    options=_generation_options(options),
                    format=fmt,
                    keep_alive=-1,
                    think=resolved_think,
                ),
                timeout=generation_s,
            )
    except BaseException as exc:
        _note_deadline_cut(exc, cut)
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
    """Сколько решений и секунд выбора потрачено в этом ходе. Решение — ответ модели: сбой вызова (тайм-аут, обрыв,
    очередь) решения не тратит, а его секунды — тратит (решение владельца 26.10)."""

    calls: int = 0
    spent_s: float = 0.0
    # Дедлайн хода (`llm_turn_deadline_s`, BUG-2026-10-27-LLM-QUEUE-GIVES-UP): 0 — нет (как было); время — `time.monotonic()`.
    started_at: float = 0.0
    deadline_at: float = 0.0
    deadline_hits: int = 0  # вызовов, которым не хватило дедлайна (очередь или генерация)
    queue_gaveup: int = 0  # вызовов, не дождавшихся слота (`queue_timeout` / `deadline`) — ход ушёл на путь сбоя


_TURN_BUDGET: contextvars.ContextVar[_TurnBudget | None] = contextvars.ContextVar("mr_llm_turn_budget", default=None)


def _turn_deadline_s() -> float:
    """`[MESSENGER_ROUTER] llm_turn_deadline_s` — секунды хода на LLM от его начала; 0 — дедлайна нет."""

    try:
        return max(0.0, float(getattr(c, "MR_LLM_TURN_DEADLINE_S", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def begin_turn_budget() -> contextvars.Token:
    """Открыть бюджет решений хода — в эндпоинте рядом с телеметрией; токен — в `end_turn_budget`.

    Вне хода (скрипты, тесты сервисов) бюджета нет: ограничение — свойство хода пациента. Дедлайн хода — тоже: он
    отсчитывается отсюда (после замка сессии), фоновые задачи хода видят его через контекст.
    """

    now = _time.monotonic()
    deadline_s = _turn_deadline_s()
    return _TURN_BUDGET.set(_TurnBudget(started_at=now, deadline_at=now + deadline_s if deadline_s else 0.0))


def end_turn_budget(token: contextvars.Token) -> None:
    try:
        _TURN_BUDGET.reset(token)
    except ValueError:  # закрыт в другом контексте (поток ответа оборван) — бюджет уйдёт вместе с ним
        pass


def _deadline_turn() -> _TurnBudget | None:
    budget = _TURN_BUDGET.get()
    return budget if budget is not None and budget.deadline_at else None


def turn_deadline_left() -> float | None:
    """Сколько секунд дедлайна хода осталось (не меньше 0); None — дедлайна нет (выключатель off или вне хода)."""

    budget = _deadline_turn()
    return None if budget is None else max(0.0, budget.deadline_at - _time.monotonic())


def turn_deadline_hit() -> bool:
    """Дедлайн хода не оставил времени хотя бы одному вызову LLM этого хода: ответ хода собран без LLM там, где она
    была нужна, — `router` отвечает честным оффером оператора, а не правилом."""

    budget = _deadline_turn()
    return bool(budget is not None and budget.deadline_hits)


def turn_load_note() -> dict[str, Any]:
    """Поля журнала хода о нагрузке LLM: `llm_deadline_s` — дедлайн хода (под выключателем), `llm_deadline_hit` — вызовов,
    которым не хватило дедлайна, `llm_queue_gaveup` — вызовов, не дождавшихся слота (`queue_timeout` → правила). Нули не
    пишутся."""

    budget = _TURN_BUDGET.get()
    if budget is None:
        return {}
    out: dict[str, Any] = {}
    if budget.deadline_at:
        out["llm_deadline_s"] = round(budget.deadline_at - budget.started_at, 1)
    if budget.deadline_hits:
        out["llm_deadline_hit"] = budget.deadline_hits
    if budget.queue_gaveup:
        out["llm_queue_gaveup"] = budget.queue_gaveup
    return out


# p95 вызова без очереди — запас, который дедлайн оставляет задаче на ответ: ждать слот можно до «остаток − запас». Журнал
# хода прода 14–23.10 (вызовы `ok`, без очереди): классификатор темы — p95 12,3 с (233 вызова; ответ ≈ 220 токенов при
# ~19 ток/с), текст ответа — 8,8 с (14); прочие места (`choose_*`, нормализатор, валидаторы) — не больше 3,5 с. Больше
# тайм-аута генерации места вызова запас не бывает.
_P95_S: dict[str, float] = {"classifier": 12.5, "renderer": 9.0}
_P95_DEFAULT_S = 4.0


def _reserve_s(site: str, generation_s: float) -> float:
    return min(float(generation_s), _P95_S.get(site, _P95_DEFAULT_S))


def _deadline_cap(generation_s: float, *, shadow: bool = False) -> tuple[float, bool]:
    """Тайм-аут генерации под дедлайном хода: не дольше остатка. (тайм-аут, обрезан ли дедлайном)."""

    budget = None if shadow else _deadline_turn()
    if budget is None:
        return generation_s, False
    left = max(0.0, budget.deadline_at - _time.monotonic())
    return (left, True) if left < generation_s else (generation_s, False)


def _note_deadline_cut(exc: BaseException, cut: bool) -> None:
    budget = _deadline_turn()
    if cut and budget is not None and isinstance(exc, asyncio.TimeoutError):
        budget.deadline_hits += 1


def turn_decisions_spent() -> int:
    """Сколько решений хода уже потрачено (вне хода — 0): вызывающий узнаёт, тратил ли лимит его вызов (сбой вызова, сбой
    бюджета и разомкнутый предохранитель лимит не тратят — их возвращать нечего)."""

    budget = _TURN_BUDGET.get()
    return budget.calls if budget is not None else 0


def turn_budget_exhausted() -> bool:
    """Лимит решений хода исчерпан (следующее решение получит отказ `budget`) — в отличие от недоступной LLM: модель
    в этом ходе отвечала, решений просто больше нет (ход (е), DESIGN-2026-10-22-TURN-DECISION-BUDGET). Сбои вызова решений
    не тратят; секунды — тратят, поэтому LLM, висящая до тайм-аута, исчерпывает лимит секундами хода. Вне хода — False."""

    budget = _TURN_BUDGET.get()
    if budget is None:
        return False
    max_calls, max_turn_s, _, _ = _decision_limits()
    return bool(max_calls and budget.calls >= max_calls) or bool(max_turn_s and max_turn_s - budget.spent_s < 1.0)


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
    slot_at: float | None = None
    queue_ms = 0
    cut = False
    try:
        async with llm_slot(queue_timeout_ms, reserve_s=_reserve_s(site, timeout), shadow=shadow):
            slot_at = _time.monotonic()
            queue_ms = int((slot_at - started_at) * 1000)
            generation_s, cut = _deadline_cap(timeout, shadow=shadow)
            res = await asyncio.wait_for(_OLLAMA_CLIENT.generate(**request), timeout=generation_s)
    except BaseException as exc:
        _note_deadline_cut(exc, cut)
        if budget is not None and isinstance(exc, Exception):
            # Модель не ответила — решение не потрачено, секунды тратятся (`finally`): при лежащей LLM лимит хода не
            # «исчерпан», и путь цены отвечает правилами, а не «не нашёл» (решение владельца 26.10).
            budget.calls -= 1
        err = error_detail(exc)
        turn_telemetry.note_llm(site, _outcome(exc), int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, err=err, shadow=shadow)
        if isinstance(exc, _GENERATION_FAILURES) and not shadow and not cut:
            # Генерацию оборвал дедлайн хода — это нагрузка (поздно получен слот), а не лежащая модель.
            _BREAKER.failure(_time.monotonic(), breaker_failures, breaker_open_s)
        if isinstance(exc, Exception) and _prompt_log_enabled():
            _log_prompt_event("prompt_error", {"site": site, "error_type": type(exc).__name__, "err": err})
        raise
    finally:
        if budget is not None:
            # Под дедлайном хода ожидание слота секунды решений не тратит: время хода держит дедлайн (решение владельца 27.10 —
            # отдельные потолки не нужны), секунды решений — время самой модели. Без дедлайна — как было, с очередью.
            if not budget.deadline_at:
                budget.spent_s += _time.monotonic() - started_at
            elif slot_at is not None:
                budget.spent_s += _time.monotonic() - slot_at
    if not shadow:
        _BREAKER.success()
    text = _extract_response_text(res)
    usage = _extract_usage(res)
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage, shadow=shadow)
    if _prompt_log_enabled():
        _log_prompt_event("prompt_response", {"site": site, "response": text, "usage": usage})
    return text, usage


@asynccontextmanager
async def llm_slot(
    queue_timeout_ms: int, *, reserve_s: float = 0.0, ceiling_s: float | None = None, shadow: bool = False
) -> AsyncGenerator[None, None]:
    """Слот LLM на время вызова.

    Без дедлайна хода — как было: очередь по приходу, ждать не дольше `queue_timeout_ms` места вызова (не меньше 1 с).
    Под дедлайном (`llm_turn_deadline_s`; вызов хода, не тень) `queue_timeout_ms` не читается: ждать можно до «остаток
    дедлайна − `reserve_s`» (p95 задачи — столько нужно, чтобы успеть ответить), слот — ходу, начатому раньше; не дождался
    или времени уже нет — `LLMDeadlineError`. `ceiling_s` — свой потолок вызова целиком (классификатор): ждать не дольше него,
    не дождался — прежний отказ очереди.
    """

    sem = _SEMAPHORE
    budget = None if shadow else _TURN_BUDGET.get()
    if budget is None or not budget.deadline_at:
        timeout_s = max(1.0, float(queue_timeout_ms) / 1000.0)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError as e:
            if budget is not None:
                budget.queue_gaveup += 1
            raise LLMQueueTimeoutError("llm_queue_timeout") from e
    else:
        wait_s = budget.deadline_at - _time.monotonic() - max(0.0, reserve_s)
        if wait_s <= 0:
            budget.deadline_hits += 1
            raise LLMDeadlineError("llm_turn_deadline")
        by_ceiling = ceiling_s is not None and ceiling_s < wait_s
        acquire = sem.acquire_by(budget.started_at) if hasattr(sem, "acquire_by") else sem.acquire()
        try:
            await asyncio.wait_for(acquire, timeout=ceiling_s if by_ceiling else wait_s)
        except asyncio.TimeoutError as e:
            budget.queue_gaveup += 1
            if by_ceiling:
                raise LLMQueueTimeoutError("llm_queue_timeout") from e
            budget.deadline_hits += 1
            raise LLMDeadlineError("llm_turn_deadline") from e
    try:
        yield
    finally:
        sem.release()


async def generate_text(
    prompt: str,
    *,
    timeout_s: int,
    queue_timeout_ms: int,
    fmt: str | None = None,
    llm: str | None = None,
    think: bool | None = None,
    options: dict[str, Any] | None = None,
    deadline_s: float | None = None,
) -> str:
    text, _usage = await generate_text_with_usage(
        prompt,
        timeout_s=timeout_s,
        queue_timeout_ms=queue_timeout_ms,
        fmt=fmt,
        llm=llm,
        think=think,
        options=options,
        deadline_s=deadline_s,
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
    cut = False
    try:
        async with llm_slot(queue_timeout_ms, reserve_s=_reserve_s(site, max(5, int(timeout_s)))):
            queue_ms = int((_time.monotonic() - started_at) * 1000)
            generation_s, cut = _deadline_cap(float(max(5, int(timeout_s))))
            stream = await asyncio.wait_for(
                _OLLAMA_CLIENT.generate(
                    model=model,
                    prompt=prompt,
                    options=ollama_settings.options_set(),
                    stream=True,
                    keep_alive=-1,
                    think=resolved_think,
                ),
                timeout=generation_s,
            )
            async for chunk in stream:
                text = _extract_response_text(chunk)
                if getattr(chunk, "done", False):
                    usage = _extract_usage(chunk)
                if text:
                    yield text
    except BaseException as exc:
        # GeneratorExit — потребитель закрыл поток (бюджет рендера): это не сбой модели.
        _note_deadline_cut(exc, cut)
        outcome = "closed" if isinstance(exc, GeneratorExit) else _outcome(exc)
        turn_telemetry.note_llm(site, outcome, int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, err=error_detail(exc))
        raise
    turn_telemetry.note_llm(site, "ok", int((_time.monotonic() - started_at) * 1000), queue_ms=queue_ms, usage=usage)
