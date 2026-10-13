"""Разбор реплики на ход (L-15b, шаг 1): одинаковое вычисление внутри хода считается один раз.

Ценовая реплика за ход разбирается до пяти раз: понимание хода (`price_list_services`), корзина
(`_multi_price_items` в `price_info`), срок (`readiness_from_catalog`, `readiness_by_choice`), подбор
анализов (`_resolve_multi_price_items` в `lab_tests`). Вход у них почти всегда один и тот же, а
разбор списка на настоящем прайсе стоит секунды CPU: резолвер — около 170 мс на фрагмент или окно
сплошного списка. Замер 13.10 (стенд, мир 20261008, `test`, набор под деплой): повторы — 23% времени
ходов с разбором, p95 таких ходов 8,1 → 6,0 с без них (журнал, STEP-2026-10-13-L15B-1-PARSE-CACHE).

Здесь — только память ОДИНАКОВЫХ вычислений: ключ = функция + точный вход + отпечаток среза прайса
(как у `_CATALOG_NAMES_CACHE`). Поведение не меняется ни в одном режиме. Единый вход для всех
потребителей (одна реплика — один разбор) поведение меняет — это выключатель `turn_parse`
следующих шагов L-15b, не этот.

Время жизни — один ход: держатель создаёт `turn_understanding.begin_turn()` (начало
`orchestrator.run_pipeline` и `router.route_patient_message`). Вне хода (скрипты, тесты функций)
держателя нет — всё считается как раньше. Дочерние задачи `asyncio.gather` и потоки
`asyncio.to_thread` получают КОПИЮ контекста — с тем же объектом держателя; поэтому `set()` — только
в `begin()`, дальше объект лишь пополняется, под `threading.Lock`. Результат из памяти отдаётся
глубокой копией: потребители вправе менять свои строки (корзина, подбор анализов).

Разбор реплики на фразы (L-15b, шаг 4 — контракт C): одна услуга или несколько решает выбор LLM из
фраз, нарезанных кодом (`services/_turn_phrases.py`). Его результат — `PhraseParse` (фразы-срезы
реплики в её порядке, сила каждой по данным МИС, статус и источник решения).
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Hashable, TypeVar

from . import turn_telemetry
from .runtime_config import config as _cfg

if TYPE_CHECKING:
    from .choice import Strength

_R = TypeVar("_R")


class _TurnParse:
    """Вычисления разбора этого хода: ключ → результат; счётчики — для тестов и замеров."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[Hashable, Any] = {}
        self.computed = 0
        self.reused = 0
        # Слот разбора реплики на фразы (L-15b, шаг 4): задача хода, её результат и реплика.
        self.phrases_task: asyncio.Future[PhraseParse] | None = None
        self.phrases_result: PhraseParse | None = None
        self.phrases_reply = ""

    def remember(self, key: Hashable, compute: Callable[[], _R]) -> _R:
        """Результат `compute()` для `key`: из памяти хода (копией) или посчитанный сейчас.

        Два потока с одним ключом могут посчитать его оба — результат один и тот же, в памяти
        остаётся первый; считать под замком нельзя: разбор зовёт разбор (корзина → позиции).
        """

        with self._lock:
            found = key in self._results
            if found:
                self.reused += 1
                stored = self._results[key]
        if found:
            return copy.deepcopy(stored)
        result = compute()
        with self._lock:
            self._results.setdefault(key, copy.deepcopy(result))
            self.computed += 1
        return result

    def peek(self, key: Hashable, compute: Callable[[], _R]) -> _R:
        """Как `remember`, но без записи: тень читает память хода и не пополняет её."""

        with self._lock:
            found = key in self._results
            if found:
                self.reused += 1
                stored = self._results[key]
        return copy.deepcopy(stored) if found else compute()

    def keep_phrases(self, task: asyncio.Future[PhraseParse]) -> None:
        """Результат задачи разбора — под замок (читают и потоки); отмена и сбой — без результата."""

        if task.cancelled() or task.exception() is not None:
            return
        with self._lock:
            self.phrases_result = task.result()


_CURRENT: contextvars.ContextVar[_TurnParse | None] = contextvars.ContextVar("mr_turn_parse", default=None)


def begin() -> None:
    """Новый ход — пустая память разбора (зовёт `turn_understanding.begin_turn`)."""

    _CURRENT.set(_TurnParse())


def current() -> _TurnParse | None:
    return _CURRENT.get()


def per_turn(fingerprint: Callable[[Any], Hashable]) -> Callable[[Callable[[str, Any], _R]], Callable[[str, Any], _R]]:
    """Помнить `fn(текст, срез прайса)` до конца хода; ключ — функция, текст и `fingerprint(срез)`.

    Вне хода и для нестрокового текста — прежний вызов без памяти.
    """

    def decorate(fn: Callable[[str, Any], _R]) -> Callable[[str, Any], _R]:
        name = f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(query_text: str, retail_rows: Any) -> _R:
            memo = _CURRENT.get()
            if memo is None or not isinstance(query_text, str):
                return fn(query_text, retail_rows)
            key = (name, query_text, fingerprint(retail_rows))
            if _READ_ONLY.get():
                return memo.peek(key, lambda: fn(query_text, retail_rows))
            return memo.remember(key, lambda: fn(query_text, retail_rows))

        return wrapper

    return decorate


# --------------------------------------------------------------------------- разбор реплики на фразы (контракт C)


class ParseStatus(StrEnum):
    """Чем кончился разбор реплики на фразы (L-15b, шаг 4)."""

    EMPTY = "empty"  # в реплике нет слов, кроме обвязки
    STRONG_WHOLE = "strong_whole"  # вся реплика — сильное совпадение по данным МИС
    STRONG_ALL = "strong_all"  # разрез по разделителям, все фразы сильные
    SINGLE = "single"  # кандидат один — реплика целиком
    TOO_MANY = "too_many"  # 16 фрагментов и больше — разрез правил, без LLM
    PICKED = "picked"  # фразы выбрала LLM
    NONE_FIT = "none_fit"  # LLM: ни одна фраза не услуга — не список
    INVALID = "invalid"  # ответ LLM нарушил контракт (в т. ч. пересекающиеся фразы)
    UNAVAILABLE = "unavailable"  # LLM не ответила
    LATE = "late"  # тень не успела к концу хода
    ERROR = "error"  # сбой кода разбора


_DECIDED = frozenset({ParseStatus.STRONG_WHOLE, ParseStatus.STRONG_ALL, ParseStatus.SINGLE, ParseStatus.PICKED, ParseStatus.NONE_FIT})


@dataclass(frozen=True)
class ParsedPhrase:
    """Фраза реплики `reply[start:end]` — одна услуга; `rows` — строки прайса сильной фразы (читатель
    копирует, прежде чем менять)."""

    text: str
    start: int
    end: int
    strength: Strength
    rows: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PhraseParse:
    """Разбор ценовой реплики на фразы-услуги: статус, откуда решение (`data` | `llm` | `none`),
    фразы в порядке реплики и фрагменты по разделителям, не покрытые выбором (`leftover`)."""

    reply: str
    status: str
    source: str
    items: tuple[ParsedPhrase, ...] = ()
    leftover: tuple[str, ...] = ()
    candidates: int = 0
    ms: int = 0

    @property
    def is_list(self) -> bool:
        return len(self.items) >= 2

    @property
    def decided(self) -> bool:
        """Разбором можно действовать: данные или внятный ответ LLM (не сбой, не опоздание)."""

        return self.status in _DECIDED

    def for_text(self, text: str) -> bool:
        """Читатель берёт разбор, только если его вход — эта реплика."""

        return str(text or "") == self.reply

    def as_debug(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "phrases": [item.text for item in self.items],
            "strong": [item.strength == "strong" for item in self.items],
            "leftover": list(self.leftover),
            "candidates": self.candidates,
            "ms": self.ms,
        }


# --------------------------------------------------------------------------- слот разбора на фразы (контракт C)
#
# Выключатель `[MESSENGER_ROUTER] turn_parse = off | shadow | on` (по умолчанию off). Разбор запускает
# `turn_understanding.observe_price_turn` — раз за ход, после понимания. Под `shadow` — фоновой задачей
# во время хода (решение владельца 13.10, Q1): решение LLM вне бюджета хода и предохранителя, память
# разбора и сессию задача не трогает, пишет только `tu_parse_*`; конец хода — `drain` (ждём не дольше
# секунды, потом отмена — статус `late`). Под `on` разбор ждут сразу — его решение из бюджета хода;
# читатели под `on` (шаг 5: понимание, корзина, путь одной услуги) берут его через `acting_for(вход)` —
# только решённый и только для своей реплики. Вне хода и под `off` слота нет.

MODES = ("off", "shadow", "on")

_READ_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar("mr_turn_parse_read_only", default=False)


def mode() -> str:
    """`turn_parse`: off | shadow | on; иное — off."""

    raw = str(getattr(_cfg, "MR_TURN_PARSE", "off") or "").strip().lower()
    return raw if raw in MODES else "off"


def read_only_memory() -> None:
    """Задача (тень) читает память разбора хода, но не пополняет её — копия контекста задачи, а не хода."""

    _READ_ONLY.set(True)


def start_phrases(factory: Callable[[], Awaitable[PhraseParse]], *, reply: str = "") -> asyncio.Future[PhraseParse] | None:
    """Запустить разбор реплики раз за ход (из корутины хода); повтор — та же задача. Вне хода и под
    `off` — None."""

    memo = _CURRENT.get()
    if memo is None or mode() == "off":
        return None
    with memo._lock:
        if memo.phrases_task is None:
            memo.phrases_reply = str(reply or "")
            memo.phrases_task = asyncio.ensure_future(factory())
            memo.phrases_task.add_done_callback(memo.keep_phrases)
        return memo.phrases_task


def phrases_now() -> PhraseParse | None:
    """Разбор хода без ожидания (можно из потока): None — не запускался или ещё считается."""

    memo = _CURRENT.get()
    if memo is None:
        return None
    with memo._lock:
        return memo.phrases_result


async def phrases() -> PhraseParse | None:
    """Дождаться разбора хода; не запускался — None."""

    memo = _CURRENT.get()
    task = memo.phrases_task if memo is not None else None
    if task is None:
        return None
    if not task.done():
        await asyncio.wait({task})
    return phrases_now()


def acting_phrases() -> PhraseParse | None:
    """Разбор, по которому разрешено действовать (шаг 5): только под `on` и только решённый."""

    if mode() != "on":
        return None
    parse = phrases_now()
    return parse if parse is not None and parse.decided else None


def acting_for(text: str) -> PhraseParse | None:
    """Разбор, по которому действует читатель с входом `text` (L-15b, шаг 5): под `on`, решённый и этой
    реплики (`for_text`). None — прежний путь правил: разбора нет, он не решён или вход читателя — не
    реплика хода (дозаполнение, контекст прошлого хода)."""

    parse = acting_phrases()
    return parse if parse is not None and parse.for_text(text) else None


async def drain(grace_s: float) -> None:
    """Конец хода: ждём разбор не дольше `grace_s`, затем отменяем — статус `late` (в телеметрию хода).
    Разбора нет или он готов — возврат без ожидания (под `off` ход не меняется ни на такт)."""

    memo = _CURRENT.get()
    task = memo.phrases_task if memo is not None else None
    if task is None or task.done():
        return
    try:
        await asyncio.wait({task}, timeout=grace_s)
    finally:
        if not task.done():
            task.cancel()
    if not task.done():
        await asyncio.wait({task}, timeout=grace_s)  # отмена доходит до задачи — её записи раньше строки хода
    if task.cancelled():
        late = PhraseParse(reply=memo.phrases_reply, status=ParseStatus.LATE.value, source="none")
        with memo._lock:
            memo.phrases_result = late
        turn_telemetry.note(tu_parse_status=late.status, tu_parse_src=late.source)


def debug() -> dict[str, Any] | None:
    """Разбор хода для debug-ответа (фразы пациента — только здесь, не в телеметрии)."""

    memo = _CURRENT.get()
    if memo is None or memo.phrases_task is None:
        return None
    parse = phrases_now()
    return parse.as_debug() if parse is not None else {"status": "pending"}
