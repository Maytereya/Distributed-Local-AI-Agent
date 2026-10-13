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

import contextvars
import copy
import functools
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, Hashable, TypeVar

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
