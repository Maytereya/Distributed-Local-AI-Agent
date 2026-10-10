"""Выбор из кандидатов — одна механика для всех мест, где LLM понимает смысл слов пациента.

Ход (в) рефакторинга, L-12; принцип 2 конституции (`docs/ARCHITECTURE_PRINCIPLES.md`): правила и
поиск собирают кандидатов, LLM возвращает номер, исход или «ничего не подходит». До модуля девять
LLM-слоёв делали это каждый по-своему: три способа разбора, четыре политики сбоя, сбой кэшировался
до рестарта, «не нашла» значило то же, что «упала».

Контракт:
- LLM видит пронумерованный список 1..N (id вариантов не видит) или перечень названий; ответ
  ограничен JSON Schema — перечень допустимых номеров (названий), `maxItems`, исходы, коды причин,
  `search_terms` (до 3). Пустой выбор разрешён: иначе схема «втиснет» модель в номер;
- ответ LLM — недоверенный ввод: не JSON, номер вне списка, лишние номера, bool или строка вместо
  числа, неизвестный код → INVALID (такому ответу не верим целиком); пусто → NONE_FIT; исход →
  OUTCOME; сбой, тайм-аут, бюджет хода, предохранитель → UNAVAILABLE;
- что делать при UNAVAILABLE / INVALID / NONE_FIT, решает вызывающий: модуль даёт механику,
  политику задаёт клиент;
- кэшируются только внятные ответы (PICKED, NONE_FIT, OUTCOME), коротко (`choose_cache_ttl_s`);
  сбой не кэшируется;
- текст пациента — данные, а не инструкция (SEC-3): он в роли user, без «ёлочек» и переносов строк.

Промпт задачи — `prompts/choose_<задача>.txt` и `app_data/prompts/mr_choose_<задача>.txt`: над
строкой `<<DATA>>` — инструкция (роль system), под ней — данные хода с `<<QUESTION>>` и
`<<OPTIONS>>`. Место вызова в телеметрии и на стенде — имя промпта.

Перенос клиента без смены промпта (`LegacyWire`): прежний промпт и прежняя форма ответа, механика —
модуля. Так перенос доказывается на записанных ответах прод-LLM: на стенде (`replay`) ответы обязаны
совпасть с прежним путём. Удаляется вместе с прежним путём клиента.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from . import llm_runtime, turn_telemetry
from .prompt_registry import load_prompt_text
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

_DATA_MARK = "<<DATA>>"
_NONE = "none"
_MAX_TERMS = 3
_MAX_TERM_LEN = 60
_MAX_QUESTION_LEN = 1000
_CACHE_MAX = 512


class Strength(str, Enum):
    """Сила совпадения варианта с данными (L-13): решает, нужен ли выбор LLM, — политика клиента."""

    STRONG = "strong"  # дословное название, синоним МИС, код — из данных
    WEAK = "weak"  # буквы, часть слова, похожесть


class Status(str, Enum):
    PICKED = "picked"
    NONE_FIT = "none_fit"
    OUTCOME = "outcome"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass(frozen=True)
class Option:
    """Вариант выбора: `id` — для кода (LLM его не видит), `text` — что видит LLM."""

    id: str
    text: str
    strength: Strength = Strength.WEAK


@dataclass(frozen=True)
class Choice:
    status: Status
    ids: tuple[str, ...] = ()
    outcome: str = ""
    reason: str = ""
    terms: tuple[str, ...] = ()
    latency_ms: int = 0
    cached: bool = False

    @property
    def failed(self) -> bool:
        """LLM не дала внятного ответа (сбой или мусор) — решает политика вызывающего."""

        return self.status in (Status.UNAVAILABLE, Status.INVALID)


@dataclass(frozen=True)
class Task:
    """Задача выбора: что можно ответить и как долго ждать.

    :param name: имя задачи; промпт — `choose_<name>`
    :param max_picks: сколько вариантов можно выбрать; 0 — сколько угодно из списка
    :param by_text: ответ — название из перечня (а не номер): модель называет, код сверяет
    :param outcomes: особые исходы задачи («urgent») — кроме выбора
    :param reasons: коды причин пустого выбора
    :param max_terms: сколько терминов второго круга поиска просить (не больше 3)
    """

    name: str
    max_picks: int = 0
    by_text: bool = False
    outcomes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    max_terms: int = 0
    num_predict: int = 128
    timeout_s: float = 8.0
    queue_timeout_ms: int = 2000

    @property
    def site(self) -> str:
        return f"choose_{self.name}"


@dataclass(frozen=True)
class LegacyWire:
    """Прежний промпт и прежняя форма ответа клиента (только для доказательства переноса).

    :param prompt: (реплика, варианты) → прежний промпт одной строкой
    :param answer: (сырой ответ, варианты) → ответ в форме модуля (`picks`, `outcome`, `reason`,
        `search_terms`) или None, если прежний разбор его не принял бы; строгость — модуля
    :param fmt: прежний формат ответа ("json" или None)
    :param options: прежние параметры генерации — поверх общих настроек, как было
    """

    prompt: Callable[[str, Sequence[Option]], str]
    answer: Callable[[str, Sequence[Option]], dict[str, Any] | None]
    fmt: str | None = "json"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Verdict:
    """Ответ модели без привязки к id: кэш хранит номера, id берутся из вариантов вызова."""

    status: Status
    indices: tuple[int, ...] = ()
    outcome: str = ""
    reason: str = ""
    terms: tuple[str, ...] = ()


_CACHEABLE = frozenset({Status.PICKED, Status.NONE_FIT, Status.OUTCOME})
_CACHE: "OrderedDict[str, tuple[float, _Verdict]]" = OrderedDict()


# --------------------------------------------------------------------------- промпт и схема


def answer_schema(task: Task, options: Sequence[Option]) -> dict[str, Any]:
    """JSON Schema ответа: выбрать можно только из списка; пустой выбор разрешён."""

    count = len(options)
    picks: dict[str, Any] = {"type": "array", "maxItems": min(task.max_picks or count, count)}
    if count:
        allowed: list[Any] = [o.text for o in options] if task.by_text else list(range(1, count + 1))
        picks["items"] = {"type": "string" if task.by_text else "integer", "enum": allowed}
    props: dict[str, Any] = {"picks": picks}
    if task.outcomes:
        props["outcome"] = {"type": "string", "enum": [_NONE, *task.outcomes]}
    if task.reasons:
        props["reason"] = {"type": "string", "enum": [_NONE, *task.reasons]}
    if task.max_terms:
        props["search_terms"] = {
            "type": "array",
            "items": {"type": "string", "maxLength": _MAX_TERM_LEN},
            "maxItems": min(task.max_terms, _MAX_TERMS),
        }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _clean_question(text: str) -> str:
    """Текст пациента — данные (SEC-3): без «ёлочек» рамки и переносов строк, разумной длины."""

    flat = " ".join(str(text or "").replace("«", '"').replace("»", '"').split())
    return flat[:_MAX_QUESTION_LEN]


def _options_block(task: Task, options: Sequence[Option]) -> str:
    if not options:
        return "(вариантов нет)"
    if task.by_text:
        return "\n".join(f"- {o.text}" for o in options)
    return "\n".join(f"{i}. {o.text}" for i, o in enumerate(options, 1))


def render(task: Task, question: str, options: Sequence[Option]) -> tuple[str, str]:
    """(инструкция для роли system, данные хода для роли user)."""

    system, mark, data = load_prompt_text(task.site).partition(_DATA_MARK)
    if not mark:
        raise ValueError(f"в промпте {task.site} нет строки {_DATA_MARK}")
    # Реплика подставляется последней: её текст уже ничем не заменяется.
    user = data.strip().replace("<<OPTIONS>>", _options_block(task, options)).replace("<<QUESTION>>", _clean_question(question))
    return system.strip(), user


# --------------------------------------------------------------------------- разбор


def _index_of(task: Task, options: Sequence[Option], item: Any) -> int | None:
    if task.by_text:
        if not isinstance(item, str):
            return None
        return next((i for i, o in enumerate(options) if o.text == item), None)
    if isinstance(item, bool) or not isinstance(item, int):
        return None
    return item - 1 if 1 <= item <= len(options) else None


def _judge(task: Task, options: Sequence[Option], data: Any) -> _Verdict:
    """Ответ в форме модуля → вердикт. Любое нарушение контракта — INVALID."""

    if not isinstance(data, dict) or not isinstance(data.get("picks"), list):
        return _Verdict(Status.INVALID)
    indices: list[int] = []
    for item in data["picks"]:
        index = _index_of(task, options, item)
        if index is None:
            return _Verdict(Status.INVALID)
        if index not in indices:
            indices.append(index)
    if len(indices) > (task.max_picks or len(options)):
        return _Verdict(Status.INVALID)
    outcome = data.get("outcome", _NONE) if task.outcomes else _NONE
    reason = data.get("reason", _NONE) if task.reasons else _NONE
    if outcome not in (_NONE, *task.outcomes) or reason not in (_NONE, *task.reasons):
        return _Verdict(Status.INVALID)
    raw_terms = data.get("search_terms", []) if task.max_terms else []
    if not isinstance(raw_terms, list):
        return _Verdict(Status.INVALID)
    terms: list[str] = []
    for term in raw_terms:
        text = str(term or "").strip()
        if text and len(text) <= _MAX_TERM_LEN and text not in terms:
            terms.append(text)
    common = {"reason": "" if reason == _NONE else reason, "terms": tuple(terms[: min(task.max_terms, _MAX_TERMS)])}
    if outcome != _NONE:
        return _Verdict(Status.OUTCOME, outcome=outcome, **common)
    if indices:
        return _Verdict(Status.PICKED, indices=tuple(indices), **common)
    return _Verdict(Status.NONE_FIT, **common)


def parse_answer(task: Task, options: Sequence[Option], raw: str) -> Choice:
    """Сырой ответ модели (JSON по схеме задачи) → выбор."""

    try:
        data = json.loads(str(raw or "").strip())
    except (TypeError, ValueError):
        data = None
    return _choice(_judge(task, options, data), options)


def _choice(verdict: _Verdict, options: Sequence[Option], *, latency_ms: int = 0, cached: bool = False) -> Choice:
    return Choice(
        status=verdict.status,
        ids=tuple(options[i].id for i in verdict.indices),
        outcome=verdict.outcome,
        reason=verdict.reason,
        terms=verdict.terms,
        latency_ms=latency_ms,
        cached=cached,
    )


# --------------------------------------------------------------------------- кэш


def _ttl_s() -> float:
    try:
        return float(getattr(_cfg, "MR_CHOOSE_CACHE_TTL_S", 1800) or 0)
    except (TypeError, ValueError):
        return 0.0


def _cache_key(site: str, system: str | None, user: str, schema: Any) -> str:
    raw = json.dumps([site, system, user, schema], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> _Verdict | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry[0] < time.monotonic():
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return entry[1]


def _cache_put(key: str, verdict: _Verdict) -> None:
    ttl = _ttl_s()
    if ttl <= 0 or verdict.status not in _CACHEABLE:
        return
    _CACHE[key] = (time.monotonic() + ttl, verdict)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


# --------------------------------------------------------------------------- выбор


async def choose(task: Task, question: str, options: Sequence[Option], *, legacy: LegacyWire | None = None) -> Choice:
    """Какие варианты подходят к реплике пациента — решает LLM, разбирает и проверяет код.

    :param task: задача выбора
    :param question: реплика пациента (как написал он сам)
    :param options: кандидаты, найденные правилами и поиском по данным
    :param legacy: прежний промпт клиента — перенос без смены промпта
    :return: выбор; при сбое — UNAVAILABLE / INVALID, политику выбирает вызывающий
    """

    options = tuple(options)
    if legacy is None:
        system, user = render(task, question, options)
        schema: Any = answer_schema(task, options)
        gen_options: dict[str, Any] = {"temperature": 0, "num_predict": task.num_predict}
    else:
        system, user = None, legacy.prompt(question, options)
        schema, gen_options = legacy.fmt, dict(legacy.options)
    key = _cache_key(task.site, system, user, schema)
    hit = _cache_get(key)
    if hit is not None:
        return _noted(task, _choice(hit, options, cached=True), len(options))
    started = time.monotonic()
    try:
        raw, _usage = await llm_runtime.generate_structured(
            site=task.site,
            system=system,
            user=user,
            schema=schema,
            options=gen_options,
            timeout_s=task.timeout_s,
            queue_timeout_ms=task.queue_timeout_ms,
            inherit_options=legacy is not None,
        )
    except Exception as exc:  # сбой LLM — UNAVAILABLE; что ответить пациенту, решает вызывающий
        logger.warning("choose_unavailable: %s %s", task.name, type(exc).__name__)
        return _noted(task, Choice(Status.UNAVAILABLE, latency_ms=_ms(started)), len(options))
    if legacy is None:
        try:
            data = json.loads(str(raw or "").strip())
        except (TypeError, ValueError):
            data = None
    else:
        data = legacy.answer(str(raw or ""), options)
    verdict = _judge(task, options, data)
    if verdict.status is Status.INVALID:
        logger.warning("choose_invalid: %s %r", task.name, str(raw or "")[:200])
    _cache_put(key, verdict)
    return _noted(task, _choice(verdict, options, latency_ms=_ms(started)), len(options))


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _noted(task: Task, choice: Choice, count: int) -> Choice:
    turn_telemetry.note_choice(
        task.name,
        choice.status.value,
        options=count,
        ms=choice.latency_ms,
        cached=choice.cached,
        reason=choice.reason,
        outcome=choice.outcome,
    )
    return choice
