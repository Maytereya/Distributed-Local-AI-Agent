"""Разбор реплики на фразы-услуги без списков слов: LLM размечает номера слов (DESIGN-2026-10-23-NO-STOPWORDS, этап 1).

Решения владельца 22.10: уйти от стоп-слов к выбору LLM по смыслу, 28 списков стоп-слов удалить; без LLM понять, какую услугу
назвал пациент, бот не пытается. Прежний генератор (`_turn_phrases`) резал реплику на окна тремя списками цены
(`_PRICE_QUERY_STOPWORDS`, `_MULTI_PRICE_NOISE_WORDS`, `_PRICE_GENERIC_SERVICE_TOKENS`), и LLM выбирала только среди этих окон:
«анализы» — обвязка, «анализ» — нет (BUG-2026-10-22-CATALOG-CONFIRM-BY-LETTERS).

Здесь окон нет. Реплика показывается LLM с номерами слов («1:Сколько 2:стоит 3:анализ 4:на 5:ферритин»), ответ — группы номеров
подряд идущих слов, по группе на услугу (принцип 2: LLM выбирает номер из перечисленного). Фраза — `reply[начало:конец]` по номерам —
подстрока реплики по построению. Строки прайса к фразе — по данным (`phrase_rows`), как у прежнего разбора.

Без LLM (решение (а)): действует только реплика, которая сама — название или код клиники (`strong_whole`) или одно слово
(`single`); сбой — `unavailable` без разреза: читатели — оффер или «уточните название», а не разрез по буквам.

Выключатель `[MESSENGER_ROUTER] turn_spans_llm = off | on` (off — прежний генератор).
Выключатель `list_condition_beside = off | on` (off — как было): фразы списка несут слова своего отрезка вне фраз (`_with_beside`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .. import llm_runtime, turn_telemetry
from ..choice import Strength
from ..prompt_registry import load_prompt_text
from ..runtime_config import config as _cfg
from ..turn_parse import ParsedPhrase, ParseStatus, PhraseParse
from . import _price_by_parse
from ._common import _normalise_catalog_text
from ._prices_helpers import _PRICE_TOKEN_RE, list_condition_beside_on
from ._strong_match import _catalog_index
from ._turn_phrases import _clause_break, phrase_rows

logger = logging.getLogger(__name__)

TASK_NAME = "turn_spans"
SITE = TASK_NAME  # не задача `choose()`: свой контракт ответа (группы номеров), промпт `turn_spans`
_MAX_WORDS = 40  # длиннее — `too_many`: не разметка, а письмо (p95 ценовой реплики — около 15 слов)
_MAX_PHRASES = 16  # решение владельца 13.10 (Q3): не больше 16 услуг в одной реплике
_QUEUE_TIMEOUT_MS = 1500
# Ответ растёт с числом услуг: пара «[12, 13], » — 7–8 токенов (карточка 2, 22.10: при 3 токенах на слово длинный список
# обрезался — все INVALID; при таймауте 5 с список из 15+ позиций не успевал — все UNAVAILABLE).
_TOKENS_FIXED = 24
_TOKENS_PER_WORD = 8
_TOKENS_MAX = 400
_TIMEOUT_BASE_S = 5.0
_TIMEOUT_PER_WORD_S = 0.25
_TIMEOUT_MAX_S = 12.0

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"phrases": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}},
    "required": ["phrases"],
}


def enabled() -> bool:
    """`turn_spans_llm = on`: разбор реплики разметкой слов LLM вместо генератора окон по спискам."""

    return str(getattr(_cfg, "MR_TURN_SPANS_LLM", "off") or "off").strip().lower() == "on"


@dataclass(frozen=True)
class _Word:
    start: int
    end: int
    text: str


def words(reply: str) -> list[_Word]:
    return [_Word(m.start(), m.end(), m.group(0)) for m in _PRICE_TOKEN_RE.finditer(str(reply or ""))]


def _verbatim_rows(reply: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки, которые реплика называет сама, как написана: название клиники целиком или код услуги — без вырезания слов вопроса."""

    text = str(reply or "").strip()
    if text.isdigit():
        if _price_by_parse.enabled() and _price_by_parse.number_is_choice(text):
            return []  # номер из показанного прошлым ответом, а не код услуги (цена, этап 2, Ш1)
        return [row for row in catalog if str(row.get("serviceHomecode") or "").strip() == text]
    by_norm, _by_name = _catalog_index(catalog)
    return [catalog[pos] for pos in sorted(by_norm.get(_normalise_catalog_text(text), ()))]


def _result(reply: str, status: ParseStatus, source: str, n_words: int, items: Sequence[ParsedPhrase] = ()) -> PhraseParse:
    phrases = _with_beside(reply, items) if list_condition_beside_on() else tuple(items)
    return PhraseParse(reply=reply, status=status.value, source=source, items=phrases, candidates=n_words)


def _with_beside(reply: str, items: Sequence[ParsedPhrase]) -> tuple[ParsedPhrase, ...]:
    """Фразы списка со словами реплики вне фраз (`ParsedPhrase.beside`; `list_condition_beside = on`).

    Правило 3 промпта разметки держит условие внутри фразы («ферритин по ОМС»), но живая LLM его нарушает (у одной услуги
    оторвала «по ОМС» от дословного названия в 59 из 82 свипа П2, карточка Ш7) — и в списке позиция печатала цену без суда
    условия: суд читает `beside`, а его заполнял только прежний генератор окон (BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS).

    Механика, без списков слов: отрезки реплики — по знаку препинания между словами (`_turn_phrases._clause_break`). Слово вне
    фраз в отрезке с фразами — у соседних фраз этого отрезка слева и справа (между двумя — у обеих: суд LLM дешевле чужой
    цены); отрезок без фраз («ферритин (по ОМС), глюкоза») — у ближайшей фразы слева, до первой фразы — у первой. Условие это
    или разговор («сколько стоит», «первый раз»), решают гард П2 и LLM корзины (`phrase_price_items`), не этот код. Одна
    фраза — без поля: путь одной услуги смотрит всю реплику сам."""

    if len(items) < 2:
        return tuple(items)
    toks = words(reply)
    clause = [0] * len(toks)  # номер отрезка каждого слова
    for k in range(1, len(toks)):
        clause[k] = clause[k - 1] + int(_clause_break(reply, toks[k - 1], toks[k]))
    order = sorted(range(len(items)), key=lambda i: (items[i].start, items[i].end))
    inside = [{clause[k] for k, tok in enumerate(toks) if items[i].start <= tok.start and tok.end <= items[i].end} for i in order]
    beside: dict[int, list[str]] = {}
    for k, tok in enumerate(toks):
        if any(item.start <= tok.start and tok.end <= item.end for item in items):
            continue
        here = [pos for pos, clauses in enumerate(inside) if clause[k] in clauses]
        left = [pos for pos in (here or range(len(order))) if items[order[pos]].end <= tok.start]
        right = [pos for pos in here if items[order[pos]].start >= tok.end]
        owners = [*left[-1:], *right[:1]] if here else [left[-1] if left else 0]
        for pos in owners:
            beside.setdefault(order[pos], []).append(tok.text)
    return tuple(replace(item, beside=" ".join(beside[i])) if i in beside else item for i, item in enumerate(items))


def _item(reply: str, first: _Word, last: _Word, catalog: list[dict[str, Any]]) -> ParsedPhrase:
    text = reply[first.start : last.end]
    rows = phrase_rows(text, catalog)
    return ParsedPhrase(text=text, start=first.start, end=last.end, strength=Strength.STRONG if rows else Strength.WEAK, rows=tuple(rows))


def decide_without_llm(reply: str, catalog: list[dict[str, Any]]) -> PhraseParse | None:
    """Разбор, которому LLM не нужна: пусто, реплика — название или код клиники целиком, одно слово, слишком длинная. None — решает LLM."""

    toks = words(reply)
    if not toks:
        return _result(reply, ParseStatus.EMPTY, "none", 0)
    rows = _verbatim_rows(reply, catalog)
    if rows:
        item = ParsedPhrase(text=reply[toks[0].start : toks[-1].end], start=toks[0].start, end=toks[-1].end, strength=Strength.STRONG, rows=tuple(rows))
        return _result(reply, ParseStatus.STRONG_WHOLE, "data", len(toks), [item])
    if len(toks) == 1:
        return _result(reply, ParseStatus.SINGLE, "data", 1, [_item(reply, toks[0], toks[0], catalog)])
    if len(toks) > _MAX_WORDS:
        return _result(reply, ParseStatus.TOO_MANY, "none", len(toks))
    return None


def timeout_for(n_words: int) -> float:
    """Ожидание ответа разметки: длинный перечень — длинный ответ."""

    return min(_TIMEOUT_MAX_S, _TIMEOUT_BASE_S + _TIMEOUT_PER_WORD_S * n_words)


def prompt(reply: str, toks: Sequence[_Word]) -> tuple[str, str]:
    """Инструкция (system) и данные хода (user): реплика пациента — данные, а не инструкция (SEC-3)."""

    template = load_prompt_text(SITE)
    system, _, data = template.partition("<<DATA>>")
    numbered = " ".join(f"{k}:{w.text}" for k, w in enumerate(toks, 1))
    return system.strip(), data.replace("<<QUESTION>>", reply).replace("<<WORDS>>", numbered).strip()


def judge(raw: Any, n_words: int) -> list[tuple[int, int]] | None:
    """Группы → отрезки (первый, последний) номер с нуля; None — ответ нарушил контракт.

    Контракт промпта — группа `[первое, последнее]` (модель так и отвечает: карточка 22.10 — «[[1, 3]]» значит слова 1–3);
    принимаются и `[k]` (одно слово) и перечень подряд `[k, k+1, …]`. Номера 1..n, отрезки не пересекаются, групп не больше
    `_MAX_PHRASES`.
    """

    if not isinstance(raw, dict) or not isinstance(raw.get("phrases"), list):
        return None
    groups = raw["phrases"]
    if len(groups) > _MAX_PHRASES:
        return None
    spans: list[tuple[int, int]] = []
    for group in groups:
        if not isinstance(group, list) or not group or not all(isinstance(k, int) and not isinstance(k, bool) for k in group):
            return None
        if len(group) == 1:
            first = last = group[0]
        elif len(group) == 2:
            first, last = group
        elif group == list(range(group[0], group[0] + len(group))):
            first, last = group[0], group[-1]
        else:
            return None
        if not 1 <= first <= last <= n_words:
            return None
        spans.append((first - 1, last - 1))
    spans.sort()
    if any(prev[1] >= cur[0] for prev, cur in zip(spans, spans[1:])):
        return None
    return spans


async def parse_turn_spans(reply: str, *, facet: str | None, catalog: list[dict[str, Any]]) -> PhraseParse:
    """Разбор реплики разметкой слов LLM; `facet` не нужен — вопрос о сроке или цене LLM не размечает как услугу."""

    del facet
    started = time.monotonic()
    reply = str(reply or "")
    decided = await asyncio.to_thread(decide_without_llm, reply, catalog)
    if decided is not None:
        return _with_ms(decided, started)
    toks = words(reply)
    system, user = prompt(reply, toks)
    try:
        raw, _usage = await llm_runtime.generate_structured(
            site=SITE,
            system=system,
            user=user,
            schema=_SCHEMA,
            options={"temperature": 0, "num_predict": min(_TOKENS_MAX, _TOKENS_FIXED + _TOKENS_PER_WORD * len(toks))},
            timeout_s=timeout_for(len(toks)),
            queue_timeout_ms=_QUEUE_TIMEOUT_MS,
        )
    except Exception as exc:  # сбой LLM — без разреза по буквам (решение владельца 22.10, вариант (а))
        logger.warning("turn_spans_unavailable: %s %s", type(exc).__name__, llm_runtime.error_detail(exc) or "-")
        reason = "budget" if isinstance(exc, llm_runtime.LLMBudgetExhaustedError) else ""
        turn_telemetry.note_choice(TASK_NAME, "unavailable", options=len(toks), ms=_ms(started), cached=False, reason=reason, outcome="", shadow=False)
        return _with_ms(_result(reply, ParseStatus.UNAVAILABLE, "llm", len(toks)), started)
    try:
        data = json.loads(str(raw or "").strip())
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("phrases"), list) and len(data["phrases"]) > _MAX_PHRASES:
        # Больше 16 услуг в одной реплике — не разбор, а письмо: как `too_many` прежнего разбора (решение владельца 13.10, Q3).
        turn_telemetry.note_choice(TASK_NAME, "too_many", options=len(toks), ms=_ms(started), cached=False, reason="", outcome="", shadow=False)
        return _with_ms(_result(reply, ParseStatus.TOO_MANY, "llm", len(toks)), started)
    spans = judge(data, len(toks))
    if spans is None:
        logger.warning("turn_spans_invalid: %r", str(raw or "")[:200])
        turn_telemetry.note_choice(TASK_NAME, "invalid", options=len(toks), ms=_ms(started), cached=False, reason="", outcome="", shadow=False)
        return _with_ms(_result(reply, ParseStatus.INVALID, "llm", len(toks)), started)
    status = ParseStatus.PICKED if spans else ParseStatus.NONE_FIT
    turn_telemetry.note_choice(TASK_NAME, status.value, options=len(toks), ms=_ms(started), cached=False, reason="", outcome="", shadow=False)
    items = await asyncio.to_thread(lambda: [_item(reply, toks[i], toks[j], catalog) for i, j in spans])
    return _with_ms(_result(reply, status, "llm", len(toks), items), started)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _with_ms(parse: PhraseParse, started: float) -> PhraseParse:
    return replace(parse, ms=_ms(started))
