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

Уточнения позиции (DESIGN-2026-10-28-POSITION-QUALIFIERS, этап 2; `position_qualifiers = on`): тот же вызов отвечает и полем
`qualifiers` — у каждой услуги её уточнения (номера слов и вид: оплата, условие, материал, метод, сопутствующая услуга) или
«ни к одной» (`of = 0`). Уточнение к позиции относит LLM, а не соседство слов (попытка 26.10 `list_condition_beside` раздавала
слова кодом — 8 потерь верных цен). Границы групп те же; битое уточнение отбрасывается, группы остаются.
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
from ..turn_parse import ParsedPhrase, ParseStatus, PhraseParse, Qualifier, QualifierKind
from . import _price_by_parse
from ._common import _normalise_catalog_text
from ._prices_helpers import _PRICE_TOKEN_RE
from ._strong_match import _catalog_index
from ._turn_phrases import phrase_rows

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
# Уточнения позиции: промпт с полем `qualifiers` (старый промпт под off не меняется — повтор из кэша off байт в байт). Уточнение —
# около 20 токенов ответа («{"of": 1, "words": [2, 3], "kind": "payment"}»): запас ответа и ожидания — на 2–3 уточнения.
PROMPT_QUALIFIERS = "turn_spans_qualifiers"
_QUALIFIER_TOKENS = 64
_QUALIFIER_TIMEOUT_S = 2.0
_SCHEMA_QUALIFIERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "phrases": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "qualifiers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "of": {"type": "integer"},
                    "words": {"type": "array", "items": {"type": "integer"}},
                    "kind": {"type": "string", "enum": [kind.value for kind in QualifierKind]},
                },
                "required": ["of", "words", "kind"],
            },
        },
    },
    "required": ["phrases", "qualifiers"],
}


def enabled() -> bool:
    """`turn_spans_llm = on`: разбор реплики разметкой слов LLM вместо генератора окон по спискам."""

    return str(getattr(_cfg, "MR_TURN_SPANS_LLM", "off") or "off").strip().lower() == "on"


def qualifiers_enabled() -> bool:
    """`position_qualifiers = on`: разметка отмечает уточнения позиции (DESIGN-2026-10-28-POSITION-QUALIFIERS, этап 2)."""

    return str(getattr(_cfg, "MR_POSITION_QUALIFIERS", "off") or "off").strip().lower() == "on"


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


def _result(
    reply: str,
    status: ParseStatus,
    source: str,
    n_words: int,
    items: Sequence[ParsedPhrase] = (),
    loose: Sequence[Qualifier] = (),
) -> PhraseParse:
    return PhraseParse(reply=reply, status=status.value, source=source, items=tuple(items), candidates=n_words, loose=tuple(loose))


def _item(reply: str, first: _Word, last: _Word, catalog: list[dict[str, Any]], qualifiers: Sequence[Qualifier] = ()) -> ParsedPhrase:
    text = reply[first.start : last.end]
    rows = phrase_rows(text, catalog)
    outside = sorted((item for item in qualifiers if not item.inside), key=lambda item: item.start)
    full = " ".join(part for _start, part in sorted([(first.start, text), *((item.start, item.text) for item in outside)]))
    return ParsedPhrase(
        text=text,
        start=first.start,
        end=last.end,
        strength=Strength.STRONG if rows else Strength.WEAK,
        rows=tuple(rows),
        beside=" ".join(item.text for item in outside),
        qualifiers=tuple(qualifiers),
        full=full if qualifiers else "",
    )


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

    template = load_prompt_text(PROMPT_QUALIFIERS if qualifiers_enabled() else SITE)
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
        span = _span(group, n_words)
        if span is None:
            return None
        spans.append(span)
    spans.sort()
    if any(prev[1] >= cur[0] for prev, cur in zip(spans, spans[1:])):
        return None
    return spans


def _span(group: Any, n_words: int) -> tuple[int, int] | None:
    """Группа номеров `[первое, последнее]` (или `[k]`, или подряд `[k, k+1, …]`) → отрезок с нуля; None — не по контракту."""

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
    return first - 1, last - 1


def judge_qualifiers(
    raw: Any, reply: str, toks: Sequence[_Word], spans: Sequence[tuple[int, int]]
) -> tuple[dict[int, list[Qualifier]], list[Qualifier]]:
    """Поле `qualifiers` ответа → уточнения по позициям (ключ — номер отрезка в `spans`, порядок реплики) и ни к одной (`of = 0`).

    `of` — номер группы в ответе LLM (с 1), а `spans` отсортированы по месту в реплике — номер пересчитывается по самой группе.
    Битое уточнение (номер слова или группы вне списка, вид вне перечня, слова чужой группы или наполовину своей) — пропуск: группы
    действуют и без него.
    """

    by_item: dict[int, list[Qualifier]] = {}
    loose: list[Qualifier] = []
    items = raw.get("qualifiers") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return by_item, loose
    order = [_span(group, len(toks)) for group in raw.get("phrases") or []]
    kinds = {kind.value: kind for kind in QualifierKind}
    for entry in items:
        if not isinstance(entry, dict) or entry.get("kind") not in kinds:
            continue
        words = _span(entry.get("words"), len(toks))
        of = entry.get("of")
        if words is None or not isinstance(of, int) or isinstance(of, bool) or not 0 <= of <= len(order):
            continue
        own = order[of - 1] if of else None
        if of and own not in spans:
            continue
        if any(first <= words[1] and words[0] <= last for first, last in spans if (first, last) != own):
            continue  # слова чужой группы — не уточнение этой позиции
        inside = own is not None and own[0] <= words[0] and words[1] <= own[1]
        if own is not None and not inside and own[0] <= words[1] and words[0] <= own[1]:
            continue  # наполовину в своей группе — граница не ясна
        start, end = toks[words[0]].start, toks[words[1]].end
        qualifier = Qualifier(kind=kinds[entry["kind"]], text=reply[start:end], start=start, end=end, inside=inside)
        if own is None:
            loose.append(qualifier)
        else:
            by_item.setdefault(spans.index(own), []).append(qualifier)
    return by_item, loose


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
    qualified = qualifiers_enabled()
    tokens = min(_TOKENS_MAX, _TOKENS_FIXED + _TOKENS_PER_WORD * len(toks)) + (_QUALIFIER_TOKENS if qualified else 0)
    try:
        raw, _usage = await llm_runtime.generate_structured(
            site=SITE,
            system=system,
            user=user,
            schema=_SCHEMA_QUALIFIERS if qualified else _SCHEMA,
            options={"temperature": 0, "num_predict": tokens},
            timeout_s=timeout_for(len(toks)) + (_QUALIFIER_TIMEOUT_S if qualified else 0.0),
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
    by_item, loose = judge_qualifiers(data, reply, toks, spans) if qualified else ({}, [])
    if by_item or loose:
        kinds = [q.kind.value for quals in by_item.values() for q in quals]
        turn_telemetry.note(tu_qualifiers=",".join(sorted(kinds)), tu_qualifiers_loose=",".join(sorted(q.kind.value for q in loose)))
    items = await asyncio.to_thread(
        lambda: [_item(reply, toks[i], toks[j], catalog, by_item.get(k, ())) for k, (i, j) in enumerate(spans)]
    )
    return _with_ms(_result(reply, status, "llm", len(toks), items, loose), started)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _with_ms(parse: PhraseParse, started: float) -> PhraseParse:
    return replace(parse, ms=_ms(started))
