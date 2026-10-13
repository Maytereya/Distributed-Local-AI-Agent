"""Фразы-кандидаты ценовой реплики и разбор без LLM (L-15b, шаг 4 — контракт C).

Одна услуга или несколько — это выбор LLM из фраз, которые нарезал код (принцип 2 конституции:
правила предлагают, LLM выбирает). Здесь — генератор кандидатов и обход, когда выбор не нужен:

- **фраза ⊂ реплика.** Фраза — всегда `reply[start:end]`, никогда не склейка токенов: строку правил
  (фрагмент, окно сегментации, ядро без фразы о сроке) `_locate` выравнивает по токенам реплики (по
  порядку, без регистра, ё = е); не нашлась — отбрасывается. Вежливость и слова вопроса с краёв фразы
  снимаются; несбалансированная скобка — конец тянется до «)», обрывок с ней сам не предлагается;
- **источники:** W — вся реплика без обвязки (у вопроса о сроке — без фразы о сроке); (б) фрагменты по
  разделителям (`_split_price_query_items`; «Прием (осмотр, консультация) врача-терапевта» не рвётся);
  (в) окна сегментации сплошного списка (`_resolve_multi_price_items` — из памяти хода, если понимание
  его уже посчитало); (г) сильные окна (1–3 слова с сильным совпадением по данным МИС) и промежутки
  между ними — разрез «ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ», которого сегментация не даёт (слов меньше 4);
- **потолок** — 16 фраз (решение владельца 13.10, Q3): приоритет W, (б), (в), (г); фрагментов (б) 16 и
  больше — `too_many`, LLM не зовём;
- **обход без LLM:** W пуста — `empty`; W сильная по данным (`phrase_rows` — `strong_rows`, дословное
  название сравнивается и как написано) — `strong_whole`; разрез по
  разделителям (б) из ≥2 фраз, все сильные, остаток — обвязка, и одного названия прайса на всё нет —
  `strong_all`; кандидат один — `single`. Сильные окна без разделителей (в)/(г) LLM не обходят: и
  «ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ» решает выбор (решение владельца 13.10, Q2);
- **выбор LLM** (`parse_turn_phrases`): `choose()` с задачей `turn_phrases` — фразы не пересекаются
  (`Task.disjoint`), подсказки строк прайса в v1 нет (подталкивала бы решать «одна или несколько» по
  похожести букв); промпт `choose_turn_phrases`. Сбой и мусор — статусы `unavailable` / `invalid` без
  фраз: что ответить пациенту, решают читатели разбора (шаг 5).

Новых регулярок и словарей нет: токены — `_PRICE_TOKEN_RE`, обвязка — `_PRICE_QUERY_STOPWORDS`,
`_MULTI_PRICE_NOISE_WORDS`, `_PRICE_GENERIC_SERVICE_TOKENS`; разрезы — функции разбора корзины.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..choice import Choice, Option, Status, Strength, Task, choose
from ..turn_parse import ParsedPhrase, PhraseParse, ParseStatus, per_turn
from ._common import _normalise_catalog_text
from ._prices_helpers import (
    _MULTI_PRICE_NOISE_WORDS,
    _PRICE_GENERIC_SERVICE_TOKENS,
    _PRICE_QUERY_STOPWORDS,
    _PRICE_TOKEN_RE,
    _catalog_fingerprint,
    _catalog_name_covering,
    _is_multi_price_noise,
    _price_query_content_tokens,
    _resolve_multi_price_items,
    _split_price_query_items,
    strip_readiness_phrasing,
)
from ._strong_match import strong_rows

_MAX_PHRASES = 16  # решение владельца 13.10 (Q3)
_MAX_TEXT = 160  # p95 названия в мини-прод — 134 знака
_MAX_WINDOW_CONTENT = 12  # сильные окна — только у короткой реплики
_WINDOWS = (3, 2, 1)

# Обвязка с краёв фразы: слова вопроса о цене и вежливость. Внутри промежутка между сильными окнами
# обвязка ещё и родовые слова («анализ на») — сами по себе услугу они не называют.
_WRAPPER = frozenset(_PRICE_QUERY_STOPWORDS) | _MULTI_PRICE_NOISE_WORDS
_FILLER = _WRAPPER | frozenset(_PRICE_GENERIC_SERVICE_TOKENS)

SOURCE_WHOLE, SOURCE_SPLIT, SOURCE_SEGMENT, SOURCE_STRONG = "whole", "split", "segment", "strong"


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    key: str


@dataclass(frozen=True)
class Phrase:
    """Фраза-кандидат: срез реплики `[start:end]` и откуда она."""

    start: int
    end: int
    raw: str
    source: str

    @property
    def text(self) -> str:
        """Как фразу видит LLM: одной строкой, «ёлочки» пациента — прямые кавычки (SEC-3)."""

        return _display(self.raw)


@dataclass(frozen=True)
class Candidates:
    """Кандидаты реплики: W, варианты для выбора (W первой, остальные по началу) и разрезы правил."""

    reply: str
    probe: str
    whole: Phrase | None
    options: tuple[Phrase, ...] = ()
    partitions: dict[str, tuple[Phrase, ...]] = field(default_factory=dict)
    too_many: bool = False


def _display(raw: str) -> str:
    return " ".join(str(raw).replace("«", '"').replace("»", '"').split())


@per_turn(_catalog_fingerprint)
def phrase_rows(raw: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки сильного совпадения фразы разбора по данным МИС (одно определение — `strong_rows`).

    Фраза разбора — срез реплики уже без обвязки, поэтому дословное название сравнивается и как
    написано: вырезатель фразы правил теряет дефис и предлоги («C-пептид» → «c пептид»), и название
    прайса, набранное целиком, иначе считалось бы слабым (L-15b, шаг 5).

    В памяти хода (шаг 6, O2): генератор спрашивает одну фразу до трёх раз (решение без LLM, строки
    вариантов, запасной разрез) — половина вызовов `strong_rows` была повтором в том же ходе.
    """

    return strong_rows(raw, catalog, verbatim=True)


def _key(word: str) -> str:
    return word.lower().replace("ё", "е")


def _tokens(text: str) -> list[_Token]:
    return [_Token(m.start(), m.end(), _key(m.group(0))) for m in _PRICE_TOKEN_RE.finditer(str(text or ""))]


def _locate(text: str, toks: Sequence[_Token], *, cursor: int = 0, taken: set[int] | None = None) -> tuple[int, int] | None:
    """Где строка правил стоит в реплике: (первый, последний) номер токена — подряд, от курсора, мимо
    занятых; не нашлась — None."""

    want = [t.key for t in _tokens(text)]
    if not want:
        return None
    n = len(want)
    for i in range(cursor, len(toks) - n + 1):
        if [t.key for t in toks[i : i + n]] == want and not (taken and taken.intersection(range(i, i + n))):
            return i, i + n - 1
    return None


def bare(text: str) -> str:
    """Фраза без обвязки с краёв, словами без регистра — сравнивать разрезы (согласие с правилами)."""

    keys = [t.key for t in _tokens(text)]
    while keys and keys[0] in _WRAPPER:
        keys.pop(0)
    while keys and keys[-1] in _WRAPPER:
        keys.pop()
    return " ".join(keys)


def _depth(reply: str, start: int, end: int) -> int:
    return reply.count("(", start, end) - reply.count(")", start, end)


def _closed_end(reply: str, start: int, end: int) -> int | None:
    """Конец фразы с закрытыми скобками: «(…» тянется до «)», если между ними нет слов; иначе None."""

    depth = _depth(reply, start, end)
    while depth > 0:
        close = reply.find(")", end)
        if close < 0 or _PRICE_TOKEN_RE.search(reply, end, close):
            return None
        end, depth = close + 1, depth - 1
    return end if depth == 0 else None


def _phrase(reply: str, toks: Sequence[_Token], i: int, j: int, source: str, *, strict: bool = True) -> Phrase | None:
    """Фраза из токенов `i..j` реплики: без обвязки с краёв и со скобками; пусто или обрывок — None."""

    while i <= j and toks[i].key in _WRAPPER:
        i += 1
    while j >= i and toks[j].key in _WRAPPER:
        j -= 1
    if i > j:
        return None
    start, end = toks[i].start, toks[j].end
    closed = _closed_end(reply, start, end)
    if closed is None:
        if strict:
            return None
        closed = end
    return Phrase(start, closed, reply[start:closed], source)


def _whole(reply: str, toks: Sequence[_Token], probe: str) -> tuple[Phrase | None, tuple[int, int] | None]:
    """W — реплика без обвязки; у вопроса о сроке — ядро без фразы о сроке, выровненное в реплику."""

    if probe == reply:
        found: tuple[int, int] | None = (0, len(toks) - 1) if toks else None
    else:
        core = _tokens(probe)
        lo, hi = 0, len(core) - 1
        while lo <= hi and core[lo].key in _WRAPPER:
            lo += 1
        while hi >= lo and core[hi].key in _WRAPPER:
            hi -= 1
        found = _locate(probe[core[lo].start : core[hi].end], toks) if lo <= hi else None
    if found is None:
        return None, None
    phrase = _phrase(reply, toks, *found, SOURCE_WHOLE, strict=False)
    return phrase, found


def _split_pieces(reply: str, toks: Sequence[_Token], fragments: list[str]) -> list[Phrase]:
    """(б) Фрагменты по разделителям; фрагмент с незакрытой «(» сливается со следующими до баланса."""

    located: list[tuple[int, int]] = []
    cursor = 0
    for fragment in fragments:
        hit = _locate(fragment, toks, cursor=cursor)
        if hit is not None:
            located.append(hit)
            cursor = hit[1] + 1
    out: list[Phrase] = []
    open_from: int | None = None
    for i, j in located:
        first = i if open_from is None else open_from
        start, end = toks[first].start, toks[j].end
        if _closed_end(reply, start, end) is None and _depth(reply, start, end) > 0:
            open_from = first  # обрывок «Прием (осмотр» — ждём «консультация) врача-терапевта»
            continue
        open_from = None
        phrase = _phrase(reply, toks, first, j, SOURCE_SPLIT)
        if phrase is not None:
            out.append(phrase)
    return out


def _segment_pieces(reply: str, toks: Sequence[_Token], probe: str, catalog: list[dict[str, Any]]) -> list[Phrase]:
    """(в) Окна сегментации сплошного списка — при тех же условиях, что у понимания хода
    (`price_list_services`): тогда разбор берётся из памяти хода, а не считается второй раз."""

    tokens = _price_query_content_tokens(probe)
    if len(tokens) < 4 or _catalog_name_covering(tokens, catalog):
        return []
    items, unrecognized = _resolve_multi_price_items(probe, catalog)
    if not items:
        return []
    taken: set[int] = set()
    out: list[Phrase] = []
    for text in [str(item.get("fragment") or "") for item in items] + [str(part) for part in unrecognized]:
        hit = _locate(text, toks, taken=taken)
        if hit is None:
            continue
        taken.update(range(hit[0], hit[1] + 1))
        phrase = _phrase(reply, toks, *hit, SOURCE_SEGMENT)
        if phrase is not None:
            out.append(phrase)
    return sorted(out, key=lambda p: p.start)


def _strong_window_pieces(reply: str, toks: Sequence[_Token], span: tuple[int, int], catalog: list[dict[str, Any]]) -> list[Phrase]:
    """(г) Сильные окна 1–3 слова (жадно слева, длинные первыми, без шум-слова внутри) и промежутки
    между ними, если в промежутке есть слово кроме обвязки. Разрез — только при сильном окне и ≥2 фразах."""

    lo, hi = span
    if not 2 <= sum(1 for t in toks[lo : hi + 1] if not _is_multi_price_noise(t.key)) <= _MAX_WINDOW_CONTENT:
        return []
    out: list[Phrase] = []
    gap: list[int] = []
    strong_found = False

    def flush() -> None:
        if gap and any(toks[k].key not in _FILLER for k in gap):
            phrase = _phrase(reply, toks, gap[0], gap[-1], SOURCE_STRONG)
            if phrase is not None:
                out.append(phrase)
        gap.clear()

    i = lo
    while i <= hi:
        found: Phrase | None = None
        width = 0
        if not _is_multi_price_noise(toks[i].key):
            for size in _WINDOWS:
                window = toks[i : min(i + size, hi + 1)]
                if len(window) < size or any(_is_multi_price_noise(t.key) for t in window):
                    continue
                phrase = _phrase(reply, toks, i, i + size - 1, SOURCE_STRONG)
                if phrase is not None and phrase_rows(phrase.raw, catalog):
                    found, width = phrase, size
                    break
        if found is None:
            gap.append(i)
            i += 1
            continue
        flush()
        out.append(found)
        strong_found = True
        i += width
    flush()
    return out if strong_found and len(out) >= 2 else []


def phrase_candidates(reply: str, *, facet: str | None, catalog: list[dict[str, Any]]) -> Candidates:
    """Фразы-кандидаты реплики. Чистая функция: без LLM и без состояния (память хода — только кэш).

    :param reply: реплика пациента
    :param facet: свойство из понимания хода (`readiness` — W без фразы о сроке)
    :param catalog: строки прайса региона
    """

    reply = str(reply or "")
    probe = strip_readiness_phrasing(reply)[0] if facet == "readiness" else reply
    toks = _tokens(reply)
    whole, span = _whole(reply, toks, probe)
    if whole is None or span is None:
        return Candidates(reply=reply, probe=probe, whole=None)
    fragments = _split_price_query_items(probe)
    separated = len(fragments) >= 2
    split = _split_pieces(reply, toks, fragments) if separated else []
    segment = [] if separated else _segment_pieces(reply, toks, probe, catalog)
    strong = [] if separated else _strong_window_pieces(reply, toks, span, catalog)
    partitions = {name: tuple(cut) for name, cut in ((SOURCE_SPLIT, split), (SOURCE_SEGMENT, segment), (SOURCE_STRONG, strong)) if len(cut) >= 2}

    chosen: list[Phrase] = []
    spans: set[tuple[int, int]] = set()
    texts: set[str] = set()
    for phrase in [whole, *split, *segment, *strong]:
        norm = _normalise_catalog_text(phrase.text)
        if (phrase.start, phrase.end) in spans or norm in texts or not norm:
            continue
        spans.add((phrase.start, phrase.end))
        texts.add(norm)
        if len(phrase.text) <= _MAX_TEXT and len(chosen) < _MAX_PHRASES:
            chosen.append(phrase)
    head = [p for p in chosen if p is whole]
    options = (*head, *sorted((p for p in chosen if p is not whole), key=lambda p: (p.start, p.end)))
    return Candidates(reply=reply, probe=probe, whole=whole, options=options, partitions=partitions, too_many=len(split) >= _MAX_PHRASES)


# --------------------------------------------------------------------------- разбор без LLM


def parsed(phrase: Phrase, rows: Sequence[dict[str, Any]]) -> ParsedPhrase:
    return ParsedPhrase(text=phrase.raw, start=phrase.start, end=phrase.end, strength=Strength.STRONG if rows else Strength.WEAK, rows=tuple(rows))


def leftover(cands: Candidates, items: Sequence[ParsedPhrase]) -> tuple[str, ...]:
    """Фрагменты по разделителям, которых не покрыл выбор, — для честного «Не распознал»."""

    return tuple(
        p.raw for p in cands.partitions.get(SOURCE_SPLIT, ()) if not any(p.start < it.end and it.start < p.end for it in items)
    )


def result(cands: Candidates, status: ParseStatus, source: str, items: Sequence[ParsedPhrase] = ()) -> PhraseParse:
    return PhraseParse(
        reply=cands.reply,
        status=status.value,
        source=source,
        items=tuple(items),
        leftover=leftover(cands, items),
        candidates=len(cands.options),
    )


def _only_wrapper_outside(cands: Candidates, cut: Sequence[Phrase]) -> bool:
    whole = cands.whole
    assert whole is not None
    return all(
        t.key in _FILLER
        for t in _tokens(cands.reply)
        if whole.start <= t.start and t.end <= whole.end and not any(p.start <= t.start and t.end <= p.end for p in cut)
    )


def decide_without_llm(cands: Candidates, catalog: list[dict[str, Any]]) -> PhraseParse | None:
    """Разбор, которому выбор LLM не нужен; None — решает `choose()`."""

    whole = cands.whole
    if whole is None:
        return result(cands, ParseStatus.EMPTY, "none")
    rows = phrase_rows(whole.raw, catalog)
    if rows:
        return result(cands, ParseStatus.STRONG_WHOLE, "data", [parsed(whole, rows)])
    if cands.too_many:
        return result(cands, ParseStatus.TOO_MANY, "none")
    cut = cands.partitions.get(SOURCE_SPLIT, ())
    if cut:
        cut_rows = [phrase_rows(p.raw, catalog) for p in cut]
        if (
            all(cut_rows)
            and _only_wrapper_outside(cands, cut)
            and not _catalog_name_covering(_price_query_content_tokens(cands.probe), catalog)
        ):
            return result(cands, ParseStatus.STRONG_ALL, "data", [parsed(p, r) for p, r in zip(cut, cut_rows)])
    if len(cands.options) <= 1:
        return result(cands, ParseStatus.SINGLE, "data", [parsed(whole, ())])
    return None


# --------------------------------------------------------------------------- выбор фраз LLM (choose)

# Решение — одно из двух на ход (лимит хода не меняется, решение владельца 13.10, Q4). Тень ждёт
# очередь к LLM не дольше 1 с (`llm_slot` округляет 300 мс до своего минимума — 1 с; Q1).
PHRASES_TASK = Task(name="turn_phrases", max_picks=0, disjoint=True, num_predict=48, timeout_s=5.0, queue_timeout_ms=1500)
_SHADOW_QUEUE_MS = 300
# Ответ — номера фраз: ≈3,3 токена на номер плюс постоянная часть. 48 токенов обрезали ответ на 15–16
# номерах (все 3 INVALID замера шага 6, по 3,4–4,0 с впустую) — бюджет растёт с числом вариантов, у
# коротких списков (≤8 вариантов) он прежний.
_TOKENS_PER_OPTION = 4
_TOKENS_FIXED = 16


def phrases_task(n_options: int, *, shadow: bool = False) -> Task:
    """Задача выбора фраз под `n_options` вариантов: бюджет ответа не меньше прежних 48 токенов."""

    budget = max(PHRASES_TASK.num_predict, _TOKENS_FIXED + _TOKENS_PER_OPTION * n_options)
    task = replace(PHRASES_TASK, num_predict=budget)
    return replace(task, queue_timeout_ms=_SHADOW_QUEUE_MS) if shadow else task


def _prepare(
    reply: str, facet: str | None, catalog: list[dict[str, Any]]
) -> tuple[Candidates, PhraseParse | None, tuple[tuple[dict[str, Any], ...], ...], tuple[ParsedPhrase, ...]]:
    """Кандидаты, разбор без LLM (если он есть), строки сильных совпадений каждого варианта и разрез по
    разделителям для политики сбоя (`fallback_split`)."""

    cands = phrase_candidates(reply, facet=facet, catalog=catalog)
    decided = decide_without_llm(cands, catalog)
    if decided is not None:
        return cands, decided, (), ()
    return cands, None, tuple(tuple(phrase_rows(p.raw, catalog)) for p in cands.options), fallback_split(cands, catalog)


def fallback_split(cands: Candidates, catalog: list[dict[str, Any]]) -> tuple[ParsedPhrase, ...]:
    """Разрез, которым читатели действуют, если LLM не ответила (политика сбоя, L-15b шаг 5): фрагменты по
    разделителям (б) со силой по данным, когда их ≥2 и одно название прайса не покрывает всю реплику
    («Билирубин и его фракции» не режется). Окна сплошного текста (в)/(г) без LLM не режут (решение
    владельца 13.10, Q2) — тогда пусто, и читатели идут прежним путём правил."""

    cut = cands.partitions.get(SOURCE_SPLIT, ())
    if len(cut) < 2 or _catalog_name_covering(_price_query_content_tokens(cands.probe), catalog):
        return ()
    return tuple(parsed(p, phrase_rows(p.raw, catalog)) for p in cut)


def _merged_list(cands: Candidates, picked: Sequence[int], rows: Sequence[Sequence[dict[str, Any]]], split: Sequence[ParsedPhrase]) -> bool:
    """LLM выбрала одну фразу — всю реплику W, она слабая, а разрез по разделителям есть (`split` —
    ≥2 фразы и одно название прайса не покрывает реплику: см. `fallback_split`)."""

    return (
        len(picked) == 1
        and cands.whole is not None
        and cands.options[picked[0]] is cands.whole
        and not rows[picked[0]]
        and len(split) >= 2
    )


def _from_choice(
    cands: Candidates, choice: Choice, rows: Sequence[Sequence[dict[str, Any]]], split: tuple[ParsedPhrase, ...] = ()
) -> PhraseParse:
    if choice.status is Status.PICKED:
        picked = sorted((int(i) for i in choice.ids), key=lambda k: (cands.options[k].start, cands.options[k].end))
        if _merged_list(cands, picked, rows, split):
            # LLM слила список в одну (слабую) фразу — вся реплика, хотя разделители режут её на ≥2 позиции
            # и одного названия прайса на неё нет (разбор 95 ответов шага 6, G4: 6 «хуже» из 8 — путь одной
            # услуги отвечал на список одной ценой). Не решение, а сбой: читатели берут разрез по
            # разделителям (политика сбоя шага 5).
            return replace(result(cands, ParseStatus.MERGED, "llm"), split=split)
        return result(cands, ParseStatus.PICKED, "llm", [parsed(cands.options[k], rows[k]) for k in picked])
    if choice.status is Status.NONE_FIT:
        return result(cands, ParseStatus.NONE_FIT, "llm")  # ни одна фраза не услуга — путь одной услуги
    # Сбой и мусор (пересечение фраз; исходов у задачи нет): фраз нет, читателям — разрез по разделителям.
    status = ParseStatus.UNAVAILABLE if choice.status is Status.UNAVAILABLE else ParseStatus.INVALID
    return replace(result(cands, status, "llm"), split=split)


async def parse_turn_phrases(
    reply: str, *, facet: str | None, catalog: list[dict[str, Any]], shadow: bool, prepared: asyncio.Event | None = None
) -> PhraseParse:
    """Разбор ценовой реплики на фразы-услуги: без LLM, если выбор не нужен, иначе — `choose()`.

    Генератор и обход — в потоке (CPU); вызов LLM — место `choose_turn_phrases`. При сбое (`unavailable`,
    `invalid`) фраз нет, а разрез по разделителям — в `split` (политика сбоя читателей, шаг 5).

    :param reply: реплика пациента
    :param facet: свойство из понимания хода (`readiness` — W без фразы о сроке)
    :param catalog: строки прайса региона
    :param shadow: тень — решение вне бюджета хода и предохранителя, очередь к LLM ≤1 с
    :param prepared: событие «генератор готов» — под `on` по нему стартуют правила понимания (шаг 6, O3b)
    """

    started = time.monotonic()
    try:
        cands, decided, rows, split = await asyncio.to_thread(_prepare, reply, facet, catalog)
    finally:
        if prepared is not None:
            prepared.set()
    if decided is None:
        options = [
            Option(id=str(k), text=f"«{p.text}»", strength=Strength.STRONG if rows[k] else Strength.WEAK, span=(p.start, p.end))
            for k, p in enumerate(cands.options)
        ]
        task = phrases_task(len(options), shadow=shadow)
        decided = _from_choice(cands, await choose(task, reply, options, shadow=shadow), rows, split)
    return replace(decided, ms=int((time.monotonic() - started) * 1000))
