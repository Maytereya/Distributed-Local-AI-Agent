"""Корзина: строки каждой позиции списка выбирает LLM — одним вызовом на весь список (ход (в), L-15a).

Корзина правил брала на позицию то, что угадал буквенный резолвер, и вторую строку «на всякий
случай» (`_select_patient_price_rows(limit=2)`): рядом с правильной строкой вставали чужие — «Глюкоза в
моче» к глюкозе, «СРБ в синовиальной жидкости» к СРБ (карточка хода (б)-1: строк вне разметки 101 →
222). Здесь нарезка на позиции остаётся за правилами (генератор кандидатов), а какие строки — та
услуга, решает модель:

- для каждой позиции — её кандидаты (строки корзины правил и поиск по словам позиции);
- сильное совпадение по данным (дословное название, синоним МИС, код — `_strong_match`) — без LLM (L-13);
- остальные позиции — один вызов `choose()`: плоский нумерованный список, строка помечена своей
  позицией («позиция» → название — цена); модель отмечает номера; ничего для позиции — «не нашёл»;
- сбой LLM: сильные позиции — по-прежнему их строки по данным, слабые — строки корзины правил, как
  раньше; сильных нет — None, корзина правил целиком.

Под `turn_parse = on` (L-15b, шаг 5) позиции — фразы разбора хода (`basket_from_phrases`): одна услуга
или несколько уже решили данные МИС или выбор LLM, сила фразы — из разбора; выбор строк слабых позиций —
тот же вызов. Политика сбоя под `on` (поправка владельца 13.10 к (ii)): LLM строки не выбрала (сбой,
мусор, вызов запрещён) — сильные позиции с ценой по данным, слабая — несколько вариантов кода (до 5) с
полными названиями и ценами; одна-единственная найденная строка ценой «той самой» не подаётся — её
название без цены (вариант B L-13, `price_names_without_prices_text`); форму решит замер шага 6.

Выключателя нет с 16.10 (решение владельца, §5 итога облака): разбор хода — единственный режим, и `choose_basket = off`
решал только, звать ли LLM для строк слабых позиций; LLM не зовётся лишь в режиме `strict` и при сбое.

Калькулятор (ТЗ 17.08 §6, базовый объём; ответы заказчика 16.09) — `basket_calculation`, под `basket_calc = on`: итог по
позициям и взятие биоматериала по данным. Базовый вариант позиции — без cito/капиллярной/на дому…, если пациент их не просил
(`_prefer_base_variants` в `_positions`); вариантов после этого больше одного — развилка, и итог — «от … до …», а не одна
выдуманная сумма (решение владельца 22.07: дезинформация ценой хуже уточнения). Складывать ли позиции вообще, решает LLM
(`basket_total`): перечень разных услуг — итог; одна услуга разными словами, варианты на выбор, «есть ли / что это / одно ли это»
(g0868, g0700) — перечень того, что есть в прайсе, без «Итого»; сбой LLM — тоже без «Итого».
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .. import turn_telemetry
from ..choice import Option, Status, Task, choose
from ..runtime_config import config as _cfg
from ._biomaterial import service_mis_names
from ._common import _normalise_catalog_text
from ._price_select_llm import _prefer_base_variants, _record_synonym_suggestion, _row_cost, _row_key, _row_name
from ._prices_helpers import _annotate_price_rows_with_care_context, _query_nonbase_price_flags, _score_price_rows, phrase_price_items
from ._strong_match import strong_rows

logger = logging.getLogger(__name__)

BASKET_TASK = Task(name="price_basket", num_predict=192, timeout_s=12, queue_timeout_ms=3000)
_PER_ITEM = 6
_MAX_OPTIONS = 30
_VARIANTS = 5  # слабая позиция без выбора LLM — до 5 вариантов кода (поправка владельца 13.10)


def variants_max() -> int:
    """Сколько вариантов кода у слабой позиции при сбое LLM (`[MESSENGER_ROUTER] parse_fail_variants`, 2–5;
    иное — 5): замер шага 6 решает, 5 или 3 (T5)."""

    try:
        value = int(getattr(_cfg, "MR_PARSE_FAIL_VARIANTS", _VARIANTS))
    except (TypeError, ValueError):
        return _VARIANTS
    return value if 2 <= value <= _VARIANTS else _VARIANTS


def _ranked(retail_rows: list[dict[str, Any]], query: str, *, limit: int) -> list[dict[str, Any]]:
    """Ранжированные строки прайса — кандидаты позиции: строки, у которых ранжирование нашло совпадение слов (оценка > 0);
    запасной поиск подстрокой («с» — любая строка с буквой «с») — только если слова позиции стоят в названии целыми словами:
    иначе это шум, из которого LLM корзины однажды выбрала «Выезд на дом» (g0702, BUG-2026-10-17-BASKET-HOME-VISIT-ROW;
    выключатель `basket_scored_candidates` снят 22.10 — неделя `on` на проде с деплоя 11)."""

    scored = _score_price_rows(retail_rows, query, limit=limit)
    rows = [item["row"] for item in scored if isinstance(item.get("row"), dict) and item.get("score", 0) > 0]
    if rows:
        return rows
    # Запасной поиск подстрокой — только строки, где слова позиции стоят в названии целыми словами: короткое обозначение
    # («лг», «фсг») находится, буква «с» внутри чужого слова — нет.
    words = _normalise_catalog_text(query).split()
    return [
        item["row"]
        for item in scored
        if isinstance(item.get("row"), dict) and words and set(words) <= set(_normalise_catalog_text(_row_name(item["row"])).split())
    ]


def _candidates(item: dict[str, Any], retail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if item.get("unsatisfiable"):
        return []  # уточнение, которого в прайсе нет («по ОМС»): не «почти то» — «Не распознал» (П2)
    if item.get("judge"):
        return list(item["judge"])  # сильная фраза на суде условия: LLM решает о её строках, не ищет «почти то»
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in [*(item.get("prices") or []), *_ranked(retail_rows, str(item.get("fragment") or item["service_name"]), limit=_PER_ITEM)]:
        if isinstance(row, dict) and _row_key(row) not in seen:
            seen.add(_row_key(row))
            out.append(row)
    return out


def _word_candidates(fragment: str, retail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Кандидаты слабой позиции по её отдельным значимым словам — когда по фразе целиком код не нашёл
    ничего (шаг 6: «т4 свободный» → «сТ4 (FT4) Свободный тироксин», «кальций общий» → строки кальция).

    Только варианты для выбора LLM корзины: строку выбирает LLM, при её сбое позиция остаётся «Не распознал»
    (кандидаты по словам — не та услуга с ценой). Слово — токен фразы вне обвязки и родовых слов
    (`_turn_phrases._FILLER`); строки слова — сильное совпадение по данным МИС (название, синоним, код) и
    ранжирование прайса. Новых регулярок и словарей нет.
    """

    from ._turn_phrases import _FILLER, _tokens

    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for token in _tokens(fragment):
        if token.key in _FILLER:
            continue
        word = fragment[token.start : token.end]
        for row in [*strong_rows(word, retail_rows), *_ranked(retail_rows, word, limit=_PER_ITEM)]:
            if isinstance(row, dict) and _row_key(row) not in seen:
                seen.add(_row_key(row))
                out.append(row)
    return out[: _PER_ITEM * 2]


def _context_candidates(index: int, items: list[dict[str, Any]], retail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Кандидаты позиции списка с общей частью соседей — когда по самой позиции код не нашёл ничего.

    Перечисление часто держит общее слово один раз: голову в начале («витамин б9 и б12», «Витамины В1, В2,
    …, В12» — «б12», «В2» без «витамин») или хвост в конце («т3 и т4 свободные» — «т3» без «свободные»).
    Фраза позиции остаётся срезом реплики (контракт C), а поиск идёт по «голова ближайшей предыдущей
    позиции из ≥2 слов + позиция» и «позиция + хвост ближайшей следующей позиции из ≥2 слов» (голова — все
    слова, кроме последнего; хвост — все, кроме первого). Только варианты выбора LLM, как кандидаты по
    словам; новых регулярок и словарей нет.
    """

    from ._turn_phrases import _FILLER, _tokens

    def words(i: int) -> list[str]:
        text = str(items[i].get("fragment") or items[i]["service_name"])
        return [text[t.start : t.end] for t in _tokens(text) if t.key not in _FILLER]

    fragment = " ".join(words(index))
    queries: list[str] = []
    before = next((words(i) for i in range(index - 1, -1, -1) if len(words(i)) >= 2), None)
    if before:
        queries.append(" ".join([*before[:-1], fragment]))
    after = next((words(i) for i in range(index + 1, len(items)) if len(words(i)) >= 2), None)
    if after:
        queries.append(" ".join([fragment, *after[1:]]))
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for query in queries:
        for row in [*strong_rows(query, retail_rows), *_ranked(retail_rows, query, limit=_PER_ITEM)]:
            if isinstance(row, dict) and _row_key(row) not in seen:
                seen.add(_row_key(row))
                out.append(row)
    return out[:_PER_ITEM]


def _position(fragment: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    name = _row_name(rows[0]) if len(rows) == 1 else fragment
    return {"service_name": name, "prices": _annotate_price_rows_with_care_context(list(rows)), "fragment": fragment}


def _variants(fragment: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Слабая позиция без выбора LLM (политика сбоя под `turn_parse = on`): (строки с ценой, название без цены).

    Кандидаты кода без непрошеных cito/капиллярной/на дому — по одной строке на название, до `variants_max()`:
    два названия и больше — варианты с полными названиями и ценами; одно — его название без цены (вариант
    B L-13: одна найденная строка — не «та самая цена»); ни одного — («», «») — «Не распознал».
    """

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = variants_max()
    for row in _prefer_base_variants(list(candidates), fragment):
        key = _normalise_catalog_text(_row_name(row))
        if key and key not in seen and len(rows) < limit:
            seen.add(key)
            rows.append(row)
    if len(rows) >= 2:
        return rows, ""
    return [], _row_name(rows[0]) if rows else ""


def _fair_shares(sizes: list[int], total: int = _MAX_OPTIONS) -> list[int]:
    """Сколько кандидатов каждой слабой позиции показать LLM: бюджет поровну, а недобор позиций, у которых
    кандидатов меньше доли, — тем, у кого их больше (не меньше прежних двух на позицию). Прежняя доля
    `max(2, 30 // N)` при восьми слабых — три строки, хотя половина бюджета пустовала у позиций без кандидатов
    (STEP-2026-10-14-L15B-6-T3, п. 7)."""

    total = max(total, 2 * sum(1 for size in sizes if size))
    shares = [0] * len(sizes)
    left = total
    open_ = [i for i, size in enumerate(sizes) if size]
    while left > 0 and open_:
        each = max(1, left // len(open_))
        for i in list(open_):
            take = min(each, sizes[i] - shares[i], left)
            shares[i] += take
            left -= take
            if shares[i] >= sizes[i]:
                open_.remove(i)
            if left <= 0:
                break
    return shares


async def _pick(
    question: str,
    items: list[dict[str, Any]],
    retail_rows: list[dict[str, Any]],
    *,
    llm_allowed: bool = True,
    fair_share: bool = False,
) -> tuple[dict[int, list[dict[str, Any]]], list[tuple[int, str, list[dict[str, Any]]]], bool]:
    """Строки позиций: сильные — по данным, слабые — выбор LLM одним вызовом.

    `fair_share` (корзина из фраз, `turn_parse = on`) — бюджет вариантов по `_fair_shares`; иначе прежняя
    доля `max(2, 30 // N)` (корзина правил под off не меняется).

    :return: (выбранные строки по номеру позиции, слабые позиции с кандидатами, сбой LLM)
    """

    chosen: dict[int, list[dict[str, Any]]] = {}
    weak: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, item in enumerate(items):
        fragment = str(item.get("fragment") or item["service_name"])
        if item.get("strong"):
            # Сильная по разбору хода: строки — её данные, кандидаты не нужны (шаг 6, O5 — ранжирование
            # прайса для неё считалось и не читалось).
            chosen[index] = list(item["strong"])
            continue
        candidates = _candidates(item, retail_rows)
        # Сила позиции из разбора хода (`strong`, весь срез) — не пересчитывается по кандидатам.
        strong = list(item["strong"]) if "strong" in item else strong_rows(fragment, retail_rows, pool=candidates)
        if strong:
            chosen[index] = strong
        else:
            weak.append((index, fragment, candidates))
    if not weak:
        return chosen, weak, False
    # Корзина из фраз (`fair_share`): позиции, у которой по фразе целиком кандидатов нет, LLM получает
    # кандидатов по общей части соседей по списку, затем по словам; при сбое LLM варианты строятся из прежних
    # (пустых) — «Не распознал». Уточнение,
    # которого в прайсе нет («по ОМС», П2), — без кандидатов и по словам.
    offered = [
        candidates
        or (
            _context_candidates(index, items, retail_rows) or _word_candidates(fragment, retail_rows)
            if fair_share and llm_allowed and not items[index].get("unsatisfiable")
            else []
        )
        for index, fragment, candidates in weak
    ]
    per_item = max(2, _MAX_OPTIONS // len(weak))
    limits = _fair_shares([len(rows) for rows in offered]) if fair_share else [per_item] * len(weak)
    options: list[Option] = []
    owners: dict[str, tuple[int, dict[str, Any]]] = {}
    for (index, fragment, _whole_cands), rows, limit in zip(weak, offered, limits):
        label = _option_label(items[index], fragment)
        for row in rows[:limit]:
            option_id = str(len(options))
            owners[option_id] = (index, row)
            options.append(Option(id=option_id, text=f"{label} → {_row_name(row)} — {_row_cost(row)} руб."))
    if not options:
        return chosen, weak, False  # у слабых позиций нет ни одного кандидата — выбирать не из чего
    if not llm_allowed:
        return chosen, weak, True
    choice = await choose(BASKET_TASK, question, options)
    if choice.failed:
        return chosen, weak, True
    for option_id in sorted(choice.ids, key=int):
        index, row = owners[option_id]
        chosen.setdefault(index, []).append(row)
    return chosen, weak, False


def _option_label(item: dict[str, Any], fragment: str) -> str:
    """Позиция в варианте для LLM: «фраза»; у сильной фразы на суде условия — и слова, которые пациент добавил
    к ней в своём фрагменте (правило 5 промпта корзины: условие, которого нет в строке, — строку не выбирать)."""

    condition = str(item.get("condition") or "")
    return f"«{fragment}» [добавлено: «{condition}»]" if condition else f"«{fragment}»"


def _not_found_label(item: dict[str, Any], fragment: str) -> str:
    """«Не распознал» сильной фразы, которую суд условия не подтвердил, — со словами пациента («ферритин омс»),
    а не «ферритин»: ферритин в прайсе есть, нет его с этим условием. У слабой фразы на суде (`phrase`) — вся фраза, как
    написал пациент: условие может стоять и перед названием («по омс ферритин»)."""

    if item.get("phrase"):
        return str(item["phrase"])
    condition = str(item.get("condition") or "")
    return f"{fragment} {condition}" if condition else fragment


def _positions(
    items: list[dict[str, Any]], chosen: dict[int, list[dict[str, Any]]], unpriced: dict[int, str] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    positions: list[dict[str, Any]] = []
    not_found: list[str] = []
    for index, item in enumerate(items):
        fragment = str(item.get("fragment") or item["service_name"])
        rows = chosen.get(index)
        if rows:
            positions.append(_position(fragment, _prefer_base_variants(rows, fragment)))
        elif unpriced and unpriced.get(index):
            # Одна найденная строка без выбора LLM — название без цены (`multi_price_payload_from_items`).
            positions.append({"service_name": fragment, "prices": [], "fragment": fragment, "unpriced": unpriced[index]})
        else:
            not_found.append(_not_found_label(item, fragment))
    return positions, not_found


def _without_repeats(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строка прайса — один раз в ответе (корзина из фраз). Строка, которую выбрали у нескольких позиций,
    остаётся у той, где строк меньше — у позиции, которую она называет точнее («Общий анализ мочи» — у «общий
    анализ мочи», а не вариантом у «общий анализ крови»; «Триглицериды» — у «триглицериды», а не у
    «холестерин»); при равенстве — у первой. Позиция, у которой строк не осталось, сливается с той — её фраза
    названа той же услугой (не «Не распознал»); слитые позиции помечены `merged` — одна оставшаяся позиция всё
    равно ответ корзины, а не повод уйти в путь одной услуги. Как дедуп услуг у корзины правил
    (`_resolve_multi_price_items`); STEP-2026-10-14-L15B-6-T3, п. 7."""

    owner: dict[tuple[str, str, str], int] = {}
    for index, position in enumerate(positions):
        rows = position.get("prices") or []
        for row in rows:
            key = _row_key(row)
            held = owner.get(key)
            if held is None or len(rows) < len(positions[held].get("prices") or []):
                owner[key] = index
    out: list[dict[str, Any]] = []
    merged = False
    for index, position in enumerate(positions):
        prices = position.get("prices") or []
        if not prices:
            out.append(position)  # название без цены (политика сбоя) — как было
            continue
        kept: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in prices:
            key = _row_key(row)
            if owner.get(key) == index and key not in seen:
                seen.add(key)
                kept.append(row)
        if not kept:
            merged = True
            continue
        out.append(position if len(kept) == len(prices) else {**position, "prices": kept})
    return [{**position, "merged": True} for position in out] if merged else out


def _settle_weak_as_variants(
    weak: list[tuple[int, str, list[dict[str, Any]]]], chosen: dict[int, list[dict[str, Any]]]
) -> dict[int, str]:
    """Политика сбоя под `on`: слабые позиции — варианты кода (`_variants`); названия без цены — по номеру позиции."""

    unpriced: dict[int, str] = {}
    for index, fragment, candidates in weak:
        rows, name = _variants(fragment, candidates)
        if rows:
            chosen[index] = rows
        elif name:
            unpriced[index] = name
    return unpriced


async def choose_basket_rows(
    question: str,
    items: list[dict[str, Any]],
    unrecognized: list[str],
    retail_rows: list[dict[str, Any]],
    *,
    llm_allowed: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Позиции корзины правил с выбранными строками и нераспознанное (корзина, когда разбор не годится для действия).

    Политика сбоя — как у корзины из фраз: слабые позиции без выбора LLM — варианты кода (`_variants`), а не строки
    корзины правил (T7 шаг 3б: прежняя ветка «слабые — строки правил» жила только под `turn_parse = off`, снята).

    :param question: реплика пациента (как написал он сам)
    :param items: позиции корзины правил (`fragment`, `service_name`, `prices`)
    :param unrecognized: позиции, которые правила не распознали
    :param retail_rows: строки прайса региона
    :param llm_allowed: можно ли звать LLM; нельзя — как сбой
    """

    chosen, weak, failed = await _pick(question, items, retail_rows, llm_allowed=llm_allowed)
    unpriced = _settle_weak_as_variants(weak, chosen) if failed else {}
    positions, not_found = _positions(items, chosen, unpriced)
    return positions, [*unrecognized, *not_found]


def _journal_fragment(fragment: str) -> str:
    """Фраза слабой позиции для журнала синонимов (L-14, шаг 2) — как фраза одиночной услуги: без речевой обвязки с
    краёв (`turn_understanding._bare_phrase`, те же списки): «а ггтп» → «ггтп»; позиция из одной обвязки — пусто, в
    журнал не пишется. Фразы разбора режут края только по `_turn_phrases._WRAPPER`, «а» в нём нет."""

    from ..turn_understanding import _bare_phrase  # понимание хода импортирует сервисы цены — не на уровне модуля

    return _bare_phrase(fragment)


async def basket_from_phrases(
    question: str,
    phrases: Sequence[Any],
    leftover: Sequence[str],
    retail_rows: list[dict[str, Any]],
    *,
    llm_allowed: bool = True,
    journal: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Корзина из фраз разбора хода (`turn_parse = on`, L-15b, шаг 5): позиции — фразы в порядке реплики.

    Сильная фраза — её строки по данным МИС без LLM; слабые — один вызов `choose()` по кандидатам правил
    (`phrase_price_items`); LLM строки не выбрала (сбой, мусор, вызов запрещён) — политика сбоя: варианты
    кода или название без цены (`_variants`). Фрагменты по разделителям, не покрытые разбором
    (`leftover`), и позиции без строк — «Не распознал».

    :param question: реплика пациента (как написал он сам)
    :param phrases: фразы разбора (`turn_parse.ParsedPhrase`)
    :param leftover: фрагменты реплики вне выбранных фраз
    :param retail_rows: строки прайса региона
    :param llm_allowed: можно ли звать LLM (режим `strict` — нельзя)
    :param journal: фразы выбрала LLM разбора — слабые позиции и решение LLM по ним (выбранные строки или
        «не нашла», в том числе когда кандидатов нет вовсе) пишутся в журнал синонимов (`source: basket`);
        без решения LLM по строкам (сбой) — не пишутся
    :return: (позиции с выбранными строками, нераспознанное)
    """

    items = phrase_price_items(phrases, retail_rows)
    chosen, weak, failed = await _pick(question, items, retail_rows, llm_allowed=llm_allowed, fair_share=True)
    # Сильная фраза на суде условия (`judge`) — не слабая позиция: синонима клинике она не даёт, а при сбое LLM —
    # «Не распознал», не варианты с ценой (П2: дезинформация ценой хуже уточнения).
    judged = {index for index, item in enumerate(items) if item.get("judge")}
    if judged:
        turn_telemetry.note(strong_phrase_guard="basket_fallback" if failed else ("basket_rows" if any(chosen.get(i) for i in judged) else "basket_not_found"))
    if journal and not failed:
        for index, fragment, _candidates in weak:
            phrase = _journal_fragment(fragment)
            if phrase and index not in judged:
                _record_synonym_suggestion(phrase, [], list(chosen.get(index) or []), source="basket")
    unpriced = _settle_weak_as_variants([entry for entry in weak if entry[0] not in judged], chosen) if failed else {}
    positions, not_found = _positions(items, chosen, unpriced)
    return _without_repeats(positions), [*leftover, *not_found]


# --- Калькулятор: итог и взятие биоматериала (ТЗ 17.08 §6, ответы заказчика 16.09, пп. 2–3; `basket_calc = on`) ---

# Строки прайса взятия — как их назвал заказчик 16.09 («Взятие крови из вены», а не «Взятие крови без последующего
# исследования»); цена — из прайса по названию, не константой. Порядок — порядок перечня под итогом.
_VEIN = "Взятие крови из вены"
_FINGER = "Взятие крови из пальца"
_SWAB = "Взятие мазка (соскоба)"
_TAKES = (_VEIN, _FINGER, _SWAB)
# Слова, которыми клиника пишет материал в МИС (`biomatNames`) и капиллярность в названии строки прайса. Это правило
# заказчика (16.09) над данными клиники, а не разбор слов пациента: «капиллярная» в названии → палец; мазок / соскоб →
# мазок; иначе кровь → вена.
_CAPILLARY_ROW = "капиллярная"
_BLOOD = "кровь"
_VEIN_WORD = "вены"
_CAPILLARY = frozenset({_CAPILLARY_ROW, "пальца"})
_SWAB_WORDS = frozenset({"соскоб", "мазок"})


def calc_enabled() -> bool:
    """`[MESSENGER_ROUTER] basket_calc = on`: под позициями корзины — итог и взятие биоматериала; мусор — off (как было)."""

    return str(getattr(_cfg, "MR_BASKET_CALC", "off") or "").strip().lower() == "on"


def _words(text: str) -> set[str]:
    return set(_normalise_catalog_text(text).split())


def _material_takes(material: str) -> frozenset[str]:
    """Какое взятие нужно для одного материала МИС («кровь из вены», «соскоб из уретры»); пусто — материал пациент приносит
    сам или берёт врач (моча, кал, биоптат)."""

    words = _words(material)
    takes: set[str] = set()
    if words & _SWAB_WORDS:
        takes.add(_SWAB)
    if _BLOOD in words:
        if words & _CAPILLARY:
            takes.add(_FINGER)
        if _VEIN_WORD in words or not words & _CAPILLARY:
            takes.add(_VEIN)
    return frozenset(takes)


def _row_takes(row: dict[str, Any]) -> tuple[frozenset[str] | None, bool]:
    """Взятие для строки прайса по данным: (взятие или None — из данных не выводится, лабораторная ли услуга).

    `biomatNames` перечисляет ДОПУСТИМЫЕ материалы, а не выбранный (у ОАК — «кровь из вены, кровь капиллярная»): взятие
    выводится, если всем допустимым материалам, которые берут в клинике, нужно одно и то же; материалы без взятия (моча,
    сперма, пунктат) его не отменяют — перечень под итогом предупреждает о доплате, когда материал берут («Chlamydia» —
    соскоб или моча → «Взятие мазка (соскоба)»). Кровь из вены или из пальца у строки без «капиллярная» в названии — из
    вены (капиллярный вариант — своя строка прайса, ответ заказчика 16.09). Взятие разное («кровь из вены, соскоб с шейки
    матки») — из данных не выводится. Услуги нет в справочнике МИС — о взятии данных нет (не лабораторная по данным).
    """

    name = _row_name(row)
    if _CAPILLARY_ROW in _words(name):
        return frozenset({_FINGER}), True
    materials, _synonyms = service_mis_names(name)
    if not materials:
        return None, False
    options = {takes for takes in map(_material_takes, materials) if takes}
    if len(options) <= 1:
        return (options.pop() if options else frozenset()), True
    if options == {frozenset({_VEIN}), frozenset({_FINGER})}:
        return frozenset({_VEIN}), True
    return None, True


def _amount(row: dict[str, Any]) -> int:
    """Сумма строки, как её печатает рендер (`renderer._extract_price_amount`: округление, не отбрасывание копеек)."""

    for key in ("servicePrice", "price", "service_price", "amount", "cost"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return int(round(float(value)))
    return _row_cost(row)


def basket_calculation(positions: Sequence[dict[str, Any]], retail_rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Итог и взятие биоматериала по позициям корзины (`basket_calc = on`); off или позиций с ценой нет — None.

    Итог — сумма по позициям с ценой: у позиции одна цена (после базового варианта — `_positions`) — она; несколько
    вариантов с разными ценами — развилка: итог «от суммы меньших до суммы больших», позиция — в `forks`. Взятие — строки
    прайса `_TAKES`, которые нужны позициям по данным (`_row_takes`); у развилки — только если всем её вариантам нужно одно и
    то же. Взятие, которое пациент назвал сам, — его позиция, в перечень не входит. `take_unknown` — у лабораторной позиции
    взятие из данных не выводится или строки взятия нет в прайсе: тогда под итогом остаётся одна оговорка заказчика.

    :param positions: позиции корзины (`service_name`, `prices`)
    :param retail_rows: строки прайса региона — цены взятия
    :return: `positions` (групп с ценой, как их печатает рендер), `total_min`, `total_max`, `forks` (заголовки развилок), `takes`
        (`serviceName`, `cost`), `take_unknown`, `total_unknown` (у показанного варианта цена 0 — не указана: итога нет)
    """

    if not calc_enabled():
        return None
    # Позиции — как их печатает рендер: группа на заголовок «По услуге «…»» (две позиции с одним названием — одна группа,
    # `renderer._group_family_variants_by_care_context`): итог считается по тому, что пациент видит.
    groups: dict[str, list[dict[str, Any]]] = {}
    capillary_asked: set[str] = set()
    for position in positions:
        rows = [row for row in position.get("prices") or [] if isinstance(row, dict)]
        if rows:
            label = str(position.get("service_name") or "")
            groups.setdefault(label, []).extend(rows)
            if "capillary" in _query_nonbase_price_flags(str(position.get("fragment") or "")):
                capillary_asked.add(label)
    if not groups:
        return None
    total_min = total_max = 0
    forks: list[str] = []
    needed: set[str] = set()
    unknown = no_price = False
    for label, rows in groups.items():
        amounts = [_amount(row) for row in rows]
        # 0 в прайсе — цена не указана, а не цена: с ней сумма неизвестна, итог не печатается (g0345: «от 2 800» из «аборт — 0 руб.»).
        no_price = no_price or min(amounts) <= 0
        total_min += min(amounts)
        total_max += max(amounts)
        if min(amounts) != max(amounts):
            forks.append(label)
        verdicts = [_row_takes(row) for row in rows]
        takes = {take for take, _lab in verdicts}
        if len(takes) == 1 and None not in takes and not (label in capillary_asked and _FINGER not in next(iter(takes))):
            needed |= next(iter(takes))
        elif any(lab for _take, lab in verdicts):
            # Взятие у вариантов разное — или пациент просил «из пальца», а показанные строки не капиллярные (их выбрала LLM;
            # «из вены» спорило бы с его словами, g0363): из данных не выводится — только оговорка.
            unknown = True
    named = {_normalise_catalog_text(_row_name(row)) for rows in groups.values() for row in rows}
    by_name = {_normalise_catalog_text(_row_name(row)): row for row in reversed(retail_rows) if isinstance(row, dict)}
    shown: list[dict[str, Any]] = []
    for take in _TAKES:
        key = _normalise_catalog_text(take)
        if take not in needed or key in named:
            continue
        row = by_name.get(key)
        if row is None:
            unknown = True  # строки взятия нет в прайсе — перечень без неё, оговорка остаётся
            continue
        shown.append({"serviceName": _row_name(row), "cost": _amount(row)})
    turn_telemetry.note(
        basket_total="single" if len(groups) < 2 else ("no_price" if no_price else ("range" if forks else "exact")),
        basket_takes=len(shown),
        **({"basket_take_unknown": True} if unknown else {}),
    )
    return {
        "positions": len(groups),
        "total_min": total_min,
        "total_max": total_max,
        "forks": forks,
        "takes": shown,
        "take_unknown": unknown,
        "total_unknown": no_price,
    }


# Складывать ли позиции — решает LLM по реплике (принцип 2): перечень разных услуг — итог; одна услуга разными словами,
# варианты на выбор, «есть ли / что это / одно ли это» — не сумма (g0868 «Чек ап… — есть ли комплексные анализы?», g0700
# «Витамин Д это витамин д 25 он и просто витамин д?»). Пациент назвал позиции сам — по буквам этого не различить.
BASKET_TOTAL_TASK = Task(name="basket_total", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500)
_SUM = "sum"
_ALTERNATIVES = "alternatives"
_BASKET_TOTAL_OPTIONS = (
    Option(id=_SUM, text="Узнать стоимость нескольких разных анализов или услуг, которые пациент перечислил, — общая сумма за все"),
    Option(
        id=_ALTERNATIVES,
        text="Узнать про одну услугу или про варианты на выбор: есть ли такие, что это, одно ли это, чем отличаются, какой выбрать",
    ),
)


async def basket_calc(
    question: str, positions: Sequence[dict[str, Any]], retail_rows: Sequence[dict[str, Any]], *, llm_allowed: bool = True
) -> dict[str, Any] | None:
    """Расчёт корзины (`basket_calculation`) и решение LLM, складывать ли позиции (`basket_total`): итог печатается, только если
    пациент перечислил разные услуги. Альтернативы и вопрос «есть ли / что это», «ни одна», сбой LLM, вызов запрещён (`strict`) —
    `summed: False`: позиции и взятие — как были, «Итого» нет (лучше без суммы, чем сумма альтернатив). Итога и так нет (одна
    позиция, цена 0) — LLM не зовётся.

    :param question: реплика пациента (как написал он сам)
    :param positions: позиции корзины
    :param retail_rows: строки прайса региона
    :param llm_allowed: можно ли звать LLM
    """

    calc = basket_calculation(positions, retail_rows)
    if calc is None or calc["positions"] < 2 or calc["total_unknown"]:
        return calc
    kind = "llm_off"
    if llm_allowed:
        choice = await choose(BASKET_TOTAL_TASK, question, _BASKET_TOTAL_OPTIONS)
        if choice.status is Status.PICKED and choice.ids:
            kind = choice.ids[0]
        else:
            kind = "none" if choice.status is Status.NONE_FIT else "failed"
    turn_telemetry.note(basket_total_kind=kind)
    if kind == _SUM:
        return calc
    turn_telemetry.note(basket_total="not_summed")
    return {**calc, "summed": False}
