"""Модуль домена цен.

Содержит pilot-миграцию price-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from agent_logic_2.nayka_api import api_price

from .. import shown_context, turn_parse, turn_telemetry
from . import _price_by_parse
from ..policies import handoff_message
from ..text_templates import PRICE_CLARIFY_NAME_TEXT, price_names_without_prices_text, price_not_found_text
from ._addresses_helpers import _extract_homecode_query
from ._common import (
    DOCTORS_TOP_N,
    _as_int,
    _coerce_top_n,
    _get_first_present,
    _normalise_input,
    _service_fallback,
)
from ._doctors_helpers import (
    _classify_catalog_service_kind,
    _collect_role_unit_names,
    _compact_specialization,
    _doctor_matches_primary_specialty,
    _doctor_sort_key,
    _extract_specialty_from_text,
    _has_reliable_doctor_service_link,
    _is_clean_consultation_row_name,
    _is_consultation_service_query,
    _is_direct_specialty_text_match,
    _pick_display_specialization,
    _select_effective_price_service_name,
    _should_prefer_retail_query_candidate,
    _specialty_label_for_doctor,
)
from ._prepare import _is_prepare_requested_in_price_query
from ._basket_select import basket_from_phrases, choose_basket_rows
from ._turn_phrases import _FILLER
from ._turn_phrases import _tokens as _phrase_tokens
from ._price_select_llm import (
    PriceSelection,
    distinct_services,
    general_clarify_on,
    reconcile_with_proposal,
    rows_overlap,
    select_price_rows,
)
from ._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _DOCTOR_PRICE_HINT_RE,
    _PRICE_CONSULT_HINT_RE,
    _PRICE_REQUEST_RE,
    _annotate_price_rows_with_care_context,
    _apply_service_synonyms,
    _build_compound_price_clarify_payload,
    _multi_price_items,
    multi_price_payload_from_items,
    _build_price_family_payload,
    _build_synonym_fork_payload,
    _extract_price_service_from_query,
    _is_city_only_reply,
    _is_generic_uzi_price_request,
    _price_show_all_requested,
    _is_strong_doctor_price_match,
    _match_drops_unsatisfiable_qualifier,
    _strong_phrase_guard_on,
    _price_family_payload_from_context,
    _price_query_tokens,
    _price_row_score,
    _rank_price_rows,
    _resolve_ambiguous_price_kind_with_llm,
    _select_patient_price_rows,
    query_has_unsatisfiable_qualifier,
    resolve_price_service_name_from_catalog,
    restrict_rows_to_family_letter,
)
from ._regions import (
    _has_explicit_non_samara_regions,
    _region_matches_samara_tokens,
)

if TYPE_CHECKING:
    from .core import Services

logger = logging.getLogger(__name__)


def _llm_family_payload(service_name: str, rows: list[dict[str, Any]], *, visible_limit: int = 10) -> dict[str, Any]:
    """Несколько строк, которые LLM признала той услугой, — варианты на выбор пациенту.

    Та же форма, что у `_build_price_family_payload`: рендер и выбор варианта следующим
    ходом работают без изменений.
    """

    variants = _annotate_price_rows_with_care_context(list(rows))
    visible_count = min(len(variants), visible_limit)
    remaining_count = max(0, len(variants) - visible_count)
    return {
        "service_name": service_name,
        "service_kind": "family_query",
        "family_variants": variants,
        "showing_all": False,
        "visible_limit": visible_limit,
        "remaining_count": remaining_count,
        "show_all_hint": (
            f"По вашему запросу найдено еще {remaining_count} вариантов. "
            'Чтобы показать их, напишите: "все".'
        )
        if remaining_count
        else "",
        "note": "price_llm_select",
    }


def _shown_phrase(query: str) -> str:
    """Фраза услуги в тексте пациенту («Не нашёл стоимость для «…»», «По запросу «…» нашёл похожие»). Под `price_by_parse = on`
    (цена, этап 2, Ш2) — фраза разбора этой реплики (срез реплики), а не вырезатель правил: «стоят ферритин» пациент больше не
    читает; одной фразы нет — пусто, текст без названия. Off — как было."""

    if _price_by_parse.enabled():
        return str(_price_by_parse.phrase(str(query or "")) or "").strip()
    return str(_extract_price_service_from_query(query) or query or "").strip()


def _search_text(query: str, query_text: str) -> str:
    """Что искать в прайсе — резолвер, семейство, развилка синонима МИС, буква семейства. Под `price_by_parse = on` (Ш2) — фраза
    разбора этой реплики через синонимы кода, как реплика; разбор решил «услуги нет» — пусто (услугу держит контекст); одной фразы
    нет (разбора нет, вход — не реплика хода, список) — реплика, как было."""

    if not _price_by_parse.enabled():
        return query_text
    named = _price_by_parse.phrase(str(query or ""))
    return query_text if named is None else _apply_service_synonyms(named)


def _service_phrase(query: str, query_text: str) -> str | None:
    """Фраза услуги — запасной поиск строк и кандидат розницы. Под `price_by_parse = on` (Ш2) — фраза разбора (услуги нет — None);
    одной фразы нет — вырезатель, как было."""

    if _price_by_parse.enabled():
        named = _price_by_parse.phrase(str(query or ""))
        if named is not None:
            return _apply_service_synonyms(named) or None
    return _extract_price_service_from_query(query_text)


def _parse_failed_clarify(query: str, entities: dict[str, Any]) -> dict[str, Any] | None:
    """Без LLM услугу не угадываем (решение владельца 22.10 (а); цена, этап 2, Ш4, `price_by_parse = on`): разбор этой реплики не
    решён (LLM не ответила, ответ не по контракту, сбой) — «уточните название», без поиска по словам и без списка правил с ценами.
    Реплика, которая сама — название или код клиники, сюда не доходит: её решают данные без LLM (`strong_whole`), и ответ на дословное
    название и есть путь без LLM. Цена попутно (запись, карточка врача; `__price_question` нет) — как было. Пометка — ответа политики
    (`price_match_policy: …`): услугу, которую до него подставило заземление по буквам, ответ снимает из контекста
    (BUG-2026-10-13-L13-CONTEXT-LEAK) — иначе «а сколько стоит?» следующим ходом назвал бы цену догадки."""

    if not _price_by_parse.enabled() or not entities.get("__price_question") or not _price_by_parse.failed(str(query or "")):
        return None
    turn_telemetry.note(price_policy="parse_failed_clarify")
    note = "price_match_policy: parse_failed"
    return {"service_name": "", "prices": [], "retail_prices": [], "doctors": [], "clarify_text": PRICE_CLARIFY_NAME_TEXT, "note": note, "entities_used": entities}


def _match_policy_payload(selection: Any, query_text: str, entities: dict[str, Any]) -> dict[str, Any] | None:
    """Политика силы совпадения (L-13) под `price_match_policy = on`: вместо ответа правил — честное
    «не нашёл» (LLM не нашла услугу и во втором круге) или названия без цен (LLM не ответила внятно
    на слабом совпадении; названий целым словом нет — перевод на оператора). Иначе None — дальше как
    раньше. Одна форма для `price_info` и `service_bundle_info`: оба рендера печатают `clarify_text`.
    """

    status = getattr(selection, "status", "")
    if status not in ("none_fit", "unavailable"):
        return None
    absent = status == "none_fit" and getattr(selection, "reason", "") == "qualifier_absent"
    # Суд условия ответил «той услуги с этим условием нет»: фраза разбора бывает названием без оторванного условия («ферритин» из
    # «ферритин по ОМС»), и «не нашёл для «ферритин»» читалось бы как «ферритина нет» — под `price_by_parse = on` без названия.
    phrase = "" if absent and _price_by_parse.enabled() else _shown_phrase(query_text)
    used = {**entities, "service_name_effective": phrase}
    note = f"price_match_policy: {status}"
    if status == "unavailable" and not selection.names:
        return _service_fallback(
            note=note,
            handoff_message=handoff_message("low_confidence"),
            entities=used,
            reason="low_confidence",
            extra={"prices": [], "retail_prices": []},
        )
    text = price_not_found_text(phrase) if status == "none_fit" else price_names_without_prices_text(phrase, selection.names)
    return {"service_name": phrase, "prices": [], "retail_prices": [], "doctors": [], "clarify_text": text, "note": note, "entities_used": used}


async def _multi_price_payload(
    query: str, query_text: str, retail_rows: list[dict[str, Any]], entities: dict[str, Any]
) -> dict[str, Any] | None:
    """Корзина: строки позиций выбирает LLM одним вызовом (ход (в), L-15a); сбой — политика сбоя корзины.

    Под `turn_parse = on` (L-15b, шаг 5) позиции — фразы разбора хода этой реплики: разбор решил одну
    услугу (`strong_whole`, `single`, `none_fit`, одна выбранная фраза) — не корзина, путь одной услуги;
    несколько фраз — корзина из них (`basket_from_phrases`); LLM не ответила — разрез по разделителям
    (`turn_parse.acting_for`). Иначе позиции режут правила, как раньше. Политика сбоя под `on`: слабая
    позиция без выбора LLM — варианты кода или название без цены, не одна строка правил с ценой."""

    llm_allowed = bool(entities.get("__price_question")) and str(entities.get("__runtime_llm_mode") or "").strip().lower() != "strict"
    choosing = llm_allowed
    from .. import turn_understanding  # понимание хода импортирует сервисы — импорт на вызове
    parse = turn_understanding.parse_for(str(query or ""))
    if parse is not None:
        if not parse.is_list:
            return None
        positions, unrecognized = await basket_from_phrases(
            str(query or ""),
            parse.items,
            parse.leftover,
            retail_rows,
            llm_allowed=choosing,
            journal=parse.status == turn_parse.ParseStatus.PICKED,
        )
        return multi_price_payload_from_items(positions, _names_a_service(unrecognized), allow_single=True)
    # Разбора нет или он не годится для действия (`empty`, `too_many`, `error`, вход — не реплика хода): корзина правил
    # с политикой сбоя разбора — строки слабых позиций выбирает LLM, при сбое — варианты кода (T7 шаг 3б: прежняя
    # корзина правил без выбора, путь `turn_parse = off`, снята).
    if turn_understanding.one_enabled() and not turn_understanding.rules_list(query_text, retail_rows):
        # Ход (б), этап 2: «список ли» — одно решение понимания хода (с защитами K1), корзина правил сама его не принимает
        # (разбор 21.10: в 19 из 21 расхождения корзина без K1 резала одну услугу на чужие строки).
        return None
    items, unrecognized = _multi_price_items(query_text, retail_rows)
    if not items:
        return None
    refined = await choose_basket_rows(str(query or query_text), items, unrecognized, retail_rows, llm_allowed=choosing)
    return multi_price_payload_from_items(*refined, allow_single=True)


def _names_a_service(fragments: list[str]) -> list[str]:
    """«Не распознал» корзины из фраз (под `turn_parse = on`): фрагмент только из обвязки и родовых слов
    («анализов» из «Стоимость анализов: …») услугу не называет — не печатаем. Мерило — то же, что у
    генератора фраз для промежутков между позициями (`_turn_phrases._FILLER`), без новых словарей
    (STEP-2026-10-14-L15B-6-T3, п. 8)."""

    return [fragment for fragment in fragments if any(token.key not in _FILLER for token in _phrase_tokens(str(fragment or "")))]


def _parse_phrase_judged_by_llm(query: str) -> str:
    """Класс A (решение владельца 15.10; единственный путь с 17.10 — выключатель `parse_price_guard` снят): фраза,
    которую LLM разбора выбрала одной слабой услугой, — её «есть ли такое в прайсе» решает выбор строки, а не гард П2
    по словарю каталога. Иначе пусто — гард как раньше (сбой LLM выбора строки — ответ правил, а у них гард)."""

    from .. import turn_understanding  # понимание хода импортирует сервисы — импорт на вызове

    parse = turn_understanding.parse_for(query)
    if parse is None or parse.is_list or len(parse.items) != 1 or parse.status != turn_parse.ParseStatus.PICKED:
        return ""
    item = parse.items[0]
    return "" if item.rows else str(item.text or "")


def _p2_name_from_entity(entity_service_name: str, query_service_name: str | None, service_name: str, query_text: str) -> bool:
    """Судья П2 (единственный путь с 17.10 — выключатель `p2_llm_judge` снят): резолвер по реплике названия не дал, и
    путь цены ведёт название из сущности NLU / контекста (без него — путь одной услуги или «Уточните»). Уточнение П2
    проверяет вызывающий по строкам прайса."""

    return (
        bool(str(entity_service_name or "").strip())
        and not query_service_name
        and bool(str(service_name or "").strip())
        and not _is_city_only_reply(query_text)
    )


async def _p2_llm_verdict(
    query: str,
    retail_rows: list[dict[str, Any]],
    *,
    candidates: list[dict[str, Any]],
    shown: list[dict[str, Any]],
    hints: list[str],
    context: str,
    runtime_llm_mode: str,
) -> PriceSelection | None:
    """«Та ли это услуга с этим свойством» (BUG-2026-10-15-P2-ENTITY-BYPASS): резолвер отказал
    из-за уточнения, которого в прайсе нет (П2, `query_has_unsatisfiable_qualifier`), а название пришло из сущности
    NLU / контекста. Гард по буквам здесь не годен (встречный свип 15.10: «первый приём невролога», «для иностранного
    гражданина» — разговорное слово, не уточнение) — судит выбор строки (`judge_qualifier`, правило 3 промпта отсекает
    «по ОМС»), кандидаты — строки названия.

    Выбор LLM (строки) — они; LLM ответила «не та» (`none_fit`) — честное «не нашёл» (`_match_policy_payload`);
    сбой LLM, ответ не по делу («слишком общее», «не услуга»), `unavailable` — None: вызывающий продолжает ход как без
    выключателя (прежний выбор строки и политика L-13 — с 16.10 единственная, T7 шаг 2)."""

    selection = await select_price_rows(
        query,
        retail_rows,
        candidates=candidates,
        shown=shown,
        hints=hints,
        context=context,
        runtime_llm_mode=runtime_llm_mode,
        judge_qualifier=True,
        budget_clarify=True,  # отказ лимита решений хода — уточнение (`price_budget_clarify`)
    )
    if selection is None or (not selection.rows and selection.status != "none_fit"):
        turn_telemetry.note(p2_llm_judge="fallback")
        return None
    turn_telemetry.note(p2_llm_judge="rows" if selection.rows else "not_found")
    return selection


def _parse_strong_selection(query: str) -> PriceSelection | None:
    """Под `turn_parse = on` (L-15b, шаг 5): разбор этой реплики решил одну услугу, сильную по данным МИС
    (дословное название, синоним МИС, код), — её строки без выбора LLM, вместо ответа правил по буквам
    (решение владельца 13.10: STRONG — только данные МИС). Иначе None — выбор строки как раньше."""

    from .. import turn_understanding  # понимание хода импортирует сервисы — импорт на вызове

    parse = turn_understanding.parse_for(query)
    if parse is None or parse.is_list or len(parse.items) != 1 or not parse.items[0].rows:
        return None
    return PriceSelection(rows=tuple(dict(row) for row in parse.items[0].rows), status="strong")


def _qualifier_outside_strong_phrase(query: str, strong: PriceSelection, retail_rows: list[dict[str, Any]]) -> bool:
    """В реплике ВНЕ сильной фразы разбора есть слово, которого нет в прайсе вовсе («по ОМС» рядом с «приёмом
    терапевта»). Мерило — гард П2 пути одной услуги (`_match_drops_unsatisfiable_qualifier`), «найденная услуга» —
    фраза и названия её строк: слово фразы и словоформы названий уточнением не считаются. Срабатывает только там,
    где тот же гард по всей реплике отказал бы и под turn_parse = off, — это повод спросить LLM, а не вердикт."""

    from .. import turn_understanding  # понимание хода импортирует сервисы — импорт на вызове

    parse = turn_understanding.parse_for(query)
    phrase = str(parse.items[0].text or "") if parse is not None and parse.items else ""
    names = [str(row.get("serviceName") or row.get("name") or "") for row in strong.rows]
    return _match_drops_unsatisfiable_qualifier(query, " ".join([phrase, *names]), retail_rows)


# Причины пустого выбора, при которых суд условия у сильной фразы говорит «той услуги с этим условием нет»:
# правило 3 промпта модуля (`qualifier_absent`) и пустой код (модель ответила «none», ничего не выбрав).
_CONDITION_ABSENT = frozenset({"qualifier_absent", ""})


async def _parse_strong_answer(
    query: str, retail_rows: list[dict[str, Any]], *, context: str, runtime_llm_mode: str
) -> PriceSelection | None:
    """Ответ сильной фразы разбора (`turn_parse = on`) с гардом П2 (BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS).

    LLM разбора может выбрать сильную фразу, оторвав от неё условие («приём терапевта» из «приём терапевта по ОМС»),
    — тогда её строки по данным МИС — платная цена без того, о чём спрашивал пациент. Под `strong_phrase_guard = llm`
    фраза без слов вне прайса отдаёт строки, как раньше; со словом вне прайса его судит выбор строки, один круг
    (`judge_qualifier`, правило 3 промпта отсекает «по ОМС»; разговорное слово — «первый приём» — LLM пропускает,
    словарь бы заглушил: урок BUG-2026-10-15-P2-ENTITY-BYPASS), кандидаты — строки фразы. LLM судит условие, а не
    личность услуги — её решают данные МИС (решение владельца 13.10):

    - выбор строк, пересекающийся со строками фразы, — строки фразы (условие им не мешает; соседи, которых LLM
      добавила к синониму МИС, — её мнение о самой услуге, а не об условии); выбор только других строк — список: строки
      фразы и выбор LLM (неоднозначность — список, решение 22.07: «ОАК срочный» → + Cito верно, «ФГДС под седацией» →
      профиль анализов «под наркозом» — чужое, живой свип 16.10);
    - «условия нет» (`qualifier_absent`, пустой код) — `none_fit`: «не нашёл» (`_match_policy_payload`);
    - «другая услуга» (`other_service`) — строки сильной фразы: спор LLM с синонимом МИС не довод против данных;
    - «слишком общее», «не услуга», сбой LLM — None: сильная фраза ход не решает, дальше путь правил с гардом — как под
      turn_parse = off (дезинформация ценой хуже уточнения).

    :return: выбор строк (или решение «не нашёл»); None — разбора сильной фразы нет или ход ведут правила
    """

    strong = _parse_strong_selection(query)
    if strong is None:
        return None
    # Под `price_by_parse = on` гард тот же (карточка Ш7, 25.10, RISK-2026-10-25-STRONG-PHRASE-NO-GUARD): без него живая LLM разметки
    # отрывала «по ОМС» от дословного названия в 59 вопросах из 82 свипа П2 — платная цена вместо «не нашёл»; правило 3 её промпта
    # модель не держит.
    if not _strong_phrase_guard_on() or not _qualifier_outside_strong_phrase(query, strong, retail_rows):
        return strong
    from .. import turn_understanding  # понимание хода импортирует сервисы — импорт на вызове
    parse = turn_understanding.parse_for(query)
    phrase = str(parse.items[0].text or "") if parse is not None and parse.items else ""
    selection = await select_price_rows(
        query,
        retail_rows,
        candidates=list(strong.rows),
        shown=list(strong.rows),
        hints=[phrase],
        context=context,
        runtime_llm_mode=runtime_llm_mode,
        judge_qualifier=True,
        one_round=True,
        budget_clarify=True,  # отказ лимита решений хода — уточнение (`price_budget_clarify`)
    )
    if selection is not None and selection.rows:
        turn_telemetry.note(strong_phrase_guard="rows")
        if rows_overlap(selection.rows, strong.rows):
            return strong
        return PriceSelection(rows=(*strong.rows, *selection.rows), status="picked")
    reason = selection.reason if selection is not None and selection.status == "none_fit" else None
    if reason in _CONDITION_ABSENT:
        turn_telemetry.note(strong_phrase_guard="not_found")
        return selection
    if reason == "other_service":
        turn_telemetry.note(strong_phrase_guard="strong")
        return strong
    turn_telemetry.note(strong_phrase_guard="fallback")
    return None


def _names_no_service(query: str) -> bool:
    """L-14, шаг 3: понимание этой реплики есть, а услуги в ней не названо («а цена?»,
    «сколько стоит?» — упоминания нет). Тогда фраза вырезателя («а») не перебивает показанную услугу контекста
    и не ищется в прайсе: вопрос — о том, что пациент уже видел (находка 1 STEP-…-L14-PREP-CONSUMERS). Под
    `turn_understanding_one = on` (ход (б), этап 2, U8) — только действующее понимание (`of_reply`), в тени — прежний путь."""

    from .. import turn_understanding  # понимание хода импортирует сервисы цены — не на уровне модуля

    tu = turn_understanding.of_reply(str(query or "")) if turn_understanding.one_enabled() else turn_understanding.current()
    return tu is not None and tu.for_text(str(query or "")) and not tu.services


def _rows_shown_before(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Строки прайса, которые показал прошлый ответ, — если реплика услуги не называет («сколько стоит?», «а
    цена?»; `_names_no_service`). Вопрос — о том, что пациент уже видел: отвечаем ровно
    этими строками, а не семейством по названию из контекста (находка 4: после одной строки «Глюкоза» повтор
    показывал «Глюкозу в моче» и «в синовиальной жидкости»). None — прежний путь."""

    shown = shown_context.previous()
    if not shown or not _names_no_service(query):
        return None
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("serviceName") or row.get("name") or "").strip() if isinstance(row, dict) else ""
        by_name.setdefault(name, row)
    found = [by_name[name] for name in shown if name in by_name]
    return found or None


def _llm_selection_service_name(query_text: str, rows: list[dict[str, Any]], *, reply: str = "") -> str:
    """Подпись к ответу из выбора LLM: одна строка — её название, несколько — фраза пациента. Под `price_by_parse = on` (Ш2) —
    фраза разбора реплики `reply`; её нет — название первой строки (слова реплики подписью не печатаются)."""

    first = str(rows[0].get("serviceName") or rows[0].get("name") or "").strip() if rows else ""
    if len(rows) == 1:
        return first
    if _price_by_parse.enabled():
        return _shown_phrase(reply) or first
    return str(_extract_price_service_from_query(query_text) or query_text or "").strip()


async def _prefetch_price_snapshots() -> None:
    """Сегодняшние срезы прайса и priceUnits — скачать в потоке, если их нет: дальше ход читает их синхронно
    (резолвер, гард уточнения, контекст оказания), и скачивание не должно идти в цикле событий (DATA-7)."""

    try:
        await asyncio.to_thread(api_price.prefetch_price_today)
    except Exception:
        logger.warning("предзагрузка срезов прайса не удалась — ход читает последние", exc_info=True)


async def price_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает информацию о ценах на услуги или приём у врача.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: словарь с ценовыми строками или сообщением об ошибке
    """

    await _prefetch_price_snapshots()

    async def _load_with_retry(fn: Any, *args: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.12)
                    continue
        if last_exc is not None:
            raise last_exc
        return []

    doctor_id = _as_int(entities.get("doctor_id"))
    doctor_name = _get_first_present(entities, ["doctor_name", "doctor", "fio", "last_name", "doctor_last_name"]) or ""
    resolved_doctor_fio: str | None = None
    if not doctor_id and doctor_name:
        doctor_id, resolved_doctor_fio = await self._resolve_doctor_id_from_name(doctor_name)
    if not doctor_id and query and _DOCTOR_PRICE_HINT_RE.search(str(query or "")):
        # Fallback для фраз вида "сколько стоит ... у Белохвостиковой":
        # извлекаем врача из полного текста запроса, даже если classifier не выделил doctor_name.
        doctor_id, q_resolved_fio = await self._resolve_doctor_id_from_name(str(query))
        if doctor_id and q_resolved_fio:
            resolved_doctor_fio = q_resolved_fio
            doctor_name = q_resolved_fio
    # Нормализуем пациентские синонимы к терминам каталога (BUG-2026-06-23-01:
    # «забор крови»→«взятие крови», «электромиография»→«ЭМГ») до матчинга услуги.
    entity_service_name = _apply_service_synonyms(_get_first_present(entities, ["service_name", "test_name"]) or "")
    query_text = _apply_service_synonyms(str(query or "").strip())
    search_text = _search_text(query, query_text)
    has_price_request = bool(query_text and _PRICE_REQUEST_RE.search(query_text))
    if _price_show_all_requested(query_text, query):
        family_payload = _price_family_payload_from_context(entities, show_all=True)
        if family_payload:
            family_payload["entities_used"] = entities
            return family_payload
    if _is_generic_uzi_price_request(query_text):
        return {
            "prices": [],
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
            "note": "price_info: generic_uzi_clarify",
            "entities_used": entities,
        }
    doctor_query_specialty = _extract_specialty_from_text(query_text) if (doctor_id and has_price_request) else ""
    p2_from_entity = False

    if doctor_id:
        # Для doctor-specific PRICE не приземляемся в городский retail-catalog:
        # иначе вопрос "у Иванова" может маппиться в случайную услугу по всему прайсу.
        query_service_name = _service_phrase(query, query_text) if query_text else None
        if not query_service_name:
            query_service_name = entity_service_name
        # "сколько стоит прием у <врач>" без специальности:
        # принудительно удерживаем консультационный контекст вместо stale service_name.
        if has_price_request and _PRICE_CONSULT_HINT_RE.search(query_text) and not doctor_query_specialty:
            if not _is_consultation_service_query(str(query_service_name or "")):
                query_service_name = "консультация"
        service_name = query_service_name or ""
        # Если это doctor-specific price без явной услуги, оставляем query как fallback
        # (для редких строк doctor_price, где нет стандартных маркеров).
        if not service_name and has_price_request and _DOCTOR_PRICE_HINT_RE.search(query_text):
            service_name = query_text
    else:
        if entity_service_name and _is_city_only_reply(query_text):
            query_service_name = None
        else:
            query_service_name = resolve_price_service_name_from_catalog(
                search_text,
                current_service_name=entity_service_name,
            )
            # Гард неудовлетворимого уточнения — ЧЕТВЁРТЫЙ стык. Отказ резолвера
            # обходился этим fallback'ом: «приём терапевта по ОМС» → сырая фраза
            # «прием терапевт» → ранжирование → платная цена. Если каталог такого
            # свойства не знает вовсе — честное «не нашёл», а не цена без него.
            if not query_service_name and not query_has_unsatisfiable_qualifier(query_text) and not _names_no_service(query):
                query_service_name = _service_phrase(query, query_text)
        service_name = _select_effective_price_service_name(
            entity_service_name,
            query_service_name,
        )
        p2_from_entity = _p2_name_from_entity(entity_service_name, query_service_name, service_name, query_text)
    needle = _normalise_input(service_name)

    if doctor_id:
        # doctor prices: branch-level regionId из /doctorServicePricesByRegion cache
        try:
            prices = await _load_with_retry(api_price.load_doctor_prices)
        except Exception:
            return _service_fallback(
                note="price_info source unavailable",
                handoff_message=handoff_message("service_error_prices"),
                entities=entities,
                extra={"prices": []},
            )
        doc_prices = [p for p in prices if _as_int(p.get("doctorId")) == doctor_id]
        if needle:
            ranked = _rank_price_rows(doc_prices, service_name, limit=10)
            if (
                not ranked
                and has_price_request
                and _PRICE_CONSULT_HINT_RE.search(query_text)
            ):
                ranked = [
                    p for p in doc_prices
                    if _is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                ]
            doc_prices = ranked
        if not needle:
            if has_price_request and _PRICE_CONSULT_HINT_RE.search(query_text):
                consult_rows = [
                    p for p in doc_prices
                    if _is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                ]
                if consult_rows:
                    doc_prices = consult_rows
            doc_prices = sorted(
                [p for p in doc_prices if isinstance(p, dict)],
                key=lambda p: (_normalise_input(str(p.get("serviceName") or "")), _as_int(p.get("cost")) or 0),
            )
        doc_prices = _annotate_price_rows_with_care_context(doc_prices[:10])
        return {
            "prices": doc_prices,
            "note": "price_info: doctorServicePricesByRegion (branch-level regionId)",
            "entities_used": {
                **entities,
                "doctor_id_resolved": doctor_id,
                "doctor_name_resolved": resolved_doctor_fio or doctor_name or "",
                "service_name_effective": service_name,
            },
        }

    # Цена, этап 2, Ш4: до корзины правил и поиска по словам — разбор не решён, без LLM услугу (и список) не угадываем.
    clarify = _parse_failed_clarify(query, entities)
    if clarify is not None:
        return clarify
    # retail prices: city-level regionId в /priceByRegion/{cityRegionId}
    try:
        price_rows = await _load_with_retry(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
    except Exception:
        return _service_fallback(
            note="price_info source unavailable",
            handoff_message=handoff_message("service_error_prices"),
            entities=entities,
            extra={"prices": []},
        )
    retail_rows_clean = [p for p in price_rows if isinstance(p, dict)]
    p2_judged = p2_from_entity and query_has_unsatisfiable_qualifier(query_text, retail_rows_clean)
    # Мульти-услуговый запрос («ВИЧ, гепатит, ОАК»): активируется только если
    # ≥2 фрагментов приземлились на реальные услуги каталога. Иначе — fallback
    # на существующий single-service путь, старые кейсы остаются без изменений.
    multi_payload = await _multi_price_payload(query, query_text, retail_rows_clean, entities)
    if multi_payload is not None:
        multi_payload["entities_used"] = {
            **entities,
            "service_name_effective": str(multi_payload.get("service_name") or "").strip(),
        }
        return multi_payload
    # Синоним МИС указывает на несколько услуг («оак» → 3, «рак» → 5): показываем
    # ВСЕ варианты, выбирает пациент. Стоит ПОСЛЕ мульти-пути, чтобы корзина
    # («оак, ферритин») по-прежнему считалась как корзина, и ДО семейного —
    # словарь клиники доверенней лексической догадки.
    synonym_fork = _build_synonym_fork_payload(search_text, retail_rows_clean)
    if synonym_fork is not None and not doctor_id:
        synonym_fork["entities_used"] = {
            **entities,
            "service_name_effective": str(synonym_fork.get("service_name") or "").strip(),
        }
        return synonym_fork
    # Что спросил пациент, решает LLM (класс «цена не той услуги», решение владельца
    # 30.09); правила собирают кандидатов, их ответ — первый кандидат и ответ по
    # умолчанию, если LLM молчит. LLM зовётся не больше одного раза за вызов.
    async def _llm_select(proposal: list[dict[str, Any]], shown: list[dict[str, Any]], *, entity_name: bool = False) -> Any:
        if not entities.get("__price_question"):
            return None  # цена попутно (запись, карточка врача) — по правилам, без LLM
        strong = await _parse_strong_answer(
            str(query or ""),
            retail_rows_clean,
            context=entity_service_name,
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
        )
        if strong is not None:
            return strong
        if entity_name and p2_judged:
            verdict = await _p2_llm_verdict(
                str(query or ""),
                retail_rows_clean,
                candidates=proposal,
                shown=shown,
                hints=[service_name],
                context=entity_service_name,
                runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
            )
            if verdict is not None:
                return verdict
            # Сбой судьи — ход как без судьи: прежний выбор строки и его политика (L-13), а не ответ правил мимо неё.
        judged = _parse_phrase_judged_by_llm(str(query or ""))
        return await select_price_rows(
            str(query or ""),
            retail_rows_clean,
            candidates=proposal,
            shown=shown,
            hints=[service_name, judged],
            context=entity_service_name,
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
            judge_qualifier=bool(judged),
            budget_clarify=True,  # отказ лимита решений хода — уточнение (`price_budget_clarify`)
            general_as_none_fit=general_clarify_on(),  # «слишком общее» — уточнение, не цены правил (`price_general_clarify`)
        )

    def _llm_answer(rows: list[dict[str, Any]]) -> dict[str, Any]:
        name = _llm_selection_service_name(query_text, rows, reply=str(query or ""))
        if len(rows) > 1:
            payload = _llm_family_payload(name, rows)
        else:
            payload = {
                "prices": _annotate_price_rows_with_care_context(list(rows)),
                "service_kind": "lab" if _classify_catalog_service_kind(
                    name,
                    query_text=query_text,
                    retail_rows=list(rows),
                    has_exact_doctor_link=False,
                    is_consult_query=False,
                ) == "lab" else "",
                "note": "price_info: price_llm_select",
            }
        payload["entities_used"] = {**entities, "service_name_effective": name}
        return payload

    shown_before = None if doctor_id else _rows_shown_before(str(query or ""), retail_rows_clean)
    if shown_before:
        if len(shown_before) == 1:
            payload = _llm_answer(shown_before)
        else:  # подпись — услуга контекста, а не слова вопроса («По запросу «сколько стоит»»)
            payload = _llm_family_payload(entity_service_name or str(shown_before[0].get("serviceName") or ""), shown_before)
            payload["entities_used"] = {**entities, "service_name_effective": entity_service_name}
        payload["note"] = "price_info: shown_before"
        return payload
    family_payload = _build_price_family_payload(
        search_text,
        retail_rows_clean,
        show_all=False,
        visible_limit=10,
    )
    if family_payload and not doctor_id:
        variants = list(family_payload.get("family_variants") or [])
        shown = variants[: int(family_payload.get("visible_limit") or 10)]
        selection = await _llm_select(variants, shown)
        stop = _match_policy_payload(selection, str(query or ""), entities)
        if stop is not None:
            return stop
        llm_rows = reconcile_with_proposal(shown, selection)
        if llm_rows is not None:
            return _llm_answer(llm_rows)
        family_payload["entities_used"] = {
            **entities,
            "service_name_effective": str(family_payload.get("service_name") or "").strip(),
        }
        return family_payload
    if not needle:
        selection = await _llm_select([], [])
        stop = _match_policy_payload(selection, str(query or ""), entities)
        if stop is not None:
            return stop
        llm_rows = reconcile_with_proposal([], selection)
        if llm_rows is not None:
            return _llm_answer(llm_rows)
        return {"prices": [], "note": "no service query", "entities_used": entities}
    retail_query = service_name
    if query_text and not _is_city_only_reply(query_text) and not _names_no_service(query):
        query_candidate = _service_phrase(query, query_text)
        if _should_prefer_retail_query_candidate(query_candidate or "", service_name):
            retail_query = query_candidate or service_name
    # Гард различающего токена семейства для МАЛЫХ семейств (гепатит А/D/E —
    # <3 вариантов, family-mode их не берёт): если запрос несёт букву, ранжируем
    # ТОЛЬКО по строкам этой буквы, иначе всплывёт чужой гепатит (В за запрос про А).
    # Для запросов без дискриминатора список не меняется.
    family_scoped_rows = restrict_rows_to_family_letter(
        search_text, [p for p in price_rows if isinstance(p, dict)]
    )
    matches = _select_patient_price_rows(
        family_scoped_rows,
        retail_query,
        limit=10,
    )
    # BUG-H sibling: query-кандидат из текста мог нести framing-шум («анализ ттг»),
    # который даёт ПУСТУЮ выборку (AND-семантика: нет строки и с «анализ», и с «ттг»),
    # хотя резолвленная услуга (service_name) находится. Откатываемся на неё, чтобы
    # не терять цену на «сколько стоит анализ на <X>». Срабатывает только на пустом
    # результате — рабочие кейсы не затрагиваются.
    if not matches and retail_query != service_name:
        matches = _select_patient_price_rows(
            family_scoped_rows,
            service_name,
            limit=10,
        )
    # Рендер показывает пациенту до пяти строк — их и сверяем.
    selection = await _llm_select(matches, matches[:5], entity_name=True)
    stop = _match_policy_payload(selection, str(query or ""), entities)
    if stop is not None:
        return stop
    llm_rows = reconcile_with_proposal(matches[:5], selection)
    if llm_rows is not None:
        return _llm_answer(llm_rows)
    matches = _annotate_price_rows_with_care_context(matches)
    return {
        "prices": matches,
        "service_kind": "lab" if _classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=matches,
            has_exact_doctor_link=False,
            is_consult_query=False,
        ) == "lab" else "",
        "note": f"price_info: priceByRegion({SAMARA_PRICE_REGION_ID})",
        "entities_used": {
            **entities,
            "service_name_effective": service_name,
        },
    }


async def service_bundle_info(
    self: "Services",
    query: str,
    entities: dict[str, Any],
    *,
    top_n: int | None = None,
) -> dict[str, Any]:
    top_limit = _coerce_top_n(top_n, default=DOCTORS_TOP_N)
    await _prefetch_price_snapshots()
    # Нормализуем пациентские синонимы к терминам каталога (BUG-2026-06-23-01:
    # «забор крови»→«взятие крови», «электромиография»→«ЭМГ») до матчинга услуги.
    entity_service_name = _apply_service_synonyms(_get_first_present(entities, ["service_name", "test_name"]) or "")
    query_text = _apply_service_synonyms(str(query or "").strip())
    search_text = _search_text(query, query_text)
    if _price_show_all_requested(query_text, query):
        family_payload = _price_family_payload_from_context(entities, show_all=True)
        if family_payload:
            family_payload["entities_used"] = entities
            return family_payload
    if _is_generic_uzi_price_request(query_text):
        return {
            "service_name": "УЗИ",
            "retail_prices": [],
            "doctors": [],
            "prepare": "",
            "show_prepare": False,
            "top_n_applied": top_limit,
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
            "note": "service_bundle_info: generic_uzi_clarify",
            "entities_used": entities,
        }
    clarify = _parse_failed_clarify(query, entities)
    if clarify is not None:
        clarify["top_n_applied"] = top_limit
        return clarify
    try:
        retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
    except Exception:
        retail_rows = []
    retail_rows = [p for p in retail_rows if isinstance(p, dict)]

    # Что спросил пациент, решает LLM (класс «цена не той услуги», решение владельца
    # 30.09): «приём эндокринолога» → не «приём хирурга (эндокринологическое
    # отделение)». Правила собирают кандидатов, их ответ — первый кандидат и ответ по
    # умолчанию, если LLM молчит. LLM зовётся не больше одного раза за вызов.
    async def _llm_select(
        proposal: list[dict[str, Any]], shown: list[dict[str, Any]], rules_guess: str = "", *, p2_judged: bool = False
    ) -> Any:
        if not entities.get("__price_question"):
            return None  # цена попутно (запись, карточка врача) — по правилам, без LLM
        strong = await _parse_strong_answer(
            str(query or ""),
            retail_rows,
            context=entity_service_name,
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
        )
        if strong is not None:
            return strong
        if p2_judged:
            verdict = await _p2_llm_verdict(
                str(query or ""),
                retail_rows,
                candidates=proposal,
                shown=shown,
                hints=[rules_guess],
                context=entity_service_name,
                runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
            )
            if verdict is not None:
                return verdict
            # Сбой судьи — ход как без судьи (прежний выбор строки и политика L-13).
        return await select_price_rows(
            str(query or ""),
            retail_rows,
            candidates=proposal,
            shown=shown,
            hints=[rules_guess],
            context=entity_service_name,
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
            budget_clarify=True,  # отказ лимита решений хода — уточнение (`price_budget_clarify`)
            general_as_none_fit=general_clarify_on(),  # «слишком общее» — уточнение, не цены правил (`price_general_clarify`)
        )

    def _llm_variants(rows: list[dict[str, Any]]) -> dict[str, Any]:
        llm_payload = _llm_family_payload(_llm_selection_service_name(query_text, rows, reply=str(query or "")), rows)
        llm_payload["top_n_applied"] = top_limit
        llm_payload["entities_used"] = entities
        return llm_payload

    llm_rows: list[dict[str, Any]] | None = _rows_shown_before(str(query or ""), retail_rows)
    if llm_rows is not None and len(llm_rows) > 1:  # показанные варианты — те же, что видел пациент
        shown_payload = _llm_family_payload(entity_service_name or str(llm_rows[0].get("serviceName") or ""), llm_rows)
        shown_payload.update(top_n_applied=top_limit, entities_used=entities, note="service_bundle_info: shown_before")
        return shown_payload
    family_payload = None if llm_rows else _build_price_family_payload(
        search_text,
        retail_rows,
        show_all=False,
        visible_limit=10,
    )
    if family_payload:
        variants = list(family_payload.get("family_variants") or [])
        shown = variants[: int(family_payload.get("visible_limit") or 10)]
        selection = await _llm_select(variants, shown)
        stop = _match_policy_payload(selection, str(query or ""), entities)
        if stop is not None:
            return stop
        llm_rows = reconcile_with_proposal(shown, selection)
        if llm_rows is None:
            family_payload["top_n_applied"] = top_limit
            family_payload["entities_used"] = entities
            return family_payload
        if len(llm_rows) > 1:
            return _llm_variants(llm_rows)
    if llm_rows:
        # Одна строка — дальше обычный путь: цена, врачи по её коду.
        query_service_name = _llm_selection_service_name(query_text, llm_rows, reply=str(query or ""))
    elif entity_service_name and _is_city_only_reply(query_text):
        query_service_name = None
    else:
        query_service_name = resolve_price_service_name_from_catalog(
            search_text,
            current_service_name=entity_service_name,
        )
        # Тот же гард, что в `price_info` (см. комментарий там): fallback на сырую
        # фразу не имеет права обходить отказ по неудовлетворимому уточнению.
        if not query_service_name and not query_has_unsatisfiable_qualifier(query_text, retail_rows) and not _names_no_service(query):
            query_service_name = _service_phrase(query, query_text)
    service_name = (
        query_service_name
        if llm_rows
        else _select_effective_price_service_name(entity_service_name, query_service_name)
    )
    p2_judged = (
        not llm_rows
        and _p2_name_from_entity(entity_service_name, query_service_name, service_name, query_text)
        and query_has_unsatisfiable_qualifier(query_text, retail_rows)
    )
    needle = _normalise_input(service_name)

    out: dict[str, Any] = {
        "service_name": service_name,
        "retail_prices": [],
        "doctors": [],
        "prepare": "",
        "show_prepare": False,
        "top_n_applied": top_limit,
        "note": "service_bundle_info",
        "entities_used": {
            **entities,
            "service_name_effective": service_name,
        },
    }
    if not needle:
        if llm_rows is None:
            selection = await _llm_select([], [])
            stop = _match_policy_payload(selection, str(query or ""), entities)
            if stop is not None:
                return stop
            llm_rows = reconcile_with_proposal([], selection)
        if not llm_rows:
            out["note"] = "service_bundle_info: no service query"
            return out
        if len(llm_rows) > 1:
            return _llm_variants(llm_rows)
        service_name = _llm_selection_service_name(query_text, llm_rows, reply=str(query or ""))
        needle = _normalise_input(service_name)

    # 1) Retail price by city-level regionId (Самара = 3).
    retail_query = service_name
    retail_prefers_query_candidate = False
    if query_text and not _is_city_only_reply(query_text) and not _names_no_service(query):
        query_candidate = _service_phrase(query, query_text)
        retail_prefers_query_candidate = _should_prefer_retail_query_candidate(
            query_candidate or "",
            service_name,
        )
        if retail_prefers_query_candidate:
            retail_query = query_candidate or service_name
    try:
        out["retail_prices"] = _select_patient_price_rows(
            retail_rows,
            retail_query,
            limit=5,
        )
        # BUG-H sibling: framing-шум в query-кандидате («анализ ттг») даёт ПУСТУЮ
        # выборку, хотя резолвленная услуга находится. Откат на неё (только на пустом
        # результате), и не подменяем service_name на шумный кандидат.
        if not out["retail_prices"] and retail_query != service_name:
            out["retail_prices"] = _select_patient_price_rows(retail_rows, service_name, limit=5)
            if out["retail_prices"]:
                retail_prefers_query_candidate = False
                retail_query = service_name
        out["retail_prices"] = _annotate_price_rows_with_care_context(out["retail_prices"])
    except Exception:
        out["retail_prices"] = []
        out["note"] = "service_bundle_info: retail source unavailable"
    if retail_prefers_query_candidate and retail_query:
        out["service_name"] = retail_query
    if llm_rows is None:
        # Сверяем с тем, что пациент увидел бы: у анализа — до пяти вариантов, у
        # приёма и процедуры — одна верхняя строка (рендер показывает её).
        shown_kind = _classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=out["retail_prices"],
            has_exact_doctor_link=False,
            is_consult_query=_is_consultation_service_query(service_name),
        )
        shown = out["retail_prices"] if shown_kind == "lab" else out["retail_prices"][:1]
        selection = await _llm_select(out["retail_prices"], shown, service_name, p2_judged=p2_judged)
        stop = _match_policy_payload(selection, str(query or ""), entities)
        if stop is not None:
            return stop
        llm_rows = reconcile_with_proposal(shown, selection)
        if llm_rows is not None and len(llm_rows) > 1 and not rows_overlap(llm_rows, out["retail_prices"]):
            # Правила промахнулись целиком, а LLM нашла несколько вариантов — их и
            # показываем на выбор, без врачей чужой услуги.
            return _llm_variants(llm_rows)
        if llm_rows is not None and general_clarify_on() and distinct_services(llm_rows):
            # LLM выбрала несколько разных услуг (не варианты одной): дальше путь показал бы первую строку с её врачами —
            # «консультация врача» → приём хирурга (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 3). Список на выбор.
            return _llm_variants(llm_rows)
    if llm_rows:
        out["retail_prices"] = _annotate_price_rows_with_care_context(list(llm_rows))
        service_name = str(llm_rows[0].get("serviceName") or llm_rows[0].get("name") or service_name).strip()
        out["service_name"] = service_name
        out["llm_selected"] = True
        out["entities_used"] = {**entities, "service_name_effective": service_name}

    compound_payload = _build_compound_price_clarify_payload(
        query_text=query_text,
        entities=entities,
        primary_service_name=service_name,
        retail_rows=retail_rows,
        primary_retail_prices=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
    )
    if compound_payload:
        compound_payload["top_n_applied"] = top_limit
        compound_payload["entities_used"] = {
            **entities,
            "service_name_effective": service_name,
        }
        return compound_payload

    # 2) Top-N doctors by ord among doctors that have the matched service in doctor prices.
    top_retail = out["retail_prices"][0] if isinstance(out.get("retail_prices"), list) and out["retail_prices"] else {}
    target_homecode = _normalise_input(
        str(top_retail.get("serviceHomecode") or top_retail.get("homecode") or "")
    )
    is_consult_query = _is_consultation_service_query(service_name)
    preliminary_kind = _classify_catalog_service_kind(
        service_name,
        query_text=query_text,
        retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
        has_exact_doctor_link=False,
        is_consult_query=is_consult_query,
    )
    query_norm = _normalise_input(service_name)
    query_tokens = _price_query_tokens(service_name)
    homecode_query = _extract_homecode_query(service_name)
    matched_price_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
    exact_link_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
    samara_tokens: set[str] = set()
    by_id: dict[int, dict[str, Any]] = {}
    doctor_prices: list[dict[str, Any]] = []
    service_kind = preliminary_kind
    if preliminary_kind not in {"lab", "diagnostic_no_doctor"}:
        samara_tokens = await self._samara_region_tokens()
        doctors = await self._ensure_doctors_cache_loaded()
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            doc_id = _as_int(doc.get("id"))
            if doc_id is None:
                continue
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            if _has_explicit_non_samara_regions(raw_regions):
                continue
            if samara_tokens and raw_regions and not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
            by_id[doc_id] = doc

        try:
            doctor_prices = await asyncio.to_thread(api_price.load_doctor_prices)
        except Exception:
            doctor_prices = []

        for row in doctor_prices:
            if not isinstance(row, dict):
                continue
            # Пропускаем «нулевые» doctor_prices — это, как правило,
            # незаполненные админом записи (data drafts). Пациент
            # видел в выдаче «Саушкина (Лор) — 0 руб», «Джовмардов
            # (Лор) — 0 руб» — заведомо неверная цена для приёма
            # ЛОР-врача (розница 2 000 руб). Если для специальности
            # есть retail-цена, она и так показана сверху ответа,
            # а конкретный врач без заполненной цены пусть всплывает
            # через doctors_info, но не клеймится «0 руб».
            row_cost = _as_int(row.get("cost")) or 0
            if row_cost <= 0:
                continue
            doctor_id = _as_int(row.get("doctorId"))
            if doctor_id is None or doctor_id not in by_id:
                continue
            row_name_norm = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
            row_homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
            score, matched = _price_row_score(
                row,
                query=query_norm,
                tokens=query_tokens,
                homecode_query=homecode_query,
            )
            if not is_consult_query and target_homecode and row_homecode and target_homecode == row_homecode:
                score = max(score, 260)
                matched = max(matched, 1)
            if score <= 0:
                continue
            if not is_consult_query and not _is_strong_doctor_price_match(
                query_norm=query_norm,
                query_tokens=query_tokens,
                row_name_norm=row_name_norm,
                matched_tokens=matched,
                target_homecode=target_homecode,
                row_homecode=row_homecode,
            ):
                continue
            item = (score, matched, -row_cost, doctor_id, row)
            matched_price_rows.append(item)
            if target_homecode and row_homecode and target_homecode == row_homecode:
                exact_link_rows.append(item)

        has_reliable_doctor_link = bool(exact_link_rows) or _has_reliable_doctor_service_link(
            matched_price_rows,
            query_norm,
        )
        service_kind = _classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=has_reliable_doctor_link,
            is_consult_query=is_consult_query,
        )
    if service_kind == "ambiguous":
        service_kind = await _resolve_ambiguous_price_kind_with_llm(
            query_text,
            out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=bool(exact_link_rows) or _has_reliable_doctor_service_link(
                matched_price_rows,
                query_norm,
            ),
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
        )
    if service_kind == "operator":
        return _service_fallback(
            note="service_bundle_info ambiguous operator fallback",
            handoff_message=handoff_message("ambiguous_price_service"),
            entities=entities,
            reason="ambiguous_price_service",
            extra={
                "retail_prices": out.get("retail_prices") or [],
                "service_name": service_name,
            },
        )
    if service_kind == "family_query":
        family_payload = _build_price_family_payload(
            search_text,
            retail_rows,
            show_all=False,
            visible_limit=10,
        )
        if family_payload:
            family_payload["entities_used"] = entities
            family_payload["top_n_applied"] = top_limit
            return family_payload
    out["service_kind"] = service_kind

    if service_kind in {"doctor_consult", "procedure_with_doctor"}:
        candidate_rows = (
            exact_link_rows
            if (service_kind == "procedure_with_doctor" or out.get("llm_selected")) and exact_link_rows
            else matched_price_rows
        )
        allow_soft_substring_fallback = service_kind == "doctor_consult" and len(query_tokens) <= 1
        if not candidate_rows and query_norm and allow_soft_substring_fallback:
            for row in doctor_prices:
                if not isinstance(row, dict):
                    continue
                doctor_id = _as_int(row.get("doctorId"))
                if doctor_id is None or doctor_id not in by_id:
                    continue
                service_row_name = _normalise_input(str(row.get("serviceName") or ""))
                if query_norm and query_norm in service_row_name:
                    cost = _as_int(row.get("cost")) or 0
                    candidate_rows.append((1, 1, -cost, doctor_id, row))

        query_specialty = _extract_specialty_from_text(query_text) or _extract_specialty_from_text(service_name)
        if service_kind == "doctor_consult" and not query_specialty:
            # Специальность не распознана («рефлексотерапевта», «репродуктолога») —
            # фильтра по специальности ниже не будет, и в список шли любые врачи с
            # «приёмом» в прайсе: на рефлексотерапевта бот называл колопроктолога и
            # лимфолога с их ценами. Верим только точной связи из МИС — врач оказывает
            # ИМЕННО эту услугу (тот же код, что у розничной строки). Нет связи — нет
            # списка (свип 30.09, tests/test_price_consult_doctor_links.py).
            candidate_rows = []
            for row in doctor_prices if target_homecode else []:
                if not isinstance(row, dict):
                    continue
                row_code = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
                linked_doctor_id = _as_int(row.get("doctorId"))
                row_cost = _as_int(row.get("cost")) or 0
                if row_code == target_homecode and row_cost > 0 and linked_doctor_id in by_id:
                    candidate_rows.append((260, 1, -row_cost, linked_doctor_id, row))

        candidate_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        best_row_by_doctor: dict[int, dict[str, Any]] = {}
        for _, _, _, doctor_id, row in candidate_rows:
            if doctor_id not in best_row_by_doctor:
                best_row_by_doctor[doctor_id] = row

        # Врачи, у которых в МИС есть ровно эта услуга (тот же код, что у розничной
        # строки). Одного кода мало: МИС отдаёт «грязные» строки (приём кардиолога у
        # ревматолога — test_service_bundle_info_consult_filters_doctors_by_primary_specialty).
        exact_code_doctor_ids: set[int] = set()
        for row in doctor_prices if target_homecode else []:
            if not isinstance(row, dict) or (_as_int(row.get("cost")) or 0) <= 0:
                continue
            row_code = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
            linked_doctor_id = _as_int(row.get("doctorId"))
            if row_code == target_homecode and linked_doctor_id is not None:
                exact_code_doctor_ids.add(linked_doctor_id)

        def _eligible(doctor_id: int) -> bool:
            if doctor_id not in by_id:
                return False
            if service_kind != "doctor_consult" or not query_specialty:
                return True
            # Строку выбрала LLM, и у врача в МИС ровно эта услуга по коду: разбор
            # специальности по тексту ему не судья — «пластического хирурга» он сводит
            # к «хирургу» и отдал бы приём общего хирурга (свип 01.10).
            if out.get("llm_selected") and doctor_id in exact_code_doctor_ids:
                return True
            doc = by_id[doctor_id]
            if _doctor_matches_primary_specialty(doc, query_specialty):
                return True
            # Не основное подразделение, но ровно эта специальность И ровно эта услуга
            # по коду МИС: инфекционист с основным подразделением «гепатолог» (свип 30.09).
            return doctor_id in exact_code_doctor_ids and any(
                _is_direct_specialty_text_match(unit, query_specialty)
                for unit in _collect_role_unit_names(doc, main_value=False)
            )

        # Специальность — ДО обрезки до top_limit: раньше обрезали по порядку сортировки
        # среди всех врачей со словом «приём», и нужного специалиста вытесняли
        # терапевты — список выходил пустым (свип 30.09, «приём инфекциониста»).
        doctor_cards = sorted(
            [by_id[doctor_id] for doctor_id in best_row_by_doctor if _eligible(doctor_id)],
            key=_doctor_sort_key,
        )[:top_limit]

        # Параллелизируем availability_snapshot для всех кандидатов.
        # Раньше последовательный цикл `for doc ...: await snapshot(...)`
        # подвешивал PRICE-консультацию специалиста на 240+ секунд при
        # медленном Nayka /doctorSchedule (один surname Трубин у нас
        # отдавался 224с). Теперь все 4-8 кандидатов едут параллельно,
        # верхняя граница на availability-фазу ≈ 30с (per-doctor timeout
        # внутри `_schedule_by_specialty` уже 60с, для чисто
        # availability-проверки ставим 25с — мы не показываем дни/слоты,
        # а только bool «есть/нет окон»).
        # Жалоба заказчика 2026-05-05: «сколько стоит приём кардиолога»
        # уходил в 4-минутный таймаут (eval P3 / CRIT_PRICE_CONSULT_UROLOGIST_001).
        _AVAILABILITY_TIMEOUT_S = 25.0
        _AVAILABILITY_CONCURRENCY = 8
        _availability_sem = asyncio.Semaphore(_AVAILABILITY_CONCURRENCY)
        _empty_availability = {
            "available": False,
            "nearest_slot": "",
            "regions_with_slots": [],
            "note": "availability_timeout",
        }

        async def _fetch_doctor_availability(doc: dict[str, Any]) -> dict[str, Any] | None:
            """Возвращает payload для одного врача либо None, если врача надо
            пропустить (specialty mismatch / no doctor_id)."""
            doctor_id = _as_int(doc.get("id"))
            if doctor_id is None:
                return None
            async with _availability_sem:
                try:
                    availability = await asyncio.wait_for(
                        self._doctor_availability_snapshot(
                            str(doc.get("fio") or ""),
                            samara_tokens=samara_tokens,
                        ),
                        timeout=_AVAILABILITY_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    availability = dict(_empty_availability)
                except Exception:
                    availability = dict(_empty_availability, note="availability_error")
            price_row = best_row_by_doctor.get(doctor_id, {})
            return {
                "id": doctor_id,
                "fio": str(doc.get("fio") or "").strip(),
                "ord": _as_int(doc.get("ord")),
                "specialization": _compact_specialization(
                    _pick_display_specialization(
                        doc,
                        preferred_specialty=query_specialty,
                        preferred_service=service_name,
                    )
                ),
                "specialty_label": _specialty_label_for_doctor(
                    doc,
                    preferred_specialty=query_specialty,
                ),
                "regions": [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()],
                "service_price": _as_int(price_row.get("cost")),
                "available": bool(availability.get("available")),
                "nearest_slot": str(availability.get("nearest_slot") or ""),
                "regions_with_slots": list(availability.get("regions_with_slots") or []),
                "availability_note": str(availability.get("note") or ""),
            }

        availability_results = await asyncio.gather(
            *(_fetch_doctor_availability(doc) for doc in doctor_cards)
        )
        out_doctors = [r for r in availability_results if r is not None]
        out["doctors"] = out_doctors
    else:
        out["doctors"] = []
        out["note"] = (
            f"{out['note']}; " if str(out.get("note") or "").strip() else ""
        ) + f"service_bundle_info: {service_kind or 'no_doctors'}"

    # 3) Preparation guidance by service/test name.
    # В PRICE показываем подготовку только по явному запросу пациента.
    # Иначе блок шумит и мешает основной задаче (цена/врач/расписание).
    show_prepare = _is_prepare_requested_in_price_query(query_text)
    out["show_prepare"] = show_prepare
    if show_prepare and not is_consult_query:
        prepare_payload = await self.test_prepare(service_name, {"service_name": service_name})
        if isinstance(prepare_payload, dict) and not prepare_payload.get("handoff_required"):
            out["prepare"] = str(prepare_payload.get("prepare") or "").strip()

    return out
