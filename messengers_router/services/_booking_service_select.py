"""Услугу-цель записи выбирает LLM, правила только собирают кандидатов — как у цены (решение владельца 30.09).

BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS: в записи услугу решал лексический резолвер прайса, и его догадка уходила
дальше как «exact» без суда смысла — «запишите на гастроскопию» → «Есть возможность записи на Гастрин…» (анализ, 750
руб.), «запишите на массаж» → «Массаж век». У цены этот класс закрыт выбором строки (`_price_select_llm`); путь записи
его не вызывал. Здесь — тот же выбор (`select_price_rows`, модуль `choose()`, промпт `choose_price_service`, тот же
выключатель `llm_price_service_select`): вопрос — фраза услуги из реплики (слоты записи — дата, филиал — не условие
услуги), кандидат — догадка правил.

Исход (`BookingServiceVerdict.action`):
- `keep` — догадка правил остаётся: её одну подтвердили данные МИС (сильное совпадение) или LLM; выбор не звали
  (выключатель, strict, в реплике нет слов об услуге) — как было;
- `replace` — LLM (или данные МИС) назвали другую одну услугу — цель она («гастроскопия» → ФГДС вторым кругом);
- `choose` — несколько услуг (общее название группы — «массаж») или LLM недоступна на слабом совпадении: названия на
  выбор пациенту, без записи
  (решение владельца 18.10 — как L-13 у цены: дезинформация услугой хуже уточнения);
  (одно название — подтверждение «Похоже, вы имели в виду …?»; названий целым словом нет — сама догадка правил на
  подтверждение);
- `clarify` — LLM: названо слишком общее («массаж», «биопсию») — пациента просят назвать услугу, примеры — названия
  целым словом (вариант B), если есть; «в реплике нет услуги» (`not_a_service`) — не уточнение, а `refuse`: услуга
  снимается, цель — специальность или врач, если есть, иначе честный отказ гарда B′;
- `refuse` — LLM не нашла услугу и во втором круге: цель не подтверждена — честный отказ гарда B′ с оффером оператора.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ._common import _normalise_catalog_text
from ._price_select_llm import select_price_rows

# Сколько названий показать пациенту на выбор — как вариант B у цены.
_CHOICE_LIMIT = 5
# Код причины пустого выбора «названа группа услуг» (`PRICE_TASK.reasons`): уточнить с примерами.
_TOO_GENERAL = "too_general"
# «В реплике нет названия услуги» — услуга снимается без примеров: примеры варианта B по словам «время», «день» — бессмыслица
# («На какое время запись к неврологу» → «например: Запись фотоизображения на флеш-карту», карточка test 18.10, g0792);
# цель записи — специальность или врач, если они есть, иначе честный отказ гарда B′.
_NOT_A_SERVICE = "not_a_service"


@dataclass(frozen=True)
class BookingServiceVerdict:
    action: str  # keep | replace | choose | clarify | refuse
    name: str = ""
    names: tuple[str, ...] = ()
    status: str = ""  # статус выбора строки (`PriceSelection.status`) — для флага хода


def _name(row: dict[str, Any]) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip()


def _distinct_names(rows: Any) -> tuple[str, ...]:
    out: list[str] = []
    for row in rows or ():
        name = _name(row) if isinstance(row, dict) else str(row or "").strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _from_names(names: tuple[str, ...], canonical: str, status: str) -> BookingServiceVerdict:
    """Запись — на одну услугу: одно название — цель; несколько (общее название группы, «массаж») — выбор пациента,
    даже если догадка правил среди них: молча выбрать одну из группы — та же подмена."""

    if not names:
        return BookingServiceVerdict("keep", canonical, status=status)
    if len(names) > 1:
        return BookingServiceVerdict("choose", names=names[:_CHOICE_LIMIT], status=status)
    if _normalise_catalog_text(names[0]) == _normalise_catalog_text(canonical):
        return BookingServiceVerdict("keep", canonical, status=status)
    return BookingServiceVerdict("replace", names[0], status=status)


async def judge_booking_service(
    phrase: str,
    canonical: str,
    *,
    catalog: Callable[[], Awaitable[list[dict[str, Any]]]],
    reply: str = "",
    runtime_llm_mode: str = "",
) -> BookingServiceVerdict:
    """Та ли услуга в догадке правил `canonical`, которую назвал пациент фразой `phrase` (реплика целиком — `reply`).

    `catalog` — строки каталога услуг сервисного слоя (`Services._ensure_service_catalog_rows_loaded`: прайс и цены
    врачей) — те же, по которым правила нашли догадку.
    """

    phrase = str(phrase or "").strip()
    canonical = str(canonical or "").strip()
    keep = BookingServiceVerdict("keep", canonical)
    if not phrase or not canonical:
        return keep
    # Дословное название строки прайса в реплике — сильное совпадение по данным (`_strong_match`): выбор не нужен.
    # Фраза услуги теряет скобку («Эзофагогастродуоденоскопия (ФГДС)» → без «(ФГДС)»), поэтому сверка — по реплике.
    if _normalise_catalog_text(canonical) in _normalise_catalog_text(reply or phrase):
        return BookingServiceVerdict("keep", canonical, status="strong")
    try:
        rows = [row for row in await catalog() if isinstance(row, dict)]
    except Exception:
        return keep
    if not rows:
        return keep
    target = _normalise_catalog_text(canonical)
    candidates = [row for row in rows if _normalise_catalog_text(_name(row)) == target]
    selection = await select_price_rows(
        phrase,
        rows,
        candidates=candidates,
        # Пациент увидит одно название — догадку правил; строки с тем же названием (адреса, варианты цены) — не список.
        shown=candidates[:1],
        hints=(canonical,),
        runtime_llm_mode=runtime_llm_mode,
        general_as_none_fit=True,
        price_answer=False,  # суд записи — прежний промпт: правила пути цены (`price_select_rules`) его не касаются
    )
    if selection is None:
        return keep
    if selection.status in {"strong", "picked"}:
        return _from_names(_distinct_names(selection.rows), canonical, selection.status)
    if selection.status == "unavailable":
        # Названий целым словом нет — на выбор сама догадка правил («Похоже, вы имели в виду …?»): решает пациент.
        names = _distinct_names(selection.names) or (canonical,)
        return BookingServiceVerdict("choose", names=names[:_CHOICE_LIMIT], status=selection.status)
    if selection.status == "none_fit" and selection.reason == _TOO_GENERAL:
        # «Массаж», «биопсию» — названа группа услуг: какую именно, решает пациент (примеры — названия варианта B).
        return BookingServiceVerdict("clarify", names=_distinct_names(selection.names)[:_CHOICE_LIMIT], status=_TOO_GENERAL)
    if selection.status == "none_fit" and selection.reason == _NOT_A_SERVICE:
        return BookingServiceVerdict("refuse", status=_NOT_A_SERVICE)
    if selection.status == "none_fit":
        return BookingServiceVerdict("refuse", status=selection.status)
    return keep
