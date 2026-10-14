"""Срок готовности: при сильном совпадении — кодом из МИС, при слабом услугу выбирает LLM, срок печатает код.

Вопрос о сроке («сколько готовится хгч», «срок готовности липидограммы») при сильном совпадении по
данным получает срок из поля `deadline` справочника МИС кодом (`readiness_from_catalog`, 4df0745).
Слабое совпадение (буквы, падеж, часть названия) раньше шло в честный перевод — 14 переводов о сроке
в карточке «понимание хода on» (11.10). Его решает выбор модели, как строку прайса
(`readiness_by_choice`):

- кандидаты — строки, которые находит поиск правил по словам вопроса (без фразы о сроке);
- модель видит только названия (не сроки и не цены) и отмечает номера той услуги;
- выбранные названия — сроки из справочника кодом (принцип 1), не больше трёх; срока нет или у
  строк одного названия сроки разные — это название не показывается;
- ничего не выбрала, сбой, список услуг — None: честный перевод, как раньше.

Выключатель `[MESSENGER_ROUTER] choose_readiness = off | on` (по умолчанию off; действует под
пониманием хода `on`, где вопрос о сроке распознаётся).

Что считать сильным, задаёт `[MESSENGER_ROUTER] strong_match` (L-15b, шаг 3): `legacy` (по умолчанию) —
прежнее определение срока (название целиком или до скобок, словарь кода `_PRICE_SERVICE_ALIASES`, синоним
МИС, код) по услуге, которую угадал резолвер; `data` — одно определение с ценой и корзиной, только
данные МИС (`_strong_match.strong_rows`): «ттг» и «оак» по словарю кода — уже выбор LLM.

Под `turn_parse = on` (L-15b, шаг 5) услугу вопроса называет разбор хода (`turn_parse.acting_for`): фраза
одна и сильная по данным МИС — срок её строк кодом, без LLM; одна слабая — её выбирает LLM по кандидатам
этой фразы; несколько фраз (список) или ни одной — честный перевод, как у списка раньше. Разбора нет или
он не решён — как раньше. Слабая фраза и решение LLM по ней (выбранные услуги или «не нашла») — в журнал
синонимов для клиники (`source: readiness`).
"""

from __future__ import annotations

from typing import Any

from .. import turn_parse
from ..choice import Option, Status, Task, choose
from ..runtime_config import config as _cfg
from ._biomaterial import mis_synonym_candidates
from ._common import _normalise_catalog_text
from ._price_select_llm import _mention_phrase, _record_synonym_suggestion, journal_phrase_mode
from ._prices_helpers import (
    _PRICE_SERVICE_ALIASES,
    _apply_service_synonyms,
    _catalog_rows,
    _extract_price_service_from_query,
    _normalise_input,
    _rank_price_rows,
    format_deadline,
    price_list_services,
    resolve_price_service_name_from_catalog,
    strip_readiness_phrasing,
)
from ._strong_match import strong_match_mode, strong_rows

READINESS_TASK = Task(name="readiness_service", max_picks=3, num_predict=64, timeout_s=12, queue_timeout_ms=3000)
_CANDIDATES = 10


def readiness_mode() -> str:
    raw = str(getattr(_cfg, "MR_CHOOSE_READINESS", "off") or "").strip().lower()
    return raw if raw in ("off", "on") else "off"


def _one_deadline(rows: list[dict[str, Any]]) -> str | None:
    """Срок строк одной услуги: один и есть — он; разные или пустой — None (честный перевод)."""

    formatted = {format_deadline(str(row.get("deadline") or "").strip()) for row in rows}
    return formatted.pop() if len(formatted) == 1 and None not in formatted else None


def _deadline(name: str, catalog: list[dict[str, Any]]) -> str | None:
    key = _normalise_input(name)
    return _one_deadline([row for row in catalog if _normalise_input(str(row.get("serviceName") or "")) == key])


def _legacy_strong_service_match(fragment: str, service_name: str, retail_rows: list[dict[str, Any]]) -> bool:
    """Прежнее определение сильного у срока (`strong_match = legacy`, 4df0745): фраза — название услуги
    целиком или до скобок, курируемая аббревиатура, синоним МИС или код. Уходит, когда `data` принят."""

    key = _normalise_input(_extract_price_service_from_query(fragment) or fragment)
    name = _normalise_input(service_name)
    if key in (name, _normalise_input(service_name.split("(")[0].split("[")[0])):
        return True
    if key in _PRICE_SERVICE_ALIASES or service_name in mis_synonym_candidates(fragment):
        return True
    return key.isdigit() and any(
        str(row.get("serviceHomecode") or "").strip() == key and _normalise_input(str(row.get("serviceName") or "")) == name
        for row in retail_rows
    )


def _single_phrase(parse: turn_parse.PhraseParse) -> turn_parse.ParsedPhrase | None:
    """Фраза вопроса о сроке по разбору хода: одна — она; список или ни одной услуги — None (перевод)."""

    return parse.items[0] if len(parse.items) == 1 else None


def _deadline_of_parsed(parse: turn_parse.PhraseParse) -> list[dict[str, str]] | None:
    """Срок по разбору хода: одна фраза, сильная по данным МИС, и у неё одна услуга — срок её строк кодом.
    Слабая фраза — None (её услугу выберет LLM, `readiness_by_choice`), список и синоним на несколько
    услуг — тоже None."""

    phrase = _single_phrase(parse)
    if phrase is None or not phrase.rows:
        return None
    names = list(dict.fromkeys(str(row.get("serviceName") or "").strip() for row in phrase.rows))
    if len({_normalise_catalog_text(name) for name in names}) != 1:
        return None
    deadline = _one_deadline(list(phrase.rows))
    return [{"name": names[0], "deadline": deadline}] if deadline else None


def readiness_from_catalog(question: str, *, rows: list[dict[str, Any]] | None = None) -> list[dict[str, str]] | None:
    """Сроки готовности услуги, названной в вопросе о сроке, — из справочника МИС (поле `deadline`).

    Только одна услуга и только при сильном совпадении (определение — `strong_match`). None — список
    или слабое совпадение (их выбирает `choose()`, ход (в)), у услуги нет срока или у её строк сроки
    разные: тогда честный перевод, а не догадка.

    :param question: реплика пациента с вопросом о сроке
    :param rows: строки прайса (по умолчанию — срез региона)
    :return: [{"name", "deadline"}] или None
    """

    catalog = _catalog_rows(rows)
    parse = turn_parse.acting_for(question)
    if parse is not None:
        return _deadline_of_parsed(parse)
    core, _ = strip_readiness_phrasing(question)
    if not core.strip():
        return None
    if price_list_services(core, rows=catalog)[0]:
        return None  # список: строку каждой позиции выбирает choose() (ход (в)), до него — перевод
    if strong_match_mode() == "data":
        strong = strong_rows(core, catalog)
        names = list(dict.fromkeys(str(row.get("serviceName") or "").strip() for row in strong))
        # Одна услуга по данным: синоним на несколько услуг — не одна, её выбирает LLM.
        if len({_normalise_catalog_text(name) for name in names}) != 1:
            return None
        name, deadline = names[0], _one_deadline(strong)
    else:
        name = resolve_price_service_name_from_catalog(core, rows=catalog)
        if not name or not _legacy_strong_service_match(core, name, catalog):
            return None
        deadline = _deadline(name, catalog)
    return [{"name": name, "deadline": deadline}] if deadline else None


async def readiness_by_choice(question: str, *, rows: list[dict[str, Any]] | None = None) -> list[dict[str, str]] | None:
    """Сроки готовности услуг, которые выбрала модель; None — честный перевод.

    :param question: реплика пациента с вопросом о сроке
    :param rows: строки прайса (по умолчанию — срез региона)
    :return: [{"name", "deadline"}] или None
    """

    catalog = _catalog_rows(rows)
    first: list[str] = []
    weak_phrase = ""  # слабая фраза разбора — в журнал синонимов с выбором LLM (L-15b, шаг 5)
    parse = turn_parse.acting_for(question)
    if parse is not None:
        phrase = _single_phrase(parse)
        if phrase is None:
            return None
        weak_phrase = "" if phrase.rows else phrase.text
        if weak_phrase and journal_phrase_mode() == "mention":  # фраза журнала — упоминание хода (L-14, шаг 2)
            mentioned = _mention_phrase(question)
            weak_phrase = weak_phrase if mentioned is None else mentioned
        first = list(dict.fromkeys(str(row.get("serviceName") or "").strip() for row in phrase.rows))
        if len({_normalise_catalog_text(name) for name in first}) == 1:
            return None  # одна услуга по данным, а срока у неё нет — соседняя строка не годится: перевод
        # Синоним МИС на несколько услуг — выбор только из его услуг; слабая фраза — кандидаты правил по ней.
        core = _apply_service_synonyms(phrase.text)
    else:
        core, _ = strip_readiness_phrasing(question)
        if not core.strip() or price_list_services(core, rows=catalog)[0]:
            return None
    names: list[str] = []
    guess = "" if first else resolve_price_service_name_from_catalog(core, rows=catalog)
    ranked = [] if first else [str(row.get("serviceName") or "") for row in _rank_price_rows(catalog, core, limit=_CANDIDATES)]
    for name in [*first, guess or "", *ranked]:
        name = name.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return None
    choice = await choose(READINESS_TASK, question, [Option(id=name, text=name) for name in names])
    if weak_phrase and choice.status in (Status.PICKED, Status.NONE_FIT):
        picked = [next((row for row in catalog if str(row.get("serviceName") or "").strip() == name), {"serviceName": name}) for name in choice.ids]
        _record_synonym_suggestion(weak_phrase, [], picked, source="readiness")
    if choice.status is not Status.PICKED:
        return None
    found = [{"name": name, "deadline": deadline} for name in choice.ids if (deadline := _deadline(name, catalog))]
    return found or None
