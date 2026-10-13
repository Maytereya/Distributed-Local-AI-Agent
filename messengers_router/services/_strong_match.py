"""Сильное совпадение по данным МИС — одно определение для цены, корзины, срока и L-15b.

Решение владельца 13.10 (вариант А): сильное — то, что клиника сказала своими данными, и только оно;
такие строки показываются без выбора LLM:
- дословное название строки прайса;
- синоним из справочника МИС (`serviceSynonyms`) — все услуги синонима (`mis_synonym_candidates`):
  у неоднозначного синонима их несколько («холестерин» → Холестерол, ЛПНП, ЛПВП, Триглицериды), и
  молча выбирать одну нельзя;
- код услуги (`serviceHomecode`), если пациент написал число.

Не сильное — слабое, его строку выбирает LLM (`choose()`): «название до скобок» (скобка бывает
значимым уточнением — «ОАК (без лейкоцитарной формулы и СОЭ)» на «оак» — урезанный вариант, а не
общий анализ) и курируемый словарь кода `_PRICE_SERVICE_ALIASES` («холестерин» → «Липидограмма»,
хотя клиника этим словом называет четыре анализа). Содержимое словаря — клинике как предложения в
справочник синонимов; сам словарь остаётся генератором кандидатов (резолвер, сегментация списка).

Нормализация — `_normalise_catalog_text` (цена и корзина сравнивали так с L-13): нижний регистр,
ё → е, всё, кроме букв, цифр, дефиса и пробела, — пробел. Пунктуация пациента («ферритин?», название
со скобками и запятыми, набранное без них) не делает дословное название слабым. На живом прайсе
(3 671 строка, 3 542 названия) так совпадают лишь 3 пары названий, и это одна и та же услуга, записанная
с разной пунктуацией («Ударно-волновая терапия (1 зона)» / «…, 1 зона»). Прежнее определение срока
(`_normalise_input`) пунктуацию сравнивало.

Срок переходит на это определение под выключателем `[MESSENGER_ROUTER] strong_match = legacy | data`
(по умолчанию `legacy` — прежнее определение срока с «до скобок» и словарём кода): срок в наборе
деплоя, и смена определения меняет его ответы.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..runtime_config import config as _cfg
from ._biomaterial import mis_synonym_candidates
from ._common import _normalise_catalog_text
from ._prices_helpers import _extract_price_service_from_query

_MODES = ("legacy", "data")


def strong_match_mode() -> str:
    """Определение сильного у срока: `legacy` (прежнее) | `data` (это); иное — `legacy`."""

    raw = str(getattr(_cfg, "MR_STRONG_MATCH", "legacy") or "").strip().lower()
    return raw if raw in _MODES else "legacy"


def _name(row: dict[str, Any]) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip()


def strong_rows(
    phrase: str,
    catalog: Sequence[dict[str, Any]],
    *,
    pool: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Строки прайса, сильно совпавшие с фразой пациента по данным МИС; пусто — совпадение слабое.

    :param phrase: реплика или фраза позиции, как написал пациент (слова вопроса о цене снимаются)
    :param catalog: строки прайса региона — по ним ищется код услуги
    :param pool: где искать название и синоним — кандидаты правил (цена, корзина); None — весь `catalog`
    :return: строки в порядке `pool` (затем строки кода), без повторов
    """

    key = _normalise_catalog_text(_extract_price_service_from_query(phrase) or phrase)
    if not key:
        return []
    synonyms = set(mis_synonym_candidates(phrase))
    by_code = [row for row in catalog if key.isdigit() and str(row.get("serviceHomecode") or "").strip() == key]
    out: list[dict[str, Any]] = []
    for row in [*(catalog if pool is None else pool), *by_code]:
        name = _name(row)
        if (key == _normalise_catalog_text(name) or name in synonyms or row in by_code) and row not in out:
            out.append(row)
    return out
