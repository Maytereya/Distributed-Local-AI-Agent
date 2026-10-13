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

import threading
from typing import Any, Sequence

from ..runtime_config import config as _cfg
from ._biomaterial import mis_synonym_candidates
from ._common import _normalise_catalog_text
from ._prices_helpers import _catalog_fingerprint, _extract_price_service_from_query

_MODES = ("legacy", "data")


def strong_match_mode() -> str:
    """Определение сильного у срока: `legacy` (прежнее) | `data` (это); иное — `legacy`."""

    raw = str(getattr(_cfg, "MR_STRONG_MATCH", "legacy") or "").strip().lower()
    return raw if raw in _MODES else "legacy"


def _name(row: dict[str, Any]) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip()


# Индекс среза для `strong_rows` без `pool` (L-15b, шаг 6, O1): нормализованное название → позиции строк и
# название как есть → позиции (для синонимов МИС). Без него каждый вызов нормализовал все 3 671 название
# (16–22 мс; генератор фраз зовёт `strong_rows` до 47 раз за ход). Ключ — сам срез (держим ссылку: `id`
# освобождённого списка мог бы достаться другому) и его отпечаток (срез, дописанный на месте).
_INDEX_LIMIT = 4
_INDEX: list[tuple[Sequence[dict[str, Any]], tuple[int, str, str], dict[str, list[int]], dict[str, list[int]]]] = []
_INDEX_LOCK = threading.Lock()


def _catalog_index(catalog: Sequence[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    rows = catalog if isinstance(catalog, list) else list(catalog)
    stamp = _catalog_fingerprint(rows)
    with _INDEX_LOCK:
        for known, known_stamp, by_norm, by_name in _INDEX:
            if known is catalog and known_stamp == stamp:
                return by_norm, by_name
    by_norm: dict[str, list[int]] = {}
    by_name: dict[str, list[int]] = {}
    for pos, row in enumerate(rows):
        name = _name(row)
        by_norm.setdefault(_normalise_catalog_text(name), []).append(pos)
        by_name.setdefault(name, []).append(pos)
    with _INDEX_LOCK:
        _INDEX[:] = [entry for entry in _INDEX if entry[0] is not catalog][-(_INDEX_LIMIT - 1) :]
        _INDEX.append((catalog, stamp, by_norm, by_name))
    return by_norm, by_name


def strong_rows(
    phrase: str,
    catalog: Sequence[dict[str, Any]],
    *,
    pool: Sequence[dict[str, Any]] | None = None,
    verbatim: bool = False,
) -> list[dict[str, Any]]:
    """Строки прайса, сильно совпавшие с фразой пациента по данным МИС; пусто — совпадение слабое.

    :param phrase: реплика или фраза позиции, как написал пациент (слова вопроса о цене снимаются)
    :param catalog: строки прайса региона — по ним ищется код услуги
    :param pool: где искать название и синоним — кандидаты правил (цена, корзина); None — весь `catalog`
    :param verbatim: дословное название сравнивается и с фразой как написана — у фразы разбора хода
        (L-15b): она уже без обвязки, а вырезатель фразы правил теряет в названии дефис и предлоги
        («Глюкозотолерантный тест (глюкоза+ C-пептид)» → «… c пептид»). Цена и корзина правил зовут
        без него — их поведение не меняется
    :return: строки в порядке `pool` (затем строки кода), без повторов
    """

    key = _normalise_catalog_text(_extract_price_service_from_query(phrase) or phrase)
    names = {key, _normalise_catalog_text(phrase)} if verbatim else {key}
    names.discard("")
    if not names:
        return []
    synonyms = set(mis_synonym_candidates(phrase))
    by_code = [row for row in catalog if key.isdigit() and str(row.get("serviceHomecode") or "").strip() == key]
    out: list[dict[str, Any]] = []
    if pool is None:
        # Те же строки в том же порядке, что и проход по всему срезу: позиции совпадений по индексу.
        by_norm, by_name = _catalog_index(catalog)
        positions = {pos for pos, row in enumerate(catalog) if row in by_code} if by_code else set()
        for name in names:
            positions.update(by_norm.get(name, ()))
        for name in synonyms:
            positions.update(by_name.get(name, ()))
        source = [catalog[pos] for pos in sorted(positions)]
        for row in [*source, *by_code]:
            if row not in out:
                out.append(row)
        return out
    for row in [*pool, *by_code]:
        name = _name(row)
        if (_normalise_catalog_text(name) in names or name in synonyms or row in by_code) and row not in out:
            out.append(row)
    return out
