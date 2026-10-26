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

Цена, этап 2, Ш1 (`price_by_parse = on`, DESIGN-2026-10-25-PRICE-STAGE2): ключ — фраза как есть, без вырезателя правил
(`_extract_price_service_from_query`, 11 списков): что в реплике услуга, решил разбор хода, и читатели передают его фразу;
реплика целиком сильна, только если она сама — название, синоним или код (решение 22.10 (а)). Число — код услуги, только
если это вся фраза и не номер из показанного прошлым ответом (`_price_by_parse.number_is_choice`: коды 1–20 в прайсе все
заняты, и «1» после списка цен было услугой с кодом 1).

Буква семейства (`strong_family_letter = on`, BUG-2026-10-27-STRONG-DROPS-LETTER): поиск синонима выбрасывает предлоги и у
реплики, и у ключа словаря — «с», «в», «а», «и», «у» среди них. Синонимы клиники «гепатит в» (HBsAg) и «гепатит с» (Анти-HCV)
становились одной цепочкой «гепатит», и «гепатит с», «гепатит а», «гепатит» были сильны строке «Гепатит В - HBsAg» — цена
чужого гепатита без выбора LLM. Под `on` синоним сильный, только если буквы семейства в ключе те же, что во фразе; строка,
названная другой буквой того же семейства, не сильная. Буква семейства — одиночная буква сразу после слова, у ствола которого
прайс сам различает строки буквами (`_letter_families`: «Гепатит А / В / С / D / E / G», «Витамин А / C / D / E», «Протеин С /
S»); после прочих слов одиночная буква — предлог («глюкоза в крови»). Цифры, «т3»/«т4», «b12», «IgG» поиск не теряет: токен вне
синонима не обвязка, и синоним не засчитывается; дословное название сравнивается целиком.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Sequence

from ..runtime_config import config as _cfg
from . import _price_by_parse
from ._biomaterial import _TOKEN_RE, _tokens, mis_synonym_candidates
from ._common import _normalise_catalog_text, _normalise_input
from ._prices_helpers import _catalog_fingerprint, _extract_price_service_from_query, _fold_family_letter

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


def family_letter_enabled() -> bool:
    """`[MESSENGER_ROUTER] strong_family_letter = on`: сильное совпадение не теряет букву семейства; мусор — off."""

    return str(getattr(_cfg, "MR_STRONG_FAMILY_LETTER", "off") or "").strip().lower() == "on"


_FAMILIES: list[tuple[Sequence[dict[str, Any]], tuple[int, str, str], dict[str, frozenset[str]]]] = []


@functools.lru_cache(maxsize=16384)
def _letter_marks(name: str) -> tuple[tuple[str, str], ...]:
    """(ствол, буква) различителей в названии строки: одиночная буква после слова от 6 букв — через пробел или тире, и за ней не
    идёт кириллическое слово («Гепатит В - HBsAg», «Вирус гепатита C [кач.]», «Протеин С»; «Глюкоза в моче» — предлог).

    Кэш по названию: срез читатели передают и новым списком (`_catalog_rows`), и разбор всех 3 671 названия стоил бы 47 мс на вызов.
    """

    text = _normalise_input(name)
    spans = list(_TOKEN_RE.finditer(text))
    marks: list[tuple[str, str]] = []
    for pos in range(1, len(spans)):
        word, letter = spans[pos - 1].group(), spans[pos].group()
        if len(letter) != 1 or not letter.isalpha() or len(word) < 6 or not word.isalpha():
            continue
        gap = text[spans[pos - 1].end() : spans[pos].start()]
        if gap.strip() not in ("", "-", "–", "—"):
            continue
        if pos + 1 < len(spans) and not text[spans[pos].end() : spans[pos + 1].start()].strip():
            following = spans[pos + 1].group()[0]
            if "а" <= following <= "я" or following == "ё":
                continue
        marks.append((word[:6], _fold_family_letter(letter)))
    return tuple(marks)


def _letter_families(catalog: Sequence[dict[str, Any]]) -> dict[str, frozenset[str]]:
    """Буквенные семейства прайса: ствол (6 букв слова) → буквы, которыми прайс различает строки этого ствола (≥ 2 разных).

    Из данных, без списка болезней в коде — как `_family_discriminator_context` (22.07); на срезе 08.10 — восемь стволов:
    гепатит, витамин, протеин, группы (стрептококк А / В), класса (Ig), фактор, энтеротоксин и «мутаций в S гене».
    """

    rows = catalog if isinstance(catalog, list) else list(catalog)
    stamp = _catalog_fingerprint(rows)
    with _INDEX_LOCK:
        for known, known_stamp, families in _FAMILIES:
            if known is catalog and known_stamp == stamp:
                return families
    found: dict[str, set[str]] = {}
    for row in rows:
        for stem, letter in _letter_marks(_name(row)):
            found.setdefault(stem, set()).add(letter)
    families = {stem: frozenset(letters) for stem, letters in found.items() if len(letters) >= 2}
    with _INDEX_LOCK:
        _FAMILIES[:] = [entry for entry in _FAMILIES if entry[0] is not catalog][-(_INDEX_LIMIT - 1) :]
        _FAMILIES.append((catalog, stamp, families))
    return families


def family_letters(text: str, families: dict[str, frozenset[str]]) -> tuple[tuple[str, str], ...]:
    """Буквы семейства в тексте: (ствол, буква) — одиночная буква сразу после слова семейства, если прайс различает его строки
    этой буквой («гепатит с», «гепатиту с», «гепатит-с»); кириллица и латиница — одна буква (`_fold_family_letter`).

    :param text: фраза пациента, ключ синонима или название строки
    :param families: семейства прайса (`_letter_families`)
    :return: пары (ствол, буква) по порядку; пусто — буквы семейства нет
    """

    tokens = _tokens(text)
    found: list[tuple[str, str]] = []
    for prev, token in zip(tokens, tokens[1:]):
        if len(token) != 1 or not token.isalpha() or len(prev) < 6 or not prev.isalpha():
            continue
        letter = _fold_family_letter(token)
        if letter in families.get(prev[:6], ()):
            found.append((prev[:6], letter))
    return tuple(found)


def _names_other_letter(name: str, named: tuple[tuple[str, str], ...], families: dict[str, frozenset[str]]) -> bool:
    """Строка названа другой буквой того же семейства, чем фраза («Гепатит В - HBsAg» на «гепатит с»); без буквы — нет."""

    own = family_letters(name, families)
    for stem, letter in named:
        letters = {mine for root, mine in own if root == stem}
        if letters and letter not in letters:
            return True
    return False


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

    if _price_by_parse.enabled():
        key = _normalise_catalog_text(phrase)
        names = {key}
        code = key.isdigit() and not _price_by_parse.number_is_choice(key)
    else:
        key = _normalise_catalog_text(_extract_price_service_from_query(phrase) or phrase)
        names = {key, _normalise_catalog_text(phrase)} if verbatim else {key}
        code = key.isdigit()
    names.discard("")
    if not names:
        return []
    families = _letter_families(catalog) if family_letter_enabled() else {}
    named = family_letters(phrase, families) if families else ()
    if families:
        synonyms = set(mis_synonym_candidates(phrase, accept=lambda synonym: family_letters(synonym, families) == named))
    else:
        synonyms = set(mis_synonym_candidates(phrase))
    by_code = [row for row in catalog if code and str(row.get("serviceHomecode") or "").strip() == key]
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
    else:
        for row in [*pool, *by_code]:
            name = _name(row)
            if (_normalise_catalog_text(name) in names or name in synonyms or row in by_code) and row not in out:
                out.append(row)
    if named:
        out = [row for row in out if not _names_other_letter(_name(row), named, families)]
    return out
