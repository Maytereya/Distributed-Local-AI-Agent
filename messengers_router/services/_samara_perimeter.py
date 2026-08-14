"""Самарский периметр обслуживания: места, которые мы обслуживаем как Самару.

## Зачем модуль существует

Прод 10.08: «Может ли сдать спермограмму в южном городе» → бот выдал дамп
восемнадцати филиалов Самары. Пациент для бота был «без города»: `match_city`
возвращает None для «Придорожный», «Южный город», «в южном городе», «Мехзавод».
Ложного отсечения не было, но и пользы тоже.

Хуже: «пр-т Николаевский» (адрес филиала в Южном городе) `match_city` резолвит
в город **«Николаевский»** из справочника городов → ранний гард роутера считает
это ДРУГИМ городом и уводит пациента к оператору «Сейчас могу помочь только по
Самаре». То есть собственный филиал клиники выглядел как чужой город.

## Данные

Узел `id=3 «Самара»` — 32 филиала; тридцать в черте города, три вне её:

- `пос.Придорожный («Южный город»), пр-т Николаевский, д. 2`
- `п.Мехзавод, квартал 10, д.13`
- `ул.Челышевская, 3 (ЮГ-2)` — тоже Южный город

Словарь мест строится **ИЗ ДАННЫХ** филиалов (поля `name` и `addressForSite`):
населённый пункт после `пос./п./г./с.`, содержимое кавычек и содержимое скобок
(так в словарь попадают «Придорожный», «Южный город», «ЮГ-2», «Мехзавод»).

К ним добавлен закрытый список спутников по решению владельца №5 (14.08):
Новокуйбышевск, Кинель, Смышляевка, Стройкерамика — их обслуживаем как Самару и
к оператору как «другой город» НЕ отправляем.
"""

from __future__ import annotations

import re
from typing import Any

from ..russian_nlu import normalize_ru
from ._samara_branches import fallback_branches, load_snapshot

# Решение владельца №5 (14.08). Список ЗАКРЫТ: новые окрестности — уточнять у
# заказчика, не додумывать.
SATELLITE_PLACES: tuple[str, ...] = (
    "новокуйбышевск",
    "кинель",
    "смышляевка",
    "стройкерамика",
)

# Населённый пункт в адресе филиала: «пос.Придорожный», «п.Мехзавод».
_SETTLEMENT_RE = re.compile(r"\b(?:пос|п|г|с|пгт|дер|д)\.\s*([А-ЯЁA-Z][\w\-]{2,})")
# Содержимое кавычек («Южный город») и скобок ((ЮГ-2)).
_QUOTED_RE = re.compile(r"[«\"']([^»\"']{2,40})[»\"']")
_PARENTHESISED_RE = re.compile(r"\(([^)]{2,40})\)")

# Токены, которые не являются названием места (мусор из скобок вроде «Самара»).
_NOT_A_PLACE = frozenset({"самара", "самары", "самаре"})

_PLACES_CACHE: dict[int, frozenset[str]] = {}


# Падежные/родовые окончания русского языка. Нужны, чтобы «в южном городе»
# сматчилось с «Южный город», а «в Придорожном» — с «Придорожный», и при этом
# «кино» НЕ сматчилось с «Кинель». Список фиксированный и покрывает склонение
# прилагательных и существительных; порядок — от длинных к коротким.
_WORD_ENDINGS: tuple[str, ...] = (
    "ыми", "ими", "ого", "его", "ому", "ему", "ых", "их", "ым", "им", "ой", "ей",
    "ая", "яя", "ую", "юю", "ые", "ие", "ом", "ем", "ый", "ий", "ах", "ям", "ам",
    "ов", "ев", "ь", "а", "я", "у", "ю", "е", "и", "ы", "о",
)
# Двухбуквенное окончание — надёжный признак склонения («южн-ом», «южн-ый»),
# основу можно оставить короткой. Однобуквенное («кин-о») слишком часто
# оказывается частью корня, поэтому там требуем основу подлиннее — иначе
# «кино» ложно сматчилось бы с «Кинель».
_MIN_STEM_LONG_ENDING = 3
_MIN_STEM_SHORT_ENDING = 4


def _word_stem(word: str) -> str:
    """Основа слова: срезаем одно падежное окончание.

    «южном»→«южн», «южный»→«южн», «придорожном»→«придорожн»,
    «кинель»/«кинеле»→«кинел», «кино»→«кино» (срез дал бы ложный матч).
    """

    low = word.lower()
    for ending in _WORD_ENDINGS:
        if len(low) <= len(ending) or not low.endswith(ending):
            continue
        trimmed = low[: -len(ending)]
        floor = _MIN_STEM_LONG_ENDING if len(ending) >= 2 else _MIN_STEM_SHORT_ENDING
        return trimmed if len(trimmed) >= floor else low
    return low


def _norm(text: str) -> str:
    cleaned = re.sub(r"[«»\"'`]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", normalize_ru(cleaned)).strip()


def _places_from_branch(row: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for field in ("addressForSite", "name"):
        raw = str(row.get(field) or "")
        if not raw:
            continue
        for match in _SETTLEMENT_RE.finditer(raw):
            out.add(_norm(match.group(1)))
        for pattern in (_QUOTED_RE, _PARENTHESISED_RE):
            for match in pattern.finditer(raw):
                out.add(_norm(match.group(1)))
    return {p for p in out if p and p not in _NOT_A_PLACE and len(p) >= 3}


def perimeter_places(branches: list[dict[str, Any]] | None = None) -> frozenset[str]:
    """Места самарского периметра: из данных филиалов + спутники по решению №5.

    :param branches: филиалы узла Самары (если не переданы — берём снапшот)
    :return: нормализованные названия мест
    """

    rows = branches
    if rows is None:
        rows = load_snapshot() or fallback_branches()
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    key = len(rows)
    cached = _PLACES_CACHE.get(key)
    if cached is not None and branches is None:
        return cached

    places: set[str] = set(SATELLITE_PLACES)
    for row in rows:
        places |= _places_from_branch(row)
    frozen = frozenset(places)
    if branches is None:
        _PLACES_CACHE.clear()
        _PLACES_CACHE[key] = frozen
    return frozen


def match_perimeter_place(
    text: str, branches: list[dict[str, Any]] | None = None
) -> str | None:
    """Находит в реплике место самарского периметра.

    Сопоставление по ОСНОВАМ слов (одно падежное окончание срезается): так
    «в южном городе» матчит «Южный город», «в Придорожном» — «Придорожный», а
    «кино» НЕ матчит «Кинель».

    :param text: реплика пациента
    :param branches: филиалы (для тестов)
    :return: каноническое (нормализованное) название места либо None
    """

    norm = _norm(text)
    if not norm:
        return None
    tokens = re.findall(r"[a-zа-яё0-9\-]+", norm)
    if not tokens:
        return None
    stems = [_word_stem(t) for t in tokens]
    best: str | None = None
    for place in perimeter_places(branches):
        place_words = re.findall(r"[a-zа-яё0-9\-]+", place)
        if not place_words:
            continue
        place_stems = [_word_stem(w) for w in place_words]
        span = len(place_stems)
        for i in range(len(stems) - span + 1):
            if stems[i : i + span] == place_stems:
                if best is None or len(place) > len(best):
                    best = place
                break
    return best


def is_samara_perimeter_text(text: str, branches: list[dict[str, Any]] | None = None) -> bool:
    """True, если реплика называет место, которое мы обслуживаем как Самару."""

    return match_perimeter_place(text, branches) is not None


_ADDRESS_TOKENS_CACHE: dict[int, frozenset[str]] = {}


def _branch_address_stems(branches: list[dict[str, Any]] | None = None) -> frozenset[str]:
    """Основы всех слов из адресов и названий наших филиалов."""

    rows = branches
    if rows is None:
        rows = load_snapshot() or fallback_branches()
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    key = len(rows)
    cached = _ADDRESS_TOKENS_CACHE.get(key)
    if cached is not None and branches is None:
        return cached
    stems: set[str] = set()
    for row in rows:
        for field in ("addressForSite", "name"):
            for word in re.findall(r"[a-zа-яё0-9\-]+", _norm(str(row.get(field) or ""))):
                if len(word) >= 4:
                    stems.add(_word_stem(word))
    frozen = frozenset(stems)
    if branches is None:
        _ADDRESS_TOKENS_CACHE.clear()
        _ADDRESS_TOKENS_CACHE[key] = frozen
    return frozen


def is_branch_address_word(value: str, branches: list[dict[str, Any]] | None = None) -> bool:
    """True, если «город» из `match_city` — на самом деле слово из адреса филиала.

    Справочник городов содержит «Николаевский» и «Спутник», а у клиники есть
    филиал на **пр-те Николаевском** в Южном городе. Без этой проверки адрес
    собственного филиала читался как ДРУГОЙ город и пациента уводило к
    оператору. Проверка catalog-derived и не зависит от того, какой срез
    филиалов загружен (живой снапшот или seed): слово «Николаевский» есть в
    `addressForSite` в обоих.

    :param value: значение, которое вернул `match_city`
    :param branches: филиалы (для тестов)
    :return: True, если это слово из наших же адресов
    """

    norm = _norm(value)
    if not norm:
        return False
    words = [w for w in re.findall(r"[a-zа-яё0-9\-]+", norm) if len(w) >= 4]
    if not words:
        return False
    stems = _branch_address_stems(branches)
    return all(_word_stem(w) in stems for w in words)


def branch_name_for_place(
    place: str, branches: list[dict[str, Any]] | None = None
) -> str | None:
    """Название филиала, стоящего в этом месте периметра.

    Нужно, чтобы на «в южном городе» отвечать КОНКРЕТНЫМ филиалом, а не дампом
    восемнадцати адресов Самары. Спутники (Новокуйбышевск, Кинель…) своих
    филиалов не имеют — для них возвращаем None, обслуживаем как Самару в целом.

    :param place: нормализованное место (из `match_perimeter_place`)
    :param branches: филиалы (для тестов)
    :return: значение поля `name` филиала либо None
    """

    target = _norm(place)
    if not target:
        return None
    rows = branches
    if rows is None:
        rows = load_snapshot() or fallback_branches()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if target in _places_from_branch(row):
            name = str(row.get("name") or "").strip()
            if name:
                return name
    return None
