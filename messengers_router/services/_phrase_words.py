"""Слова реплики и сильные окна по данным МИС — общие части разбора, которые читают живые модули.

Вынесены из генератора окон `_turn_phrases` (К1 карты мёртвого кода, ответ «Рефакторинг Next» 29.10: снимать генератор, а не
файл): токены реплики (`_tokens`), обвязка с краёв фразы (`_WRAPPER`) и промежутков (`_FILLER`), фраза без обвязки (`bare`),
сильные строки фразы (`phrase_rows`) и сильное окно внутри фразы (`strong_window`, `strong_inside`). Их берут разметка слов
`_turn_spans`, подготовка, корзина, цена и понимание хода. Новых регулярок и словарей нет: токены — `_PRICE_TOKEN_RE`, обвязка —
`_PRICE_QUERY_STOPWORDS`, `_MULTI_PRICE_NOISE_WORDS`, `_PRICE_GENERIC_SERVICE_TOKENS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..turn_parse import per_turn
from ._common import _normalise_catalog_text
from ._prices_helpers import (
    _MULTI_PRICE_NOISE_WORDS,
    _PRICE_GENERIC_SERVICE_TOKENS,
    _PRICE_QUERY_STOPWORDS,
    _PRICE_TOKEN_RE,
    _catalog_fingerprint,
    _is_multi_price_noise,
    query_names_service_in_other_words,
)
from ._strong_match import strong_rows

_WINDOWS = (3, 2, 1)

# Обвязка с краёв фразы: слова вопроса о цене и вежливость. Внутри промежутка между сильными окнами
# обвязка ещё и родовые слова («анализ на») — сами по себе услугу они не называют.
_WRAPPER = frozenset(_PRICE_QUERY_STOPWORDS) | _MULTI_PRICE_NOISE_WORDS
_FILLER = _WRAPPER | frozenset(_PRICE_GENERIC_SERVICE_TOKENS)

@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    key: str



def _key(word: str) -> str:
    return word.lower().replace("ё", "е")


def _tokens(text: str) -> list[_Token]:
    return [_Token(m.start(), m.end(), _key(m.group(0))) for m in _PRICE_TOKEN_RE.finditer(str(text or ""))]


def bare(text: str) -> str:
    """Фраза без обвязки с краёв, словами без регистра — сравнивать разрезы (согласие с правилами)."""

    keys = [t.key for t in _tokens(text)]
    while keys and keys[0] in _WRAPPER:
        keys.pop(0)
    while keys and keys[-1] in _WRAPPER:
        keys.pop()
    return " ".join(keys)


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


def strong_window(fragment: str, catalog: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str] | None:
    """Одно сильное по данным МИС окно (1–3 слова, длинные первыми, без шум-слова внутри) внутри фразы и слова фразы вокруг
    него без обвязки: «нужна ли подготовка к экг» → («экг», строки, «нужна ли подготовка к»). Окон нет, вокруг нет слов или
    в фразе другое сильное окно вне выбранного (склейка двух названий) — None. Механика окон разбора без новых словарей;
    что за слова вокруг — решает вызывающий (`strong_inside` для корзины, `prepare._catalog_subject` для подготовки)."""

    toks = _tokens(fragment)
    hits: list[tuple[int, int, str, list[dict[str, Any]]]] = []  # окна всех размеров, длинные первыми
    for size in _WINDOWS:
        for i in range(len(toks) - size + 1):
            window = toks[i : i + size]
            if any(_is_multi_price_noise(t.key) for t in window):
                continue
            raw = fragment[window[0].start : window[-1].end]
            rows = phrase_rows(raw, catalog)
            if rows:
                hits.append((i, i + size, raw, rows))
    if not hits:
        return None
    lo, hi, raw, rows = hits[0]
    name = _normalise_catalog_text(raw)
    if any(_normalise_catalog_text(hit[2]) != name and not (lo <= hit[0] and hit[1] <= hi) for hit in hits[1:]):
        return None  # два разных названия в одной фразе — не условие при услуге, а склейка: как было
    rest = " ".join(fragment[t.start : t.end] for k, t in enumerate(toks) if not lo <= k < hi and t.key not in _FILLER)
    return (raw, rows, rest) if rest else None


def strong_inside(fragment: str, catalog: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str] | None:
    """Окно `strong_window` внутри слабой фразы корзины, если слова вокруг могут быть условием при услуге: «ферритин первый
    раз» → («ферритин», строки, «первый раз»); что это за слова (условие или разговор), решает LLM корзины.

    Слова вокруг, которые сами называют услугу, — не условие, а склейка перечня без запятой (дополнение 20.10, карточка
    корзин dev, r0121): другое сильное окно любого размера вне выбранного («бактериоскопия мазка цитология …», отсекает
    `strong_window`) или хоть одно слово, известное прайсу («тимофеевка луговая ферритин …»,
    `query_names_service_in_other_words` — словарь гарда П2). Тогда None — как было: суд одного окна молча выкинул бы
    вторую услугу из ответа. Проверка — только здесь: подготовке нужно само окно («подготовка к экг» — слова прайса вокруг
    услуги, слияние деплоя 11)."""

    found = strong_window(fragment, catalog)
    if found is None or not query_names_service_in_other_words(found[2], catalog):
        return None  # вокруг нет слов или среди них слово прайса — суда нет, как было
    return found
