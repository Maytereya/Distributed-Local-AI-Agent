"""Класс `filler_lists_disagree`: одно понятие, разложенное по трём спискам,
которые разошлись.

## Что произошло

«Речевая обвязка» закодирована в проекте трижды и ведётся порознь:

* `_PRICE_QUERY_STOPWORDS` — токенайзер прайса;
* `_MULTI_PRICE_NOISE_WORDS` — разбор корзины;
* `_SPEECH_WRAPPER` — слой синонимов МИС.

Из 29 слов, которые два последних списка уже считали обвязкой, **27 ломали
резолвер**, потому что первый о них не знал:

    «ферритин»                → Ферритин ✓
    «прайс ферритин»          → None
    «записаться на ферритин»  → None
    «посчитайте ферритин»     → None

Механизм не в гарде: лишний токен раздувает `token_count`, а
`_is_price_match_strong` при двух токенах требует, чтобы совпали ОБА. Строка
просто не доживала до кандидатов.

## Почему нельзя было слить списки механически

`_SPEECH_WRAPPER` намеренно лингвистический — в нём есть «анализ», «анализы»,
«тест». Эти слова стоят в НАЗВАНИЯХ услуг: «анализ» в 49 строках каталога,
«тест» в 55, «после» в 29, «перед» в 9. Вычеркнуть их из токенайзера значило бы
сломать матч «Общий анализ крови».

Поэтому состав отобран по данным: берутся слова, которых нет в
`_PRICE_QUERY_STOPWORDS`, и отбрасываются все, что встречаются в названиях
каталога. Этот тест пересчитывает отбор на живом прайсе — если клиника заведёт
услугу со словом «прайс» в названии, он покраснеет.

## Инвариант

Слово, которое ХОТЬ ОДИН слой считает речевой обвязкой и которого нет ни в
одном названии услуги, не может мешать найти услугу в другом слое.
"""

from __future__ import annotations

import re

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services._biomaterial import _SPEECH_WRAPPER
from messengers_router.services._common import _normalise_input
from messengers_router.services._prices_helpers import (
    _MULTI_PRICE_NOISE_WORDS,
    _PRICE_SPEECH_ACT_STOPWORDS,
    SAMARA_PRICE_REGION_ID,
    resolve_price_service_name_from_catalog,
)

# Слова из `_SPEECH_WRAPPER`/`_MULTI_PRICE_NOISE_WORDS`, которые каталог
# использует в названиях услуг. Их НЕЛЬЗЯ стричь токенайзером — перечислены
# явно, чтобы тест ниже отличал «правомерное исключение» от «забыли добавить».
_CATALOG_USES_THESE = frozenset({"анализ", "тест", "после", "перед", "себя"})


def _rows():
    return [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _catalog_vocabulary(rows) -> set[str]:
    words: set[str] = set()
    for row in rows:
        words |= set(re.findall(r"[a-zа-яё0-9]+", _normalise_input(str(row.get("serviceName") or ""))))
    return words


def _filler_known_to_other_layers() -> set[str]:
    """Слова-обвязка из двух соседних слоёв, годные к проверке."""
    both = set(_SPEECH_WRAPPER) | set(_MULTI_PRICE_NOISE_WORDS)
    return {w for w in both if w.isalpha() and len(w) >= 3}


def test_catalog_sanity():
    rows = _rows()
    assert len(rows) > 1000, "нужен живой прайс региона"
    assert len(_catalog_vocabulary(rows)) > 1000


def test_speech_act_stopwords_absent_from_catalog():
    """Отбор пересчитывается на живых данных, а не верится на слово.

    Каждое слово, добавленное в стоп-лист прайса из соседних слоёв, обязано
    отсутствовать в названиях услуг. Иначе стрижка ломала бы матч.
    """

    vocabulary = _catalog_vocabulary(_rows())
    collisions = sorted(w for w in _PRICE_SPEECH_ACT_STOPWORDS if w in vocabulary)
    assert not collisions, (
        "эти слова стрижёт токенайзер прайса, но каталог их использует в названиях: "
        f"{collisions} — матч по таким услугам сломается"
    )

    # Обратная сторона: группа не должна молча разрастись за счёт слов,
    # которых нет ни в одном соседнем слое — тогда это уже наш выдуманный
    # список, а не сведение существующих.
    foreign = sorted(_PRICE_SPEECH_ACT_STOPWORDS - _filler_known_to_other_layers())
    assert not foreign, f"слова без источника в соседних слоях: {foreign}"


def test_filler_word_does_not_hide_the_service():
    """Инвариант класса: обвязка, известная другому слою, не мешает найти услугу.

    Свип идёт по самим спискам, а не по паре примеров: он расширяется вместе с
    ними. Исключены слова, которые каталог использует в названиях — для них
    поведение определяется матчем, а не обвязкой.
    """

    rows = _rows()
    assert resolve_price_service_name_from_catalog("ферритин", rows=rows), "предусловие: голое слово находится"

    broken = sorted(
        word
        for word in _filler_known_to_other_layers() - _CATALOG_USES_THESE
        if resolve_price_service_name_from_catalog(f"{word} ферритин", rows=rows) is None
    )
    assert not broken, f"обвязка прячет услугу: {broken}"


@pytest.mark.parametrize(
    "query",
    [
        "прайс ферритин",
        "записаться на ферритин",
        "посчитайте ферритин",
        "завтра ферритин",
        "уточните цену ферритин",
    ],
)
def test_natural_phrasings_resolve(query: str):
    """Обычные пациентские фразы, каждая из которых отдавала None."""
    assert resolve_price_service_name_from_catalog(query, rows=_rows())


@pytest.mark.parametrize(
    "query",
    [
        "приём терапевта по ОМС",
        "онлайн консультация",
        "приём кардиолога удалённо",
        "консультация по телефону",
        "приём по видеосвязи",
    ],
)
def test_unsatisfiable_qualifier_still_blocked(query: str):
    """Анти-under-block: ослабление не открыло дорогу уточнениям, которых нет.

    Расширение стоп-листа могло бы заодно проглотить «по ОМС» и «онлайн» —
    проверяем, что гард П2 по-прежнему гасит эти запросы.
    """
    assert resolve_price_service_name_from_catalog(query, rows=_rows()) is None


@pytest.mark.parametrize(
    ("query", "must_contain"),
    [
        ("общий анализ крови", "общий анализ крови"),
        ("анализ на ферритин", "ферритин"),
        ("подготовка перед фгдс", "фгдс"),
    ],
)
def test_catalog_words_are_not_stripped(query: str, must_contain: str):
    """Анти-регрессия: слова, которые каталог использует, остались рабочими."""
    found = resolve_price_service_name_from_catalog(query, rows=_rows())
    assert found, f"{query!r} перестал находиться"
    assert must_contain in _normalise_input(found), (query, found)
