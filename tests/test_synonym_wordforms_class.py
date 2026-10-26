"""Класс: словоформы синонима МИС дают одни и те же кандидаты выбора строки цены (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 1).

Карточка 24.10: синоним клиники «коагулограмма» — у двух гемостазиограмм. «Сколько стоит коагулограмма» — обе
гемостазиограммы; «Стоимость коагулограммы», «Цена на коагулограмму» — «Определение времени свертываемости крови по
Сухареву» (чужая цена): синоним сравнивался с фразой буква в букву, строк синонима в списке LLM не было, и она выбрала
похожее. Тот же класс, что «анализ / анализы».

Инварианты (выключатель `synonym_wordforms` снят 31.10 — неделя `on` на проде с деплоя 17):
- любая словоформа синонима ставит строки его услуг первыми в список LLM, как дословный синоним; сильным совпадением она не
  становится — выбирает LLM (общее начало склеивает и разные слова);
- словоформа — строгая (`_same_word_strict`): общее начало длинного слова без окончания; «коагулограмма» ≠ «коагуляция»;
- синоним, который буквы и так находят в названиях прайса («терапевт» — «врача-терапевта»), строк не добавляет — как было.
"""

from __future__ import annotations

import pytest

from messengers_router.choice import Choice, Status
from messengers_router.services import _biomaterial as bio
from messengers_router.services import _price_select_llm as sel
from tests.test_price_candidates_by_meaning_class import _mis_synonyms, _Oracle

ROWS = [
    {"serviceId": 715, "serviceName": "Расширенная гемостазиограмма", "cost": 2880, "priceUnitId": 145},
    {"serviceId": 7935, "serviceName": '"Гемостазиограмма (скрининг)"', "cost": 950, "priceUnitId": 145},
    {"serviceId": 471, "serviceName": "Определение времени свертываемости крови по Сухареву", "cost": 250, "priceUnitId": 142},
    {"serviceId": 5001, "serviceName": "Аргоноплазменная коагуляция шейки матки", "cost": 4000, "priceUnitId": 150},
    {"serviceId": 5002, "serviceName": "Радиоволновая коагуляция", "cost": 3500, "priceUnitId": 150},
    {"serviceId": 7001, "serviceName": "Прием (осмотр, консультация) врача-терапевта", "cost": 2700, "priceUnitId": 149},
    {"serviceId": 7002, "serviceName": "Прием (осмотр, консультация) врача-хирурга", "cost": 3000, "priceUnitId": 149},
]
_SYNONYMS = {
    "коагулограмма": ("Расширенная гемостазиограмма", '"Гемостазиограмма (скрининг)"'),
    "терапевт": ("Прием (осмотр, консультация) врача-терапевта",),
}
_GEMOSTAZ = {"Расширенная гемостазиограмма", '"Гемостазиограмма (скрининг)"'}
_FORMS = [
    "коагулограмма",
    "коагулограммы",
    "коагулограмме",
    "коагулограмму",
    "коагулограммой",
    "Стоимость коагулограммы",
    "Цена на коагулограмму",
    "Сколько стоит анализ на коагулограмму",
]


def _synonym_names(monkeypatch, text: str) -> set[str]:
    _mis_synonyms(monkeypatch, _SYNONYMS)
    return {row["serviceName"] for row in sel._synonym_rows([text], ROWS)}


@pytest.mark.parametrize("text", _FORMS)
def test_every_wordform_of_a_synonym_offers_its_services(text, monkeypatch) -> None:
    assert _synonym_names(monkeypatch, text) == _GEMOSTAZ


@pytest.mark.parametrize("text", ["приём терапевта", "консультация терапевту", "терапевт"])
def test_synonym_the_letters_already_find_adds_nothing(text, monkeypatch) -> None:
    assert _synonym_names(monkeypatch, text) == set()


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("коагулограммы", "коагулограмма", True),
        ("коагулограммами", "коагулограмма", True),
        ("терапевта", "терапевт", True),
        ("сифилиса", "сифилис", True),
        ("спермограмму", "спермограмма", True),
        ("коагулограмма", "коагуляция", False),
        ("креатинина", "креатинфосфокиназа", False),
        ("прогестерон", "прогестин", False),
        ("гастроскопию", "гастрин", False),
        ("кровь", "крови", False),  # короче порога — целиком, как `_same_word`
    ],
)
def test_strict_wordform(a, b, same) -> None:
    assert sel._same_word_strict(a, b) is same
    assert sel._same_word_strict(b, a) is same


def test_wordform_synonym_is_not_a_strong_match(monkeypatch) -> None:
    _mis_synonyms(monkeypatch, _SYNONYMS)
    assert bio.mis_synonym_candidates("коагулограммы") == ()  # сильное совпадение — только дословное (решение 13.10)
    assert sel.strong_rows("коагулограммы", ROWS) == []


@pytest.mark.parametrize("question", ["Стоимость коагулограммы", "Цена на коагулограмму"])
def test_llm_sees_the_synonym_rows_first_and_decides(question, monkeypatch) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_CANDIDATES_BY_MEANING", "on", raising=False)
    monkeypatch.setattr(sel, "_record_synonym_suggestion", lambda *a, **k: None)
    _mis_synonyms(monkeypatch, _SYNONYMS)
    oracle = _Oracle(Choice(Status.NONE_FIT, reason="other_service"))
    monkeypatch.setattr(sel, "choose", oracle)
    import asyncio

    asyncio.run(sel.select_price_rows(question, ROWS))
    first = oracle.calls[0][:2] if oracle.calls else []
    assert first and all("гемостазиограмма" in text.lower() for text in first), oracle.calls
