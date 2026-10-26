"""Короткий ответ после списка цен с подсказкой «напишите: "все"» — BUG-2026-10-24-SHOW-ALL-HINT-ON-PROD (прод с деплоя 10).

Было: «сколько стоит оак» → две строки ОАК и «найдено ещё 6 вариантов, напишите: "все"»; пациент отвечает «1» (первая строка) —
одна функция ответа на висящий вопрос читала цифру как пункт 1 подсказки («все»), и бот раскрывал весь список, где хвост,
добранный по словам «общий анализ крови», нёс «Общий анализ мочи» и «ЗППП: анализ крови». «да» — то же.

Решение владельца 26.10: цифра — номер строки показанного списка, ответ — эта строка, без LLM; «да» на подсказку — показать все;
полный список — только строки той же услуги.

Инвариант класса: короткий ответ после нумерованного списка цен не показывает цен строк, которых пациент не видел, кроме явной
просьбы показать все («все», «покажи все», «да» на подсказку), — и тогда только строк той же услуги; номер N — ровно N-я строка
показанного (и раскрытого) списка. Мир — мини-прод, LLM недоступна (путь правил, как в мире гейта): номер решается без LLM.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from messengers_router import pending_answer, renderer
from messengers_router.services import _prices_helpers as H
from messengers_router.services._common import _normalise_catalog_text
from tests.test_l14_two_turn_consumers import _play, _priced
from tests.test_price_turns_golden import FIXTURE, mini_prod_world  # noqa: F401 — фикстура

_OAK = "сколько стоит оак"
_ROW = re.compile(r"^(\d+)\. (.+?) — \d[\d\s]* руб", re.M)
_OAK_WORDS = "общий анализ крови"


def _rows(text: str) -> list[str]:
    """Нумерованные строки цены, как их видит пациент, в порядке номеров."""

    return [name for _number, name in _ROW.findall(text)]


def _dialog(turns: list[str], name: str, world, monkeypatch) -> list[str]:
    return [turn.text for turn in _play(turns, f"hint-{name}", world, monkeypatch, "deploy")]


@pytest.mark.parametrize("first", [_OAK, "сколько стоит гепатит", "сколько стоит тестостерон", "сколько стоит мазок"])
def test_number_answers_that_row_of_the_shown_list(first, mini_prod_world, monkeypatch):  # noqa: F811
    """«N» после нумерованного списка — ровно N-я показанная строка (её цена, без чужих)."""

    shown = _rows(_dialog([first], first, mini_prod_world, monkeypatch)[0])
    assert len(shown) >= 2, shown
    for number, name in enumerate(shown, 1):
        reply = _dialog([first, str(number)], f"{first}-{number}", mini_prod_world, monkeypatch)[1]
        assert _priced(reply) == [_normalise_catalog_text(name)], (number, reply)


@pytest.mark.parametrize("reply", ["3", "9", "15"])
def test_number_beyond_the_shown_rows_shows_no_unseen_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Номер строки, которой пациент не видел (под подсказкой «ещё N вариантов»), — не раскрытие и не чужая цена."""

    first, second = _dialog([_OAK, reply], f"beyond-{reply}", mini_prod_world, monkeypatch)
    assert 'напишите: "все"' in first
    assert set(_priced(second)) <= set(_priced(first)), second


@pytest.mark.parametrize("reply", ["все", "покажи все", "все варианты", "да"])
def test_show_all_lists_only_rows_of_the_same_service(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Явная просьба показать все — показанные строки на месте, новые — только ОАК (не ОАМ, не «ЗППП: анализ крови»)."""

    first, second = _dialog([_OAK, reply], f"all-{reply}", mini_prod_world, monkeypatch)
    shown, now = _rows(first), _rows(second)
    assert len(now) > len(shown) and now[: len(shown)] == shown, second
    assert all(_OAK_WORDS in name.casefold() for name in now), now


@pytest.mark.parametrize(
    ("first", "words"),
    [("сколько стоит глюкозотолерантный тест", ("глюкозотолерант", "тест")), ("сколько стоит исследование кала", ("исследован", "кал"))],
)
def test_show_all_of_a_rules_family_keeps_every_word_of_the_service(first, words, mini_prod_world, monkeypatch):  # noqa: F811
    """Семейство правил: хвост за «все» — строки, где есть каждое слово услуги; расширение по одному корню тянуло чужие
    («глюкозотолерантный тест» → «Экспресс-тест для определения гепатита С», «исследование кала» → гистология)."""

    shown, expanded = (_rows(text) for text in _dialog([first, "все"], f"rules-{first}", mini_prod_world, monkeypatch))
    new = [name for name in expanded if name not in shown]  # в группах формата оказания хвост встаёт в свою группу
    # у «глюкозотолерантного теста» весь хвост был чужим — подсказки «все» больше нет, «все» новых строк не добавляет
    assert len(shown) >= 2 and set(shown) <= set(expanded), expanded
    assert all(all(word in name.casefold() for word in words) for name in new), new


def test_number_after_show_all_is_the_row_of_the_full_list(mini_prod_world, monkeypatch):  # noqa: F811
    """После раскрытия на виду весь список: «4» — его четвёртая строка."""

    _first, expanded, picked = _dialog([_OAK, "все", "4"], "after-all", mini_prod_world, monkeypatch)
    assert len(_rows(expanded)) >= 4
    assert _priced(picked) == [_normalise_catalog_text(_rows(expanded)[3])], picked


def test_number_on_the_hint_is_not_an_answer_to_it_and_spends_no_llm(monkeypatch):
    """Одна функция: цифра на подсказку «все» — не пункт подсказки (решение без LLM), «все» и «да» — показать все."""

    async def no_llm(*_args, **_kwargs):
        raise AssertionError("цифра на подсказку решается без LLM")

    async def decide(reply: str) -> tuple[pending_answer.Outcome, pending_answer.Source, bool]:
        found = await pending_answer.answer("show_all", 'Чтобы показать их, напишите: "все".', pending_answer.SHOW_ALL_OPTIONS, reply)
        return found.outcome, found.source, pending_answer.shows_all()  # решение хода живёт в контексте хода

    monkeypatch.setattr(pending_answer, "choose", no_llm)
    for reply in ("1", "2.", "№3"):
        assert asyncio.run(decide(reply)) == (pending_answer.Outcome.NONE, pending_answer.Source.FAST, False), reply
    for reply in ("все", "да"):
        assert asyncio.run(decide(reply))[2], reply


def _context(variants: list[dict], visible: int) -> dict:
    return {"_price_family_context": {"service_name": "x", "family_variants": variants, "visible_limit": visible}}


def test_row_by_number_follows_the_rendered_numbering():
    """Номер читается так, как рендер пронумеровал строки: подряд (и в одной группе формата оказания) — N-я строка; групп
    несколько — у каждой своя нумерация, номер неоднозначен, строки нет."""

    plain = [{"serviceName": f"Услуга {n}", "cost": 100 * n} for n in range(1, 6)]
    payload = {"service_name": "x", "family_variants": plain, "visible_limit": 3, "showing_all": False}
    printed = dict(_ROW.findall(renderer._format_price_family_variants(payload)))
    assert printed == {"1": "Услуга 1", "2": "Услуга 2", "3": "Услуга 3"}
    for number, name in printed.items():
        assert H._price_family_row_from_context(number, _context(plain, 3))["serviceName"] == name
    for reply in ("4", "0", "1 и 2", "первый", ""):
        assert H._price_family_row_from_context(reply, _context(plain, 3)) is None, reply
    assert H._price_family_row_from_context("1", _context(plain[:1], 1)) is None  # одна строка — не список

    one_group = [{**row, "care_setting_label": "дневной стационар"} for row in plain]
    printed = dict(_ROW.findall(renderer._format_price_family_variants({**payload, "family_variants": one_group})))
    assert printed == {"1": "Услуга 1", "2": "Услуга 2", "3": "Услуга 3"}
    assert H._price_family_row_from_context("2", _context(one_group, 3))["serviceName"] == "Услуга 2"

    grouped = [{**row, "care_setting_label": "поликлиника" if n % 2 else "дневной стационар"} for n, row in enumerate(plain)]
    text = renderer._format_price_family_variants({**payload, "family_variants": grouped})
    assert [number for number, _name in _ROW.findall(text)].count("1") == 2  # нумерация в группах своя
    assert H._price_family_row_from_context("1", _context(grouped, 3)) is None


def test_oak_tail_holds_only_oak_rows():
    """Хвост списка ОАК (его раскрывает «все») — только строки ОАК: на прайсе мини-прода ранжирование по словам «общий анализ
    крови» тянуло ОАМ и «ЗППП: анализ крови»."""

    path = FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    payload = H._build_oak_canonical_payload(_OAK, rows, show_all=True)
    names = [str(row["serviceName"]) for row in payload["family_variants"]]
    assert len(names) >= 3 and all(_OAK_WORDS in name.casefold() for name in names), names
