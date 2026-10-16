"""Гард П2 у сильной фразы разбора хода (BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS, `strong_phrase_guard`).

П2 (решение владельца): уточнение, которого клиника не оказывает («по ОМС»), — честное «не нашёл» или «Уточните», а не
платная цена «почти той» услуги. Под `turn_parse = on` LLM разбора выбирает фразу из перечня кода и может оторвать от
услуги условие (прод 16.10: «приём терапевта» из «сколько стоит приём терапевта по ОМС» — вопреки правилу 3 своего
промпта): фраза сильная по данным МИС, читатель цены отдавал её строки напрямую, гард П2 не спрашивался — платная цена.
Под `turn_parse = off` тот же ход — «Уточните».

Фикс — LLM-first, как класс A (`parse_price_guard`) и `p2_llm_judge`: слово вне сильной фразы, которого нет в прайсе, —
повод спросить выбор строки (`judge_qualifier`, правило 3 промпта отсекает «по ОМС»), а не вердикт словаря: словарь
глушил разговорные слова («первый приём», «делается»; встречный свип P2-ENTITY-BYPASS — 18 ложных из 28).

Инварианты класса (через обработчик эндпоинта, мини-прод, прод-набор 16.10, оракул LLM):
- `off` — как было с 15.10 (откат): строки сильной фразы, выбор строки LLM не спрашивается;
- `llm` + LLM «не та» — без цены («не нашёл»); `llm` + LLM выбрала строку — цена;
- `llm` + сбой LLM выбора строки — ответ тот же, что под `turn_parse = off` (гард правил): цены нет;
- разговорное слово вне сильной фразы судит LLM, не словарь: выбрала строку — цена, «не та» — без цены;
- реплики без слова вне прайса — ответ под `llm` тот же, что под `off`, выбор строки не зовётся;
- два хода: короткий вопрос после «не нашёл» не получает цену того, чего пациент не видел.
"""

from __future__ import annotations

import json
import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_shadow import _deploy
from tests.test_turn_parse_t3 import _FOLLOW_UPS, _ROWS_MARKS, _Oracle, _assert_follow_up_stays_on_what_was_shown
from ollama import GenerateResponse

# Прод 16.10 поверх набора деплоя (`_deploy`): разбор хода, выбор строки промптом модуля, L-13, класс A.
_PROD_1016 = {
    "MR_TURN_PARSE": "on",
    "MR_CHOOSE_PRICE": "on",
    "MR_PARSE_PRICE_GUARD": "llm",
    "MR_CONTEXT_SHOWN": "on",
    "MR_JOURNAL_PHRASE": "mention",
}
# Фразы, которые «выбирает» LLM разбора: сильная фраза без условия (как прод выбрал «приём терапевта»).
_PHRASES = ("ферритин", "глюкоза")
# Строки, которые оракул «выбирает» на выборе строки: префикс строки варианта «название — цена».
_PICK = ("Ферритин — 530", "Глюкоза — 190")

_STRONG_P2 = (
    "сколько стоит ферритин по омс",
    "сколько стоит глюкоза по ОМС?",
    "ферритин по полису омс сколько стоит",
    "сколько стоит ферритин бесплатно по омс",
)
_CONVERSATIONAL = (  # слово вне фразы, которого нет в прайсе, — не условие (урок P2-ENTITY-BYPASS: «первый», «делается»)
    "Сколько стоит ферритин, делается ли он у вас?",
    "сколько стоит первый ферритин",
)
_OUTSIDE_THE_CLASS = (
    "сколько стоит ферритин",
    "сколько стоит анализ на ферритин",
    "ферритин цена",
    "сколько стоит глюкоза",
)


class _Down:
    """LLM недоступна во всех местах (разбора тоже нет)."""

    async def generate(self, **_kwargs):
        raise ConnectionError("тест: LLM недоступна")


class _Reason(_Oracle):
    """Разбор — сильная фраза без условия; выбор строки (промпт модуля) — пусто с кодом причины и терминами второго
    круга (их суд условия использовать не должен: один круг)."""

    def __init__(self, reason: str) -> None:
        super().__init__(_PHRASES, ("—",))
        self.reason = reason

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if system is not None and any(mark in seen for mark in _ROWS_MARKS):
            self.asked.append("rows")
            answer = {"picks": [], "reason": self.reason, "search_terms": ["ферритин сыворотки"]}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        return await super().generate(prompt=prompt, system=system, **kwargs)


def _llm(kind: str) -> _Oracle:
    """Разбор у всех — сильная фраза без условия; выбор строки: `not_it` — «подходящих нет», `pick` — строки
    `_PICK`, `rows_down` — сбой (только выбор строки, разбор ответил)."""

    return {
        "not_it": lambda: _Oracle(_PHRASES, ("—",)),
        "pick": lambda: _Oracle(_PHRASES, _PICK),
        "rows_down": lambda: _Oracle(_PHRASES, ()),
    }[kind]()


def _run(turns, world, monkeypatch, *, guard: str, llm, parse: str = "on", session: str) -> tuple[list[dict], list[dict], object]:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    for key, value in _PROD_1016.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(cfg, "MR_TURN_PARSE", parse, raising=False)
    monkeypatch.setattr(cfg, "MR_STRONG_PHRASE_GUARD", guard, raising=False)
    client = _llm(llm) if isinstance(llm, str) else llm
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", client)
    with ev_world.FrozenWorld(world, "day") as frozen:
        record = run_dialog(list(turns), session)
        rows = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return record, rows, client


def _priced(text: str) -> bool:
    return bool(re.search(r"\d\s*руб", text))


def _sid(*parts: str) -> str:
    return "p2sp-" + "-".join(re.sub(r"\W+", "_", part)[-14:] for part in parts)


def test_precondition_the_strong_phrase_leaves_the_qualifier_outside(mini_prod_world):  # noqa: F811
    """Предусловие класса на мини-прайсе: фраза без условия — сильная по данным МИС (её строки и есть ответ без
    гарда), в перечне кода есть и фраза с условием (правило 3 промпта разбора предлагает её LLM), а гард П2 по
    всей реплике отказывает — под `turn_parse = off` цены нет. Разговорные слова гард по буквам тоже считает
    уточнением — поэтому судит LLM."""

    from agent_logic_2.nayka_api import api_price

    from messengers_router.services import _prices_helpers as H
    from messengers_router.services import _turn_phrases as P

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        for phrase in _PHRASES:
            assert P.phrase_rows(phrase, rows), phrase
        options = [option.raw for option in P.phrase_candidates(_STRONG_P2[0], facet=None, catalog=rows).options]
        assert "ферритин по омс" in options and "ферритин" in options, options
        for reply in (*_STRONG_P2, *_CONVERSATIONAL):
            assert H.query_has_unsatisfiable_qualifier(reply, rows), reply
        for reply in _OUTSIDE_THE_CLASS:
            assert not H.query_has_unsatisfiable_qualifier(reply, rows), reply


@pytest.mark.parametrize("reply", _STRONG_P2)
def test_turn_parse_off_has_no_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Эталон класса — ход под `turn_parse = off`: гард П2 отказывает, цены нет."""

    [turn], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="pick", parse="off", session=_sid("poff", reply))
    assert not _priced(turn["text"]), turn["text"]


@pytest.mark.parametrize("reply", _STRONG_P2)
def test_off_keeps_the_answer_of_15_10(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Откат (`off`) — как было с 15.10: строки сильной фразы с ценой, выбор строки LLM не спрашивается."""

    [turn], [row], client = _run([reply], mini_prod_world, monkeypatch, guard="off", llm="not_it", session=_sid("off", reply))
    assert _priced(turn["text"]), turn["text"]
    assert "rows" not in client.asked
    assert "strong_phrase_guard" not in row


@pytest.mark.parametrize("reply", _STRONG_P2)
def test_the_llm_says_not_it_and_there_is_no_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    [turn], [row], client = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="not_it", session=_sid("not", reply))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert "rows" in client.asked and row["strong_phrase_guard"] == "not_found"
    assert row["tu_parse_status"] == "picked"


@pytest.mark.parametrize("reply", _STRONG_P2)
def test_the_llm_picks_the_row_and_it_is_priced(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """LLM выбора строки решила, что условие строке не мешает, — цена (решение за LLM, не за словарём)."""

    [turn], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="pick", session=_sid("pick", reply))
    assert _priced(turn["text"]), turn["text"]
    assert row["strong_phrase_guard"] == "rows"


@pytest.mark.parametrize("reply", (*_STRONG_P2, *_CONVERSATIONAL))
def test_a_failed_row_choice_answers_as_turn_parse_off(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM выбора строки (разбор ответил) — сильная фраза ход не решает: ответ пути правил с гардом, тот же,
    что под `turn_parse = off` (дезинформация ценой хуже уточнения: при сбое — не цена «почти той» услуги)."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="rows_down", parse="off", session=_sid("doff", reply))
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="rows_down", session=_sid("don", reply))
    assert on["text"] == off["text"], (on["text"], off["text"])
    assert row["strong_phrase_guard"] == "fallback"


@pytest.mark.parametrize("reply", _CONVERSATIONAL)
def test_a_conversational_word_is_judged_by_the_llm_not_by_the_dictionary(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный инвариант (`guard_over_blocks_silently`): «делается», «первый» гард по буквам считает уточнением.
    Под `llm` их судит LLM: выбрала строку — цена, как с 15.10; «не та» — без цены."""

    [before], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="off", llm="pick", session=_sid("coff", reply))
    [picked], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="pick", session=_sid("cpick", reply))
    assert _priced(before["text"]) and picked["text"] == before["text"], picked["text"]
    assert row["strong_phrase_guard"] == "rows"
    [not_it], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="not_it", session=_sid("cnot", reply))
    assert not _priced(not_it["text"]) and row["strong_phrase_guard"] == "not_found", not_it["text"]


@pytest.mark.parametrize("reply", _OUTSIDE_THE_CLASS)
def test_replies_outside_the_class_do_not_change(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Сильная фраза без слова вне прайса — ответ под `llm` тот же, что под `off`, выбор строки не зовётся."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="off", llm="not_it", session=_sid("ooff", reply))
    [on], [row], client = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="not_it", session=_sid("oon", reply))
    assert on["text"] == off["text"], on["text"]
    assert "strong_phrase_guard" not in row


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_not_found_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода: после «не нашёл» короткий вопрос («а сколько?», «да», цифра) не получает цену сильной фразы,
    которой пациент не видел."""

    (first, second), _rows, _ = _run(
        [_STRONG_P2[0], follow_up], mini_prod_world, monkeypatch, guard="llm", llm="not_it", session=_sid("fu", follow_up)
    )
    assert not _priced(first["text"]), first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)


def test_second_turn_with_the_condition_after_a_price(mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода: «сколько стоит ферритин» → цена; «а ферритин по омс?» — новая реплика с условием судится заново:
    LLM «не та» — без цены (а не цена из первого ответа)."""

    (first, second), rows, _ = _run(
        ["сколько стоит ферритин", "а ферритин по омс?"], mini_prod_world, monkeypatch, guard="llm", llm="not_it", session=_sid("two")
    )
    assert _priced(first["text"]), first["text"]
    assert not _priced(second["text"]), second["text"]
    assert rows[-1].get("strong_phrase_guard") == "not_found"


def test_the_condition_is_judged_in_one_round(mini_prod_world, monkeypatch):  # noqa: F811
    """«Условия нет» (`qualifier_absent`, правило 3) — «не нашёл» сразу: второй круг по терминам LLM не идёт (он искал
    бы «почти то», а под `turn_parse = on` упирался в лимит решений хода и превращал ответ в `unavailable` — живой
    свип 16.10: 19 из 36 ходов «по омс»)."""

    oracle = _Reason("qualifier_absent")
    [turn], [row], _ = _run([_STRONG_P2[0]], mini_prod_world, monkeypatch, guard="llm", llm=oracle, session=_sid("qa"))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert oracle.asked.count("rows") == 1 and row["strong_phrase_guard"] == "not_found"
    assert not any(call.get("outcome") == "budget" for call in row.get("llm") or []), row.get("llm")


@pytest.mark.parametrize("reply", (_STRONG_P2[0], *_CONVERSATIONAL))
def test_other_service_keeps_the_mis_rows(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """«Другая услуга» (`other_service`) — LLM спорит с данными МИС о самой услуге, а судить ей здесь только условие:
    строки сильной фразы (решение 13.10 — сильное совпадение решают данные МИС), ответ как с 15.10 (живой свип 16.10:
    «альфа-липопротеин» → синоним МИС «Аполипопротеин A1» терял цену)."""

    [before], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="off", llm="not_it", session=_sid("os-off", reply))
    [after], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=_Reason("other_service"), session=_sid("os", reply))
    assert after["text"] == before["text"] and _priced(after["text"]), after["text"]
    assert row["strong_phrase_guard"] == "strong"


@pytest.mark.parametrize("reason", ["too_general", "not_a_service"])
def test_not_a_service_name_answers_as_turn_parse_off(reason, mini_prod_world, monkeypatch):  # noqa: F811
    """«Слишком общее» / «не услуга» — сильная фраза ход не решает: путь правил с гардом, как под `turn_parse = off`."""

    reply = _STRONG_P2[0]
    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm="rows_down", parse="off", session=_sid("ng-off", reason))
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=_Reason(reason), session=_sid("ng", reason))
    assert on["text"] == off["text"] and not _priced(on["text"]), on["text"]
    assert row["strong_phrase_guard"] == "fallback"


def test_the_judge_keeps_the_mis_rows_and_takes_only_other_rows(mini_prod_world, monkeypatch):  # noqa: F811
    """Выбор LLM, пересекающийся со строками сильной фразы, — ответ строками фразы (соседи, добавленные LLM, — её мнение
    о самой услуге: живой свип 16.10 «глюкоза» → + «Экспресс-тест на глюкозу»); выбор только других строк — он (LLM
    нашла строку, у которой это условие есть, или чужую: «ФГДС под седацией» → профиль анализов «под наркозом») — список
    из строк фразы и выбора LLM (решение 22.07)."""

    reply = "Сколько стоит глюкоза? делается у вас?"
    [before], _rows, _ = _run([reply], mini_prod_world, monkeypatch, guard="off", llm="not_it", session=_sid("mis-off"))
    assert _priced(before["text"]) and "Экспресс" not in before["text"], before["text"]
    wider = _Oracle(_PHRASES, ("Глюкоза — 190", "Экспресс-тест на глюкозу"))
    [kept], [row], client = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=wider, session=_sid("mis-wide"))
    assert "rows" in client.asked and row["strong_phrase_guard"] == "rows"
    assert kept["text"] == before["text"], kept["text"]
    other = _Oracle(_PHRASES, ("Экспресс-тест на глюкозу",))
    [moved], [row], _ = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=other, session=_sid("mis-other"))
    assert "Экспресс-тест на глюкозу" in moved["text"] and "Глюкоза — 190" in moved["text"], moved["text"]
    assert row["strong_phrase_guard"] == "rows"
