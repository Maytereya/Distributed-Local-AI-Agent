"""Подсказка «…напишите: "все"» — висящий вопрос бота (`_expects` типа `show_all`, наблюдатель; решения владельца 15.10).

Семейный список цены с подсказкой ждёт ответа «все» (полный список по контексту семейства). Ответ раньше разбирали
только правила (`_is_price_show_all_request`), а ответ городом — отдельное правило «только город»; решение владельца —
разбирать через `pending_reply`, где подсказка бота — висящий вопрос. Этот шаг — только наблюдатель: поле пишется, ответы
не меняются; ответ на подсказку решает одна функция ответа на висящий вопрос (`pending_answer`).

Инварианты (через обработчик эндпоинта, мини-прод):
- ответ с подсказкой → `_expects` = {kind: pick, type: show_all, options: ["все"]}; вопрос — текст ответа с подсказкой;
- «все» → полный список, как было; подсказки нет — поле снято;
- ответ без подсказки (одна цена) — поля `show_all` нет;
- ход после подсказки несёт `tu_pending_type = show_all`: вопрос и пункт «все» видит модель.

Путь прода (переворот 27.10: `turn_spans_llm`, `price_by_parse` = on): без LLM цена — «Уточните название», подсказки нет. LLM
диалога — пути прода (`test_pending_reply_show_all._prod_llm`: разметка называет услугу, выбор строки признаёт её строки).
"""

from __future__ import annotations

from tests.test_pending_reply import (  # noqa: F401 — фикстуры
    _PendingOracle,
    _answer,
    _play,
    mini_prod_world,
    synonym_fork,
)
from tests.test_pending_reply_show_all import _prod_llm

_FAMILY = "сколько стоит оак"
_HINT = 'напишите: "все"'


def test_the_hint_is_a_pending_pick_question(mini_prod_world, monkeypatch):  # noqa: F811
    records, states, _rows, _o = _play([_FAMILY], "show-all-ask", "off", mini_prod_world, monkeypatch, oracle=_prod_llm(_PendingOracle(), monkeypatch))
    assert _HINT in records[0]["text"], records[0]["text"]
    expected = states[0]["_expects"]
    assert (expected["kind"], expected["type"], expected["options"], expected["asks"]) == ("pick", "show_all", ["все"], 1)
    assert _HINT in expected["question"]


def test_all_shows_the_full_list_and_clears_the_question(mini_prod_world, monkeypatch):  # noqa: F811
    records, states, _rows, _o = _play([_FAMILY, "все"], "show-all-all", "off", mini_prod_world, monkeypatch, oracle=_prod_llm(_PendingOracle(), monkeypatch))
    assert _HINT in records[0]["text"], records[0]["text"]
    assert _HINT not in records[1]["text"] and records[1]["text"].count(" руб.") > records[0]["text"].count(" руб."), records[1]["text"]
    assert "_expects" not in states[1] or states[1]["_expects"].get("type") != "show_all"


def test_a_single_price_answer_asks_nothing(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _prod_llm(_PendingOracle(), monkeypatch, phrases=("ферритин",), rows_of="ферритин")
    records, states, _rows, _o = _play(["сколько стоит ферритин"], "show-all-none", "off", mini_prod_world, monkeypatch, oracle=oracle)
    assert records[0]["text"].count(" руб.") == 1, records[0]["text"]  # одна цена — дошли до ответа о цене
    assert _HINT not in records[0]["text"]
    assert (states[0].get("_expects") or {}).get("type") != "show_all"


def test_the_one_function_sees_the_show_all_question(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _PendingOracle({"покажите все варианты пожалуйста": _answer(picks=(1,))})
    _records, _states, rows, _llm = _play([_FAMILY, "покажите все варианты пожалуйста"], "show-all-one", "on", mini_prod_world, monkeypatch, oracle=_prod_llm(oracle, monkeypatch))
    assert rows[1]["tu_pending_type"] == "show_all" and rows[1]["tu_pending_outcome"] == "pick"
    assert oracle.calls and oracle.calls[0]["prompt"].rstrip().endswith("1. все")
