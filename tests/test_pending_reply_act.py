"""Ответ на оффер оператора — одна функция ответа на висящий вопрос (класс 5; дизайн
`docs/DESIGN_2026-10-18_one_pending_answer.md`, решения владельца 16.10 и Q1–Q5 18.10).

Аудит 14.10, пункт 5: на «Перевести на оператора?» ответ «конечно переведите» правила не понимают — оффер снимается без
перевода; стенд 18.10: «не надо оператора, я сам» — перевод (явный запрос оператора по буквам раньше разбора ответа).
Ответ решает одна функция: якорь согласия — перевод без LLM; иначе LLM; «нет» модели — отказ, перевода нет,
даже если в реплике есть слово «оператор»; сбой — прежние правила.

Инварианты класса (через обработчик эндпоинта, мини-прод, оракул LLM):
- согласие словами → перевод; при сбое LLM — правила, без перевода (как было);
- отказ от оператора словами → без перевода, «Хорошо, продолжаем»;
- пауза — без перевода; сбой LLM — ответ правил (`src = fallback`);
- голый якорь и «якорь + хвост» — без вызова LLM; срочное — раньше всего;
- два хода: после перевода следующий ответ не про оффер.

Путь прода (переворот 27.10: `turn_spans_llm`, `price_by_parse` = on): вопрос о цене после перевода без LLM — «Уточните
название»; в двухходовом тесте LLM — пути прода (`test_pending_reply_show_all._prod_llm`) поверх оракула висящего вопроса.
"""

from __future__ import annotations

import pytest

from tests.test_pending_reply import (  # noqa: F401 — mini_prod_world, synonym_fork — фикстуры
    _OFFER_QUESTION,
    _STOPLIST_OFFER,
    _PendingOracle,
    _answer,
    _expects,
    _offer_after_first_turn,
    _play,
    _prepared,
    mini_prod_world,
    synonym_fork,
)
from tests.test_pending_reply_show_all import _prod_llm

_YES = "конечно переведите"  # без запятой — не «якорь + хвост»: решает модель
_MANUAL = "Соединяю с оператором"
_DECLINED = "Хорошо, продолжаем диалог"


_DOWN = ConnectionError("ollama лежит")


def _offer(reply: str, mode: str, world, monkeypatch, *, oracle=None, name: str = "act"):
    return _play([_STOPLIST_OFFER, reply], f"{name}-{mode}", mode, world, monkeypatch, before=_offer_after_first_turn(), oracle=oracle)


def test_yes_in_other_words_is_transferred_under_on(mini_prod_world, monkeypatch):  # noqa: F811
    down, *_ = _offer(_YES, "down", mini_prod_world, monkeypatch, oracle=_PendingOracle(exc=_DOWN))
    assert not down[1]["handoff"]  # сбой LLM — правила «конечно переведите» не понимают, как было
    on, _states, rows, oracle = _offer(_YES, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({_YES: _answer("yes")}))
    assert on[1]["handoff"] and _MANUAL in on[1]["text"], on[1]
    assert len(oracle.calls) == 1
    assert (rows[1]["tu_pending_outcome"], rows[1]["tu_pending_src"]) == ("yes", "llm")
    [entry] = [entry for entry in rows[1]["choose"] if entry["task"] == "pending_reply"]
    assert not entry.get("shadow")  # решение хода, в его бюджете


@pytest.mark.parametrize("reply", ["не надо оператора, я сам", "нет, оператор не нужен"])
def test_refusing_the_operator_in_words_is_not_a_transfer(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on, states, rows, _o = _offer(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("no")}), name="refuse")
    assert not on[1]["handoff"], on[1]["text"]
    assert not states[1].get("_operator_offer_pending")


def test_refusal_from_the_model_says_we_go_on(mini_prod_world, monkeypatch):  # noqa: F811
    reply = "не надо оператора, я сам"
    on, *_ = _offer(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("no")}), name="refuse-text")
    assert _DECLINED in on[1]["text"], on[1]["text"]


def test_pause_is_not_a_transfer(mini_prod_world, monkeypatch):  # noqa: F811
    reply = "я подумаю пока"
    on, *_ = _offer(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("pause")}), name="pause")
    assert not on[1]["handoff"] and _DECLINED in on[1]["text"], on[1]["text"]


def test_llm_failure_keeps_the_rules_answer(mini_prod_world, monkeypatch):  # noqa: F811
    on, on_states, rows, _o = _offer(_YES, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle(exc=_DOWN), name="down")
    assert not on[1]["handoff"] and on_states[1].get("_operator_offer_pending") is not True  # правила не поняли — оффер снят
    assert rows[1]["tu_pending_src"] == "fallback"


@pytest.mark.parametrize("reply", ["да", "конечно, переведите", "да, пожалуйста, соедините"])
def test_an_anchor_does_not_call_the_llm(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on, _states, rows, oracle = _offer(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("no")}), name="bare")
    assert oracle.calls == [] and on[1]["handoff"]
    assert (rows[1]["tu_pending_outcome"], rows[1]["tu_pending_src"]) == ("yes", "fast")


def test_urgent_comes_before_the_decision(mini_prod_world, monkeypatch):  # noqa: F811
    reply = "нет, мне плохо, задыхаюсь"
    before = _prepared({"_operator_offer_pending": True}, _expects(_OFFER_QUESTION))
    records, _states, _rows, oracle = _play([reply], "act-urgent", "on", mini_prod_world, monkeypatch, before=before, oracle=_PendingOracle({reply: _answer("no")}))
    assert records[0]["handoff"] and oracle.calls == []


def test_after_the_transfer_the_next_turn_is_not_about_the_offer(mini_prod_world, monkeypatch):  # noqa: F811
    records, states, _rows, _oracle = _play(
        [_STOPLIST_OFFER, _YES, "сколько стоит ферритин"], "act-two", "on", mini_prod_world, monkeypatch,
        before=_offer_after_first_turn(),
        oracle=_prod_llm(_PendingOracle({_YES: _answer("yes")}), monkeypatch, phrases=("мрт", "ферритин"), rows_of="ферритин"),
    )
    assert records[1]["handoff"]
    assert not states[1].get("_operator_offer_pending")
    assert "Ферритин" in records[2]["text"] and not records[2]["handoff"], records[2]
