"""Ответ на подсказку «…напишите: "все"» — одна функция ответа на висящий вопрос (класс 5, `pending_reply = on`;
решения владельца 15.10 — подсказка — висящий вопрос, ответ разбирать через LLM, не дописывать слова в регулярки).

Было: «показать все» узнаёт только `_prices_helpers._is_price_show_all_request` — целая реплика «все», «всё», «покажи
все», «показать все», «все варианты / услуги / анализы». «покажите все варианты пожалуйста», «покажите все», «полный
список», «давай все» уходят мимо цены («Не совсем понял…»), «да, давайте все» забирает вторичный оффер (подбор анализов).

Стало: под `on` ответ решает одна функция (`pending_answer`): пункт «все» или «да» — «показать все» тем же путём, что у
правил (`_price_family_payload_from_context(..., show_all=True)`): полный список того же семейства. «другая тема»,
ничего, сбой — как было; реплику из одного определения «все» LLM не решает (сужение без LLM).

Инварианты класса (через обработчик эндпоинта, мини-прод, оракул LLM):
- ответ «все» другими словами → полный список (строк с ценой больше, подсказки нет); тот же диалог под off — как было;
- другая тема / ничего / сбой LLM — ответы и состояние как под off; «нет» — не полный список;
- реплику из одного определения «все» LLM не решает (`src = fast`);
- решение — в бюджете хода (не тень);
- два хода: после полного списка короткий вопрос не получает цену невиданной услуги.
"""

from __future__ import annotations

import pytest

from tests.test_pending_reply import (  # noqa: F401 — mini_prod_world, synonym_fork — фикстуры
    _PendingOracle,
    _answer,
    _pending_fields,
    _play,
    mini_prod_world,
    synonym_fork,
)
from tests.test_turn_parse_t3 import _assert_follow_up_stays_on_what_was_shown

_FAMILY = "сколько стоит оак"
_HINT = 'напишите: "все"'
_PRICE = " руб."
# Ответы «все» другими словами — правила их не понимают (`_is_price_show_all_request`); оракул выбирает пункт «все».
_IN_OTHER_WORDS = ("покажите все варианты пожалуйста", "да, давайте все", "полный список", "покажите все")
_PICK_ALL = _answer(picks=(1,))


def _types(monkeypatch, value: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PENDING_REPLY_TYPES", value, raising=False)


def _after_hint(reply, mode: str, world, monkeypatch, *, types: str = "", oracle=None, name: str = "sa", more: tuple[str, ...] = ()):
    _types(monkeypatch, types)
    return _play([_FAMILY, reply, *more], f"{name}-{mode}", mode, world, monkeypatch, oracle=oracle)


def _full_list(first: dict, second: dict) -> bool:
    return _HINT in first["text"] and _HINT not in second["text"] and second["text"].count(_PRICE) > first["text"].count(_PRICE)


@pytest.mark.parametrize("reply", _IN_OTHER_WORDS)
def test_all_in_other_words_shows_the_full_list_under_on(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off, *_ = _after_hint(reply, "off", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _PICK_ALL}))
    assert not _full_list(off[0], off[1]), off[1]["text"]  # как было: правила не поняли
    on, states, rows, oracle = _after_hint(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _PICK_ALL}))
    assert _full_list(on[0], on[1]), on[1]["text"]
    # та же выдача, что у ответа «все» правилами
    by_rules, *_ = _after_hint("все", "off", mini_prod_world, monkeypatch, name="sa-rules")
    assert on[1]["text"] == by_rules[1]["text"]
    assert (states[1].get("_expects") or {}).get("type") != "show_all"
    assert len(oracle.calls) == 1
    assert (rows[1]["tu_pending_outcome"], rows[1]["tu_pending_src"]) == ("pick", "llm")
    [entry] = [entry for entry in rows[1]["choose"] if entry["task"] == "pending_reply"]
    assert not entry.get("shadow")  # решение хода, в его бюджете


def test_yes_from_the_llm_also_shows_all(mini_prod_world, monkeypatch):  # noqa: F811
    reply = "да, давайте все"
    on, _states, rows, _o = _after_hint(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("yes")}), name="sa-yes")
    assert _full_list(on[0], on[1]), on[1]["text"]
    assert (rows[1]["tu_pending_outcome"], rows[1]["tu_pending_src"]) == ("yes", "llm")


@pytest.mark.parametrize(
    ("reply", "llm"),
    [
        ("покажите все варианты пожалуйста", _answer("new_topic")),
        ("покажите все варианты пожалуйста", _answer("none")),
        ("покажите все варианты пожалуйста", ConnectionError("ollama лежит")),  # сбой — правила
        ("покажите все варианты пожалуйста", "не json"),  # мусор — правила
    ],
    ids=["new_topic", "none", "down", "invalid"],
)
def test_anything_but_all_keeps_the_rules_answer(reply, llm, mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _PendingOracle(exc=llm) if isinstance(llm, BaseException) else _PendingOracle({reply: llm})
    off, off_states, *_ = _after_hint(reply, "off", mini_prod_world, monkeypatch, oracle=_PendingOracle(), name="sa-keep")
    on, on_states, _rows, _o = _after_hint(reply, "on", mini_prod_world, monkeypatch, oracle=oracle, name="sa-keep")
    assert on == off and on_states == off_states


def test_no_is_not_the_full_list(mini_prod_world, monkeypatch):  # noqa: F811
    reply = "не надо, спасибо"
    on, _states, rows, _o = _after_hint(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("no")}), name="sa-no")
    assert not _full_list(on[0], on[1]) and not on[1]["handoff"], on[1]["text"]


@pytest.mark.parametrize("types", ["", "operator_offer", "show_all_typo"])
def test_on_decides_every_type_whatever_the_list_says(types, mini_prod_world, monkeypatch):  # noqa: F811
    """Список `pending_reply_types` одна функция не читает: под `on` решаются все типы разом (решение 16.10)."""

    reply = "покажите все варианты пожалуйста"
    on, _states, _rows, _o = _after_hint(reply, "on", mini_prod_world, monkeypatch, types=types, oracle=_PendingOracle({reply: _PICK_ALL}), name=f"sa-list-{types}")
    assert _full_list(on[0], on[1]), on[1]["text"]


@pytest.mark.parametrize("reply", ["все", "покажи все", "Все варианты!"])
def test_what_the_rules_read_as_all_does_not_call_the_llm(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off, off_states, *_ = _after_hint(reply, "off", mini_prod_world, monkeypatch, oracle=_PendingOracle(), name="sa-bare")
    on, on_states, rows, oracle = _after_hint(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("no")}), name="sa-bare")
    assert oracle.calls == [] and on == off and on_states == off_states
    assert _full_list(on[0], on[1]), on[1]["text"]
    assert (rows[1]["tu_pending_type"], rows[1]["tu_pending_outcome"], rows[1]["tu_pending_src"]) == ("show_all", "pick", "fast")


@pytest.mark.parametrize("follow_up", ["а сколько?", "а цена?"])
def test_after_the_full_list_a_short_question_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    reply = "покажите все варианты пожалуйста"
    records, _states, _rows, _o = _after_hint(
        reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _PICK_ALL}), name="sa-two", more=(follow_up,)
    )
    assert _full_list(records[0], records[1]), records[1]["text"]
    _assert_follow_up_stays_on_what_was_shown(records[1], records[2])
