"""Ответ на висящий вопрос бота: решение LLM исполняется по типу вопроса (D2.3, `pending_reply = on`,
`pending_reply_types`; решение владельца 15.10 — асимметрия безопасности по типу).

Аудит 14.10, пункт 5: на «Перевести на оператора?» ответ «конечно, переведите» правила не понимают — оффер снимается
без перевода. Под `on` и типе `operator_offer` в `pending_reply_types` LLM решает в ходе (из его бюджета) и может только
ПОДНЯТЬ ответ до «да» — перевод (безопасное направление); «нет», «пауза», «другая тема», сбой — ответ правил.

Инварианты класса (через обработчик эндпоинта, мини-прод, оракул LLM):
- «да» не голым якорем → перевод на оператора; тот же диалог под off — без перевода (как было);
- «нет» от LLM, сбой LLM, тип не в списке, пустой список — ответы и состояние как под off;
- голый якорь и «да» правил — без вызова LLM; срочное — раньше всего;
- вызов LLM — в бюджете хода (не тень); телеметрия `tu_pending_acted`;
- два хода: после перевода следующий ответ не про оффер.
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
    _pending_fields,
    _play,
    _prepared,
    mini_prod_world,
    synonym_fork,
)

_YES = "конечно, переведите"
_MANUAL = "Соединяю с оператором"


def _types(monkeypatch, value: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PENDING_REPLY_TYPES", value, raising=False)


def _offer(reply: str, mode: str, world, monkeypatch, *, types: str = "operator_offer", oracle=None, name: str = "act"):
    _types(monkeypatch, types)
    return _play([_STOPLIST_OFFER, reply], f"{name}-{mode}", mode, world, monkeypatch, before=_offer_after_first_turn(), oracle=oracle)


def test_yes_in_other_words_is_transferred_under_on(mini_prod_world, monkeypatch):  # noqa: F811
    off, *_ = _offer(_YES, "off", mini_prod_world, monkeypatch, oracle=_PendingOracle({_YES: _answer("yes")}))
    assert not off[1]["handoff"]  # как было: правила «конечно, переведите» не поняли
    on, _states, rows, oracle = _offer(_YES, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({_YES: _answer("yes")}))
    assert on[1]["handoff"] and _MANUAL in on[1]["text"], on[1]
    assert len(oracle.calls) == 1
    assert rows[1]["tu_pending_outcome"] == "yes" and rows[1]["tu_pending_acted"] == "yes"
    [entry] = [entry for entry in rows[1]["choose"] if entry["task"] == "pending_reply"]
    assert not entry.get("shadow")  # решение хода, в его бюджете


@pytest.mark.parametrize(
    ("reply", "llm"),
    [
        ("нет, спасибо, сама разберусь", _answer("no")),  # «нет» LLM ответ правил не меняет
        (_YES, _answer("pause")),
        (_YES, ConnectionError("ollama лежит")),  # сбой — правила
    ],
    ids=["no", "pause", "down"],
)
def test_anything_but_yes_keeps_the_rules_answer(reply, llm, mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _PendingOracle(exc=llm) if isinstance(llm, BaseException) else _PendingOracle({reply: llm})
    off, off_states, *_ = _offer(reply, "off", mini_prod_world, monkeypatch, oracle=_PendingOracle(), name="keep")
    on, on_states, rows, _o = _offer(reply, "on", mini_prod_world, monkeypatch, oracle=oracle, name="keep")
    assert on == off and on_states == off_states
    assert "tu_pending_acted" not in rows[1]


@pytest.mark.parametrize("types", ["", "catalog_confirm", "operator_offer_typo"])
def test_on_without_the_type_in_the_list_is_the_shadow(types, mini_prod_world, monkeypatch):  # noqa: F811
    off, *_ = _offer(_YES, "off", mini_prod_world, monkeypatch, types=types, oracle=_PendingOracle(), name=f"list-{types}")
    on, _states, rows, _o = _offer(_YES, "on", mini_prod_world, monkeypatch, types=types, oracle=_PendingOracle({_YES: _answer("yes")}), name=f"list-{types}")
    assert on == off and "tu_pending_acted" not in rows[1]


@pytest.mark.parametrize("reply", ["да", "переведите на оператора"])
def test_a_bare_anchor_or_a_rules_yes_does_not_call_the_llm(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on, _states, rows, oracle = _offer(reply, "on", mini_prod_world, monkeypatch, oracle=_PendingOracle({reply: _answer("no")}), name="bare")
    assert oracle.calls == [] and on[1]["handoff"]
    # «да» — голый якорь; явный запрос оператора перехватывает предпроверка раньше точки решения
    assert _pending_fields(rows[1]).get("tu_pending_outcome") in ("fast_rules", None)


def test_urgent_comes_before_the_decision(mini_prod_world, monkeypatch):  # noqa: F811
    _types(monkeypatch, "operator_offer")
    reply = "нет, мне плохо, задыхаюсь"
    before = _prepared({"_operator_offer_pending": True}, _expects(_OFFER_QUESTION))
    records, _states, _rows, oracle = _play([reply], "act-urgent", "on", mini_prod_world, monkeypatch, before=before, oracle=_PendingOracle({reply: _answer("no")}))
    assert records[0]["handoff"] and oracle.calls == []


def test_after_the_transfer_the_next_turn_is_not_about_the_offer(mini_prod_world, monkeypatch):  # noqa: F811
    _types(monkeypatch, "operator_offer")
    records, states, _rows, _oracle = _play(
        [_STOPLIST_OFFER, _YES, "сколько стоит ферритин"], "act-two", "on", mini_prod_world, monkeypatch,
        before=_offer_after_first_turn(), oracle=_PendingOracle({_YES: _answer("yes")}),
    )
    assert records[1]["handoff"]
    assert not states[1].get("_operator_offer_pending")
    assert "Ферритин" in records[2]["text"] and not records[2]["handoff"], records[2]
