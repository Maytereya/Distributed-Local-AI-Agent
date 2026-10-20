"""Ход (б), телеметрия до этапа 2 (консилиум 20.10 §4, STEP-2026-10-21-B-TELEMETRY).

До переноса читателей цены на один объект понимания нужно видеть на проде: с какой меткой ход вышел из NLU и с какой
ответил (`tu_label_nlu` / `tu_label_final`), откуда понимание взяло услуги (`tu_understanding_src`), какой гард условия
решил (`price_guard` — полный набор значений, `none` тоже пишется), каков вердикт суда П2 (`tu_p2_verdict`) и сколько
раз читатель цены работал без разбора этой реплики (`input_not_reply` / `input_no_parse`).

Инвариант класса: у каждого хода, дошедшего до NLU, есть `tu_label_nlu` и `tu_label_final` (= `label`); у каждого
ценового хода — `price_guard` из закрытого набора и `tu_p2_verdict`; у ценового хода с пониманием —
`tu_understanding_src` ∈ {object, rules}; уточнение, которого нет в прайсе («по ОМС»), не пишет `price_guard = none`.
Через обработчик эндпоинта, LLM — подделка настоящего типа. Ответы от записи не меняются (гард — эталоны и replay).
"""

from __future__ import annotations

import json

from tests.test_llm_primary_rules_fallback_class import _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_LLM_ANSWER = json.dumps({"label": "OTHER", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
_PRICE_GUARDS = {"llm", "dictionary", "llm_bypass", "k3", "none"}
_REPLIES = (
    "сколько стоит ферритин",
    "сколько стоит ферритин и глюкоза",
    "прием терапевта по омс сколько стоит",
    "хочу записаться к кардиологу",
    "какой у вас адрес",
    "спасибо большое",
)


def _tel(world, monkeypatch, reply: str) -> dict:
    return _turn(world, monkeypatch, reply, engine="legacy_v2", fallback="off", answer=_LLM_ANSWER)["telemetry"]


def test_every_nlu_turn_has_label_after_nlu_and_final(mini_prod_world, monkeypatch):  # noqa: F811
    for reply in _REPLIES:
        tel = _tel(mini_prod_world, monkeypatch, reply)
        if "nlu_src" not in tel:
            continue  # ход решён до NLU
        assert tel.get("tu_label_nlu"), (reply, tel)
        assert tel.get("tu_label_final") == tel.get("label"), (reply, tel)


def test_every_price_turn_says_which_guard_and_verdict(mini_prod_world, monkeypatch):  # noqa: F811
    price_turns = 0
    for reply in _REPLIES:
        tel = _tel(mini_prod_world, monkeypatch, reply)
        if tel.get("tu_label_final") != "PRICE":
            assert "tu_p2_verdict" not in tel, (reply, tel)
            continue
        price_turns += 1
        assert tel.get("price_guard") in _PRICE_GUARDS, (reply, tel)
        assert tel.get("tu_p2_verdict"), (reply, tel)
        if "tu_list" in tel:  # понимание построено
            assert tel.get("tu_understanding_src") in {"object", "rules"}, (reply, tel)
    assert price_turns >= 2, "корпус вырожден: ценовых ходов почти нет"


def test_qualifier_absent_from_price_is_not_guard_none(mini_prod_world, monkeypatch):  # noqa: F811
    """«по ОМС» — уточнение, которого в прайсе нет: решал гард условия (словарь, суд LLM или K3), не «гарда не было»."""

    tel = _tel(mini_prod_world, monkeypatch, "прием терапевта по омс сколько стоит")
    if tel.get("tu_label_final") == "PRICE":
        assert tel.get("price_guard") in _PRICE_GUARDS - {"none"}, tel


def test_derived_fields_only_on_price_turns():
    from messengers_router import turn_telemetry

    assert turn_telemetry._derived({"tu_label_final": "ADDRESS"}) == {}
    assert turn_telemetry._derived({"tu_label_final": "PRICE"}) == {"price_guard": "none", "tu_p2_verdict": "none"}
    assert turn_telemetry._derived({"tu_label_final": "PRICE", "price_guard": "k3", "strong_phrase_guard": "rows"}) == {
        "tu_p2_verdict": "rows"
    }
