"""Ход (б), этап 2: читатели понимания хода берут решение из объекта, а не решают заново по буквам (STEP-2026-10-21-B-READERS).

Дизайн 19.10 §3 (таблица политик хода), итог ночи 21.10 §2.10. Инвариант класса — под `turn_understanding_one = on`, когда
понимание этой реплики есть и действует (`turn_understanding_price = on`):
- U6 «названа ли услуга» — одно решение хода в производителе понимания (`TurnUnderstanding.names_service`: упоминание
  или прежнее определение `names_service`, один вызов за ход); планировщик цены читает поле и сам не решает. Только «нет
  упоминания» не годится — replay 21.10: 5 потерь верной цены из 1068 ходов (вопрос владельцу);
- понимания этой реплики нет (вход — не реплика хода, тень, выключатель off) — прежний путь.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import turn_understanding
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.planner import build_plan
from messengers_router.services import _price_select_llm
from messengers_router.turn_understanding import ServiceMention, TurnUnderstanding


@pytest.fixture
def switches(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    def set_(*, one: str = "on", price: str = "on") -> None:
        monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_ONE", one, raising=False)
        monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", price, raising=False)

    return set_


@pytest.fixture
def understood():
    tokens = []

    def put(tu: TurnUnderstanding | None) -> None:
        tokens.append(turn_understanding._CURRENT.set(tu))

    yield put
    for token in reversed(tokens):
        turn_understanding._CURRENT.reset(token)


def _mention(phrase: str) -> ServiceMention:
    return ServiceMention(phrase=phrase, candidates=(), strength="weak", chosen=None, source="rules")


def _asks_name(text: str) -> bool:
    """Планировщик цены переспросил название услуги (висящий вопрос со слотом `service_name`)."""

    memory = MemoryStore()
    state = SessionState(session_id="t")
    plan = build_plan(RouteDecision(label="PRICE", confidence=0.9, entities={}, flags=set()), state, text, memory)
    pending = memory.get_pending(state) or {}
    return not plan.steps and "service_name" in (pending.get("missing") or [])


@pytest.fixture
def letters(monkeypatch):
    """Шпион на второе определение «названа ли» по буквам."""

    calls: list[str] = []

    def spy(text: str) -> bool:
        calls.append(text)
        return True

    monkeypatch.setattr(_price_select_llm, "names_service", spy)
    return calls


# --- U6: «названа ли услуга» -------------------------------------------------------------------------------------------

def test_u6_producer_decides_named_once_mention_first(switches, understood, letters):
    switches(one="on")
    tu = turn_understanding._decide_named(TurnUnderstanding(services=(_mention("ггтп"),), reply="сколько стоит ггтп"), "сколько стоит ггтп")
    assert tu.names_service is True and letters == []
    tu = turn_understanding._decide_named(TurnUnderstanding(reply="Сколько стоит прием к оталорингологу?"), "Сколько стоит прием к оталорингологу?")
    assert tu.names_service is True and letters == ["Сколько стоит прием к оталорингологу?"]
    assert turn_understanding.current() is tu


@pytest.mark.parametrize("one, decided", [("on", True), ("off", None)])
def test_u6_observe_price_turn_writes_the_decision_only_under_one(switches, understood, letters, monkeypatch, one, decided):
    switches(one=one)
    text = "Посев на Candida показывает точный тип грибка?"

    async def built(_text):
        return TurnUnderstanding(reply=_text)

    monkeypatch.setattr(turn_understanding, "_observe_parse_on_lazy", built)
    monkeypatch.setattr(turn_understanding, "agree_telemetry", lambda: False)
    tu = asyncio.run(turn_understanding.observe_price_turn("PRICE", user_text=text, pending=None))
    assert tu is not None and tu.names_service is decided


@pytest.mark.parametrize("named, asks", [(False, True), (True, False)])
def test_u6_planner_reads_the_decision_and_does_not_decide_again(switches, understood, letters, named, asks):
    switches(one="on")
    text = "Узнать цену анализов"
    understood(TurnUnderstanding(services_rule="parse_single", facet="price", reply=text, names_service=named))
    assert _asks_name(text) is asks
    assert letters == []


@pytest.mark.parametrize(
    "one, price, reply, decided",
    [
        ("off", "on", "Узнать цену анализов", False),  # выключатель off — как было
        ("on", "shadow", "Узнать цену анализов", False),  # тень понимания — читатели не действуют
        ("on", "on", "другая реплика", False),  # вход планировщика — не реплика хода
        ("on", "on", "Узнать цену анализов", None),  # решения нет
    ],
)
def test_u6_without_decision_of_this_reply_the_old_definition_decides(switches, understood, letters, one, price, reply, decided):
    switches(one=one, price=price)
    understood(TurnUnderstanding(services_rule="parse_single", reply=reply, names_service=decided))
    assert not _asks_name("Узнать цену анализов")  # шпион прежнего определения говорит «названа»
    assert letters == ["Узнать цену анализов"]
