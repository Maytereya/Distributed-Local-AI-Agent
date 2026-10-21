"""Ход (б), этап 2: читатели понимания хода берут решение из объекта, а не решают заново по буквам (STEP-2026-10-21-B-READERS).

Дизайн 19.10 §3 (таблица политик хода), итог ночи 21.10 §2.10. Инвариант класса — под `turn_understanding_one = on`, когда
понимание этой реплики есть и действует (`turn_understanding_price = on`):
- U6 «названа ли услуга» — одно решение хода в производителе понимания (`TurnUnderstanding.names_service`: упоминание
  или прежнее определение `names_service`, один вызов за ход); планировщик цены читает поле и сам не решает. Только «нет
  упоминания» не годится — replay 21.10: 5 потерь верной цены из 1068 ходов (вопрос владельцу);
- U1 (подстановка каталога) и U2 (грундер) при исходе `one_strong` (одна услуга, сильная по данным МИС, одно название
  прайса) берут это название из понимания и не ищут каталог по буквам реплики; слабая фраза, несколько названий, список — как было.
  У подбора анализов точное совпадение — довод П1 «это цена»: U1 для TEST_ASSIST — только когда тему решила LLM (`topic_by_llm`,
  свойство вопроса для П1 решает она же), иначе П1 без LLM отвечал бы ценой на вопрос о сроке (replay dev 21.10, g0168);
- понимания этой реплики нет (вход — не реплика хода, тень, выключатель off) — прежний путь.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import entity_grounder, router, turn_understanding
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


# --- U1 / U2: исход one_strong — название из понимания, не каталог по буквам ----------------------------------------------

_CHOSEN = "Посев на грибы рода Candida (чувствительность к антимикотическим препаратам)"
_REPLY = "Посев на кандиду сколько стоит"


def _strong(chosen: str | None = _CHOSEN, *, strength: str = "strong", reply: str = _REPLY) -> TurnUnderstanding:
    mention = ServiceMention(phrase="посев на кандиду", candidates=(chosen,) if chosen else (), strength=strength, chosen=chosen, source="parse")
    return TurnUnderstanding(services=(mention,), services_rule="parse_strong_whole", facet="price", reply=reply)


class _Catalog:
    """Шпион каталога по буквам: что спросили — то записано; отвечает «точно — Чужая услуга»."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def match_catalog_service(self, query, **_kw):
        self.asked.append(query)
        return {"status": "exact", "canonical": "Чужая услуга", "query": query}

    async def match_catalog_doctor(self, *_a, **_kw):
        return None


def _inject(label: str = "PRICE", flags: frozenset[str] = frozenset()) -> tuple[RouteDecision, _Catalog]:
    catalog = _Catalog()
    decision = RouteDecision(label=label, confidence=0.9, entities={}, flags=set(flags))
    out = asyncio.run(router._inject_catalog_candidates(decision, user_text=_REPLY, state=SessionState(session_id="t"), services=catalog))
    return out, catalog


@pytest.mark.parametrize("label, flags", [("PRICE", frozenset()), ("TEST_ASSIST", frozenset({"topic_by_llm"}))])
def test_u1_one_strong_takes_the_name_from_the_understanding(switches, understood, label, flags):
    switches(one="on")
    understood(_strong())
    decision, catalog = _inject(label, flags)
    assert decision.entities.get("service_name") == _CHOSEN
    assert {"catalog_service_exact", "understanding_service_chosen"} <= set(decision.flags)
    assert catalog.asked == []


@pytest.mark.parametrize(
    "one, tu",
    [
        ("off", _strong()),  # выключатель off — как было
        ("on", _strong(strength="weak", chosen=None)),  # слабая фраза — выбор строки, каталог как было
        ("on", _strong(chosen=None)),  # сильная, но названий несколько — не выбираем за пациента
        ("on", _strong(reply="другая реплика")),  # понимание не этой реплики
    ],
)
def test_u1_otherwise_the_catalog_decides_as_before(switches, understood, one, tu):
    switches(one=one)
    understood(tu)
    decision, catalog = _inject()
    assert catalog.asked == [_REPLY]
    assert decision.entities.get("service_name") == "Чужая услуга"


def test_u1_test_assist_by_rule_does_not_widen_p1(switches, understood):
    """Тему решило правило — довод П1 даёт, как было, каталог по буквам (g0168: вопрос о сроке не становится ценой)."""

    switches(one="on")
    understood(_strong())
    decision, catalog = _inject("TEST_ASSIST")
    assert catalog.asked == [_REPLY] and "understanding_service_chosen" not in decision.flags


class _Grounding(_Catalog):
    async def resolve_doctor_name(self, name):
        return None


def _ground(monkeypatch, label: str = "PRICE") -> tuple[dict, set, list[str]]:
    resolved: list[str] = []

    def resolver(text, current_service_name=""):
        resolved.append(text)
        return "Чужая услуга"

    monkeypatch.setattr(entity_grounder, "resolve_price_service_name_from_catalog", resolver)
    decision = RouteDecision(label=label, confidence=0.9, entities={"service_name": "кандида"}, flags=set())
    out = asyncio.run(entity_grounder.ground_decision_entities(decision, _REPLY, SessionState(session_id="t"), _Grounding()))
    return out.entities, out.flags, resolved


def test_u2_one_strong_keeps_the_name_of_the_understanding(switches, understood, monkeypatch):
    switches(one="on")
    understood(_strong())
    entities, flags, resolved = _ground(monkeypatch)
    assert entities.get("service_name") == _CHOSEN and "entity_service_name_from_understanding" in flags
    assert resolved == []


@pytest.mark.parametrize("one, tu", [("off", _strong()), ("on", _strong(strength="weak", chosen=None)), ("on", _strong(chosen=None))])
def test_u2_otherwise_grounds_by_letters_as_before(switches, understood, monkeypatch, one, tu):
    switches(one=one)
    understood(tu)
    entities, flags, resolved = _ground(monkeypatch)
    assert resolved == [_REPLY] and "entity_service_name_from_understanding" not in flags


def test_u2_test_assist_grounds_as_before(switches, understood, monkeypatch):
    switches(one="on")
    understood(_strong())
    entities, flags, _resolved = _ground(monkeypatch, label="TEST_ASSIST")
    assert "entity_service_name_from_understanding" not in flags
