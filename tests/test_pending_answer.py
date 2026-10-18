"""Одна функция ответа на висящий вопрос (`messengers_router.pending_answer`) — ядро без подключения к ходу.

Дизайн `docs/DESIGN_2026-10-18_one_pending_answer.md` §2; консилиум 18.10. Инварианты класса:
- сужение без LLM — только якоря из правил («да», «нет», …) и дословный пункт / номер; якоря — только у вопросов
  «да/нет» (у выборов «не нужно» значит своё);
- «якорь + хвост»: хвост режет код; «да» с хвостом на вопросе о цене — не «да» (хвост — обычным путём), на оффере
  оператора — «да», на отмене оформления — решает модель;
- вето отказа после модели: «да» модели на «не отменяйте …» — «нет» (отмена оформления), на прочих — «ничего»;
- «да, второй» от модели — пункт; номер вне списка, мусор, сбой — запасной путь (`fallback`), не исход;
- решение, не закрывшее ход, возвращает ходу одно решение LLM (Q5).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import llm_runtime, pending_answer as pa
from messengers_router.choice import Choice, Status

BRANCHES = ("пр. Ленина, 5", "ул. Победы, 83", "список врачей", "любой филиал")


class _Oracle:
    """Подделка `choose()` настоящего типа: отвечает заданным выбором и запоминает вызовы."""

    def __init__(self, choice: Choice | BaseException):
        self.choice = choice
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    async def __call__(self, task, question, options, *, asked="", **_kw):
        self.calls.append((question, tuple(o.text for o in options), asked))
        if isinstance(self.choice, BaseException):
            raise self.choice
        return self.choice


def _run(monkeypatch, oracle, qtype, text, options=(), question="Вопрос бота?"):
    monkeypatch.setattr(pa, "choose", oracle)
    pa.reset()
    return asyncio.run(pa.answer(qtype, question, tuple(options), text))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("да", (pa.Outcome.YES, None, False)),
        ("Нет.", (pa.Outcome.NO, None, False)),
        ("нет, мне ферритин", (pa.Outcome.NO, "мне ферритин", False)),
        ("конечно, переведите", (pa.Outcome.YES, "переведите", False)),
        ("да но для ребенка", (pa.Outcome.YES, "для ребенка", True)),
        ("да, но для ребенка", (pa.Outcome.YES, "для ребенка", True)),
        ("нет, а глюкоза сколько?", (pa.Outcome.NO, "глюкоза сколько?", True)),
        ("да нет", None),  # не якорь — решает модель
        ("ладно, не надо", None),  # «да» с отказом в хвосте — не якорь
        ("давайте во второй", None),  # нет разделителя
        ("наркоз не нужен", None),
    ],
)
def test_split_anchor(text, expected):
    assert pa.split_anchor(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1", BRANCHES[0]), ("2.", BRANCHES[1]), ("№3", BRANCHES[2]), ("5", None), ("ПР ЛЕНИНА 5", BRANCHES[0]), ("Ленина", None)],
)
def test_exact_pick(text, expected):
    assert pa.exact_pick(text, BRANCHES) == expected


def test_anchor_decides_without_llm_on_yes_no_questions(monkeypatch):
    oracle = _Oracle(Choice(Status.OUTCOME, outcome="no"))
    monkeypatch.setattr(pa, "choose", oracle)

    async def turn():
        pa.reset()
        found = await pa.answer("operator_offer", "Перевести на оператора?", (), "конечно, переведите")
        return found, pa.current("operator_offer", "конечно, переведите"), pa.current("catalog_confirm"), pa.current(text="другое")

    found, same, other_type, other_text = asyncio.run(turn())
    assert (found.outcome, found.source) == (pa.Outcome.YES, pa.Source.FAST) and not oracle.calls
    assert same is found and other_type is None and other_text is None


@pytest.mark.parametrize(
    ("text", "negated"),
    [("не отменяйте", True), ("нет-нет, продолжаем", True), ("да, давайте отменим", False), ("отменяйте", False), ("не против", False)],
)
def test_negation_is_the_first_branch_of_the_refusal_rule(text, negated):
    from messengers_router.policies import _normalize_confirmation_text

    assert pa._negated(_normalize_confirmation_text(text)) is negated


@pytest.mark.parametrize("qtype", ["catalog_confirm", "secondary_offer", "show_all"])
def test_yes_with_tail_on_price_questions_is_not_yes(qtype, monkeypatch):
    oracle = _Oracle(Choice(Status.OUTCOME, outcome="yes"))
    found = _run(monkeypatch, oracle, qtype, "да, но для ребёнка")
    assert found.outcome is pa.Outcome.NONE and found.rest == "для ребёнка" and not oracle.calls
    assert not found.closes


def test_yes_with_comma_tail_on_catalog_asks_the_model(monkeypatch):
    oracle = _Oracle(Choice(Status.OUTCOME, outcome="yes"))
    found = _run(monkeypatch, oracle, "catalog_confirm", "да, верно, его")
    assert (found.outcome, found.rest, found.source) == (pa.Outcome.YES, None, pa.Source.LLM) and len(oracle.calls) == 1


def test_operator_offer_yes_with_tail_is_yes_without_tail(monkeypatch):
    oracle = _Oracle(Choice(Status.OUTCOME, outcome="no"))
    found = _run(monkeypatch, oracle, "operator_offer", "да, пожалуйста, соедините")
    assert (found.outcome, found.rest, found.source) == (pa.Outcome.YES, None, pa.Source.FAST) and found.closes and not oracle.calls


def test_no_with_tail_keeps_the_tail(monkeypatch):
    found = _run(monkeypatch, _Oracle(Choice(Status.UNAVAILABLE)), "catalog_confirm", "нет, мне ферритин")
    assert (found.outcome, found.rest, found.source) == (pa.Outcome.NO, "мне ферритин", pa.Source.FAST)


def test_abort_yes_with_tail_asks_the_model(monkeypatch):
    oracle = _Oracle(Choice(Status.OUTCOME, outcome="yes"))
    found = _run(monkeypatch, oracle, "appointment_abort", "да, давайте отменим")
    assert found.outcome is pa.Outcome.YES and found.source is pa.Source.LLM and len(oracle.calls) == 1


@pytest.mark.parametrize(
    ("qtype", "expected"),
    [("appointment_abort", pa.Outcome.NO), ("catalog_confirm", pa.Outcome.NONE), ("operator_offer", pa.Outcome.NONE)],
)
def test_refusal_vetoes_model_yes(qtype, expected, monkeypatch):
    found = _run(monkeypatch, _Oracle(Choice(Status.OUTCOME, outcome="yes")), qtype, "не отменяйте, просто уточните адрес")
    assert found.outcome is expected


@pytest.mark.parametrize("text", ["не нужно", "нет", "да"])
def test_pick_questions_have_no_anchors(text, monkeypatch):
    oracle = _Oracle(Choice(Status.PICKED, ids=("2",)))
    found = _run(monkeypatch, oracle, "anesthesia", text, options=("с наркозом", "без наркоза"))
    assert found.source is pa.Source.LLM and found.pick == "без наркоза" and len(oracle.calls) == 1


def test_number_on_branch_question_is_a_pick_without_llm(monkeypatch):
    oracle = _Oracle(Choice(Status.UNAVAILABLE))
    found = _run(monkeypatch, oracle, "appointment_branch", "1", options=BRANCHES)
    assert (found.outcome, found.pick, found.source) == (pa.Outcome.PICK, BRANCHES[0], pa.Source.FAST) and not oracle.calls


def test_yes_with_number_from_model_is_a_pick(monkeypatch):
    found = _run(monkeypatch, _Oracle(Choice(Status.OUTCOME, ids=("2",), outcome="yes")), "appointment_branch", "да, давайте во второй", options=BRANCHES)
    assert (found.outcome, found.pick) == (pa.Outcome.PICK, BRANCHES[1])


@pytest.mark.parametrize(
    "choice",
    [Choice(Status.UNAVAILABLE), Choice(Status.INVALID), Choice(Status.PICKED, ids=("9",)), ConnectionError("ollama лежит")],
    ids=["unavailable", "invalid", "out-of-range", "error"],
)
def test_failure_is_fallback_not_an_outcome(choice, monkeypatch):
    found = _run(monkeypatch, _Oracle(choice), "appointment_branch", "давайте во второй", options=BRANCHES)
    assert found.source is pa.Source.FALLBACK and not found.decided


def test_unknown_type_or_empty_reply_is_none(monkeypatch):
    assert _run(monkeypatch, _Oracle(Choice(Status.UNAVAILABLE)), "appointment_confirm", "да") is None
    assert _run(monkeypatch, _Oracle(Choice(Status.UNAVAILABLE)), "operator_offer", "  ") is None


def test_decision_that_does_not_close_the_turn_is_refunded(monkeypatch):
    token = llm_runtime.begin_turn_budget()
    try:
        llm_runtime._TURN_BUDGET.get().calls = 1  # решение функции уже посчитано бюджетом
        found = _run(monkeypatch, _Oracle(Choice(Status.OUTCOME, outcome="new_topic")), "appointment_branch", "а сколько стоит приём?", options=BRANCHES)
        assert found.outcome is pa.Outcome.NEW_TOPIC and llm_runtime._TURN_BUDGET.get().calls == 0
        llm_runtime._TURN_BUDGET.get().calls = 1
        found = _run(monkeypatch, _Oracle(Choice(Status.PICKED, ids=("2",))), "appointment_branch", "давайте во второй", options=BRANCHES)
        assert found.closes and llm_runtime._TURN_BUDGET.get().calls == 1
    finally:
        llm_runtime.end_turn_budget(token)
