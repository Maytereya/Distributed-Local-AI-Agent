"""Класс: отказ лимита решений хода на выборе строки цены — уточнение, не ответ правил по буквам (выключатель
`price_budget_clarify` снят 29.10 после недели `on` на проде, STEP-2026-10-29-SWITCH-REMOVAL; тест пути off удалён).

Решение владельца 22.10 (DESIGN-2026-10-22-TURN-DECISION-BUDGET, вариант А): g0577 «Со скольки лет берут анализ крови из вены?» —
суд условия получил отказ лимита решений хода, и ответ правил дал цену чужой строки «Анализ крови на аминокислоты».

Инварианты:
- лимит исчерпан + путь ответа о цене (`budget_clarify=True`) — строк нет: названия целым словом на выбор
  (`unavailable` + `names`) или «не нашёл» (`none_fit`, причина `budget`);
- сбой самой модели (лимит не исчерпан) — как было (ответ правил при «недоступна»);
- суд записи / переспроса (`budget_clarify` не передан) — как было.
"""

from __future__ import annotations

import pytest

from messengers_router import llm_runtime
from messengers_router.choice import Choice, Status
from tests.test_price_candidates_by_meaning_class import ROWS, _Oracle, _run

_Q = "сколько стоит приём терапевта первичный по омс"


def _ask(monkeypatch, *, exhausted: bool, budget_clarify: bool = True):
    monkeypatch.setattr(llm_runtime, "turn_budget_exhausted", lambda: exhausted)
    oracle = _Oracle(Choice(Status.UNAVAILABLE, reason="budget" if exhausted else ""))
    return _run(monkeypatch, "on", _Q, oracle, candidates=[ROWS[-1]], shown=[ROWS[-1]], budget_clarify=budget_clarify)


def test_on_budget_denial_gives_no_rows(monkeypatch) -> None:
    selection = _ask(monkeypatch, exhausted=True)
    assert selection is not None and not selection.rows
    assert (selection.status, selection.reason) == ("none_fit", "budget") or (selection.status == "unavailable" and selection.names)


@pytest.mark.parametrize(("exhausted", "budget_clarify"), [(False, True), (True, False)])
def test_otherwise_as_before(exhausted, budget_clarify, monkeypatch) -> None:
    before = _ask(monkeypatch, exhausted=False, budget_clarify=False)
    after = _ask(monkeypatch, exhausted=exhausted, budget_clarify=budget_clarify)
    assert (after is None) == (before is None)
    if after is not None:
        assert (after.status, after.reason, after.names) == (before.status, before.reason, before.names)
