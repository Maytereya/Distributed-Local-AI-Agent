"""Модель очереди к слотам LLM (`evalbench.load_sim`, BUG-2026-10-27-LLM-QUEUE-GIVES-UP) — на синтетических ходах.

Ход нового пациента — длинный классификатор, начатого — короткие решения: «сейчас» короткое решение сдаётся по своей очереди
(1,5 с), пока классификаторы держат оба слота; политика кода (дедлайн 50 с, EDF, запас p95) — никто не сдаётся; ходов не больше,
чем слотов, — очереди нет ни в одной политике.
"""

from __future__ import annotations

from evalbench import load_sim

_TURNS = [[("classifier", 12.0), ("turn_spans", 1.0), ("choose_price_service", 2.0)]]


def test_short_task_gives_up_now_and_waits_under_the_deadline():
    now = load_sim.simulate(_TURNS, 4, "now", trials=5)
    code = load_sim.simulate(_TURNS, 4, "code", trials=5)
    assert now[3] > 0 and now[4] > 0  # «сейчас» — сдаются, ходы уходят на правила
    assert code[3] == 0 and code[4] == 0 and code[2] < 50.0  # ждут и укладываются в дедлайн


def test_no_queue_when_turns_fit_the_slots():
    for policy in ("now", "oracle", "code"):
        p50, _p95, mx, gave, failed = load_sim.simulate(_TURNS, 2, policy, trials=3)
        assert (gave, failed) == (0, 0) and p50 == mx == 15.0, policy


def test_deadline_cuts_when_the_queue_is_longer_than_the_deadline():
    code = load_sim.simulate(_TURNS, 20, "code", trials=3)
    assert code[4] > 0 and code[2] <= 50.0  # дедлайн срабатывает — ход не дольше дедлайна
