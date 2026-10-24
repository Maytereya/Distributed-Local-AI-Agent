"""Сводка телеметрии хода за окно (`scripts/turn_telemetry_summary.py`): пороги «годен» нагрузочной проверки.

Инварианты: в счёт идут ходы окна (`ts ≥ since`), не строки после ответа (`kind`) и не решения тени; вердикт «не годен»
называет каждый нарушенный порог; без ходов — код 2.
"""

from __future__ import annotations

import json

from scripts import turn_telemetry_summary as S


def _turn(ts: str, *, ms: int = 2000, choose=(), llm=(), parse: str = "") -> dict:
    row = {"v": 1, "ts": ts, "ms": ms, "choose": list(choose), "llm": list(llm)}
    if parse:
        row["tu_parse_status"] = parse
    return row


def _write(tmp_path, rows) -> str:
    (tmp_path / "turns_20261015.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return str(tmp_path)


def test_a_quiet_window_is_fit(tmp_path, capsys):
    ok_choice = {"task": "turn_phrases", "status": "picked", "options": 3, "ms": 900}
    rows = [_turn("2026-10-15T08:00:00", choose=[ok_choice], llm=[{"site": "choose_turn_phrases", "outcome": "ok", "ms": 900}], parse="picked") for _ in range(10)]
    rows.append(_turn("2026-10-15T07:00:00", choose=[{"task": "turn_phrases", "status": "unavailable", "options": 3, "ms": 9000}]))  # до окна
    rows.append({"v": 1, "kind": "pending_shadow", "ts": "2026-10-15T08:01:00", "llm": [{"outcome": "timeout", "err": "x"}]})  # строка после ответа
    assert S.main(["--since", "2026-10-15T07:42:26", "--journals", _write(tmp_path, rows)]) == 0
    out = capsys.readouterr().out
    assert "ходов 10" in out and "ВЕРДИКТ: годен" in out and "запасной путь правил 0 из 10" in out


def test_every_broken_threshold_is_named(tmp_path, capsys):
    rows = [
        _turn("2026-10-15T09:00:00", choose=[{"task": "price_basket", "status": "unavailable", "options": 2, "ms": 5000}],
              llm=[{"site": "choose_price_basket", "outcome": "timeout", "ms": 5000, "queue_ms": 1500, "err": "ETIMEDOUT"},
                   {"site": "choose_x", "outcome": "budget", "ms": 0}, {"site": "choose_y", "outcome": "circuit_open", "ms": 0}],
              parse="unavailable"),
        _turn("2026-10-15T09:01:00", choose=[{"task": "turn_phrases", "status": "picked", "options": 2, "ms": 800, "shadow": True}]),
    ]
    assert S.main(["--since", "2026-10-15T08:00:00", "--journals", _write(tmp_path, rows)]) == 1
    out = capsys.readouterr().out
    for reason in ("UNAVAILABLE+INVALID ≥ 2%", "p95 выбора ≥ 4 с", "p95 queue_ms ≥ 1 с", "circuit_open", "err не пуст", "budget > 2%"):
        assert reason in out, (reason, out)
    assert "запасной путь правил 1 из 1" in out


def test_no_turns_in_the_window(tmp_path, capsys):
    assert S.main(["--since", "2026-10-16T00:00:00", "--journals", _write(tmp_path, [_turn("2026-10-15T09:00:00")])]) == 2


def test_price_turns_show_who_judged_the_qualifier(tmp_path, capsys):
    """Ценовые ходы: решения L-13 и судей уточнения П2 — счётчиками по значениям; без ценовых полей строки нет."""

    rows = [_turn("2026-10-16T01:00:00", ms=3000) for _ in range(3)]
    rows[0].update(price_policy="strong")
    rows[1].update(price_policy="not_found", strong_phrase_guard="not_found")
    rows[2].update(price_guard="llm", strong_phrase_guard="fallback")
    assert S.main(["--since", "2026-10-16T00:30:00", "--journals", _write(tmp_path, rows)]) == 0
    out = capsys.readouterr().out
    assert "цена (3 ходов, p95 мс 3000)" in out, out
    assert "strong_phrase_guard {'not_found': 1, 'fallback': 1}" in out and "price_policy {'strong': 1, 'not_found': 1}" in out
    assert "p2_llm_judge" not in out
    S.main(["--since", "2026-10-16T00:30:00", "--journals", _write(tmp_path, [_turn("2026-10-16T01:00:00")])])
    assert "цена (" not in capsys.readouterr().out


def test_summary_shows_tool_timing_and_decision_denials():
    """Ход (е), 22.10: сводка показывает время по инструментам и долю ходов с отказом лимита решений."""

    from scripts.turn_telemetry_summary import summarize

    rows = [
        {"ms": 900, "tool_ms": {"price_info": 400}, "tu_decisions_spent": 2},
        {"ms": 1200, "tool_ms": {"price_info": 700, "test_prepare": 50}, "tu_decisions_spent": 3, "tu_decisions_denied": 1},
    ]
    text = "\n".join(summarize(rows)[0])
    assert "price_info n=2" in text and "test_prepare n=1" in text
    assert "ходов с отказом лимита 1 из 2" in text
