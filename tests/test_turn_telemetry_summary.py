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
