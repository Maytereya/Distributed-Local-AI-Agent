"""Отчёт `evalbench.list_readers` — расхождения «список ли» у понимания и корзины (ход (б), консилиум 20.10 §1)."""

from __future__ import annotations

from evalbench import list_readers


def test_divergence_is_listed_by_rule_and_id(monkeypatch):
    table = {"а": ("list_split", True, True), "б": ("segment_unrecognized", False, True), "в": ("not_list", False, False)}
    monkeypatch.setattr(list_readers, "verdicts", lambda question, rows: table[question])
    turns = [
        {"id": "g1", "label": "PRICE", "question": "а"},
        {"id": "g2", "label": "TEST_ASSIST", "question": "б"},
        {"id": "g3", "label": "PRICE", "question": "в"},
        {"id": "g4", "label": "ADDRESS", "question": "а"},
    ]
    m = list_readers.measure(turns, [])
    assert m["n"] == 3
    assert dict(m["diverge"]) == {"segment_unrecognized:only_basket": ["g2"]}
    assert "g2" in list_readers.render("x", m)
