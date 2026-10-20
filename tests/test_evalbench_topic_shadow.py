"""Отчёт `evalbench.topic_shadow` — матрица «правило × LLM» на синтетических ходах (STEP-2026-10-21-B-TOPIC-SHADOW)."""

from __future__ import annotations

from evalbench import topic_shadow


def _t(i: str, rule: str, rule_label: str, llm: str, final: str | None = None, cat: str = "цена") -> dict:
    tel = {"nlu_src": "rule", "nlu_rule": rule, "tu_topic_shadow_rule": rule_label, "tu_topic_shadow": llm, "tu_label_final": final or rule_label}
    return {"id": i, "category": cat, "telemetry": tel}


def test_matrix_counts_disagreements_and_accuracy_against_gold():
    turns = [
        _t("a", "rule_price", "PRICE", "PRICE"),
        _t("b", "rule_test_assist", "TEST_ASSIST", "PRICE", final="PRICE"),
        _t("c", "rule_test_assist", "TEST_ASSIST", "TEST_ASSIST"),
        _t("d", "rule_price", "PRICE", "unavailable"),
        {"id": "e", "telemetry": {"nlu_src": "llm"}},
    ]
    gold = {"a": {"label": "PRICE"}, "b": {"label": "PRICE", "disputed": True}, "c": {"label": "PRICE"}}
    m = topic_shadow.measure(turns, gold)

    assert (m["n"], m["decided"], dict(m["failures"])) == (4, 3, {"unavailable": 1})
    assert m["matrix"] == {("TEST_ASSIST", "PRICE"): 1}
    assert m["by_rule"]["rule_test_assist"] == {"n": 2, "agree": 1}
    assert m["acc"]["all"] == {"n": 3, "rule_ok": 1, "llm_ok": 2, "final_ok": 2}
    assert m["acc"]["clear"] == {"n": 2, "rule_ok": 1, "llm_ok": 1, "final_ok": 1}
    text = topic_shadow.render("x", m)
    assert "TEST_ASSIST → PRICE | 1" in text
