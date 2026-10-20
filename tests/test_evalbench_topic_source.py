"""Замер «кто решил тему хода» (`evalbench.topic_source`, план хода (б) 20.10) — на синтетических ходах, без данных.

Инвариант: слой хода — ровно один из `rule | llm | pending | unrecorded`; ход правила — только без вызова
LLM-классификатора; имя правила — флаги ветки (`rule_*` и флаги веток без префикса), без `rule_none` и спасения цены;
переписывание метки после NLU считается отдельно и слой не меняет. Отчёт — только счётчики, без реплик.
"""

from __future__ import annotations

import json

from evalbench import topic_source as ts


def _turn(tid: str, *, source="llm_primary", nlu_llm=False, flags=(), label="PRICE", category="цена", outcome="answer"):
    return {
        "id": tid, "category": category, "label": label, "outcome": outcome, "flags": list(flags), "price_pairs": [],
        "question": f"секретная реплика {tid}",
        "telemetry": {"decision_source": source, "nlu_llm": nlu_llm, "rule": next((f for f in flags if f.startswith("rule_")), None)},
    }


_TURNS = [
    _turn("a", flags=["rule_price", "pii_phone", "multi_intent_detected"]),
    _turn("b", flags=["rule_test_assist", "test_assist_catalog_hit_is_price"], category="результаты"),
    _turn("c", nlu_llm=True, flags=["rule_none"], category="прочее", outcome="handoff"),
    _turn("d", source="router_precheck", category="запись"),
    _turn("e", flags=["urgent"], label="URGENT", category="медвопрос"),
    _turn("f", flags=[], label=None, category="прочее"),
]


def test_each_turn_gets_one_layer():
    assert [ts.layer(t) for t in _TURNS] == ["rule", "rule", "llm", "pending", "rule", "unrecorded"]


def test_rule_name_is_branch_flags_only():
    assert ts.rule_name(["rule_price", "pii_phone", "multi_intent_detected"]) == "rule_price"
    assert ts.rule_name(["rule_none"]) is None
    assert ts.rule_name(["rule_price_rescue"]) is None
    assert ts.rule_name(["urgent", "pii_email"]) == "urgent"


def test_override_after_nlu_is_counted_but_keeps_layer():
    m = ts.measure({"name": "x", "code_sha": "0", "sets": []}, _TURNS, {})

    assert dict(m["layers"]) == {"rule": 3, "llm": 1, "pending": 1, "unrecorded": 1}
    assert m["override"]["test_assist_catalog_hit_is_price"] == 1
    assert m["by_rule"]["rule_test_assist"]["overridden"] == 1


def test_report_has_counts_but_no_patient_text(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"name": "synthetic", "code_sha": "0", "sets": []}), encoding="utf-8")
    (run / "turns.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in _TURNS), encoding="utf-8")
    header, turns = ts.load_run(str(run))

    text = ts.render(ts.measure(header, turns, {}))

    assert "правило — 75%, LLM — 25%" in text
    assert "`rule_price`" in text
    assert "секретная реплика" not in text
