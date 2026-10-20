"""Отчёт `evalbench.role_shadow` — тень «слово → роль» на синтетических ходах (STEP-2026-10-21-G-ROLE-PICK-SHADOW)."""

from __future__ import annotations

from evalbench import role_shadow


def test_counts_layers_agreement_and_accuracy():
    turns = [
        {"id": "a", "telemetry": {"role_layer": "exact", "role_exact": "кардиолог", "role_llm": "кардиолог", "role_llm_agree": True}},
        {"id": "b", "telemetry": {"role_layer": "miss", "role_exact": None, "role_llm": "офтальмолог", "role_rules_unmapped": True}},
        {"id": "c", "telemetry": {"role_layer": "miss", "role_exact": None, "role_llm": "none"}},
        {"id": "d", "telemetry": {}},
    ]
    expects = {"a": "кардиолог", "b": "офтальмолог", "c": "нет", "d": "нет"}
    m = role_shadow.measure(turns, expects)
    assert m["n"] == 3 and m["layers"] == {"exact": 1, "miss": 2}
    assert (m["agree"]["role_llm_agree:yes"], m["agree"]["role_llm_agree:n"], m["agree"]["role_rules_unmapped"]) == (1, 1, 1)
    assert (m["acc"]["n"], m["acc"]["no_shadow"], m["acc"]["exact_ok"], m["acc"]["llm_ok"]) == (4, 1, 1, 3)
    assert role_shadow._ok("хирург, флеболог", "хирург")
