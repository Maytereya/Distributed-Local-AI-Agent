import unittest

from reporting_contract import apply_facts, balances, empty_result, expire_result, unknown_contract, validate_contract


def facts(phase="answer", tasks=None, event="none", reason="none"):
    return {"version": 1, "context": "continue", "phase": phase,
            "tasks": tasks if tasks is not None else [{"key": "PRICE", "state": "solved", "kind": "structured_answer"}],
            "reason": reason, "operator_event": event}


def apply(state, meta, delivered=True, route="bot"):
    return apply_facts(state, meta, delivered=delivered, actual_route=route)


class AppealRulesTests(unittest.TestCase):
    def test_success_requires_scenario_evidence_and_confirmed_delivery(self):
        state = apply(empty_result(), facts())
        self.assertEqual(state["outcome"], "S")
        self.assertEqual(apply(empty_result(), facts(), delivered=False)["outcome"], "X")
        self.assertEqual(apply(empty_result(), unknown_contract())["outcome"], "X")
        self.assertEqual(expire_result(state, 24 * 3600)["outcome"], "S")

    def test_compound_query_is_partial_and_survives_inactivity(self):
        state = apply(empty_result(), facts(tasks=[
            {"key": "PRICE", "state": "solved", "kind": "structured_answer"},
            {"key": "DOCTOR_SCHEDULE", "state": "pending", "kind": "none"}]))
        self.assertEqual(state["outcome"], "P")
        self.assertEqual(expire_result(state, 6 * 3600)["outcome"], "P")
        state = apply(state, facts("clarification", tasks=[{"key":"DOCTOR_SCHEDULE","state":"pending","kind":"none"}]))
        self.assertEqual(state["outcome"], "P")
        self.assertEqual(expire_result(state, 6 * 3600)["outcome"], "P")
        state = apply(state, facts(tasks=[{"key": "DOCTOR_SCHEDULE", "state": "solved", "kind": "structured_answer"}]))
        self.assertEqual(state["outcome"], "S")

    def test_decline_remains_event_after_resolution_partial_or_handoff(self):
        state = apply(empty_result(), facts("operator_offer", tasks=[], event="offered"))
        declined = apply(state, facts("declined", tasks=[], event="declined", reason="operator_declined"))
        self.assertEqual(declined["outcome"], "R")
        solved = apply(declined, facts())
        self.assertEqual((solved["outcome"], solved["declined"]), ("S", True))
        partial = apply(declined, facts(tasks=[
            {"key": "PRICE", "state": "solved", "kind": "structured_answer"},
            {"key": "ADDRESS", "state": "pending", "kind": "none"}]))
        self.assertEqual((partial["outcome"], partial["declined"]), ("P", True))
        handed = apply(declined, facts(), route="operator")
        self.assertEqual((handed["outcome"], handed["declined"]), ("H", True))
        self.assertEqual(apply(handed, facts())["outcome"], "H")

    def test_six_hours_apply_only_to_unfinished_with_complete_events(self):
        waiting = apply(empty_result(), facts("clarification", tasks=[{"key":"TEST_RESULT","state":"pending","kind":"none"}]))
        self.assertEqual(expire_result(waiting, 21599)["outcome"], "W")
        self.assertEqual(expire_result(waiting, 21600)["outcome"], "U")
        self.assertEqual(expire_result(empty_result(), 21600)["outcome"], "X")
        unknown = apply(empty_result(), unknown_contract())
        self.assertEqual(expire_result(unknown, 21600)["outcome"], "X")

    def test_operator_offer_or_acceptance_does_not_invent_actual_handoff(self):
        offered = apply(empty_result(), facts("operator_offer", tasks=[], event="offered"))
        self.assertEqual((offered["outcome"], offered["route"]), ("W", "bot"))
        accepted = apply(offered, facts("operator_offer", tasks=[], event="accepted"))
        self.assertNotEqual(accepted["outcome"], "H")
        self.assertEqual(apply(accepted, unknown_contract(), delivered=False, route="operator")["outcome"], "H")

    def test_raw_patient_data_and_malformed_types_are_discarded(self):
        for dirty in (dict(facts(), patient="PATIENT_CANARY"), dict(facts(), reason="PATIENT_CANARY"),
                      dict(facts(), tasks=[{"key":"PRICE","state":"solved","kind":"PATIENT_CANARY"}]),
                      dict(facts(), version=True), dict(facts(), phase={}), dict(facts(), context=[]),
                      dict(facts(), tasks=[{"key":"PRICE","state":[],"kind":"none"}])):
            safe = validate_contract(dirty)
            self.assertEqual(safe["reason"], "invalid_contract")
            self.assertNotIn("PATIENT_CANARY", str(safe))

    def test_seven_outcomes_routes_and_percentage_denominator_balance(self):
        results = [dict(empty_result(), outcome=code, route="operator" if code == "H" else "bot") for code in "SPURHWX"]
        report = balances(results)
        self.assertEqual((report["total"], report["completed_known"]), (7, 5))
        self.assertEqual((report["success_percent"], report["partial_percent"]), (20.0, 20.0))
        self.assertIsNone(balances([])["success_percent"])
        with self.assertRaises(ValueError):
            balances([dict(empty_result(), outcome="H", route="unknown")])


if __name__ == "__main__":
    unittest.main()
