import json
import unittest
from types import SimpleNamespace as NS
from messengers_router.reporting_facts import scenario_facts


def context(label="PRICE", payload=None, *, structured=True, flags=(), pending=False, secondary=()):
    ctx = NS(decision=NS(label=label,context_action="new_topic",flags=set(flags),clarify_needed=False,
             entities={"secondary_intents":list(secondary)}), response=NS(handoff=False),plan=NS(label=label),
             evidence=NS(items=payload or {}),state=NS(last_entities={}),should_clarify=pending,
             _reporting_structured=structured,text="Сколько стоит услуга и какое расписание врача?")
    return ctx, NS(get_pending=lambda _: pending)


class RagReportingTests(unittest.TestCase):
    def test_compound_price_and_schedule_reports_partial_facts_without_text(self):
        ctx, memory = context(payload={"price":{"prices":[{"price":123,"name":"PATIENT_CANARY"}]}},secondary=["DOCTOR_SCHEDULE"])
        ctx.response.text = "PATIENT_CANARY"
        value = scenario_facts(ctx,memory)
        self.assertEqual([t["state"] for t in value["tasks"]],["solved","pending"])
        self.assertNotIn("PATIENT_CANARY",json.dumps(value))

    def test_price_in_evidence_is_not_success_when_renderer_asks_clarification(self):
        ctx,memory = context(payload={"price":{"prices":[{"price":123}],"clarify_text":"question"}},pending=True)
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"clarification")
        ctx._reporting_structured = False
        ctx.should_clarify = False
        memory.get_pending = lambda _:False
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"unknown")

    def test_only_contextual_operator_refusal_and_scenario_result_link_count(self):
        ctx,memory = context(flags=["operator_offer_declined"])
        self.assertEqual(scenario_facts(ctx,memory)["operator_event"],"declined")
        ctx,memory = context(label="TEST_RESULT",payload={"test_result_status":{"result_links":["https://naykalab.ru/result"]}})
        self.assertEqual(scenario_facts(ctx,memory)["tasks"][0]["kind"],"results_link")
        ctx.evidence.items["test_result_status"]["missing_fields"] = ["name"]
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"clarification")

    def test_known_tax_scenario_and_greeting(self):
        ctx,memory = context(label="OTHER",payload={"main_index_info":{"note":"main_index_info: tax direct link"}})
        self.assertEqual(scenario_facts(ctx,memory)["tasks"][0]["key"],"TAX_DOCUMENT")
        ctx.decision.flags = {"smalltalk_greeting"}
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"neutral")

    def test_optional_followup_is_not_an_unanswered_patient_question(self):
        ctx,memory = context(payload={"service_bundle":{"retail_prices":[{"servicePrice":"320"}]}},secondary=["TEST_ASSIST"])
        ctx.text = "Сколько стоит общий анализ крови?"
        value = scenario_facts(ctx,memory)
        self.assertEqual(value["phase"],"answer")
        self.assertEqual(len(value["tasks"]),1)

    def test_acknowledgement_does_not_hide_handoff_or_operator_refusal(self):
        ctx,memory=context(label="OTHER",flags=["operator_offer_declined"])
        ctx.text="ок"
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"declined")
        ctx.decision.flags={"operator_offer_confirmed"}
        ctx.response.handoff=True
        self.assertEqual(scenario_facts(ctx,memory)["operator_event"],"accepted")

    def test_unselected_unsupported_evidence_does_not_claim_unanswered(self):
        ctx,memory=context(payload={"unsupported_catalog":True},structured=False)
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"unknown")
        ctx._reporting_structured=True
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"unanswered")

    def test_price_family_counts_only_rendered_rows_with_actual_prices(self):
        ctx,memory=context(payload={"service_bundle":{"service_kind":"family_query","visible_limit":1,
            "family_variants":[{"name":"synthetic"},{"servicePrice":320}]}})
        self.assertEqual(scenario_facts(ctx,memory)["phase"],"unanswered")
        ctx.evidence.items["service_bundle"]["showing_all"]=True
        self.assertEqual(scenario_facts(ctx,memory)["tasks"][0]["state"],"solved")
