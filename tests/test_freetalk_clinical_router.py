import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.clinical_router import (
    clarify_type_for_slots,
    clarify_question_for_slots,
    merge_missing_slots_from_plan,
    parse_clinical_decision,
)
from localragagent.freetalk.prompts import build_clinical_router_prompt, build_post_tool_verifier_prompt


def test_parse_clinical_decision_maps_schedule_alias_and_plan():
    decision = parse_clinical_decision(
        {
            "intent": "schedule",
            "confidence": 0.77,
            "entities": {"doctor_name": "Дразнин"},
        },
        include_meili_tools=True,
    )
    assert decision.intent == "doctor_schedule"
    assert decision.confidence == 0.77
    assert decision.tool_plan[:2] == ["doctors_schedule_week", "doctors_info"]


def test_parse_clinical_decision_respects_meili_flag():
    decision = parse_clinical_decision(
        {
            "intent": "clinic_documents",
            "confidence": 0.8,
        },
        include_meili_tools=False,
    )
    assert decision.intent == "clinic_documents"
    assert decision.tool_plan == []


def test_merge_missing_slots_from_plan_for_doctor_and_service():
    missing = merge_missing_slots_from_plan(
        ["doctors_info", "price_info"],
        entities={},
    )
    assert "doctor_name_or_specialty" in missing
    assert "service_or_analysis_name" in missing


def test_clarify_question_for_slots_is_specific():
    text = clarify_question_for_slots("doctor_schedule", ["doctor_name_or_specialty"])
    assert "распис" in text.lower()


def test_parse_clinical_decision_keeps_result_lookup_entities():
    decision = parse_clinical_decision(
        {
            "intent": "test_result",
            "confidence": 0.81,
            "entities": {
                "surname": "Иванов",
                "year": "1990",
                "filial": "Самара",
                "number": "12345",
            },
            "tool_plan": ["test_result_status"],
        },
        include_meili_tools=False,
    )
    assert decision.intent == "test_result"
    assert decision.entities["surname"] == "Иванов"
    assert decision.entities["year"] == "1990"
    assert decision.entities["filial"] == "Самара"
    assert decision.entities["number"] == "12345"


def test_merge_missing_slots_from_plan_for_test_result_is_granular():
    missing = merge_missing_slots_from_plan(
        ["test_result_status"],
        entities={"surname": "Иванов"},
    )
    assert "result_year" in missing
    assert "result_filial" in missing
    assert "result_number" in missing
    assert "result_surname" not in missing


def test_parse_clinical_decision_infers_missing_auth_data_clarify_type():
    decision = parse_clinical_decision(
        {
            "intent": "test_result",
            "confidence": 0.8,
            "missing_slots": ["result_surname", "result_year"],
            "clarify_question": "Уточните фамилию и год рождения.",
        },
        include_meili_tools=False,
    )
    assert decision.clarify_type == "missing_auth_data"


def test_parse_clinical_decision_respects_explicit_confirm_candidate_type():
    decision = parse_clinical_decision(
        {
            "intent": "price",
            "confidence": 0.85,
            "entities": {},
            "missing_slots": [],
            "clarify_type": "confirm_candidate",
            "clarify_question": "Правильно понял, что нужна услуга ФКС с наркозом?",
        },
        include_meili_tools=False,
    )
    assert decision.clarify_type == "confirm_candidate"


def test_clarify_type_for_slots_marks_identify_and_missing_auth_data():
    assert clarify_type_for_slots("doctor_schedule", ["doctor_name_or_specialty"]) == "identify"
    assert clarify_type_for_slots("test_result", ["result_number"]) == "missing_auth_data"


def test_clinical_router_prompt_mentions_clarify_types():
    prompt = build_clinical_router_prompt(
        system_prompt="FT",
        summary="",
        turns=[],
        user_message="Покажи расписание врача",
        dialog_state={"clarify_type": "identify"},
    )
    assert "clarify_type" in prompt
    assert "confirm_candidate" in prompt
    assert "missing_auth_data" in prompt


def test_post_tool_verifier_prompt_mentions_clarify_types():
    prompt = build_post_tool_verifier_prompt(
        user_message="Проверь результат анализа",
        intent="test_result",
        tool_name="test_result_status",
        drafted_answer="",
        tool_payload={"missing_fields": ["фамилия"]},
    )
    assert "clarify_type" in prompt
    assert "narrow_choice" in prompt
    assert "missing_auth_data" in prompt
