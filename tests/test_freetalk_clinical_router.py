import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.routing_contract import (
    clarify_type_for_slots,
    clarify_question_for_slots,
    merge_missing_slots_from_plan,
    parse_clinical_decision,
)
from localragagent.freetalk.routing_prompting import build_clinical_router_prompt, build_post_tool_verifier_prompt


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
    assert "doctor_name" in missing
    assert "specialty" in missing
    assert "service_or_analysis_name" in missing


def test_clarify_question_for_slots_is_specific():
    text = clarify_question_for_slots("doctor_schedule", ["doctor_name", "specialty"])
    assert "распис" in text.lower()


def test_parse_clinical_decision_keeps_result_lookup_entities():
    decision = parse_clinical_decision(
        {
            "intent": "test_result",
            "confidence": 0.81,
            "entities": {
                "result_surname": "Иванов",
                "result_year_of_birth": "1990",
                "result_analysis_code": "Бг",
                "result_analysis_number": "12345",
            },
            "tool_plan": ["test_result_status"],
        },
        include_meili_tools=False,
    )
    assert decision.intent == "test_result"
    assert decision.entities["result_surname"] == "Иванов"
    assert decision.entities["result_year_of_birth"] == "1990"
    assert decision.entities["result_analysis_code"] == "Бг"
    assert decision.entities["result_analysis_number"] == "12345"


def test_merge_missing_slots_from_plan_for_test_result_is_granular():
    missing = merge_missing_slots_from_plan(
        ["test_result_status"],
        entities={"result_surname": "Иванов"},
    )
    assert "result_year_of_birth" in missing
    assert "result_analysis_code" in missing
    assert "result_analysis_number" in missing
    assert "result_surname" not in missing


def test_parse_clinical_decision_infers_missing_auth_data_clarify_type():
    decision = parse_clinical_decision(
        {
            "intent": "test_result",
            "confidence": 0.8,
            "missing_slots": ["result_surname", "result_year_of_birth"],
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


def test_parse_clinical_decision_drops_branch_clarify_for_price():
    decision = parse_clinical_decision(
        {
            "intent": "price",
            "entities": {"service_name": "Общий анализ крови"},
            "missing_slots": ["branch_name"],
            "clarify_type": "narrow_choice",
            "clarify_question": "Вы хотите узнать стоимость ОАК в филиале Наука?",
            "tool_plan": ["price_info"],
        },
        include_meili_tools=False,
    )

    assert decision.missing_slots == []
    assert decision.clarify_type == ""
    assert decision.clarify_question == ""


def test_parse_clinical_decision_keeps_non_branch_missing_slots_for_price():
    decision = parse_clinical_decision(
        {
            "intent": "price",
            "missing_slots": ["branch_name", "service_or_analysis_name"],
            "clarify_type": "narrow_choice",
            "clarify_question": "Уточните филиал.",
            "tool_plan": ["price_info"],
        },
        include_meili_tools=False,
    )

    assert decision.missing_slots == ["service_or_analysis_name"]
    assert decision.clarify_type == ""
    assert decision.clarify_question == ""


def test_parse_clinical_decision_normalizes_and_drops_unknown_missing_slots():
    decision = parse_clinical_decision(
        {
            "intent": "test_result",
            "missing_slots": [
                "surname",
                "birth_year",
                "result_filial",
                "result_number",
                "made_up_slot",
            ],
            "tool_plan": ["test_result_status"],
        },
        include_meili_tools=False,
    )

    assert decision.missing_slots == [
        "result_surname",
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    ]


def test_parse_clinical_decision_expands_doctor_or_specialty_missing_slot():
    decision = parse_clinical_decision(
        {
            "intent": "doctor_schedule",
            "missing_slots": ["doctor_name_or_specialty", "doctor_id"],
        },
        include_meili_tools=False,
    )

    assert decision.missing_slots == ["doctor_name", "specialty"]


def test_parse_clinical_decision_drops_schedule_narrowing_missing_slots():
    decision = parse_clinical_decision(
        {
            "intent": "doctor_schedule",
            "entities": {"doctor_name": "Дразнин Антон Владимирович"},
            "missing_slots": ["date", "time", "branch_name"],
            "clarify_type": "narrow_choice",
            "clarify_question": "На какую неделю нужно расписание?",
            "tool_plan": ["doctors_schedule_week"],
        },
        include_meili_tools=False,
    )

    assert decision.missing_slots == []
    assert decision.clarify_type == ""
    assert decision.clarify_question == ""


def test_clarify_type_for_slots_marks_identify_and_missing_auth_data():
    assert clarify_type_for_slots("doctor_schedule", ["doctor_name", "specialty"]) == "identify"
    assert clarify_type_for_slots("test_result", ["result_analysis_number"]) == "missing_auth_data"


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
