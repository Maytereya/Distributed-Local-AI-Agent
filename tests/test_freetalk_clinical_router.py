import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.clinical_router import (
    clarify_question_for_slots,
    merge_missing_slots_from_plan,
    parse_clinical_decision,
)


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
