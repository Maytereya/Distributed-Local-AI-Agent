from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.contracts import DialogState
from localragagent.freetalk.turn_policy import (
    CONTROL_ACTION_HANDOFF_OPERATOR,
    CONTROL_ACTION_RESET_SESSION,
    TURN_KIND_CONTACT,
    TURN_KIND_HANDOFF_OPERATOR,
    TURN_KIND_NEGATIVE_FEEDBACK,
    TURN_KIND_HARD_RESET,
    classify_turn,
)


def _active_state() -> DialogState:
    return DialogState(
        route="clinical",
        intent="price",
        missing_slots=["service_or_analysis_name"],
        phase="collecting",
        open_question="Уточните услугу.",
        flow_active=True,
        flow_kind="clarify",
        flow_stage="collecting",
        flow_interruptible=True,
        expected_slots=["service_or_analysis_name"],
    )


def test_operator_request_is_global_control_inside_slot_flow():
    decision = classify_turn("переведи на оператора", dialog_state=_active_state())

    assert decision.kind == TURN_KIND_HANDOFF_OPERATOR
    assert decision.control_action == CONTROL_ACTION_HANDOFF_OPERATOR
    assert decision.flow_relation == "interrupt"


def test_stop_and_reset_are_full_session_reset_controls():
    for text in ("стоп", "сбрось диалог", "reset", "hard reset"):
        decision = classify_turn(text, dialog_state=_active_state())

        assert decision.kind == TURN_KIND_HARD_RESET
        assert decision.control_action == CONTROL_ACTION_RESET_SESSION
        assert decision.flow_relation == "interrupt"


def test_negative_feedback_is_global_control_without_active_flow():
    decision = classify_turn("ты несешь бред", dialog_state=DialogState())

    assert decision.kind == TURN_KIND_NEGATIVE_FEEDBACK
    assert decision.control_action == CONTROL_ACTION_RESET_SESSION
    assert decision.flow_relation == "none"


def test_clinic_contacts_are_contact_source_even_with_web_words():
    decision = classify_turn("найди в сети телефоны клиники Наука Самара")

    assert decision.kind == TURN_KIND_CONTACT
    assert decision.source_mode == "contact"
