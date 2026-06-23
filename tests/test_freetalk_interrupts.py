from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.contracts import DialogState
from localragagent.freetalk.interrupt_policy import (
    FLOW_INTERRUPT_CONFIRM_TEXT,
    INTERRUPT_ARBITER_DECISIONS,
    SESSION_RESET_CONFIRM_TEXT,
    TOPIC_SWITCH_CONFIRM_TEXT,
    apply_interrupt_arbiter_decision,
    apply_interrupt_precheck,
    detect_interrupt_intent,
    parse_interrupt_arbiter_payload,
)


def test_detect_interrupt_intent_skips_slot_correction():
    assert detect_interrupt_intent("не этого врача") == "none"
    assert detect_interrupt_intent("другой филиал") == "none"


def test_detect_interrupt_intent_classifies_hard_reset_and_feedback():
    assert detect_interrupt_intent("очисти диалог") == "hard_reset"
    assert detect_interrupt_intent("это бред") == "feedback"
    assert detect_interrupt_intent("неправильно, стоп") == "interrupt"


def test_feedback_requires_active_flow_to_trigger_interrupt_guard():
    inactive = apply_interrupt_precheck(
        user_message="это бред",
        dialog_state=DialogState(),
        memory_entities={},
    )
    assert inactive.handled is False

    active = apply_interrupt_precheck(
        user_message="это бред",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_schedule",
            missing_slots=["doctor_name"],
            phase="collecting",
            open_question="Уточните врача.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните врача.",
            expected_slots=["doctor_name"],
        ),
        memory_entities={},
    )
    assert active.handled is True
    assert active.reply_text == FLOW_INTERRUPT_CONFIRM_TEXT
    assert active.next_state is not None
    assert active.next_state.phase == "interrupt_confirm_flow"


def test_hard_reset_builds_session_confirm_state():
    result = apply_interrupt_precheck(
        user_message="очисти диалог",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_info",
            phase="collecting",
            open_question="Уточните врача.",
        ),
        memory_entities={},
    )
    assert result.handled is True
    assert result.reply_text == SESSION_RESET_CONFIRM_TEXT
    assert result.next_state is not None
    assert result.next_state.phase == "interrupt_confirm_session"


def test_topic_switch_builds_confirm_state_for_active_flow():
    result = apply_interrupt_precheck(
        user_message="Сколько стоит общий анализ крови?",
        dialog_state=DialogState(
            route="clinical",
            intent="appointment",
            phase="appointment_collecting",
            open_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            flow_active=True,
            flow_kind="appointment",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            expected_slots=["patient_name"],
        ),
        memory_entities={},
    )
    assert result.handled is True
    assert result.reply_text == TOPIC_SWITCH_CONFIRM_TEXT
    assert result.next_state is not None
    assert result.next_state.phase == "interrupt_confirm_topic_switch"


def test_general_question_during_active_clarify_reenters_without_confirm():
    result = apply_interrupt_precheck(
        user_message="Спазм диафрагмы у взрослых людей. Насколько частая проблема и в каких симптомах может выражаться?",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_info",
            phase="collecting",
            open_question="Уточните врача.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните врача.",
            expected_slots=["doctor_name"],
        ),
        memory_entities={},
    )

    assert result.handled is True
    assert result.clear_state is True
    assert result.reentry_message.startswith("Спазм диафрагмы")
    assert result.next_state is None


def test_expected_slot_answer_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="Суворов",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_schedule",
            phase="collecting",
            open_question="Уточните врача.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните врача.",
            expected_slots=["doctor_name"],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_expected_specialty_answer_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="уролог",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_info",
            phase="collecting",
            open_question="Уточните специальность.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните специальность.",
            expected_slots=["specialty"],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_expected_date_answer_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="на 17 мая",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_schedule",
            phase="collecting",
            open_question="Уточните дату.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните дату.",
            expected_slots=["date"],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_expected_time_answer_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="вечером",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_schedule",
            phase="collecting",
            open_question="Уточните время.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните время.",
            expected_slots=["time"],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_expected_branch_answer_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="на Ленина",
        dialog_state=DialogState(
            route="clinical",
            intent="address",
            phase="collecting",
            open_question="Уточните филиал.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните филиал.",
            expected_slots=["branch_or_city"],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_expected_service_answer_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="общий анализ мочи",
        dialog_state=DialogState(
            route="clinical",
            intent="price",
            phase="collecting",
            open_question="Уточните услугу.",
            flow_active=True,
            flow_kind="clarify",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните услугу.",
            expected_slots=["service_or_analysis_name"],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_expected_result_fields_are_not_treated_as_topic_switch():
    code_result = apply_interrupt_precheck(
        user_message="Бг",
        dialog_state=DialogState(
            route="clinical",
            intent="test_result",
            phase="collecting",
            open_question="Уточните код анализа.",
            flow_active=True,
            flow_kind="result_lookup",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните код анализа.",
            expected_slots=["result_analysis_code"],
        ),
        memory_entities={},
    )
    number_result = apply_interrupt_precheck(
        user_message="12345",
        dialog_state=DialogState(
            route="clinical",
            intent="test_result",
            phase="collecting",
            open_question="Уточните номер анализа.",
            flow_active=True,
            flow_kind="result_lookup",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните номер анализа.",
            expected_slots=["result_analysis_number"],
        ),
        memory_entities={},
    )
    assert code_result.handled is False
    assert number_result.handled is False


def test_flow_local_signals_are_not_sent_to_interrupt_arbiter():
    uncertainty = apply_interrupt_precheck(
        user_message="не знаю",
        dialog_state=DialogState(
            route="clinical",
            intent="appointment",
            phase="appointment_collecting",
            open_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            flow_active=True,
            flow_kind="appointment",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            expected_slots=["patient_name"],
        ),
        memory_entities={},
    )
    no_preference = apply_interrupt_precheck(
        user_message="без разницы",
        dialog_state=DialogState(
            route="clinical",
            intent="appointment",
            phase="appointment_collecting",
            open_question="Выберите, пожалуйста, дату и время для записи.",
            flow_active=True,
            flow_kind="appointment",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Выберите, пожалуйста, дату и время для записи.",
            expected_slots=["date", "time", "patient_name"],
        ),
        memory_entities={},
    )
    partial_tuple = apply_interrupt_precheck(
        user_message="Иванов, 1990",
        dialog_state=DialogState(
            route="clinical",
            intent="test_result",
            phase="collecting",
            open_question="Уточните код анализа и номер анализа.",
            flow_active=True,
            flow_kind="result_lookup",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Уточните код анализа и номер анализа.",
            expected_slots=["result_analysis_code", "result_analysis_number"],
        ),
        memory_entities={},
    )
    assert uncertainty.handled is False
    assert uncertainty.arbiter_needed is False
    assert no_preference.handled is False
    assert no_preference.arbiter_needed is False
    assert partial_tuple.handled is False
    assert partial_tuple.arbiter_needed is False


def test_confirmation_answer_with_candidate_correction_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="нет, Суворов",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_info",
            phase="confirm_candidate",
            open_question="Вы имели в виду Трубина? Ответьте: да или нет.",
            flow_active=True,
            flow_kind="confirmation",
            flow_stage="confirm",
            flow_interruptible=True,
            flow_resume_question="Вы имели в виду Трубина? Ответьте: да или нет.",
            expected_slots=[],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_confirmation_answer_with_affirmation_phrase_is_not_treated_as_topic_switch():
    result = apply_interrupt_precheck(
        user_message="да, этот",
        dialog_state=DialogState(
            route="clinical",
            intent="doctor_info",
            phase="confirm_candidate",
            open_question="Вы имели в виду Трубина? Ответьте: да или нет.",
            flow_active=True,
            flow_kind="confirmation",
            flow_stage="confirm",
            flow_interruptible=True,
            flow_resume_question="Вы имели в виду Трубина? Ответьте: да или нет.",
            expected_slots=[],
        ),
        memory_entities={},
    )
    assert result.handled is False


def test_parse_interrupt_arbiter_payload_accepts_known_decisions_only():
    assert "switch" in INTERRUPT_ARBITER_DECISIONS
    assert parse_interrupt_arbiter_payload({"decision": "switch"}) == "switch"
    assert parse_interrupt_arbiter_payload({"decision": "bad"}) == "unknown"
    assert parse_interrupt_arbiter_payload(None) == "unknown"


def test_apply_interrupt_arbiter_decision_builds_topic_switch_confirm():
    result = apply_interrupt_arbiter_decision(
        decision="switch",
        user_message="ладно, другой вопрос",
        dialog_state=DialogState(
            route="clinical",
            intent="appointment",
            phase="appointment_collecting",
            open_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            flow_active=True,
            flow_kind="appointment",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            expected_slots=["patient_name"],
        ),
    )
    assert result.handled is True
    assert result.reply_text == TOPIC_SWITCH_CONFIRM_TEXT
    assert result.next_state is not None
    assert result.next_state.phase == "interrupt_confirm_topic_switch"
