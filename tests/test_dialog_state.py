# tests/test_dialog_state.py
"""Unit tests for DialogState dataclass in mess_types."""
import dataclasses
import logging
from messengers_router import dialog_graph as dialog_graph_mod
from messengers_router.mess_types import DialogState, SessionState


# --- Construction ---

def test_dialog_state_defaults():
    ds = DialogState()
    assert ds.label == "OTHER"
    assert ds.phase == ""
    assert ds.entities == {}
    assert ds.candidate_entities == {}
    assert ds.missing_slots == []
    assert ds.clarify_count == 0
    assert ds.open_question == ""
    assert ds.confidence == 0.0


def test_dialog_state_is_dataclass():
    assert dataclasses.is_dataclass(DialogState)


# --- is_active() ---

def test_is_active_false_for_other():
    ds = DialogState(label="OTHER")
    assert not ds.is_active()


def test_is_active_false_for_empty_label():
    ds = DialogState(label="")
    assert not ds.is_active()


def test_is_active_true_for_real_label():
    for label in ("APPOINTMENT", "DOCTOR_SCHEDULE", "PRICE", "TEST_RESULT"):
        ds = DialogState(label=label)
        assert ds.is_active(), f"Expected is_active() True for label={label!r}"


# --- clear() ---

def test_clear_resets_all_fields():
    ds = DialogState(
        label="APPOINTMENT",
        phase="appointment_collecting",
        entities={"doctor_name": "Иванова"},
        candidate_entities={"specialty": "кардиолог"},
        missing_slots=["patient_name"],
        clarify_count=2,
        open_question="Ваше ФИО?",
        confidence=0.80,
    )
    ds.clear()
    assert ds.label == "OTHER"
    assert ds.phase == ""
    assert ds.entities == {}
    assert ds.candidate_entities == {}
    assert ds.missing_slots == []
    assert ds.clarify_count == 0
    assert ds.open_question == ""
    assert ds.confidence == 0.0


def test_clear_makes_is_active_false():
    ds = DialogState(label="APPOINTMENT")
    assert ds.is_active()
    ds.clear()
    assert not ds.is_active()


# --- merge_entities() ---

def test_merge_entities_adds_missing_keys():
    ds = DialogState(label="APPOINTMENT", entities={"doctor_name": "Петрова"})
    ds.merge_entities({"specialty": "кардиолог", "city": "самара"})
    assert ds.entities["doctor_name"] == "Петрова"
    assert ds.entities["specialty"] == "кардиолог"
    assert ds.entities["city"] == "самара"


def test_merge_entities_does_not_overwrite_existing():
    ds = DialogState(label="DOCTOR_SCHEDULE", entities={"specialty": "терапевт"})
    ds.merge_entities({"specialty": "хирург"})
    assert ds.entities["specialty"] == "терапевт"  # existing preserved


def test_merge_entities_skips_empty_values():
    ds = DialogState(label="APPOINTMENT", entities={})
    ds.merge_entities({"doctor_name": "", "specialty": None, "city": []})
    assert ds.entities == {}


def test_merge_entities_skips_none_values():
    ds = DialogState(entities={"doctor_name": "Иванова"})
    ds.merge_entities({"patient_name": None})
    assert "patient_name" not in ds.entities


# --- SessionState integration ---

def test_session_state_has_dialog_field():
    ss = SessionState(session_id="test-123")
    assert hasattr(ss, "dialog")
    assert isinstance(ss.dialog, DialogState)


def test_session_state_dialog_defaults_to_other():
    ss = SessionState(session_id="test-456")
    assert ss.dialog.label == "OTHER"
    assert not ss.dialog.is_active()


def test_session_state_dialog_is_independent_per_instance():
    """Each SessionState must have its own DialogState (not shared default)."""
    s1 = SessionState(session_id="s1")
    s2 = SessionState(session_id="s2")
    s1.dialog.label = "APPOINTMENT"
    assert s2.dialog.label == "OTHER"


def test_dialog_graph_logs_warning_on_invalid_state(caplog):
    state = SessionState(session_id="graph-invalid", last_entities={"_dialog_state": "BROKEN"})

    with caplog.at_level(logging.WARNING):
        result = dialog_graph_mod.get_dialog_state(state)

    assert result == dialog_graph_mod.DialogState.IDLE
    assert "dialog_state_deserialize_failed" in caplog.text
