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


# --- Stage 4: evidence handoff contract (both code paths) ---

def test_evidence_requires_handoff_catches_nested_dict():
    """Service fallbacks store handoff_required inside a nested payload dict.
    The helper must find it even though the top-level key is a domain name."""
    from messengers_router.mess_types import Evidence
    from messengers_router.policies import evidence_requires_handoff

    ev = Evidence()
    ev.put("doctor_schedule", {"handoff_required": True, "handoff_reason": "service_error"})

    required, msg, reason = evidence_requires_handoff(ev)
    assert required is True
    assert reason == "service_error"


def test_evidence_requires_handoff_ignores_toplevel_boolean():
    """The helper only walks values that are dicts.
    A top-level boolean True is intentionally NOT caught by it — the router
    combines both checks so neither case is missed."""
    from messengers_router.mess_types import Evidence
    from messengers_router.policies import evidence_requires_handoff

    ev = Evidence()
    ev.put("handoff_required", True)  # executor-style top-level boolean

    required, _, _ = evidence_requires_handoff(ev)
    assert required is False  # helper alone doesn't catch this — by design


def test_graph_engine_transitions_to_hitl_for_service_fallback_handoff():
    """GraphEngine must reach HITL_PENDING when a service fallback stores
    handoff_required inside a nested dict (not caught by evidence.get alone)."""
    from messengers_router.mess_types import Evidence, RouteDecision, SessionState
    from messengers_router.dialog_graph import GraphEngine, DialogState

    ev = Evidence()
    ev.put("doctor_schedule", {"handoff_required": True, "handoff_reason": "service_error"})

    from messengers_router.policies import evidence_requires_handoff
    handoff_planned = (
        bool(ev.get("handoff_required"))
        or evidence_requires_handoff(ev)[0]
    )

    engine = GraphEngine()
    state = SessionState(session_id="stage4-test")
    decision = RouteDecision(label="DOCTOR_SCHEDULE", needs_handoff=False)

    out = engine.next(
        session=state,
        decision=decision,
        pending=None,
        handoff_planned=handoff_planned,
    )

    assert out.state == DialogState.HITL_PENDING
    assert out.transition.reason == "handoff"


def test_graph_engine_transitions_to_hitl_for_executor_toplevel_handoff():
    """GraphEngine must reach HITL_PENDING when executor sets a top-level
    handoff_required boolean (not caught by evidence_requires_handoff alone)."""
    from messengers_router.mess_types import Evidence, RouteDecision, SessionState
    from messengers_router.dialog_graph import GraphEngine, DialogState
    from messengers_router.policies import evidence_requires_handoff

    ev = Evidence()
    ev.put("handoff_required", True)   # executor style
    ev.put("handoff_reason", "service_error")

    handoff_planned = (
        bool(ev.get("handoff_required"))
        or evidence_requires_handoff(ev)[0]
    )

    engine = GraphEngine()
    state = SessionState(session_id="stage4-executor-test")
    decision = RouteDecision(label="DOCTOR_SCHEDULE", needs_handoff=False)

    out = engine.next(
        session=state,
        decision=decision,
        pending=None,
        handoff_planned=handoff_planned,
    )

    assert out.state == DialogState.HITL_PENDING
    assert out.transition.reason == "handoff"
