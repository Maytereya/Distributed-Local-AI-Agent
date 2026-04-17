# tests/test_memory_dialog_state.py
"""Tests for save_dialog_state / load_dialog_state in MemoryStore."""
import logging

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import DialogState, SessionState


def _store() -> MemoryStore:
    return MemoryStore()


def _state(session_id: str = "s1") -> SessionState:
    return SessionState(session_id=session_id)


# --- save_dialog_state ---

def test_save_writes_snapshot_to_last_entities():
    store = _store()
    state = _state()
    state.dialog = DialogState(
        label="APPOINTMENT",
        phase="appointment_collecting",
        entities={"doctor_name": "Иванова"},
        candidate_entities={"specialty": "кардиолог"},
        missing_slots=["patient_name"],
        clarify_count=1,
        open_question="Ваше ФИО?",
        confidence=0.80,
    )
    store.save_dialog_state(state)

    snap = state.last_entities["_dialog_state"]
    assert snap["label"] == "APPOINTMENT"
    assert snap["phase"] == "appointment_collecting"
    assert snap["entities"] == {"doctor_name": "Иванова"}
    assert snap["candidate_entities"] == {"specialty": "кардиолог"}
    assert snap["missing_slots"] == ["patient_name"]
    assert snap["clarify_count"] == 1
    assert snap["open_question"] == "Ваше ФИО?"
    assert snap["confidence"] == 0.80


def test_save_creates_independent_copy():
    """Mutating state.dialog after save must not affect the snapshot."""
    store = _store()
    state = _state()
    state.dialog = DialogState(label="PRICE", entities={"service_name": "анализ крови"})
    store.save_dialog_state(state)

    state.dialog.entities["service_name"] = "изменено"
    snap = state.last_entities["_dialog_state"]
    assert snap["entities"]["service_name"] == "анализ крови"


# --- load_dialog_state ---

def test_load_restores_dialog_state():
    store = _store()
    state = _state()
    state.last_entities["_dialog_state"] = {
        "label": "DOCTOR_SCHEDULE",
        "phase": "",
        "entities": {"specialty": "терапевт"},
        "candidate_entities": {},
        "missing_slots": ["doctor_name"],
        "clarify_count": 0,
        "open_question": "Уточните врача?",
        "confidence": 0.70,
    }
    ds = store.load_dialog_state(state)

    assert ds.label == "DOCTOR_SCHEDULE"
    assert ds.entities == {"specialty": "терапевт"}
    assert ds.missing_slots == ["doctor_name"]
    assert ds.open_question == "Уточните врача?"
    assert state.dialog is ds  # state.dialog updated in-place


def test_load_returns_existing_when_no_snapshot():
    store = _store()
    state = _state()
    state.dialog.label = "PRICE"

    result = store.load_dialog_state(state)
    assert result.label == "PRICE"  # existing dialog untouched


def test_load_returns_existing_on_malformed_snapshot():
    store = _store()
    state = _state()
    state.last_entities["_dialog_state"] = "not a dict"

    result = store.load_dialog_state(state)
    assert result.label == "OTHER"  # default DialogState returned


def test_load_logs_warning_on_invalid_dialog_snapshot(caplog):
    store = _store()
    state = _state()
    state.dialog = DialogState(label="PRICE", entities={"service_name": "ЭКГ"})
    state.last_entities["_dialog_state"] = {
        "label": "PRICE",
        "phase": "",
        "entities": {"service_name": "ЭКГ"},
        "candidate_entities": {},
        "missing_slots": [],
        "clarify_count": "oops",
        "open_question": "",
        "confidence": 0.7,
    }

    with caplog.at_level(logging.WARNING):
        result = store.load_dialog_state(state)

    assert result is state.dialog
    assert result.label == "PRICE"
    assert "memory_deserialize_failed" in caplog.text


# --- round-trip ---

def test_save_load_round_trip():
    store = _store()
    state = _state()
    original = DialogState(
        label="TEST_RESULT",
        phase="",
        entities={"surname": "Иванов", "year": "1990"},
        missing_slots=["number"],
        clarify_count=2,
        confidence=0.65,
    )
    state.dialog = original
    store.save_dialog_state(state)

    # Reset dialog to default, then reload
    state.dialog = DialogState()
    restored = store.load_dialog_state(state)

    assert restored.label == "TEST_RESULT"
    assert restored.entities == {"surname": "Иванов", "year": "1990"}
    assert restored.missing_slots == ["number"]
    assert restored.clarify_count == 2
    assert abs(restored.confidence - 0.65) < 1e-9
