"""
BUG-2026-06-07-01 class-invariant test.

Invariant: for a doctor-selected appointment where ADDRESS evidence is empty
and DOCTOR_SCHEDULE evidence carries the doctor's own multi-branch schedule,
the branch offer contains the DOCTOR's branches and NEVER the lab-only generic
branch "ул.Гагарина, 64" — regardless of appointment_action (reschedule / new).

Root cause: build_appointment_schedule_preview_response returns early for
reschedule/cancel (line ~468) BEFORE it calls
hydrate_appointment_context_from_schedule, so appointment_branch_options stays
empty → build_appointment_step_response falls through to safe_get_branches
fallback which can surface lab-only "Гагарина 64".

Fix: in build_appointment_step_response's APPOINTMENT_STEP_BRANCH block,
when doctor_selected and stored_options empty, re-run hydrate from the
DOCTOR_SCHEDULE evidence payload already in Evidence.
"""

import pytest

from messengers_router import evidence_keys as ek
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.response_builder import build_appointment_step_response
from messengers_router.services import Services

_LAB_ONLY_BRANCH = "ул.Гагарина, 64"
_DOCTOR_BRANCH_1 = "ул.Победы, 83"
_DOCTOR_BRANCH_2 = "ул.Аминева, 8"

# Synthetic DOCTOR_SCHEDULE payload: doctor Трубин has two own branches.
_TRUBIN_SCHEDULE_PAYLOAD = {
    "schedule": [
        {
            "fio": "Трубин Алексей Юрьевич",
            "regions": [_DOCTOR_BRANCH_1, _DOCTOR_BRANCH_2],
            "schedule": {
                _DOCTOR_BRANCH_1: [{"date": "2026-06-20", "slots": ["09:00", "10:00"]}],
                _DOCTOR_BRANCH_2: [{"date": "2026-06-21", "slots": ["11:00"]}],
            },
        }
    ]
}


@pytest.mark.parametrize("appointment_action", ["reschedule", ""])
def test_doctor_branch_offer_uses_doctors_own_schedule_not_lab_fallback(
    monkeypatch,
    appointment_action: str,
) -> None:
    """
    CLASS invariant (BUG-2026-06-07-01): when a doctor is named and
    DOCTOR_SCHEDULE evidence is populated with the doctor's own regions/windows,
    the branch step MUST offer those doctor-specific branches regardless of
    appointment_action ("reschedule" is the regressed path, "" is the baseline
    new-appointment path).

    safe_get_branches is patched to return ONLY the lab-only branch to prove
    the fix bypasses that fallback when doctor schedule data is present.
    """
    # Patch safe_get_branches to return only the lab-only Гагарина 64 branch,
    # proving the fix does NOT use this fallback when doctor schedule is available.
    monkeypatch.setattr(
        "messengers_router.response_builder.safe_get_branches",
        lambda _svc: [{"id": "branch_9358", "name": _LAB_ONLY_BRANCH, "aliases": "гагарина 64 (самара)"}],
    )

    entities: dict = {
        "city": "Самара",
        "doctor_name": "Трубин Алексей Юрьевич",
        "patient_name": "Иванов Иван Иванович",
        "__appointment_mode": True,
    }
    if appointment_action:
        entities["appointment_action"] = appointment_action

    state = SessionState(
        session_id=f"test-bug-2026-06-07-01-{appointment_action or 'new'}",
        last_entities=entities,
    )

    evidence = Evidence(
        items={
            ek.DOCTOR_SCHEDULE: _TRUBIN_SCHEDULE_PAYLOAD,
            # ADDRESS evidence intentionally absent / empty
        }
    )

    memory = MemoryStore()
    svc = Services()

    env = build_appointment_step_response("APPOINTMENT", evidence, state, svc, memory)

    assert env is not None, "Expected a branch-step response, got None"
    text = str(env.text or "")

    # Core invariant: lab-only branch must NEVER appear.
    assert _LAB_ONLY_BRANCH not in text, (
        f"[action={appointment_action!r}] Lab-only branch {_LAB_ONLY_BRANCH!r} "
        f"must NOT appear in doctor branch offer; got: {text!r}"
    )
    # The doctor's own branch(es) should be offered.
    assert _DOCTOR_BRANCH_1 in text or _DOCTOR_BRANCH_2 in text, (
        f"[action={appointment_action!r}] Expected at least one of the doctor's "
        f"own branches ({_DOCTOR_BRANCH_1!r}, {_DOCTOR_BRANCH_2!r}) in the branch "
        f"offer; got: {text!r}"
    )
    # Postcondition: the stored branch options must also be the doctor's branches.
    stored = state.last_entities.get("appointment_branch_options")
    assert isinstance(stored, list) and stored, (
        f"[action={appointment_action!r}] appointment_branch_options should be "
        f"populated after branch step; got: {stored!r}"
    )
    assert _LAB_ONLY_BRANCH not in stored, (
        f"[action={appointment_action!r}] Lab-only branch must not be in stored "
        f"appointment_branch_options; got: {stored!r}"
    )


def test_specialty_only_path_still_gets_clarification_not_lab_branch(
    monkeypatch,
) -> None:
    """
    Regression guard: Fix C (BUG-2026-06-04-04) must remain intact.
    When specialty is set but NO doctor is selected and ADDRESS evidence is
    empty, safe_get_branches fallback should be suppressed → clarification
    prompt, not lab-only branch.
    (Mirrors test_response_builder_appointment_specialty_empty_evidence_offers_clarification
    in test_messenger_services.py — kept here as a cross-reference anchor.)
    """
    from messengers_router.policies import APPOINTMENT_STEP_BRANCH

    monkeypatch.setattr(
        "messengers_router.response_builder.safe_get_branches",
        lambda _svc: [{"id": "branch_9358", "name": _LAB_ONLY_BRANCH, "aliases": "гагарина 64 (самара)"}],
    )

    state = SessionState(
        session_id="test-fix-c-regression-guard",
        last_entities={
            "city": "Самара",
            "specialty": "кардиолог",
            "__appointment_mode": True,
            "appointment_step": APPOINTMENT_STEP_BRANCH,
        },
    )
    evidence = Evidence(
        items={
            ek.ADDRESS: {"addresses": [], "branches": [], "note": "empty"},
        }
    )
    memory = MemoryStore()
    svc = Services()

    env = build_appointment_step_response("APPOINTMENT", evidence, state, svc, memory)

    assert env is not None
    text = str(env.text or "")
    assert _LAB_ONLY_BRANCH not in text, (
        f"Fix C regression: lab-only branch must NOT appear for specialty-only path; "
        f"got: {text!r}"
    )
    assert "уточните" in text.lower() or "адрес" in text.lower() or "филиал" in text.lower(), (
        f"Fix C regression: expected clarification prompt for specialty-only empty-address path; "
        f"got: {text!r}"
    )
