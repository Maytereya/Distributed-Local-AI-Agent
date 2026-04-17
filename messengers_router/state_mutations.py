"""Typed setters for appointment-domain state mutations.

Stage 11 of the refactor plan (OPTION B pilot): wraps the direct mutations of
``state.last_entities[...]`` and ``state.dialog.phase`` that drive the
APPOINTMENT flow so the key names stop leaking to 20+ call sites.

Scope is intentionally narrow — appointment only — so we can validate the
pattern before generalising to doctor/service/patient/price domains. Each
setter is a dumb assignment plus (where relevant) a phase transition; no
validation beyond ``None``/value checks. That keeps the migration
behaviour-preserving.
"""

from __future__ import annotations

from typing import Any

from .mess_types import AppointmentPhase, SessionState

# Canonical appointment-flow keys. Kept module-private so callers go through
# the setters below.
_FLOW_ACTIVE = "appointment_flow_active"
_CONFIRM_PENDING = "appointment_confirm_pending"
_CONFIRMED = "appointment_confirmed"
_CANCEL_PENDING = "appointment_cancel_pending"
_TOPIC_SWITCH_PENDING = "appointment_topic_switch_pending"
_SELECTION_MODE = "appointment_selection_mode"
_BRANCH_OPTIONS = "appointment_branch_options"

# Bundle of all transient confirmation flags we clear when the flow exits
# (either completed or abandoned).
_CONFIRMATION_FLAGS = (
    _FLOW_ACTIVE,
    _CONFIRM_PENDING,
    _CONFIRMED,
    _CANCEL_PENDING,
    _TOPIC_SWITCH_PENDING,
)


# --- Flow lifecycle -------------------------------------------------------

def activate_appointment_flow(state: SessionState) -> None:
    """Start the appointment collecting phase."""

    state.last_entities[_FLOW_ACTIVE] = True
    state.dialog.phase = AppointmentPhase.COLLECTING


def deactivate_appointment_flow(state: SessionState) -> None:
    """Drop the active-flow marker without touching phase."""

    state.last_entities.pop(_FLOW_ACTIVE, None)


# --- Confirmation step ---------------------------------------------------

def mark_appointment_confirm_pending(state: SessionState) -> None:
    """Transition to the CONFIRM phase awaiting user ack."""

    state.last_entities[_CONFIRM_PENDING] = True
    state.dialog.phase = AppointmentPhase.CONFIRM


def clear_appointment_confirm_pending(state: SessionState) -> None:
    state.last_entities.pop(_CONFIRM_PENDING, None)


def finalize_appointment_confirmation(state: SessionState) -> None:
    """User confirmed — close the flow and mark as confirmed."""

    state.last_entities[_CONFIRMED] = True
    state.last_entities.pop(_CONFIRM_PENDING, None)
    state.last_entities.pop(_FLOW_ACTIVE, None)
    state.dialog.phase = AppointmentPhase.CONFIRMED


def reject_appointment_confirmation(state: SessionState, extra_keys: tuple[str, ...] = ()) -> None:
    """User rejected confirmation — rewind to COLLECTING and reopen the flow."""

    state.last_entities[_CONFIRMED] = False
    state.last_entities.pop(_CONFIRM_PENDING, None)
    for key in extra_keys:
        state.last_entities.pop(key, None)
    state.last_entities[_FLOW_ACTIVE] = True
    state.dialog.phase = AppointmentPhase.COLLECTING


def reactivate_appointment_collecting(state: SessionState) -> None:
    """Rewind to COLLECTING phase, dropping any confirmation markers.

    Used on datetime follow-up paths in ``router.py`` that want to resume the
    appointment flow without preserving a stale confirm-pending or confirmed
    marker from an earlier turn. Does not touch ``patient_name`` — callers
    decide whether to drop it based on the surrounding context.
    """

    state.last_entities.pop(_CONFIRM_PENDING, None)
    state.last_entities.pop(_CONFIRMED, None)
    state.dialog.phase = AppointmentPhase.COLLECTING


def reset_appointment_confirmation_flags(state: SessionState) -> None:
    """Drop every transient confirmation marker and put dialog into IDLE.

    Used by the cancel branch in ``build_appointment_step_response`` — after
    we've rendered the handoff message, we clear everything so the next turn
    starts clean.
    """

    for key in _CONFIRMATION_FLAGS:
        state.last_entities.pop(key, None)
    state.dialog.phase = AppointmentPhase.IDLE


# --- Interruption pendings -----------------------------------------------

def mark_appointment_cancel_pending(state: SessionState) -> None:
    state.last_entities[_CANCEL_PENDING] = True


def clear_appointment_cancel_pending(state: SessionState) -> None:
    state.last_entities.pop(_CANCEL_PENDING, None)


def mark_appointment_topic_switch_pending(state: SessionState) -> None:
    state.last_entities[_TOPIC_SWITCH_PENDING] = True


def clear_appointment_topic_switch_pending(state: SessionState) -> None:
    state.last_entities.pop(_TOPIC_SWITCH_PENDING, None)


# --- Selection mode (branch vs doctor) -----------------------------------

def set_appointment_selection_mode(state: SessionState, mode: str) -> None:
    state.last_entities[_SELECTION_MODE] = mode


def clear_appointment_selection_mode(state: SessionState) -> None:
    state.last_entities.pop(_SELECTION_MODE, None)


# --- Branch options list --------------------------------------------------

def set_appointment_branch_options(state: SessionState, options: list[Any]) -> None:
    state.last_entities[_BRANCH_OPTIONS] = options


def clear_appointment_branch_options(state: SessionState) -> None:
    state.last_entities.pop(_BRANCH_OPTIONS, None)
