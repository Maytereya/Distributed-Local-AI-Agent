# DialogState Migration: Replace appointment boolean flags with state.dialog.phase

**Branch:** `refactor/core` → merges into `origin/release`  
**Test command:** `cd /Users/maxten/Dev/Distributed-Local-AI-Agent2 && PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`  
**Current baseline:** 514 passed, 0 failed — must stay green after every task.

---

## Background

`DialogState` and `AppointmentPhase` types exist in `messengers_router/mess_types.py`. Every `SessionState` already has a `dialog: DialogState` field. However, the appointment flow state machine still uses boolean flags written to `state.last_entities`:

| Current flag | Meaning | Target |
|---|---|---|
| `appointment_flow_active = True` | User is in booking flow | `state.dialog.phase = AppointmentPhase.COLLECTING` |
| `appointment_confirm_pending = True` | Awaiting yes/no confirmation | `state.dialog.phase = AppointmentPhase.CONFIRM` |
| `appointment_confirmed = True` | Confirmed; handoff pending | `state.dialog.phase = AppointmentPhase.CONFIRMED` |
| `appointment_cancel_pending = True` | Asking "really cancel?" | Keep in `last_entities` (transient, 1-turn state) |
| `appointment_topic_switch_pending = True` | Asking "switch topic?" | Keep in `last_entities` (transient, 1-turn state) |

The migration maps the **three persistent phase flags** to `state.dialog.phase`. The two transient flags (`cancel_pending`, `topic_switch_pending`) stay in `last_entities` because they are cleared within 1–2 turns.

---

## Files to modify

| File | Lines with flags |
|---|---|
| `messengers_router/appointment_flow_guard.py` | 236, 240, 272, 306, 314, 320, 334, 348, 352–354, 367–368, 371, 384, 398 |
| `messengers_router/router.py` | 959, 1355–1356, 1409–1410, 1519, 1649, 1651–1652, 1687, 1815, 2123, 2211 |

---

## Strategy

Do this in **two phases**:

1. **Phase 1** — Write to BOTH `last_entities` and `state.dialog.phase`. Read from `last_entities` as before. This is a safe dual-write that keeps everything working while you migrate.
2. **Phase 2** — Switch reads to use `state.dialog.phase`. Remove dual-writes once reads are confirmed correct.

This document covers **Phase 1 only**. Phase 2 can follow after Phase 1 is green for a release.

---

## Phase 1 implementation

### Step 1 — appointment_flow_guard.py

**Read the file first** (`messengers_router/appointment_flow_guard.py`, 424 lines).

At every place where `state.last_entities["appointment_flow_active"] = True` is written, add a parallel write:

```python
# OLD:
state.last_entities["appointment_flow_active"] = True

# NEW:
state.last_entities["appointment_flow_active"] = True  # keep for compat
state.dialog.phase = AppointmentPhase.COLLECTING
```

At every place where `appointment_confirm_pending` is set to `True`:

```python
# OLD:
state.last_entities["appointment_confirm_pending"] = True

# NEW:
state.last_entities["appointment_confirm_pending"] = True
state.dialog.phase = AppointmentPhase.CONFIRM
```

At every place where `appointment_confirmed` is set to `True`:

```python
# OLD:
state.last_entities["appointment_confirmed"] = True

# NEW:
state.last_entities["appointment_confirmed"] = True
state.dialog.phase = AppointmentPhase.CONFIRMED
```

At every place where `appointment_confirm_pending` is **popped/cleared** AND `appointment_flow_active` is **popped/cleared** simultaneously (meaning the flow just ended):

```python
# OLD:
state.last_entities.pop("appointment_confirm_pending", None)
state.last_entities.pop("appointment_flow_active", None)

# NEW:
state.last_entities.pop("appointment_confirm_pending", None)
state.last_entities.pop("appointment_flow_active", None)
state.dialog.phase = AppointmentPhase.IDLE  # = "" — clears the phase
```

At every place where `appointment_confirm_pending` is popped but `appointment_flow_active` is kept (meaning we're back to collecting):

```python
# OLD:
state.last_entities.pop("appointment_confirm_pending", None)
state.last_entities["appointment_flow_active"] = True

# NEW:
state.last_entities.pop("appointment_confirm_pending", None)
state.last_entities["appointment_flow_active"] = True
state.dialog.phase = AppointmentPhase.COLLECTING
```

**Exact locations in appointment_flow_guard.py:**

Line 240:
```python
state.last_entities["appointment_confirm_pending"] = True
# add: state.dialog.phase = AppointmentPhase.CONFIRM
```

Line 272:
```python
state.last_entities["appointment_flow_active"] = True
# add: state.dialog.phase = AppointmentPhase.COLLECTING
```

Lines 352–354 (confirmed YES path):
```python
state.last_entities["appointment_confirmed"] = True
state.last_entities.pop("appointment_confirm_pending", None)
state.last_entities.pop("appointment_flow_active", None)
# add: state.dialog.phase = AppointmentPhase.CONFIRMED
```

Lines 367–371 (confirmed NO path — back to collecting):
```python
state.last_entities["appointment_confirmed"] = False
state.last_entities.pop("appointment_confirm_pending", None)
# ... (date key pops) ...
state.last_entities["appointment_flow_active"] = True
# add after: state.dialog.phase = AppointmentPhase.COLLECTING
```

---

### Step 2 — router.py

**Read the file first** (`messengers_router/router.py`, 2280 lines).

Apply the same dual-write pattern at every flag-write location.

**Exact locations:**

Line 959: reads `state.last_entities.get("appointment_flow_active")` — **no change** (read-only, Phase 1 does not change reads).

Line 1355–1356: reads — no change.

Line 1409–1410:
```python
state.last_entities.pop("appointment_confirm_pending", None)
state.last_entities.pop("appointment_confirmed", None)
# These are pops (clearing). Add:
# state.dialog.phase = AppointmentPhase.IDLE
# But ONLY if the flow is actually ending here — check surrounding context first.
```
Look at the surrounding 10 lines before modifying. If the code then sets `appointment_flow_active = True` a few lines later, do NOT clear `dialog.phase` here.

Line 1519: reads — no change.

Line 1649:
```python
if not bool(state.last_entities.get("appointment_flow_active")) and not looks_like_patient_fio(user_text):
    state.last_entities.pop("appointment_confirm_pending", None)
    state.last_entities.pop("appointment_confirmed", None)
    # Add:
    # state.dialog.phase = AppointmentPhase.IDLE
```

Line 1687: reads — no change.

Line 1815: reads — no change.

Line 2123: reads — no change.

Line 2211:
```python
state.last_entities["appointment_flow_active"] = True
# Add:
# state.dialog.phase = AppointmentPhase.COLLECTING
```

---

### Step 3 — reset_appointment_runtime_state() in appointment_flow_guard.py

This function (line 132) already calls `dialog.clear()` via the `AppointmentPhase.is_active()` check. Verify it sets `dialog.phase = ""` on reset. The existing code is:

```python
def reset_appointment_runtime_state(state: SessionState) -> None:
    for key in _APPOINTMENT_RUNTIME_KEYS:
        state.last_entities.pop(key, None)
    dialog: DialogState = state.dialog
    if dialog.is_active() and (
        dialog.label == "APPOINTMENT" or AppointmentPhase.is_active(dialog.phase)
    ):
        dialog.clear()
```

`dialog.clear()` sets `phase = ""` which equals `AppointmentPhase.IDLE`. This is already correct. No change needed.

---

## What NOT to do

1. Do NOT change any **reads** of `state.last_entities.get("appointment_flow_active")` in Phase 1. Phase 1 is writes-only.
2. Do NOT remove `state.last_entities["appointment_flow_active"] = True` lines — keep them alongside the new `state.dialog.phase` writes.
3. Do NOT touch `appointment_cancel_pending` or `appointment_topic_switch_pending` — these transient flags stay in `last_entities` permanently.
4. Do NOT modify tests — the dual-write approach keeps all existing test assertions valid.
5. Read each surrounding context block (±15 lines) before modifying, especially for lines 1409–1410 in router.py where the semantics depend on what comes before/after.

---

## Verification

After all changes:
```
PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval
```

Must show 514+ passed, 0 failed.

Also confirm the `AppointmentPhase` import is present in `appointment_flow_guard.py`:
```python
from .mess_types import AppointmentPhase, DialogState, ResponseEnvelope, SessionState
```
It's already there at line 15 — no import change needed.

---

## Commit message

```
refactor(appointment): dual-write dialog.phase alongside last_entities boolean flags

Phase 1 of DialogState migration: all writes to appointment_flow_active,
appointment_confirm_pending, and appointment_confirmed now also set
state.dialog.phase to the corresponding AppointmentPhase constant.
Reads still use last_entities for compat. Phase 2 will switch reads.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```
