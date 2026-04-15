# Refactor Handoff: messengers_router LLM-First Refactor

**Branch:** `refactor/core` → merges into `origin/release`  
**Python:** `venv/bin/pytest` with `PYTHONPATH=.`  
**Test command:** `cd /Users/maxten/Dev/Distributed-Local-AI-Agent2 && PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`  
**Never break:** 510 tests must stay green after every task.

---

## Context (read this first)

This is a medical chatbot (Ollama, Russian, WhatsApp) for a Samara clinic. The codebase lives in `messengers_router/`. The refactor goal is LLM-first NLU, typed state management, and elimination of code duplication.

**Root cause of prod bugs:** `nlu_pipeline._merge()` used to silently promote regex intent over LLM when `llm.confidence < 0.45/0.55`. This caused correct LLM classifications to be overridden by crude keyword matches when user phrasing varied naturally.

---

## What is ALREADY DONE (do NOT redo)

All committed on branch `refactor/core`.

| SHA | What was done |
|-----|--------------|
| `e46968f` | Created `messengers_router/russian_nlu.py` — `normalize_ru()` + `ENTITY_WHITELIST` frozenset |
| `27c1337` | Replaced 14 inline `.lower().replace("ё","е")` copies in classifier, entity_grounder, llm_doesnt_work_fallback, policies with `normalize_ru()` |
| `5939fe9` | Removed duplicate `ALLOWED_ENTITY_KEYS` from `prompt_contracts.py` and `_ALLOWED_ENTITY_KEYS` from `classifier.py`; both now import `ENTITY_WHITELIST` from `russian_nlu` |
| `e26ee7f` | Collapsed `_extract_schedule_doctor_name`, `_extract_appointment_doctor_name`, `_extract_price_doctor_name` in `classifier.py` into single `_extract_doctor_name(text, *, mode)` |
| `e781a3b` | Added `ConfidencePolicy` frozen dataclass + `CONFIDENCE` singleton to `mess_types.py` (named thresholds: `rule_hardcode=0.85`, `high=0.75`, `moderate=0.70`, `price_floor=0.55`, `refine_min=0.45`, `llm_promote_rich=0.45`, `llm_promote_hybrid=0.55`, `promoted_floor=0.60`, `llm_default=0.20`) |
| `389bbd4` | **Core fix:** rewrote `_merge()` in `nlu_pipeline.py` — LLM intent always wins; rule only donates entities to fill gaps; safety labels (URGENT/COMPLAINT/MEDICAL_ADVICE) remain hard overrides |
| `e7c626c` | Replaced 11 magic confidence floats in `classifier.py` with `CONFIDENCE.*` fields; fixed stale test reference `_extract_appointment_doctor_name` → `_extract_doctor_name(mode="appointment")` |
| `84707cb` | Added `DialogState` dataclass to `mess_types.py` (fields: `label`, `phase`, `entities`, `candidate_entities`, `missing_slots`, `clarify_count`, `open_question`, `confidence`; methods: `is_active()`, `clear()`, `merge_entities()`); added `dialog: DialogState` field to `SessionState` |
| `8fabc6d` | Added `save_dialog_state()` / `load_dialog_state()` to `MemoryStore` in `memory.py` |
| `10c92eb` | Added `AppointmentPhase` class to `mess_types.py` (constants: `IDLE=""`, `COLLECTING`, `CONFIRM`, `CONFIRMED`, `CANCEL_CONFIRM`; class methods: `values()`, `is_active(phase)`) |
| `24f2d78` | Completed Task 4.2: extracted `_is_topic_switch_intent()` in `appointment_flow_guard.py`, added `DialogState` reset sync, and verified with full test suite |
| `ad35894` | Completed Task 5.1: added `messengers_router/orchestrator.py` with the 5-stage pipeline skeleton and `tests/test_orchestrator_pipeline.py` |
| `de827ea` | Completed Task 5.2: added `MR_USE_ORCHESTRATOR` runtime flag and wired the A/B gate through `patient_routing_stream()` |

Tests added: `test_russian_nlu.py`, `test_confidence_policy.py`, `test_nlu_merge_policy.py`, `test_dialog_state.py`, `test_memory_dialog_state.py`

## Session Update (2026-04-15)

- Task 4.2 is done in `24f2d78`.
- Task 5.1 is done in `ad35894`.
- Task 5.2 is done in `de827ea`.
- Task 6.1 is done in `84d2522`.
- Task 6.2 is done in the latest `refactor/core` commit after Task 6.1.
- Current green baseline: `510 passed, 3 warnings`.

### Important implementation notes for the next agent

1. **Task 5.2 deviation from the original text:** the orchestrator gate was added in `patient_routing_stream()`, not `route_patient_message()`. Reason: `route_patient_message()` returns `(RouteDecision, Plan, Evidence)`, while the orchestrator currently returns `ResponseEnvelope` through `ctx.response`. Putting the gate into the stream preserves API compatibility and still gives a real A/B entry point.
2. **Task 6.1 facade compatibility requirement:** `messengers_router.services` is not a dumb `import *` facade. The package `services/__init__.py` must preserve:
   - direct attribute access to underscore helpers (example: `_prepare_roots_match`)
   - `monkeypatch.setattr(svc_mod, ...)` compatibility for tests that expect old `services.py` module semantics
   The current implementation uses a proxy module class to sync top-level monkeypatch assignments into `services_legacy`.
3. **Relative import correction for Task 6.1:** from inside `messengers_router/services/__init__.py`, the legacy module must be imported as sibling `messengers_router.services_legacy` via `from .. import services_legacy`, not `from .services_legacy`.
4. **Task 6.2 extraction pattern:** the pilot migration does not rewrite the `Services` class body inline. Instead, doctor-domain methods are implemented in `messengers_router/services/doctors.py` and rebound onto `Services` at the bottom of `services_legacy.py`. This keeps the step small and preserves the external `Services` API.
5. **Task 6.2 circular-import avoidance:** `services/doctors.py` uses a lazy helper (`_legacy_module()`) to access shared helpers from `services_legacy` at runtime. Do not replace this with a top-level `from .. import services_legacy` import unless you intentionally redesign the import graph.

---

## What STILL NEEDS TO BE DONE

Tasks are ordered — do them in sequence. Each task is a single focused commit.
Tasks 4.2, 5.1, 5.2, 6.1, and 6.2 are already done. Do not redo them. Historical task definitions are kept below only as implementation context.

---

### Task 4.2 — DONE (`24f2d78`) — Collapse 3 repeated intent-check chains in `appointment_flow_guard.py`

**File:** `messengers_router/appointment_flow_guard.py`

**Problem:** Three functions (`should_keep_appointment_flow_override`, `is_new_topic_while_confirm_pending`, `is_likely_topic_switch_from_appointment`) each contain nearly identical blocks like:

```python
if (
    detect_test_result_intent(low)
    or detect_test_assist_intent(low)
    or detect_prepare_intent(low)
    or detect_price_intent(low)
    or detect_address_intent(low)
    or detect_schedule_intent(low)
    or detect_doctor_info_intent(low)
    or detect_doc_request_intent(low)
    or detect_nonbookable_walkin_intent(text)
):
    return True/False
```

**Fix:** Extract a private helper `_is_topic_switch_intent(text: str) -> bool` that encapsulates the full union of these detectors. Then replace the repeated blocks in each of the three functions with a call to `_is_topic_switch_intent(text)`.

**The helper should call** (check what each function actually uses — they differ slightly):
- `detect_test_result_intent(low)`
- `detect_test_assist_intent(low)`
- `detect_prepare_intent(low)`
- `detect_price_intent(low)`
- `detect_address_intent(low)`
- `detect_schedule_intent(low)` 
- `detect_doctor_info_intent(low)`
- `detect_doc_request_intent(low)`
- `detect_nonbookable_walkin_intent(text)` (takes original text, not lowercased)
- `detect_news_intent(low)`

Use the **union** — include all detectors from all three functions. Each calling function can still add its own extra conditions after the helper call (e.g., `is_new_topic_while_confirm_pending` also checks `("?" in text) and (len(text.split()) >= 4)`).

**Also wire in `AppointmentPhase`:** In `reset_appointment_runtime_state()`, also call `state.dialog.clear()` if `state.dialog.is_active()` — this keeps the typed `DialogState` in sync when the boolean flags are cleared.

The import of `AppointmentPhase` and `DialogState` should come from `.mess_types`.

**Tests:** Run the full suite. No new test file required for this one — the existing baseline covered the behavior.

**Commit message:**
```
refactor(appointment): extract _is_topic_switch_intent helper + wire AppointmentPhase reset

Collapsed 3 near-identical intent-check chains in appointment_flow_guard.py
into a single _is_topic_switch_intent(text) helper. Also wires state.dialog.clear()
into reset_appointment_runtime_state() to keep DialogState in sync with flag resets.
```

---

### Task 5.1 — DONE (`ad35894`) — Create `messengers_router/orchestrator.py` with 5 named pipeline stages

**File to CREATE:** `messengers_router/orchestrator.py`

This is a new file. It implements a clean 5-stage pipeline that will eventually replace the 600-line `route_patient_message()` in `router.py`. For now it is **additive only** — `router.py` is not touched yet.

**The 5 stages:**

```python
# messengers_router/orchestrator.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .mess_types import DialogState, RouteDecision, SessionState, ResponseEnvelope

@dataclass
class OrchestratorContext:
    """Carries state through the 5-stage pipeline."""
    text: str
    state: SessionState
    decision: RouteDecision | None = None
    should_clarify: bool = False
    clarify_text: str = ""
    tool_results: dict[str, Any] = field(default_factory=dict)
    response: ResponseEnvelope | None = None
    short_circuit: bool = False  # True = skip remaining stages
    short_circuit_reason: str = ""
```

The 5 stages are **async functions** with this signature pattern:
```python
async def stage_name(ctx: OrchestratorContext, ...) -> OrchestratorContext:
```

**Stage 1: `early_guards(ctx)`**
- Check for URGENT / COMPLAINT / MEDICAL_ADVICE via `classifier.deterministic_rule_decision()`
- If a safety label fires → set `ctx.decision`, set `ctx.short_circuit = True`, `ctx.short_circuit_reason = "safety"`
- Return `ctx`

**Stage 2: `nlu_route(ctx, runtime_options)`**
- Call `nlu_pipeline.analyze_with_candidates(ctx.text, ctx.state, runtime_options)`
- Store the `NLUResult.decision` into `ctx.decision`
- Call `ctx.state.dialog.merge_entities(ctx.decision.entities)`
- Update `ctx.state.dialog.label = ctx.decision.label`
- Update `ctx.state.dialog.confidence = ctx.decision.confidence`
- Return `ctx`

**Stage 3: `clarify_gate(ctx)`**
- If `ctx.decision.clarify_needed` is True → set `ctx.should_clarify = True`, `ctx.clarify_text = ctx.decision.clarify_reason`
- If `ctx.state.dialog.missing_slots` is non-empty → set `ctx.should_clarify = True`
- Return `ctx`

**Stage 4: `tool_loop(ctx, services)`** *(stub for now)*
- If `ctx.should_clarify` → skip (return immediately)
- Otherwise: placeholder — `ctx.tool_results = {}` — real tool dispatch comes later
- Return `ctx`

**Stage 5: `render(ctx)`** *(stub for now)*
- If `ctx.should_clarify` → `ctx.response = ResponseEnvelope(text=ctx.clarify_text)`
- Otherwise: `ctx.response = ResponseEnvelope(text="")` — real rendering comes later
- Return `ctx`

**Main entry point:**
```python
async def run_pipeline(
    text: str,
    state: SessionState,
    runtime_options=None,
) -> OrchestratorContext:
    ctx = OrchestratorContext(text=text, state=state)
    ctx = await early_guards(ctx)
    if ctx.short_circuit:
        return ctx
    ctx = await nlu_route(ctx, runtime_options)
    ctx = await clarify_gate(ctx)
    ctx = await tool_loop(ctx, services=None)
    ctx = await render(ctx)
    return ctx
```

**Important:** Import `classifier` and `nlu_pipeline` lazily inside the function body (not at module top level) if there are circular import issues. The module must be importable without starting Ollama.

**Tests to write** (`tests/test_orchestrator_pipeline.py`):
- Test that `OrchestratorContext` has all fields with correct defaults
- Test that `early_guards` sets `short_circuit=True` when passed a mock that returns URGENT
- Test that `clarify_gate` sets `should_clarify=True` when `decision.clarify_needed=True`
- Test that `run_pipeline` returns an `OrchestratorContext` (mock the NLU to avoid Ollama)

**Commit message:**
```
feat(orchestrator): add 5-stage pipeline skeleton in orchestrator.py

Stages: early_guards → nlu_route → clarify_gate → tool_loop → render.
OrchestratorContext dataclass carries state through the pipeline.
tool_loop and render are stubs; real dispatch/rendering comes in Task 5.2+.
```

---

### Task 5.2 — DONE (`de827ea`) — Wire `router.py` to call orchestrator behind `MR_USE_ORCHESTRATOR=1` flag

**File:** `messengers_router/router.py`  
**File:** `messengers_router/runtime_config.py` (add flag)

**Step 1:** Add `MR_USE_ORCHESTRATOR: bool = False` to `runtime_config.py` (read from env var `MR_USE_ORCHESTRATOR`).

Check how existing flags like `MR_NLU_ENGINE` are read in `runtime_config.py` — follow the same pattern.

**Step 2:** In `router.py`, find `route_patient_message()` (the main entry point, ~600 lines). At the very top of the function body, add:

```python
from .runtime_config import config as _c
if _c.MR_USE_ORCHESTRATOR:
    from .orchestrator import run_pipeline
    ctx = await run_pipeline(text, state, runtime_options=runtime_options)
    # For now: if orchestrator returned a response, use it; otherwise fall through
    if ctx.response and ctx.response.text:
        return ctx.response
```

This is an A/B gate — `MR_USE_ORCHESTRATOR=0` (default) keeps the old path untouched.

**Tests:** Set `MR_USE_ORCHESTRATOR=1` via monkeypatch and assert the function returns a `ResponseEnvelope` without crashing. The orchestrator stubs are enough for this test.

**Commit message:**
```
feat(router): wire orchestrator behind MR_USE_ORCHESTRATOR env flag

When MR_USE_ORCHESTRATOR=1, route_patient_message() delegates to the new
5-stage pipeline. Default is 0 (old path). Allows A/B testing without
touching production behavior.
```

---

### Task 6.1 — DONE (current HEAD after next commit) — Create `messengers_router/services/` package skeleton

**Goal:** Begin extracting the 3000-line `services.py` into domain-focused submodules. This is additive — `services.py` is NOT deleted yet.

**Step 1:** Create the directory + files:
```
messengers_router/services/__init__.py     # re-exports everything from services_legacy
messengers_router/services/_legacy.py      # symlink shim (see below)
```

Actually, the cleanest approach: **rename `services.py` → `services_legacy.py`** (a git mv), then create `messengers_router/services/__init__.py` that does:

```python
# messengers_router/services/__init__.py
"""Services package — domain-specific API adapters.

During migration, all symbols are re-exported from services_legacy.
New domain modules will be added as submodules and removed from legacy.
"""
from .. import services_legacy as _legacy
```

Status note: `services.py` did **not** define `__all__`, and plain `import *` was not enough to preserve monkeypatch compatibility. The working implementation uses:
- `globals().update(...)` to mirror the legacy namespace, including underscore helpers
- a proxy module class that forwards top-level `setattr` / `delattr` calls into `services_legacy`

**Important:** After renaming, search for `from .services import` and `from messengers_router.services import` across the codebase and update any imports that break. Run the full test suite to confirm nothing broke.

**Step 2:** Create placeholder submodule files (empty for now):
```
messengers_router/services/doctors.py     # will hold doctor schedule/info logic
messengers_router/services/appointments.py # will hold appointment booking logic
messengers_router/services/lab_tests.py   # will hold test result / prepare logic
messengers_router/services/addresses.py   # will hold address/branch logic
messengers_router/services/prices.py      # will hold pricing logic
```

Each placeholder should just have a module docstring saying what it will contain. No code yet.

**Commit message:**
```
refactor(services): create services/ package skeleton, rename services.py → services_legacy.py

services/__init__.py re-exports everything from services_legacy for zero-breakage
migration. Placeholder submodule files created for: doctors, appointments,
lab_tests, addresses, prices. Full test suite must pass.
```

---

### Task 6.2 — DONE (latest `refactor/core` commit) — Move one domain into its submodule (doctors.py as pilot)

**This is a pilot migration.** Move doctor-related functions from `services_legacy.py` into `services/doctors.py`.

**Step 1:** Read `services_legacy.py` and identify all functions/classes related to doctor info and doctor schedule (look for `get_doctor`, `fetch_doctor`, `doctor_info`, `schedule`, etc.).

**Step 2:** Move those functions to `services/doctors.py`. Add proper imports.

**Step 3:** In `services_legacy.py`, replace the moved functions with imports from `.services.doctors`:
```python
from .services.doctors import get_doctor_info, get_doctor_schedule  # etc.
```

**Step 4:** `services/__init__.py` continues to re-export everything — callers don't need to change.

**Step 5:** Run full test suite.

**Commit message:**
```
refactor(services): migrate doctor functions to services/doctors.py

Pilot domain extraction. services_legacy.py re-imports from doctors.py.
All external callers unchanged via services/__init__.py facade.
```

---

### Task 7.1 — Full eval run + any fixes

**Run:**
```bash
cd /Users/maxten/Dev/Distributed-Local-AI-Agent2
PYTHONPATH=. venv/bin/pytest tests/ -q --ignore=tests/eval 2>&1 | tail -5
```

If there are failures from earlier tasks, fix them before proceeding.

Also check for remaining inline `.lower().replace("ё", "е")` copies that were noted in scope but not yet migrated (these are in `router.py`, `services_legacy.py`, `renderer.py`, `flow_policy.py`, `appointment_flow_guard.py`, `topic_registry.py`, `doctor_name_port.py`). Fix them by importing `normalize_ru` from `russian_nlu` — follow the exact same pattern as Task 1.2.

**Commit message:**
```
refactor(nlu): roll out normalize_ru to remaining files in router/renderer/flow_policy
```

---

### Task 7.2 — Simplify `llm_doesnt_work_fallback.py` stemmer (optional, low risk)

**File:** `messengers_router/llm_doesnt_work_fallback.py`

The hand-rolled Russian stemmer in this file has magic weights and is fragile. This is a **low priority** task — only do it if time permits and all other tasks are green.

The stemmer only runs when Ollama is completely down. It's a safety net, not a main path. Document it with a comment explaining its purpose rather than rewriting it.

---

### Task 7.3 — Remove `MR_USE_ORCHESTRATOR` flag (LAST STEP, do after eval is green)

Only do this after Tasks 5.1, 5.2, and 7.1 are complete and the orchestrator is handling real traffic in tests.

**File:** `router.py`, `runtime_config.py`

Remove the `if _c.MR_USE_ORCHESTRATOR:` branch. Make `run_pipeline()` the sole routing path. Delete the `MR_USE_ORCHESTRATOR` entry from `runtime_config.py`.

**Commit message:**
```
feat(router): remove MR_USE_ORCHESTRATOR flag — orchestrator is now sole routing path
```

---

## Key files to know

| File | Purpose |
|------|---------|
| `messengers_router/mess_types.py` | All shared types: `Label`, `RouteDecision`, `SessionState`, `DialogState`, `AppointmentPhase`, `ConfidencePolicy`, `CONFIDENCE` |
| `messengers_router/russian_nlu.py` | `normalize_ru()` + `ENTITY_WHITELIST` — single source of truth |
| `messengers_router/nlu_pipeline.py` | NLU orchestration: `_merge()` (LLM-first), `analyze_with_candidates()` |
| `messengers_router/classifier.py` | LLM classification + deterministic rules + entity extraction |
| `messengers_router/memory.py` | `MemoryStore`: session state, pending slots, `save/load_dialog_state()` |
| `messengers_router/appointment_flow_guard.py` | Appointment flow state machine (boolean-flag based, being migrated to `AppointmentPhase`) |
| `messengers_router/router.py` | Main entry point `route_patient_message()` — 600 lines, target for orchestrator |
| `messengers_router/services.py` | 3000-line API adapter monolith — being split into `services/` package |

## Rules for the other agent

1. **Run tests after every task.** Command: `PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`
2. **Never touch `tests/eval/`** — those are the eval cases, not unit tests.
3. **All work on branch `refactor/core`**. Never push to `release` directly.
4. **One commit per task.** Use the commit messages above verbatim.
5. **The `Co-Authored-By` line must be in every commit:**
   ```
   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```
6. **Read files before editing** — the repo has a pre-tool hook that requires this.
7. **Do NOT edit `tests/eval/`**, `app_data/prompts/`, or any `.md` docs files except as directed.
8. **If a test fails that you didn't write**, investigate before assuming it's pre-existing. Check git blame.
