# Orchestrator Phase 2: Replace Legacy Bridge with Real tool_loop + render

**Branch:** `refactor/core` → merges into `origin/release`  
**Test command:** `cd /Users/maxten/Dev/Distributed-Local-AI-Agent2 && PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`  
**Current baseline:** 513 passed, 0 failed — must stay green after every task.

---

## What exists now (do NOT change)

The 5-stage orchestrator pipeline is live in `messengers_router/orchestrator.py`. All requests go through it. However `tool_loop()` is currently a **bridge stub** — it just calls the old monolithic `route_patient_message()` internally:

```python
# current tool_loop — this is the stub to replace
async def tool_loop(ctx, services, memory, runtime_options):
    if ctx.short_circuit or ctx.should_clarify:
        return ctx
    from . import router
    decision, plan, evidence = await router.route_patient_message(
        ctx.text, ctx.state, services, memory, runtime_options=runtime_options,
    )
    ctx.decision = decision
    ctx.plan = plan
    ctx.evidence = evidence
    return ctx
```

And `render()` only handles the `short_circuit` and `should_clarify` cases — for the normal path it returns `ctx.response = None` and lets `patient_routing_stream()` in router.py handle the real rendering downstream.

**The goal of this phase:** make `tool_loop()` and `render()` do the real work, so that `route_patient_message()` is no longer called at all. Once done, `route_patient_message()` becomes dead code and can be deleted.

---

## Architecture you need to understand

### Current flow (what happens right now):

```
patient_routing_stream()
  → run_pipeline()
      → early_guards()       ← safety check, may short_circuit
      → nlu_route()          ← LLM classification, updates ctx.decision
      → clarify_gate()       ← sets ctx.should_clarify if slots missing
      → tool_loop()          ← STUB: calls route_patient_message() internally
      → render()             ← STUB: only handles safety/clarify cases
  ← returns OrchestratorContext
  if ctx.response:           ← only safety or clarify responses come from here
      yield ctx.response
  else:
      decision/plan/evidence = ctx.decision/plan/evidence
      # ... 200 more lines of rendering in patient_routing_stream()
```

### Target flow (what you are building):

```
patient_routing_stream()
  → run_pipeline()
      → early_guards()
      → nlu_route()
      → clarify_gate()
      → tool_loop()          ← calls build_plan() + execute_plan() directly
      → render()             ← calls renderer.render_stream() / render_urgent() etc.
  ← returns OrchestratorContext with ctx.response always set
  yield ctx.response         ← always comes from orchestrator
```

---

## Key functions you will use

All already exist in the codebase. You are wiring them, not rewriting them.

### `router.build_plan(decision, state, user_text, memory, runtime_options) → Plan`
Located at `messengers_router/router.py` line ~1260.  
Takes the `RouteDecision`, builds the tool `Plan` to execute.

### `router.execute_plan(plan, state, services) → Evidence`
Located at `messengers_router/router.py` line ~1270. Async.  
Runs the plan against the backend APIs, returns `Evidence`.

### `renderer.render_stream(user_text, decision, evidence, runtime_options) → AsyncGenerator[str, None]`
Located at `messengers_router/renderer.py` line ~874. Async generator.  
Takes decision + evidence, generates the LLM response text chunks.

### `renderer.render_urgent() → ResponseEnvelope`
### `renderer.render_complaint() → ResponseEnvelope`  
### `renderer.render_medical_advice() → ResponseEnvelope`
Located at `messengers_router/renderer.py` lines 800–840. Sync.

### Pre-pending handlers (these run BEFORE NLU in current router, handle multi-turn state):
- `router._handle_appointment_action_pending()` — line ~488
- `router._handle_compound_price_pending()` — line ~593
- `router._handle_catalog_confirm_pending()` — line ~735

These 3 handlers intercept certain pending states and return early before NLU runs. Right now they live inside `route_patient_message()`. In the new pipeline they need to run **inside `tool_loop()`** (after `nlu_route` but with ability to short-circuit).

### `router._handle_secondary_queue()` logic (lines ~1370–1430 of router.py):
This handles the "secondary intent queue" — when a user was offered a secondary intent and replies "yes". Also needs to move into `tool_loop()`.

---

## The 3 tasks to implement

### Task A — Move pre-pending handlers into `tool_loop()`

**What to do:**

Replace the current `tool_loop()` stub in `orchestrator.py` with this logic:

```python
async def tool_loop(
    ctx: OrchestratorContext,
    services: Any | None = None,
    memory: Any | None = None,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    if ctx.short_circuit or ctx.should_clarify:
        return ctx
    if services is None or memory is None:
        return ctx

    from . import router

    # 1. Pre-pending handlers — may short-circuit with their own decision+evidence
    for handler in (
        router._handle_catalog_confirm_pending,
        router._handle_appointment_action_pending,
        router._handle_compound_price_pending,
    ):
        result = await handler(
            user_text=ctx.text,
            state=ctx.state,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
        if result is not None:
            ctx.decision, ctx.plan, ctx.evidence = result
            ctx.short_circuit = True
            ctx.short_circuit_reason = "pending_handler"
            return ctx

    # 2. Build plan from NLU decision
    plan = router.build_plan(
        ctx.decision,
        ctx.state,
        ctx.text,
        memory,
        runtime_options=runtime_options,
    )
    ctx.plan = plan

    # 3. Execute plan against services
    evidence = await router.execute_plan(plan, ctx.state, services)
    ctx.evidence = evidence

    return ctx
```

**Important:** The 3 handler functions (`_handle_catalog_confirm_pending`, `_handle_appointment_action_pending`, `_handle_compound_price_pending`) are private (underscore prefix). Calling them from orchestrator.py is acceptable during this migration phase since both files are in the same package. Do NOT make them public yet — that's a separate cleanup task.

**The secondary queue logic** (lines ~1370–1430 of router.py) is more complex and entangled with `memory.merge_entities`. Leave it in `route_patient_message()` for now — Task A only handles the three `_handle_*_pending` functions.

---

### Task B — Implement real `render()` in orchestrator

**What to do:**

Replace the current `render()` stub with a real implementation. The key insight is: `render_stream()` is an async generator (yields text chunks), but `ResponseEnvelope` holds a complete string. So you need to collect the generator output into one string.

```python
async def render(ctx: OrchestratorContext, runtime_options: Any | None = None) -> OrchestratorContext:
    from . import renderer

    # Safety short-circuits (already handled correctly — keep as-is)
    if ctx.short_circuit and ctx.decision is not None:
        if ctx.decision.label == "URGENT":
            ctx.response = renderer.render_urgent()
        elif ctx.decision.label == "COMPLAINT":
            ctx.response = renderer.render_complaint()
        elif ctx.decision.label == "MEDICAL_ADVICE":
            ctx.response = renderer.render_medical_advice()
        else:
            # pending_handler short-circuit: fall through to normal render below
            pass
        if ctx.response is not None:
            return ctx

    # Clarify: return the clarify text directly
    if ctx.should_clarify:
        ctx.response = ResponseEnvelope(text=ctx.clarify_text)
        return ctx

    # Normal path: collect render_stream output
    if ctx.decision is not None and ctx.evidence is not None:
        chunks: list[str] = []
        async for chunk in renderer.render_stream(
            ctx.text,
            ctx.decision,
            ctx.evidence,
            runtime_options=runtime_options,
        ):
            chunks.append(chunk)
        full_text = "".join(chunks)
        needs_handoff = bool(ctx.decision.needs_handoff)
        ctx.response = ResponseEnvelope(
            text=full_text,
            attachments=list(ctx.evidence.items.get("attachments") or []),
            handoff=needs_handoff,
        )
        return ctx

    # Fallback: empty response (should not happen in normal flow)
    ctx.response = ResponseEnvelope(text="")
    return ctx
```

**Also update `run_pipeline()`** to pass `runtime_options` to `render()`:
```python
ctx = await render(ctx, runtime_options=runtime_options)
```

---

### Task C — Simplify `patient_routing_stream()` in `router.py`

Once Tasks A and B are done, the orchestrator always sets `ctx.response`. The ~200-line rendering block in `patient_routing_stream()` after the orchestrator call can be simplified.

**Current code in `patient_routing_stream()` (around line 2055):**
```python
try:
    ctx = await run_pipeline(...)
    if ctx.response and ctx.response.text:
        if ctx.response.handoff:
            _reset_state_after_handoff(state, memory)
        yield ctx.response
        return
    decision = ctx.decision
    plan = ctx.plan
    evidence = ctx.evidence
    if decision is None or plan is None or evidence is None:
        raise RuntimeError(...)
except Exception as e:
    ...  # error handling
flow_label = plan.label
# ... 200 more lines of debug, rendering, history management
```

**After Tasks A+B, replace with:**
```python
try:
    ctx = await run_pipeline(
        user_text, state,
        services=services, memory=memory,
        runtime_options=runtime_options,
    )
except Exception as e:
    fallback_text = handoff_message("service_error")
    state_update: dict[str, Any] = {}
    if debug:
        state_update = {"debug": {"route_error": str(e)}}
    _reset_state_after_handoff(state, memory)
    yield ResponseEnvelope(
        text=fallback_text, attachments=[], handoff=True, state_update=state_update,
    )
    return

if ctx.response is None:
    # Should not happen — defensive fallback
    yield ResponseEnvelope(text=handoff_message("service_error"), handoff=True)
    return

if ctx.response.handoff:
    _reset_state_after_handoff(state, memory)

# Debug envelope (keep this)
if debug and ctx.decision and ctx.plan and ctx.evidence:
    pending = memory.get_pending(state)
    yield ResponseEnvelope(
        text="", attachments=[], handoff=False,
        state_update={"debug": _debug_meta(ctx.decision, ctx.plan, ctx.evidence, state, pending)},
    )

# History + summary update (keep these)
memory.append_turn(state, role="user", text=user_text)
memory.append_turn(state, role="assistant", text=ctx.response.text)
update_summary(state, reason="normal")

yield ctx.response
return
```

**CAUTION on Task C:** Before deleting any code from `patient_routing_stream()`, check each block carefully:
- History update (`memory.append_turn`) — must be preserved, move to after orchestrator
- Summary update (`update_summary`) — must be preserved
- Debug envelope — keep but simplify
- Intro text for first turn (`INTRO_TEXT`) — check if `render_stream` already handles this via `Evidence`; if not, keep it
- `_reset_state_after_handoff` — keep in error branch and handoff branch

**Do Task C last and test heavily.** If any doubt, leave `patient_routing_stream()` untouched and just let `ctx.response` be yielded. The 200 extra lines are dead code but harmless if the early `yield ctx.response; return` path always fires.

---

## Test strategy

**After Task A:** Run full suite. Write one test in `tests/test_orchestrator_pipeline.py`:
```python
async def test_tool_loop_calls_build_and_execute_plan(monkeypatch):
    # mock build_plan to return a Plan, mock execute_plan to return Evidence
    # assert ctx.plan and ctx.evidence are set after tool_loop
```

**After Task B:** Run full suite. Write one test:
```python
async def test_render_collects_stream_into_response_envelope(monkeypatch):
    # mock render_stream to yield ["hello ", "world"]
    # assert ctx.response.text == "hello world"
```

**After Task C:** Run full suite twice — once normally, once with `MR_DEBUG=1` set to exercise the debug branch.

---

## What NOT to do

1. **Do NOT delete `route_patient_message()`** until all 3 tasks are done and 513 tests still pass. The secondary queue logic still lives there.
2. **Do NOT touch** `early_guards()`, `nlu_route()`, `clarify_gate()` — they are correct as-is.
3. **Do NOT touch** the appointment precheck logic (`run_appointment_precheck()`) at the top of `patient_routing_stream()` — it runs before the orchestrator and is correct.
4. **Do NOT change** the `services/` package structure — it was just refactored.
5. **Read files before editing** — there is a pre-tool hook that requires this.

---

## Commit messages (use verbatim)

**Task A:**
```
refactor(orchestrator): replace legacy bridge in tool_loop with build_plan + execute_plan

tool_loop() now calls build_plan() + execute_plan() directly instead of
delegating to route_patient_message(). Pre-pending handlers
(_handle_catalog_confirm_pending, _handle_appointment_action_pending,
_handle_compound_price_pending) wired in as short-circuit checks.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Task B:**
```
refactor(orchestrator): implement real render() — collects render_stream into ResponseEnvelope

render() now calls renderer.render_stream() and joins the async generator
output into a complete ResponseEnvelope. run_pipeline() always returns
ctx.response != None after this change.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**Task C:**
```
refactor(router): simplify patient_routing_stream — orchestrator now owns full response

Removed ~200 lines of rendering logic from patient_routing_stream() that
is now handled inside the orchestrator pipeline. History + summary updates
preserved. route_patient_message() is now dead code (kept for one more
release before deletion).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## File locations quick reference

| File | Line range | What's there |
|------|-----------|--------------|
| `messengers_router/orchestrator.py` | full file | The pipeline to modify |
| `messengers_router/router.py` | 488–740 | The 3 `_handle_*_pending` functions |
| `messengers_router/router.py` | 1260–1272 | `build_plan()` + `execute_plan()` |
| `messengers_router/router.py` | 1274–end | `route_patient_message()` — read but don't modify |
| `messengers_router/router.py` | ~2055–2250 | `patient_routing_stream()` rendering block (Task C) |
| `messengers_router/renderer.py` | 800–840 | `render_urgent/complaint/medical_advice()` |
| `messengers_router/renderer.py` | 874–930 | `render_stream()` async generator |
| `tests/test_orchestrator_pipeline.py` | full file | Existing orchestrator tests — extend, don't break |
