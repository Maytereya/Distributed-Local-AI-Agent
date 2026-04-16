# Orchestrator Phase 3: Implement Real tool_loop + render

**Branch:** `refactor/core` → merges into `origin/release`  
**Test command:** `cd /Users/maxten/Dev/Distributed-Local-AI-Agent2 && PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`  
**Current baseline:** 514 passed, 0 failed — must stay green after every task.

---

## Progress update (2026-04-16, Commit 1 completed)

Task B is now complete on `refactor/core`.

- `render()` no longer returns `None` for the normal path.
- It now executes the same response-ordering that previously lived in
  `patient_routing_stream()`:
  - safety short-circuits
  - clarify gate
  - greeting shortcut
  - recovery-policy output
  - pending-clarification output
  - deterministic/pre-built responses via `response_builder.py`
  - `render_stream()` fallback
- Deterministic builders are reused instead of reimplemented.
- `ctx.response` is now populated for all normal orchestrator paths.
- Secondary-offer side effects are now applied inside orchestrator render.

Verification after Commit 1:

- Full suite:
  `PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`
- Result:
  `516 passed, 3 warnings`

### Remaining work

Only Task C remains for this phase:

- slim `patient_routing_stream()` to a shell around `run_pipeline()`
- keep debug envelope + history/summary updates
- keep handoff reset at the shell boundary

Commit 2 should modify `router.py` only after re-reading the now-current
`patient_routing_stream()` implementation.

---

## Progress update (2026-04-16, Commit 2 completed)

Task C is now complete on `refactor/core`.

- `patient_routing_stream()` is now a shell around `run_pipeline()`.
- The large post-orchestrator rendering block was removed.
- The shell now keeps only:
  - error fallback around `run_pipeline()`
  - optional debug envelope from `ctx.decision/plan/evidence`
  - history append (`user` + `assistant`)
  - summary refresh
  - handoff reset at the boundary
  - final `yield ctx.response`

This means the orchestrator is now the sole owner of normal response
construction, while `patient_routing_stream()` keeps the transport/session
wrapper responsibilities only.

Verification after Commit 2:

- Full suite:
  `PYTHONPATH=. venv/bin/pytest tests/ -x -q --ignore=tests/eval`
- Result:
  `516 passed, 3 warnings`

### Net result of Phase 3

- `tool_loop()` owns orchestration after NLU.
- `render()` owns response construction.
- `patient_routing_stream()` is reduced to prechecks + shell behavior.

---

## Current state (what exists right now)

File: `messengers_router/orchestrator.py` (213 lines)

`tool_loop()` currently bridges to the legacy monolith:
```python
async def tool_loop(ctx, services, memory, runtime_options):
    if ctx.short_circuit or ctx.should_clarify:
        return ctx
    if services is None or memory is None:
        ctx.tool_results = {}
        return ctx
    from . import router
    decision, plan, evidence = await router.route_patient_message(
        ctx.text, ctx.state, services, memory, runtime_options=runtime_options,
    )
    ctx.decision = decision
    ctx.plan = plan
    ctx.evidence = evidence
    ctx.state.dialog.label = ctx.decision.label
    ctx.state.dialog.confidence = ctx.decision.confidence
    ctx.state.dialog.merge_entities(ctx.decision.entities)
    return ctx
```

`render()` currently returns `ctx.response = None` for the normal path (plan/evidence set),
letting the ~200-line block in `patient_routing_stream()` do the actual rendering.

---

## What you must do — 3 tasks in order

The complete spec for all three tasks is already written in:
`docs/superpowers/plans/2026-04-15-orchestrator-phase2-handoff.md`

Read that file first. Below is the exact implementation to use.

---

### Task A — Replace tool_loop bridge with real handlers

**File:** `messengers_router/orchestrator.py`

**Read the file before editing** (pre-tool hook requires it).

Replace the entire `tool_loop()` function (lines 112–148) with:

```python
async def tool_loop(
    ctx: OrchestratorContext,
    services: Any | None = None,
    memory: Any | None = None,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Выполняет plan + execute, заменяя legacy-bridge.

    :param ctx: контекст пайплайна
    :param services: сервисный слой
    :param memory: хранилище pending/state
    :param runtime_options: runtime-настройки LLM/NLU
    :return: обновлённый контекст
    """

    if ctx.short_circuit or ctx.should_clarify:
        return ctx
    if services is None or memory is None:
        ctx.tool_results = {}
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

**Why this works:**
- `_handle_catalog_confirm_pending`, `_handle_appointment_action_pending`, `_handle_compound_price_pending` are already defined in `router.py` at lines ~735, ~488, ~593 respectively. They accept keyword-only args `(user_text, state, services, memory, runtime_options)` and return `tuple[RouteDecision, Plan, Evidence] | None`.
- `build_plan` is at `router.py` line ~1260 — synchronous, takes `(decision, state, user_text, memory, runtime_options)`.
- `execute_plan` is at `router.py` line ~1270 — async, takes `(plan, state, services)`.

**After Task A:** Run the full test suite. Write this test in `tests/test_orchestrator_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_tool_loop_calls_build_and_execute_plan(monkeypatch):
    from messengers_router.orchestrator import OrchestratorContext, tool_loop
    from messengers_router.mess_types import (
        Evidence, Plan, RouteDecision, SessionState, DialogState,
    )
    import messengers_router.router as router_mod

    state = SessionState(session_id="t1")
    decision = RouteDecision(label="PRICE", confidence=0.8)
    ctx = OrchestratorContext(text="цена анализа", state=state, decision=decision)

    captured = {}

    def fake_build_plan(dec, st, text, mem, *, runtime_options=None):
        captured["build"] = True
        return Plan(label="PRICE")

    async def fake_execute_plan(plan, st, svc):
        captured["execute"] = True
        return Evidence(items={"price": "100"})

    # patch all 3 handlers to return None (not triggered)
    monkeypatch.setattr(router_mod, "_handle_catalog_confirm_pending", lambda **kw: None)
    monkeypatch.setattr(router_mod, "_handle_appointment_action_pending", lambda **kw: None)
    monkeypatch.setattr(router_mod, "_handle_compound_price_pending", lambda **kw: None)
    monkeypatch.setattr(router_mod, "build_plan", fake_build_plan)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    result = await tool_loop(ctx, services=object(), memory=object())
    assert captured.get("build") is True
    assert captured.get("execute") is True
    assert result.plan is not None
    assert result.evidence is not None
```

---

### Task B — Implement real render()

**File:** `messengers_router/orchestrator.py`

Read the file again (it was modified by Task A).

Replace the entire `render()` function (currently lines 151–180) with:

```python
async def render(
    ctx: OrchestratorContext,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Собирает финальный ResponseEnvelope.

    :param ctx: контекст пайплайна
    :param runtime_options: runtime-настройки LLM/NLU
    :return: обновлённый контекст
    """

    from . import renderer

    # Safety short-circuits
    if ctx.short_circuit and ctx.decision is not None:
        if ctx.decision.label == "URGENT":
            ctx.response = renderer.render_urgent()
        elif ctx.decision.label == "COMPLAINT":
            ctx.response = renderer.render_complaint()
        elif ctx.decision.label == "MEDICAL_ADVICE":
            ctx.response = renderer.render_medical_advice()
        # pending_handler short-circuit: falls through to normal render below
        if ctx.response is not None:
            return ctx

    # Clarify gate
    if ctx.should_clarify:
        ctx.response = ResponseEnvelope(text=ctx.clarify_text)
        return ctx

    # Normal path: collect render_stream output into a single string
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

    # Defensive fallback — should not reach here in normal flow
    ctx.response = ResponseEnvelope(text="")
    return ctx
```

Also update `run_pipeline()` — it already calls `render(ctx, runtime_options=runtime_options)` at line 212. If the old version didn't pass `runtime_options`, fix that. The call should be:
```python
ctx = await render(ctx, runtime_options=runtime_options)
```

**After Task B:** Run the full test suite. Write this test in `tests/test_orchestrator_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_render_collects_stream_into_response_envelope(monkeypatch):
    from messengers_router.orchestrator import OrchestratorContext, render
    from messengers_router.mess_types import (
        Evidence, RouteDecision, SessionState,
    )
    import messengers_router.renderer as renderer_mod

    state = SessionState(session_id="t2")
    decision = RouteDecision(label="PRICE", confidence=0.8)
    evidence = Evidence(items={"price": "100"})
    ctx = OrchestratorContext(
        text="цена анализа", state=state, decision=decision, evidence=evidence,
    )
    ctx.plan = object()  # non-None so render knows plan was built

    async def fake_render_stream(text, dec, ev, *, runtime_options=None):
        yield "привет "
        yield "мир"

    monkeypatch.setattr(renderer_mod, "render_stream", fake_render_stream)

    result = await render(ctx)
    assert result.response is not None
    assert result.response.text == "привет мир"
```

---

### Task C — Simplify patient_routing_stream() in router.py

**ONLY do this after Tasks A and B are green.**

**File:** `messengers_router/router.py`

Read the file before editing. Find the block in `patient_routing_stream()` around line 2055 that starts with:
```python
try:
    ctx = await run_pipeline(...)
    if ctx.response and ctx.response.text:
        if ctx.response.handoff:
```

This block currently falls through to ~200 lines of rendering when `ctx.response` is None. After Task B, `ctx.response` is ALWAYS set, so those 200 lines are dead code. However:

**CAUTION:** Before deleting anything, grep for these calls within the 200-line block and confirm each one is handled in the new path:
- `memory.append_turn(state, role=..., text=...)` — must be preserved
- `update_summary(state, reason="normal")` — must be preserved  
- `_debug_meta(...)` / debug envelope — must be preserved
- `_reset_state_after_handoff(state, memory)` — must be preserved in handoff + error branches
- `INTRO_TEXT` — check if `render_stream` already includes intro via Evidence; if not, keep

Replace the `try/except` block (from `try: ctx = await run_pipeline(...)` to the end of the giant rendering block) with:

```python
try:
    ctx = await run_pipeline(
        user_text,
        state,
        services=services,
        memory=memory,
        runtime_options=runtime_options,
    )
except Exception as e:
    fallback_text = handoff_message("service_error")
    state_update: dict[str, Any] = {}
    if debug:
        state_update = {"debug": {"route_error": str(e)}}
    _reset_state_after_handoff(state, memory)
    yield ResponseEnvelope(
        text=fallback_text,
        attachments=[],
        handoff=True,
        state_update=state_update,
    )
    return

if ctx.response is None:
    # Defensive: should not happen
    yield ResponseEnvelope(text=handoff_message("service_error"), handoff=True)
    return

if ctx.response.handoff:
    _reset_state_after_handoff(state, memory)

# Debug envelope (keep this — it feeds the debug UI)
if debug and ctx.decision and ctx.plan and ctx.evidence:
    pending = memory.get_pending(state)
    yield ResponseEnvelope(
        text="",
        attachments=[],
        handoff=False,
        state_update={"debug": _debug_meta(ctx.decision, ctx.plan, ctx.evidence, state, pending)},
    )

# History + summary (must be preserved)
memory.append_turn(state, role="user", text=user_text)
memory.append_turn(state, role="assistant", text=ctx.response.text)
update_summary(state, reason="normal")

yield ctx.response
return
```

**If you are unsure about any code inside the 200-line block being safe to delete:** leave the entire original block in place and just add a guard at the top:

```python
if ctx.response is not None and ctx.response.text:
    if ctx.response.handoff:
        _reset_state_after_handoff(state, memory)
    memory.append_turn(state, role="user", text=user_text)
    memory.append_turn(state, role="assistant", text=ctx.response.text)
    update_summary(state, reason="normal")
    yield ctx.response
    return
# ... existing 200 lines follow as fallback (dead code but harmless)
```

This is the safe option. Use it if there's any doubt.

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

## What NOT to do

1. Do NOT delete `route_patient_message()` — secondary queue logic still lives there.
2. Do NOT touch `early_guards()`, `nlu_route()`, `clarify_gate()`.
3. Do NOT touch the `run_appointment_precheck()` call at the TOP of `patient_routing_stream()` — it runs before the orchestrator and is correct.
4. Read files before editing — pre-tool hook enforces this.
5. Do NOT skip the test run between tasks.
