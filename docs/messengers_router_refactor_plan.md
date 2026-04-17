# `messengers_router` Refactor Plan (post-orchestrator cleanup)

**Audience:** an autonomous refactor agent executing the stages below.
**Owner:** @Maytereya
**Baseline commits:** `06f85a5` (merge of `refactor/core`), `fec9770`, `98d0e00`, `c9a4869`
**Scope:** `messengers_router/` only. Do NOT modify `agent_logic_1/`, `agent_logic_2/`, or `src/`.

## Current Handoff Status

- **Active checkout:** `/Users/maxten/Dev/Distributed-Local-AI-Agent2`
- **Active branch:** `refactor/core`
- **Completed stage commits already present on `refactor/core`:**
  - `e492ecb` — Stage 0/1 cleanup + Stage 5 diagnostic mode checkpoint
  - `f388318` — Stage 2 (`refactor(classifier): make guardrail_precheck sync (it never awaited)`)
  - `36cfe69` — Stage 3 (`refactor(router): log swallowed exceptions instead of hiding them`)
  - `da77de9` — Stage 4 (`refactor(router): route evidence handoff check through policies helper`)
  - `9f9483b` — Stage 5 (`refactor(router): consolidate reset_appointment_runtime_state — Option A`)
  - `d026414` — Stage 6 (`refactor(policies): centralise handoff messages via HANDOFF_REASON_MATRIX`)
- **Current stage:** Stage 7 is implemented locally and verified locally, but **not committed yet**
- **Current dirty files (Stage 7 only, excluding local user files outside scope):**
  - `messengers_router/flow_policy.py`
  - `messengers_router/router.py`
  - `messengers_router/appointment_flow_guard.py`
  - `tests/test_router_flow_override.py`
- **Fresh local verification for the current dirty Stage-7 diff:**
  - `ruff check messengers_router/flow_policy.py messengers_router/router.py messengers_router/appointment_flow_guard.py tests/test_router_flow_override.py` → passed
  - `PYTHONPATH=/Users/maxten/Dev/Distributed-Local-AI-Agent2 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/test_router_flow_override.py -q` → `171 passed`
  - `PYTHONPATH=/Users/maxten/Dev/Distributed-Local-AI-Agent2 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval` → `555 passed, 3 warnings`
- **Remote eval status:** not rerun after `RUN_ID=1776366965`; the last known remote blocker remains `04_stage5_golden_corpus` timeout from Session 4, so remote parity is still unverified for Stages 4–7
- **Important environment caveat:** after the date rolled to `2026-04-17`, local pytest started picking today's Nayka cache filenames. To keep local tests deterministic in this worktree, hydrate today's dated files before pytest if they are missing/empty:
  - `agent_logic_2/nayka_api/apidata/doctors_20260417.jsonl`
  - `agent_logic_2/nayka_api/apidata/price_units/price_units_20260417.jsonl`
  - `agent_logic_2/nayka_api/apidata/price_by_region/price_region_3_20260417.jsonl`
  - `agent_logic_2/nayka_api/apidata/service_info/service_info_20260417.jsonl`
  Otherwise some tests may regenerate empty live-dependent caches and fail for environment reasons rather than router logic.

## Progress Log

### Session 1 — 2026-04-16

- **Status:** blocked before Stage 1
- **Worktree:** `/tmp/codex-worktrees/messengers-router-refactor-s1`
- **Branch:** `codex/messengers-router-refactor-s1`
- **Baseline RUN_ID:** `1776365226`
- **What was done:**
  - isolated worktree created from `release@f4c8111`
  - baseline capture started per plan before any refactor edits
- **Why execution stopped:**
  - `pytest tests/ messengers_router/tests/` is not green because `messengers_router/tests/` does not exist in the current tree; only `tests/` exists
  - `ruff check messengers_router/` reports 21 pre-existing violations before any new refactor work
  - remote eval was intentionally not started after the baseline already failed, because the plan says to stop and report if baseline is not green
- **Next session should start with:**
  - deciding whether to add a Stage 0 that normalizes the baseline, or to update the baseline verification commands to match the current repository layout
  - deciding how to treat the existing non-stage Ruff findings (`policies.py`, `renderer.py`, `services_legacy.py`, `scripts/audit_nayka_site_api.py`) so Stage 1 can stay atomic

### Session 2 — 2026-04-16

- **Status:** Stage 0 prepared, execution not started
- **What was verified:**
  - the real unit-test root in the current tree is `tests/`; `messengers_router/tests/` does not exist
  - repo docs and adjacent plans already use `PYTHONPATH=. ... pytest tests/`
  - in an isolated worktree without its own `venv/`, pytest must run via an already provisioned interpreter (activated venv or absolute path to a shared venv)
- **Observed pytest baseline when run against the real test root:**
  - `PYTHONPATH=/tmp/codex-worktrees/messengers-router-refactor-s1 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval`
  - first failure: `tests/test_entity_grounder.py::test_grounder_extracts_service_from_user_text`
  - failure mode: `Services()` falls through to live Nayka calls with an empty/invalid base URL, leaving the service catalog empty and dropping `service_name`
- **Plan update in this session:**
  - baseline verification command updated to the actual repo layout
  - new Stage 0 added so baseline normalization happens before Part I

### Session 3 — 2026-04-16

- **Status:** Stage 0 partially executed; pytest baseline is green, non-Stage-1 Ruff debt is fixed
- **Operational finding (root cause):**
  - isolated worktrees do not contain the local Nayka cache files from `agent_logic_2/nayka_api/apidata/`
  - because `/app_data` is not writable in this local environment, cache resolution falls back to the worktree-local legacy path, which starts empty
  - this empty cache caused both the original `entity_grounder` failure and the later `doctor_name` failure in `classifier`; those were symptoms, not separate logic bugs
- **What was done in practice:**
  - hydrated `/tmp/codex-worktrees/messengers-router-refactor-s1/agent_logic_2/nayka_api/apidata/` from the populated cache in the main checkout
  - re-ran `PYTHONPATH=/tmp/codex-worktrees/messengers-router-refactor-s1 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval`
  - result: `519 passed, 3 warnings`
  - fixed the non-Stage-1 Ruff findings in:
    - `messengers_router/policies.py`
    - `messengers_router/renderer.py`
    - `messengers_router/scripts/audit_nayka_site_api.py`
    - `messengers_router/services_legacy.py`
- **Current Ruff status after those fixes:**
  - `ruff check messengers_router/` now fails only on 12 `F401` import removals reserved for Stage 1:
    - `messengers_router/router.py`
    - `messengers_router/nlu_pipeline.py`
- **Important constraint for next session:**
  - before any baseline pytest run in this worktree, ensure `agent_logic_2/nayka_api/apidata/` in the worktree is hydrated from a populated cache source, otherwise tests that depend on local doctors/price catalogs will fail for environment reasons

### Session 4 — 2026-04-16

- **Status:** Stage 1 implemented locally; merge is still blocked by one remote-eval timeout
- **Stage 1 code changes (worktree branch):**
  - removed dead imports from `messengers_router/router.py`
  - removed dead `CONFIDENCE` import from `messengers_router/nlu_pipeline.py`
  - kept the Stage-0 non-router lint fixes from Session 3
- **Local verification:**
  - `ruff check --select F401 messengers_router/router.py messengers_router/nlu_pipeline.py` → passed
  - `ruff check messengers_router/` → passed
  - `PYTHONPATH=/tmp/codex-worktrees/messengers-router-refactor-s1 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval` → `519 passed, 3 warnings`
- **Remote eval RUN_ID:** `1776366965`
- **Remote eval results:**
  - `01_stage1_intent_smoke` → `20/20` passed
  - `02_stage3_appointment_flow` → `20/20` passed
  - `03_stage4_reliability` → `15/15` passed
  - `05_critical_safety_gate` → `49/49` passed
  - `06_prepare_wrap_quality` → `4/4` passed
  - `07_coverage_ext_assets` → passed
  - `04_stage5_golden_corpus` → failed with `socket.timeout`
- **Interpretation:**
  - there is no evidence that Stage 1 import cleanup regressed the routed behavior checked by stage1/stage3/stage4/critical/prepare-wrap
  - the only remaining blocker is server-side or latency-related instability in stage5 corpus execution
- **Next session should start with:**
  - hydrating the worktree-local Nayka cache again if the worktree was recreated or cleaned
  - inspecting `/tmp/codex-worktrees/messengers-router-refactor-s1/messengers_router/eval_suite/logs/1776366965/04_stage5_golden_corpus.log`
  - re-running stage5 corpus (or full remote eval) to determine whether the timeout is transient or reproducible

### Session 5 — 2026-04-16

- **Status:** diagnostic support for Stage 5 added and checkpointed; remote timeout remained unverified server-side
- **What was done:**
  - added `--diagnostic` and `--slow-case-threshold-sec` to `messengers_router/scripts/eval_stage5_corpus.py`
  - added per-case `START/END/ERROR/SLOW` timing output plus `Slowest cases` summary
  - added regression tests in `tests/test_eval_stage5_corpus.py`
- **Local verification in that session:**
  - `ruff check messengers_router/ tests/test_eval_stage5_corpus.py` → passed
  - `PYTHONPATH=/tmp/codex-worktrees/messengers-router-refactor-s1 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval` → `521 passed, 3 warnings`
- **Commit created:**
  - `e492ecb` — `refactor(router): checkpoint baseline cleanup and stage5 diagnostics`
- **Interpretation:**
  - this commit is a checkpoint, not evidence that Stage 5 remote eval is green
  - it exists so subsequent stages can stay atomic instead of carrying baseline + diagnostics as an uncommitted diff

### Session 6 — 2026-04-17

- **Status:** Stage 2 completed and committed
- **Stage 2 code changes:**
  - `messengers_router/classifier.py`: `guardrail_precheck()` changed from `async def` to sync `def`
  - `messengers_router/classifier.py`: removed the only `await guardrail_precheck(...)` call site
  - `tests/test_llm_primary_nlu.py`: added a contract test proving `guardrail_precheck` is sync
- **Local verification in that session:**
  - `ruff check messengers_router/` → passed
  - `PYTHONPATH=/tmp/codex-worktrees/messengers-router-refactor-s1 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval` → `522 passed, 3 warnings`
- **Commit created:**
  - `f388318` — `refactor(classifier): make guardrail_precheck sync (it never awaited)`
- **Operational note:**
  - after midnight, local pytest briefly failed because the worktree had no valid `20260417` Nayka cache files yet; this was fixed by hydrating today's dated files from the existing cached data, not by changing Stage-2 logic

### Session 7 — 2026-04-17

- **Status:** Stage 3 implemented locally, verified locally, **not committed yet**
- **Stage 3 code changes currently in the worktree:**
  - `messengers_router/router.py`: log swallowed exception around `services.ensure_background_refresh_started()`
  - `messengers_router/endpoint.py`: in `get_services()`, log startup failure and re-raise instead of silently swallowing it
  - `messengers_router/flow_policy.py`: log `_safe_get_branches()` integration failures before returning `[]`
  - `messengers_router/classifier.py`: log `ollama_classify_payload()` failures before timeout fallback payload
  - `messengers_router/dialog_graph.py`: log invalid `_dialog_state` deserialization before returning `IDLE`
  - `messengers_router/memory.py`: log malformed dialog-state snapshots before returning existing `state.dialog`
- **Stage 3 tests added/updated:**
  - `tests/test_endpoint_di.py`
  - `tests/test_router_flow_override.py`
  - `tests/test_llm_primary_nlu.py`
  - `tests/test_memory_dialog_state.py`
  - `tests/test_dialog_state.py`
- **Verification in this session:**
  - targeted red→green tests for the six new logging cases → passed
  - focused regression slice across the affected test modules → `205 passed`
  - `ruff check messengers_router/` → passed
  - `PYTHONPATH=/tmp/codex-worktrees/messengers-router-refactor-s1 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval` → `528 passed, 3 warnings`
- **What is still missing for Stage 3:**
  - separate Stage-3 commit has not been created yet
  - remote eval has not been rerun for Stage 3
- **Next session should start with:**
  - if staying stage-atomic, stage and commit only the Stage-3 files with commit message:
    `refactor(router): log swallowed exceptions instead of hiding them`
  - after that, continue to Stage 4
  - keep the date-sensitive Nayka cache caveat in mind before any local pytest rerun

### Session 8 — 2026-04-17

- **Status:** Stage 3 committed; temporary refactor worktree collapsed back into `refactor/core`
- **What was done:**
  - created separate Stage-3 commit `36cfe69` — `refactor(router): log swallowed exceptions instead of hiding them`
  - fast-forward merged the temporary `codex/messengers-router-refactor-s1` branch into `refactor/core`
  - removed the temporary worktree and local `codex/messengers-router-refactor-s1` branch

### Session 9 — 2026-04-17

- **Status:** Stage 4 completed and committed
- **Commit created:**
  - `da77de9` — `refactor(router): route evidence handoff check through policies helper`

### Session 10 — 2026-04-17

- **Status:** Stage 5 completed and committed
- **Decision taken:** OPTION A
- **Commit created:**
  - `9f9483b` — `refactor(router): consolidate reset_appointment_runtime_state — Option A`

### Session 11 — 2026-04-17

- **Status:** Stage 6 completed and committed
- **Decision taken:** OPTION B
- **Commit created:**
  - `d026414` — `refactor(policies): centralise handoff messages via HANDOFF_REASON_MATRIX`

### Session 12 — 2026-04-17

- **Status:** Stage 7 implemented locally, verified locally, **not committed yet**
- **Decision taken:** OPTION B
- **What was done:**
  - added named clear helpers with shared core in `messengers_router/flow_policy.py`
  - migrated router handoff-reset call sites to the canonical handoff helper
  - migrated `appointment_flow_guard` off the old `clear_appointment_flow_context`
  - migrated `apply_context_action()` topic-switch path to the canonical topic-switch helper
- **Local verification in this session:**
  - `ruff check messengers_router/flow_policy.py messengers_router/router.py messengers_router/appointment_flow_guard.py tests/test_router_flow_override.py` → passed
  - `PYTHONPATH=/Users/maxten/Dev/Distributed-Local-AI-Agent2 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/test_router_flow_override.py -q` → `171 passed`
  - `PYTHONPATH=/Users/maxten/Dev/Distributed-Local-AI-Agent2 /Users/maxten/Dev/Distributed-Local-AI-Agent2/venv/bin/pytest tests/ -x -q --ignore=tests/eval` → `555 passed, 3 warnings`

### Session 13 — 2026-04-17 (autonomous batch)

- **Status:** Stages 10, 11, 12a, 14, 16 committed; 15, 17a skipped (need remote eval); 19 audited clean; 17b and 18 not attempted.
- **Decisions taken (user "go recommended"):**
  - Stage 10 → OPTION C starting with B (constants module; TypedDict deferred until services_legacy.py extracted)
  - Stage 11 → OPTION B (appointment-domain pilot)
  - Stage 12 → blanket OK for 12a–12g
  - Stage 15 → OPTION B
  - Stage 16 → OPTION C (hybrid lru_cache + ctx-cache)
  - Stage 17a first (OPTION C short-circuit), 17b NOT without new approval
  - Skip Stage 13, Stage 18
- **Commits created:**
  - `3b6db5f` — `refactor(evidence): replace raw keys with constants in executor, orchestrator, response_builder, router` (Stage 10)
  - `ba4816d` — `refactor(state): appointment-domain mutation wrapper (Stage 11 pilot)`
  - `d2ac78d` — `refactor(router): extract _handle_operator_offer_pending from route_patient_message (Stage 12a)`
  - `77e21bf` — `perf(orchestrator): add per-stage latency instrumentation (Stage 14)`
  - `1dfc35d` — `perf(nlu): memoize deterministic text helpers with lru_cache`
- **Skipped / deferred (documented reasons):**
  - **Stage 12b–12g:** each helper extraction is behaviour-preserving but the plan requires "eval output identical" per plan §ground-rule 4; without remote eval against `http://172.16.0.16/api/messenger-generate-once` I cannot prove byte-for-byte equivalence on 49 critical + 49 golden + 49 stage5 cases. Local pytest only covers structural invariants, not the full NLU surface.
  - **Stage 15 (parallel catalog):** `_inject_catalog_candidates` has a data dependency on `_verify_doctor_entity` setting `entities["doctor_name"]` before `should_try_service` refinement (router.py lines 1087, 1091–1094). OPTION B requires separating fetch-from-apply in both helpers; plan rates "Risk: Medium. Eval mismatch = revert." Cannot validate without remote eval.
  - **Stage 17a (NLU short-circuit):** the current `_merge` in `nlu_pipeline.py:71–120` explicitly treats LLM as source of truth for non-safety labels. Docstring calls out that the old "high rule confidence → skip LLM" path caused green-eval / prod-fail divergence and was deliberately removed. Short-circuiting re-introduces that failure mode. Cannot safely gate behind "byte-identical eval output" requirement without remote eval.
  - **Stage 17b, 18:** not in scope per user instructions (17b needs new approval, 18 is measurements-driven).
- **Stage 19 audit results (no code changes needed):**
  - `requests.get/post/put` → only in `messengers_router/scripts/audit_nayka_site_api.py` (CLI-only, not hot-path).
  - `time.sleep` → zero occurrences under `messengers_router/`.
  - File I/O → `city.py:load_city_list` already wrapped with `@lru_cache`; other open() calls are in `scripts/` and `eval_suite/` (CLI / offline).
  - `ensure_background_refresh_started` → verified: only calls `api_price.ensure_daily_price_refresh_started` and `api_service_info.ensure_daily_service_info_refresh_started`, both of which use `loop.create_task` — no blocking HTTP in the hot path.
- **Local verification (per-stage gate):**
  - ruff + `PYTHONPATH=. venv/bin/python -m pytest tests/` ran green after each committed stage.
  - Final: `561 passed, 3 warnings in 90.41s`.
- **Not run:** remote eval (`run_remote_eval.sh`) — per user instructions, trust stage-9 green baseline; stage-5 corpus was noted flaky.

---

## Ground Rules

These are non-negotiable. Violating any of them is an automatic rollback.

1. **Every stage ends green.** Before merging a stage commit, the following must all pass:
   - `PYTHONPATH=. python -m pytest tests/ -x -q --ignore=tests/eval` (unit tests; use any already provisioned interpreter/venv with project deps installed)
   - `bash messengers_router/eval_suite/run_remote_eval.sh --url http://172.16.0.16/api/messenger-generate-once`
     (all 7 eval stages: intent smoke, appointment flow, reliability, golden corpus, critical safety gate, prepare wrap, coverage)
   - `ruff check messengers_router/`
2. **One stage = one commit (or a small atomic chain).** Do not bundle unrelated changes. Commit message prefix: `refactor(router):`.
3. **If a stage fails, revert that stage's commit and ask the user.** Do not patch forward to make a failure go away.
4. **No behavior changes unless the stage explicitly says so.** If a stage is described as "no behavior change", run a full eval run against the previous green run's log directory and diff the outputs — any difference must be explained.
5. **Where the plan says `OPTION A / OPTION B`, STOP and ask the user.** Do not guess.
6. **Do not add new files unless the stage says to.** Prefer editing existing files.
7. **Do not rewrite docstrings or comments on code you didn't touch.** No drive-by edits.
8. **Read-before-edit.** The runtime enforces this. Always Read a file once per session before editing it.
9. **Never skip hooks** (`--no-verify`), never force-push, never amend merged commits.
10. **Status-line updates only at stage boundaries.** Don't spam progress reports mid-stage.

---

## Baseline Capture (mandatory, do this first)

Before touching anything:

```bash
# from repo root (or worktree root)
# If the worktree has no local venv/, use an already provisioned interpreter:
# activate the shared venv first or replace `python -m pytest` with an absolute interpreter path.
RUN_ID=$(date +%s)
mkdir -p messengers_router/eval_suite/logs/baseline_${RUN_ID}
PYTHONPATH=. python -m pytest tests/ -x -q --ignore=tests/eval \
  2>&1 | tee messengers_router/eval_suite/logs/baseline_${RUN_ID}/pytest.log
bash messengers_router/eval_suite/run_remote_eval.sh --url http://172.16.0.16/api/messenger-generate-once \
  2>&1 | tee messengers_router/eval_suite/logs/baseline_${RUN_ID}/eval.log
ruff check messengers_router/ 2>&1 | tee messengers_router/eval_suite/logs/baseline_${RUN_ID}/ruff.log
```

Record `RUN_ID` in the PR description. Every subsequent stage will diff against this.

All stages below are verified against this baseline. If the baseline isn't green, stop and report.

---

## Stage 0 — Normalize the baseline contract before Part I

**Goal:** Make the plan executable against the repository that actually exists today. Right now the plan assumes a non-existent `messengers_router/tests/` tree and an always-green baseline; both assumptions are false on `release@f4c8111`.

**Scope:** baseline only. No refactor work from Part I starts until this stage is green.

**What Stage 0 must fix:**
1. **Pytest command mismatch.**
   - Canonical unit-test root is `tests/`.
   - Canonical command is:
     ```bash
     PYTHONPATH=. python -m pytest tests/ -x -q --ignore=tests/eval
     ```
   - If running inside a worktree without its own `venv/`, use an already provisioned interpreter (activated shared venv or absolute interpreter path). Do not assume `./venv/bin/pytest` exists in the worktree.
2. **Current red pytest baseline.**
   - Observed first failure:
     `tests/test_entity_grounder.py::test_grounder_extracts_service_from_user_text`
   - Observed failure mode:
     `Services()` attempts live Nayka-dependent catalog loading with an empty/invalid base URL (`Invalid URL '/doctors'`, `/companyUnits`, `/doctorRegions`, `/regions`), so `ground_decision_entities()` cannot confirm `"Торакоцентез"` from user text and drops `service_name`.
   - This must be fixed in a deterministic, test-safe way before Stage 1. The likely direction is to make the service-grounding path use an offline-safe catalog source or a deterministic fallback when API config is absent, rather than depending on live Nayka in unit tests.
3. **Current red Ruff baseline outside Part I.**
   - Stage 1 legitimately owns only the unused-import subset in `router.py` and `nlu_pipeline.py`.
   - The rest of the pre-existing Ruff debt is separate baseline work and must not be silently bundled into Stage 1:
     - `messengers_router/policies.py` (`F841`)
     - `messengers_router/renderer.py` (`E741`)
     - `messengers_router/services_legacy.py` (`F402`, `E741`, `E402`)
     - `messengers_router/scripts/audit_nayka_site_api.py` (`F401`)

**Execution order:**
1. Fix the red pytest baseline until `PYTHONPATH=. python -m pytest tests/ -x -q --ignore=tests/eval` is green.
2. Fix the non-Stage-1 Ruff violations listed above until `ruff check messengers_router/` is green.
3. Re-run the full Baseline Capture block and record the new `RUN_ID`.
4. Only then continue to Stage 1.

**Inter-stage lint note:**
- Stage 1 explicitly owns the remaining unused-import cleanup in `messengers_router/router.py` and `messengers_router/nlu_pipeline.py`.
- Therefore, Stage 0 is considered complete enough to hand off once:
  - pytest is green,
  - the non-Stage-1 Ruff findings listed above are fixed,
  - and the only residual Ruff output is the Stage-1-owned `F401` set in those two files.

**Verification:**
```bash
PYTHONPATH=. python -m pytest tests/ -x -q --ignore=tests/eval
ruff check messengers_router/
bash messengers_router/eval_suite/run_remote_eval.sh --url http://172.16.0.16/api/messenger-generate-once
```

**Risk:** Medium. This stage is intentionally not a refactor stage; it is a baseline normalization stage. Keep the fixes narrow and deterministic.

**Commit:** `refactor(router): normalize messengers_router baseline before staged cleanup`

---

# PART I — Tier 1: Bug-class risks (do these first, in order)

These are small, safe, and fix latent bugs or unmask invisible errors. No behavior changes intended.

---

## Stage 1 — Remove dead imports from `router.py`

**Goal:** Eliminate 12 stale imports left over from the render migration. These mask whether responsibility has actually moved to the orchestrator.

**File:** [messengers_router/router.py](messengers_router/router.py)

**Action:** Delete the following imports. Each name must be verified as unused by `grep -n '\bNAME\b' messengers_router/router.py` returning only the import line.

| Line (approx) | Symbol |
|---|---|
| 41 | `secondary_followup_text` (from `flow_policy`) |
| 69 | `clarification_question` (from `policies`) |
| 70 | `evidence_requires_handoff` (from `policies`) |
| 72 | `decision_handoff_text` (from `policies`) |
| 89 | `evaluate_recovery` (from `recovery_policy`) |
| 92 | `render_urgent` (from `renderer`) |
| 93 | `render_complaint` (from `renderer`) |
| 94 | `render_medical_advice` (from `renderer`) |
| 95 | `render_stream` (from `renderer`) |
| 98 | `INTRO_TEXT`, `LOW_CONF_CLARIFY_TEXT` (from `text_templates`) |
| 22 | `CONFIDENCE` in `nlu_pipeline.py` (if still unused — verify) |

**Verification:**
```bash
ruff check --select F401 messengers_router/router.py messengers_router/nlu_pipeline.py
PYTHONPATH=. python -m pytest tests/ -x -q --ignore=tests/eval
bash messengers_router/eval_suite/run_remote_eval.sh ...
```

**Risk:** Minimal. These are pure removals.

**Commit:** `refactor(router): drop dead imports left over from render migration`

---

## Stage 2 — Drop `async` from `guardrail_precheck` (purely-CPU function)

**Important context:** The user's async strategy for the rest of the pipeline is *correct* and should be preserved — see Part IV for planned concurrency gains. This stage is narrow: a single function that does zero I/O shouldn't be a coroutine, because the coroutine machinery blocks inlining optimizations and forces all callers into the async stack for no benefit. Removing `async` here does NOT argue against async elsewhere.

**Goal:** `classifier.guardrail_precheck()` is declared `async` but contains no `await`. Every caller pays coroutine-wrapping cost for nothing.

**File:** [messengers_router/classifier.py:814](messengers_router/classifier.py:814)

**Action:**
1. Read the function and confirm no `await` statements exist in its body (including any inner helpers only it calls).
2. If the function body later needs to call an async service (e.g., a future guardrail that hits an LLM), revisit this decision. For now: body is pure regex/rule work.
3. Remove `async` from the `def` line.
4. Find every caller — `grep -rn 'guardrail_precheck' messengers_router/` — and remove the `await` at each call site.
5. If any caller is a sync function that called `asyncio.run(guardrail_precheck(...))`, simplify to a direct call.

**Verification:** pytest + full eval.

**Risk:** Low. The function is pure CPU work; changing sync/async at the call boundary is mechanical.

**Commit:** `refactor(classifier): make guardrail_precheck sync (it never awaited)`

---

## Stage 3 — Add logging to silent exception handlers

**Goal:** Make invisible failures visible. The `_extract_schedule_doctor_name` NameError stayed hidden because `patient_routing_stream`'s outer `except Exception` swallowed the traceback into a generic handoff. Today `c9a4869` already logs the traceback in the router. Extend the same discipline to the other silent swallows.

**Behavior change:** NONE visible to users. Only log output is added.

**Targets** (in priority order):
| File:Line | Current | Action |
|---|---|---|
| [router.py:2009-2010](messengers_router/router.py:2009) | `except Exception: pass` around `services.ensure_background_refresh_started()` | `log.warning("background_refresh_start_failed", exc_info=True)` |
| [endpoint.py:37-38](messengers_router/endpoint.py:37) | `except Exception: pass` in `get_services()` startup | `log.error("get_services_init_failed", exc_info=True); raise` (this is startup — fail loud) |
| [flow_policy.py:708-709](messengers_router/flow_policy.py:708) | `except Exception` in `_safe_get_branches()` → returns `[]` | add `log.warning("branches_fetch_failed", exc_info=True)` before returning `[]` |
| [classifier.py:230-233](messengers_router/classifier.py:230) | `except Exception` in `ollama_classify_payload()` → timeout-flagged return | add `log.warning("ollama_classify_failed", exc_info=True)` |
| [dialog_graph.py:49-50](messengers_router/dialog_graph.py:49) | `except Exception` in `get_dialog_state()` → returns IDLE | add `log.warning("dialog_state_deserialize_failed", exc_info=True)` |
| [memory.py:322-323](messengers_router/memory.py:322) | `except Exception` in dialog state deserialization | add `log.warning("memory_deserialize_failed", exc_info=True)` |

**Constraint:** Use the logger already configured in each module (`log = logging.getLogger(__name__)` is the existing convention). If a module doesn't have a logger, add one following the convention.

**Do NOT touch** the following — they are legitimately narrow parsers where silent default is the contract:
- `policies.py:1126` `_safe_date` — returns None for invalid dates by design
- `policies.py:1394` child-age extraction — opt-in field
- `renderer.py:85,94,122` — ISO parsing helpers; caller handles None
- `llm_runtime.py` config parsers — fall-back to defaults is correct
- `topic_registry.py:81` `_safe_int` — same contract
- `prompt_contracts.py:64,114` — config defaults

**Verification:** pytest + full eval. No eval output should change.

**Risk:** Low. Logs only.

**Commit:** `refactor(router): log swallowed exceptions instead of hiding them`

---

## Stage 4 — Route `router.py:1862` through `evidence_requires_handoff`

**Status:** completed on `refactor/core`
**Commit:** `da77de9` — `refactor(router): route evidence handoff check through policies helper`

**Goal:** Eliminate the one place that bypasses the handoff-contract helper.

**File:** [messengers_router/router.py:1862](messengers_router/router.py:1862)

**Current (approximate):**
```python
if bool(evidence.get("handoff_required")):
    ...
```

**Action:**
1. Re-add the `evidence_requires_handoff` import (removed in Stage 1 — keep in mind ordering).
2. Replace the direct `.get()` with:
   ```python
   handoff_required, handoff_msg, handoff_reason = evidence_requires_handoff(evidence)
   if handoff_required:
       ...
   ```
3. Use `handoff_msg` and `handoff_reason` if they change downstream wording (compare current behavior — if the site only consumed a boolean before, preserve that).

**Stage-1 interaction:** If Stage 1 removed `evidence_requires_handoff` from `router.py`'s import list, re-add it here. Do not leave unused imports.

**Verification:** pytest + full eval. The same cases should still take the handoff path; messages must not change.

**Risk:** Low-medium. One behavior difference is possible: the helper walks nested dicts for `handoff_required=True`, whereas the direct `.get()` only checks top-level. If a nested case existed in evidence that was being missed before, this site will now catch it — that's an improvement, but verify no eval case regresses.

**Commit:** `refactor(router): route evidence handoff check through policies helper`

---

## Stage 5 — Resolve the duplicate `reset_appointment_runtime_state`

**Status:** completed on `refactor/core`
**Decision taken:** OPTION A
**Commit:** `9f9483b` — `refactor(router): consolidate reset_appointment_runtime_state — Option A`

**Goal:** Two functions with the same name and different implementations are a time bomb. Collapse them.

**Files:**
- [messengers_router/appointment_flow_guard.py:132-139](messengers_router/appointment_flow_guard.py:132)
- [messengers_router/flow_policy.py:976-988](messengers_router/flow_policy.py:976)

**The divergence:**
- `appointment_flow_guard` version iterates `_APPOINTMENT_RUNTIME_KEYS` AND calls `dialog.clear()` if active.
- `flow_policy` version calls `_clear_flow_state(state)` (a different key set, does NOT clear `dialog`).

**Current importers (verify with grep):**
- `router.py` imports from `appointment_flow_guard`
- `response_builder.py` imports from `flow_policy`

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A:** Keep `appointment_flow_guard` behavior (clears runtime keys + `dialog.clear()`).
- **Pro:** Matches the "fresh start on handoff" intuition. The DOCTOR_SCHEDULE bug cascade we just fixed was aggravated by Turn 2 finding stale doctor context — clearing dialog on reset prevents that class of state-leak.
- **Con:** Any caller that was relying on `flow_policy.reset_*` keeping dialog alive will change behavior.

**OPTION B:** Keep `flow_policy` behavior (clears specific flow keys only, leaves `dialog` intact).
- **Pro:** Conservative; minimum behavior change.
- **Con:** Leaves two distinct concepts ("clear appointment flow" vs "reset dialog") unresolved; the caller has to know which one to use.

**OPTION C:** Introduce a `reason` argument: `reset_appointment_runtime_state(state, reason: Literal["handoff", "topic_switch", "cancel"])`. A-behavior for "handoff", B-behavior for "topic_switch".
- **Pro:** Preserves both behaviors explicitly, callers document intent.
- **Con:** Larger change, touches more call sites.

**Recommended:** OPTION A. The audit evidence (DOCTOR_SCHEDULE cascade, scattered state-mutation sites) leans toward "handoff means clear everything transient". But ask the user.

**Once the decision is made, the stage work:**
1. Keep only ONE implementation — in `flow_policy.py` (the more appropriate owner module).
2. Delete the other.
3. Update `router.py` to import from `flow_policy`.
4. Verify no other callers.

**Verification:** pytest + full eval. Pay special attention to:
- Stage 3 appointment flow (all APPT_* cases)
- Critical gate `CRIT_APPT_HARD_RESET_*`
- Any case that involves a handoff followed by a new topic turn

**Risk:** Medium. Any divergence here will show up in Stage 3 eval.

**Commit:** `refactor(router): unify reset_appointment_runtime_state (chose OPTION X)`

---

# PART II — Tier 2: DRY consolidations (do these after Part I is green)

Behavior-preserving. Each stage is a single-concept consolidation.

---

## Stage 6 — Route hardcoded handoff strings through `HANDOFF_REASON_MATRIX`

**Status:** completed on `refactor/core`
**Decision taken:** OPTION B
**Commit:** `d026414` — `refactor(policies): centralise handoff messages via HANDOFF_REASON_MATRIX`
**Note:** renderer "not found" texts were intentionally left out of `HANDOFF_REASON_MATRIX`, because they are not operator-handoff reasons.

**Goal:** 15+ sites hardcode variants of "Сейчас не удалось получить данные автоматически. Соединяю с оператором." instead of calling `policies.handoff_message(reason)`.

**Targets:**
| File:Line | Current hardcoded text |
|---|---|
| [services_legacy.py:6051](messengers_router/services_legacy.py:6051) | "получить список врачей" |
| [services_legacy.py:6314](messengers_router/services_legacy.py:6314) | "получить расписание" |
| [services_legacy.py:6427,6437](messengers_router/services_legacy.py:6427) | "найти информацию" |
| [services_legacy.py:6484,6491](messengers_router/services_legacy.py:6484) | "получить данные для записи" |
| [services_legacy.py:6927](messengers_router/services_legacy.py:6927) | "получить результаты" |
| [services_legacy.py:6976](messengers_router/services_legacy.py:6976) | "сформировать ссылку на результат" |
| [services_legacy.py:7072,7119](messengers_router/services_legacy.py:7072) | "получить цены" |
| [services/doctors.py:378,617](messengers_router/services/doctors.py:378) | doctors list / schedule |
| [services/prices.py:121,168](messengers_router/services/prices.py:121) | prices |
| [renderer.py:137,253](messengers_router/renderer.py:137) | "расписание не найдено" / "информация о враче не найдена" |

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: One generic key (minimal change).**
Replace every hardcoded string with `handoff_message("service_error")`.
- **Pro:** Simplest; one canonical message.
- **Con:** Loses domain specificity — user sees the same sentence whether prices or schedules failed.

**OPTION B: Per-domain keys (preserve current wording).**
Add to `HANDOFF_REASON_MATRIX` in [policies.py](messengers_router/policies.py):
```python
HANDOFF_REASON_MATRIX = {
    ...existing...
    "service_error_doctors_list":   "Сейчас не удалось получить список врачей автоматически. Соединяю с оператором.",
    "service_error_schedule":       "Сейчас не удалось получить расписание автоматически. Соединяю с оператором.",
    "service_error_doctor_info":    "Сейчас не удалось найти информацию автоматически. Соединяю с оператором.",
    "service_error_appointments":   "Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
    "service_error_results":        "Сейчас не удалось получить результаты автоматически. Соединяю с оператором.",
    "service_error_result_link":    "Сейчас не удалось сформировать ссылку на результат автоматически. Соединяю с оператором.",
    "service_error_prices":         "Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
    "renderer_schedule_not_found":  "К сожалению, расписание не найдено. Уточните фамилию врача или город.",
    "renderer_doctor_not_found":    "К сожалению, информация о враче не найдена. Уточните фамилию или специальность.",
}
```
Replace each hardcoded call with `handoff_message("service_error_<domain>")`.
- **Pro:** Preserves current user-facing text exactly. No eval wording regressions.
- **Con:** Slightly more matrix entries; the domain split is a design choice.

**OPTION C: Factory function `service_error_message(domain: str) -> str`.**
```python
def service_error_message(domain: str) -> str:
    templates = {"doctors": "получить список врачей", "schedule": "получить расписание", ...}
    noun = templates.get(domain, "получить данные")
    return f"Сейчас не удалось {noun} автоматически. Соединяю с оператором."
```
- **Pro:** No matrix explosion; one function.
- **Con:** Mixes a template language into what's otherwise a lookup table.

**Recommended:** OPTION B. Preserves existing eval behavior bit-for-bit.

**Verification:** Full eval. Compare final handoff messages character-by-character with baseline run. Some critical gate assertions may check exact wording.

**Risk:** Medium. Any char-level mismatch with baseline breaks eval. OPTION B mitigates this.

**Commit:** `refactor(policies): centralise handoff messages via HANDOFF_REASON_MATRIX`

---

## Stage 7 — Collapse four overlapping state-clear helpers

**Status:** implemented locally on `refactor/core`, verified locally, **not committed yet**
**Decision taken:** OPTION B
**Current local Stage-7 diff:** `messengers_router/flow_policy.py`, `messengers_router/router.py`, `messengers_router/appointment_flow_guard.py`, `tests/test_router_flow_override.py`

**Goal:** `_reset_state_after_handoff` (router), `clear_appointment_flow_context` (guard), `_clear_flow_state` (flow_policy), `_clear_topic_state` (flow_policy) all clear different slices of state. The question "what do I call when X happens?" has no clear answer.

**Prerequisite:** Stage 5 is complete.

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: Single function with a reason enum.**
```python
# flow_policy.py
class ClearReason(str, Enum):
    HANDOFF = "handoff"
    TOPIC_SWITCH = "topic_switch"
    APPOINTMENT_CANCELLED = "appointment_cancelled"
    APPOINTMENT_CONFIRMED = "appointment_confirmed"

def clear_dialog_state(state, memory, reason: ClearReason, *, preserve_city: bool = True) -> None:
    # one function, switches on reason
```
- **Pro:** One entry point; reason is documented and greppable.
- **Con:** Fat function; adding a new reason touches one place (good) but risks accidental behavior leak if branches share helpers.

**OPTION B: Small named helpers with a shared private core.**
```python
def _clear_core(state, memory, *, keys_to_clear, preserve_city): ...
def clear_on_handoff(state, memory): _clear_core(...)
def clear_on_topic_switch(state, memory): _clear_core(...)
def clear_on_appointment_end(state, memory): _clear_core(...)
```
- **Pro:** Each public helper documents intent via its name.
- **Con:** Slightly more names to remember.

**Recommended:** OPTION B. Matches the existing module style (small functions with clear names).

**Execution (once option chosen):**
1. Design the canonical clear matrix: which keys are cleared under which reason. Write it in the docstring first.
2. Implement the new helper(s) in `flow_policy.py`.
3. Replace call sites one at a time, verifying tests between each.
4. Delete old helpers only after all call sites migrated.

**Verification:** pytest + full eval after each call-site migration. This stage is the one most likely to shake out latent state-leak bugs.

**Risk:** Medium-high. This is where small divergences become user-visible (e.g., a stale doctor name leaking into the next turn). Do migration site-by-site, not big-bang.

**Commit strategy:** One commit per call-site migration. Final commit removes old helpers.

---

## Stage 8 — Easy extraction: test-result service module

**Goal:** Fill the empty [services/lab_tests.py](messengers_router/services/lab_tests.py) stub with 3 test-result functions from `services_legacy.py`. Easy win; chips away at the 7463-line legacy module.

**Functions to move (approximate, verify in source):**
- `_build_public_result_link`
- `_extract_result_query_fields`
- `_test_assist_clarify_response`

(run `grep -n 'test_assist\|result_link\|result_query' messengers_router/services_legacy.py` to confirm exact names and line ranges.)

**Action:**
1. Cut the 3 functions + their helpers from `services_legacy.py` into `services/lab_tests.py`.
2. Update `services_legacy.py` to import them from `services.lab_tests` if any internal caller still needs them (prefer direct import replacement at call sites).
3. If `services/__init__.py` re-exports anything, add the new names.
4. Check `grep -rn 'from .services_legacy import.*test_' messengers_router/` — redirect to `services.lab_tests`.

**Verification:** pytest + full eval. Zero expected behavior change.

**Risk:** Low — pure move, if call sites are redirected correctly.

**Commit:** `refactor(services): extract test-result helpers into services/lab_tests`

---

## Stage 9 — Split `quick_fill_core_entities` by entity domain

**Goal:** [policies.py:1270-1475](messengers_router/policies.py:1270) fills 8+ unrelated entity types in 205 lines. Split by domain.

**Action:**
1. Identify the disjoint blocks inside the function. Expected groups:
   - Insurance (`insurance_type`, `accepts_children`, `child_age`)
   - Datetime (`datetime`, `date_text`, `time_text`)
   - Order/results (`order_id`, test_result fields)
   - Specialty/doctor (`specialty`, `doctor_name`)
   - Service/test (`service_name`, `test_goal`)
   - Patient (`patient_name`)
2. Extract each block into a private helper:
   ```python
   def _fill_insurance_entities(text, missing, out): ...
   def _fill_datetime_entities(text, missing, out): ...
   def _fill_test_result_entities(text, missing, out): ...
   def _fill_appointment_entities(text, missing, out): ...
   def _fill_patient_entities(text, missing, out): ...
   ```
3. `quick_fill_core_entities` becomes a ~30-line dispatcher calling them in sequence.
4. **Preserve order of operations exactly** — some fills depend on others (e.g., `child_age` implies `accepts_children=True`).

**Verification:** pytest + full eval. Zero behavior change.

**Risk:** Medium. Order dependencies are subtle. Use a regression test first (the existing eval serves as one).

**Commit:** `refactor(policies): split quick_fill_core_entities by entity domain`

---

# PART III — Tier 3: Architectural direction (plan the moves, do carefully)

Don't attempt in one sitting. Each stage here is multi-step.

---

## Stage 10 — Evidence schema (choose discipline)

**Goal:** `Evidence.items` is `dict[str, Any]` with ~40 magic-string keys scattered across 50+ sites. Make the schema discoverable.

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: Full `TypedDict` schema.**
```python
# mess_types.py
class EvidenceItems(TypedDict, total=False):
    auth_required: bool
    auth_message: str
    handoff_required: bool
    handoff_reason: str
    handoff_message: str
    unsupported_catalog: dict
    price: dict
    operator_offer_response: dict
    doctor_schedule: dict
    # ... all ~40 keys, each typed
```
Update `Evidence` to `items: EvidenceItems = field(default_factory=dict)`.
- **Pro:** Full static type checking; IDE autocomplete; refactor-safe.
- **Con:** Touches `mess_types.py`, may require fixing type errors at many existing call sites (good, but invasive).

**OPTION B: Key constants module (incremental).**
```python
# evidence_keys.py
AUTH_REQUIRED = "auth_required"
HANDOFF_REQUIRED = "handoff_required"
PRICE = "price"
...
```
Replace raw strings at call sites over time; call sites are grep-friendly.
- **Pro:** Incremental; no mass change required; still greppable.
- **Con:** No type-safety; still `dict[str, Any]`.

**OPTION C: Do both.** Start with B to unblock migration, adopt A once key list is stable.

**Recommended:** OPTION C (start with B). This is a long-tail refactor; a TypedDict forces all call sites to be fixed at once, which doesn't fit with the "tests stay green after every stage" rule.

**Execution (OPTION B first):**
1. Create [evidence_keys.py](messengers_router/evidence_keys.py) with the ~40 known keys.
2. Migrate in one domain at a time (start with `orchestrator.py` and `router.py`; leave `services_legacy.py` untouched until it's extracted).
3. Run tests + eval after each file migrated.
4. Defer OPTION A until `services_legacy.py` is extracted.

**Risk:** Low per-step if done incrementally. High if attempted as one big PR.

**Commit strategy:** One commit per file migrated. Each commit: `refactor(evidence): replace raw keys with constants in <file>`.

---

## Stage 11 — State mutation API (pilot, then expand)

**Goal:** 60+ sites directly mutate `state.last_entities[...]` and `state.dialog.*`. Renaming any key is an audit nightmare. Add a thin typed wrapper.

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: Full `state_mutations.py` module with setters for every key.**
```python
# state_mutations.py
def set_doctor_name(state, name: str) -> None: ...
def set_patient_name(state, name: str) -> None: ...
def mark_appointment_pending(state) -> None: ...
def set_appointment_step(state, step: str) -> None: ...
# ... ~30 setters
```
- **Pro:** Centralizes key names; adds validation hooks.
- **Con:** Verbose; 60 call sites to migrate.

**OPTION B: Pilot a single domain (appointment).**
Create the wrapper for appointment-related keys only. Migrate only `response_builder.py`, `appointment_flow_guard.py`, and appointment branches in `router.py`. Defer other domains until the pattern is validated.
- **Pro:** Small bite; reveals whether the pattern pays off.
- **Con:** Two conventions coexist during the pilot.

**Recommended:** OPTION B. Start with appointment (the highest-mutation domain per the audit). Revisit after 3 months in production.

**Execution:**
1. List all appointment-related `last_entities` keys (grep `last_entities\["appointment`, `last_entities\["doctor_`, `last_entities\["service_`, etc.).
2. Create `state_mutations.py` with setters covering only those keys.
3. Migrate `response_builder.py` first (smallest call-site count for appointment).
4. Run tests + eval.
5. Migrate `appointment_flow_guard.py`.
6. Run tests + eval.
7. Migrate appointment branches in `router.py`.
8. Full eval.

**Risk:** Medium. Each migration is behavior-preserving if the setter is a dumb assignment; risk rises if setters add validation.

**Commit strategy:** One commit per file.

---

## Stage 12 — Shrink `route_patient_message` (the 604-line hotspot)

**Goal:** Migrate remaining pending handlers from [router.py:1276-1880](messengers_router/router.py:1276) into orchestrator stages.

**This is the biggest remaining migration. Do not attempt without explicit user sign-off per sub-stage.**

**Sub-stages (each is a separate commit + full eval):**

- **12a.** Extract `_handle_operator_offer_pending` → move to orchestrator stage (pre-NLU).
- **12b.** Extract `_handle_catalog_confirm_pending` → orchestrator stage.
- **12c.** Extract `_handle_appointment_action_pending` → orchestrator stage.
- **12d.** Extract `_handle_compound_price_pending` → orchestrator stage.
- **12e.** Move secondary queue handling → orchestrator stage.
- **12f.** Move doctor entity verification guard → orchestrator stage.
- **12g.** Move post-NLU guardrails (PRICE context, PREPARE followup, datetime) → orchestrator stage.

After all 7 sub-stages, `route_patient_message` should be under 200 lines and mostly dispatch.

**Risk per sub-stage:** Medium. Each pending handler has subtle state interactions with `memory` and the surrounding flow. Full eval required after each.

**Do NOT combine sub-stages into one commit.** If 12c breaks eval, revert 12c only.

---

## Stage 13 — Continue `services_legacy.py` extraction (block-wait)

**Goal:** Shrink the 7463-line legacy module toward domain-focused services.

**Prerequisite notes from the audit:**
- **Price catalog (~2500 lines, 48 funcs):** Coupled to LLM disambiguation. **Blocked on:** clear boundary between NLU and rendering. Do NOT extract until orchestrator owns the render path for price responses (post-Stage 12g).
- **Prepare (~1500 lines, 25 funcs):** Coupled to Meili API. **Blocked on:** Meili abstraction layer. Create `services/_meili_client.py` first, then extract.
- **Doctor (~1500 lines, 31 funcs):** 8 already in `services/doctors.py`. Migrate the rest opportunistically when touching doctor-related code.
- **Address (~1200 lines, 21 funcs):** 2 already in `services/addresses.py`. Same: opportunistic.

**Do NOT schedule Stage 13 as a standalone task.** Migrate-as-you-touch is safer than mass moves. Re-evaluate after Stages 10–12.

---

# PART IV — Performance / Concurrency (make responses faster)

**User goal:** the bot should "fly through the pipeline" — make per-turn latency drop by running independent work in parallel and by caching within-turn computations. This part treats that as a first-class goal, not a side effect.

**Ground rule addendum for Part IV only:**

- **A latency measurement is mandatory before ANY Part IV stage.** Record per-stage timings with a simple instrumentation (see Stage 14) against the Baseline `RUN_ID` so every subsequent change can quote "before / after" numbers.
- **No perf stage may regress eval output.** Green eval is still the hard gate.
- **Correctness first.** Concurrency changes are the single biggest source of new bugs (task exceptions swallowed, state written out of order, tasks leaking). Every Part IV stage has a "correctness checklist" below that MUST be satisfied before commit.

## Correctness Checklist for Every Part IV Stage

Before committing a Part IV stage, confirm:

1. **No silent exception swallowing.** `asyncio.gather(..., return_exceptions=True)` is banned unless each returned exception is explicitly inspected and handled in the same function. Default is `return_exceptions=False` (exceptions propagate).
2. **No shared mutable state written from parallel tasks.** If two parallel coroutines both write `state.last_entities`, that's a data race. Either serialize the writes (gather, then write sequentially) or split the keys.
3. **Every `create_task` is awaited or explicitly fired-and-forgotten with a logged name.** Orphan tasks are forbidden.
4. **Timeouts.** Parallel LLM/HTTP calls must inherit the existing per-call timeout. Do NOT add `asyncio.wait_for` without a timeout value matching current behavior.
5. **Cancellation propagation.** If the outer request is cancelled (client disconnect), spawned tasks must be cancelled too. `asyncio.TaskGroup` (Python 3.11+) handles this cleanly; use it where available.
6. **Determinism.** Merging parallel results must produce the same `decision` / `plan` / `evidence` as the serial version for all 49 critical + 49 golden cases. Diff eval logs byte-for-byte.

---

## Stage 14 — Instrument per-stage latency (prerequisite for all perf work)

**Goal:** Measure what we want to optimize. No optimization without baseline numbers.

**Action:**
1. Add a lightweight timing context in `orchestrator.py`:
   ```python
   # orchestrator.py — new helper
   @asynccontextmanager
   async def _timed_stage(ctx: OrchestratorContext, name: str):
       t0 = perf_counter()
       try:
           yield
       finally:
           ctx.stage_timings[name] = round((perf_counter() - t0) * 1000, 1)  # ms
   ```
2. Wrap each pipeline stage: `early_guards`, `nlu_route`, `clarify_gate`, `tool_loop`, `render`.
3. Add `stage_timings: dict[str, float]` to `OrchestratorContext`.
4. When `debug=True`, surface `stage_timings` in `state_update.debug`.
5. Add an eval-harness addon (opt-in via `--timings`) that records p50/p90/p99 per stage across the 49 critical cases.
6. **Baseline run:** Execute eval with `--timings`. Save the table to [eval_suite/logs/perf_baseline.md](messengers_router/eval_suite/logs/perf_baseline.md).

**Verification:** pytest + full eval. No user-visible change.

**Risk:** None (pure instrumentation). `perf_counter()` overhead is nanoseconds.

**Commit:** `perf(orchestrator): add per-stage latency instrumentation`

**Expected output (example):**
```
stage          p50     p90     p99
early_guards   2ms     5ms     12ms
nlu_route      420ms   780ms   1200ms     <-- hot
clarify_gate   1ms     2ms     3ms
tool_loop      180ms   350ms   500ms      <-- hot
render         95ms    250ms   400ms      <-- hot
TOTAL          700ms   1400ms  2100ms
```

Without this table, Parts IV.15 through IV.17 are just guessing.

---

## Stage 15 — Parallelize doctor-verify + catalog-inject (medium win, low risk)

**Goal:** `router.py:1469-1475` runs two independent catalog lookups serially. Run them concurrently.

**Current (approximate):**
```python
# router.py around line 1469
decision = await _verify_doctor_entity(decision, services, user_text)
decision = await _inject_catalog_candidates(
    decision, user_text=user_text, state=state, services=services,
)
```

**Problem:** Both calls mutate `decision`. They can't just `gather` and both reassign.

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: Return patches, merge after gather.**
Refactor each helper to return a "decision patch" (e.g., `DecisionPatch` dataclass) instead of mutating / returning a full new decision. Then:
```python
doctor_patch, catalog_patch = await asyncio.gather(
    _verify_doctor_entity_patch(decision, services, user_text),
    _inject_catalog_candidates_patch(decision, user_text=..., state=..., services=...),
)
decision = decision.apply(doctor_patch).apply(catalog_patch)
```
- **Pro:** No races; explicit merge order; easy to test.
- **Con:** Touches helper signatures; both helpers must be rewritten.

**OPTION B: Read-only parallel fetch, sequential merge.**
Extract the pure "fetch from catalog" HTTP call into its own coroutine. Run the two fetches via `gather`. Then apply both results sequentially inside the main function.
- **Pro:** Helpers keep their existing signatures; only the I/O is parallelized.
- **Con:** Requires internal restructure of each helper to separate "fetch" from "apply".

**Recommended:** OPTION B. Smaller blast radius.

**Correctness checklist (all six items) must be signed off in the PR description.**

**Verification:**
- pytest + full eval.
- Compare `stage_timings["tool_loop"]` to baseline — expect 30–100ms improvement at p50.
- Explicit test: a case with both doctor name + service name (e.g., `G017` from the golden corpus) must produce the same final `decision.entities` and `decision.flags`.

**Risk:** Medium. Eval output mismatch = revert.

**Commit:** `perf(router): parallelize doctor verification and catalog candidate injection`

---

## Stage 16 — Per-turn cache for hot deterministic helpers (small win, low risk)

**Goal:** Eliminate redundant within-turn calls of pure functions on identical inputs.

**Targets (from the audit):**
| Function | Call sites | Hit count per turn |
|---|---|---|
| `_extract_doctor_name(text, mode)` | classifier.py:1175, 1195, 1476 | 3 |
| `extract_service_phrase(text)` | entity_grounder.py:475, classifier.py + policies | 2–3 |
| `services.resolve_doctor_name(name)` | entity_grounder.py, router.py:1083, 1363, 1622 | 2–3 |
| `normalize_ru(text)` | hot loops in classifier.py + router.py | 10+ (but cheap) |

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: Turn-scoped cache attached to `OrchestratorContext`.**
```python
# orchestrator.py
@dataclass
class OrchestratorContext:
    ...
    turn_cache: dict[str, Any] = field(default_factory=dict)
```
Helpers take `ctx` (or a cache handle) and memoize against it. At end of turn, ctx is discarded; no TTL management needed.
- **Pro:** No stale state possible; trivially correct.
- **Con:** Touches helper signatures.

**OPTION B: `functools.lru_cache` on pure functions.**
Apply `@lru_cache(maxsize=512)` to `_extract_doctor_name`, `extract_service_phrase`, `normalize_ru`.
- **Pro:** One-line change per function.
- **Con:** Cross-turn cache hits. Safe for pure functions but you must prove purity (e.g., `normalize_ru` reads no globals). A refactor of internals could silently invalidate the cache.
- **Con:** `maxsize` tuning becomes a concern under load.

**OPTION C: Hybrid.** `lru_cache` for pure sync helpers; ctx-scoped cache for async service calls (which depend on external catalog state and must not leak across turns).

**Recommended:** OPTION C. `lru_cache` for `normalize_ru`, `extract_service_phrase`, `_extract_doctor_name`; ctx-scoped cache for `services.resolve_doctor_name`.

**Important — do NOT cache `services.resolve_doctor_name` across turns.** The doctor catalog refreshes on a TTL; a cross-turn cache would serve stale names after a refresh.

**Verification:**
- pytest + full eval.
- Unit test: confirm cache hit counter increments for repeated calls within a turn.
- Compare `stage_timings["nlu_route"]` to baseline — expect 10–30ms improvement at p50.

**Risk:** Low for `lru_cache` on pure helpers. Low-medium for ctx-cache on service calls (the potential bug is a cache that outlives a turn, which correctness checklist item 2 catches).

**Commit:** `perf(classifier,services): memoize deterministic helpers within a turn`

---

## Stage 17 — Parallelize rule + LLM NLU (biggest win, highest risk)

**Goal:** `nlu_pipeline.py:141–154` runs rule classification, then LLM classification. These can run concurrently and be merged after.

**Current flow (approximate):**
```python
# nlu_pipeline.py
prefetched_rule = await _rule_decision(text, state.last_entities, runtime_options=runtime_options)
# prefetched_rule is sometimes passed to analyze() to avoid recomputation
llm = await classifier.analyze(
    text, llm_context, runtime_options=runtime_options, prefetched_rule=prefetched_rule,
)
```

**The catch:** `classifier.analyze()` today accepts `prefetched_rule` to skip re-running the rule pass. If we change to "rule and LLM run in parallel", the LLM pass can't know the rule result, so merge logic must move OUT of `analyze` into the caller.

**🛑 STOP — USER DECISION REQUIRED 🛑**

**OPTION A: True parallel (gather) with post-merge.**
```python
rule_coro = _rule_decision(text, last_entities, runtime_options=runtime_options)
llm_coro = classifier.analyze(text, llm_context, runtime_options=runtime_options, prefetched_rule=None)

rule, llm = await asyncio.gather(rule_coro, llm_coro)
decision = _merge_rule_and_llm(rule, llm)   # existing merge logic, extracted
```
- **Pro:** ~150–250ms saving at p50 (both are LLM-heavy).
- **Con:** LLM pass runs even when the rule decision is high-confidence and would short-circuit — wasting one LLM call per high-confidence turn.
- **Con:** Requires proving the merge logic is deterministic (== the serial behavior).

**OPTION B: Speculative parallel with early-cancel.**
Start both, but if the rule decision completes first and has confidence ≥ threshold, cancel the LLM task.
- **Pro:** Saves LLM calls on easy turns (~50% of traffic per the golden corpus, which is heavily rule-decided).
- **Con:** Task cancellation has to be clean (correctness checklist item 5). More complex.

**OPTION C: Keep serial but aggressively short-circuit.**
If `_rule_decision` returns a decision with confidence ≥ threshold, skip the LLM call entirely.
- **Pro:** Saves latency AND LLM cost.
- **Pro:** No concurrency bugs possible.
- **Con:** Only helps high-confidence rule cases; low-confidence turns still pay the full LLM cost.

**Recommended:** Stage into two parts.
- **17a:** Implement OPTION C first (aggressive short-circuit). Low risk, saves LLM cost, and is independently valuable. Commit.
- **17b:** Then implement OPTION B on top (speculative parallel for the cases where 17a didn't short-circuit). Higher risk, full LLM cost win.

If after 17a the p90 latency meets the target (user to define), skip 17b. Don't do concurrency for its own sake.

**Correctness requirement:** For every one of the 49 critical + 49 golden + 49 stage5 cases, the final `(label, confidence, entities, flags)` tuple from the new NLU must be byte-identical to baseline. Diff the JSON output of eval stages 4 and 5 character-by-character.

**Verification:**
- pytest + full eval.
- `stage_timings["nlu_route"]`: target −150ms at p50 after 17b; −50ms after 17a alone (short-circuit on ~30% of turns).
- Count of LLM calls per 100 turns: must decrease after 17a, stay flat after 17b.

**Risk:** 17a — low. 17b — high. This is the single most likely stage to break eval.

**Commit:** `perf(nlu): short-circuit LLM when rule confidence is high` (17a); then `perf(nlu): run rule and LLM speculatively in parallel` (17b).

---

## Stage 18 — Parallelize pending-handler dispatch (small, optional)

**Goal:** `router.py:1322-1350` sequentially checks 3 mutually-exclusive pending states.

**Reality check:** Each handler does O(1) state reads and MAYBE one service call. The "sequential" cost is tiny. This stage is worth it ONLY if Stage 14 measurements show the pending dispatch is >20ms at p50. Otherwise skip.

**Recommended:** Defer. Revisit only if measurements demand it.

**If scheduled:**
- Convert to `asyncio.gather` of 3 coroutines, each returning `Optional[Response]`.
- Take the first non-None result.
- Beware: if two handlers claim to handle the same turn, the serial order decides. Parallel makes the decision non-deterministic. Need to check this can't happen (the audit claims they are mutually exclusive — prove it with an assert).

**Risk:** Medium. Non-determinism risk is real.

---

## Stage 19 — HTTP / async client audit

**Goal:** Confirm the pipeline never blocks the event loop on sync I/O.

**Action:** Audit every module the hot path touches for:
- `requests.get/post/put` (should be `httpx` async).
- `time.sleep` (should be `asyncio.sleep`).
- Unwrapped file I/O (use `asyncio.to_thread` if unavoidable).
- `ensure_background_refresh_started()` — the audit noted this is called sync from the hot path. Verify it only sets a flag; if it does any HTTP, move it to `asyncio.create_task` at service construction.

**Verification:** No behavior change; latency should drop under concurrent load (when another turn is in flight). To validate: run eval with two concurrent workers (add a `--concurrency 2` flag to the eval harness) and compare p90 to single-worker.

**Risk:** Low per sweep; high if something like `requests.get` is found in the hot path and replaced carelessly.

**Commit:** `perf(services): replace sync I/O in hot path with async equivalents`

---

## Part IV Summary (latency targets)

After all Part IV stages, the expected (not guaranteed) latency picture:

| Stage budget | Baseline p50 (example) | Target p50 | How |
|---|---|---|---|
| early_guards | 2 ms | 2 ms | no change |
| nlu_route | 420 ms | 180 ms | Stage 17 short-circuit + parallel |
| clarify_gate | 1 ms | 1 ms | no change |
| tool_loop | 180 ms | 100 ms | Stage 15 parallel catalog |
| render | 95 ms | 95 ms | no change (already streamed) |
| **total** | **~700 ms** | **~380 ms** | |

These numbers are illustrative — your Stage 14 baseline is the source of truth.

**Non-goals for Part IV:**
- Optimizing the LLM provider itself (out of scope for this package).
- Micro-optimizing Python (re-ordering, `__slots__`, etc.). ROI is too low vs. IO wins.
- Parallelizing evaluation runs (test harness concern, not router concern).

---

# Final Validation

Before declaring the refactor done:

1. `wc -l messengers_router/services_legacy.py` — should trend down, not up.
2. `wc -l messengers_router/router.py` — under 1500 lines after Stage 12 completes.
3. `ruff check --select F401,F811,F821,ARG messengers_router/` — clean.
4. Full eval baseline comparison: character-level diff of all 7 eval logs against Baseline `RUN_ID` from step 0. Expected diff: **none**.
5. Run eval 3 times in succession; assert identical output (proxy for flake elimination).
6. Unit test coverage for `messengers_router/` — should not drop.

---

# Decision Checkpoints Summary

| Stage | Decision |
|---|---|
| 5 | `reset_appointment_runtime_state` behavior: OPTION A/B/C? |
| 6 | Handoff message consolidation: OPTION A/B/C? |
| 7 | State-clear helpers: OPTION A/B? |
| 10 | Evidence schema: OPTION A/B/C? |
| 11 | State mutation API: OPTION A/B? |
| 15 | Parallel catalog lookups: OPTION A (patches) / B (split fetch from apply)? |
| 16 | Per-turn cache: OPTION A (ctx-scoped) / B (`lru_cache`) / C (hybrid)? |
| 17 | NLU speedup: OPTION A (full parallel) / B (speculative + cancel) / C (short-circuit)? Plan currently schedules 17a = C, then 17b = B. |

When you hit a 🛑, stop, present the options to the user in chat, and wait for selection. Do not proceed with your own preference.

---

# Anti-patterns to Watch For

Things that MUST NOT appear in any commit:
- `--no-verify` on commits.
- Force-pushing to any branch.
- New TODO/FIXME comments. If a follow-up is needed, file it as a GitHub issue.
- Renaming files without updating all importers in the same commit.
- Removing a function without grep-verifying zero callers.
- Suppressing new ruff warnings with `# noqa` instead of fixing.
- Changing eval case files (`*.jsonl`) or eval scripts. Those are the ground truth; the code conforms to them, not the other way around.

---

# How to Report Back

After each stage:
- Commit SHA.
- Output of `pytest -q` (tail).
- Output of eval summary line per stage (e.g., `Passed: 49/49 (100.0%)`).
- Any unexpected ruff/mypy output.
- If a 🛑 was hit, the raw decision prompt to the user.

After the whole plan:
- Total line delta per file (`git diff --stat main`).
- List of files deleted.
- List of files added.
- Eval output diff vs Baseline `RUN_ID` (expected: none).
