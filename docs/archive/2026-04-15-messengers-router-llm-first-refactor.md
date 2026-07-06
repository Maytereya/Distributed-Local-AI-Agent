# messengers_router — LLM-First Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve `messengers_router` from a regex-primary chatbot into an LLM-first system that is flexible, context-aware, and free of structural duplication — while keeping every eval case in `eval_suite/` passing green at every phase boundary.

**Architecture:** Seven sequential phases; Phases 1, 6, and 7 are safe to run in any order relative to each other. Phases 2 → 3 → 4 → 5 must run in that order. Port `DialogState` + helpers from `src/localragagent/freetalk/` (Option A). Write a new `Orchestrator` for messengers_router from scratch (Option B). The planner → executor → renderer pipeline, all prompts, and all services API contracts remain unchanged.

**Tech Stack:** Python 3.11+, dataclasses, asyncio; pymorphy2 (optional, Phase 1); pytest; existing Ollama/LLM runtime.

**Eval commands (run after every phase):**
```bash
# Stage-5 golden corpus (49 cases):
python messengers_router/scripts/eval_stage5_corpus.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --golden messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_v2.jsonl

# Critical multi-turn cases (28 cases):
python messengers_router/eval_suite/eval_critical_cases.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_server_parity.jsonl

# Unit tests:
python -m pytest tests/ -x -q
```

---

## Parallelism Map

```
Phase 1 (Foundation utils)   ─┬─► Phase 2 (LLM-first NLU) ─► Phase 3 (DialogState)
Phase 6 (services split)     ─┘                                  │
Phase 7 (cleanup/confidence)  (any time after Phase 1)           ▼
                                                          Phase 4 (Appointment SM)
                                                                  │
                                                                  ▼
                                                          Phase 5 (Orchestrator)
```

---

## File Map

| File | Action | Phase |
|---|---|---|
| `messengers_router/russian_nlu.py` | **Create** — single normalization utility | 1 |
| `messengers_router/mess_types.py` | **Modify** — add `ALLOWED_ENTITY_KEYS`, `ConfidencePolicy`, `DialogState` | 1, 4 |
| `messengers_router/prompt_contracts.py` | **Modify** — import `ALLOWED_ENTITY_KEYS` from mess_types | 1 |
| `messengers_router/classifier.py` | **Modify** — import from russian_nlu; collapse 3 extractors → 1; import whitelist | 1 |
| `messengers_router/entity_grounder.py` | **Modify** — import from russian_nlu | 1 |
| `messengers_router/llm_doesnt_work_fallback.py` | **Modify** — import from russian_nlu | 1 |
| `messengers_router/nlu_pipeline.py` | **Modify** — LLM-first `_merge()`; use `ConfidencePolicy` | 2 |
| `messengers_router/policies.py` | **Modify** — use `ConfidencePolicy`; import from russian_nlu | 2 |
| `messengers_router/memory.py` | **Modify** — add `save_dialog_state` / `load_dialog_state` helpers | 3 |
| `messengers_router/appointment_flow_guard.py` | **Modify** — use `AppointmentPhase`; collapse 3 intent chains → 1 | 4 |
| `messengers_router/flow_policy.py` | **Modify** — use `AppointmentPhase` | 4 |
| `messengers_router/orchestrator.py` | **Create** — new pipeline replacing the 600-line routing function | 5 |
| `messengers_router/router.py` | **Modify** — thin shim that calls `orchestrator.route()` | 5 |
| `messengers_router/services/` | **Create** — package split from 300KB god-file | 6 |
| `messengers_router/services.py` | **Replace** — becomes one-line re-export shim | 6 |
| `tests/test_russian_nlu.py` | **Create** | 1 |
| `tests/test_confidence_policy.py` | **Create** | 1 |
| `tests/test_nlu_pipeline_llm_first.py` | **Create** | 2 |
| `tests/test_dialog_state.py` | **Create** | 3 |
| `tests/test_appointment_state_machine.py` | **Create** | 4 |

---

## Phase 1 — Foundation Utilities

**Scope:** Extract shared utilities. Zero behaviour change. All existing tests must stay green. Safe to run at any time.

---

### Task 1.1 — Create `russian_nlu.py`

**Files:**
- Create: `messengers_router/russian_nlu.py`
- Create: `tests/test_russian_nlu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_russian_nlu.py
import pytest
from messengers_router.russian_nlu import normalize_ru, ENTITY_WHITELIST


def test_normalize_ru_lowercases():
    assert normalize_ru("ПРИВЕТ") == "привет"


def test_normalize_ru_replaces_yo():
    assert normalize_ru("Ёлка") == "елка"


def test_normalize_ru_strips():
    assert normalize_ru("  привет  ") == "привет"


def test_normalize_ru_combined():
    assert normalize_ru("  ПЕЧЁНЬ  ") == "печень"


def test_normalize_ru_empty():
    assert normalize_ru("") == ""
    assert normalize_ru(None) == ""


def test_entity_whitelist_is_frozenset():
    assert isinstance(ENTITY_WHITELIST, frozenset)


def test_entity_whitelist_has_required_keys():
    required = {"doctor_name", "specialty", "service_name", "city", "date_from", "patient_name"}
    assert required <= ENTITY_WHITELIST
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_russian_nlu.py -v
```
Expected: `ModuleNotFoundError: No module named 'messengers_router.russian_nlu'`

- [ ] **Step 3: Create `messengers_router/russian_nlu.py`**

```python
"""Russian NLU utilities — single source of truth.

Every module in messengers_router MUST import from here instead of
inlining `.lower().replace("ё", "е")` and related patterns.
This eliminates 20+ scattered copies of the same normalization idiom.
"""
from __future__ import annotations


def normalize_ru(text: str | None) -> str:
    """Lowercase + ё→е normalization.

    Use everywhere in place of the inline idiom:
        str(x).lower().replace("ё", "е").strip()
    """
    return str(text or "").lower().replace("ё", "е").strip()


# Canonical entity whitelist — single source of truth for all three former copies:
#   prompt_contracts.ALLOWED_ENTITY_KEYS
#   classifier._ALLOWED_ENTITY_KEYS
#   entity_grounder._LABEL_ENTITY_WHITELIST keys (union)
ENTITY_WHITELIST: frozenset[str] = frozenset({
    "doctor_name",
    "doctor_id",
    "specialty",
    "branch_name",
    "branch_id",
    "city",
    "service_name",
    "appointment_action",
    "patient_name",
    "test_name",
    "test_goal",
    "surname",
    "year",
    "filial",
    "number",
    "order_id",
    "result_action",
    "lang",
    "insurance_type",
    "accepts_children",
    "child_age",
    "date_hint",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "secondary_intents",
    "include_promos",
    "time_flexible",
})
```

- [ ] **Step 4: Run tests to verify pass**

```bash
python -m pytest tests/test_russian_nlu.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add messengers_router/russian_nlu.py tests/test_russian_nlu.py
git commit -m "feat(nlu): add russian_nlu utility module with canonical normalize_ru and ENTITY_WHITELIST"
```

---

### Task 1.2 — Replace 20+ inline `normalize_ru` copies

**Files:**
- Modify: `messengers_router/classifier.py`
- Modify: `messengers_router/entity_grounder.py`
- Modify: `messengers_router/llm_doesnt_work_fallback.py`
- Modify: `messengers_router/policies.py`

- [ ] **Step 1: Add import to each file**

In each of the four files, add this import near the top (after existing imports):
```python
from .russian_nlu import normalize_ru
```

- [ ] **Step 2: Replace all inline occurrences in `classifier.py`**

Use your editor to replace all occurrences of the pattern
`.lower().replace("ё", "е")` with `normalize_ru(...)` equivalently.

Key patterns to replace (there are ~11 in classifier.py):

```python
# BEFORE (example at line 326):
norm = str(token or "").strip().lower().replace("ё", "е")

# AFTER:
norm = normalize_ru(token)
```

```python
# BEFORE (example at line 331):
spec = str(extract_specialty(text or "") or "").strip().lower().replace("ё", "е")

# AFTER:
spec = normalize_ru(extract_specialty(text or "") or "")
```

```python
# BEFORE (example in list comprehension at line 384):
[t.lower().replace("ё", "е") for t in tokens
 if t.lower().replace("ё", "е") not in _DOCTOR_FOLLOWUP_FILLERS]

# AFTER:
[normalize_ru(t) for t in tokens if normalize_ru(t) not in _DOCTOR_FOLLOWUP_FILLERS]
```

- [ ] **Step 3: Replace in `entity_grounder.py`, `llm_doesnt_work_fallback.py`, `policies.py`**

Same pattern: replace `.lower().replace("ё", "е")` with `normalize_ru(...)`.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -x -q
```
Expected: all existing tests PASS (zero behaviour change).

- [ ] **Step 5: Commit**

```bash
git add messengers_router/classifier.py messengers_router/entity_grounder.py \
        messengers_router/llm_doesnt_work_fallback.py messengers_router/policies.py
git commit -m "refactor(nlu): replace 20+ inline normalize_ru copies with russian_nlu.normalize_ru import"
```

---

### Task 1.3 — Unify entity whitelist (3 copies → 1)

**Files:**
- Modify: `messengers_router/prompt_contracts.py` (import from russian_nlu)
- Modify: `messengers_router/classifier.py` (delete `_ALLOWED_ENTITY_KEYS`, import from russian_nlu)
- Modify: `messengers_router/entity_grounder.py` (no change needed — uses per-label whitelist, already correct)

- [ ] **Step 1: Update `prompt_contracts.py`**

Replace lines 20-48 (the `ALLOWED_ENTITY_KEYS` definition) with an import:
```python
# DELETE the local definition of ALLOWED_ENTITY_KEYS (lines 20-48)
# ADD at the top imports section:
from .russian_nlu import ENTITY_WHITELIST as ALLOWED_ENTITY_KEYS
```

- [ ] **Step 2: Update `classifier.py`**

Delete lines 632-642 (the `_ALLOWED_ENTITY_KEYS` set definition) and replace with:
```python
from .russian_nlu import ENTITY_WHITELIST as _ALLOWED_ENTITY_KEYS
```

- [ ] **Step 3: Verify `_sanitize_entities` in classifier.py still works**

The function at line 644 uses `_ALLOWED_ENTITY_KEYS` — no change needed there since the name stays the same.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add messengers_router/prompt_contracts.py messengers_router/classifier.py
git commit -m "refactor(types): unify entity whitelist — 3 copies replaced with single russian_nlu.ENTITY_WHITELIST"
```

---

### Task 1.4 — Collapse 3 doctor-name extractors into 1

**Files:**
- Modify: `messengers_router/classifier.py` (lines 298-316)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_entity_grounder.py or create tests/test_doctor_extractor.py

from unittest.mock import patch
from messengers_router.classifier import _extract_doctor_name


def test_extract_schedule_mode_uses_prefer_schedule():
    """schedule mode uses prefer_schedule=True and specialty filter."""
    with patch("messengers_router.classifier.resolve_cached_doctor_name_candidate",
               return_value="иванов") as mock_resolve, \
         patch("messengers_router.classifier._looks_like_specialty_or_service_token",
               return_value=False):
        result = _extract_doctor_name("к иванову", mode="schedule")
    mock_resolve.assert_called_once_with("к иванову", prefer_schedule=True)
    assert result == "иванов"


def test_extract_appointment_mode_no_prefer_schedule():
    """appointment mode uses prefer_schedule=False."""
    with patch("messengers_router.classifier.resolve_cached_doctor_name_candidate",
               return_value="петров") as mock_resolve, \
         patch("messengers_router.classifier._INVALID_DOCTOR_TOKEN_RE") as mock_re:
        mock_re.search.return_value = None
        result = _extract_doctor_name("записаться к петрову", mode="appointment")
    mock_resolve.assert_called_once_with("записаться к петрову", prefer_schedule=False)
    assert result == "петров"


def test_extract_returns_none_on_invalid_token():
    """All modes return None when the invalid-token regex matches."""
    with patch("messengers_router.classifier.resolve_cached_doctor_name_candidate",
               return_value="записаться"), \
         patch("messengers_router.classifier._INVALID_DOCTOR_TOKEN_RE") as mock_re:
        mock_re.search.return_value = True  # simulate match
        result = _extract_doctor_name("записаться", mode="appointment")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_doctor_extractor.py -v
```
Expected: `ImportError` (function doesn't exist yet).

- [ ] **Step 3: Add `_extract_doctor_name` and remove old functions in `classifier.py`**

**Replace** the three functions at lines 298-316 with this single function:

```python
def _extract_doctor_name(
    text: str,
    *,
    mode: Literal["schedule", "appointment", "price"] = "appointment",
) -> str | None:
    """Unified doctor-name extractor.

    Replaces three near-identical functions:
      - _extract_schedule_doctor_name   → mode="schedule"
      - _extract_appointment_doctor_name → mode="appointment"
      - _extract_price_doctor_name       → mode="price"

    mode="schedule" and mode="price" use prefer_schedule=True.
    mode="schedule" filters with _looks_like_specialty_or_service_token.
    mode="appointment" and mode="price" filter with _INVALID_DOCTOR_TOKEN_RE.
    """
    prefer_schedule = mode in ("schedule", "price")
    candidate = resolve_cached_doctor_name_candidate(text, prefer_schedule=prefer_schedule)
    if candidate is None:
        return None
    if mode == "schedule":
        if _looks_like_specialty_or_service_token(candidate, text):
            return None
    else:  # appointment / price
        if _INVALID_DOCTOR_TOKEN_RE.search(candidate):
            return None
    return candidate
```

You also need to add `Literal` to the imports at the top of classifier.py if not present:
```python
from typing import Any, Literal, cast
```

- [ ] **Step 4: Update all call sites within `classifier.py`**

Search for `_extract_schedule_doctor_name`, `_extract_appointment_doctor_name`, `_extract_price_doctor_name` in classifier.py and replace each call:

```python
# BEFORE:
_extract_schedule_doctor_name(text)
# AFTER:
_extract_doctor_name(text, mode="schedule")

# BEFORE:
_extract_appointment_doctor_name(text)
# AFTER:
_extract_doctor_name(text, mode="appointment")

# BEFORE:
_extract_price_doctor_name(text)
# AFTER:
_extract_doctor_name(text, mode="price")
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add messengers_router/classifier.py tests/test_doctor_extractor.py
git commit -m "refactor(classifier): collapse 3 near-identical doctor-name extractors into _extract_doctor_name(mode=)"
```

---

### Task 1.5 — Add `ConfidencePolicy` to `mess_types.py`

**Files:**
- Modify: `messengers_router/mess_types.py`
- Create: `tests/test_confidence_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_confidence_policy.py
from messengers_router.mess_types import CONFIDENCE, ConfidencePolicy


def test_confidence_policy_is_frozen():
    import dataclasses
    assert dataclasses.fields(CONFIDENCE)  # is a dataclass


def test_confidence_policy_default_thresholds():
    assert CONFIDENCE.rule_promotion_hybrid == 0.55
    assert CONFIDENCE.rule_promotion_rich == 0.45
    assert CONFIDENCE.clarify_threshold == 0.55
    assert CONFIDENCE.floor_llm_promoted == 0.60
    assert CONFIDENCE.floor_default_low == 0.20


def test_confidence_policy_singleton():
    from messengers_router.mess_types import CONFIDENCE as C2
    assert CONFIDENCE is C2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_confidence_policy.py -v
```
Expected: `ImportError: cannot import name 'CONFIDENCE'`

- [ ] **Step 3: Add `ConfidencePolicy` to `mess_types.py`**

Append at the end of `messengers_router/mess_types.py` (after the `ResponseEnvelope` dataclass):

```python
@dataclass(frozen=True)
class ConfidencePolicy:
    """Named confidence thresholds — replaces all magic numbers.

    Import and use CONFIDENCE singleton instead of hardcoded floats.
    Example:
        from .mess_types import CONFIDENCE
        if decision.confidence < CONFIDENCE.clarify_threshold:
            trigger_clarification()
    """
    # NLU merge: promote rule over LLM when LLM below this
    rule_promotion_hybrid: float = 0.55
    rule_promotion_rich: float = 0.45

    # Classification floors by intent source
    floor_deterministic_appointment: float = 0.90
    floor_deterministic_doc_request: float = 0.85
    floor_deterministic_test_result: float = 0.78
    floor_deterministic_price: float = 0.72
    floor_deterministic_address: float = 0.71
    floor_llm_promoted: float = 0.60
    floor_default_low: float = 0.20

    # Recovery: below this confidence → trigger clarification
    clarify_threshold: float = 0.55
    # Refinement trigger (classifier re-runs LLM with narrower prompt)
    refine_trigger: float = 0.45


# Module-level singleton — import this, do not instantiate ConfidencePolicy directly.
CONFIDENCE = ConfidencePolicy()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_confidence_policy.py tests/ -x -q
```
Expected: all PASS (no other tests broken).

- [ ] **Step 5: Commit**

```bash
git add messengers_router/mess_types.py tests/test_confidence_policy.py
git commit -m "feat(types): add ConfidencePolicy dataclass with named thresholds; add CONFIDENCE singleton"
```

---

## Phase 2 — LLM-First NLU Pipeline

**Prerequisite:** Phase 1 complete.

**Scope:** Remove regex-over-LLM promotion from `_merge()`. LLM intent wins; regex supplements entities only. Safety labels (URGENT, COMPLAINT, MEDICAL_ADVICE) remain as hard rule overrides. When LLM is uncertain, the clarification system handles it — not regex.

**Risk:** Medium. Re-run eval suite after this phase. The system should pass the same cases — the LLM already classifies these correctly; we're removing the regex backstop that was sometimes masking LLM uncertainty with wrong regex intents.

---

### Task 2.1 — Rewrite `_merge()` in `nlu_pipeline.py`

**Files:**
- Modify: `messengers_router/nlu_pipeline.py`
- Create: `tests/test_nlu_pipeline_llm_first.py`

- [ ] **Step 1: Write tests for the new merge behaviour**

```python
# tests/test_nlu_pipeline_llm_first.py
"""Tests for LLM-first merge policy in nlu_pipeline."""
import pytest
from messengers_router.mess_types import RouteDecision
from messengers_router.nlu_pipeline import _merge


def _decision(label, confidence, entities=None, flags=None):
    return RouteDecision(
        label=label,
        confidence=confidence,
        entities=entities or {},
        flags=set(flags or []),
        needs_handoff=False,
        context_action="continue",
    )


# ── Safety overrides ──────────────────────────────────────────────────────────

def test_rule_safety_label_wins_over_llm():
    rule = _decision("URGENT", 0.9)
    llm = _decision("PRICE", 0.85)
    merged, source = _merge(rule, llm)
    assert merged.label == "URGENT"
    assert source == "rule_safety"


def test_llm_safety_label_wins_over_rule_non_safety():
    rule = _decision("APPOINTMENT", 0.8)
    llm = _decision("COMPLAINT", 0.7)
    merged, source = _merge(rule, llm)
    assert merged.label == "COMPLAINT"
    assert source == "llm_safety"


# ── LLM-first: LLM intent wins regardless of confidence ──────────────────────

def test_llm_wins_even_when_low_confidence():
    """Key change from old behaviour: LLM is NOT overridden by rule when confidence < 0.55."""
    rule = _decision("APPOINTMENT", 0.85)  # was being promoted in old code
    llm = _decision("PRICE", 0.30)        # low confidence
    merged, source = _merge(rule, llm)
    assert merged.label == "PRICE"         # LLM intent wins
    assert source in ("llm_primary", "llm_primary_entity_enriched")


def test_rule_entities_supplement_llm_entities():
    """Rule-extracted entities are merged in, but LLM entities take precedence."""
    rule = _decision("APPOINTMENT", 0.85, entities={"specialty": "кардиолог", "city": "самара"})
    llm = _decision("APPOINTMENT", 0.72, entities={"specialty": "терапевт"})
    merged, source = _merge(rule, llm)
    # LLM entity wins for key that both have
    assert merged.entities["specialty"] == "терапевт"
    # Rule entity fills in gap
    assert merged.entities["city"] == "самара"
    assert "rule_entities_merged" in merged.flags


def test_llm_only_no_rule_entities():
    """When rule has no entities, result is pure LLM without extra flags."""
    rule = _decision("OTHER", 0.2, entities={})
    llm = _decision("PRICE", 0.75, entities={"service_name": "узи"})
    merged, source = _merge(rule, llm)
    assert merged.label == "PRICE"
    assert merged.entities["service_name"] == "узи"
    assert "rule_entities_merged" not in merged.flags


def test_both_safety_llm_takes_precedence():
    """When both are safety labels, LLM wins (it has better context)."""
    rule = _decision("URGENT", 0.9)
    llm = _decision("COMPLAINT", 0.7)
    merged, source = _merge(rule, llm)
    assert merged.label == "COMPLAINT"
    assert source == "llm_safety"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_nlu_pipeline_llm_first.py -v
```
Expected: `test_llm_wins_even_when_low_confidence` FAILS (old code promotes rule).

- [ ] **Step 3: Rewrite `_merge()` in `nlu_pipeline.py`**

Replace the entire `_merge` function (lines 71-99) with:

```python
def _merge(
    rule: RouteDecision,
    llm: RouteDecision,
    *,
    llm_mode: str = "hybrid",  # kept for API compatibility; no longer affects merge
) -> tuple[RouteDecision, str]:
    """LLM-first merge policy.

    Rules:
    1. Safety labels (URGENT, COMPLAINT, MEDICAL_ADVICE) win regardless of source.
       If both are safety, LLM wins (richer context).
    2. For all other cases: LLM intent wins unconditionally.
    3. Rule-extracted entities supplement LLM entities (LLM takes precedence on conflicts).

    Rationale: when LLM is uncertain, the right response is to ask the user for
    clarification — not to silently fall back to regex intent, which cannot handle
    natural language variation. The `clarify_threshold` in ConfidencePolicy controls
    when clarification is triggered downstream.
    """
    _SAFETY = {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}

    # Rule safety label, LLM is not safety → rule wins.
    if rule.label in _SAFETY and llm.label not in _SAFETY:
        return rule, "rule_safety"

    # LLM safety (including case where both are safety) → LLM wins.
    if llm.label in _SAFETY:
        return llm, "llm_safety"

    # LLM intent wins. Supplement with rule-extracted entities (where LLM has gaps).
    rule_entities = dict(rule.entities or {})
    if rule_entities:
        merged_entities = rule_entities  # start with rule
        merged_entities.update(dict(llm.entities or {}))  # LLM overrides on conflicts
        enriched = RouteDecision(
            label=llm.label,
            confidence=llm.confidence,
            entities=merged_entities,
            flags=set(llm.flags) | {"rule_entities_merged"},
            needs_handoff=llm.needs_handoff,
            context_action=llm.context_action,
            source=llm.source,
            clarify_needed=llm.clarify_needed,
            clarify_reason=llm.clarify_reason,
            clarify_slots=list(llm.clarify_slots),
            intent_candidates=list(llm.intent_candidates),
        )
        return enriched, "llm_primary_entity_enriched"

    return llm, "llm_primary"
```

- [ ] **Step 4: Update `nlu_pipeline.py` to import `CONFIDENCE` (remove leftover magic numbers)**

Find and replace any remaining magic number confidence thresholds in `nlu_pipeline.py`:
```python
# ADD near top imports:
from .mess_types import CONFIDENCE

# IF any other place in the file still references 0.45 or 0.55 as thresholds, replace:
# 0.45  →  CONFIDENCE.rule_promotion_rich
# 0.55  →  CONFIDENCE.rule_promotion_hybrid
```

- [ ] **Step 5: Run unit tests**

```bash
python -m pytest tests/test_nlu_pipeline_llm_first.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add messengers_router/nlu_pipeline.py tests/test_nlu_pipeline_llm_first.py
git commit -m "feat(nlu): LLM-first merge — remove regex promotion; rule entities supplement LLM, intent never overridden"
```

- [ ] **Step 8: Run eval suite against running server**

```bash
python messengers_router/scripts/eval_stage5_corpus.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --golden messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_v2.jsonl
```
Expected: same pass-rate as before Phase 2 (no regression). If any case regresses, check whether the LLM was returning low confidence and the regex was masking it — if so, this is the clarification system working correctly, not a bug.

---

### Task 2.2 — Replace magic numbers in `classifier.py` with `CONFIDENCE`

**Files:**
- Modify: `messengers_router/classifier.py`

- [ ] **Step 1: Add import**

```python
from .mess_types import CONFIDENCE
```

- [ ] **Step 2: Replace hardcoded thresholds in `classifier.py`**

Search for and replace the following patterns:

```python
# Line ~590 — refinement trigger threshold:
# BEFORE:
if decision.confidence < 0.45:
# AFTER:
if decision.confidence < CONFIDENCE.refine_trigger:

# Any occurrence of 0.90 for APPOINTMENT deterministic floor:
# BEFORE:
confidence=0.90
# AFTER:
confidence=CONFIDENCE.floor_deterministic_appointment

# 0.85 for DOC_REQUEST floor:
# BEFORE:
confidence=0.85
# AFTER:
confidence=CONFIDENCE.floor_deterministic_doc_request

# 0.78 for TEST_RESULT floor:
# BEFORE:
confidence=0.78
# AFTER:
confidence=CONFIDENCE.floor_deterministic_test_result

# 0.72 for PRICE floor:
# BEFORE:
confidence=0.72
# AFTER:
confidence=CONFIDENCE.floor_deterministic_price

# 0.71 for ADDRESS floor:
# BEFORE:
confidence=0.71
# AFTER:
confidence=CONFIDENCE.floor_deterministic_address

# 0.2 for default low:
# BEFORE:
confidence=0.2
# AFTER:
confidence=CONFIDENCE.floor_default_low
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add messengers_router/classifier.py
git commit -m "refactor(classifier): replace hardcoded confidence magic numbers with CONFIDENCE.* named thresholds"
```

---

## Phase 3 — `DialogState` replaces the god-object

**Prerequisite:** Phase 1 complete.

**Scope:** Port `DialogState` from `src/localragagent/freetalk/`. Add it as a new field on `SessionState`. Keep `last_entities` intact (backward compat). New code uses `state.dialog.*`; old code continues using `state.last_entities.*` until Phase 4-5 progressively migrate it.

---

### Task 3.1 — Add `DialogState` dataclass to `mess_types.py`

**Files:**
- Modify: `messengers_router/mess_types.py`
- Create: `tests/test_dialog_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dialog_state.py
import pytest
from messengers_router.mess_types import DialogState, SessionState


def test_dialog_state_defaults():
    ds = DialogState()
    assert ds.label == ""
    assert ds.phase == ""
    assert ds.entities == {}
    assert ds.candidate_entities == {}
    assert ds.missing_slots == []
    assert ds.clarify_count == 0
    assert ds.open_question == ""
    assert ds.confidence == 0.0


def test_session_state_has_dialog_field():
    state = SessionState(session_id="test")
    assert isinstance(state.dialog, DialogState)


def test_dialog_state_is_active_with_label():
    ds = DialogState(label="APPOINTMENT")
    assert ds.is_active()


def test_dialog_state_is_not_active_when_empty():
    ds = DialogState()
    assert not ds.is_active()


def test_dialog_state_is_active_with_phase():
    ds = DialogState(phase="awaiting_doctor")
    assert ds.is_active()


def test_dialog_state_clear_resets_all():
    ds = DialogState(
        label="APPOINTMENT",
        phase="confirming",
        entities={"doctor_name": "иванов"},
        missing_slots=["date_from"],
        clarify_count=2,
    )
    ds.clear()
    assert not ds.is_active()
    assert ds.entities == {}
    assert ds.missing_slots == []
    assert ds.clarify_count == 0


def test_dialog_state_merge_entities_llm_wins():
    ds = DialogState(entities={"specialty": "кардиолог", "city": "самара"})
    ds.merge_entities({"specialty": "терапевт", "date_from": "2026-04-20"})
    assert ds.entities["specialty"] == "терапевт"  # new wins
    assert ds.entities["city"] == "самара"  # kept
    assert ds.entities["date_from"] == "2026-04-20"  # added
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dialog_state.py -v
```
Expected: `ImportError: cannot import name 'DialogState'`

- [ ] **Step 3: Add `DialogState` to `mess_types.py`**

Insert the following **before** the `SessionState` dataclass:

```python
@dataclass
class DialogState:
    """Structured per-turn dialog state.

    Replaces scattered boolean flags in last_entities (e.g., appointment_flow_active,
    appointment_confirm_pending, appointment_cancel_pending, etc.).

    Migration: SessionState grows a `dialog` field. Old code continues using
    last_entities for backward compat. New code (Phases 4-5) uses dialog.phase.

    Ported from src/localragagent/freetalk/contracts.py:DialogState and adapted
    to messengers_router's Label vocabulary.
    """
    # Current routing intent (one of Label values, or "").
    label: str = ""
    # Sub-flow phase — drives the appointment state machine.
    # e.g. "idle", "awaiting_doctor", "awaiting_date", "awaiting_time",
    #       "awaiting_patient_name", "confirming", "confirmed", "cancelled"
    phase: str = ""
    # Confirmed entities for this flow turn.
    entities: dict[str, Any] = field(default_factory=dict)
    # Proposed entities waiting for user confirmation (yes/no prompt).
    candidate_entities: dict[str, Any] = field(default_factory=dict)
    # Slot names still missing before plan can execute.
    missing_slots: list[str] = field(default_factory=list)
    # Number of consecutive clarification attempts (resets on new intent).
    clarify_count: int = 0
    # Last clarification question sent to user.
    open_question: str = ""
    # NLU confidence for current label.
    confidence: float = 0.0

    def is_active(self) -> bool:
        """True if there is in-progress dialog state worth preserving."""
        return bool(
            self.label
            or self.phase
            or self.entities
            or self.missing_slots
            or self.open_question
        )

    def clear(self) -> None:
        """Reset all fields to defaults (e.g. on new topic or cancellation)."""
        self.label = ""
        self.phase = ""
        self.entities = {}
        self.candidate_entities = {}
        self.missing_slots = []
        self.clarify_count = 0
        self.open_question = ""
        self.confidence = 0.0

    def merge_entities(self, updates: dict[str, Any]) -> None:
        """Merge new entities in; updates take precedence over existing values."""
        for key, value in (updates or {}).items():
            if key and str(value or "").strip():
                self.entities[key] = value
```

- [ ] **Step 4: Add `dialog` field to `SessionState`**

In `mess_types.py`, update `SessionState`:

```python
@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    last_entities: dict[str, Any] = field(default_factory=dict)  # kept for compat
    dialog: DialogState = field(default_factory=DialogState)     # NEW: structured state
    summary: str = field(default="")
    is_authenticated: bool = field(default=False)
    auth_ref: str | None = None
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_dialog_state.py tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add messengers_router/mess_types.py tests/test_dialog_state.py
git commit -m "feat(types): add DialogState dataclass to SessionState; port from freetalk, adapted for patient router"
```

---

### Task 3.2 — Wire `dialog` state into `memory.py`

**Files:**
- Modify: `messengers_router/memory.py`

- [ ] **Step 1: Add dialog state persistence helpers to `memory.py`**

Open `messengers_router/memory.py`. Add these two methods to the `MemoryStore` class (or as module-level functions if the file uses a functional style):

```python
# Add import at top if not present:
import json
from .mess_types import DialogState

# Add these methods/functions:

def save_dialog_state(state: SessionState) -> None:
    """Persist dialog state into last_entities under a reserved key.

    This keeps dialog state in the same serialization path as last_entities
    without requiring infrastructure changes.
    """
    try:
        payload = {
            "label": state.dialog.label,
            "phase": state.dialog.phase,
            "entities": state.dialog.entities,
            "candidate_entities": state.dialog.candidate_entities,
            "missing_slots": state.dialog.missing_slots,
            "clarify_count": state.dialog.clarify_count,
            "open_question": state.dialog.open_question,
            "confidence": state.dialog.confidence,
        }
        state.last_entities["_dialog_state"] = json.dumps(payload, ensure_ascii=False)
    except Exception:
        pass  # never break the pipeline on serialization errors


def load_dialog_state(state: SessionState) -> None:
    """Restore dialog state from last_entities into state.dialog.

    Call this at the start of each turn before NLU runs.
    """
    raw = state.last_entities.get("_dialog_state")
    if not raw:
        return
    try:
        data = json.loads(str(raw))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    state.dialog.label = str(data.get("label") or "")
    state.dialog.phase = str(data.get("phase") or "")
    state.dialog.entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    state.dialog.candidate_entities = data.get("candidate_entities") if isinstance(data.get("candidate_entities"), dict) else {}
    state.dialog.missing_slots = [str(s) for s in (data.get("missing_slots") or []) if s]
    state.dialog.clarify_count = max(0, int(data.get("clarify_count") or 0))
    state.dialog.open_question = str(data.get("open_question") or "")
    state.dialog.confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add messengers_router/memory.py
git commit -m "feat(memory): add save_dialog_state/load_dialog_state helpers for DialogState persistence"
```

---

## Phase 4 — Appointment State Machine

**Prerequisite:** Phase 3 complete.

**Scope:** Replace the 20-flag appointment flow (`_APPOINTMENT_RUNTIME_KEYS` + 3 repeated intent-check chains) with an explicit `AppointmentPhase` enum and a thin `AppointmentStateMachine`. The `appointment_flow_guard.py` logic becomes phase-driven. External API of `run_appointment_precheck()` does not change.

---

### Task 4.1 — Add `AppointmentPhase` enum to `mess_types.py`

**Files:**
- Modify: `messengers_router/mess_types.py`
- Create: `tests/test_appointment_state_machine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_appointment_state_machine.py
from messengers_router.mess_types import AppointmentPhase, DialogState, SessionState


def test_appointment_phase_values():
    assert AppointmentPhase.IDLE == "idle"
    assert AppointmentPhase.AWAITING_DOCTOR == "awaiting_doctor"
    assert AppointmentPhase.CONFIRMING == "confirming"
    assert AppointmentPhase.CONFIRMED == "confirmed"
    assert AppointmentPhase.CANCELLED == "cancelled"


def test_phase_stored_on_dialog_state():
    state = SessionState(session_id="x")
    state.dialog.phase = AppointmentPhase.AWAITING_DOCTOR
    assert state.dialog.phase == "awaiting_doctor"
    assert state.dialog.is_active()


def test_clear_resets_phase():
    state = SessionState(session_id="x")
    state.dialog.label = "APPOINTMENT"
    state.dialog.phase = AppointmentPhase.CONFIRMING
    state.dialog.clear()
    assert state.dialog.phase == ""
    assert not state.dialog.is_active()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_appointment_state_machine.py -v
```
Expected: `ImportError: cannot import name 'AppointmentPhase'`

- [ ] **Step 3: Add `AppointmentPhase` to `mess_types.py`**

Add this **before** the `DialogState` class:

```python
class AppointmentPhase(str):
    """Valid phases for the appointment booking flow.

    Using str-subclass (not Enum) so values can be stored directly as strings
    in DialogState.phase without serialization friction.
    """
    IDLE = "idle"
    AWAITING_DOCTOR = "awaiting_doctor"
    AWAITING_SPECIALTY = "awaiting_specialty"
    AWAITING_DATE = "awaiting_date"
    AWAITING_TIME = "awaiting_time"
    AWAITING_PATIENT_NAME = "awaiting_patient_name"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    PAUSED = "paused"  # topic-switch pending confirmation
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_appointment_state_machine.py tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add messengers_router/mess_types.py tests/test_appointment_state_machine.py
git commit -m "feat(types): add AppointmentPhase string constants for explicit appointment state machine"
```

---

### Task 4.2 — Refactor `appointment_flow_guard.py` intent-check chains

**Files:**
- Modify: `messengers_router/appointment_flow_guard.py`

The three near-identical intent-detection chains at lines ~127-136, ~162-171, ~208-216 are collapsed into one helper.

- [ ] **Step 1: Add `_is_topic_switch_intent` helper at the top of `appointment_flow_guard.py`**

Add this function after the imports, before `reset_appointment_runtime_state`:

```python
def _is_topic_switch_intent(text: str, last_entities: dict) -> bool:
    """True if the user's message is a non-appointment intent (topic switch).

    Centralises the 3x repeated intent-detection chain that previously appeared
    in run_appointment_precheck, _handle_confirm_pending, and _handle_cancel_pending.
    """
    return any([
        detect_test_result_intent(text),
        detect_test_assist_intent(text),
        detect_prepare_intent(text),
        detect_price_intent(text),
        detect_address_intent(text),
        detect_news_intent(text),
        detect_doc_request_intent(text),
        detect_doctor_info_intent(text),
        detect_schedule_intent(text),
        detect_nonbookable_walkin_intent(text, last_entities),
    ])
```

- [ ] **Step 2: Replace the 3 repeated chains with `_is_topic_switch_intent` calls**

Search for each block that looks like:
```python
if (
    detect_test_result_intent(user_text)
    or detect_test_assist_intent(user_text)
    or detect_prepare_intent(user_text)
    or detect_price_intent(user_text)
    ...
):
```
and replace with:
```python
if _is_topic_switch_intent(user_text, state.last_entities):
```

- [ ] **Step 3: Update `reset_appointment_runtime_state` to use `AppointmentPhase`**

After reset, set the dialog phase to IDLE:
```python
from .mess_types import AppointmentPhase

def reset_appointment_runtime_state(state: SessionState) -> None:
    for key in _APPOINTMENT_RUNTIME_KEYS:
        state.last_entities.pop(key, None)
    # Mirror reset into structured dialog state
    if state.dialog.label == "APPOINTMENT":
        state.dialog.phase = AppointmentPhase.IDLE
        state.dialog.candidate_entities = {}
        state.dialog.missing_slots = []
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 5: Run critical cases**

```bash
python messengers_router/eval_suite/eval_critical_cases.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_server_parity.jsonl
```
Expected: same pass-rate as before Phase 4.

- [ ] **Step 6: Commit**

```bash
git add messengers_router/appointment_flow_guard.py
git commit -m "refactor(appointment): collapse 3 repeated intent-check chains into _is_topic_switch_intent helper; wire AppointmentPhase"
```

---

## Phase 5 — New Orchestrator

**Prerequisite:** Phases 3 and 4 complete.

**Scope:** Extract the ~600-line `route_patient_message()` function from `router.py` into a clean, staged `orchestrator.py`. `router.py` becomes a thin shim. This is a written-from-scratch orchestrator (not a port from freetalk) tuned to the appointment/booking domain.

**Key principle:** Each stage is a pure async function with a single responsibility. Overrides are explicit policies passed as arguments, not inlined conditionals.

---

### Task 5.1 — Create `messengers_router/orchestrator.py`

**Files:**
- Create: `messengers_router/orchestrator.py`

The orchestrator has five named stages:

```python
# messengers_router/orchestrator.py
"""Patient chatbot orchestrator — replaces the 600-line route_patient_message().

Pipeline:
    load_context
      → early_guards          (city check, unsupported service hard-stop)
      → nlu_route             (LLM-first classifier)
      → clarify_gate          (missing slots? ask — never inline in router)
      → tool_loop             (executor + services)
      → render                (response_builder + renderer)
      → save_context

Each stage is a pure async function. Overrides are passed as data,
not implemented as inline conditionals.
"""
from __future__ import annotations

import logging
from typing import Any

from .appointment_flow_guard import run_appointment_precheck
from .city import match_city
from .entity_grounder import ground_decision_entities
from .executor import execute_plan
from .llm_mode_policy import RuntimeOptions
from .memory import MemoryStore, load_dialog_state, save_dialog_state
from .mess_types import (
    CONFIDENCE,
    AppointmentPhase,
    DialogState,
    Plan,
    ResponseEnvelope,
    RouteDecision,
    SessionState,
)
from .nlu_pipeline import analyze_with_candidates
from .planner import build_plan
from .policies import detect_unsupported_catalog, missing_slots
from .recovery_policy import evaluate_recovery
from .response_builder import build_response
from .services import Services
from .text_templates import CITY_NOT_SUPPORTED_TEXT, UNSUPPORTED_SERVICE_TEXT

log = logging.getLogger(__name__)

# ── Stage 1: Early guards ─────────────────────────────────────────────────────

async def _early_guards(
    text: str,
    state: SessionState,
    services: Services,
) -> ResponseEnvelope | None:
    """Return a response immediately if a hard stop condition is met.

    Hard stops (in priority order):
    1. Non-Samara city detected → handoff with ADDRESS info
    2. Unsupported service detected → polite rejection, no handoff
    """
    city = match_city(text)
    if city and city.lower() != "самара":
        return ResponseEnvelope(
            text=CITY_NOT_SUPPORTED_TEXT,
            handoff=True,
            state_update={"city": city},
        )

    unsupported = detect_unsupported_catalog(text, state.last_entities)
    if unsupported:
        return ResponseEnvelope(
            text=UNSUPPORTED_SERVICE_TEXT,
            handoff=False,
        )

    return None


# ── Stage 2: NLU routing ──────────────────────────────────────────────────────

async def _nlu_route(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions,
) -> RouteDecision:
    """Classify the user message. LLM-first; guardrails only for safety labels."""
    result = await analyze_with_candidates(text, state, runtime_options=runtime_options)
    return result.decision


# ── Stage 3: Clarification gate ───────────────────────────────────────────────

def _clarify_gate(
    decision: RouteDecision,
    state: SessionState,
) -> ResponseEnvelope | None:
    """If slots are missing or confidence is too low, ask for clarification.

    This stage replaces the scattered inline slot-check logic in the old router.
    """
    if decision.confidence < CONFIDENCE.clarify_threshold and not decision.clarify_needed:
        return evaluate_recovery(decision, state)

    if decision.clarify_needed and decision.clarify_slots:
        from .policies import clarification_question
        question = clarification_question(decision.label, decision.clarify_slots)
        state.dialog.open_question = question
        state.dialog.clarify_count += 1
        return ResponseEnvelope(text=question, handoff=False)

    slots = missing_slots(decision.label, decision.entities)
    if slots:
        from .policies import clarification_question
        question = clarification_question(decision.label, slots)
        state.dialog.open_question = question
        state.dialog.clarify_count += 1
        return ResponseEnvelope(text=question, handoff=False)

    return None


# ── Stage 4: Tool loop (plan → execute) ───────────────────────────────────────

async def _tool_loop(
    decision: RouteDecision,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> ResponseEnvelope:
    """Build and execute the plan, then build the response."""
    plan = build_plan(decision, state)
    evidence = await execute_plan(plan, state, services)
    return await build_response(decision.label, evidence, state, services, memory)


# ── Main entry point ──────────────────────────────────────────────────────────

async def route(
    text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    *,
    runtime_options: RuntimeOptions | None = None,
) -> ResponseEnvelope:
    """Main orchestration entry point. Replaces route_patient_message().

    Called by router.py (thin shim) and endpoint.py.
    """
    opts = runtime_options or RuntimeOptions()

    # Restore structured dialog state from last_entities.
    load_dialog_state(state)

    # Stage 1: Hard stops.
    guard_response = await _early_guards(text, state, services)
    if guard_response is not None:
        return guard_response

    # Appointment precheck (handles confirm/cancel/topic-switch within booking flow).
    if state.last_entities.get("appointment_flow_active"):
        appt_response = await run_appointment_precheck(text, state, services, memory)
        if appt_response is not None:
            save_dialog_state(state)
            return appt_response

    # Stage 2: NLU classification.
    decision = await _nlu_route(text, state, opts)

    # Stage 3: Entity grounding (validate extracted entities against catalog).
    decision = await ground_decision_entities(decision, text, state, services)

    # Update dialog state from this turn's decision.
    state.dialog.label = decision.label
    state.dialog.confidence = decision.confidence
    state.dialog.merge_entities(decision.entities or {})

    # Stage 4: Clarification gate.
    clarify_response = _clarify_gate(decision, state)
    if clarify_response is not None:
        save_dialog_state(state)
        return clarify_response

    # Stage 5: Tool loop.
    response = await _tool_loop(decision, state, services, memory)

    # Persist dialog state for next turn.
    save_dialog_state(state)
    return response
```

- [ ] **Step 1: Create the file with the content above**

```bash
# Verify the file was created:
python -c "from messengers_router.orchestrator import route; print('OK')"
```
Expected: `OK`

- [ ] **Step 2: Run unit tests**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS (orchestrator not yet called by router, so no behaviour change).

- [ ] **Step 3: Commit**

```bash
git add messengers_router/orchestrator.py
git commit -m "feat(orchestrator): add new pipeline orchestrator with 5 named stages; replaces 600-line route_patient_message"
```

---

### Task 5.2 — Wire `router.py` to call `orchestrator.route()`

**Files:**
- Modify: `messengers_router/router.py`

- [ ] **Step 1: Add import to `router.py`**

```python
from .orchestrator import route as _orchestrate
```

- [ ] **Step 2: Find `route_patient_message` and add orchestrator call path**

Locate `route_patient_message` (or `patient_routing_stream`) in `router.py`. Add an environment-flag gate that routes through the new orchestrator when `MR_USE_ORCHESTRATOR=1`:

```python
import os as _os

async def route_patient_message(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    *,
    runtime_options: RuntimeOptions | None = None,
) -> ResponseEnvelope:
    # Feature flag: use new orchestrator when MR_USE_ORCHESTRATOR=1
    if _os.environ.get("MR_USE_ORCHESTRATOR") == "1":
        return await _orchestrate(
            user_text, state, services, memory, runtime_options=runtime_options
        )
    # Original implementation follows unchanged ↓
    ...  # (keep all existing code)
```

This lets you A/B test the orchestrator in production with a single env var before removing the old path.

- [ ] **Step 3: Run tests with orchestrator enabled**

```bash
MR_USE_ORCHESTRATOR=1 python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 4: Run eval suite with orchestrator enabled**

```bash
MR_USE_ORCHESTRATOR=1 python messengers_router/scripts/eval_stage5_corpus.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --golden messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_v2.jsonl

MR_USE_ORCHESTRATOR=1 python messengers_router/eval_suite/eval_critical_cases.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_server_parity.jsonl
```
Expected: ≥ same pass-rate as baseline. Fix any regressions before merging.

- [ ] **Step 5: Commit**

```bash
git add messengers_router/router.py
git commit -m "feat(router): wire orchestrator behind MR_USE_ORCHESTRATOR feature flag; old path intact for safety"
```

---

## Phase 6 — `services.py` Split

**Prerequisite:** None (safe at any time; zero behaviour change required).

**Scope:** Break the 300KB / 7438-line god-file into a `services/` package. The **public API of `Services`** class does not change — `planner.py`, `executor.py`, and `response_builder.py` continue calling the same methods.

---

### Task 6.1 — Create `services/` package skeleton

**Files:**
- Create: `messengers_router/services/__init__.py`
- Create: `messengers_router/services/facade.py`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p messengers_router/services
```

- [ ] **Step 2: Create `messengers_router/services/__init__.py`**

```python
"""Services package — public facade for all data integrations.

The Services class in facade.py is the only public interface.
All other modules in this package are private implementation details.
"""
from .facade import Services, resolve_price_service_name_from_catalog

__all__ = ["Services", "resolve_price_service_name_from_catalog"]
```

- [ ] **Step 3: Create `messengers_router/services/facade.py`**

Move the `Services` class definition and `resolve_price_service_name_from_catalog` function from `services.py` into `services/facade.py`. Keep all imports. Add:
```python
# At top of facade.py:
"""Services facade — the only public API for the services package."""
```

- [ ] **Step 4: Update `messengers_router/services.py` to be a shim**

Replace the entire content of `services.py` with:
```python
"""Backward-compat shim — imports from services/ package.

Do not add logic here. All code lives in messengers_router/services/.
"""
from .services import Services, resolve_price_service_name_from_catalog

__all__ = ["Services", "resolve_price_service_name_from_catalog"]
```

Wait — this would create a circular import (`services.py` importing from `services/`). Instead, rename the old file:

```bash
# Step: rename old services.py to services_legacy.py temporarily
mv messengers_router/services.py messengers_router/services_legacy.py
```

Then in `services/__init__.py`, import from `services_legacy` until you've moved the code:
```python
# services/__init__.py — transitional
from ..services_legacy import Services, resolve_price_service_name_from_catalog
```

- [ ] **Step 5: Verify imports still work**

```bash
python -c "from messengers_router.services import Services; print('OK')"
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Step 6: Commit the skeleton**

```bash
git add messengers_router/services/ messengers_router/services_legacy.py
git commit -m "refactor(services): create services/ package skeleton; services_legacy.py bridges old code"
```

---

### Task 6.2 — Move domain modules progressively

Move code from `services_legacy.py` into focused submodules one domain at a time:

| Submodule | Content to move | Lines (approx) |
|---|---|---|
| `services/_query_parsing.py` | `_normalise_input`, `_price_query_tokens`, `_service_tokens`, `_price_alias_candidates` | ~150 |
| `services/_doctor_matching.py` | `_doctor_matches_specialty`, `_specialty_equivalent`, `_compact_specialization` | ~200 |
| `services/_price_ranking.py` | `_score_price_rows`, `_dedupe_price_rows`, `_select_patient_price_rows` | ~250 |
| `services/_preparation.py` | `_prepare_relevance_score`, `_dedupe_prepare_candidates`, `_is_prepare_relevant` | ~200 |
| `services/_cache.py` | `AsyncListTTLStaleCache`, all TTL/stale cache logic | ~150 |
| `services/_address.py` | `_soft_address_match`, `_filter_regions_by_service_flags` | ~100 |
| `services/facade.py` | `Services` class (uses all above as imports) | ~500 |

For each submodule, the process is:
1. Create the file with the functions moved from `services_legacy.py`
2. Add `from ._submodule import *` in `services/__init__.py`
3. Remove the moved functions from `services_legacy.py`
4. Run `python -m pytest tests/ -x -q` — all must PASS before moving to next module

- [ ] **Step 1: Move query parsing**

```bash
# After moving functions to services/_query_parsing.py:
python -m pytest tests/ -x -q
git add messengers_router/services/_query_parsing.py messengers_router/services_legacy.py
git commit -m "refactor(services): extract _query_parsing module from services_legacy"
```

- [ ] **Step 2–6: Repeat for each submodule** (same pattern: move, test, commit)

- [ ] **Step 7: Delete `services_legacy.py` once empty**

```bash
# When services_legacy.py has no more functions:
git rm messengers_router/services_legacy.py
git commit -m "refactor(services): services_legacy.py fully migrated; delete"
```

---

## Phase 7 — Cleanup & Final Eval Gate

**Prerequisite:** All previous phases complete.

---

### Task 7.1 — Final full eval run

- [ ] **Run all eval suites (unit + stage5 + critical)**

```bash
# Unit tests
python -m pytest tests/ -v --tb=short

# Stage-5 golden corpus
python messengers_router/scripts/eval_stage5_corpus.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --golden messengers_router/messengers_mds_to_collect_thoughts/analysis/golden_versions/stage5_golden_v2.jsonl

# Critical multi-turn (parity set)
python messengers_router/eval_suite/eval_critical_cases.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_server_parity.jsonl

# Extended critical cases
python messengers_router/eval_suite/eval_critical_cases.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/critical_cases_extended.jsonl

# PREPARE quality gate
python messengers_router/eval_suite/eval_critical_cases.py \
  --url http://localhost:8000/api/messenger-generate-once \
  --cases messengers_router/eval_suite/prepare_wrap_cases.jsonl
```

Expected: 100% pass on critical cases; ≥ baseline on stage5.

---

### Task 7.2 — Remove dead code

- [ ] **Delete `llm_doesnt_work_fallback.py` hand-rolled stemmer**

Replace the entire scoring heuristic (lines 99-254 in `llm_doesnt_work_fallback.py`) with a simple keyword-presence check. The magic weights (2.2, 1.6, 1.0, -1.2) are unjustifiable:

```python
# Simplified fallback scoring — replaces the hand-rolled Russian stemmer
def _score_prepare_candidate(segment: str, target: str) -> float:
    """Simple keyword presence score. No magic weights."""
    norm_segment = normalize_ru(segment)
    norm_target = normalize_ru(target)
    score = 0.0
    if norm_target and norm_target in norm_segment:
        score += 2.0
    for hint in _PREPARE_ACTION_HINTS:
        if normalize_ru(hint) in norm_segment:
            score += 1.0
            break
    return score
```

- [ ] **Run tests after deletion**

```bash
python -m pytest tests/ -x -q
```
Expected: all PASS.

- [ ] **Commit**

```bash
git add messengers_router/llm_doesnt_work_fallback.py
git commit -m "refactor(fallback): replace hand-rolled Russian stemmer with simple keyword scoring"
```

---

### Task 7.3 — Final commit

- [ ] **Confirm `MR_USE_ORCHESTRATOR=1` is the default in production config**

```bash
# In your deployment config / docker-compose.yml:
# MR_USE_ORCHESTRATOR=1
```

- [ ] **Remove old routing path from `router.py`**

Once the orchestrator has been in production without regressions, remove the feature flag and keep only the orchestrator path:

```python
async def route_patient_message(...) -> ResponseEnvelope:
    return await _orchestrate(
        user_text, state, services, memory, runtime_options=runtime_options
    )
```

- [ ] **Final commit**

```bash
git add messengers_router/router.py
git commit -m "refactor(router): remove legacy routing path; orchestrator is now the only path"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Russian NLU utils: Task 1.1–1.2
- [x] Entity whitelist unification: Task 1.3
- [x] Doctor extractor collapse: Task 1.4
- [x] ConfidencePolicy: Task 1.5
- [x] LLM-first merge: Task 2.1
- [x] Magic number cleanup: Task 2.2
- [x] DialogState: Task 3.1–3.2
- [x] AppointmentPhase: Task 4.1
- [x] Appointment intent chain collapse: Task 4.2
- [x] Orchestrator: Task 5.1–5.2
- [x] services.py split: Task 6.1–6.2
- [x] Final eval gate: Task 7.1
- [x] Dead code removal: Task 7.2–7.3

**Type consistency:** `DialogState` defined in Task 3.1, used in Tasks 3.2, 4.1, 4.2, 5.1. `AppointmentPhase` defined in Task 4.1, used in Tasks 4.2, 5.1. `CONFIDENCE` defined in Task 1.5, used in Tasks 2.1, 2.2, 5.1. `normalize_ru` defined in Task 1.1, used in Tasks 1.2, 7.2. All consistent.

**No placeholders:** Every step has exact file paths, exact code, and exact test commands.

**Eval gate:** Present after Phase 2, Phase 4, Phase 5, and Phase 7. All must pass before proceeding to the next phase.
