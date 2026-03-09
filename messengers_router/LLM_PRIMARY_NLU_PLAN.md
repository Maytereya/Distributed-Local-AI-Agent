# LLM-Primary NLU Design

## Goal

Сместить обычную patient-NLU интерпретацию в сторону LLM-first without breaking:
- safety guardrails,
- Samara-only routing,
- deterministic service/tool orchestration,
- current messenger HTTP contract.

## Current Implementation

Реализовано в коде:
- `MR_NLU_ENGINE=legacy_v2|llm_primary`
- `MR_NLU_SHADOW=0|1`
- `llm_mode` semantics:
  - `strict`: legacy deterministic/fallback path
  - `hybrid`: LLM-primary NLU
  - `rich`: LLM-primary NLU + renderer self-check + optional refine

## Pipeline

### `legacy_v2`

`deterministic_rule_decision -> legacy analyze -> merge`

Это оставлено как fallback/escape hatch для rollout.

### `llm_primary`

`guardrail_pre -> primary LLM JSON classify -> deterministic postprocess -> grounding -> router`

Guardrails оставлены только для:
- `URGENT`
- `COMPLAINT`
- `MEDICAL_ADVICE`
- `doc_request_handoff`
- greeting/smalltalk shortcut

Остальные обычные labels идут через primary LLM classify.

## Structured Clarify

Primary LLM JSON now returns:
- `clarify_needed`
- `clarify_reason`
- `clarify_slots`
- `intent_candidates`

Patient-facing clarify text remains deterministic.

Причины:
- меньше latency,
- меньше prompt drift,
- easier testing,
- no free-form clarify hallucinations.

## Grounding Policy

- `doctor_name`: only after doctor cache verification
- `city`: only via `match_city`
- `service_name`: current-turn extraction first, then LLM fallback phrase if non-generic

## Debug Trace

Debug endpoint should expose:
- `guardrail_pre`
- `llm_primary_raw`
- `llm_primary_sanitized`
- `guardrail_post`
- `final_decision`

This is required for shadow rollout and regression analysis.

## Rollout Notes

- Default engine remains `legacy_v2` until eval gates are met.
- `llm_primary` should first be compared in shadow mode.
- Existing Stage 5 corpus is not sufficient as the only gate.
- Expand eval with `DOCTOR_INFO`, `DOCTOR_SCHEDULE`, `PREPARE`, `OTHER`, non-Samara and follow-up turns from WhatsApp chat corpus.

## Current Verification Snapshot

- Core messenger regression suite after refactor: `76 passed`
- Covered tests:
  - `tests/test_llm_primary_nlu.py`
  - `tests/test_specialty_nlu.py`
  - `tests/test_router_flow_override.py`
  - `tests/test_entity_grounder.py`
  - `tests/test_messenger_services.py`
  - `tests/test_api_price_cache.py`
- Stage 5 corpus eval still requires a running local debug endpoint (`/api/messenger-generate-once`).
