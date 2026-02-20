# Golden Versions Changelog

## v1
- Source: `analysis/stage5_golden_cases.jsonl`
- Captured during router v1/v1.5 stabilization cycle.
- Notes: baseline for stage5 gate and regression comparisons.

## v2
- Source: `analysis/golden_versions/stage5_golden_v2.jsonl`
- Captured during router v2 rollout (graph + dual-pass NLU + soft-HITL policy).
- Updated expectations:
  - Nonbookable intents (`ЭКГ`/`анализы`) map to `ADDRESS` instead of `APPOINTMENT`/`TEST_ASSIST`.
  - Price-dominant test-assist messages map to `PRICE`.
  - Ambiguous low-signal phrase `G037` maps to `OTHER`.
  - `OTHER` fallback now uses soft escalation (handoff not immediate) for `G046`/`G047`.
