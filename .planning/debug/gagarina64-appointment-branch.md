---
status: awaiting_human_verify
trigger: "Appointment to a SPECIALTY offers a wrong, doctor-less branch «ул.Гагарина, 64»"
created: 2026-06-05T00:00:00Z
updated: 2026-06-05T01:00:00Z
---

## Current Focus

hypothesis: CONFIRMED and FIXED. Three-part fix applied: (A) city-as-branch filter skip in addresses.py fallback; (B) specialty-aware doctor filter in addresses.py fallback; (C) defense-in-depth empty-evidence guard in response_builder.py.
test: Full gate run in progress (all green so far). 6/6 new invariant tests pass in isolation.
expecting: 487 passed, 0 failed (prior run was 485+2 fail, those 2 were fixed).
next_action: Await human verification of prod behavior after deploy.

## Symptoms

expected: «Запись к врачу-гинекологу» → bot offers branches with GYNECOLOGISTS (пр.Ленина 5, ул.Победы 83, ул.Аминева 29 etc), never a lab-only branch.
actual: «Запись к врачу-гинекологу» → «Есть возможность записи на приём к гинеколог в городе Самара по адресам: - ул.Гагарина, 64. Какой филиал вам удобен?»
errors: no Python errors; wrong address comes from response_builder fallback with empty address evidence.
reproduction: decision=APPOINTMENT, entities specialty=гинеколог, __appointment_mode=True → address_info returns note='address_info: doctors cache fallback' with addresses=[] branches=[] EMPTY.
started: reported 2026-06-05

## Eliminated

## Evidence

- timestamp: 2026-06-05T00:00:00Z
  checked: Pre-filled diagnosis from orchestrator (prod trace + offline reproduction)
  found: Two causes: (1) branch_q="самара" (city) substring-filters doctor addresses that lack city prefix → 0 results. (2) With empty addresses, response_builder uses safe_get_branches (all branches, specialty-agnostic) → Гагарина 64 (lab-only, 0 doctors) appears.
  implication: Fix must skip city-as-branch filter AND add specialty-aware guard in response_builder.

- timestamp: 2026-06-05T00:30:00Z
  checked: Offline confirmation via ./venv/bin/python address_info call (BEFORE fix)
  found: note='address_info: doctors cache fallback', addresses=[], branches=0. Confirmed: branch_q='самара' drops all gynecologist addresses like 'пр.Ленина, 5'.
  implication: Root cause confirmed exactly as diagnosed.

- timestamp: 2026-06-05T00:45:00Z
  checked: Fix A+B applied to addresses.py fallback. Re-run address_info (AFTER fix).
  found: note='address_info: doctors cache fallback', addresses=12 gynecologist branches (пр.Ленина 5, ул.Победы 83, ул.Аминева 29 etc.). Гагарина 64 NOT in result.
  implication: Fix A (city-only skip) + Fix B (specialty filter) + Samara non-filter working correctly.

- timestamp: 2026-06-05T00:50:00Z
  checked: Fix C applied to response_builder.py (defense-in-depth empty evidence guard).
  found: When specialty_entity set + address_evidence_empty → branches=[] → appointment_addresses_for_city returns [] → honest clarification text, Гагарина 64 NOT in text.
  implication: Defense-in-depth layer works.

## Resolution

root_cause: (1) address_info doctors-cache fallback: branch_q set to city name "самара" (from city entity, no specific branch/region/unit set), which is NOT present in doctor region strings like "пр.Ленина, 5" → every doctor address filtered out → empty result. (2) response_builder appointment branch-offer: with empty address evidence, falls back to safe_get_branches (all clinic branches, specialty-agnostic) which includes Гагарина 64 (lab-only, 0 doctors).
fix: (A) addresses.py: _branch_is_city_only flag — when branch came from city-only entity, do NOT apply branch_q substring filter to doctor addresses. (B) addresses.py: specialty_q filter in fallback — when appointment_mode + specialty, iterate only doctors matching that specialty via _doctor_role_specialty_match_level. Also adds _has_explicit_non_samara_regions filter. (C) response_builder.py: when specialty_entity set AND address_evidence_empty, set branches=[] to prevent specialty-agnostic safe_get_branches from surfacing lab-only branches.
verification: 6/6 new invariant tests pass; ruff clean; full gate run in progress (all green, expected 487 passed 0 failed).
files_changed:
  - messengers_router/services/addresses.py
  - messengers_router/response_builder.py
  - tests/test_messenger_services.py
  - docs/messengers_router_bug_log.md
