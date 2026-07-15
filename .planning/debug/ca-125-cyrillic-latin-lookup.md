---
status: awaiting_human_verify
trigger: "Investigate and fix: bot does not find analysis «Са-125» (CA-125 tumor marker)"
created: 2026-06-05T00:00:00Z
updated: 2026-06-05T01:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — root cause is routing gap. deterministic_rule_decision("Са-125", {}) returned None → LLM → generic TEST_ASSIST with empty entities → missing_slots fires → clarify. Fix applied and verified offline.
test: all gate tests green (213 passed test_messenger_services, 62 passed test_specialty_nlu, ruff clean)
expecting: human confirmation that «Са-125» now returns CA-125 price in live/staging env
next_action: await human verify

## Symptoms
<!-- Written during gathering, then IMMUTABLE -->

expected: user types «Са-125» → bot returns price/info for CA-125 tumor marker
actual: user types «Са-125» → bot replies generic clarify "Для какой цели хотите подобрать анализы?"
errors: no exception — wrong routing + empty entities
reproduction: offline deterministic_rule_decision("Са-125", {}, allow_refine=False) → None → falls to LLM → TEST_ASSIST with empty entities → missing_slots fires → clarify text
started: reported 2026-06-05 (dialog #166)

## Eliminated

- hypothesis: _rank_price_rows fails to match «Са-125» to catalog
  evidence: _rank_price_rows(rows, 'Са-125', limit=5) returns ['CA - 125 (яичники)'] — matching works via digit '125' token
  timestamp: 2026-06-05

- hypothesis: resolve_price_service_name_from_catalog fails
  evidence: returns 'CA - 125 (яичники)' correctly for "Са-125"
  timestamp: 2026-06-05

## Evidence

- timestamp: 2026-06-05
  checked: deterministic_rule_decision("Са-125", {}, allow_refine=False, attach_secondary=False)
  found: returns None — no deterministic rule fires
  implication: falls to LLM → generic TEST_ASSIST with empty entities

- timestamp: 2026-06-05
  checked: REQUIRED_SLOTS["TEST_ASSIST"] in policies.py
  found: ["_any_of:city,branch_name,branch_id", "_any_of:test_goal,test_name"] — test_goal and test_name are REQUIRED
  implication: LLM returning TEST_ASSIST with {} → missing_slots finds both missing → planner returns empty Plan → response_builder shows CLARIFY_TEXT_MAP["TEST_ASSIST"]

- timestamp: 2026-06-05
  checked: detect_test_assist_intent("Са-125") and detect_price_intent("Са-125")
  found: both False — no pattern matches
  implication: root cause confirmed — routing gap, not matching gap

- timestamp: 2026-06-05
  checked: _price_query_tokens("Са-125")
  found: ['125'] — only digit token extracted; "са" (2 chars) dropped by len<3 filter
  implication: matching via digit works for CA-125 family; homoglyph fold adds safety net for no-separator forms like "са125"

- timestamp: 2026-06-05
  checked: test_assist flow when query="Са-125" and test_name="Са-125"
  found: _rank_price_rows(rows, "Са-125", limit=10) → ['CA - 125 (яичники)'] cost=650
  implication: if routing worked, the bot WOULD show correct result — confirmed by end-to-end test

- timestamp: 2026-06-05
  checked: fix applied — deterministic_rule_decision("Са-125", {}, allow_refine=False)
  found: RouteDecision(label='TEST_ASSIST', entities={'test_name': 'Са-125'}, flags={'rule_test_assist_lab_code'})
  implication: routing now fires; test_name populated; missing_slots=[] with city in context

- timestamp: 2026-06-05
  checked: test_assist("Са-125", {"test_name": "Са-125", "city": "Самара"})
  found: {'tests': [{'serviceName': 'CA - 125 (яичники)', 'cost': 650.0, 'serviceHomecode': '225'}], 'note': 'test_assist: priceByRegion(3)'}
  implication: end-to-end pipeline works correctly after fix

## Resolution

root_cause: deterministic_rule_decision("Са-125", {}) returned None because no deterministic pattern recognized bare alphanumeric lab code as a specific test query. Pipeline fell to LLM which returned TEST_ASSIST with empty entities. REQUIRED_SLOTS["TEST_ASSIST"] requires test_goal/test_name → missing_slots gate fired → planner returned empty Plan → response_builder emitted CLARIFY_TEXT_MAP["TEST_ASSIST"] = "Для какой цели хотите подобрать анализы?" instead of calling test_assist() (which would have found the correct catalog entry via digit token "125").

fix:
  1. messengers_router/policies.py: added _CYR_TO_LAT map + fold_cyrillic_homoglyphs() + _LAB_CODE_RE pattern + detect_specific_lab_code_intent()
  2. messengers_router/classifier.py: imported detect_specific_lab_code_intent; added elif branch before detect_test_assist_intent → TEST_ASSIST with entities={"test_name": text} + flag "rule_test_assist_lab_code"
  3. messengers_router/services/_prices_helpers.py: added _PRICE_CYR_TO_LAT + _fold_homoglyph_token(); applied in _price_query_tokens as EXTRA digit-token candidate

verification: |
  BEFORE: deterministic_rule_decision("Са-125", {}) → None → LLM → empty TEST_ASSIST → clarify
  AFTER:  deterministic_rule_decision("Са-125", {}) → TEST_ASSIST{test_name="Са-125"} → test_assist() → CA-125(650₽)
  Gate: ruff clean on all 5 changed files; 213 passed (test_messenger_services) + 62 passed (test_specialty_nlu)
  Regression: сахар/рак/тироксин → lab_code_flag=False (no false matches)

files_changed:
  - messengers_router/policies.py
  - messengers_router/classifier.py
  - messengers_router/services/_prices_helpers.py
  - tests/test_specialty_nlu.py
  - tests/test_messenger_services.py
  - docs/messengers_router_bug_log.md
