"""Canonical key constants for ``Evidence.items``.

Stage 10 of the refactor plan: these constants replace raw string literals at
``evidence.get(...)`` / ``ev.put(...)`` call sites so the universe of evidence
keys becomes discoverable and greppable. This is OPTION B of Stage 10 — a
``TypedDict`` schema (OPTION A) is deferred until ``services_legacy.py`` is
extracted.

Keys are grouped by domain. Dynamic keys produced from tool names
(``f"{tool}_optional_suppressed"``, ``f"{tool}_error"``,
``f"{tool}_optional_error"``) stay as f-strings at the call site — they're not
a fixed vocabulary and have no single canonical name.
"""

from __future__ import annotations

# --- Auth gate -------------------------------------------------------------
AUTH_REQUIRED = "auth_required"
AUTH_MESSAGE = "auth_message"

# --- Handoff gate ----------------------------------------------------------
HANDOFF_REQUIRED = "handoff_required"
HANDOFF_REASON = "handoff_reason"
HANDOFF_MESSAGE = "handoff_message"

# --- Tool payloads (executor writes, response_builder reads) --------------
DOCTORS_INFO = "doctors_info"
DOCTOR_SCHEDULE = "doctor_schedule"
TEST_ASSIST = "test_assist"
PREPARE = "prepare"
TEST_RESULT_STATUS = "test_result_status"
PRICE = "price"
MAIN_INDEX_INFO = "main_index_info"
SERVICE_BUNDLE = "service_bundle"
ADDRESS = "address"
NEWS = "news"
UNKNOWN_TOOL = "unknown_tool"

# --- Orchestrator / policy payloads ---------------------------------------
UNSUPPORTED_CATALOG = "unsupported_catalog"
OPERATOR_OFFER_RESPONSE = "operator_offer_response"
CATALOG_CONFIRM_RESPONSE = "catalog_confirm_response"
CATALOG_HEALTH_RESPONSE = "catalog_health_response"

# --- Rendering sidechannel -------------------------------------------------
ATTACHMENTS = "attachments"
