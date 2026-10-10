"""Versioned, content-free facts shared by RAG, the CRM and reports.

This contract never contains a question, answer, name, URL, exception or prompt.
RAG facts describe the scenario; the CRM supplies actual routing and delivery.
"""
from __future__ import annotations

import copy
import re

VERSION = 1
TOPICS = frozenset({
    "PRICE", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "TEST_RESULT", "TAX_DOCUMENT",
    "APPOINTMENT", "ADDRESS", "PREPARE", "TEST_ASSIST", "NEWS", "OTHER",
    "URGENT", "COMPLAINT", "MEDICAL_ADVICE",
})
KINDS = frozenset({"tax_link", "results_link", "structured_answer", "none"})
REASONS = frozenset({
    "none", "clarification", "no_data", "not_understood", "unsupported_scenario",
    "patient_requested", "operator_declined", "technical_error", "delivery_unconfirmed",
    "missing_telemetry", "invalid_contract", "inactivity", "manual_operator",
    "existing_operator", "partial_answer",
})
PHASES = frozenset({"answer", "clarification", "operator_offer", "declined", "unanswered", "neutral", "unknown"})
OUTCOMES = ("S", "P", "U", "R", "H", "W", "X")


def unknown_contract(reason="missing_telemetry"):
    return {"version": VERSION, "context": "unknown", "phase": "unknown", "tasks": [],
            "reason": reason if isinstance(reason, str) and reason in REASONS else "missing_telemetry", "operator_event": "none"}


def validate_contract(value):
    """Fail closed and discard every unrecognised field or free-text value."""
    invalid = unknown_contract("invalid_contract")
    if not isinstance(value, dict) or set(value) != {"version", "context", "phase", "tasks", "reason", "operator_event"}:
        return invalid
    if type(value["version"]) is not int or value["version"] != VERSION:
        return invalid
    if any(not isinstance(value[key], str) for key in ("context", "phase", "reason", "operator_event")):
        return invalid
    if value["context"] not in {"new_topic", "continue", "unknown"} or value["phase"] not in PHASES:
        return invalid
    if value["reason"] not in REASONS or value["operator_event"] not in {"none", "offered", "declined", "accepted"}:
        return invalid
    tasks = value["tasks"]
    if not isinstance(tasks, list) or len(tasks) > 8:
        return invalid
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"key", "state", "kind"}:
            return invalid
        key = task["key"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z_]+(?::[1-8])?", key):
            return invalid
        if key.split(":")[0] not in TOPICS or key in seen:
            return invalid
        seen.add(key)
        if not isinstance(task["state"], str) or not isinstance(task["kind"], str):
            return invalid
        if task["state"] not in {"solved", "pending", "unknown"} or task["kind"] not in KINDS:
            return invalid
        if task["state"] == "solved" and task["kind"] == "none":
            return invalid
    if value["phase"] == "answer" and not tasks:
        return invalid
    if value["phase"] == "declined" and value["operator_event"] != "declined":
        return invalid
    return copy.deepcopy(value)


def empty_result():
    return {"outcome": "W", "route": "unknown", "tasks": {}, "declined": False,
            "telemetry_complete": False, "reason": "missing_telemetry"}


def apply_facts(previous, contract, *, delivered: bool, actual_route: str):
    """One outcome per appeal. Only actual CRM routing can establish H."""
    state = copy.deepcopy(previous)
    meta = validate_contract(contract)
    state["declined"] = state["declined"] or meta["operator_event"] == "declined"
    if state["outcome"] == "H":
        return state  # Preserve the first actual handoff and its reason.
    if actual_route == "operator":
        state.update(outcome="H", route="operator", reason=meta["reason"])
        return state
    state["route"] = actual_route if actual_route in {"bot", "unknown"} else "unknown"
    if meta["phase"] == "neutral":
        return state  # A greeting/acknowledgement is not evidence of resolution.
    if meta["phase"] == "unknown" or not delivered:
        state.update(outcome="X", telemetry_complete=False,
                     reason=meta["reason"] if delivered else "delivery_unconfirmed")
        return state
    state["telemetry_complete"] = True
    for task in meta["tasks"]:
        state["tasks"][task["key"]] = {"state": task["state"], "kind": task["kind"]}
    tasks = list(state["tasks"].values())
    solved = sum(task["state"] == "solved" for task in tasks)
    unknown = any(task["state"] == "unknown" for task in tasks)
    if unknown:
        outcome = "X"
        state["telemetry_complete"] = False
    elif tasks and solved == len(tasks):
        outcome = "S"
    elif solved:
        outcome = "P"
    elif meta["phase"] == "declined":
        outcome = "R"
    elif meta["phase"] in {"clarification", "operator_offer"}:
        outcome = "W"
    elif meta["phase"] == "unanswered":
        outcome = "U"
    else:
        outcome = "X"
        state["telemetry_complete"] = False
    state.update(outcome=outcome, reason=meta["reason"])
    return state


def expire_result(previous, inactive_seconds):
    state = copy.deepcopy(previous)
    if inactive_seconds < 6 * 3600 or state["outcome"] != "W":
        return state
    if not state["telemetry_complete"]:
        state.update(outcome="X", reason="missing_telemetry")
    elif any(task["state"] == "solved" for task in state["tasks"].values()):
        state.update(outcome="P", reason="partial_answer")
    else:
        state.update(outcome="R" if state["declined"] else "U", reason="inactivity")
    return state


def balances(results):
    counts = {key: 0 for key in OUTCOMES}
    routes = {key: 0 for key in ("bot", "operator", "unknown")}
    for result in results:
        counts[result["outcome"]] += 1
        routes[result["route"]] += 1
    total = sum(counts.values())
    if sum(routes.values()) != total or routes["operator"] != counts["H"]:
        raise ValueError("reporting_balance_mismatch")
    denominator = sum(counts[key] for key in ("S", "P", "U", "R", "H"))
    return {"total": total, "outcomes": counts, "routes": routes, "completed_known": denominator,
            "success_percent": round(100 * counts["S"] / denominator, 1) if denominator else None,
            "partial_percent": round(100 * counts["P"] / denominator, 1) if denominator else None}
