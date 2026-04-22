"""Routing resolution policy for FreeTalk."""

from __future__ import annotations

from .routing_contract import ClinicalDecision
from .contracts import DialogAct, DialogState
from .dialog_state import merge_missing_slots as _merge_missing_slots_helper
from .tool_planning import select_tool_plan


def resolve_dialog_act(
    *,
    user_message: str,
    decision: ClinicalDecision,
    dialog_state: DialogState,
    active_dialog_state: bool,
    web_search_signal: bool,
    web_search_available: bool,
    medical_regex: bool,
    medical_fallback: bool,
    doctor_followup_hint: bool,
    contextual_followup_hint: bool,
    clinical_min_confidence: float,
    include_meili_tools: bool,
    fallback_contextual_tool_plan: list[str] | None = None,
) -> DialogAct:
    if active_dialog_state:
        if decision.intent == "unknown" and str(dialog_state.intent or "").strip():
            decision.intent = str(dialog_state.intent or "").strip()
        if not decision.tool_plan and dialog_state.tool_plan:
            decision.tool_plan = list(dialog_state.tool_plan)
        if dialog_state.missing_slots:
            decision.missing_slots = _merge_missing_slots_helper(dialog_state.missing_slots, decision.missing_slots)

    route = "general"
    fallback_reason = ""
    if decision.intent != "unknown" or decision.tool_plan or decision.missing_slots:
        route = "clinical"
    elif active_dialog_state:
        route = "clinical"
    elif web_search_signal and web_search_available:
        route = "web"

    fallback_medical_signal = bool(
        medical_regex
        or medical_fallback
        or doctor_followup_hint
        or contextual_followup_hint
        or active_dialog_state
    )

    if route != "clinical" and fallback_medical_signal:
        if decision.source == "llm_router_invalid_json":
            fallback_reason = "invalid_json"
        elif decision.intent == "unknown" and not decision.tool_plan:
            fallback_reason = "unknown_intent"
        elif decision.confidence < float(clinical_min_confidence):
            fallback_reason = "low_confidence"
        else:
            fallback_reason = "heuristic_fallback"
        fallback_tool_plan = select_tool_plan(
            user_message,
            include_meili_tools=include_meili_tools,
        )
        if not fallback_tool_plan and contextual_followup_hint:
            fallback_tool_plan = list(fallback_contextual_tool_plan or [])
        fallback_intent = decision.intent
        if fallback_intent == "unknown" and fallback_tool_plan:
            fallback_intent = infer_intent_from_tool_plan(fallback_tool_plan)
        decision.intent = fallback_intent
        decision.tool_plan = fallback_tool_plan
        route = "clinical"
        decision.source = "heuristic_fallback"

    response_policy = "general_only"
    if route == "clinical":
        response_policy = "tool_only"
    elif route == "web":
        response_policy = "mixed"

    return DialogAct(
        route=route,
        intent=decision.intent,
        entities=dict(decision.entities or {}),
        confidence=decision.confidence,
        missing_slots=list(decision.missing_slots or []),
        clarify_type=str(decision.clarify_type or "").strip(),
        clarify_question=str(decision.clarify_question or "").strip(),
        tool_plan=list(decision.tool_plan or []),
        response_policy=response_policy,
        source=decision.source,
        fallback_reason=fallback_reason,
    )


def infer_intent_from_tool_plan(tool_plan: list[str]) -> str:
    plan = list(tool_plan or [])
    if "doctors_schedule_week" in plan:
        return "doctor_schedule"
    if "doctors_info" in plan:
        return "doctor_info"
    if "price_info" in plan:
        return "price"
    if "test_prepare" in plan:
        return "prepare"
    if "test_assist" in plan:
        return "tests"
    if "test_result_status" in plan:
        return "test_result"
    if "address_info" in plan:
        return "address"
    if "news_info" in plan:
        return "clinic_news"
    if "main_index_info" in plan:
        return "clinic_documents"
    if "service_bundle_info" in plan:
        return "service_info"
    return "unknown"
