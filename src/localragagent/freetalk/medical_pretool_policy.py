"""Pre-tool medical state preparation for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate_policy import (
    candidate_confirmation_target,
    candidate_entities_from_entities,
)
from .routing_contract import merge_missing_slots_from_plan
from .contracts import DialogAct, DialogState
from .dialog_state import (
    filter_missing_slots_by_entities,
    merge_missing_slots,
)
from .intent_policy import apply_intent_entity_policy
from .routing_policy import infer_intent_from_tool_plan
from .tool_planning import select_tool_plan


@dataclass(slots=True)
class MedicalPreToolState:
    intent: str
    entities: dict[str, Any]
    tool_plan: list[str]
    candidate_entities: dict[str, Any]
    confirmation_target: str
    state_clarify_type: str
    missing_slots: list[str]
    needs_clarification: bool


def resolve_current_service_name(
    *,
    effective_state: DialogState,
    dialog_act: DialogAct,
    session_memory_entities: dict[str, Any],
) -> str:
    return str(
        effective_state.entities.get("service_name")
        or effective_state.entities.get("test_name")
        or dialog_act.entities.get("service_name")
        or dialog_act.entities.get("test_name")
        or session_memory_entities.get("service_name")
        or session_memory_entities.get("test_name")
        or ""
    ).strip()


def merge_turn_entities(
    *,
    user_message: str,
    intent: str,
    effective_state: DialogState,
    dialog_act: DialogAct,
    grounded_entities: dict[str, Any],
    contextual_entities: dict[str, Any],
) -> dict[str, Any]:
    merged_entities = dict(effective_state.entities or {})
    merged_entities.update(dialog_act.entities)
    merged_entities.update(grounded_entities)
    merged_entities.update(contextual_entities)
    return apply_intent_entity_policy(
        user_message=user_message,
        intent=intent,
        entities=merged_entities,
    )


def resolve_tool_plan(
    *,
    user_message: str,
    intent: str,
    dialog_act: DialogAct,
    effective_state: DialogState,
    include_meili_tools: bool,
) -> tuple[str, list[str]]:
    tool_plan = [tool for tool in (dialog_act.tool_plan or []) if isinstance(tool, str)]
    if not tool_plan:
        tool_plan = [tool for tool in (effective_state.tool_plan or []) if isinstance(tool, str)]
    resolved_intent = str(intent or "").strip()
    if not tool_plan:
        tool_plan = select_tool_plan(user_message, include_meili_tools=include_meili_tools)
        if resolved_intent == "unknown" and tool_plan:
            resolved_intent = infer_intent_from_tool_plan(tool_plan)
    return resolved_intent, tool_plan


def finalize_pretool_state(
    *,
    intent: str,
    confidence: float,
    effective_state: DialogState,
    base_missing_slots: list[str],
    entities: dict[str, Any],
    tool_plan: list[str],
    remembered_doctor: str,
    clinical_min_confidence: float,
) -> MedicalPreToolState:
    candidate_entities = candidate_entities_from_entities(
        entities,
        current=effective_state.candidate_entities,
    )
    confirmation_target = candidate_confirmation_target(
        intent=intent,
        tool_plan=tool_plan,
        entities=entities,
        candidate_entities=candidate_entities,
        current_target=effective_state.confirmation_target,
    )
    missing_from_plan = merge_missing_slots_from_plan(tool_plan, entities, intent=intent)
    missing_slots = merge_missing_slots(
        list(base_missing_slots or []),
        missing_from_plan,
    )
    missing_slots = filter_missing_slots_by_entities(missing_slots, entities)

    resolved_entities = dict(entities or {})
    if {"doctor_name", "specialty"} & set(missing_slots) and str(remembered_doctor or "").strip():
        resolved_entities["doctor_name"] = str(remembered_doctor).strip()
        missing_slots = [slot for slot in missing_slots if slot not in {"doctor_name", "specialty"}]

    needs_clarification = bool(missing_slots)
    if not needs_clarification and confidence < float(clinical_min_confidence) and not tool_plan:
        needs_clarification = True

    return MedicalPreToolState(
        intent=intent,
        entities=resolved_entities,
        tool_plan=list(tool_plan or []),
        candidate_entities=candidate_entities,
        confirmation_target=confirmation_target,
        state_clarify_type=str(effective_state.clarify_type or "").strip(),
        missing_slots=missing_slots,
        needs_clarification=needs_clarification,
    )
def build_clinical_state(
    *,
    intent: str,
    entities: dict[str, Any],
    candidate_entities: dict[str, Any],
    confirmation_target: str,
    missing_slots: list[str],
    clarify_type: str,
    tool_plan: list[str],
    confidence: float,
    clarify_count: int,
    last_tool: str,
    phase: str,
    open_question: str,
) -> DialogState:
    return DialogState(
        route="clinical",
        intent=intent,
        entities=dict(entities or {}),
        candidate_entities=dict(candidate_entities or {}),
        confirmation_target=str(confirmation_target or ""),
        missing_slots=list(missing_slots or []),
        clarify_type=str(clarify_type or ""),
        tool_plan=list(tool_plan or []),
        response_policy="tool_only",
        confidence=float(confidence),
        clarify_count=int(clarify_count),
        last_tool=str(last_tool or ""),
        phase=str(phase or ""),
        open_question=str(open_question or ""),
    )


def rejected_candidate_slots(target: str) -> list[str]:
    if target == "doctor_name":
        return ["doctor_name", "specialty"]
    if target == "service_name":
        return ["service_or_analysis_name"]
    return []
