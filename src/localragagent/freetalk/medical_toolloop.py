"""Medical tool loop execution for FreeTalk."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import Any

from .catalog_policy import catalog_resolution_reply_if_needed
from .routing_contract import (
    clarify_question_for_slots,
    clarify_type_for_slots,
    merge_missing_slots_from_plan,
)
from .contracts import AgentReply
from .medical_pretool_policy import build_clinical_state
from .observability import log_event


def _payload_requests_handoff(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if bool(payload.get("handoff_required")):
        return True
    return bool(str(payload.get("handoff_message") or "").strip())


@dataclass(slots=True)
class MedicalToolLoopContext:
    session_id: str
    user_message: str
    intent: str
    confidence: float
    clarify_count: int
    phase: str
    tool_plan: list[str]
    missing_slots: list[str]
    entities: dict[str, Any]
    candidate_entities: dict[str, Any]


@dataclass(slots=True)
class MedicalToolLoopRuntime:
    adapter: Any
    dispatcher: Any
    max_tool_steps: int
    public_entities: Callable[[dict[str, Any]], dict[str, Any]]
    schedule_payload_stats: Callable[[dict[str, Any]], dict[str, Any]]
    render_tool_reply: Callable[..., Awaitable[str]]
    post_tool_verify: Callable[..., Awaitable[Any]]
    remember_doctor_from_tool_result: Callable[..., Awaitable[None]]
    save_session_entity_memory: Callable[[str, dict[str, Any]], Awaitable[None]]
    save_dialog_state: Callable[[str, Any], Awaitable[None]]
    clear_dialog_state: Callable[[str], Awaitable[None]]
    merge_entities: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    tool_payload_memory_entities: Callable[[str, dict[str, Any]], dict[str, Any]]
    missing_slots_from_tool_payload: Callable[[str, dict[str, Any]], list[str]]
    merge_missing_slots: Callable[[list[str], list[str]], list[str]]
    filter_missing_slots_by_entities: Callable[[list[str], dict[str, Any]], list[str]]


async def execute_medical_tool_loop(
    *,
    context: MedicalToolLoopContext,
    runtime: MedicalToolLoopRuntime,
) -> AgentReply:
    if not context.tool_plan:
        log_event("medical_plan_empty_not_found", level=logging.WARNING, session_id=context.session_id)
        return AgentReply(
            text=(
                "Не удалось однозначно определить медицинский запрос. "
                "Уточните врача, услугу, анализ или тип вопроса (цена/подготовка/расписание)."
            ),
            source="clinic_data",
        )

    catalog_resolution_reply = catalog_resolution_reply_if_needed(
        tool_plan=context.tool_plan,
        entities=context.entities,
        fallback_only=False,
    )
    if catalog_resolution_reply is not None:
        log_event(
            "medical_catalog_resolution_reply",
            level=logging.WARNING,
            session_id=context.session_id,
            tool_plan=",".join(context.tool_plan),
            entities=runtime.public_entities(context.entities),
        )
        await runtime.clear_dialog_state(context.session_id)
        return catalog_resolution_reply

    tool_entities = runtime.public_entities(context.entities)
    for tool_name in context.tool_plan[: runtime.max_tool_steps]:
        log_event("medical_tool_call_start", session_id=context.session_id, tool_name=tool_name)
        prepared_call = runtime.adapter.prepare_tool_call(
            tool_name=tool_name,
            user_message=context.user_message,
            entities=tool_entities,
        )
        result = await runtime.dispatcher.call(
            prepared_call.tool_name,
            prepared_call.backend_query,
            entities=prepared_call.backend_entities,
        )
        if result.error:
            log_event(
                "medical_tool_call_error",
                level=logging.WARNING,
                session_id=context.session_id,
                tool_name=tool_name,
                error=result.error[:200],
            )
            continue
        if not result.found:
            log_event(
                "medical_tool_call_not_found",
                session_id=context.session_id,
                tool_name=tool_name,
                payload_note=str(result.payload.get("note") or "")[:160],
            )
            continue

        adapted_result = runtime.adapter.normalize_tool_payload(
            tool_name=result.tool_name,
            payload=result.payload,
            prepared_call=prepared_call,
        )
        result.payload = dict(adapted_result.ft_payload or {})

        if result.tool_name == "doctors_schedule_week":
            stats = runtime.schedule_payload_stats(result.payload)
            log_event(
                "medical_schedule_payload_stats",
                session_id=context.session_id,
                doctors_count=stats["doctors_count"],
                regions_count=stats["regions_count"],
                days_count=stats["days_count"],
                slots_count=stats["slots_count"],
                schedule_unavailable_reason=stats["schedule_unavailable_reason"],
            )

        answer = await runtime.render_tool_reply(
            user_message=context.user_message,
            tool_name=result.tool_name,
            tool_payload=result.payload,
        )
        log_event(
            "medical_tool_success",
            session_id=context.session_id,
            tool_name=result.tool_name,
            payload_keys=",".join(sorted(str(k) for k in result.payload.keys())),
            answer_chars=len(answer),
        )
        if _payload_requests_handoff(result.payload):
            return AgentReply(
                text=answer,
                source="clinic_data",
                tool_name=result.tool_name,
                tool_payload=result.payload,
                handoff=True,
            )
        verification = await runtime.post_tool_verify(
            user_message=context.user_message,
            intent=context.intent,
            tool_name=result.tool_name,
            drafted_answer=answer,
            tool_payload=result.payload,
            missing_slots=context.missing_slots,
        )
        log_event(
            "post_tool_verifier_decision",
            session_id=context.session_id,
            tool_name=result.tool_name,
            answer_policy=verification.answer_policy,
            enough_data=verification.enough_data,
            should_clarify=verification.should_clarify,
            source=verification.source,
        )
        await runtime.remember_doctor_from_tool_result(
            session_id=context.session_id,
            tool_name=result.tool_name,
            payload=result.payload,
        )
        await runtime.save_session_entity_memory(
            context.session_id,
            runtime.merge_entities(
                context.entities,
                runtime.tool_payload_memory_entities(result.tool_name, result.payload),
            ),
        )
        if verification.answer_policy == "clarify":
            clarify_type = str(verification.clarify_type or "").strip()
            if not clarify_type:
                clarify_type = clarify_type_for_slots(context.intent, context.missing_slots)
            clarify_text = str(verification.clarify_question or "").strip()
            if not clarify_text:
                clarify_text = clarify_question_for_slots(context.intent, context.missing_slots)
            next_missing_slots = runtime.merge_missing_slots(
                context.missing_slots,
                runtime.missing_slots_from_tool_payload(result.tool_name, result.payload),
            )
            next_missing_slots = runtime.filter_missing_slots_by_entities(next_missing_slots, context.entities)
            await runtime.save_dialog_state(
                context.session_id,
                build_clinical_state(
                    intent=context.intent,
                    entities=runtime.public_entities(context.entities),
                    candidate_entities=context.candidate_entities,
                    confirmation_target="",
                    missing_slots=next_missing_slots,
                    clarify_type=clarify_type,
                    tool_plan=context.tool_plan,
                    confidence=context.confidence,
                    clarify_count=max(1, int(context.clarify_count or 0)),
                    last_tool=result.tool_name,
                    phase="post_tool",
                    open_question=clarify_text,
                ),
            )
            return AgentReply(
                text=clarify_text,
                source="clinic_data",
                tool_name=result.tool_name,
                tool_payload=result.payload,
            )
        if verification.answer_policy == "not_found":
            await runtime.clear_dialog_state(context.session_id)
            return AgentReply(
                text="В данных клиники по вашему запросу ничего не найдено.",
                source="clinic_data",
                tool_name=result.tool_name,
                tool_payload=result.payload,
            )
        await runtime.clear_dialog_state(context.session_id)
        return AgentReply(
            text=answer,
            source="clinic_data",
            tool_name=result.tool_name,
            tool_payload=result.payload,
        )

    log_event(
        "medical_all_tools_exhausted",
        level=logging.WARNING,
        session_id=context.session_id,
        tool_plan=",".join(context.tool_plan[: runtime.max_tool_steps]),
    )
    exhausted_reply = catalog_resolution_reply_if_needed(
        tool_plan=context.tool_plan,
        entities=context.entities,
        fallback_only=True,
    )
    if exhausted_reply is not None:
        await runtime.clear_dialog_state(context.session_id)
        return exhausted_reply

    retry_after_not_found = bool(
        context.tool_plan
        and str(context.phase or "").strip().lower() != "post_not_found"
    )
    if retry_after_not_found:
        missing_slots = list(context.missing_slots or [])
        if not missing_slots:
            missing_slots = merge_missing_slots_from_plan(
                context.tool_plan,
                context.entities,
                intent=context.intent,
            )
            missing_slots = runtime.filter_missing_slots_by_entities(missing_slots, context.entities)
        clarify_type = clarify_type_for_slots(context.intent, missing_slots)
        clarify_text = clarify_question_for_slots(context.intent, missing_slots)
        await runtime.save_dialog_state(
            context.session_id,
            build_clinical_state(
                intent=context.intent,
                entities=runtime.public_entities(context.entities),
                candidate_entities=context.candidate_entities,
                confirmation_target="",
                missing_slots=missing_slots,
                clarify_type=clarify_type,
                tool_plan=context.tool_plan,
                confidence=context.confidence,
                clarify_count=max(1, int(context.clarify_count or 0)),
                last_tool="",
                phase="post_not_found",
                open_question=clarify_text,
            ),
        )
        return AgentReply(
            text=f"В данных клиники по текущему запросу ничего не найдено. {clarify_text}",
            source="clinic_data",
        )

    await runtime.clear_dialog_state(context.session_id)
    return AgentReply(text="В данных клиники по вашему запросу ничего не найдено.", source="clinic_data")
