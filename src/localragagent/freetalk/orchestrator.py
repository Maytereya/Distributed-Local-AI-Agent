"""Top-level orchestration helpers for FreeTalk."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from .adapter import FreeTalkAdapter
from .contracts import AgentReply, DialogAct, DialogState, SessionContext
from .medical_toolloop import (
    MedicalToolLoopContext,
    MedicalToolLoopRuntime,
    execute_medical_tool_loop as _execute_medical_tool_loop_helper,
)
from .observability import log_event
from .routing_prompting import build_general_prompt
from .tool_dispatcher import ToolDispatcher
from .tool_planning import is_about_agent_query, should_use_web_search


@dataclass(slots=True)
class GuardRuntimeConfig:
    state_key: str
    none_state: str
    one_more_state: str
    awaiting_immediate_state: str
    awaiting_final_state: str


def _is_handoff_reply(reply: AgentReply) -> bool:
    if bool(getattr(reply, "handoff", False)):
        return True
    payload = reply.tool_payload if isinstance(reply.tool_payload, dict) else {}
    if bool(payload.get("handoff_required")):
        return True
    return bool(str(payload.get("handoff_message") or "").strip())


async def chat(
    agent: Any,
    message: str,
    session_id: str,
    *,
    guard: GuardRuntimeConfig,
    merge_text: Any,
) -> AgentReply:
    sid = str(session_id or "").strip()
    if not sid:
        sid = agent._new_session_id()
    agent._last_prompt_eval_count = 0
    user_message = str(message or "").strip()
    if not user_message:
        return AgentReply(text="Напишите сообщение текстом.", source="general_knowledge")

    context = await agent.memory.load_context(
        sid,
        history_tail_turns=agent.config.history_tail_turns,
    )
    guard_state = await agent.memory.get_meta_str(sid, guard.state_key, guard.none_state)

    guard_reply = await agent._handle_guard_decision(
        session_id=sid,
        guard_state=guard_state,
        user_message=user_message,
        context=context,
    )
    if guard_reply is not None:
        log_event(
            "context_guard_decision_handled",
            session_id=sid,
            state=guard_state or "none",
            next_session_id=guard_reply.next_session_id or "",
        )
        if not guard_reply.next_session_id:
            guard_fragments = agent._source_fragments_for_reply(guard_reply)
            guard_reply.source_fragments = guard_fragments
            agent._log_source_trace(session_id=sid, reply=guard_reply)
            await agent.memory.append_exchange(
                sid,
                user_text=user_message,
                assistant_text=guard_reply.text,
                source=guard_reply.source,
                source_fragments=guard_reply.source_fragments,
            )
            await agent._maybe_compact(sid)
        return guard_reply

    dialog_state = await agent._load_dialog_state(sid)
    remembered_doctor = await agent.memory.get_meta_str(sid, agent._last_doctor_name_key(), "")
    session_memory_entities = await agent._load_session_entity_memory(sid)
    appointment_precheck = agent._appointment_precheck(
        user_message=user_message,
        dialog_state=dialog_state,
        memory_entities=session_memory_entities,
    )
    if getattr(appointment_precheck, "handled", False):
        if getattr(appointment_precheck, "clear_state", False):
            await agent._clear_dialog_state(sid)
        clear_memory_keys = getattr(appointment_precheck, "clear_memory_keys", None)
        if isinstance(clear_memory_keys, (list, tuple)) and clear_memory_keys:
            await agent._clear_session_entity_memory_keys(sid, clear_memory_keys)
        elif getattr(appointment_precheck, "next_state", None) is not None:
            await agent._save_dialog_state(sid, appointment_precheck.next_state)
            await agent._save_session_entity_memory(sid, dict(appointment_precheck.next_state.entities or {}))
        save_memory_entities = getattr(appointment_precheck, "save_memory_entities", None)
        if isinstance(save_memory_entities, dict) and save_memory_entities:
            await agent._save_session_entity_memory(sid, save_memory_entities)
        reply = AgentReply(
            text=str(getattr(appointment_precheck, "reply_text", "") or "").strip(),
            source="clinic_data",
            handoff=bool(getattr(appointment_precheck, "handoff", False)),
        )
    else:
        dialog_act = await agent._build_dialog_act(
            user_message=user_message,
            context=context,
            dialog_state=dialog_state,
            remembered_doctor=remembered_doctor,
        )
        log_event(
            "route_selected",
            session_id=sid,
            route=dialog_act.route,
            intent=dialog_act.intent,
            source=dialog_act.source,
            fallback_reason=dialog_act.fallback_reason or "",
        )
        reply = await agent._execute_dialog_act(
            user_message=user_message,
            context=context,
            dialog_act=dialog_act,
            dialog_state=dialog_state,
        )

    if guard_state == guard.one_more_state and not reply.next_session_id:
        await agent.memory.set_meta_str(sid, guard.state_key, guard.awaiting_final_state)
        reply.text = merge_text(reply.text, agent._guard_final_choice_prompt())
    elif guard_state == guard.none_state and not reply.next_session_id:
        if agent._is_context_guard_needed(context=context, user_message=user_message):
            est_tokens = agent._estimate_context_tokens(context=context, user_message=user_message)
            await agent.memory.set_meta_str(sid, guard.state_key, guard.awaiting_immediate_state)
            reply.text = merge_text(reply.text, agent._guard_near_limit_prompt())
            log_event(
                "context_guard_triggered",
                level=logging.WARNING,
                session_id=sid,
                estimated_tokens=est_tokens,
                prompt_eval_count=agent._last_prompt_eval_count,
                context_window=agent.config.context_window_tokens,
            )

    source_fragments = agent._source_fragments_for_reply(reply)
    reply.source_fragments = source_fragments
    agent._log_source_trace(session_id=sid, reply=reply)
    if _is_handoff_reply(reply):
        await agent.memory.clear_session(sid)
        if not str(reply.next_session_id or "").strip():
            reply.next_session_id = agent._new_session_id()
        log_event(
            "freetalk_handoff_session_reset",
            level=logging.WARNING,
            session_id=sid,
            next_session_id=reply.next_session_id,
            tool_name=reply.tool_name or "",
            source=reply.source or "",
        )
        return reply
    if reply.next_session_id:
        return reply
    await agent.memory.append_exchange(
        sid,
        user_text=user_message,
        assistant_text=reply.text,
        source=reply.source,
        source_fragments=source_fragments,
    )
    await agent._maybe_compact(sid)
    return reply


async def execute_dialog_act(
    agent: Any,
    *,
    user_message: str,
    context: SessionContext,
    dialog_act: DialogAct,
    dialog_state: DialogState,
) -> AgentReply:
    route = str(dialog_act.route or "").strip().lower()
    if route == "clinical":
        return await agent._medical_reply(
            user_message,
            context,
            dialog_act=dialog_act,
            dialog_state=dialog_state,
        )
    if route == "web":
        await agent._clear_dialog_state(context.session_id)
        return await agent._web_reply(user_message, context)
    await agent._clear_dialog_state(context.session_id)
    return await agent._general_reply(user_message, context)


async def web_reply(agent: Any, user_message: str, context: SessionContext) -> AgentReply:
    if not agent.web_search:
        return await agent._general_reply(user_message, context)
    log_event("general_web_search_triggered", session_id=context.session_id, message=user_message[:140])
    web_payload = await agent.web_search.search(user_message, entities={})
    web_results = web_payload.get("results") if isinstance(web_payload, dict) else []
    if isinstance(web_results, list) and web_results:
        answer = await agent._render_tool_reply(
            user_message=user_message,
            tool_name="web_search",
            tool_payload=web_payload,
        )
        if _is_non_empty_text(answer):
            log_event(
                "general_web_search_success",
                session_id=context.session_id,
                results_count=len(web_results),
            )
            return AgentReply(
                text=answer,
                source="mixed",
                tool_name="web_search",
                tool_payload=web_payload,
            )
    note = str(web_payload.get("note") or "") if isinstance(web_payload, dict) else ""
    if "source unavailable" in note.lower():
        log_event(
            "web_search_unavailable_fallback",
            level=logging.WARNING,
            note=note,
        )
        return AgentReply(
            text=(
                "Интернет-поиск сейчас недоступен. "
                "Повторите запрос позже или задайте вопрос без требования актуальных данных."
            ),
            source="general_knowledge",
        )
    return AgentReply(
        text="В интернет-поиске по этому запросу не найдено релевантных результатов.",
        source="general_knowledge",
        tool_name="web_search",
        tool_payload=web_payload if isinstance(web_payload, dict) else {},
    )


async def general_reply(agent: Any, user_message: str, context: SessionContext) -> AgentReply:
    if is_about_agent_query(user_message):
        log_event("general_about_agent", session_id=context.session_id)
        return AgentReply(text=agent._capabilities_brief(), source="general_knowledge")

    if agent.web_search and should_use_web_search(user_message):
        log_event("general_web_search_triggered", session_id=context.session_id, message=user_message[:140])
        web_payload = await agent.web_search.search(user_message, entities={})
        web_results = web_payload.get("results") if isinstance(web_payload, dict) else []
        if isinstance(web_results, list) and web_results:
            answer = await agent._render_tool_reply(
                user_message=user_message,
                tool_name="web_search",
                tool_payload=web_payload,
            )
            if _is_non_empty_text(answer):
                log_event(
                    "general_web_search_success",
                    session_id=context.session_id,
                    results_count=len(web_results),
                )
                return AgentReply(
                    text=answer,
                    source="mixed",
                    tool_name="web_search",
                    tool_payload=web_payload,
                )
        note = str(web_payload.get("note") or "") if isinstance(web_payload, dict) else ""
        if "source unavailable" in note.lower():
            log_event(
                "web_search_unavailable_fallback",
                level=logging.WARNING,
                note=note,
            )
            return AgentReply(
                text=(
                    "Интернет-поиск сейчас недоступен. "
                    "Повторите запрос позже или задайте вопрос без требования актуальных данных."
                ),
                source="general_knowledge",
            )

    prompt = build_general_prompt(
        system_prompt=agent.system_prompt,
        summary=context.summary,
        turns=context.turns,
        user_message=user_message,
    )
    text = await agent._llm_text(prompt)
    if not _is_non_empty_text(text):
        text = "Уточните, пожалуйста, вопрос. Если это медицинская тема клиники, я запрошу данные через инструменты."
        log_event("general_llm_empty_fallback", level=logging.WARNING, session_id=context.session_id)
    else:
        log_event(
            "general_llm_answered",
            session_id=context.session_id,
            answer_chars=len(text),
        )
    return AgentReply(text=text, source="general_knowledge")


async def medical_reply(
    agent: Any,
    user_message: str,
    context: SessionContext,
    *,
    dialog_act: DialogAct | None = None,
    dialog_state: DialogState | None = None,
) -> AgentReply:
    health = await agent.services.get_catalog_health()
    if isinstance(health, dict) and not bool(health.get("ok", True)):
        log_event(
            "catalog_health_not_ok",
            level=logging.WARNING,
            details=str(health)[:300],
        )
        return AgentReply(
            text="Сейчас источник данных клиники недоступен. Попробуйте повторить запрос позже.",
            source="clinic_data",
            tool_name="get_catalog_health",
            tool_payload=health,
        )

    effective_state = agent._copy_dialog_state(dialog_state)
    session_memory_entities = await agent._load_session_entity_memory(context.session_id)
    if dialog_act is None:
        remembered_doctor = await agent.memory.get_meta_str(context.session_id, agent._last_doctor_name_key(), "")
        if not agent._dialog_state_is_active(effective_state):
            effective_state = await agent._load_dialog_state(context.session_id)
        dialog_act = await agent._build_dialog_act(
            user_message=user_message,
            context=context,
            dialog_state=effective_state,
            remembered_doctor=remembered_doctor,
        )
        if str(dialog_act.route or "").strip().lower() != "clinical":
            return await agent._execute_dialog_act(
                user_message=user_message,
                context=context,
                dialog_act=dialog_act,
                dialog_state=effective_state,
            )

    contextual_entities = agent._extract_contextual_entities(user_message)
    current_service_name = agent._resolve_current_service_name(
        effective_state=effective_state,
        dialog_act=dialog_act,
        session_memory_entities=session_memory_entities,
    )
    grounded_entities = await agent._ground_entities(
        user_message,
        intent_hint=dialog_act.intent,
        current_service_name=current_service_name,
    )
    entities = agent._merge_turn_entities(
        user_message=user_message,
        intent=dialog_act.intent,
        effective_state=effective_state,
        dialog_act=dialog_act,
        grounded_entities=grounded_entities,
        contextual_entities=contextual_entities,
    )
    dialog_act.intent, tool_plan = agent._resolve_tool_plan(
        user_message=user_message,
        intent=dialog_act.intent,
        dialog_act=dialog_act,
        effective_state=effective_state,
    )
    adapter = agent.adapter or FreeTalkAdapter()
    entities = await agent._enrich_entities_from_session_memory(
        session_id=context.session_id,
        user_message=user_message,
        tool_plan=tool_plan,
        entities=entities,
        dialog_state=effective_state,
        memory_entities=session_memory_entities,
        contextual_entities=contextual_entities,
    )
    remembered_doctor = await agent.memory.get_meta_str(context.session_id, agent._last_doctor_name_key(), "")
    pretool = agent._finalize_pretool_state(
        intent=dialog_act.intent,
        confidence=dialog_act.confidence,
        effective_state=effective_state,
        base_missing_slots=agent._merge_missing_slots(effective_state.missing_slots, dialog_act.missing_slots),
        entities=entities,
        tool_plan=tool_plan,
        remembered_doctor=remembered_doctor,
    )
    entities = pretool.entities
    candidate_entities = pretool.candidate_entities
    confirmation_target = pretool.confirmation_target
    missing_slots = pretool.missing_slots
    state_clarify_type = pretool.state_clarify_type
    if str(effective_state.phase or "").strip().lower() == "confirm_candidate" and confirmation_target:
        confirm_decision = agent._yes_no_decision(user_message)
        if confirm_decision == "yes":
            entities = agent._promote_confirmed_candidate(
                entities=entities,
                candidate_entities=candidate_entities,
                target=confirmation_target,
            )
            candidate_entities.pop(confirmation_target, None)
            effective_state.phase = ""
            effective_state.open_question = ""
            effective_state.confirmation_target = ""
            pretool = agent._finalize_pretool_state(
                intent=dialog_act.intent,
                confidence=dialog_act.confidence,
                effective_state=effective_state,
                base_missing_slots=agent._merge_missing_slots(effective_state.missing_slots, dialog_act.missing_slots),
                entities=entities,
                tool_plan=tool_plan,
                remembered_doctor=remembered_doctor,
            )
            entities = pretool.entities
            candidate_entities = pretool.candidate_entities
            confirmation_target = pretool.confirmation_target
            missing_slots = pretool.missing_slots
            state_clarify_type = pretool.state_clarify_type
        elif confirm_decision == "no":
            candidate_entities.pop(confirmation_target, None)
            clarify_slots = agent._rejected_candidate_slots(confirmation_target)
            clarify_text = agent._candidate_rejected_question(confirmation_target)
            await agent._save_dialog_state(
                context.session_id,
                agent._build_clinical_state(
                    intent=dialog_act.intent,
                    entities=agent._public_entities(entities),
                    candidate_entities=candidate_entities,
                    confirmation_target="",
                    missing_slots=agent._merge_missing_slots([], clarify_slots),
                    clarify_type="identify",
                    tool_plan=tool_plan,
                    confidence=dialog_act.confidence,
                    clarify_count=max(1, int(effective_state.clarify_count or 0)),
                    last_tool=effective_state.last_tool,
                    phase="collecting",
                    open_question=clarify_text,
                ),
            )
            await agent._save_session_entity_memory(context.session_id, entities)
            return AgentReply(text=clarify_text, source="clinic_data")

    if confirmation_target:
        confirm_text = agent._candidate_confirmation_question(
            target=confirmation_target,
            candidate_value=str(candidate_entities.get(confirmation_target) or "").strip(),
        )
        if confirm_text:
            next_attempt = int(effective_state.clarify_count or 0) + 1
            await agent._save_dialog_state(
                context.session_id,
                agent._build_clinical_state(
                    intent=dialog_act.intent,
                    entities=agent._public_entities(entities),
                    candidate_entities=candidate_entities,
                    confirmation_target=confirmation_target,
                    missing_slots=list(effective_state.missing_slots or dialog_act.missing_slots or []),
                    clarify_type="confirm_candidate",
                    tool_plan=tool_plan,
                    confidence=dialog_act.confidence,
                    clarify_count=next_attempt,
                    last_tool=effective_state.last_tool,
                    phase="confirm_candidate",
                    open_question=confirm_text,
                ),
            )
            await agent._save_session_entity_memory(context.session_id, entities)
            log_event(
                "medical_candidate_confirmation_requested",
                session_id=context.session_id,
                intent=dialog_act.intent,
                target=confirmation_target,
                candidate=str(candidate_entities.get(confirmation_target) or "")[:120],
            )
            return AgentReply(text=confirm_text, source="clinic_data")

    log_event(
        "medical_plan_selected",
        session_id=context.session_id,
        tool_plan=",".join(tool_plan),
        entities=agent._public_entities(entities),
        router_intent=dialog_act.intent,
        router_confidence=dialog_act.confidence,
        router_source=dialog_act.source,
        missing_slots=",".join(missing_slots),
        message=user_message[:160],
    )
    if pretool.needs_clarification:
        clarify_type = str(dialog_act.clarify_type or state_clarify_type or "").strip()
        if not clarify_type:
            clarify_type = agent._clarify_type_for_slots(dialog_act.intent, missing_slots)
        clarify_text = str(dialog_act.clarify_question or "").strip()
        if not clarify_text:
            clarify_text = str(effective_state.open_question or "").strip()
        if not clarify_text:
            clarify_text = agent._clarify_question_for_slots(dialog_act.intent, missing_slots)
        next_attempt = int(effective_state.clarify_count or 0) + 1
        await agent._save_dialog_state(
            context.session_id,
            agent._build_clinical_state(
                intent=dialog_act.intent,
                entities=agent._public_entities(entities),
                candidate_entities=candidate_entities,
                confirmation_target="",
                missing_slots=missing_slots,
                clarify_type=clarify_type,
                tool_plan=tool_plan,
                confidence=dialog_act.confidence,
                clarify_count=next_attempt,
                last_tool=effective_state.last_tool,
                phase="collecting",
                open_question=clarify_text,
            ),
        )
        await agent._save_session_entity_memory(context.session_id, entities)
        log_event(
            "medical_clarification_requested",
            level=logging.WARNING,
            session_id=context.session_id,
            intent=dialog_act.intent,
            attempts=next_attempt,
            missing_slots=",".join(missing_slots),
            question=clarify_text[:180],
        )
        return AgentReply(text=clarify_text, source="clinic_data")

    await agent._save_session_entity_memory(context.session_id, entities)
    await agent._clear_dialog_state(context.session_id)
    dispatcher = ToolDispatcher(agent.services.tool_handlers(include_meili_tools=agent.config.include_meili_tools))
    return await _execute_medical_tool_loop_helper(
        context=MedicalToolLoopContext(
            session_id=context.session_id,
            user_message=user_message,
            intent=dialog_act.intent,
            confidence=dialog_act.confidence,
            clarify_count=max(1, int(effective_state.clarify_count or 0)),
            phase=str(effective_state.phase or ""),
            tool_plan=tool_plan,
            missing_slots=missing_slots,
            entities=entities,
            candidate_entities=candidate_entities,
        ),
        runtime=MedicalToolLoopRuntime(
            adapter=adapter,
            dispatcher=dispatcher,
            max_tool_steps=agent.config.max_tool_steps,
            public_entities=agent._public_entities,
            schedule_payload_stats=agent._schedule_payload_stats,
            render_tool_reply=agent._render_tool_reply,
            post_tool_verify=agent._post_tool_verify,
            remember_doctor_from_tool_result=agent._remember_doctor_from_tool_result,
            save_session_entity_memory=agent._save_session_entity_memory,
            save_dialog_state=agent._save_dialog_state,
            clear_dialog_state=agent._clear_dialog_state,
            merge_entities=agent._merge_entities,
            tool_payload_memory_entities=agent._tool_payload_memory_entities,
            missing_slots_from_tool_payload=agent._missing_slots_from_tool_payload,
            merge_missing_slots=agent._merge_missing_slots,
            filter_missing_slots_by_entities=agent._filter_missing_slots_by_entities,
        ),
    )


def _is_non_empty_text(value: str) -> bool:
    return bool(str(value or "").strip())
