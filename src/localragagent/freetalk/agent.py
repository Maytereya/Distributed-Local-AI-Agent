"""FreeTalk agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import re
from typing import Any
import uuid

from localragagent.ports import freetalk_llm_port
from localragagent.ports.freetalk_services_port import LegacyServicesPort
from localragagent.ports.freetalk_web_search_port import WebSearchPort

from .clinical_router import (
    ClinicalDecision,
    clarify_question_for_slots,
    merge_missing_slots_from_plan,
    parse_clinical_decision,
)
from .config import FreeTalkConfig
from .contracts import AgentReply, SessionContext
from .memory_persist import PersistentSummaryStore
from .memory_redis import RedisMemoryStore
from .observability import log_event
from .prompts import (
    build_clinical_router_prompt,
    build_general_prompt,
    build_summary_prompt,
    build_tool_result_prompt,
    load_system_prompt,
)
from .tool_dispatcher import ToolDispatcher
from .tool_registry import is_about_agent_query, is_medical_query, select_tool_plan, should_use_web_search


def _is_non_empty_text(value: str) -> bool:
    return bool(str(value or "").strip())


def _top_list(values: list[Any], limit: int = 5) -> list[Any]:
    return list(values[: max(1, limit)])


_CTX_GUARD_STATE_KEY = "ctx_guard_state"
_LAST_DOCTOR_NAME_KEY = "last_doctor_name"
_CLINICAL_PENDING_STATE_KEY = "clinical_pending_state"
_CTX_GUARD_NONE = ""
_CTX_GUARD_AWAITING_IMMEDIATE = "awaiting_immediate"
_CTX_GUARD_ONE_MORE = "one_more"
_CTX_GUARD_AWAITING_FINAL = "awaiting_final"
_CLINICAL_MIN_CONFIDENCE = 0.58
_CLINICAL_MAX_CLARIFY_RETRIES = 2

_YES_RE = re.compile(r"^\s*(да|угу|ага|yes|yep|ok|ок|конечно)\s*[.!?]?\s*$", re.I)
_NO_RE = re.compile(r"^\s*(нет|неа|no|nope|not now|пока нет)\s*[.!?]?\s*$", re.I)
_DOCTOR_ANAPHORA_RE = re.compile(r"\b(его|него|нему|ним|он|у\s+него|у\s+него\s+же|у\s+неё|ее|её|она)\b", re.I)
_DOCTOR_FOLLOWUP_RE = re.compile(
    r"\b(доктор|врач|расписан|график|при(е|ё)м|слот|окн|чем\s+занима|о\s+нем|о\s+враче|инфо)\b",
    re.I,
)
_TOKEN_RE = re.compile(r"[a-zа-яё0-9\-]+", re.I)
_DOCTOR_NON_PERSON_TOKENS = {
    "врач",
    "доктор",
    "специалист",
    "терапевт",
    "кардиолог",
    "невролог",
    "гастроэнтеролог",
    "эндокринолог",
    "гинеколог",
    "уролог",
    "онколог",
    "педиатр",
    "хирург",
    "дерматолог",
    "аллерголог",
    "иммунолог",
    "офтальмолог",
    "лор",
    "отоларинголог",
}
_SERVICE_NON_SPECIFIC_TOKENS = {
    "услуга",
    "услуги",
    "процедура",
    "процедуры",
    "анализ",
    "анализы",
    "исследование",
    "исследования",
    "цена",
    "стоимость",
    "сколько",
    "стоит",
    "прайс",
    "подготовка",
    "врач",
    "доктор",
    "клиника",
}


def _merge_text(primary: str, appendix: str) -> str:
    a = str(primary or "").strip()
    b = str(appendix or "").strip()
    if a and b:
        return f"{a}\n\n{b}"
    return a or b


def _new_session_id() -> str:
    return f"gr_ft_{uuid.uuid4().hex[:12]}"


def _capabilities_brief() -> str:
    return (
        "Я могу:\n"
        "- Общаться свободно на любые темы.\n"
        "- По вопросам клиники (врачи, услуги, подготовка, цены, анализы, расписание, филиалы) искать ответ через инструменты данных.\n"
        "- По запросу искать информацию в интернете.\n\n"
        "Примеры:\n"
        "- \"Сколько стоит МРТ поясницы?\"\n"
        "- \"Как подготовиться к общему анализу крови?\"\n"
        "- \"Поищи в интернете последние новости по теме ...\""
    )


def _normalise_match_text(value: str) -> str:
    text = str(value or "").lower().strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_user_fragment(value: str, *, max_len: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


@dataclass(slots=True)
class FreeTalkAgent:
    config: FreeTalkConfig
    services: LegacyServicesPort
    memory: RedisMemoryStore
    persist: PersistentSummaryStore
    system_prompt: str
    web_search: WebSearchPort | None = None
    _last_prompt_eval_count: int = 0

    @classmethod
    def build(
        cls,
        *,
        config: FreeTalkConfig,
        services: LegacyServicesPort,
        memory: RedisMemoryStore,
        persist: PersistentSummaryStore,
        web_search: WebSearchPort | None = None,
    ) -> "FreeTalkAgent":
        prompt = load_system_prompt(config.system_prompt_path)
        return cls(
            config=config,
            services=services,
            memory=memory,
            persist=persist,
            system_prompt=prompt,
            web_search=web_search,
        )

    async def chat(self, message: str, session_id: str) -> AgentReply:
        sid = str(session_id or "").strip() or _new_session_id()
        self._last_prompt_eval_count = 0
        user_message = str(message or "").strip()
        if not user_message:
            return AgentReply(text="Напишите сообщение текстом.", source="general_knowledge")

        context = await self.memory.load_context(
            sid,
            history_tail_turns=self.config.history_tail_turns,
        )
        remembered_doctor = await self.memory.get_meta_str(sid, _LAST_DOCTOR_NAME_KEY, "")
        doctor_followup_hint = self._looks_like_doctor_followup_message(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
        )
        medical_regex = is_medical_query(user_message)
        medical_fallback = self._looks_like_clinic_data_query(user_message)
        medical_intent = bool(medical_regex or medical_fallback or doctor_followup_hint)
        web_search_signal = should_use_web_search(user_message, allow_for_medical=True)
        log_event(
            "route_intent_evaluated",
            session_id=sid,
            medical_regex=medical_regex,
            medical_fallback=medical_fallback,
            doctor_followup_hint=doctor_followup_hint,
            remembered_doctor=bool(str(remembered_doctor or "").strip()),
            medical_intent=medical_intent,
            web_search_signal=web_search_signal,
            about_agent=is_about_agent_query(user_message),
            message=user_message[:180],
        )
        guard_state = await self.memory.get_meta_str(sid, _CTX_GUARD_STATE_KEY, _CTX_GUARD_NONE)

        guard_reply = await self._handle_guard_decision(
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
                await self.memory.append_exchange(
                    sid,
                    user_text=user_message,
                    assistant_text=guard_reply.text,
                    source=guard_reply.source,
                )
                await self._maybe_compact(sid)
            return guard_reply

        if medical_intent:
            log_event("route_selected", session_id=sid, route="medical")
            reply = await self._medical_reply(user_message, context)
        else:
            log_event("route_selected", session_id=sid, route="general")
            await self._clear_clinical_pending_state(sid)
            reply = await self._general_reply(user_message, context)

        if guard_state == _CTX_GUARD_ONE_MORE and not reply.next_session_id:
            await self.memory.set_meta_str(sid, _CTX_GUARD_STATE_KEY, _CTX_GUARD_AWAITING_FINAL)
            reply.text = _merge_text(reply.text, self._guard_final_choice_prompt())
        elif guard_state == _CTX_GUARD_NONE and not reply.next_session_id:
            if self._is_context_guard_needed(context=context, user_message=user_message):
                est_tokens = self._estimate_context_tokens(context=context, user_message=user_message)
                await self.memory.set_meta_str(sid, _CTX_GUARD_STATE_KEY, _CTX_GUARD_AWAITING_IMMEDIATE)
                reply.text = _merge_text(reply.text, self._guard_near_limit_prompt())
                log_event(
                    "context_guard_triggered",
                    level=logging.WARNING,
                    session_id=sid,
                    estimated_tokens=est_tokens,
                    prompt_eval_count=self._last_prompt_eval_count,
                    context_window=self.config.context_window_tokens,
                )

        await self.memory.append_exchange(
            sid,
            user_text=user_message,
            assistant_text=reply.text,
            source=reply.source,
        )
        await self._maybe_compact(sid)
        return reply

    async def _general_reply(self, user_message: str, context: SessionContext) -> AgentReply:
        if is_about_agent_query(user_message):
            log_event("general_about_agent", session_id=context.session_id)
            return AgentReply(text=_capabilities_brief(), source="general_knowledge")

        if self.web_search and should_use_web_search(user_message):
            log_event("general_web_search_triggered", session_id=context.session_id, message=user_message[:140])
            web_payload = await self.web_search.search(user_message, entities={})
            web_results = web_payload.get("results") if isinstance(web_payload, dict) else []
            if isinstance(web_results, list) and web_results:
                answer = await self._render_tool_reply(
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
            system_prompt=self.system_prompt,
            summary=context.summary,
            turns=context.turns,
            user_message=user_message,
        )
        text = await self._llm_text(prompt)
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

    async def _handle_guard_decision(
        self,
        *,
        session_id: str,
        guard_state: str,
        user_message: str,
        context: SessionContext,
    ) -> AgentReply | None:
        state = str(guard_state or "")
        decision = self._yes_no_decision(user_message)

        if state == _CTX_GUARD_AWAITING_IMMEDIATE:
            if decision == "yes":
                await self.memory.clear_session(session_id)
                log_event("context_guard_clear_now", session_id=session_id)
                return AgentReply(
                    text="Диалог завершен и удален. Начинаем новую сессию.",
                    source="system",
                    next_session_id=_new_session_id(),
                )
            if decision == "no":
                await self.memory.set_meta_str(session_id, _CTX_GUARD_STATE_KEY, _CTX_GUARD_ONE_MORE)
                log_event("context_guard_one_more_accepted", session_id=session_id)
                return AgentReply(
                    text=(
                        "Принято. Я приму еще одно сообщение в текущем диалоге, "
                        "после этого попрошу финально выбрать способ завершения."
                    ),
                    source="system",
                )
            return AgentReply(
                text=(
                    "Пожалуйста, ответьте: Да или Нет.\n\n"
                    + self._guard_near_limit_prompt()
                ),
                source="system",
            )

        if state == _CTX_GUARD_AWAITING_FINAL:
            if decision == "yes":
                await self.memory.clear_session(session_id)
                log_event("context_guard_clear_final_yes", session_id=session_id)
                return AgentReply(
                    text="Диалог удален полностью. Начинаем новую сессию.",
                    source="system",
                    next_session_id=_new_session_id(),
                )
            if decision == "no":
                compact_summary = await self._ensure_compacted_summary(session_id, context)
                new_session_id = _new_session_id()
                if compact_summary:
                    await self.memory.save_summary(new_session_id, compact_summary)
                log_event(
                    "context_guard_rollover_with_summary",
                    session_id=session_id,
                    next_session_id=new_session_id,
                    summary_chars=len(compact_summary),
                )
                return AgentReply(
                    text=(
                        "Диалог сохранен в компактной памяти. "
                        "Запускаю новую сессию с этим контекстом."
                    ),
                    source="system",
                    next_session_id=new_session_id,
                )
            return AgentReply(
                text=(
                    "Нужен точный ответ: Да или Нет.\n\n"
                    + self._guard_final_choice_prompt()
                ),
                source="system",
            )

        return None

    def _yes_no_decision(self, text: str) -> str:
        value = str(text or "").strip()
        if _YES_RE.match(value):
            return "yes"
        if _NO_RE.match(value):
            return "no"
        return "unknown"

    def _guard_near_limit_prompt(self) -> str:
        return (
            "Контекстное окно почти заполнено. Следует вскоре начать новый диалог.\n"
            "Можно завершить этот диалог прямо сейчас? Да/нет."
        )

    def _guard_final_choice_prompt(self) -> str:
        return (
            "Контекст почти исчерпан, текущий диалог будет закрыт.\n"
            "Да — удалить этот диалог полностью.\n"
            "Нет — сохранить компактный контекст и начать новую сессию.\n"
            "Ответьте: Да или Нет."
        )

    def _is_context_guard_needed(self, *, context: SessionContext, user_message: str) -> bool:
        window = max(1024, int(self.config.context_window_tokens))
        reserve = max(64, int(self.config.context_response_reserve_tokens))
        budget = max(256, window - reserve)
        est_tokens = self._estimate_context_tokens(context=context, user_message=user_message)
        ratio = float(est_tokens) / float(budget)
        if ratio >= float(self.config.context_warn_ratio):
            return True
        prompt_eval = int(self._last_prompt_eval_count)
        if prompt_eval <= 0:
            return False
        return (float(prompt_eval) / float(window)) >= float(self.config.context_warn_ratio)

    def _estimate_context_tokens(self, *, context: SessionContext, user_message: str) -> int:
        parts: list[str] = [self.system_prompt, str(context.summary or ""), str(user_message or "")]
        for turn in context.turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip()
            content = str(turn.get("content") or "").strip()
            if content:
                parts.append(f"{role}: {content}")
        chars = sum(len(item) for item in parts if item)
        div = max(2, int(self.config.context_estimate_chars_per_token))
        return max(1, chars // div)

    async def _ensure_compacted_summary(self, session_id: str, context: SessionContext) -> str:
        base_context = context
        if not base_context.turns:
            base_context = await self.memory.load_context(
                session_id,
                history_tail_turns=max(self.config.compaction_trigger_turns, self.config.history_tail_turns),
            )
        summary = str(base_context.summary or "").strip()
        if not summary:
            prompt = build_summary_prompt(previous_summary="", turns=base_context.turns)
            llm_summary = await self._llm_text(prompt)
            summary = str(llm_summary or "").strip()
        if not summary:
            summary = self._fallback_summary(base_context).strip()
        if summary:
            await self.memory.save_summary(session_id, summary)
            await self.persist.append_snapshot(
                session_id=session_id,
                summary=summary,
                key_facts=self._extract_key_facts(summary),
                open_loops=[],
                extra={"reason": "context_rollover"},
            )
            log_event(
                "context_rollover_snapshot_saved",
                session_id=session_id,
                summary_chars=len(summary),
            )
        return summary

    async def _medical_reply(self, user_message: str, context: SessionContext) -> AgentReply:
        health = await self.services.get_catalog_health()
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

        pending_state = await self._load_clinical_pending_state(context.session_id)
        remembered_doctor = await self.memory.get_meta_str(context.session_id, _LAST_DOCTOR_NAME_KEY, "")
        decision = await self._route_clinical_decision(
            user_message=user_message,
            context=context,
            pending_state=pending_state,
            remembered_doctor=remembered_doctor,
        )
        entities = await self._ground_entities(user_message, intent_hint=decision.intent)
        merged_entities = dict(decision.entities)
        merged_entities.update(entities)
        entities = self._apply_intent_entity_policy(
            user_message=user_message,
            intent=decision.intent,
            entities=merged_entities,
        )
        tool_plan = self._build_tool_plan_from_decision(
            user_message=user_message,
            decision=decision,
        )
        entities = await self._enrich_entities_from_session_memory(
            session_id=context.session_id,
            user_message=user_message,
            tool_plan=tool_plan,
            entities=entities,
        )

        missing_from_plan = merge_missing_slots_from_plan(tool_plan, entities)
        missing_slots = self._merge_missing_slots(decision.missing_slots, missing_from_plan)
        if "doctor_name_or_specialty" in missing_slots and remembered_doctor:
            entities["doctor_name"] = str(remembered_doctor).strip()
            missing_slots = [slot for slot in missing_slots if slot != "doctor_name_or_specialty"]

        needs_clarification = bool(missing_slots)
        if not needs_clarification and decision.confidence < _CLINICAL_MIN_CONFIDENCE and not tool_plan:
            needs_clarification = True

        log_event(
            "medical_plan_selected",
            session_id=context.session_id,
            tool_plan=",".join(tool_plan),
            entities=self._public_entities(entities),
            router_intent=decision.intent,
            router_confidence=decision.confidence,
            router_source=decision.source,
            missing_slots=",".join(missing_slots),
            message=user_message[:160],
        )
        if needs_clarification:
            clarify_text = str(decision.clarify_question or "").strip()
            if not clarify_text:
                clarify_text = clarify_question_for_slots(decision.intent, missing_slots)
            attempts = int(pending_state.get("attempts") or 0)
            same_pending = self._is_same_pending_request(
                pending_state=pending_state,
                intent=decision.intent,
                missing_slots=missing_slots,
            )
            next_attempt = attempts + 1 if same_pending else 1
            if next_attempt > _CLINICAL_MAX_CLARIFY_RETRIES:
                await self._clear_clinical_pending_state(context.session_id)
                return AgentReply(
                    text="Не удалось однозначно уточнить запрос. Укажите, пожалуйста, фамилию врача или точное название услуги.",
                    source="clinic_data",
                )
            await self._save_clinical_pending_state(
                context.session_id,
                {
                    "intent": decision.intent,
                    "missing_slots": missing_slots,
                    "clarify_question": clarify_text,
                    "attempts": next_attempt,
                    "phase": "pre_tool",
                },
            )
            log_event(
                "medical_clarification_requested",
                level=logging.WARNING,
                session_id=context.session_id,
                intent=decision.intent,
                attempts=next_attempt,
                missing_slots=",".join(missing_slots),
                question=clarify_text[:180],
            )
            return AgentReply(text=clarify_text, source="clinic_data")

        await self._clear_clinical_pending_state(context.session_id)
        if not tool_plan:
            log_event("medical_plan_empty_not_found", level=logging.WARNING, session_id=context.session_id)
            return AgentReply(
                text=(
                    "Не удалось однозначно определить медицинский запрос. "
                    "Уточните врача, услугу, анализ или тип вопроса (цена/подготовка/расписание)."
                ),
                source="clinic_data",
            )

        catalog_resolution_reply = self._catalog_resolution_reply_if_needed(
            tool_plan=tool_plan,
            entities=entities,
            fallback_only=False,
        )
        if catalog_resolution_reply is not None:
            log_event(
                "medical_catalog_resolution_reply",
                level=logging.WARNING,
                session_id=context.session_id,
                tool_plan=",".join(tool_plan),
                entities=self._public_entities(entities),
            )
            return catalog_resolution_reply

        dispatcher = ToolDispatcher(self.services.tool_handlers(include_meili_tools=self.config.include_meili_tools))
        tool_entities = self._public_entities(entities)
        for tool_name in tool_plan[: self.config.max_tool_steps]:
            log_event("medical_tool_call_start", session_id=context.session_id, tool_name=tool_name)
            result = await dispatcher.call(tool_name, user_message, entities=tool_entities)
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

            if result.tool_name == "doctors_schedule_week":
                stats = self._schedule_payload_stats(result.payload)
                log_event(
                    "medical_schedule_payload_stats",
                    session_id=context.session_id,
                    doctors_count=stats["doctors_count"],
                    regions_count=stats["regions_count"],
                    days_count=stats["days_count"],
                    slots_count=stats["slots_count"],
                    schedule_unavailable_reason=stats["schedule_unavailable_reason"],
                )

            answer = await self._render_tool_reply(
                user_message=user_message,
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
            await self._remember_doctor_from_tool_result(
                session_id=context.session_id,
                tool_name=result.tool_name,
                payload=result.payload,
            )
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
            tool_plan=",".join(tool_plan[: self.config.max_tool_steps]),
        )
        exhausted_reply = self._catalog_resolution_reply_if_needed(
            tool_plan=tool_plan,
            entities=entities,
            fallback_only=True,
        )
        if exhausted_reply is not None:
            return exhausted_reply

        retry_after_not_found = bool(
            tool_plan
            and str(pending_state.get("phase") or "").strip().lower() != "post_not_found"
        )
        if retry_after_not_found:
            clarify_text = clarify_question_for_slots(decision.intent, missing_slots)
            await self._save_clinical_pending_state(
                context.session_id,
                {
                    "intent": decision.intent,
                    "missing_slots": missing_slots,
                    "clarify_question": clarify_text,
                    "attempts": 1,
                    "phase": "post_not_found",
                },
            )
            return AgentReply(
                text=f"В данных клиники по текущему запросу ничего не найдено. {clarify_text}",
                source="clinic_data",
            )

        await self._clear_clinical_pending_state(context.session_id)
        return AgentReply(text="В данных клиники по вашему запросу ничего не найдено.", source="clinic_data")

    async def _ground_entities(self, user_message: str, *, intent_hint: str = "") -> dict[str, Any]:
        entities: dict[str, Any] = {}
        intent = str(intent_hint or "").strip().lower()
        doctor_focus = intent in {"doctor_info", "doctor_schedule"}
        service_focus = intent in {"price", "prepare", "tests", "service_info"}
        should_try_service = service_focus or not doctor_focus

        if should_try_service:
            try:
                service_match = await self.services.match_catalog_service(user_message, current_service_name="")
            except Exception as exc:
                log_event(
                    "entity_ground_service_failed",
                    level=logging.WARNING,
                    error_type=type(exc).__name__,
                )
                service_match = {}
            if isinstance(service_match, dict):
                status = str(service_match.get("status") or "")
                canonical = str(service_match.get("canonical") or "").strip()
                entities["_ft_service_match_status"] = status
                entities["_ft_service_match_query"] = str(service_match.get("query") or "").strip()
                if status == "exact" and canonical:
                    entities["service_name"] = canonical
        else:
            entities["_ft_service_match_status"] = "skipped_doctor_focus"
            entities["_ft_service_match_query"] = ""

        try:
            doctor_match = await self.services.match_catalog_doctor(user_message)
        except Exception as exc:
            log_event(
                "entity_ground_doctor_failed",
                level=logging.WARNING,
                error_type=type(exc).__name__,
            )
            doctor_match = {}
        if isinstance(doctor_match, dict):
            status = str(doctor_match.get("status") or "")
            canonical = str(doctor_match.get("canonical") or "").strip()
            entities["_ft_doctor_match_status"] = status
            entities["_ft_doctor_match_query"] = str(doctor_match.get("query") or "").strip()
            if status in {"exact", "fuzzy"} and canonical:
                entities["doctor_name"] = canonical
                entities["doctor_name_match_status"] = status

        return entities

    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        pending_state: dict[str, Any],
        remembered_doctor: str,
    ) -> ClinicalDecision:
        prompt = build_clinical_router_prompt(
            system_prompt=self.system_prompt,
            summary=context.summary,
            turns=context.turns,
            user_message=user_message,
            pending_intent=str(pending_state.get("intent") or ""),
            pending_slots=[
                str(slot).strip()
                for slot in (pending_state.get("missing_slots") or [])
                if str(slot).strip()
            ],
            remembered_doctor=str(remembered_doctor or "").strip(),
        )
        payload = await self._llm_json(prompt)
        decision = parse_clinical_decision(
            payload,
            include_meili_tools=self.config.include_meili_tools,
        )
        if not decision.tool_plan:
            decision.tool_plan = select_tool_plan(
                user_message,
                include_meili_tools=self.config.include_meili_tools,
            )
            if decision.intent == "unknown" and decision.tool_plan:
                decision.intent = self._infer_intent_from_tool_plan(decision.tool_plan)
            if not decision.source:
                decision.source = "heuristic"
        return decision

    async def _llm_json(self, prompt: str) -> dict[str, Any]:
        self._last_prompt_eval_count = 0
        try:
            text, usage = await freetalk_llm_port.generate_text_with_usage(
                prompt,
                timeout_s=self.config.llm_timeout_s,
                queue_timeout_ms=self.config.llm_queue_timeout_ms,
                fmt="json",
                think=False,
            )
        except Exception as exc:
            log_event(
                "llm_json_generate_failed",
                level=logging.WARNING,
                error_type=type(exc).__name__,
            )
            return {}
        try:
            self._last_prompt_eval_count = int(usage.get("prompt_eval_count", 0))
        except Exception:
            self._last_prompt_eval_count = 0
        parsed = self._parse_json_object(text)
        if not parsed:
            log_event(
                "llm_json_parse_empty",
                level=logging.WARNING,
                payload_preview=str(text or "")[:220],
            )
        return parsed

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        candidates: list[str] = [text]
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return dict(parsed[0])
        return {}

    @staticmethod
    def _merge_missing_slots(primary: list[str], secondary: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for source in (primary or [], secondary or []):
            slot = str(source or "").strip().lower()
            if not slot or slot in seen:
                continue
            seen.add(slot)
            out.append(slot)
        return out

    @staticmethod
    def _is_same_pending_request(
        *,
        pending_state: dict[str, Any],
        intent: str,
        missing_slots: list[str],
    ) -> bool:
        old_intent = str(pending_state.get("intent") or "").strip().lower()
        old_slots = sorted(str(slot or "").strip().lower() for slot in (pending_state.get("missing_slots") or []))
        new_slots = sorted(str(slot or "").strip().lower() for slot in (missing_slots or []))
        return bool(old_intent and old_intent == str(intent or "").strip().lower() and old_slots == new_slots)

    async def _load_clinical_pending_state(self, session_id: str) -> dict[str, Any]:
        raw = await self.memory.get_meta_str(session_id, _CLINICAL_PENDING_STATE_KEY, "")
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _save_clinical_pending_state(self, session_id: str, state: dict[str, Any]) -> None:
        payload = state if isinstance(state, dict) else {}
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except Exception:
            encoded = "{}"
        await self.memory.set_meta_str(session_id, _CLINICAL_PENDING_STATE_KEY, encoded)

    async def _clear_clinical_pending_state(self, session_id: str) -> None:
        await self.memory.set_meta_str(session_id, _CLINICAL_PENDING_STATE_KEY, "")

    @staticmethod
    def _infer_intent_from_tool_plan(tool_plan: list[str]) -> str:
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

    def _build_tool_plan_from_decision(self, *, user_message: str, decision: ClinicalDecision) -> list[str]:
        from_router = [tool for tool in (decision.tool_plan or []) if isinstance(tool, str)]
        if from_router:
            return from_router
        return select_tool_plan(user_message, include_meili_tools=self.config.include_meili_tools)

    def _apply_intent_entity_policy(self, *, user_message: str, intent: str, entities: dict[str, Any]) -> dict[str, Any]:
        out = dict(entities or {})
        route_intent = str(intent or "").strip().lower()
        if route_intent in {"doctor_info", "doctor_schedule"}:
            if str(out.get("doctor_name") or "").strip():
                out.pop("service_name", None)
                out.pop("test_name", None)
        if route_intent in {"clinic_news", "clinic_documents"}:
            out.pop("doctor_name", None)
            out.pop("specialty", None)
        if route_intent in {"price", "prepare", "tests", "service_info"} and not str(out.get("service_name") or "").strip():
            if str(out.get("doctor_name") or "").strip() and re.search(r"\b(врач|доктор|у\s+[а-яё\-]{3,})\b", user_message, re.I):
                out.pop("service_name", None)
        return out

    @staticmethod
    def _public_entities(entities: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (entities or {}).items() if not str(k).startswith("_ft_")}

    @staticmethod
    def _looks_like_specific_doctor_lookup(value: str) -> bool:
        tokens = [t.lower() for t in _TOKEN_RE.findall(str(value or "")) if len(t) >= 3]
        if not tokens:
            return False
        return any(token not in _DOCTOR_NON_PERSON_TOKENS for token in tokens)

    @staticmethod
    def _looks_like_specific_service_lookup(value: str) -> bool:
        tokens = [t.lower() for t in _TOKEN_RE.findall(str(value or "")) if len(t) >= 3]
        if not tokens:
            return False
        return any(token not in _SERVICE_NON_SPECIFIC_TOKENS for token in tokens)

    def _catalog_resolution_reply_if_needed(
        self,
        *,
        tool_plan: list[str],
        entities: dict[str, Any],
        fallback_only: bool = False,
    ) -> AgentReply | None:
        doctor_tools = {"doctors_info", "doctors_schedule_week"}
        if any(tool in doctor_tools for tool in tool_plan):
            doctor_status = str(entities.get("_ft_doctor_match_status") or "").strip().lower()
            doctor_query = str(entities.get("_ft_doctor_match_query") or "").strip()
            if doctor_status == "unavailable" and not fallback_only:
                return AgentReply(
                    text="Сейчас каталог врачей клиники недоступен. Повторите запрос немного позже.",
                    source="clinic_data",
                )
            if doctor_status == "miss" and fallback_only and self._looks_like_specific_doctor_lookup(doctor_query):
                label = _clean_user_fragment(doctor_query)
                suffix = f" «{label}»" if label else ""
                return AgentReply(
                    text=f"В данных клиники врач{suffix} не найден. Проверьте фамилию или уточните специальность.",
                    source="clinic_data",
                )

        service_tools = {"price_info", "service_bundle_info", "test_prepare", "test_assist"}
        if any(tool in service_tools for tool in tool_plan):
            service_status = str(entities.get("_ft_service_match_status") or "").strip().lower()
            service_query = str(entities.get("_ft_service_match_query") or "").strip()
            if service_status == "unavailable" and not fallback_only:
                return AgentReply(
                    text="Сейчас каталог услуг клиники недоступен. Повторите запрос немного позже.",
                    source="clinic_data",
                )
            if service_status == "miss" and fallback_only and self._looks_like_specific_service_lookup(service_query):
                label = _clean_user_fragment(service_query)
                suffix = f" «{label}»" if label else ""
                return AgentReply(
                    text=f"В данных клиники услуга или анализ{suffix} не найдены. Уточните название.",
                    source="clinic_data",
                )

        return None

    async def _enrich_entities_from_session_memory(
        self,
        *,
        session_id: str,
        user_message: str,
        tool_plan: list[str],
        entities: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(entities or {})
        uses_doctor_tools = bool({"doctors_schedule_week", "doctors_info"} & set(tool_plan or []))
        if not uses_doctor_tools:
            return out

        current_doctor = str(out.get("doctor_name") or "").strip()
        if current_doctor:
            return out

        if not _DOCTOR_ANAPHORA_RE.search(str(user_message or "")):
            return out

        remembered = await self.memory.get_meta_str(session_id, _LAST_DOCTOR_NAME_KEY, "")
        remembered = str(remembered or "").strip()
        if not remembered:
            return out

        out["doctor_name"] = remembered
        out["doctor_name_source"] = "session_memory"
        log_event(
            "medical_entities_enriched_from_memory",
            session_id=session_id,
            doctor_name=remembered,
            message=user_message[:140],
        )
        return out

    async def _remember_doctor_from_tool_result(
        self,
        *,
        session_id: str,
        tool_name: str,
        payload: dict[str, Any],
    ) -> None:
        doctor = self._extract_primary_doctor_name(tool_name, payload)
        if not doctor:
            return
        await self.memory.set_meta_str(session_id, _LAST_DOCTOR_NAME_KEY, doctor)
        log_event(
            "doctor_context_stored",
            session_id=session_id,
            tool_name=tool_name,
            doctor_name=doctor,
        )

    @staticmethod
    def _extract_primary_doctor_name(tool_name: str, payload: dict[str, Any]) -> str:
        entities_used = payload.get("entities_used") if isinstance(payload, dict) else {}
        if isinstance(entities_used, dict):
            for key in ("doctor_name_resolved", "doctor_name", "doctor_query", "doctor_resolved"):
                value = str(entities_used.get(key) or "").strip()
                if value:
                    return value

        doctors = payload.get("doctors") if isinstance(payload, dict) else []
        if isinstance(doctors, list):
            for row in doctors:
                if not isinstance(row, dict):
                    continue
                fio = str(row.get("fio") or "").strip()
                if fio:
                    return fio

        schedule = payload.get("schedule") if isinstance(payload, dict) else []
        if isinstance(schedule, list):
            for row in schedule:
                if not isinstance(row, dict):
                    continue
                fio = str(row.get("fio") or "").strip()
                if fio:
                    return fio

        if tool_name == "doctors_schedule_week":
            direct = str(payload.get("doctor_name") or payload.get("fio") or "").strip()
            if direct:
                return direct

        return ""

    async def _render_tool_reply(self, *, user_message: str, tool_name: str, tool_payload: dict[str, Any]) -> str:
        rendered = self._fallback_render(tool_name, tool_payload).strip()
        if _is_non_empty_text(rendered):
            log_event(
                "tool_rendered_deterministic",
                tool_name=tool_name,
                answer_chars=len(rendered),
                payload_note=str(tool_payload.get("note") or "")[:120],
            )
            return rendered
        log_event(
            "tool_render_fallback",
            level=logging.WARNING,
            user_message=user_message[:120],
            tool_name=tool_name,
            payload_note=str(tool_payload.get("note") or "")[:120],
        )
        prompt = build_tool_result_prompt(
            system_prompt=self.system_prompt,
            user_message=user_message,
            tool_name=tool_name,
            tool_payload=tool_payload,
        )
        llm_text = await self._llm_text(prompt)
        if _is_non_empty_text(llm_text):
            log_event(
                "tool_rendered_by_llm_fallback",
                level=logging.WARNING,
                tool_name=tool_name,
                answer_chars=len(llm_text),
            )
            return llm_text
        return "Нашел данные, но не удалось корректно сформировать ответ. Уточните запрос, и я отвечу точнее."

    def _looks_like_clinic_data_query(self, user_message: str) -> bool:
        text = _normalise_match_text(user_message)
        if not text:
            return False
        if "клиник" not in text and "наука" not in text:
            return False
        signals = (
            "кто",
            "выведи",
            "покажи",
            "спис",
            "врач",
            "доктор",
            "терап",
            "специал",
            "принима",
            "распис",
            "услуг",
            "процед",
            "анализ",
            "подготов",
            "цена",
            "прайс",
            "стоим",
            "филиал",
            "адрес",
            "результат",
            "наука",
        )
        return any(token in text for token in signals)

    def _looks_like_doctor_followup_message(self, *, user_message: str, remembered_doctor: str) -> bool:
        text = str(user_message or "").strip()
        if not text:
            return False
        remembered = str(remembered_doctor or "").strip()
        if not remembered:
            return False
        if _DOCTOR_ANAPHORA_RE.search(text):
            return True
        if _DOCTOR_FOLLOWUP_RE.search(text):
            return True
        lowered = text.lower().replace("ё", "е")
        remembered_parts = [part.strip().lower().replace("ё", "е") for part in remembered.split() if part.strip()]
        if not remembered_parts:
            return False
        if len(lowered.split()) <= 3:
            return any(part in lowered for part in remembered_parts)
        return False

    async def _maybe_compact(self, session_id: str) -> None:
        turn_count = await self.memory.get_turn_count(session_id)
        if turn_count < self.config.compaction_trigger_turns:
            return

        last_compacted_turn = await self.memory.get_meta_int(session_id, "last_compacted_turn", 0)
        min_delta = max(4, self.config.compaction_trigger_turns // 2)
        if turn_count - last_compacted_turn < min_delta:
            return

        context = await self.memory.load_context(
            session_id,
            history_tail_turns=max(self.config.compaction_trigger_turns, self.config.history_tail_turns),
        )
        prompt = build_summary_prompt(previous_summary=context.summary, turns=context.turns)
        summary = await self._llm_text(prompt)
        if not _is_non_empty_text(summary):
            summary = self._fallback_summary(context)
        summary = summary.strip()
        if not summary:
            return

        await self.memory.save_summary(session_id, summary)
        await self.memory.set_meta_int(session_id, "last_compacted_turn", turn_count)
        await self.persist.append_snapshot(
            session_id=session_id,
            summary=summary,
            key_facts=self._extract_key_facts(summary),
            open_loops=[],
            extra={"turn_count": turn_count},
        )
        log_event(
            "dialog_compacted",
            session_id=session_id,
            turn_count=turn_count,
            summary_chars=len(summary),
        )

    async def _llm_text(self, prompt: str) -> str:
        self._last_prompt_eval_count = 0
        try:
            text, usage = await freetalk_llm_port.generate_text_with_usage(
                prompt,
                timeout_s=self.config.llm_timeout_s,
                queue_timeout_ms=self.config.llm_queue_timeout_ms,
                fmt=None,
                think=False,
            )
        except Exception as exc:
            log_event(
                "llm_generate_failed",
                level=logging.ERROR,
                error_type=type(exc).__name__,
            )
            return ""
        try:
            self._last_prompt_eval_count = int(usage.get("prompt_eval_count", 0))
        except Exception:
            self._last_prompt_eval_count = 0
        return str(text or "").strip()

    def _fallback_summary(self, context: SessionContext) -> str:
        recent = [str(t.get("content") or "").strip() for t in context.turns if isinstance(t, dict)]
        recent = [x for x in recent if x][-self.config.summary_keep_turns :]
        if not recent:
            return context.summary.strip()
        lines = [f"- {line}" for line in recent]
        base = context.summary.strip()
        if base:
            return f"{base}\n" + "\n".join(lines)
        return "\n".join(lines)

    @staticmethod
    def _extract_key_facts(summary: str) -> list[str]:
        text = str(summary or "").strip()
        if not text:
            return []
        chunks = [item.strip(" -\t\r\n") for item in text.replace("\r", "\n").split("\n")]
        chunks = [chunk for chunk in chunks if chunk]
        return _top_list(chunks, limit=5)

    @staticmethod
    def _format_iso_date_short(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed.strftime("%d.%m")
        except Exception:
            pass
        if len(raw) >= 10:
            return raw[:10]
        return raw

    @classmethod
    def _schedule_day_line(cls, day: dict[str, Any]) -> tuple[str, bool]:
        if not isinstance(day, dict):
            return "", False

        date_label = cls._format_iso_date_short(str(day.get("date") or ""))
        slots_raw = day.get("slots")
        slots: list[str] = []
        if isinstance(slots_raw, list):
            for item in slots_raw:
                text = str(item or "").strip()
                if not text:
                    continue
                slots.append(text[:5] if len(text) >= 5 else text)

        if slots:
            title = date_label or "Ближайшая дата"
            return f"{title}: {', '.join(_top_list(slots, limit=6))}", True

        start = str(day.get("start") or "").strip()
        end = str(day.get("end") or "").strip()
        if start or end:
            title = date_label or "Ближайшая дата"
            interval = f"{start[:5] if start else ''}-{end[:5] if end else ''}".strip("-")
            if interval:
                return f"{title}: {interval}", False

        return "", False

    @classmethod
    def _render_schedule_details(cls, payload: dict[str, Any]) -> str:
        schedule = payload.get("schedule")
        if not isinstance(schedule, list) or not schedule:
            return ""

        lines: list[str] = ["Нашел расписание:"]
        rendered = 0
        has_free_slots = False
        for row in _top_list(schedule, 3):
            if not isinstance(row, dict):
                continue

            rendered += 1
            fio = str(row.get("fio") or "").strip() or "Врач"
            lines.append(f"{rendered}. {fio}")

            row_schedule = row.get("schedule")
            row_lines = 0
            if isinstance(row_schedule, dict):
                for region_name, days in list(row_schedule.items())[:2]:
                    day_lines: list[str] = []
                    if isinstance(days, list):
                        for day in days[:4]:
                            preview, day_has_free = cls._schedule_day_line(day)
                            if not preview:
                                continue
                            day_lines.append(preview)
                            if day_has_free:
                                has_free_slots = True
                    if not day_lines:
                        continue

                    region = str(region_name or "").strip()
                    if region:
                        lines.append(f"   {region}")
                    for preview in day_lines:
                        lines.append(f"   • {preview}")
                    row_lines += len(day_lines)

            if row_lines == 0:
                lines.append("   Свободные окна по этому врачу не найдены, уточните дату или филиал.")
            lines.append("")

        if rendered == 0:
            return ""

        reason = str(payload.get("schedule_unavailable_reason") or "").strip().lower()
        if not has_free_slots and reason == "no_free_slots_2_weeks":
            lines.append("На ближайшие две недели свободных слотов по этому запросу нет.")
        elif has_free_slots:
            lines.append("Если нужно, уточню ближайшие окна по филиалу и дате.")

        return "\n".join([line for line in lines if str(line).strip()]).strip()

    @staticmethod
    def _schedule_payload_stats(payload: dict[str, Any]) -> dict[str, Any]:
        schedule = payload.get("schedule")
        doctors_count = 0
        regions_count = 0
        days_count = 0
        slots_count = 0
        if isinstance(schedule, list):
            for row in schedule:
                if not isinstance(row, dict):
                    continue
                doctors_count += 1
                row_schedule = row.get("schedule")
                if not isinstance(row_schedule, dict):
                    continue
                for _, days in row_schedule.items():
                    regions_count += 1
                    if not isinstance(days, list):
                        continue
                    for day in days:
                        if not isinstance(day, dict):
                            continue
                        days_count += 1
                        day_slots = day.get("slots")
                        if isinstance(day_slots, list):
                            slots_count += sum(1 for slot in day_slots if str(slot or "").strip())
        return {
            "doctors_count": doctors_count,
            "regions_count": regions_count,
            "days_count": days_count,
            "slots_count": slots_count,
            "schedule_unavailable_reason": str(payload.get("schedule_unavailable_reason") or "").strip(),
        }

    def _fallback_render(self, tool_name: str, payload: dict[str, Any]) -> str:
        clarify_text = str(payload.get("clarify_text") or "").strip()
        if clarify_text:
            return clarify_text

        handoff_text = str(payload.get("handoff_message") or "").strip()
        if handoff_text:
            return handoff_text

        if tool_name == "test_result_status":
            links = [str(x).strip() for x in (payload.get("result_links") or []) if str(x).strip()]
            if payload.get("ready") and links:
                return f"Результат готов. Ссылка: {links[0]}"
            missing = payload.get("missing_fields") or []
            if missing:
                return "Чтобы проверить результат, уточните: " + ", ".join(str(x) for x in missing)
            return "По указанным данным результат пока не найден или еще не готов."

        if tool_name == "test_prepare":
            text = str(payload.get("prepare") or "").strip()
            if text:
                return text

        if tool_name == "price_info":
            prices = [x for x in (payload.get("prices") or []) if isinstance(x, dict)]
            if prices:
                lines = ["Нашел цены по запросу:"]
                for row in _top_list(prices, 5):
                    name = str(row.get("serviceName") or row.get("name") or "").strip()
                    cost = row.get("cost")
                    if name and cost:
                        lines.append(f"- {name}: {cost} ₽")
                    elif name:
                        lines.append(f"- {name}")
                return "\n".join(lines)
            return "По вашему запросу цены в данных клиники не найдены."

        if tool_name == "test_assist":
            tests = [x for x in (payload.get("tests") or []) if isinstance(x, dict)]
            if tests:
                lines = ["Подходящие анализы:"]
                for row in _top_list(tests, 5):
                    name = str(row.get("serviceName") or row.get("name") or "").strip()
                    cost = row.get("cost")
                    if name and cost:
                        lines.append(f"- {name}: {cost} ₽")
                    elif name:
                        lines.append(f"- {name}")
                return "\n".join(lines)
            return "Подходящие анализы в данных клиники не найдены."

        if tool_name == "address_info":
            addresses = [str(x).strip() for x in (payload.get("addresses") or []) if str(x).strip()]
            if addresses:
                lines = ["Нашел адреса филиалов:"]
                lines.extend(f"- {addr}" for addr in _top_list(addresses, 6))
                return "\n".join(lines)
            return "Адреса по вашему запросу в данных клиники не найдены."

        if tool_name == "doctors_info":
            doctors = [x for x in (payload.get("doctors") or []) if isinstance(x, dict)]
            if doctors:
                lines = ["Нашел врачей по запросу:"]
                for doc in _top_list(doctors, 6):
                    fio = str(doc.get("fio") or "").strip()
                    spec = str(doc.get("specialization") or "").strip()
                    if fio and spec:
                        lines.append(f"- {fio} ({spec})")
                    elif fio:
                        lines.append(f"- {fio}")
                return "\n".join(lines)
            return "Врачей по вашему запросу в данных клиники не найдено."

        if tool_name == "doctors_schedule_week":
            detailed = self._render_schedule_details(payload)
            if detailed:
                return detailed
            reason = str(payload.get("schedule_unavailable_reason") or "").strip().lower()
            if reason == "no_free_slots_2_weeks":
                return "На ближайшие две недели свободных слотов по этому запросу нет."
            return "Расписание по вашему запросу в данных клиники не найдено."

        if tool_name == "service_bundle_info":
            prices = [x for x in (payload.get("retail_prices") or []) if isinstance(x, dict)]
            doctors = [x for x in (payload.get("doctors") or []) if isinstance(x, dict)]
            lines: list[str] = []
            if prices:
                top = prices[0]
                name = str(top.get("serviceName") or top.get("name") or "").strip()
                cost = top.get("cost")
                if name and cost:
                    lines.append(f"Услуга: {name}. Цена от {cost} ₽.")
            if doctors:
                lines.append("Врачи по услуге:")
                for doc in _top_list(doctors, 4):
                    fio = str(doc.get("fio") or "").strip()
                    if fio:
                        lines.append(f"- {fio}")
            prepare = str(payload.get("prepare") or "").strip()
            if prepare:
                lines.append("")
                lines.append("Подготовка:")
                lines.append(prepare)
            if lines:
                return "\n".join(lines)
            return "По этой услуге в данных клиники сейчас нет релевантной информации."

        if tool_name == "main_index_info":
            content = str(payload.get("content") or "").strip()
            if content:
                return content
            return "По вашему запросу в базе знаний клиники ничего не найдено."

        if tool_name == "news_info":
            news = [x for x in (payload.get("news") or []) if isinstance(x, dict)]
            if news:
                lines = ["Нашел новости клиники:"]
                for item in _top_list(news, 5):
                    title = str(item.get("title") or item.get("name") or "").strip()
                    url = str(item.get("url") or item.get("link") or "").strip()
                    if title and url:
                        lines.append(f"- {title} ({url})")
                    elif title:
                        lines.append(f"- {title}")
                    elif url:
                        lines.append(f"- {url}")
                return "\n".join(lines)
            return "Новости по вашему запросу не найдены."

        if tool_name == "web_search":
            results = [x for x in (payload.get("results") or []) if isinstance(x, dict)]
            if results:
                lines = [
                    "Нашел в интернете:",
                ]
                for row in _top_list(results, 5):
                    title = str(row.get("title") or "").strip()
                    url = str(row.get("url") or "").strip()
                    snippet = str(row.get("snippet") or "").strip()
                    source = str(row.get("source") or "").strip()
                    if title and url:
                        lines.append(f"- {title} ({url})")
                    elif title:
                        lines.append(f"- {title}")
                    elif url:
                        lines.append(f"- {url}")
                    if snippet:
                        lines.append(f"  {snippet}")
                    if source:
                        lines.append(f"  Источник поиска: {source}")
                lines.append("")
                lines.append("Это общая информация из интернет-поиска, не из данных клиники.")
                return "\n".join(lines)
            return "В интернет-поиске по этому запросу не найдено релевантных результатов."

        content = str(payload.get("content") or "").strip()
        if content:
            return content
        return "Нашел данные, но не удалось корректно сформировать ответ. Уточните вопрос, и я попробую точнее."
