"""FreeTalk agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any
import uuid

from ..ports import freetalk_llm_port
from ..ports.freetalk_services_port import LegacyServicesPort
from ..ports.freetalk_web_search_port import WebSearchPort

from .adapter import FreeTalkAdapter
from .candidate_policy import (
    candidate_confirmation_question as _candidate_confirmation_question_helper,
    candidate_confirmation_target as _candidate_confirmation_target_helper,
    candidate_entities_from_entities as _candidate_entities_from_entities_helper,
    candidate_rejected_question as _candidate_rejected_question_helper,
    promote_confirmed_candidate as _promote_confirmed_candidate_helper,
)
from .catalog_policy import (
    catalog_resolution_reply_if_needed as _catalog_resolution_reply_if_needed_helper,
    looks_like_specific_doctor_lookup as _looks_like_specific_doctor_lookup_helper,
    looks_like_specific_service_lookup as _looks_like_specific_service_lookup_helper,
)
from .routing_contract import (
    ClinicalDecision,
    clarify_type_for_slots,
    clarify_question_for_slots,
    parse_clinical_decision,
)
from .config import FreeTalkConfig
from .contracts import AgentReply, DialogAct, DialogState, PostToolVerification, SessionContext
from .dialog_state import (
    clear_dialog_state as _clear_dialog_state_helper,
    copy_dialog_state as _copy_dialog_state_helper,
    dialog_state_from_payload as _dialog_state_from_payload_helper,
    dialog_state_is_active as _dialog_state_is_active_helper,
    dialog_state_payload as _dialog_state_payload_helper,
    filter_missing_slots_by_entities as _filter_missing_slots_by_entities_helper,
    load_dialog_state as _load_dialog_state_helper,
    load_session_entity_memory as _load_session_entity_memory_helper,
    merge_entities as _merge_entities_helper,
    merge_missing_slots as _merge_missing_slots_helper,
    save_dialog_state as _save_dialog_state_helper,
    save_session_entity_memory as _save_session_entity_memory_helper,
)
from .followup_policy import (
    contextual_followup_tool_plan as _contextual_followup_tool_plan_helper,
    extract_branch_reference as _extract_branch_reference_helper,
    extract_city_reference as _extract_city_reference_helper,
    extract_contextual_entities as _extract_contextual_entities_helper,
    extract_date_filters as _extract_date_filters_helper,
    extract_service_variant as _extract_service_variant_helper,
    extract_time_filters as _extract_time_filters_helper,
    looks_like_contextual_clinical_followup as _looks_like_contextual_clinical_followup_helper,
    looks_like_doctor_followup_message as _looks_like_doctor_followup_message_helper,
    looks_like_service_followup_message as _looks_like_service_followup_message_helper,
)
from .grounding_policy import ground_entities as _ground_entities_helper
from .intent_policy import apply_intent_entity_policy as _apply_intent_entity_policy_helper
from .memory_persist import PersistentSummaryStore
from .memory_policy import (
    enrich_entities_from_session_memory as _enrich_entities_from_session_memory_helper,
    extract_primary_doctor_name as _extract_primary_doctor_name_helper,
    extract_schedule_memory_entities as _extract_schedule_memory_entities_helper,
    remember_doctor_from_tool_result as _remember_doctor_from_tool_result_helper,
    tool_payload_memory_entities as _tool_payload_memory_entities_helper,
)
from .memory_redis import RedisMemoryStore
from .medical_pretool_policy import (
    build_clinical_state as _build_clinical_state_helper,
    finalize_pretool_state as _finalize_pretool_state_helper,
    merge_turn_entities as _merge_turn_entities_helper,
    rejected_candidate_slots as _rejected_candidate_slots_helper,
    resolve_current_service_name as _resolve_current_service_name_helper,
    resolve_tool_plan as _resolve_tool_plan_helper,
)
from .observability import log_event
from .orchestrator import (
    GuardRuntimeConfig,
    chat as _chat_orchestrator_helper,
    execute_dialog_act as _execute_dialog_act_orchestrator_helper,
    general_reply as _general_reply_orchestrator_helper,
    medical_reply as _medical_reply_orchestrator_helper,
    web_reply as _web_reply_orchestrator_helper,
)
from .routing_prompting import (
    build_clinical_router_prompt,
    build_post_tool_verifier_prompt,
    build_summary_prompt,
    build_tool_result_prompt,
    load_system_prompt,
)
from .post_tool_policy import (
    heuristic_post_tool_verification as _heuristic_post_tool_verification_helper,
    parse_post_tool_verification as _parse_post_tool_verification_helper,
)
from .rendering import (
    fallback_render as _fallback_render_helper,
    format_iso_date_short as _format_iso_date_short_helper,
    render_schedule_details as _render_schedule_details_helper,
    schedule_day_line as _schedule_day_line_helper,
    schedule_payload_stats as _schedule_payload_stats_helper,
)
from .routing_policy import resolve_dialog_act as _resolve_dialog_act_helper
from .tool_planning import is_about_agent_query, is_medical_query, select_tool_plan, should_use_web_search


def _is_non_empty_text(value: str) -> bool:
    return bool(str(value or "").strip())


def _top_list(values: list[Any], limit: int = 5) -> list[Any]:
    return list(values[: max(1, limit)])


_CTX_GUARD_STATE_KEY = "ctx_guard_state"
_LAST_DOCTOR_NAME_KEY = "last_doctor_name"
_CLINICAL_DIALOG_STATE_KEY = "clinical_dialog_state"
_CLINICAL_ENTITY_MEMORY_KEY = "clinical_entity_memory"
_CTX_GUARD_NONE = ""
_CTX_GUARD_AWAITING_IMMEDIATE = "awaiting_immediate"
_CTX_GUARD_ONE_MORE = "one_more"
_CTX_GUARD_AWAITING_FINAL = "awaiting_final"
_CLINICAL_MIN_CONFIDENCE = 0.58

_YES_RE = re.compile(r"^\s*(да|угу|ага|yes|yep|ok|ок|конечно)\s*[.!?]?\s*$", re.I)
_NO_RE = re.compile(r"^\s*(нет|неа|no|nope|not now|пока нет)\s*[.!?]?\s*$", re.I)
_DOCTOR_ANAPHORA_RE = re.compile(r"\b(его|него|нему|ним|он|у\s+него|у\s+него\s+же|у\s+неё|ее|её|она)\b", re.I)
_DOCTOR_FOLLOWUP_RE = re.compile(
    r"\b(доктор|врач|расписан|график|при(е|ё)м|слот|окн|чем\s+занима|о\s+нем|о\s+враче|инфо)\b",
    re.I,
)
_SERVICE_FOLLOWUP_RE = re.compile(
    r"\b(услуг|анализ|процедур|подготовк|цена|стоим|сколько|где|филиал|адрес|с\s+наркозом|без\s+наркоза)\b",
    re.I,
)
_SERVICE_VARIANT_RE = re.compile(r"\b(с\s+наркозом|без\s+наркоза|с\s+контрастом|без\s+контраста)\b", re.I)
_DATE_FILTER_RE = re.compile(
    r"\b("
    r"сегодня|завтра|послезавтра|"
    r"на\s+следующ(?:ей|ую)\s+неделе|"
    r"на\s+этой\s+неделе|"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\.\d{1,2}(?:\.\d{2,4})?"
    r")\b",
    re.I,
)
_TIME_FILTER_RE = re.compile(
    r"\b("
    r"утром|дн[её]м|вечером|"
    r"после\s+\d{1,2}(?::\d{2})?|"
    r"до\s+\d{1,2}(?::\d{2})?|"
    r"в\s+\d{1,2}:\d{2}|"
    r"\d{1,2}:\d{2}"
    r")\b",
    re.I,
)
_BRANCH_WORD_RE = re.compile(r"\b(филиал|адрес|локаци)\w*\b", re.I)
_SHORT_BRANCH_RE = re.compile(r"^\s*(?:а\s+)?(?:на|в)\s+([^?.!,]+?)\s*[?!.]?\s*$", re.I)
_BRANCH_CAPTURE_RE = re.compile(
    r"\b(?:филиал(?:е|ом)?|адрес(?:е|ом)?)(?:\s+на)?\s+([^?.!,]+?)(?:\s*[?!.]|$)",
    re.I,
)
_TIME_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_AFTER_RE = re.compile(r"\bпосле\s+(\d{1,2})(?::(\d{2}))?\b", re.I)
_TIME_BEFORE_RE = re.compile(r"\bдо\s+(\d{1,2})(?::(\d{2}))?\b", re.I)
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
_BRANCH_FOLLOWUP_STOPWORDS = {
    "сегодня",
    "завтра",
    "послезавтра",
    "утром",
    "вечером",
    "днем",
    "днём",
    "следующей",
    "следующую",
    "этой",
    "неделе",
    "неделю",
    "понедельник",
    "вторник",
    "среду",
    "четверг",
    "пятницу",
    "субботу",
    "воскресенье",
}
_SESSION_MEMORY_ENTITY_KEYS: tuple[str, ...] = (
    "doctor_name",
    "specialty",
    "service_name",
    "test_name",
    "service_variant",
    "city",
    "branch_name",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "date",
    "time",
    "result_surname",
    "result_year_of_birth",
    "result_analysis_code",
    "result_analysis_number",
    "doctor_id",
)


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
        "- \"Кто делает УЗИ шеи?\"\n"
        "- \"Как подготовиться к общему анализу крови?\"\n"
        "- \"Поищи в интернете последние новости по теме ...\""
    )


def _normalise_match_text(value: str) -> str:
    text = str(value or "").lower().strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _compose_service_with_variant(service_name: str, service_variant: str) -> str:
    base = str(service_name or "").strip()
    variant = str(service_variant or "").strip().lower()
    if not base or not variant:
        return base
    if variant in base.lower():
        return base
    return f"{base} {variant}".strip()


@dataclass(slots=True)
class FreeTalkAgent:
    config: FreeTalkConfig
    services: LegacyServicesPort
    memory: RedisMemoryStore
    persist: PersistentSummaryStore
    system_prompt: str
    web_search: WebSearchPort | None = None
    adapter: FreeTalkAdapter | None = None
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
            adapter=FreeTalkAdapter(),
        )

    async def chat(self, message: str, session_id: str) -> AgentReply:
        return await _chat_orchestrator_helper(
            self,
            message,
            session_id,
            guard=GuardRuntimeConfig(
                state_key=_CTX_GUARD_STATE_KEY,
                none_state=_CTX_GUARD_NONE,
                one_more_state=_CTX_GUARD_ONE_MORE,
                awaiting_immediate_state=_CTX_GUARD_AWAITING_IMMEDIATE,
                awaiting_final_state=_CTX_GUARD_AWAITING_FINAL,
            ),
            merge_text=_merge_text,
        )

    async def _build_dialog_act(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> DialogAct:
        memory_entities = await self._load_session_entity_memory(context.session_id)
        if is_about_agent_query(user_message):
            log_event(
                "route_intent_evaluated",
                session_id=context.session_id,
                route="general",
                llm_intent="about_agent",
                llm_confidence=1.0,
                llm_source="heuristic",
                medical_regex=False,
                medical_fallback=False,
                doctor_followup_hint=False,
                contextual_followup_hint=False,
                remembered_doctor=bool(str(remembered_doctor or "").strip()),
                web_search_signal=False,
                fallback_reason="",
                message=user_message[:180],
            )
            return DialogAct(
                route="general",
                intent="about_agent",
                response_policy="general_only",
                source="heuristic",
            )

        web_search_signal = should_use_web_search(user_message, allow_for_medical=True)
        active_dialog_state = self._dialog_state_is_active(dialog_state)
        decision = await self._route_clinical_decision(
            user_message=user_message,
            context=context,
            dialog_state=dialog_state,
            remembered_doctor=remembered_doctor,
        )
        medical_regex = is_medical_query(user_message)
        medical_fallback = self._looks_like_clinic_data_query(user_message)
        doctor_followup_hint = self._looks_like_doctor_followup_message(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
        )
        contextual_followup_hint = self._looks_like_contextual_clinical_followup(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
            memory_entities=memory_entities,
        )
        fallback_contextual_tool_plan: list[str] = []
        if contextual_followup_hint:
            fallback_contextual_tool_plan = self._contextual_followup_tool_plan(
                user_message=user_message,
                remembered_doctor=remembered_doctor,
                memory_entities=memory_entities,
            )
        dialog_act = _resolve_dialog_act_helper(
            user_message=user_message,
            decision=decision,
            dialog_state=dialog_state,
            active_dialog_state=active_dialog_state,
            web_search_signal=web_search_signal,
            web_search_available=self.web_search is not None,
            medical_regex=medical_regex,
            medical_fallback=medical_fallback,
            doctor_followup_hint=doctor_followup_hint,
            contextual_followup_hint=contextual_followup_hint,
            clinical_min_confidence=_CLINICAL_MIN_CONFIDENCE,
            include_meili_tools=self.config.include_meili_tools,
            fallback_contextual_tool_plan=fallback_contextual_tool_plan,
        )
        if dialog_act.source == "heuristic_fallback" and dialog_act.fallback_reason:
            log_event(
                "router_fallback_applied",
                level=logging.WARNING,
                session_id=context.session_id,
                reason=dialog_act.fallback_reason,
                llm_intent=dialog_act.intent,
                llm_confidence=dialog_act.confidence,
                tool_plan=",".join(dialog_act.tool_plan),
            )

        log_event(
            "route_intent_evaluated",
            session_id=context.session_id,
            route=dialog_act.route,
            llm_intent=dialog_act.intent,
            llm_confidence=dialog_act.confidence,
            llm_source=dialog_act.source,
            medical_regex=medical_regex,
            medical_fallback=medical_fallback,
            doctor_followup_hint=doctor_followup_hint,
            contextual_followup_hint=contextual_followup_hint,
            remembered_doctor=bool(str(remembered_doctor or "").strip()),
            web_search_signal=web_search_signal,
            fallback_reason=dialog_act.fallback_reason,
            message=user_message[:180],
        )
        return dialog_act

    async def _execute_dialog_act(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_act: DialogAct,
        dialog_state: DialogState,
    ) -> AgentReply:
        return await _execute_dialog_act_orchestrator_helper(
            self,
            user_message=user_message,
            context=context,
            dialog_act=dialog_act,
            dialog_state=dialog_state,
        )

    async def _web_reply(self, user_message: str, context: SessionContext) -> AgentReply:
        return await _web_reply_orchestrator_helper(self, user_message, context)

    def _source_fragments_for_reply(self, reply: AgentReply) -> list[dict[str, str]]:
        text = str(reply.text or "").strip()
        if not text:
            return []
        source = str(reply.source or "").strip().lower()
        if source == "clinic_data":
            return [{"text": text, "source": "clinic_data"}]
        if source == "general_knowledge":
            return [{"text": text, "source": "general_knowledge"}]
        if source == "mixed":
            if str(reply.tool_name or "").strip() == "web_search":
                return [{"text": text, "source": "web_search"}]
            marker = "Это общая информация, не из данных клиники."
            if marker in text:
                head, _, tail = text.partition(marker)
                fragments: list[dict[str, str]] = []
                if str(head or "").strip():
                    fragments.append({"text": str(head).strip(), "source": "clinic_data"})
                fragments.append({"text": marker, "source": "general_knowledge"})
                if str(tail or "").strip():
                    fragments.append({"text": str(tail).strip(), "source": "general_knowledge"})
                return fragments
            return [{"text": text, "source": "general_knowledge"}]
        if source == "system":
            return [{"text": text, "source": "general_knowledge"}]
        return [{"text": text, "source": "general_knowledge"}]

    def _log_source_trace(self, *, session_id: str, reply: AgentReply) -> None:
        fragments = list(reply.source_fragments or [])
        if not fragments:
            return
        tags: list[str] = []
        total_chars = 0
        for frag in fragments:
            if not isinstance(frag, dict):
                continue
            tag = str(frag.get("source") or "").strip().lower()
            text = str(frag.get("text") or "")
            if not tag:
                continue
            total_chars += len(text)
            if tag not in tags:
                tags.append(tag)
        log_event(
            "answer_source_trace",
            session_id=session_id,
            reply_source=reply.source,
            tags=",".join(tags),
            fragments_count=len(fragments),
            total_chars=total_chars,
            tool_name=reply.tool_name or "",
        )

    async def _general_reply(self, user_message: str, context: SessionContext) -> AgentReply:
        return await _general_reply_orchestrator_helper(self, user_message, context)

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

    async def _medical_reply(
        self,
        user_message: str,
        context: SessionContext,
        *,
        dialog_act: DialogAct | None = None,
        dialog_state: DialogState | None = None,
    ) -> AgentReply:
        return await _medical_reply_orchestrator_helper(
            self,
            user_message,
            context,
            dialog_act=dialog_act,
            dialog_state=dialog_state,
        )

    async def _ground_entities(
        self,
        user_message: str,
        *,
        intent_hint: str = "",
        current_service_name: str = "",
    ) -> dict[str, Any]:
        return await _ground_entities_helper(
            self.services,
            user_message,
            intent_hint=intent_hint,
            current_service_name=current_service_name,
        )

    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        prompt = build_clinical_router_prompt(
            system_prompt=self.system_prompt,
            summary=context.summary,
            turns=context.turns,
            user_message=user_message,
            pending_intent=str(dialog_state.intent or ""),
            pending_slots=[
                str(slot).strip()
                for slot in (dialog_state.missing_slots or [])
                if str(slot).strip()
            ],
            remembered_doctor=str(remembered_doctor or "").strip(),
            dialog_state=self._dialog_state_payload(dialog_state),
        )
        payload = await self._llm_json(prompt)
        decision = parse_clinical_decision(
            payload,
            include_meili_tools=self.config.include_meili_tools,
        )
        if not payload:
            decision.source = "llm_router_invalid_json"
        else:
            decision.source = "llm_router"
        return decision

    async def _post_tool_verify(
        self,
        *,
        user_message: str,
        intent: str,
        tool_name: str,
        drafted_answer: str,
        tool_payload: dict[str, Any],
        missing_slots: list[str],
    ) -> PostToolVerification:
        fallback = self._heuristic_post_tool_verification(
            intent=intent,
            tool_name=tool_name,
            drafted_answer=drafted_answer,
            tool_payload=tool_payload,
            missing_slots=missing_slots,
        )
        prompt = build_post_tool_verifier_prompt(
            user_message=user_message,
            intent=intent,
            tool_name=tool_name,
            drafted_answer=drafted_answer,
            tool_payload=tool_payload,
        )
        payload = await self._llm_json(prompt)
        if not payload:
            return fallback
        verified = self._parse_post_tool_verification(
            payload,
            fallback=fallback,
        )
        if verified.answer_policy == "clarify" and not str(verified.clarify_question or "").strip():
            question = clarify_question_for_slots(intent, missing_slots)
            verified.clarify_question = question
        if verified.answer_policy == "clarify" and not str(verified.clarify_type or "").strip():
            verified.clarify_type = clarify_type_for_slots(intent, missing_slots)
        if verified.answer_policy == "direct" and not verified.enough_data:
            verified.answer_policy = "clarify" if verified.should_clarify else "not_found"
        return verified

    def _heuristic_post_tool_verification(
        self,
        *,
        intent: str,
        tool_name: str,
        drafted_answer: str,
        tool_payload: dict[str, Any],
        missing_slots: list[str],
    ) -> PostToolVerification:
        return _heuristic_post_tool_verification_helper(
            intent=intent,
            tool_name=tool_name,
            drafted_answer=drafted_answer,
            tool_payload=tool_payload,
            missing_slots=missing_slots,
        )

    def _parse_post_tool_verification(
        self,
        payload: dict[str, Any],
        *,
        fallback: PostToolVerification,
    ) -> PostToolVerification:
        return _parse_post_tool_verification_helper(payload, fallback=fallback)

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
        return _merge_missing_slots_helper(primary, secondary)

    @staticmethod
    def _filter_missing_slots_by_entities(missing_slots: list[str], entities: dict[str, Any]) -> list[str]:
        return _filter_missing_slots_by_entities_helper(missing_slots, entities)

    @staticmethod
    def _merge_entities(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        return _merge_entities_helper(base, updates)

    @staticmethod
    def _copy_dialog_state(dialog_state: DialogState | None) -> DialogState:
        return _copy_dialog_state_helper(dialog_state)

    @staticmethod
    def _dialog_state_is_active(dialog_state: DialogState | None) -> bool:
        return _dialog_state_is_active_helper(dialog_state)

    @staticmethod
    def _dialog_state_payload(dialog_state: DialogState) -> dict[str, Any]:
        return _dialog_state_payload_helper(dialog_state)

    @staticmethod
    def _dialog_state_from_payload(payload: dict[str, Any] | None) -> DialogState:
        return _dialog_state_from_payload_helper(payload)

    async def _load_dialog_state(self, session_id: str) -> DialogState:
        return await _load_dialog_state_helper(self.memory, session_id)

    async def _save_dialog_state(self, session_id: str, dialog_state: DialogState) -> None:
        await _save_dialog_state_helper(self.memory, session_id, dialog_state)

    async def _clear_dialog_state(self, session_id: str) -> None:
        await _clear_dialog_state_helper(self.memory, session_id)

    async def _load_session_entity_memory(self, session_id: str) -> dict[str, Any]:
        return await _load_session_entity_memory_helper(self.memory, session_id)

    async def _save_session_entity_memory(self, session_id: str, entities: dict[str, Any]) -> None:
        await _save_session_entity_memory_helper(self.memory, session_id, entities)

    @staticmethod
    def _last_doctor_name_key() -> str:
        return _LAST_DOCTOR_NAME_KEY

    @staticmethod
    def _new_session_id() -> str:
        return _new_session_id()

    @staticmethod
    def _capabilities_brief() -> str:
        return _capabilities_brief()

    @staticmethod
    def _candidate_entities_from_entities(
        entities: dict[str, Any],
        *,
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _candidate_entities_from_entities_helper(entities, current=current)

    @staticmethod
    def _candidate_confirmation_target(
        *,
        intent: str,
        tool_plan: list[str],
        entities: dict[str, Any],
        candidate_entities: dict[str, Any],
        current_target: str = "",
    ) -> str:
        return _candidate_confirmation_target_helper(
            intent=intent,
            tool_plan=tool_plan,
            entities=entities,
            candidate_entities=candidate_entities,
            current_target=current_target,
        )

    @staticmethod
    def _candidate_confirmation_question(*, target: str, candidate_value: str) -> str:
        return _candidate_confirmation_question_helper(target=target, candidate_value=candidate_value)

    @staticmethod
    def _candidate_rejected_question(target: str) -> str:
        return _candidate_rejected_question_helper(target)

    @staticmethod
    def _promote_confirmed_candidate(
        *,
        entities: dict[str, Any],
        candidate_entities: dict[str, Any],
        target: str,
    ) -> dict[str, Any]:
        return _promote_confirmed_candidate_helper(
            entities=entities,
            candidate_entities=candidate_entities,
            target=target,
        )

    def _missing_slots_from_tool_payload(self, tool_name: str, payload: dict[str, Any]) -> list[str]:
        if tool_name != "test_result_status":
            return []
        slots_ft = payload.get("missing_slots_ft") if isinstance(payload, dict) else []
        if isinstance(slots_ft, list) and slots_ft:
            return self._merge_missing_slots(
                [],
                [str(item or "").strip().lower() for item in slots_ft if str(item or "").strip()],
            )
        raw_fields = payload.get("missing_fields") if isinstance(payload, dict) else []
        if not isinstance(raw_fields, list):
            return []
        missing: list[str] = []
        for item in raw_fields:
            text = str(item or "").strip().lower()
            if not text:
                continue
            if "фам" in text:
                missing.append("result_surname")
            elif "год" in text:
                missing.append("result_year_of_birth")
            elif "код" in text or "фили" in text:
                missing.append("result_analysis_code")
            elif "номер" in text or "заказ" in text or "order" in text:
                missing.append("result_analysis_number")
        return self._merge_missing_slots([], missing)

    def _tool_payload_memory_entities(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _tool_payload_memory_entities_helper(tool_name, payload)

    @staticmethod
    def _clarify_question_for_slots(intent: str, missing_slots: list[str]) -> str:
        return clarify_question_for_slots(intent, missing_slots)

    @staticmethod
    def _clarify_type_for_slots(intent: str, missing_slots: list[str]) -> str:
        return clarify_type_for_slots(intent, missing_slots)

    @staticmethod
    def _resolve_current_service_name(
        *,
        effective_state: DialogState,
        dialog_act: DialogAct,
        session_memory_entities: dict[str, Any],
    ) -> str:
        return _resolve_current_service_name_helper(
            effective_state=effective_state,
            dialog_act=dialog_act,
            session_memory_entities=session_memory_entities,
        )

    @staticmethod
    def _merge_turn_entities(
        *,
        user_message: str,
        intent: str,
        effective_state: DialogState,
        dialog_act: DialogAct,
        grounded_entities: dict[str, Any],
        contextual_entities: dict[str, Any],
    ) -> dict[str, Any]:
        return _merge_turn_entities_helper(
            user_message=user_message,
            intent=intent,
            effective_state=effective_state,
            dialog_act=dialog_act,
            grounded_entities=grounded_entities,
            contextual_entities=contextual_entities,
        )

    def _resolve_tool_plan(
        self,
        *,
        user_message: str,
        intent: str,
        dialog_act: DialogAct,
        effective_state: DialogState,
    ) -> tuple[str, list[str]]:
        return _resolve_tool_plan_helper(
            user_message=user_message,
            intent=intent,
            dialog_act=dialog_act,
            effective_state=effective_state,
            include_meili_tools=self.config.include_meili_tools,
        )

    @staticmethod
    def _finalize_pretool_state(
        *,
        intent: str,
        confidence: float,
        effective_state: DialogState,
        base_missing_slots: list[str],
        entities: dict[str, Any],
        tool_plan: list[str],
        remembered_doctor: str,
    ):
        return _finalize_pretool_state_helper(
            intent=intent,
            confidence=confidence,
            effective_state=effective_state,
            base_missing_slots=base_missing_slots,
            entities=entities,
            tool_plan=tool_plan,
            remembered_doctor=remembered_doctor,
            clinical_min_confidence=_CLINICAL_MIN_CONFIDENCE,
        )

    @staticmethod
    def _build_clinical_state(
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
        return _build_clinical_state_helper(
            intent=intent,
            entities=entities,
            candidate_entities=candidate_entities,
            confirmation_target=confirmation_target,
            missing_slots=missing_slots,
            clarify_type=clarify_type,
            tool_plan=tool_plan,
            confidence=confidence,
            clarify_count=clarify_count,
            last_tool=last_tool,
            phase=phase,
            open_question=open_question,
        )

    @staticmethod
    def _rejected_candidate_slots(target: str) -> list[str]:
        return _rejected_candidate_slots_helper(target)

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
        return _apply_intent_entity_policy_helper(
            user_message=user_message,
            intent=intent,
            entities=entities,
        )

    @staticmethod
    def _public_entities(entities: dict[str, Any]) -> dict[str, Any]:
        hidden_keys = {"doctor_name_match_status", "doctor_name_candidate", "service_name_candidate"}
        return {
            k: v
            for k, v in (entities or {}).items()
            if not str(k).startswith("_ft_") and str(k) not in hidden_keys
        }

    @staticmethod
    def _looks_like_specific_doctor_lookup(value: str) -> bool:
        return _looks_like_specific_doctor_lookup_helper(value)

    @staticmethod
    def _looks_like_specific_service_lookup(value: str) -> bool:
        return _looks_like_specific_service_lookup_helper(value)

    def _catalog_resolution_reply_if_needed(
        self,
        *,
        tool_plan: list[str],
        entities: dict[str, Any],
        fallback_only: bool = False,
    ) -> AgentReply | None:
        return _catalog_resolution_reply_if_needed_helper(
            tool_plan=tool_plan,
            entities=entities,
            fallback_only=fallback_only,
        )

    def _looks_like_contextual_clinical_followup(
        self,
        *,
        user_message: str,
        remembered_doctor: str,
        memory_entities: dict[str, Any],
    ) -> bool:
        return _looks_like_contextual_clinical_followup_helper(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
            memory_entities=memory_entities,
        )

    def _contextual_followup_tool_plan(
        self,
        *,
        user_message: str,
        remembered_doctor: str,
        memory_entities: dict[str, Any],
    ) -> list[str]:
        return _contextual_followup_tool_plan_helper(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
            memory_entities=memory_entities,
        )

    @staticmethod
    def _extract_contextual_entities(user_message: str) -> dict[str, Any]:
        return _extract_contextual_entities_helper(user_message)

    @staticmethod
    def _extract_service_variant(text: str) -> str:
        return _extract_service_variant_helper(text)

    @staticmethod
    def _extract_city_reference(text: str) -> str:
        return _extract_city_reference_helper(text)

    @staticmethod
    def _extract_branch_reference(text: str) -> str:
        return _extract_branch_reference_helper(text)

    @staticmethod
    def _extract_date_filters(text: str) -> dict[str, Any]:
        return _extract_date_filters_helper(text)

    @staticmethod
    def _extract_time_filters(text: str) -> dict[str, Any]:
        return _extract_time_filters_helper(text)

    async def _enrich_entities_from_session_memory(
        self,
        *,
        session_id: str,
        user_message: str,
        tool_plan: list[str],
        entities: dict[str, Any],
        dialog_state: DialogState | None = None,
        memory_entities: dict[str, Any] | None = None,
        contextual_entities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await _enrich_entities_from_session_memory_helper(
            memory=self.memory,
            session_id=session_id,
            user_message=user_message,
            tool_plan=tool_plan,
            entities=entities,
            dialog_state=dialog_state,
            memory_entities=memory_entities,
            contextual_entities=contextual_entities,
            load_session_entity_memory=_load_session_entity_memory_helper,
            dialog_state_is_active=_dialog_state_is_active_helper,
            looks_like_doctor_followup_message=_looks_like_doctor_followup_message_helper,
            looks_like_service_followup_message=_looks_like_service_followup_message_helper,
            compose_service_with_variant=_compose_service_with_variant,
        )

    async def _remember_doctor_from_tool_result(
        self,
        *,
        session_id: str,
        tool_name: str,
        payload: dict[str, Any],
    ) -> None:
        await _remember_doctor_from_tool_result_helper(
            memory=self.memory,
            session_id=session_id,
            tool_name=tool_name,
            payload=payload,
        )

    @staticmethod
    def _extract_primary_doctor_name(tool_name: str, payload: dict[str, Any]) -> str:
        return _extract_primary_doctor_name_helper(tool_name, payload)

    @staticmethod
    def _extract_schedule_memory_entities(payload: dict[str, Any]) -> dict[str, Any]:
        return _extract_schedule_memory_entities_helper(payload)

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
        return _looks_like_doctor_followup_message_helper(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
        )

    def _looks_like_service_followup_message(self, *, user_message: str, remembered_service: str) -> bool:
        return _looks_like_service_followup_message_helper(
            user_message=user_message,
            remembered_service=remembered_service,
        )

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
        return _format_iso_date_short_helper(value)

    @classmethod
    def _schedule_day_line(cls, day: dict[str, Any]) -> tuple[str, bool]:
        return _schedule_day_line_helper(day)

    @classmethod
    def _render_schedule_details(cls, payload: dict[str, Any]) -> str:
        return _render_schedule_details_helper(payload)

    @staticmethod
    def _schedule_payload_stats(payload: dict[str, Any]) -> dict[str, Any]:
        return _schedule_payload_stats_helper(payload)

    def _fallback_render(self, tool_name: str, payload: dict[str, Any]) -> str:
        return _fallback_render_helper(tool_name, payload)
