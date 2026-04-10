"""FreeTalk agent orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
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
    clarify_type_for_slots,
    clarify_question_for_slots,
    merge_missing_slots_from_plan,
    normalize_clarify_type,
    parse_clinical_decision,
)
from .config import FreeTalkConfig
from .contracts import AgentReply, DialogAct, DialogState, PostToolVerification, SessionContext
from .memory_persist import PersistentSummaryStore
from .memory_redis import RedisMemoryStore
from .observability import log_event
from .prompts import (
    build_clinical_router_prompt,
    build_general_prompt,
    build_post_tool_verifier_prompt,
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
_CLINICAL_DIALOG_STATE_KEY = "clinical_dialog_state"
_LEGACY_CLINICAL_PENDING_STATE_KEY = "clinical_pending_state"
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
    "region",
    "branch",
    "branch_name",
    "filial",
    "date_from",
    "date_to",
    "time_from",
    "time_to",
    "date",
    "time",
    "surname",
    "year",
    "number",
    "order_number",
    "order_id",
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


def _clean_user_fragment(value: str, *, max_len: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
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
                guard_fragments = self._source_fragments_for_reply(guard_reply)
                guard_reply.source_fragments = guard_fragments
                self._log_source_trace(session_id=sid, reply=guard_reply)
                await self.memory.append_exchange(
                    sid,
                    user_text=user_message,
                    assistant_text=guard_reply.text,
                    source=guard_reply.source,
                    source_fragments=guard_reply.source_fragments,
                )
                await self._maybe_compact(sid)
            return guard_reply

        dialog_state = await self._load_dialog_state(sid)
        remembered_doctor = await self.memory.get_meta_str(sid, _LAST_DOCTOR_NAME_KEY, "")
        dialog_act = await self._build_dialog_act(
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
        reply = await self._execute_dialog_act(
            user_message=user_message,
            context=context,
            dialog_act=dialog_act,
            dialog_state=dialog_state,
        )

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

        source_fragments = self._source_fragments_for_reply(reply)
        reply.source_fragments = source_fragments
        self._log_source_trace(session_id=sid, reply=reply)
        await self.memory.append_exchange(
            sid,
            user_text=user_message,
            assistant_text=reply.text,
            source=reply.source,
            source_fragments=source_fragments,
        )
        await self._maybe_compact(sid)
        return reply

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
        if active_dialog_state:
            if decision.intent == "unknown" and str(dialog_state.intent or "").strip():
                decision.intent = str(dialog_state.intent or "").strip()
            if not decision.tool_plan and dialog_state.tool_plan:
                decision.tool_plan = list(dialog_state.tool_plan)
            if dialog_state.missing_slots:
                decision.missing_slots = self._merge_missing_slots(dialog_state.missing_slots, decision.missing_slots)

        route = "general"
        fallback_reason = ""
        if decision.intent != "unknown" or decision.tool_plan or decision.missing_slots:
            route = "clinical"
        elif active_dialog_state:
            route = "clinical"
        elif web_search_signal and self.web_search is not None:
            route = "web"

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
            elif decision.confidence < _CLINICAL_MIN_CONFIDENCE:
                fallback_reason = "low_confidence"
            else:
                fallback_reason = "heuristic_fallback"
            fallback_tool_plan = select_tool_plan(
                user_message,
                include_meili_tools=self.config.include_meili_tools,
            )
            if not fallback_tool_plan and contextual_followup_hint:
                fallback_tool_plan = self._contextual_followup_tool_plan(
                    user_message=user_message,
                    remembered_doctor=remembered_doctor,
                    memory_entities=memory_entities,
                )
            fallback_intent = decision.intent
            if fallback_intent == "unknown" and fallback_tool_plan:
                fallback_intent = self._infer_intent_from_tool_plan(fallback_tool_plan)
            decision.intent = fallback_intent
            decision.tool_plan = fallback_tool_plan
            route = "clinical"
            decision.source = "heuristic_fallback"
            log_event(
                "router_fallback_applied",
                level=logging.WARNING,
                session_id=context.session_id,
                reason=fallback_reason,
                llm_intent=decision.intent,
                llm_confidence=decision.confidence,
                tool_plan=",".join(fallback_tool_plan),
            )

        log_event(
            "route_intent_evaluated",
            session_id=context.session_id,
            route=route,
            llm_intent=decision.intent,
            llm_confidence=decision.confidence,
            llm_source=decision.source,
            medical_regex=medical_regex,
            medical_fallback=medical_fallback,
            doctor_followup_hint=doctor_followup_hint,
            contextual_followup_hint=contextual_followup_hint,
            remembered_doctor=bool(str(remembered_doctor or "").strip()),
            web_search_signal=web_search_signal,
            fallback_reason=fallback_reason,
            message=user_message[:180],
        )

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

    async def _execute_dialog_act(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_act: DialogAct,
        dialog_state: DialogState,
    ) -> AgentReply:
        route = str(dialog_act.route or "").strip().lower()
        if route == "clinical":
            return await self._medical_reply(
                user_message,
                context,
                dialog_act=dialog_act,
                dialog_state=dialog_state,
            )
        if route == "web":
            await self._clear_dialog_state(context.session_id)
            return await self._web_reply(user_message, context)
        await self._clear_dialog_state(context.session_id)
        return await self._general_reply(user_message, context)

    async def _web_reply(self, user_message: str, context: SessionContext) -> AgentReply:
        if not self.web_search:
            return await self._general_reply(user_message, context)
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
        return AgentReply(
            text="В интернет-поиске по этому запросу не найдено релевантных результатов.",
            source="general_knowledge",
            tool_name="web_search",
            tool_payload=web_payload if isinstance(web_payload, dict) else {},
        )

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

    async def _medical_reply(
        self,
        user_message: str,
        context: SessionContext,
        *,
        dialog_act: DialogAct | None = None,
        dialog_state: DialogState | None = None,
    ) -> AgentReply:
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

        effective_state = self._copy_dialog_state(dialog_state)
        session_memory_entities = await self._load_session_entity_memory(context.session_id)
        if dialog_act is None:
            remembered_doctor = await self.memory.get_meta_str(context.session_id, _LAST_DOCTOR_NAME_KEY, "")
            if not self._dialog_state_is_active(effective_state):
                effective_state = await self._load_dialog_state(context.session_id)
            dialog_act = await self._build_dialog_act(
                user_message=user_message,
                context=context,
                dialog_state=effective_state,
                remembered_doctor=remembered_doctor,
            )
            if str(dialog_act.route or "").strip().lower() != "clinical":
                return await self._execute_dialog_act(
                    user_message=user_message,
                    context=context,
                    dialog_act=dialog_act,
                    dialog_state=effective_state,
                )

        contextual_entities = self._extract_contextual_entities(user_message)
        current_service_name = str(
            effective_state.entities.get("service_name")
            or effective_state.entities.get("test_name")
            or dialog_act.entities.get("service_name")
            or dialog_act.entities.get("test_name")
            or session_memory_entities.get("service_name")
            or session_memory_entities.get("test_name")
            or ""
        ).strip()
        entities = await self._ground_entities(
            user_message,
            intent_hint=dialog_act.intent,
            current_service_name=current_service_name,
        )
        merged_entities = dict(effective_state.entities or {})
        merged_entities.update(dialog_act.entities)
        merged_entities.update(entities)
        merged_entities.update(contextual_entities)
        entities = self._apply_intent_entity_policy(
            user_message=user_message,
            intent=dialog_act.intent,
            entities=merged_entities,
        )
        tool_plan = [tool for tool in (dialog_act.tool_plan or []) if isinstance(tool, str)]
        if not tool_plan:
            tool_plan = [tool for tool in (effective_state.tool_plan or []) if isinstance(tool, str)]
        if not tool_plan:
            tool_plan = select_tool_plan(user_message, include_meili_tools=self.config.include_meili_tools)
            if dialog_act.intent == "unknown" and tool_plan:
                dialog_act.intent = self._infer_intent_from_tool_plan(tool_plan)
        entities = await self._enrich_entities_from_session_memory(
            session_id=context.session_id,
            user_message=user_message,
            tool_plan=tool_plan,
            entities=entities,
            dialog_state=effective_state,
            memory_entities=session_memory_entities,
            contextual_entities=contextual_entities,
        )
        candidate_entities = self._candidate_entities_from_entities(
            entities,
            current=effective_state.candidate_entities,
        )
        confirmation_target = self._candidate_confirmation_target(
            intent=dialog_act.intent,
            tool_plan=tool_plan,
            entities=entities,
            candidate_entities=candidate_entities,
            current_target=effective_state.confirmation_target,
        )
        state_clarify_type = str(effective_state.clarify_type or "").strip()
        if str(effective_state.phase or "").strip().lower() == "confirm_candidate" and confirmation_target:
            confirm_decision = self._yes_no_decision(user_message)
            if confirm_decision == "yes":
                entities = self._promote_confirmed_candidate(
                    entities=entities,
                    candidate_entities=candidate_entities,
                    target=confirmation_target,
                )
                candidate_entities.pop(confirmation_target, None)
                effective_state.phase = ""
                effective_state.open_question = ""
                effective_state.confirmation_target = ""
            elif confirm_decision == "no":
                candidate_entities.pop(confirmation_target, None)
                clarify_slot = (
                    "doctor_name_or_specialty"
                    if confirmation_target == "doctor_name"
                    else "service_or_analysis_name"
                )
                clarify_text = self._candidate_rejected_question(confirmation_target)
                await self._save_dialog_state(
                    context.session_id,
                    DialogState(
                        route="clinical",
                        intent=dialog_act.intent,
                        entities=self._public_entities(entities),
                        candidate_entities=candidate_entities,
                        confirmation_target="",
                        missing_slots=self._merge_missing_slots([], [clarify_slot]),
                        clarify_type="identify",
                        tool_plan=tool_plan,
                        response_policy="tool_only",
                        confidence=dialog_act.confidence,
                        clarify_count=max(1, int(effective_state.clarify_count or 0)),
                        last_tool=effective_state.last_tool,
                        phase="collecting",
                        open_question=clarify_text,
                    ),
                )
                await self._save_session_entity_memory(context.session_id, entities)
                return AgentReply(text=clarify_text, source="clinic_data")

        if confirmation_target:
            confirm_text = self._candidate_confirmation_question(
                target=confirmation_target,
                candidate_value=str(candidate_entities.get(confirmation_target) or "").strip(),
            )
            if confirm_text:
                next_attempt = int(effective_state.clarify_count or 0) + 1
                await self._save_dialog_state(
                    context.session_id,
                    DialogState(
                        route="clinical",
                        intent=dialog_act.intent,
                        entities=self._public_entities(entities),
                        candidate_entities=candidate_entities,
                        confirmation_target=confirmation_target,
                        missing_slots=list(effective_state.missing_slots or dialog_act.missing_slots or []),
                        clarify_type="confirm_candidate",
                        tool_plan=tool_plan,
                        response_policy="tool_only",
                        confidence=dialog_act.confidence,
                        clarify_count=next_attempt,
                        last_tool=effective_state.last_tool,
                        phase="confirm_candidate",
                        open_question=confirm_text,
                    ),
                )
                await self._save_session_entity_memory(context.session_id, entities)
                log_event(
                    "medical_candidate_confirmation_requested",
                    session_id=context.session_id,
                    intent=dialog_act.intent,
                    target=confirmation_target,
                    candidate=str(candidate_entities.get(confirmation_target) or "")[:120],
                )
                return AgentReply(text=confirm_text, source="clinic_data")

        missing_from_plan = merge_missing_slots_from_plan(tool_plan, entities)
        missing_slots = self._merge_missing_slots(
            self._merge_missing_slots(effective_state.missing_slots, dialog_act.missing_slots),
            missing_from_plan,
        )
        missing_slots = self._filter_missing_slots_by_entities(missing_slots, entities)
        remembered_doctor = await self.memory.get_meta_str(context.session_id, _LAST_DOCTOR_NAME_KEY, "")
        if "doctor_name_or_specialty" in missing_slots and remembered_doctor:
            entities["doctor_name"] = str(remembered_doctor).strip()
            missing_slots = [slot for slot in missing_slots if slot != "doctor_name_or_specialty"]

        needs_clarification = bool(missing_slots)
        if not needs_clarification and dialog_act.confidence < _CLINICAL_MIN_CONFIDENCE and not tool_plan:
            needs_clarification = True

        log_event(
            "medical_plan_selected",
            session_id=context.session_id,
            tool_plan=",".join(tool_plan),
            entities=self._public_entities(entities),
            router_intent=dialog_act.intent,
            router_confidence=dialog_act.confidence,
            router_source=dialog_act.source,
            missing_slots=",".join(missing_slots),
            message=user_message[:160],
        )
        if needs_clarification:
            clarify_type = str(dialog_act.clarify_type or state_clarify_type or "").strip()
            if not clarify_type:
                clarify_type = clarify_type_for_slots(dialog_act.intent, missing_slots)
            clarify_text = str(dialog_act.clarify_question or "").strip()
            if not clarify_text:
                clarify_text = str(effective_state.open_question or "").strip()
            if not clarify_text:
                clarify_text = clarify_question_for_slots(dialog_act.intent, missing_slots)
            next_attempt = int(effective_state.clarify_count or 0) + 1
            await self._save_dialog_state(
                context.session_id,
                DialogState(
                    route="clinical",
                    intent=dialog_act.intent,
                    entities=self._public_entities(entities),
                    candidate_entities=candidate_entities,
                    confirmation_target="",
                    missing_slots=missing_slots,
                    clarify_type=clarify_type,
                    tool_plan=tool_plan,
                    response_policy="tool_only",
                    confidence=dialog_act.confidence,
                    clarify_count=next_attempt,
                    last_tool=effective_state.last_tool,
                    phase="collecting",
                    open_question=clarify_text,
                ),
            )
            await self._save_session_entity_memory(context.session_id, entities)
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

        await self._save_session_entity_memory(context.session_id, entities)
        await self._clear_dialog_state(context.session_id)
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
            await self._clear_dialog_state(context.session_id)
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
            verification = await self._post_tool_verify(
                user_message=user_message,
                intent=dialog_act.intent,
                tool_name=result.tool_name,
                drafted_answer=answer,
                tool_payload=result.payload,
                missing_slots=missing_slots,
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
            await self._remember_doctor_from_tool_result(
                session_id=context.session_id,
                tool_name=result.tool_name,
                payload=result.payload,
            )
            await self._save_session_entity_memory(
                context.session_id,
                self._merge_entities(
                    entities,
                    self._tool_payload_memory_entities(result.tool_name, result.payload),
                ),
            )
            if verification.answer_policy == "clarify":
                clarify_type = str(verification.clarify_type or "").strip()
                if not clarify_type:
                    clarify_type = clarify_type_for_slots(dialog_act.intent, missing_slots)
                clarify_text = str(verification.clarify_question or "").strip()
                if not clarify_text:
                    clarify_text = clarify_question_for_slots(dialog_act.intent, missing_slots)
                next_missing_slots = self._merge_missing_slots(
                    missing_slots,
                    self._missing_slots_from_tool_payload(result.tool_name, result.payload),
                )
                next_missing_slots = self._filter_missing_slots_by_entities(next_missing_slots, entities)
                await self._save_dialog_state(
                    context.session_id,
                    DialogState(
                        route="clinical",
                        intent=dialog_act.intent,
                        entities=self._public_entities(entities),
                        candidate_entities=candidate_entities,
                        confirmation_target="",
                        missing_slots=next_missing_slots,
                        clarify_type=clarify_type,
                        tool_plan=tool_plan,
                        response_policy="tool_only",
                        confidence=dialog_act.confidence,
                        clarify_count=max(1, int(effective_state.clarify_count or 0)),
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
                await self._clear_dialog_state(context.session_id)
                return AgentReply(
                    text="В данных клиники по вашему запросу ничего не найдено.",
                    source="clinic_data",
                    tool_name=result.tool_name,
                    tool_payload=result.payload,
                )
            await self._clear_dialog_state(context.session_id)
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
            await self._clear_dialog_state(context.session_id)
            return exhausted_reply

        retry_after_not_found = bool(
            tool_plan
            and str(effective_state.phase or "").strip().lower() != "post_not_found"
        )
        if retry_after_not_found:
            clarify_type = clarify_type_for_slots(dialog_act.intent, missing_slots)
            clarify_text = clarify_question_for_slots(dialog_act.intent, missing_slots)
            await self._save_dialog_state(
                context.session_id,
                DialogState(
                    route="clinical",
                    intent=dialog_act.intent,
                    entities=self._public_entities(entities),
                    candidate_entities=candidate_entities,
                    confirmation_target="",
                    missing_slots=missing_slots,
                    clarify_type=clarify_type,
                    tool_plan=tool_plan,
                    response_policy="tool_only",
                    confidence=dialog_act.confidence,
                    clarify_count=max(1, int(effective_state.clarify_count or 0)),
                    last_tool="",
                    phase="post_not_found",
                    open_question=clarify_text,
                ),
            )
            return AgentReply(
                text=f"В данных клиники по текущему запросу ничего не найдено. {clarify_text}",
                source="clinic_data",
            )

        await self._clear_dialog_state(context.session_id)
        return AgentReply(text="В данных клиники по вашему запросу ничего не найдено.", source="clinic_data")

    async def _ground_entities(
        self,
        user_message: str,
        *,
        intent_hint: str = "",
        current_service_name: str = "",
    ) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        intent = str(intent_hint or "").strip().lower()
        doctor_focus = intent in {"doctor_info", "doctor_schedule"}
        service_focus = intent in {"price", "prepare", "tests", "service_info"}
        should_try_service = service_focus or not doctor_focus

        if should_try_service:
            try:
                service_match = await self.services.match_catalog_service(
                    user_message,
                    current_service_name=str(current_service_name or "").strip(),
                )
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
                elif status == "fuzzy" and canonical:
                    entities["service_name_candidate"] = canonical
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
            if status == "exact" and canonical:
                entities["doctor_name"] = canonical
                entities["doctor_name_match_status"] = status
            elif status == "fuzzy" and canonical:
                entities["doctor_name_match_status"] = status
                entities["doctor_name_candidate"] = canonical

        return entities

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
        clarify_text = str(tool_payload.get("clarify_text") or "").strip()
        if clarify_text:
            return PostToolVerification(
                enough_data=False,
                should_clarify=True,
                clarify_type=clarify_type_for_slots(intent, missing_slots),
                clarify_question=clarify_text,
                answer_policy="clarify",
                source="heuristic",
            )

        if tool_name == "test_result_status":
            missing_fields = tool_payload.get("missing_fields") or []
            if isinstance(missing_fields, list) and missing_fields:
                question = "Чтобы проверить результат, уточните: " + ", ".join(str(x) for x in missing_fields)
                return PostToolVerification(
                    enough_data=False,
                    should_clarify=True,
                    clarify_type="missing_auth_data",
                    clarify_question=question,
                    answer_policy="clarify",
                    source="heuristic",
                )

        reason = str(tool_payload.get("schedule_unavailable_reason") or "").strip().lower()
        if tool_name == "doctors_schedule_week" and reason == "no_free_slots_2_weeks":
            return PostToolVerification(
                enough_data=True,
                should_clarify=False,
                clarify_question="",
                answer_policy="direct",
                source="heuristic",
            )

        text = str(drafted_answer or "").strip().lower()
        if not text or "не удалось корректно сформировать ответ" in text:
            question = clarify_question_for_slots(intent, missing_slots)
            return PostToolVerification(
                enough_data=False,
                should_clarify=True,
                clarify_type=clarify_type_for_slots(intent, missing_slots),
                clarify_question=question,
                answer_policy="clarify",
                source="heuristic",
            )

        return PostToolVerification(
            enough_data=True,
            should_clarify=False,
            clarify_type="",
            clarify_question="",
            answer_policy="direct",
            source="heuristic",
        )

    def _parse_post_tool_verification(
        self,
        payload: dict[str, Any],
        *,
        fallback: PostToolVerification,
    ) -> PostToolVerification:
        if not isinstance(payload, dict):
            return fallback

        enough_data = self._coerce_bool(payload.get("enough_data"), fallback.enough_data)
        should_clarify = self._coerce_bool(payload.get("should_clarify"), fallback.should_clarify)
        clarify_type = normalize_clarify_type(payload.get("clarify_type"))
        clarify_question = str(payload.get("clarify_question") or "").strip()
        answer_policy = str(payload.get("answer_policy") or "").strip().lower()
        if answer_policy not in {"direct", "clarify", "not_found"}:
            if should_clarify:
                answer_policy = "clarify"
            elif enough_data:
                answer_policy = "direct"
            else:
                answer_policy = "not_found"

        if should_clarify and answer_policy != "clarify":
            answer_policy = "clarify"
        if answer_policy == "clarify" and not clarify_question:
            clarify_question = fallback.clarify_question
        if answer_policy == "clarify" and not clarify_type:
            clarify_type = fallback.clarify_type
        if answer_policy != "clarify":
            clarify_type = ""

        return PostToolVerification(
            enough_data=enough_data,
            should_clarify=should_clarify,
            clarify_type=clarify_type,
            clarify_question=clarify_question,
            answer_policy=answer_policy,
            source="llm",
        )

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return bool(default)

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
        for bucket in (primary or [], secondary or []):
            values = bucket if isinstance(bucket, list) else [bucket]
            for source in values:
                slot = str(source or "").strip().lower()
                if not slot or slot in seen:
                    continue
                seen.add(slot)
                out.append(slot)
        return out

    @staticmethod
    def _filter_missing_slots_by_entities(missing_slots: list[str], entities: dict[str, Any]) -> list[str]:
        out: list[str] = []
        doctor_known = bool(str(entities.get("doctor_name") or entities.get("doctor_id") or "").strip())
        specialty_known = bool(str(entities.get("specialty") or "").strip())
        service_known = bool(
            str(
                entities.get("service_name")
                or entities.get("test_name")
                or entities.get("service_variant")
                or ""
            ).strip()
        )
        result_surname = bool(str(entities.get("surname") or "").strip())
        result_year = bool(str(entities.get("year") or "").strip())
        result_filial = bool(
            str(
                entities.get("filial")
                or entities.get("branch_name")
                or entities.get("branch")
                or entities.get("region")
                or entities.get("city")
                or ""
            ).strip()
        )
        result_number = bool(
            str(
                entities.get("number")
                or entities.get("order_number")
                or entities.get("order_id")
                or ""
            ).strip()
        )
        for slot in (missing_slots or []):
            name = str(slot or "").strip().lower()
            if not name:
                continue
            if name == "doctor_name_or_specialty" and (doctor_known or specialty_known):
                continue
            if name == "service_or_analysis_name" and (service_known or doctor_known):
                continue
            if name == "result_surname" and result_surname:
                continue
            if name == "result_year" and result_year:
                continue
            if name == "result_filial" and result_filial:
                continue
            if name == "result_number" and result_number:
                continue
            if name == "result_identifiers" and (result_surname and result_year and result_filial and result_number):
                continue
            out.append(name)
        return out

    @staticmethod
    def _merge_entities(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        out = dict(base or {})
        for key, value in (updates or {}).items():
            name = str(key or "").strip()
            if not name:
                continue
            text = str(value or "").strip()
            if not text:
                continue
            out[name] = value
        return out

    @staticmethod
    def _copy_dialog_state(dialog_state: DialogState | None) -> DialogState:
        if not isinstance(dialog_state, DialogState):
            return DialogState()
        return DialogState(
            route=str(dialog_state.route or ""),
            intent=str(dialog_state.intent or ""),
            entities=dict(dialog_state.entities or {}),
            candidate_entities=dict(dialog_state.candidate_entities or {}),
            confirmation_target=str(dialog_state.confirmation_target or ""),
            missing_slots=list(dialog_state.missing_slots or []),
            clarify_type=normalize_clarify_type(dialog_state.clarify_type),
            tool_plan=list(dialog_state.tool_plan or []),
            response_policy=str(dialog_state.response_policy or ""),
            confidence=float(dialog_state.confidence or 0.0),
            clarify_count=int(dialog_state.clarify_count or 0),
            last_tool=str(dialog_state.last_tool or ""),
            phase=str(dialog_state.phase or ""),
            open_question=str(dialog_state.open_question or ""),
        )

    @staticmethod
    def _dialog_state_is_active(dialog_state: DialogState | None) -> bool:
        if not isinstance(dialog_state, DialogState):
            return False
        intent = str(dialog_state.intent or "").strip().lower()
        return bool(
            (intent and intent != "unknown")
            or dialog_state.entities
            or dialog_state.missing_slots
            or str(dialog_state.clarify_type or "").strip()
            or dialog_state.tool_plan
            or str(dialog_state.confirmation_target or "").strip()
            or str(dialog_state.phase or "").strip()
        )

    @staticmethod
    def _dialog_state_payload(dialog_state: DialogState) -> dict[str, Any]:
        payload = asdict(dialog_state) if isinstance(dialog_state, DialogState) else {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _dialog_state_from_payload(payload: dict[str, Any] | None) -> DialogState:
        data = payload if isinstance(payload, dict) else {}
        entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
        candidate_entities = (
            data.get("candidate_entities")
            if isinstance(data.get("candidate_entities"), dict)
            else {}
        )
        missing_slots = data.get("missing_slots") if isinstance(data.get("missing_slots"), list) else []
        tool_plan = data.get("tool_plan") if isinstance(data.get("tool_plan"), list) else []
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        try:
            clarify_count = int(data.get("clarify_count", 0))
        except Exception:
            clarify_count = 0
        return DialogState(
            route=str(data.get("route") or "").strip() or "general",
            intent=str(data.get("intent") or "").strip() or "unknown",
            entities={str(k): v for k, v in entities.items() if str(k).strip() and str(v or "").strip()},
            candidate_entities={
                str(k): v for k, v in candidate_entities.items() if str(k).strip() and str(v or "").strip()
            },
            confirmation_target=str(data.get("confirmation_target") or "").strip(),
            missing_slots=[str(x).strip() for x in missing_slots if str(x).strip()],
            clarify_type=normalize_clarify_type(data.get("clarify_type")),
            tool_plan=[str(x).strip() for x in tool_plan if str(x).strip()],
            response_policy=str(data.get("response_policy") or "").strip() or "general_only",
            confidence=max(0.0, min(1.0, confidence)),
            clarify_count=max(0, clarify_count),
            last_tool=str(data.get("last_tool") or "").strip(),
            phase=str(data.get("phase") or "").strip(),
            open_question=str(data.get("open_question") or "").strip(),
        )

    async def _load_dialog_state(self, session_id: str) -> DialogState:
        if self.memory is None:
            return DialogState()
        raw = await self.memory.get_meta_str(session_id, _CLINICAL_DIALOG_STATE_KEY, "")
        text = str(raw or "").strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = {}
            state = self._dialog_state_from_payload(parsed if isinstance(parsed, dict) else {})
            if self._dialog_state_is_active(state):
                return state

        legacy_raw = await self.memory.get_meta_str(session_id, _LEGACY_CLINICAL_PENDING_STATE_KEY, "")
        legacy_text = str(legacy_raw or "").strip()
        if not legacy_text:
            return DialogState()
        try:
            legacy = json.loads(legacy_text)
        except Exception:
            legacy = {}
        if not isinstance(legacy, dict):
            return DialogState()
        return DialogState(
            route="clinical",
            intent=str(legacy.get("intent") or "").strip() or "unknown",
            entities={},
            candidate_entities={},
            confirmation_target="",
            missing_slots=[str(x).strip() for x in (legacy.get("missing_slots") or []) if str(x).strip()],
            clarify_type="",
            tool_plan=[],
            response_policy="tool_only",
            confidence=0.0,
            clarify_count=max(0, int(legacy.get("attempts") or 0)),
            last_tool="",
            phase=str(legacy.get("phase") or "").strip(),
            open_question=str(legacy.get("clarify_question") or "").strip(),
        )

    async def _save_dialog_state(self, session_id: str, dialog_state: DialogState) -> None:
        if self.memory is None:
            return
        payload = self._dialog_state_payload(dialog_state)
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except Exception:
            encoded = "{}"
        await self.memory.set_meta_str(session_id, _CLINICAL_DIALOG_STATE_KEY, encoded)
        await self.memory.set_meta_str(session_id, _LEGACY_CLINICAL_PENDING_STATE_KEY, "")

    async def _clear_dialog_state(self, session_id: str) -> None:
        if self.memory is None:
            return
        await self.memory.set_meta_str(session_id, _CLINICAL_DIALOG_STATE_KEY, "")
        await self.memory.set_meta_str(session_id, _LEGACY_CLINICAL_PENDING_STATE_KEY, "")

    async def _load_session_entity_memory(self, session_id: str) -> dict[str, Any]:
        if self.memory is None:
            return {}
        raw = await self.memory.get_meta_str(session_id, _CLINICAL_ENTITY_MEMORY_KEY, "")
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): v for k, v in parsed.items() if str(k).strip() and str(v or "").strip()}

    async def _save_session_entity_memory(self, session_id: str, entities: dict[str, Any]) -> None:
        if self.memory is None:
            return
        current = await self._load_session_entity_memory(session_id)
        updates: dict[str, Any] = {}
        for key in _SESSION_MEMORY_ENTITY_KEYS:
            value = entities.get(key)
            if not str(value or "").strip():
                continue
            updates[key] = value
        merged = self._merge_entities(current, updates)
        if not merged:
            return
        try:
            encoded = json.dumps(merged, ensure_ascii=False)
        except Exception:
            encoded = "{}"
        await self.memory.set_meta_str(session_id, _CLINICAL_ENTITY_MEMORY_KEY, encoded)

    @staticmethod
    def _candidate_entities_from_entities(
        entities: dict[str, Any],
        *,
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = dict(current or {})
        service_candidate = str(entities.get("service_name_candidate") or "").strip()
        doctor_candidate = str(entities.get("doctor_name_candidate") or "").strip()
        if service_candidate:
            out["service_name"] = service_candidate
        if doctor_candidate:
            out["doctor_name"] = doctor_candidate
        return out

    @staticmethod
    def _candidate_confirmation_target(
        *,
        intent: str,
        tool_plan: list[str],
        entities: dict[str, Any],
        candidate_entities: dict[str, Any],
        current_target: str = "",
    ) -> str:
        target = str(current_target or "").strip()
        if target and str(candidate_entities.get(target) or "").strip():
            if target == "doctor_name" and not str(entities.get("doctor_name") or "").strip():
                return target
            if target == "service_name" and not str(entities.get("service_name") or "").strip():
                return target

        plan = set(tool_plan or [])
        route_intent = str(intent or "").strip().lower()
        service_candidate = str(candidate_entities.get("service_name") or "").strip()
        doctor_candidate = str(candidate_entities.get("doctor_name") or "").strip()
        service_exact = bool(str(entities.get("service_name") or "").strip())
        doctor_exact = bool(str(entities.get("doctor_name") or "").strip())

        if (
            service_candidate
            and not service_exact
            and (
                route_intent in {"price", "prepare", "tests", "service_info", "address"}
                or bool({"price_info", "service_bundle_info", "test_prepare", "test_assist", "address_info"} & plan)
            )
        ):
            return "service_name"
        if (
            doctor_candidate
            and not doctor_exact
            and (
                route_intent in {"doctor_info", "doctor_schedule"}
                or bool({"doctors_info", "doctors_schedule_week"} & plan)
            )
        ):
            return "doctor_name"
        return ""

    @staticmethod
    def _candidate_confirmation_question(*, target: str, candidate_value: str) -> str:
        label = str(candidate_value or "").strip()
        if not label:
            return ""
        if target == "doctor_name":
            return f"Правильно понял, что нужен врач «{label}»? Ответьте: Да или Нет."
        if target == "service_name":
            return f"Правильно понял, что нужна услуга «{label}»? Ответьте: Да или Нет."
        return ""

    @staticmethod
    def _candidate_rejected_question(target: str) -> str:
        if target == "doctor_name":
            return "Хорошо. Уточните, пожалуйста, фамилию врача или специальность."
        if target == "service_name":
            return "Хорошо. Уточните, пожалуйста, точное название услуги или анализа."
        return "Хорошо. Уточните запрос чуть подробнее."

    @staticmethod
    def _promote_confirmed_candidate(
        *,
        entities: dict[str, Any],
        candidate_entities: dict[str, Any],
        target: str,
    ) -> dict[str, Any]:
        out = dict(entities or {})
        value = str(candidate_entities.get(target) or "").strip()
        if not value:
            return out
        out[target] = value
        if target == "doctor_name":
            out["doctor_name_match_status"] = "fuzzy_confirmed"
        if target == "service_name":
            out["_ft_service_match_status"] = "fuzzy_confirmed"
        return out

    def _missing_slots_from_tool_payload(self, tool_name: str, payload: dict[str, Any]) -> list[str]:
        if tool_name != "test_result_status":
            return []
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
                missing.append("result_year")
            elif "фили" in text:
                missing.append("result_filial")
            elif "номер" in text or "заказ" in text or "order" in text:
                missing.append("result_number")
        return self._merge_missing_slots([], missing)

    def _tool_payload_memory_entities(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        doctor_name = self._extract_primary_doctor_name(tool_name, payload)
        if doctor_name:
            out["doctor_name"] = doctor_name
        if tool_name in {"price_info", "service_bundle_info", "test_prepare", "test_assist"}:
            entities_used = payload.get("entities_used") if isinstance(payload, dict) else {}
            if isinstance(entities_used, dict):
                service_name = str(
                    entities_used.get("service_name_effective")
                    or entities_used.get("service_name")
                    or ""
                ).strip()
                if service_name:
                    out["service_name"] = service_name
        if tool_name == "doctors_schedule_week":
            out.update(self._extract_schedule_memory_entities(payload))
        if tool_name == "address_info":
            addresses = [str(x).strip() for x in (payload.get("addresses") or []) if str(x).strip()]
            if addresses:
                out["branch_name"] = addresses[0]
        return out

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
        hidden_keys = {"doctor_name_match_status", "doctor_name_candidate", "service_name_candidate"}
        return {
            k: v
            for k, v in (entities or {}).items()
            if not str(k).startswith("_ft_") and str(k) not in hidden_keys
        }

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

    def _looks_like_contextual_clinical_followup(
        self,
        *,
        user_message: str,
        remembered_doctor: str,
        memory_entities: dict[str, Any],
    ) -> bool:
        if not memory_entities and not str(remembered_doctor or "").strip():
            return False
        if self._extract_contextual_entities(user_message):
            return True
        remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
        if remembered_service and self._looks_like_service_followup_message(
            user_message=user_message,
            remembered_service=remembered_service,
        ):
            return True
        return self._looks_like_doctor_followup_message(
            user_message=user_message,
            remembered_doctor=remembered_doctor,
        )

    def _contextual_followup_tool_plan(
        self,
        *,
        user_message: str,
        remembered_doctor: str,
        memory_entities: dict[str, Any],
    ) -> list[str]:
        contextual_entities = self._extract_contextual_entities(user_message)
        has_schedule_filters = bool(
            {"branch", "branch_name", "city", "region", "date", "date_from", "date_to", "time", "time_from", "time_to"}
            & set(contextual_entities.keys())
        )
        remembered_doctor = str(remembered_doctor or memory_entities.get("doctor_name") or "").strip()
        remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
        if remembered_doctor and has_schedule_filters:
            return ["doctors_schedule_week", "doctors_info"]
        if remembered_service:
            if "branch" in contextual_entities or "branch_name" in contextual_entities:
                return ["address_info", "service_bundle_info"]
            if "service_variant" in contextual_entities:
                return ["service_bundle_info", "price_info", "address_info"]
        return []

    @staticmethod
    def _extract_contextual_entities(user_message: str) -> dict[str, Any]:
        text = str(user_message or "").strip()
        if not text:
            return {}
        out: dict[str, Any] = {}
        variant = FreeTalkAgent._extract_service_variant(text)
        if variant:
            out["service_variant"] = variant
        branch = FreeTalkAgent._extract_branch_reference(text)
        if branch:
            out["branch"] = branch
        city = FreeTalkAgent._extract_city_reference(text)
        if city:
            out["city"] = city
            out["region"] = city
        out.update(FreeTalkAgent._extract_date_filters(text))
        out.update(FreeTalkAgent._extract_time_filters(text))
        return out

    @staticmethod
    def _extract_service_variant(text: str) -> str:
        match = _SERVICE_VARIANT_RE.search(str(text or ""))
        return str(match.group(1) or "").strip().lower() if match else ""

    @staticmethod
    def _extract_city_reference(text: str) -> str:
        if re.search(r"\bсамар\w*\b", str(text or ""), re.I):
            return "Самара"
        return ""

    @staticmethod
    def _extract_branch_reference(text: str) -> str:
        def _clean_branch_candidate(value: str) -> str:
            candidate = str(value or "").strip(" ,.")
            if not candidate:
                return ""
            candidate = re.split(
                r"\b(утром|дн[её]м|вечером|сегодня|завтра|послезавтра|после\s+\d{1,2}(?::\d{2})?|до\s+\d{1,2}(?::\d{2})?|\d{1,2}:\d{2})\b",
                candidate,
                maxsplit=1,
                flags=re.I,
            )[0].strip(" ,.")
            return candidate

        source = str(text or "").strip()
        if not source:
            return ""
        lowered = source.lower()
        if "другом филиале" in lowered or "другой филиал" in lowered:
            return "другой филиал"
        explicit = _BRANCH_CAPTURE_RE.search(source)
        if explicit:
            candidate = _clean_branch_candidate(str(explicit.group(1) or ""))
            if candidate:
                return candidate
        short = _SHORT_BRANCH_RE.match(source)
        if not short:
            return ""
        candidate = _clean_branch_candidate(str(short.group(1) or ""))
        if not candidate:
            return ""
        low_candidate = candidate.lower()
        if low_candidate in _BRANCH_FOLLOWUP_STOPWORDS:
            return ""
        if _DATE_FILTER_RE.search(candidate) or _TIME_FILTER_RE.search(candidate):
            return ""
        if len(candidate.split()) > 5:
            return ""
        return candidate

    @staticmethod
    def _extract_date_filters(text: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        source = str(text or "").strip()
        if not source:
            return out
        today = datetime.now().date()
        lowered = source.lower()
        if "послезавтра" in lowered:
            target = today + timedelta(days=2)
            iso = target.isoformat()
            return {"date": iso, "date_from": iso, "date_to": iso}
        if "завтра" in lowered:
            target = today + timedelta(days=1)
            iso = target.isoformat()
            return {"date": iso, "date_from": iso, "date_to": iso}
        if "сегодня" in lowered:
            iso = today.isoformat()
            return {"date": iso, "date_from": iso, "date_to": iso}
        if "на следующей неделе" in lowered or "на следующую неделю" in lowered:
            return {"date": "next_week"}
        if "на этой неделе" in lowered:
            return {"date": "this_week"}
        iso_match = _DATE_ISO_RE.search(source)
        if iso_match:
            iso = f"{int(iso_match.group(1)):04d}-{int(iso_match.group(2)):02d}-{int(iso_match.group(3)):02d}"
            return {"date": iso, "date_from": iso, "date_to": iso}
        dot_match = _DATE_DOT_RE.search(source)
        if dot_match:
            day = int(dot_match.group(1))
            month = int(dot_match.group(2))
            year_raw = str(dot_match.group(3) or "").strip()
            year = today.year
            if year_raw:
                year = int(year_raw)
                if year < 100:
                    year += 2000
            try:
                target = date(year, month, day)
            except Exception:
                return out
            if not year_raw and target < today:
                try:
                    target = date(year + 1, month, day)
                except Exception:
                    return out
            iso = target.isoformat()
            return {"date": iso, "date_from": iso, "date_to": iso}
        return out

    @staticmethod
    def _extract_time_filters(text: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        source = str(text or "").strip()
        if not source:
            return out
        lowered = source.lower()
        if "утром" in lowered:
            out["time"] = "утром"
            out["time_from"] = "08:00"
            out["time_to"] = "12:00"
        elif "днем" in lowered or "днём" in lowered:
            out["time"] = "днем"
            out["time_from"] = "12:00"
            out["time_to"] = "17:00"
        elif "вечером" in lowered:
            out["time"] = "вечером"
            out["time_from"] = "17:00"
            out["time_to"] = "21:00"
        after_match = _TIME_AFTER_RE.search(source)
        if after_match:
            minutes = int(after_match.group(2) or 0)
            out["time_from"] = f"{int(after_match.group(1)):02d}:{minutes:02d}"
            out["time"] = out["time_from"]
        before_match = _TIME_BEFORE_RE.search(source)
        if before_match:
            minutes = int(before_match.group(2) or 0)
            out["time_to"] = f"{int(before_match.group(1)):02d}:{minutes:02d}"
        exact_match = _TIME_HHMM_RE.search(source)
        if exact_match:
            exact = f"{int(exact_match.group(1)):02d}:{int(exact_match.group(2)):02d}"
            out["time"] = exact
            out["time_from"] = exact
        return out

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
        out = dict(entities or {})
        memory_entities = dict(memory_entities or {})
        if not memory_entities:
            memory_entities = await self._load_session_entity_memory(session_id)
        contextual_entities = dict(contextual_entities or {})
        active_state = self._dialog_state_is_active(dialog_state)
        enriched_keys: list[str] = []

        plan = set(tool_plan or [])
        uses_doctor_tools = bool({"doctors_schedule_week", "doctors_info"} & plan)
        uses_service_tools = bool({"price_info", "service_bundle_info", "test_prepare", "test_assist"} & plan)
        uses_address_tools = "address_info" in plan
        uses_result_tool = "test_result_status" in plan
        has_contextual_entities = bool(contextual_entities)

        if uses_doctor_tools and not str(out.get("doctor_name") or "").strip():
            remembered = await self.memory.get_meta_str(session_id, _LAST_DOCTOR_NAME_KEY, "")
            remembered = str(remembered or memory_entities.get("doctor_name") or "").strip()
            if remembered and (
                active_state
                or has_contextual_entities
                or _DOCTOR_ANAPHORA_RE.search(str(user_message or ""))
                or self._looks_like_doctor_followup_message(
                    user_message=user_message,
                    remembered_doctor=remembered,
                )
            ):
                out["doctor_name"] = remembered
                out["doctor_name_source"] = "session_memory"
                enriched_keys.append("doctor_name")

        remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
        if (
            (uses_service_tools or uses_address_tools)
            and not str(out.get("service_name") or out.get("test_name") or "").strip()
        ):
            if remembered_service and (
                active_state
                or has_contextual_entities
                or self._looks_like_service_followup_message(
                    user_message=user_message,
                    remembered_service=remembered_service,
                )
            ):
                out["service_name"] = remembered_service
                enriched_keys.append("service_name")
        if remembered_service and str(out.get("service_variant") or "").strip():
            combined = _compose_service_with_variant(
                str(out.get("service_name") or remembered_service),
                str(out.get("service_variant") or ""),
            )
            if combined and combined != str(out.get("service_name") or "").strip():
                out["service_name"] = combined
                enriched_keys.append("service_name")

        if uses_result_tool:
            for key in ("surname", "year", "filial", "number", "order_number", "order_id"):
                if str(out.get(key) or "").strip():
                    continue
                value = str(memory_entities.get(key) or "").strip()
                if not value:
                    continue
                out[key] = value
                enriched_keys.append(key)

        if active_state or has_contextual_entities:
            for key in ("branch_name", "branch", "region", "city", "date", "date_from", "date_to", "time", "time_from", "time_to"):
                if str(out.get(key) or "").strip():
                    continue
                value = str(memory_entities.get(key) or "").strip()
                if not value:
                    continue
                out[key] = value
                enriched_keys.append(key)

        if enriched_keys:
            log_event(
                "medical_entities_enriched_from_memory",
                session_id=session_id,
                keys=",".join(sorted(set(enriched_keys))),
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
        if self.memory is None:
            return
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

    @staticmethod
    def _extract_schedule_memory_entities(payload: dict[str, Any]) -> dict[str, Any]:
        schedule = payload.get("schedule") if isinstance(payload, dict) else []
        if not isinstance(schedule, list):
            return {}
        out: dict[str, Any] = {}
        for row in schedule:
            if not isinstance(row, dict):
                continue
            row_schedule = row.get("schedule")
            if not isinstance(row_schedule, dict):
                continue
            for region_name, days in row_schedule.items():
                region = str(region_name or "").strip()
                if region:
                    out["branch_name"] = region
                    out["branch"] = region
                if not isinstance(days, list):
                    return out
                for day in days:
                    if not isinstance(day, dict):
                        continue
                    date_value = str(day.get("date") or "").strip()
                    if date_value:
                        out["date"] = date_value
                        out["date_from"] = date_value
                        out["date_to"] = date_value
                    slots = [str(x).strip() for x in (day.get("slots") or []) if str(x).strip()]
                    if slots:
                        out["time"] = slots[0][:5]
                        out["time_from"] = slots[0][:5]
                    else:
                        start = str(day.get("start") or "").strip()
                        end = str(day.get("end") or "").strip()
                        if start:
                            out["time_from"] = start[:5]
                        if end:
                            out["time_to"] = end[:5]
                    return out
                return out
        return out

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

    def _looks_like_service_followup_message(self, *, user_message: str, remembered_service: str) -> bool:
        text = str(user_message or "").strip()
        if not text:
            return False
        remembered = str(remembered_service or "").strip()
        if not remembered:
            return False
        if _SERVICE_FOLLOWUP_RE.search(text):
            return True
        if _SERVICE_VARIANT_RE.search(text):
            return True
        if self._extract_contextual_entities(text):
            return True
        lowered = text.lower().replace("ё", "е")
        remembered_parts = [part.strip().lower().replace("ё", "е") for part in remembered.split() if part.strip()]
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
