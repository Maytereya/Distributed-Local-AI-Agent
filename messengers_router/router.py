"""Главный оркестратор мессенджерного диалога.

Выполняет pipeline: classify -> slot filling -> plan -> services -> rendering,
ведет APPOINTMENT flow, pending-уточнения, handoff решения и debug meta.

Ответственность модуля:
1) Координировать этапы пайплайна (NLU/graph/recovery/plan/services/render).
2) Обновлять session state и pending-слоты без дублирования бизнес-логики.
3) Сохранять API-совместимость ответа (`text`, `attachments`, `handoff`, `state_update`).

Модуль не должен превращаться в "ручной классификатор": интент-детекторы,
recovery-политика и сервисные интеграции вынесены в отдельные компоненты.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, Any

from agent_logic_2 import config as c

from .mess_types import Evidence, Plan, PlanStep, ResponseEnvelope, RouteDecision, SessionState
from .classifier import analyze
from .context_summary import update_summary
from .dialog_graph import GraphEngine
from .entity_grounder import ground_decision_entities
from .flow_policy import (
    _apply_context_action,
    _apply_pending_override,
    _fill_date_from_schedule_windows,
    _get_secondary_queue,
    _hydrate_appointment_context_from_schedule,
    _is_appointment_waiting_patient_name,
    _looks_like_patient_fio,
    _normalize_secondary_labels,
    _safe_get_branches,
    _secondary_followup_text,
    _set_secondary_queue,
    quick_fill_entities_from_text,
)
from .nlu_pipeline import analyze_with_candidates
from .llm_mode_policy import RuntimeOptions
from .policies import (
    require_auth_for_test_result,
    missing_slots,
    clarification_question,
    evidence_requires_handoff,
    handoff_message,
    appointment_step_policy,
    APPOINTMENT_STEP_BRANCH,
    APPOINTMENT_STEP_DATETIME,
    APPOINTMENT_STEP_PATIENT,
    APPOINTMENT_STEP_CONFIRM,
    APPOINTMENT_CONFIRM_YES,
    APPOINTMENT_CONFIRM_NO,
    appointment_confirmation_transition,
    extract_price_rub,
    appointment_summary,
    appointment_addresses_for_city,
    appointment_text_branch_prompt,
    appointment_text_datetime_prompt,
    appointment_text_patient_name_prompt,
    appointment_text_confirm_prompt,
    appointment_text_confirmed_handoff,
    appointment_text_reask_datetime,
    appointment_text_reask_confirm,
    decision_handoff_text,
    apply_verified_doctor_override,
    detect_nonbookable_walkin_intent,
    detect_test_assist_intent,
    detect_test_result_intent,
    detect_prepare_intent,
    detect_price_intent,
    detect_address_intent,
    detect_doc_request_intent,
    detect_schedule_intent,
    detect_doctor_info_intent,
    has_datetime_signal,
    looks_like_branch_hint,
    nonbookable_service_hint,
    appointment_service_display,
)
from .recovery_policy import contextual_reply_kind, evaluate_recovery, explicit_operator_requested
from .services import Services
from .renderer import (
    render_urgent,
    render_complaint,
    render_medical_advice,
    render_stream,
    format_service_bundle_for_patient,
    format_price_for_patient,
    format_doctor_schedule_for_patient,
    format_doctor_info_for_patient,
    format_address_for_patient,
    format_news_for_patient,
)
from .memory import MemoryStore
from .text_templates import INTRO_TEXT, LOW_CONF_CLARIFY_TEXT
from .city import match_city

_DEFAULT_CITY = "Самара"
_SAMARA_ONLY_OPERATOR_TEXT = "Сейчас могу помочь только по Самаре. Соединяю с оператором."
_GRAPH_ENGINE = GraphEngine()
_DOCTOR_NOISE_TOKENS = {
    "хочу",
    "нужно",
    "надо",
    "можно",
    "запись",
    "записаться",
    "прием",
    "приём",
    "подскажите",
    "скажите",
    "когда",
    "где",
    "да",
    "нет",
}
_PATIENT_NAME_FRAGMENT_RE = re.compile(r"^\s*[А-ЯЁа-яё\-]{2,}\s+[А-ЯЁа-яё\-]{1,}\s*$")


def _reset_appointment_state_flags(state: SessionState) -> None:
    for k in ("appointment_flow_active", "appointment_confirm_pending", "appointment_confirmed"):
        state.last_entities.pop(k, None)


def _env_flag(name: str, default: bool) -> bool:
    cfg_flags = {
        "MR_ROUTER_V2_ENABLE": c.MR_ROUTER_V2_ENABLE,
        "MR_ROUTER_V2_SHADOW": c.MR_ROUTER_V2_SHADOW,
        "MR_NLU_SHADOW": c.MR_NLU_SHADOW,
    }
    raw = cfg_flags.get(name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _copy_decision(decision: RouteDecision, **overrides: Any) -> RouteDecision:
    data = {
        "label": decision.label,
        "confidence": decision.confidence,
        "entities": dict(decision.entities),
        "flags": set(decision.flags),
        "needs_handoff": decision.needs_handoff,
        "context_action": decision.context_action,
        "source": decision.source,
        "clarify_needed": decision.clarify_needed,
        "clarify_reason": decision.clarify_reason,
        "clarify_slots": list(decision.clarify_slots),
        "intent_candidates": list(decision.intent_candidates),
    }
    data.update(overrides)
    return RouteDecision(**data)


def _remember_question(state: SessionState, kind: str, slots: list[str] | None = None) -> None:
    state.last_entities["last_question_kind"] = str(kind or "").strip()
    clean_slots = [str(x).strip() for x in (slots or []) if str(x).strip()]
    if clean_slots:
        state.last_entities["last_clarify_slots"] = clean_slots[:8]
    else:
        state.last_entities.pop("last_clarify_slots", None)


def _extract_nlu_trace(evidence: Evidence) -> dict[str, Any]:
    for item in reversed(evidence.debug_trace or []):
        if not isinstance(item, dict):
            continue
        nlu = item.get("nlu")
        if not isinstance(nlu, dict):
            continue
        trace = nlu.get("nlu_trace") or nlu.get("trace")
        if isinstance(trace, dict):
            return trace
    return {}


def _is_samara_city(city: str | None) -> bool:
    if not city:
        return False
    return str(city).strip().lower().replace("ё", "е") == "самара"


def _should_keep_appointment_flow_override(user_text: str) -> bool:
    """
    Разрешаем мягкий OTHER->APPOINTMENT override только для реплик,
    похожих на продолжение сценария записи (время/город/филиал/ФИО/да-нет).
    Новые темы ("анализы", "результаты", "расписание") не должны
    притягиваться назад в активный APPOINTMENT flow.
    """
    text = str(user_text or "").strip()
    if not text:
        return False

    low = text.lower()
    if (
        detect_test_result_intent(low)
        or detect_test_assist_intent(low)
        or detect_prepare_intent(low)
        or detect_price_intent(low)
        or detect_address_intent(low)
        or detect_doc_request_intent(low)
        or detect_nonbookable_walkin_intent(text)
        or detect_schedule_intent(low)
        or detect_doctor_info_intent(low)
    ):
        return False

    reply_kind = contextual_reply_kind(text)
    if reply_kind in {"yes", "no"}:
        return True
    if has_datetime_signal(text):
        return True
    if match_city(text):
        return True
    if looks_like_branch_hint(text):
        return True
    if _looks_like_patient_fio(text):
        return True
    # На шаге ввода ФИО допускаем "Фамилия И" как продолжение потока записи.
    if _PATIENT_NAME_FRAGMENT_RE.fullmatch(text):
        return True
    return False


def _is_new_topic_while_confirm_pending(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text:
        return False
    if contextual_reply_kind(text) in {"yes", "no"}:
        return False

    low = text.lower()
    if (
        detect_prepare_intent(low)
        or detect_price_intent(low)
        or detect_test_assist_intent(low)
        or detect_test_result_intent(low)
        or detect_address_intent(low)
        or detect_schedule_intent(low)
        or detect_doctor_info_intent(low)
        or detect_doc_request_intent(low)
        or detect_nonbookable_walkin_intent(text)
    ):
        return True

    # Длинная вопросительная реплика с высокой вероятностью новая тема.
    return ("?" in text) and (len(text.split()) >= 4)


def _early_debug_state_update(
    debug: bool,
    *,
    label: str,
    handoff: bool,
    flags: set[str] | None = None,
    context_action: str = "continue",
    confidence: float = 0.95,
) -> dict[str, Any]:
    if not debug:
        return {}
    return {
        "debug": {
            "decision": {
                "label": label,
                "confidence": confidence,
                "context_action": context_action,
                "source": "router_precheck",
                "flags": sorted(list(flags or set())),
                "entities": {},
                "needs_handoff": handoff,
                "clarify_needed": False,
                "clarify_reason": "",
                "clarify_slots": [],
                "intent_candidates": [],
            },
            "plan": {"label": label, "steps": []},
            "evidence": {"items": {}, "debug_trace": [{"router_precheck": {"reason": context_action}}]},
        }
    }


async def _verify_doctor_entity(
    decision: RouteDecision,
    services: Services,
    user_text: str,
) -> RouteDecision:
    """
    Подтверждает doctor_name только через кэш врачей.
    Если совпадения нет — doctor_name удаляется из entities.
    """
    entities = dict(decision.entities)
    flags = set(decision.flags)
    # Верифицируем doctor_name в любом label, если поле присутствует.
    # Это защищает flow-override OTHER -> APPOINTMENT от ложного "врача"
    # из ФИО пациента (например: "12 февраля 09:00 Тен Максим Александрович").
    needs_doctor_verification = bool(entities.get("doctor_name")) or decision.label in {
        "DOCTOR_SCHEDULE",
        "DOCTOR_INFO",
        "APPOINTMENT",
    } or (decision.context_action == "overwrite_doctor")
    if not needs_doctor_verification:
        return decision

    raw = str(entities.get("doctor_name") or "").strip()
    if raw:
        raw_norm = " ".join(raw.lower().replace("ё", "е").split())
        raw_tokens = [t for t in raw_norm.split(" ") if t]
        if len(raw_tokens) == 1 and raw_tokens[0] in _DOCTOR_NOISE_TOKENS:
            entities.pop("doctor_name", None)
            flags.add("doctor_name_unverified")
            raw = ""
    resolved: str | None = None
    if raw:
        resolved = await services.resolve_doctor_name(raw)
    if not resolved and decision.context_action == "overwrite_doctor":
        resolved = await services.resolve_doctor_name(user_text)

    context_action = decision.context_action
    if resolved:
        entities["doctor_name"] = resolved
        flags.add("doctor_name_verified")
    elif raw:
        entities.pop("doctor_name", None)
        flags.add("doctor_name_unverified")
        if decision.label == "APPOINTMENT" and _looks_like_patient_fio(raw):
            entities["patient_name"] = raw
            flags.add("patient_name_from_unverified_doctor")
            context_action = "continue"
    elif decision.context_action == "overwrite_doctor" and decision.label == "OTHER":
        # Не подтвердили нового врача по кэшу — считаем, что это не переключение врача.
        context_action = "continue"

    return RouteDecision(
        label=decision.label,
        confidence=decision.confidence,
        entities=entities,
        flags=flags,
        needs_handoff=decision.needs_handoff,
        context_action=context_action,
        source=decision.source,
        clarify_needed=decision.clarify_needed,
        clarify_reason=decision.clarify_reason,
        clarify_slots=list(decision.clarify_slots),
        intent_candidates=list(decision.intent_candidates),
    )


async def _sanitize_doctor_in_entities(
    entities: dict[str, Any],
    services: Services,
    *,
    label: str,
) -> dict[str, Any]:
    """
    Любой doctor_name, появившийся после quick-fill/merge, должен быть
    подтвержден через кэш врачей.
    """
    out = dict(entities or {})
    raw = str(out.get("doctor_name") or "").strip()
    if not raw:
        return out

    resolved = await services.resolve_doctor_name(raw)
    if resolved:
        out["doctor_name"] = resolved
        return out

    out.pop("doctor_name", None)
    if label == "APPOINTMENT" and _looks_like_patient_fio(raw) and not out.get("patient_name"):
        out["patient_name"] = raw
    return out


async def _backfill_appointment_doctor_from_text(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> None:
    """
    В APPOINTMENT pending иногда приходит короткая реплика врача
    ("к Дразнину", "Дразнин"), которая не проходит quick-fill контекст.
    Делаем безопасный fallback через resolve_doctor_name (по кэшу врачей).
    """
    pending = memory.get_pending(state)
    if not isinstance(pending, dict) or pending.get("label") != "APPOINTMENT":
        return

    if state.last_entities.get("doctor_id") or state.last_entities.get("doctor_name"):
        return

    missing = pending.get("missing")
    if not isinstance(missing, list):
        return
    need_doctor_or_service = any(
        isinstance(m, str) and ("doctor_id" in m or "doctor_name" in m or "specialty" in m or "service_name" in m)
        for m in missing
    )
    if not need_doctor_or_service:
        return

    raw = str(user_text or "").strip()
    if not raw:
        return

    probes: list[str] = [raw]
    low = raw.lower()
    if low.startswith("к "):
        tail = raw[2:].strip()
        if tail:
            probes.append(tail)

    resolved: str | None = None
    for p in probes:
        resolved = await services.resolve_doctor_name(p)
        if resolved:
            break

    if not resolved:
        return

    memory.merge_entities(state, {"doctor_name": resolved}, label="APPOINTMENT")

    # Если в service_name ранее ошибочно попало то же слово (например, "Дразнин"),
    # очищаем его, чтобы план строился по doctor flow.
    svc = str(state.last_entities.get("service_name") or "").strip()
    if svc:
        svc_norm = " ".join(svc.lower().replace("ё", "е").split())
        raw_norm = " ".join(raw.lower().replace("ё", "е").split())
        resolved_norm = " ".join(str(resolved).lower().replace("ё", "е").split())
        if svc_norm in {raw_norm, resolved_norm}:
            state.last_entities.pop("service_name", None)

# ----------------------------
# Planning & execution
# ----------------------------

def build_plan(decision: RouteDecision, state: SessionState, user_text: str, memory: MemoryStore) -> Plan:
    pending = memory.get_pending(state)
    effective_label = _apply_pending_override(decision, pending, user_text=user_text)

    entities = state.last_entities
    missing = missing_slots(effective_label, entities)

    if missing:
        memory.set_pending(state, label=effective_label, missing_slots=missing)
        return Plan(label=effective_label, steps=[])

    memory.clear_pending(state)

    label = effective_label
    steps: list[PlanStep] = []

    if "doc_request_main_index" in decision.flags or "doc_request_handoff" in decision.flags:
        steps.append(PlanStep(tool="main_index_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "TEST_RESULT":
        steps.append(PlanStep(tool="test_result_status", input={"query": user_text, "entities": dict(entities)}, auth="none"))
        return Plan(label=label, steps=steps)

    if label == "TEST_ASSIST":
        steps.append(PlanStep(tool="test_assist", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_SCHEDULE":
        steps.append(PlanStep(tool="doctors_schedule_week", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_INFO":
        steps.append(PlanStep(tool="doctors_info", input={"query": user_text, "entities": dict(entities)}))
        if entities.get("doctor_id") or entities.get("doctor_name"):
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}, required=False))
        return Plan(label=label, steps=steps)

    if label == "APPOINTMENT":
        if entities.get("doctor_id") or entities.get("doctor_name"):
            steps.append(PlanStep(tool="doctors_schedule_week", input={"query": user_text, "entities": dict(entities)}))
        else:
            address_entities = dict(entities)
            address_entities["__appointment_mode"] = True
            steps.append(PlanStep(tool="address_info", input={"query": user_text, "entities": address_entities}))
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}, required=False))
        return Plan(label=label, steps=steps)

    if label == "PRICE":
        service_known = bool(str(entities.get("service_name") or entities.get("test_name") or "").strip())
        doctor_known = bool(entities.get("doctor_id") or entities.get("doctor_name"))
        if service_known and not doctor_known:
            steps.append(PlanStep(tool="service_bundle_info", input={"query": user_text, "entities": dict(entities)}))
        else:
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "ADDRESS":
        steps.append(PlanStep(tool="address_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "PREPARE":
        steps.append(PlanStep(tool="test_prepare", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "NEWS":
        steps.append(PlanStep(tool="news_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    return Plan(label=label, steps=[])


async def execute_plan(plan: Plan, state: SessionState, services: Services) -> Evidence:
    ev = Evidence()

    for step in plan.steps:
        if step.auth == "patient_token":
            need_auth, msg = require_auth_for_test_result(state.is_authenticated)
            if need_auth:
                ev.put("auth_required", True)
                ev.put("auth_message", msg)
                return ev

        tool = step.tool
        inp = step.input
        q = inp.get("query", "")
        ent = inp.get("entities") or {}

        try:
            if tool == "doctors_info":
                ev.put("doctors_info", await services.doctors_info(q, ent))
            elif tool == "doctors_schedule_week":
                ev.put("doctor_schedule", await services.doctors_schedule_week(q, ent))
            elif tool == "appointment_help":
                ev.put("appointment", await services.appointment_help(q, ent))
            elif tool == "test_assist":
                ev.put("test_assist", await services.test_assist(q, ent))
            elif tool == "test_prepare":
                ev.put("prepare", await services.test_prepare(q, ent))
            elif tool == "test_result_status":
                ev.put("test_result_status", await services.test_result_status(q, ent))
            elif tool == "price_info":
                ev.put("price", await services.price_info(q, ent))
            elif tool == "main_index_info":
                ev.put("main_index_info", await services.main_index_info(q, ent))
            elif tool == "service_bundle_info":
                ev.put("service_bundle", await services.service_bundle_info(q, ent))
            elif tool == "address_info":
                ev.put("address", await services.address_info(q, ent))
            elif tool == "news_info":
                ev.put("news", await services.news_info(q, ent))
            else:
                ev.put("unknown_tool", tool)
        except Exception as e:
            ev.put(
                f"{tool}_error",
                {
                    "tool": tool,
                    "message": str(e),
                    "query": q,
                },
            )
            ev.put("handoff_required", True)
            ev.put("handoff_reason", "service_error")
            ev.put(
                "handoff_message",
                handoff_message("service_error"),
            )
            return ev

    return ev


async def route_patient_message(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence]:
    # Вежливое переключение на вторичный интент по короткому "да/нет".
    queue = _get_secondary_queue(state)
    if (
        state.last_entities.get("_secondary_offer_pending")
        and queue
        and not state.last_entities.get("appointment_confirm_pending")
        and not state.last_entities.get("appointment_flow_active")
    ):
        reply_kind = contextual_reply_kind(user_text)
        if reply_kind == "yes":
            next_label = queue.pop(0)
            _set_secondary_queue(state, queue)
            state.last_entities["_secondary_offer_pending"] = False
            decision = RouteDecision(
                label=next_label,  # type: ignore[arg-type]
                confidence=0.9,
                entities={"secondary_intent_from_queue": True},
                flags={"secondary_intent_activated"},
                needs_handoff=False,
            )
            plan = build_plan(decision, state, user_text, memory=memory)
            evidence = await execute_plan(plan, state, services)
            return decision, plan, evidence
        if reply_kind == "no":
            _set_secondary_queue(state, [])
            state.last_entities["_secondary_offer_pending"] = False
        else:
            # Пользователь продолжил диалог в другом направлении.
            state.last_entities["_secondary_offer_pending"] = False

    # Диагностические NLU-поля не должны засорять долгоживущий session state.
    # Актуальный trace отдаем только через debug-канал.
    for key in ("_nlu_candidates", "_nlu_merged_from", "_nlu_shadow"):
        state.last_entities.pop(key, None)

    nlu_debug: dict[str, Any] = {}
    use_v2_nlu = _env_flag("MR_ROUTER_V2_ENABLE", True)
    shadow_nlu = _env_flag("MR_ROUTER_V2_SHADOW", False) or _env_flag("MR_NLU_SHADOW", False)
    if use_v2_nlu:
        nlu_result = await analyze_with_candidates(user_text, state, runtime_options=runtime_options)
        decision = nlu_result.decision
        nlu_debug["candidates"] = [
            {
                "source": c.source,
                "label": c.label,
                "confidence": c.confidence,
                "flags": c.flags[:8],
            }
            for c in nlu_result.candidates
        ]
        nlu_debug["merged_from"] = nlu_result.merged_from
        if nlu_result.trace:
            nlu_debug["nlu_trace"] = nlu_result.trace
            nlu_debug["trace"] = nlu_result.trace
        if shadow_nlu:
            legacy = await analyze(user_text, state.last_entities)
            nlu_debug["shadow"] = {
                "legacy_label": legacy.label,
                "legacy_conf": legacy.confidence,
                "v2_label": decision.label,
                "v2_conf": decision.confidence,
            }
    else:
        decision = await analyze(user_text, state.last_entities, runtime_options=runtime_options)
    decision = await _verify_doctor_entity(decision, services, user_text)
    decision = _apply_context_action(decision, state, user_text)
    promoted_label, promoted_flags = apply_verified_doctor_override(decision.label, set(decision.flags), user_text)
    if promoted_label != decision.label or promoted_flags != decision.flags:
        decision = _copy_decision(
            decision,
            label=promoted_label,  # type: ignore[arg-type]
            confidence=max(decision.confidence, 0.65),
            flags=promoted_flags,
            needs_handoff=False,
        )

    # При запросах по специальности (без явного врача) чистим залипшего врача из state.
    if (
        decision.label in {"DOCTOR_INFO", "DOCTOR_SCHEDULE"}
        and decision.entities.get("specialty")
        and not any(decision.entities.get(k) for k in ("doctor_name", "doctor_id", "last_name", "doctor_last_name"))
    ):
        for k in ("doctor_name", "doctor_id", "last_name", "doctor_last_name"):
            state.last_entities.pop(k, None)

    if decision.label in {"APPOINTMENT", "TEST_ASSIST"} and not detect_prepare_intent(user_text):
        merged_ctx = dict(state.last_entities)
        merged_ctx.update(decision.entities or {})
        if detect_nonbookable_walkin_intent(user_text, merged_ctx):
            entities = dict(decision.entities)
            if not entities.get("service_name"):
                svc = nonbookable_service_hint(user_text)
                if svc:
                    entities["service_name"] = svc
            decision = _copy_decision(
                decision,
                label="ADDRESS",
                confidence=max(decision.confidence, 0.78),
                entities=entities,
                flags=set(decision.flags) | {"policy_nonbookable_walkin"},
                needs_handoff=False,
                context_action="continue",
                source="guardrail_post",
            )

    # Сохраняем сценарий записи на операторских уточнениях (ветка "да/нет", короткие ответы и т.п.).
    if (
        decision.label == "OTHER"
        and decision.context_action == "continue"
        and state.last_entities.get("appointment_flow_active")
        and _should_keep_appointment_flow_override(user_text)
    ):
        decision = _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            flags=set(decision.flags) | {"flow_appointment_override"},
            needs_handoff=False,
            context_action="continue",
        )

    # Защита активного APPOINTMENT flow: не даем случайной переклассификации
    # увести реплику "дата/время/ФИО" в чужой интент.
    if (
        decision.label in {"TEST_RESULT", "DOCTOR_SCHEDULE", "DOCTOR_INFO"}
        and state.last_entities.get("appointment_flow_active")
        and _should_keep_appointment_flow_override(user_text)
    ):
        decision = _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.60),
            flags=set(decision.flags) | {"flow_appointment_guard_override"},
            needs_handoff=False,
            context_action="continue",
        )

    if decision.label == "DOCTOR_SCHEDULE":
        # Очищаем хвосты сценария записи, чтобы расписание не фильтровалось
        # старым branch/date/time из предыдущих шагов.
        for k in (
            "appointment_flow_active",
            "appointment_confirm_pending",
            "appointment_confirmed",
            "appointment_branch_options",
            "service_name",
            "test_name",
            "branch_id",
            "branch_name",
            "date_from",
            "date_to",
            "time_from",
            "time_to",
            "date_hint",
        ):
            state.last_entities.pop(k, None)
        city_hint = match_city(user_text)
        if city_hint and not decision.entities.get("city"):
            decision.entities["city"] = city_hint

    if decision.label == "TEST_RESULT":
        # При переходе к результатам анализов завершаем хвост APPOINTMENT flow.
        # Исключение: если прямо сейчас ждем ФИО пациента и пользователь прислал ФИО,
        # не сбрасываем запись из-за случайной переклассификации.
        pending_now = memory.get_pending(state)
        keep_appointment_flow = (
            (_is_appointment_waiting_patient_name(pending_now) and _looks_like_patient_fio(user_text))
            or (
                bool(state.last_entities.get("appointment_flow_active"))
                and _should_keep_appointment_flow_override(user_text)
            )
        )
        if not keep_appointment_flow:
            _reset_appointment_state_flags(state)

    # Entity grounding: принимаем только подтвержденные/разрешенные сущности.
    pending_before_merge = memory.get_pending(state)
    grounding = await ground_decision_entities(
        decision=decision,
        user_text=user_text,
        state=state,
        services=services,
        pending=pending_before_merge,
    )
    if grounding.entities != decision.entities or grounding.flags:
        decision = _copy_decision(
            decision,
            entities=grounding.entities,
            flags=set(decision.flags) | set(grounding.flags),
        )

    # merge entities from LLM+rules
    memory.merge_entities(state, decision.entities, label=decision.label)
    # Явный город в текущей реплике должен уметь исправлять/обновлять контекст
    # даже если city уже был заполнен ранее неверно.
    city_hint_now = match_city(user_text)
    if city_hint_now and decision.label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        memory.merge_entities(state, {"city": city_hint_now}, label=decision.label)
    elif decision.label in {"APPOINTMENT", "ADDRESS", "TEST_ASSIST", "DOCTOR_INFO", "DOCTOR_SCHEDULE"}:
        if not str(state.last_entities.get("city") or "").strip():
            memory.merge_entities(state, {"city": _DEFAULT_CITY}, label=decision.label)
    sec_now = _normalize_secondary_labels(decision.entities.get("secondary_intents"))
    if sec_now:
        existing = _get_secondary_queue(state)
        merged = [x for x in existing if x != decision.label]
        for x in sec_now:
            if x != decision.label and x not in merged:
                merged.append(x)
        _set_secondary_queue(state, merged)

    # quick fill on current turn (before pending exists)
    pending = memory.get_pending(state)
    if not pending:
        missing_now = missing_slots(decision.label, state.last_entities)
        if missing_now:
            quick_now = quick_fill_entities_from_text(user_text, state.last_entities, missing_now, services)
            if quick_now:
                quick_now = await _sanitize_doctor_in_entities(quick_now, services, label=decision.label)
                memory.merge_entities(state, quick_now, label=decision.label)
        elif decision.label == "APPOINTMENT" and state.last_entities.get("appointment_flow_active"):
            # В активном сценарии записи продолжаем извлекать филиал/дату/время
            # даже если формально required slots уже заполнены.
            quick_flow = quick_fill_entities_from_text(
                user_text,
                state.last_entities,
                [
                    "_any_of:doctor_id,doctor_name,specialty",
                    "_any_of:city,branch_name,branch_id",
                    "date_from",
                    "time_from",
                    "patient_name",
                ],
                services,
            )
            if quick_flow:
                quick_flow = await _sanitize_doctor_in_entities(quick_flow, services, label=decision.label)
                memory.merge_entities(state, quick_flow, label=decision.label)

    # if pending exists, try quick fill missing slots (NO LLM)
    pending = memory.get_pending(state)
    if pending:
        pend_label = pending.get("label")
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        if isinstance(pend_label, str) and isinstance(missing, list) and missing:
            quick = quick_fill_entities_from_text(user_text, state.last_entities, missing, services)
            if quick:
                quick = await _sanitize_doctor_in_entities(quick, services, label=pend_label)
                memory.merge_entities(state, quick, label=pend_label)
        if pend_label == "APPOINTMENT":
            await _backfill_appointment_doctor_from_text(user_text, state, services, memory)

    _fill_date_from_schedule_windows(state, decision.label)

    plan = build_plan(decision, state, user_text, memory=memory)
    evidence = await execute_plan(plan, state, services)
    if nlu_debug:
        evidence.debug_trace.append({"nlu": nlu_debug})
    # Graph state transition (FSM слой)
    pending_after_plan = memory.get_pending(state)
    graph_out = _GRAPH_ENGINE.next(
        session=state,
        decision=decision,
        pending=pending_after_plan,
        handoff_planned=bool(evidence.get("handoff_required")),
    )
    evidence.debug_trace.append(
        {
            "graph_transition": {
                "from": graph_out.transition.from_state.value,
                "to": graph_out.transition.to_state.value,
                "reason": graph_out.transition.reason,
            }
        }
    )
    update_summary(
        state,
        reason="topic_switch" if decision.context_action in {"new_topic", "overwrite_doctor"} else "",
    )
    return decision, plan, evidence



def _debug_meta(decision: RouteDecision, plan: Plan, evidence: Evidence, state: SessionState, pending: Any) -> dict[str, Any]:
    nlu_trace = _extract_nlu_trace(evidence)
    return {
        "decision": {
            "label": decision.label,
            "confidence": decision.confidence,
            "context_action": decision.context_action,
            "source": decision.source,
            "flags": sorted(list(decision.flags)),
            "entities": decision.entities,
            "needs_handoff": decision.needs_handoff,
            "clarify_needed": decision.clarify_needed,
            "clarify_reason": decision.clarify_reason,
            "clarify_slots": list(decision.clarify_slots),
            "intent_candidates": list(decision.intent_candidates),
        },
        "plan": {
            "label": plan.label,
            "steps": [
                {"tool": s.tool, "input": s.input, "required": s.required, "auth": s.auth}
                for s in plan.steps
            ],
        },
        "evidence": {
            "items": evidence.items,
            "debug_trace": evidence.debug_trace,
        },
        "pending": pending,
        "history_tail": (state.history or [])[-10:],
        "last_entities": state.last_entities,
        "summary": state.summary,
        "nlu_trace": nlu_trace,
    }


def _build_price_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "PRICE":
        return None
    price_payload = evidence.get("price")
    if not isinstance(price_payload, dict):
        return None
    render_entities = dict(state.last_entities or {})
    used = price_payload.get("entities_used")
    if isinstance(used, dict):
        doctor_id_resolved = used.get("doctor_id_resolved")
        doctor_name_resolved = used.get("doctor_name_resolved")
        service_name_effective = str(used.get("service_name_effective") or "").strip()
        if doctor_id_resolved:
            render_entities["doctor_id"] = doctor_id_resolved
        if isinstance(doctor_name_resolved, str) and doctor_name_resolved.strip():
            render_entities["doctor_name"] = doctor_name_resolved.strip()
        if service_name_effective:
            render_entities["service_name"] = service_name_effective
            render_entities.pop("test_name", None)
    text = format_price_for_patient(price_payload, render_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_service_bundle_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "PRICE":
        return None
    payload = evidence.get("service_bundle")
    if not isinstance(payload, dict):
        return None
    text = format_service_bundle_for_patient(payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_doctor_info_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "DOCTOR_INFO":
        return None
    doctors_info_payload = evidence.get("doctors_info")
    if not isinstance(doctors_info_payload, dict):
        return None
    text = format_doctor_info_for_patient(doctors_info_payload, state.last_entities)
    price_payload = evidence.get("price")
    if isinstance(price_payload, dict):
        raw_prices = price_payload.get("prices")
        if isinstance(raw_prices, list) and raw_prices:
            text = (
                f"{text}\n\n"
                "Примеры стоимости услуг этого врача:\n"
                f"{format_price_for_patient(price_payload, state.last_entities)}"
            ).strip()
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_test_result_response(flow_label: str, evidence: Evidence) -> ResponseEnvelope | None:
    if flow_label != "TEST_RESULT":
        return None
    result_status = evidence.get("test_result_status")
    if not isinstance(result_status, dict):
        return None

    if result_status.get("ready") is True:
        note = str(result_status.get("note") or "")
        preview = str(result_status.get("result_preview") or "").strip()
        links_raw = result_status.get("result_links")
        links = [str(x).strip() for x in links_raw] if isinstance(links_raw, list) else []
        links = [x for x in links if x]
        if note == "result_link_constructed":
            text = "Сформировал ссылку для просмотра результата по указанным данным."
        else:
            text = "Результаты по вашим данным найдены."
        if links:
            if len(links) == 1:
                text = f"{text}\n\nСсылка на результат: {links[0]}"
            else:
                lines = "\n".join(f"- {u}" for u in links[:5])
                text = f"{text}\n\nСсылки на результаты:\n{lines}"
        if preview and not links:
            text = f"{text}\n\n{preview}"
        return ResponseEnvelope(text=text, attachments=[], handoff=False)

    missing = result_status.get("missing_fields")
    if isinstance(missing, list) and missing:
        return ResponseEnvelope(
            text=clarification_question("TEST_RESULT", [str(m) for m in missing]),
            attachments=[],
            handoff=False,
        )

    preview = str(result_status.get("result_preview") or "").strip()
    text = preview or "По указанным данным результаты пока не найдены или ещё не готовы."
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_doctor_schedule_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "DOCTOR_SCHEDULE":
        return None
    schedule_payload = evidence.get("doctor_schedule")
    if not isinstance(schedule_payload, dict):
        return None
    _hydrate_appointment_context_from_schedule(state, schedule_payload)
    # После показа расписания оставляем "живой" контекст записи:
    # короткие реплики вида "на 16:30" должны интерпретироваться
    # как продолжение сценария APPOINTMENT, а не как новый OTHER.
    state.last_entities["appointment_flow_active"] = True
    text = format_doctor_schedule_for_patient(schedule_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_address_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    memory: MemoryStore,
    decision: RouteDecision,
    user_text: str,
) -> ResponseEnvelope | None:
    if flow_label != "ADDRESS":
        return None
    address_payload = evidence.get("address")
    if not isinstance(address_payload, dict):
        return None

    branches_raw = address_payload.get("branches")
    addresses_raw = address_payload.get("addresses")
    has_branches = isinstance(branches_raw, list) and any(
        str((x or {}).get("address") if isinstance(x, dict) else x).strip() for x in branches_raw
    )
    has_addresses = isinstance(addresses_raw, list) and any(str(x).strip() for x in addresses_raw)
    if not has_branches and not has_addresses:
        # Если город/адрес не найден, оставляем ADDRESS pending на повторный ввод города.
        # Это предотвращает выпадение в OTHER после опечатки ("Самраа" -> "Самара").
        state.last_entities.pop("city", None)
        memory.set_pending(state, label="ADDRESS", missing_slots=["_any_of:city,branch_name,branch_id"])

    walkin_hint: str | None = None
    if any("nonbookable" in str(f) for f in decision.flags):
        walkin_hint = nonbookable_service_hint(user_text) or str(state.last_entities.get("service_name") or "").strip()
        if walkin_hint == "":
            walkin_hint = None
    text = format_address_for_patient(
        address_payload,
        state.last_entities,
        nonbookable_service=walkin_hint,
    )
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_news_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "NEWS":
        return None
    news_payload = evidence.get("news")
    if not isinstance(news_payload, dict):
        return None
    text = format_news_for_patient(news_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_main_index_info_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get("main_index_info")
    if not isinstance(payload, dict):
        return None
    content = str(payload.get("content") or "").strip()
    if content:
        return ResponseEnvelope(text=content, attachments=[], handoff=False)
    return None


def _build_appointment_schedule_preview_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
) -> ResponseEnvelope | None:
    if flow_label != "APPOINTMENT":
        return None
    schedule_payload = evidence.get("doctor_schedule")
    if not isinstance(schedule_payload, dict):
        return None
    if not (state.last_entities.get("doctor_name") or state.last_entities.get("doctor_id")):
        return None
    if state.last_entities.get("date_from") or state.last_entities.get("date_hint"):
        return None
    if state.last_entities.get("time_from"):
        return None

    _hydrate_appointment_context_from_schedule(state, schedule_payload)
    state.last_entities["appointment_flow_active"] = True
    text = format_doctor_schedule_for_patient(schedule_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _build_appointment_step_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    if flow_label != "APPOINTMENT":
        return None

    entities = state.last_entities
    state.last_entities["appointment_flow_active"] = True

    appointment_step = appointment_step_policy(entities)
    service = appointment_service_display(entities)
    city = str(entities.get("city") or "").strip()

    if appointment_step == APPOINTMENT_STEP_BRANCH:
        if not city:
            city = _DEFAULT_CITY
            state.last_entities["city"] = city
        stored_options = state.last_entities.get("appointment_branch_options")
        addresses = []
        if isinstance(stored_options, list):
            addresses = [str(x).strip() for x in stored_options if str(x).strip()]
        if not addresses:
            branches = _safe_get_branches(services)
            addresses = appointment_addresses_for_city(
                evidence.get("address"),
                branches,
                city=city or None,
                limit=5,
            )
        state.last_entities["appointment_branch_options"] = addresses
        return ResponseEnvelope(
            text=appointment_text_branch_prompt(service, city, addresses),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_DATETIME:
        state.last_entities.pop("appointment_branch_options", None)
        price_rub = extract_price_rub(evidence.get("price"))
        branch = str(entities.get("branch_name") or entities.get("city") or "выбранном филиале").strip()
        return ResponseEnvelope(
            text=appointment_text_datetime_prompt(service, branch, price_rub),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_PATIENT:
        # Фиксируем pending patient_name, чтобы короткие/частичные ФИО
        # не выбивали диалог в другой интент (например, TEST_RESULT).
        memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])
        return ResponseEnvelope(
            text=appointment_text_patient_name_prompt(),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_CONFIRM:
        state.last_entities["appointment_confirm_pending"] = True
        summary = appointment_summary(entities)
        return ResponseEnvelope(text=appointment_text_confirm_prompt(summary), handoff=False)

    return None


async def patient_routing_stream(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    debug: bool = False,
    runtime_options: RuntimeOptions | None = None,
) -> AsyncGenerator[ResponseEnvelope, None]:
    try:
        services.ensure_background_refresh_started()
    except Exception:
        pass

    # Явный запрос оператора должен иметь абсолютный приоритет.
    if explicit_operator_requested(user_text):
        memory.clear_pending(state)
        state.last_entities["_nlu_unclear_count"] = 0
        state.last_entities["appointment_flow_active"] = False
        yield ResponseEnvelope(
            text=handoff_message("manual_operator"),
            attachments=[],
            handoff=True,
            state_update=_early_debug_state_update(
                debug,
                label="OTHER",
                handoff=True,
                flags={"manual_operator"},
                context_action="new_topic",
                confidence=1.0,
            ),
        )
        return

    city_now = match_city(user_text)
    if city_now and not _is_samara_city(city_now):
        memory.clear_pending(state)
        state.last_entities.pop("city", None)
        _reset_appointment_state_flags(state)
        update_summary(state, reason="handoff")
        yield ResponseEnvelope(
            text=_SAMARA_ONLY_OPERATOR_TEXT,
            attachments=[],
            handoff=True,
            state_update=_early_debug_state_update(
                debug,
                label="ADDRESS",
                handoff=True,
                flags={"city_not_supported"},
                context_action="new_topic",
                confidence=1.0,
            ),
        )
        return

    # Подтверждение записи обрабатываем до NLU/route, чтобы rich/hybrid режим
    # не влиял на handoff-переход.
    if state.last_entities.get("appointment_confirm_pending"):
        confirm_transition = appointment_confirmation_transition(user_text)
        if confirm_transition == APPOINTMENT_CONFIRM_YES:
            summary = appointment_summary(state.last_entities)
            state.last_entities["appointment_confirmed"] = True
            state.last_entities.pop("appointment_confirm_pending", None)
            state.last_entities.pop("appointment_flow_active", None)
            yield ResponseEnvelope(
                text=appointment_text_confirmed_handoff(summary),
                handoff=True,
                state_update=_early_debug_state_update(
                    debug,
                    label="APPOINTMENT",
                    handoff=True,
                    flags={"appointment_confirm_yes"},
                    confidence=1.0,
                ),
            )
            return
        if confirm_transition == APPOINTMENT_CONFIRM_NO:
            state.last_entities["appointment_confirmed"] = False
            state.last_entities.pop("appointment_confirm_pending", None)
            for k in ("date_from", "date_to", "time_from", "time_to", "date_hint"):
                state.last_entities.pop(k, None)
            state.last_entities["appointment_flow_active"] = True
            yield ResponseEnvelope(
                text=appointment_text_reask_datetime(),
                handoff=False,
                state_update=_early_debug_state_update(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_confirm_no"},
                    confidence=1.0,
                ),
            )
            return
        if _is_new_topic_while_confirm_pending(user_text):
            # Пользователь сменил тему: выходим из шага подтверждения.
            state.last_entities.pop("appointment_confirm_pending", None)
            state.last_entities.pop("appointment_flow_active", None)
            for k in (
                "date_from",
                "date_to",
                "time_from",
                "time_to",
                "date_hint",
                "appointment_windows",
                "appointment_branch_options",
            ):
                state.last_entities.pop(k, None)
            memory.clear_pending(state)
        else:
            yield ResponseEnvelope(
                text=appointment_text_reask_confirm(),
                handoff=False,
                state_update=_early_debug_state_update(
                    debug,
                    label="APPOINTMENT",
                    handoff=False,
                    flags={"appointment_confirm_reask"},
                    confidence=1.0,
                ),
            )
            return

    try:
        decision, plan, evidence = await route_patient_message(
            user_text,
            state,
            services,
            memory,
            runtime_options=runtime_options,
        )
    except Exception as e:
        fallback_text = handoff_message("service_error")
        state_update: dict[str, Any] = {}
        if debug:
            state_update = {"debug": {"route_error": str(e)}}
        yield ResponseEnvelope(
            text=fallback_text,
            attachments=[],
            handoff=True,
            state_update=state_update,
        )
        return
    flow_label = plan.label

    if debug:
        pending = memory.get_pending(state)
        yield ResponseEnvelope(
            text="",
            attachments=[],
            handoff=False,
            state_update={"debug": _debug_meta(decision, plan, evidence, state, pending)},
        )

    user_turn_count = sum(1 for h in (state.history or []) if isinstance(h, dict) and h.get("role") == "user")
    if "smalltalk_greeting" in decision.flags and user_turn_count <= 1:
        yield ResponseEnvelope(text=INTRO_TEXT, attachments=[], handoff=False)
        return

    if decision.label == "URGENT":
        yield render_urgent()
        return
    if decision.label == "COMPLAINT":
        yield render_complaint()
        return
    if decision.label == "MEDICAL_ADVICE":
        yield render_medical_advice()
        return

    # Смягченный fallback для неуверенного NLU:
    # сначала уточняем, а к оператору передаем только после 3-го непонимания
    # или при явном запросе "оператор".
    # Важно: не перебиваем активный pending/flow (иначе ломается естественный диалог).
    pending_now = memory.get_pending(state)
    flow_active = bool(state.last_entities.get("appointment_flow_active"))
    recovery = evaluate_recovery(
        user_text=user_text,
        decision=decision,
        flow_label=flow_label,
        pending_exists=bool(pending_now),
        flow_active=flow_active,
        state_entities=state.last_entities,
        summary=state.summary,
        max_unclear=3,
    )
    if recovery.kind == "handoff":
        update_summary(state, reason="handoff")
        yield ResponseEnvelope(text=recovery.text or handoff_message("low_confidence"), handoff=True)
        return
    if recovery.kind == "clarify":
        _remember_question(
            state,
            recovery.reason or "clarify",
            list(decision.clarify_slots) if decision.clarify_slots else [],
        )
        yield ResponseEnvelope(text=recovery.text or LOW_CONF_CLARIFY_TEXT, handoff=False)
        return

    if flow_label == "DOCTOR_SCHEDULE":
        # есть специальность, но нет врача → уточняем
        doctor_known = bool(
            decision.entities.get("doctor_name")
            or decision.entities.get("doctor_last_name")
            or decision.entities.get("last_name")
            or state.last_entities.get("doctor_name")
            or state.last_entities.get("doctor_id")
        )
        specialty_known = bool(
            decision.entities.get("specialty")
            or state.last_entities.get("specialty")
        )
        if (
                specialty_known
                and not doctor_known
        ):
            # По specialty-сценарию (например, "гастроэнтеролог ближайший")
            # даем пройти к сервису расписания, который сам подбирает врача.
            pass

    pending = memory.get_pending(state)
    if not plan.steps and pending:
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        if flow_label == "APPOINTMENT":
            state.last_entities["appointment_flow_active"] = True
        _remember_question(state, f"pending:{flow_label}", missing if isinstance(missing, list) else [])
        yield ResponseEnvelope(
            text=clarification_question(flow_label, missing if isinstance(missing, list) else []),
            handoff=False,
        )
        return

    if evidence.get("auth_required"):
        yield ResponseEnvelope(text=evidence.get("auth_message", "Нужна авторизация."), handoff=False)
        return

    handoff_required, handoff_msg, handoff_reason = evidence_requires_handoff(evidence)
    if handoff_required:
        yield ResponseEnvelope(text=handoff_message(handoff_reason, handoff_msg), handoff=True)
        return

    main_index_resp = _build_main_index_info_response(evidence)
    if main_index_resp is not None:
        yield main_index_resp
        return

    service_bundle_resp = _build_service_bundle_response(flow_label, evidence, state)
    if service_bundle_resp is not None:
        yield service_bundle_resp
        return

    price_resp = _build_price_response(flow_label, evidence, state)
    if price_resp is not None:
        yield price_resp
        return

    test_result_resp = _build_test_result_response(flow_label, evidence)
    if test_result_resp is not None:
        yield test_result_resp
        return

    doctor_schedule_resp = _build_doctor_schedule_response(flow_label, evidence, state)
    if doctor_schedule_resp is not None:
        yield doctor_schedule_resp
        return

    doctor_info_resp = _build_doctor_info_response(flow_label, evidence, state)
    if doctor_info_resp is not None:
        yield doctor_info_resp
        return

    address_resp = _build_address_response(flow_label, evidence, state, memory, decision, user_text)
    if address_resp is not None:
        yield address_resp
        return

    news_resp = _build_news_response(flow_label, evidence, state)
    if news_resp is not None:
        yield news_resp
        return

    # Если пользователь сразу хочет записаться к конкретному врачу, сначала
    # показываем его актуальные окна, а не отправляем в общий сценарий "город -> филиал".
    appointment_preview_resp = _build_appointment_schedule_preview_response(flow_label, evidence, state)
    if appointment_preview_resp is not None:
        yield appointment_preview_resp
        return

    appointment_step_resp = _build_appointment_step_response(flow_label, evidence, state, services, memory)
    if appointment_step_resp is not None:
        yield appointment_step_resp
        return

    try:
        async for chunk in render_stream(user_text, decision, evidence, runtime_options=runtime_options):
            yield ResponseEnvelope(text=chunk, attachments=[], handoff=False)
    except Exception:
        yield ResponseEnvelope(
            text=handoff_message("renderer_error"),
            attachments=[],
            handoff=True,
        )
        return

    if decision.needs_handoff:
        yield ResponseEnvelope(text=decision_handoff_text(decision.flags), attachments=[], handoff=True)

    secondary = _get_secondary_queue(state)
    followup = _secondary_followup_text(secondary)
    if (
        followup
        and not state.last_entities.get("_secondary_offer_pending")
        and not decision.needs_handoff
        and flow_label not in {"APPOINTMENT", "URGENT", "COMPLAINT", "MEDICAL_ADVICE", "TEST_RESULT"}
    ):
        state.last_entities["_secondary_offer_pending"] = True
        yield ResponseEnvelope(text=followup, attachments=[], handoff=False)
