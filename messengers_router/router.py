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

import os
import re
from typing import AsyncGenerator, Any

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
    format_doctor_schedule_for_patient,
    format_doctor_info_for_patient,
    format_address_for_patient,
    format_news_for_patient,
)
from .memory import MemoryStore
from .city import match_city

_INTRO_TEXT = (
    "Здравствуйте! Это ИИ-помощник клиники «Наука».\n"
    "Через меня вы можете записаться к врачу, перенести или отменить запись, "
    "узнать результаты анализов и получить информацию об услугах."
)
_LOW_CONF_CLARIFY_TEXT = (
    "Уточните, пожалуйста, запрос чуть подробнее, чтобы я не ошибся: "
    "что именно нужно — запись, расписание врача, стоимость, адрес или результаты анализов?"
)
_DEFAULT_CITY = "Самара"
_SAMARA_ONLY_OPERATOR_TEXT = "Сейчас могу помочь только по Самаре. Соединяю с оператором."
_PRICE_TO_OPERATOR_TEXT = "По вопросам стоимости соединяю с оператором."
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


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
                ev.put("doctors_info", await services.doctors_info(q, ent, output_max=5))
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

    use_v2_nlu = _env_flag("MR_ROUTER_V2_ENABLE", True)
    shadow_nlu = _env_flag("MR_ROUTER_V2_SHADOW", False)
    if use_v2_nlu:
        nlu_result = await analyze_with_candidates(user_text, state, runtime_options=runtime_options)
        decision = nlu_result.decision
        state.last_entities["_nlu_candidates"] = [
            {
                "source": c.source,
                "label": c.label,
                "confidence": c.confidence,
                "flags": c.flags[:8],
            }
            for c in nlu_result.candidates
        ]
        state.last_entities["_nlu_merged_from"] = nlu_result.merged_from
        if shadow_nlu:
            legacy = await analyze(user_text, state.last_entities)
            state.last_entities["_nlu_shadow"] = {
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
        decision = RouteDecision(
            label=promoted_label,  # type: ignore[arg-type]
            confidence=max(decision.confidence, 0.65),
            entities=dict(decision.entities),
            flags=promoted_flags,
            needs_handoff=False,
            context_action=decision.context_action,
        )

    # При запросах по специальности (без явного врача) чистим залипшего врача из state.
    if (
        decision.label in {"DOCTOR_INFO", "DOCTOR_SCHEDULE"}
        and decision.entities.get("specialty")
        and not any(decision.entities.get(k) for k in ("doctor_name", "doctor_id", "last_name", "doctor_last_name"))
    ):
        for k in ("doctor_name", "doctor_id", "last_name", "doctor_last_name"):
            state.last_entities.pop(k, None)

    if decision.label in {"APPOINTMENT", "TEST_ASSIST"}:
        merged_ctx = dict(state.last_entities)
        merged_ctx.update(decision.entities or {})
        if detect_nonbookable_walkin_intent(user_text, merged_ctx):
            entities = dict(decision.entities)
            if not entities.get("service_name"):
                svc = nonbookable_service_hint(user_text)
                if svc:
                    entities["service_name"] = svc
            decision = RouteDecision(
                label="ADDRESS",
                confidence=max(decision.confidence, 0.78),
                entities=entities,
                flags=set(decision.flags) | {"policy_nonbookable_walkin"},
                needs_handoff=False,
                context_action="continue",
            )

    # Сохраняем сценарий записи на операторских уточнениях (ветка "да/нет", короткие ответы и т.п.).
    if (
        decision.label == "OTHER"
        and decision.context_action == "continue"
        and state.last_entities.get("appointment_flow_active")
        and _should_keep_appointment_flow_override(user_text)
    ):
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            entities=decision.entities,
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
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.60),
            entities=decision.entities,
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
            for k in ("appointment_flow_active", "appointment_confirm_pending", "appointment_confirmed"):
                state.last_entities.pop(k, None)

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
        decision = RouteDecision(
            label=decision.label,
            confidence=decision.confidence,
            entities=grounding.entities,
            flags=set(decision.flags) | set(grounding.flags),
            needs_handoff=decision.needs_handoff,
            context_action=decision.context_action,
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
    return {
        "decision": {
            "label": decision.label,
            "confidence": decision.confidence,
            "context_action": decision.context_action,
            "flags": sorted(list(decision.flags)),
            "entities": decision.entities,
            "needs_handoff": decision.needs_handoff,
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
    }


async def patient_routing_stream(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    debug: bool = False,
    runtime_options: RuntimeOptions | None = None,
) -> AsyncGenerator[ResponseEnvelope, None]:
    # Явный запрос оператора должен иметь абсолютный приоритет.
    if explicit_operator_requested(user_text):
        memory.clear_pending(state)
        state.last_entities["_nlu_unclear_count"] = 0
        state.last_entities["appointment_flow_active"] = False
        yield ResponseEnvelope(
            text=handoff_message("manual_operator"),
            attachments=[],
            handoff=True,
        )
        return

    city_now = match_city(user_text)
    if city_now and not _is_samara_city(city_now):
        memory.clear_pending(state)
        state.last_entities.pop("city", None)
        for k in ("appointment_flow_active", "appointment_confirm_pending", "appointment_confirmed"):
            state.last_entities.pop(k, None)
        update_summary(state, reason="handoff")
        yield ResponseEnvelope(
            text=_SAMARA_ONLY_OPERATOR_TEXT,
            attachments=[],
            handoff=True,
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
            )
            return
        yield ResponseEnvelope(
            text=appointment_text_reask_confirm(),
            handoff=False,
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
        yield ResponseEnvelope(text=_INTRO_TEXT, attachments=[], handoff=False)
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

    # По текущей политике стоимость не выдаем автоматически:
    # любые ценовые запросы передаем оператору.
    if decision.label == "PRICE":
        memory.clear_pending(state)
        for k in ("appointment_flow_active", "appointment_confirm_pending", "appointment_confirmed"):
            state.last_entities.pop(k, None)
        update_summary(state, reason="handoff")
        yield ResponseEnvelope(text=_PRICE_TO_OPERATOR_TEXT, handoff=True)
        return

    if "doc_request_handoff" in decision.flags:
        yield ResponseEnvelope(
            text=handoff_message("doc_request_handoff"),
            handoff=True,
        )
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
        yield ResponseEnvelope(text=recovery.text or _LOW_CONF_CLARIFY_TEXT, handoff=False)
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

    if flow_label == "TEST_RESULT":
        result_status = evidence.get("test_result_status")
        if isinstance(result_status, dict):
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
                yield ResponseEnvelope(text=text, attachments=[], handoff=False)
                return

            missing = result_status.get("missing_fields")
            if isinstance(missing, list) and missing:
                yield ResponseEnvelope(
                    text=clarification_question("TEST_RESULT", [str(m) for m in missing]),
                    attachments=[],
                    handoff=False,
                )
                return

            preview = str(result_status.get("result_preview") or "").strip()
            text = preview or "По указанным данным результаты пока не найдены или ещё не готовы."
            yield ResponseEnvelope(text=text, attachments=[], handoff=False)
            return

    schedule_payload = evidence.get("doctor_schedule")
    if flow_label == "DOCTOR_SCHEDULE" and isinstance(schedule_payload, dict):
        _hydrate_appointment_context_from_schedule(state, schedule_payload)
        # После показа расписания оставляем "живой" контекст записи:
        # короткие реплики вида "на 16:30" должны интерпретироваться
        # как продолжение сценария APPOINTMENT, а не как новый OTHER.
        state.last_entities["appointment_flow_active"] = True
        text = format_doctor_schedule_for_patient(schedule_payload, state.last_entities)
        yield ResponseEnvelope(text=text, attachments=[], handoff=False)
        return

    doctors_info_payload = evidence.get("doctors_info")
    if flow_label == "DOCTOR_INFO" and isinstance(doctors_info_payload, dict):
        text = format_doctor_info_for_patient(doctors_info_payload, state.last_entities)
        yield ResponseEnvelope(text=text, attachments=[], handoff=False)
        return

    address_payload = evidence.get("address")
    if flow_label == "ADDRESS" and isinstance(address_payload, dict):
        branches_raw = address_payload.get("branches")
        addresses_raw = address_payload.get("addresses")
        has_branches = isinstance(branches_raw, list) and any(str((x or {}).get("address") if isinstance(x, dict) else x).strip() for x in branches_raw)
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
        yield ResponseEnvelope(text=text, attachments=[], handoff=False)
        return

    news_payload = evidence.get("news")
    if flow_label == "NEWS" and isinstance(news_payload, dict):
        text = format_news_for_patient(news_payload, state.last_entities)
        yield ResponseEnvelope(text=text, attachments=[], handoff=False)
        return

    # Если пользователь сразу хочет записаться к конкретному врачу, сначала
    # показываем его актуальные окна, а не отправляем в общий сценарий "город -> филиал".
    if (
        flow_label == "APPOINTMENT"
        and isinstance(schedule_payload, dict)
        and (state.last_entities.get("doctor_name") or state.last_entities.get("doctor_id"))
        and not (state.last_entities.get("date_from") or state.last_entities.get("date_hint"))
        and not state.last_entities.get("time_from")
    ):
        _hydrate_appointment_context_from_schedule(state, schedule_payload)
        state.last_entities["appointment_flow_active"] = True
        text = format_doctor_schedule_for_patient(schedule_payload, state.last_entities)
        yield ResponseEnvelope(text=text, attachments=[], handoff=False)
        return

    if flow_label == "APPOINTMENT":
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
            yield ResponseEnvelope(
                text=appointment_text_branch_prompt(service, city, addresses),
                handoff=False,
            )
            return

        if appointment_step == APPOINTMENT_STEP_DATETIME:
            state.last_entities.pop("appointment_branch_options", None)
            price_rub = extract_price_rub(evidence.get("price"))
            branch = str(entities.get("branch_name") or entities.get("city") or "выбранном филиале").strip()
            yield ResponseEnvelope(
                text=appointment_text_datetime_prompt(service, branch, price_rub),
                handoff=False,
            )
            return

        if appointment_step == APPOINTMENT_STEP_PATIENT:
            # Фиксируем pending patient_name, чтобы короткие/частичные ФИО
            # не выбивали диалог в другой интент (например, TEST_RESULT).
            memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])
            yield ResponseEnvelope(
                text=appointment_text_patient_name_prompt(),
                handoff=False,
            )
            return

        if appointment_step == APPOINTMENT_STEP_CONFIRM:
            state.last_entities["appointment_confirm_pending"] = True
            summary = appointment_summary(entities)
            yield ResponseEnvelope(text=appointment_text_confirm_prompt(summary), handoff=False)
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
