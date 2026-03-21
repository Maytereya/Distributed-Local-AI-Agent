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

from .mess_types import Evidence, Plan, ResponseEnvelope, RouteDecision, SessionState
from .classifier import analyze
from .context_summary import update_summary
from .dialog_graph import GraphEngine
from .entity_grounder import (
    ground_decision_entities,
    sanitize_doctor_entities,
    verify_doctor_entities_in_decision,
)
from .flow_policy import (
    apply_context_action,
    fill_date_from_schedule_windows,
    get_secondary_queue,
    is_appointment_waiting_patient_name,
    is_short_prepare_followup,
    looks_like_patient_fio,
    normalize_secondary_labels,
    quick_fill_entities_from_text,
    secondary_followup_text,
    set_secondary_queue,
)
from .appointment_flow_guard import (
    clear_appointment_flow_context,
    reset_appointment_runtime_state,
    run_appointment_precheck,
    should_keep_appointment_flow_override,
)
from .nlu_pipeline import analyze_with_candidates
from .llm_mode_policy import RuntimeOptions
from .planner import build_plan as planner_build_plan
from .executor import execute_plan as executor_execute_plan
from .response_builder import (
    build_address_response as response_build_address_response,
    build_appointment_schedule_preview_response as response_build_appointment_schedule_preview_response,
    build_appointment_step_response as response_build_appointment_step_response,
    build_doctor_info_response as response_build_doctor_info_response,
    build_doctor_schedule_response as response_build_doctor_schedule_response,
    build_first_structured_response as response_build_first_structured_response,
    build_main_index_info_response as response_build_main_index_info_response,
    build_news_response as response_build_news_response,
    build_price_response as response_build_price_response,
    build_service_bundle_response as response_build_service_bundle_response,
    build_test_result_response as response_build_test_result_response,
)
from .policies import (
    missing_slots,
    clarification_question,
    evidence_requires_handoff,
    handoff_message,
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
    nonbookable_service_hint,
)
from .recovery_policy import contextual_reply_kind, evaluate_recovery, explicit_operator_requested
from .services import Services
from .renderer import (
    render_urgent,
    render_complaint,
    render_medical_advice,
    render_stream,
)
from .memory import MemoryStore
from .text_templates import INTRO_TEXT, LOW_CONF_CLARIFY_TEXT
from .city import match_city
from .topic_registry import (
    build_topic_flag,
    match_topic,
)
from .runtime_config import config as c

_DEFAULT_CITY = "Самара"
_SAMARA_ONLY_OPERATOR_TEXT = "Сейчас могу помочь только по Самаре. Соединяю с оператором."
_GRAPH_ENGINE = GraphEngine()
_TOPIC_OVERRIDE_ALLOW_FROM_OTHER = {"PREPARE", "NEWS"}
_TOPIC_OVERRIDE_MIN_SCORE = 2
_SECONDARY_SOFT_YES_RE = re.compile(
    r"^\s*(?:(?:да|ок|окей|хорошо|ладно)(?:\s+(?:спасибо|благодарю|благодарствую|спс))?|"
    r"(?:спасибо|благодарю|благодарствую|спс))\s*[!.,?;:]*\s*$",
    re.I,
)

# Backward-compat alias for tests/internal callers.
_should_keep_appointment_flow_override = should_keep_appointment_flow_override


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


def _is_secondary_soft_yes(text: str) -> bool:
    return bool(_SECONDARY_SOFT_YES_RE.match(str(text or "")))


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


def _apply_topic_registry_override(
    decision: RouteDecision,
    topic_match: Any,
) -> RouteDecision:
    topic_label = str(getattr(topic_match, "label", "") or "").strip().upper()
    topic_id = str(getattr(topic_match, "topic_id", "") or "").strip()
    topic_score = int(getattr(topic_match, "score", 0) or 0)
    if not topic_label or not topic_id:
        return decision

    flags = set(decision.flags)
    flags.add("topic_registry_match")
    flags.add(build_topic_flag(topic_id))

    # Безопасный приоритет критичных лейблов: registry не перезатирает
    # срочные/медицинские/жалобные контуры.
    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return _copy_decision(decision, flags=flags | {"topic_registry_hint_only"})

    # Лучший сценарий: labels совпали -> считаем это подтверждением.
    if decision.label == topic_label:
        return _copy_decision(
            decision,
            confidence=max(decision.confidence, 0.80),
            flags=flags | {"topic_registry_confirm"},
        )

    # Расхождение labels:
    # - по умолчанию registry работает как hint-only;
    # - override разрешаем только из OTHER и только по ограниченному allowlist.
    if decision.label != "OTHER":
        return _copy_decision(decision, flags=flags | {"topic_registry_hint_only"})

    if topic_label not in _TOPIC_OVERRIDE_ALLOW_FROM_OTHER:
        return _copy_decision(decision, flags=flags | {"topic_registry_hint_only"})

    if topic_score < _TOPIC_OVERRIDE_MIN_SCORE:
        return _copy_decision(decision, flags=flags | {"topic_registry_hint_only"})

    return _copy_decision(
        decision,
        label=topic_label,  # type: ignore[arg-type]
        confidence=max(decision.confidence, 0.78),
        flags=flags | {"topic_registry_override_from_other"},
        needs_handoff=False,
        source="topic_registry",
        context_action="continue",
    )


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


def _is_appointment_datetime_followup(user_text: str) -> bool:
    text = str(user_text or "").strip()
    if not text or not has_datetime_signal(text):
        return False
    low = text.lower()
    if (
        detect_prepare_intent(low)
        or detect_test_assist_intent(low)
        or detect_test_result_intent(low)
        or detect_doc_request_intent(low)
        or detect_schedule_intent(low)
        or detect_doctor_info_intent(low)
        or detect_address_intent(low)
        or detect_price_intent(low)
        or detect_nonbookable_walkin_intent(text)
    ):
        return False
    return True


def _apply_appointment_continuity_overrides(
    decision: RouteDecision,
    state: SessionState,
    user_text: str,
) -> RouteDecision:
    """
    Единая post-policy точка удержания APPOINTMENT flow.
    Приоритет:
    1) реплика с датой/временем
    2) OTHER + контекстное продолжение
    3) guard для TEST_RESULT/DOCTOR_* при активной записи
    """
    if not state.last_entities.get("appointment_flow_active"):
        return decision

    if (
        has_datetime_signal(user_text)
        and decision.label in {"OTHER", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "TEST_RESULT", "ADDRESS", "PRICE"}
        and not detect_prepare_intent(user_text)
        and not detect_test_assist_intent(user_text)
        and not detect_test_result_intent(user_text)
        and not detect_doc_request_intent(user_text)
    ):
        return _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.66),
            flags=set(decision.flags) | {"flow_datetime_appointment_override"},
            needs_handoff=False,
            context_action="continue",
        )

    if (
        decision.label in {"OTHER", "ADDRESS"}
        and decision.context_action == "continue"
        and should_keep_appointment_flow_override(user_text)
    ):
        return _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            flags=set(decision.flags) | {"flow_appointment_override"},
            needs_handoff=False,
            context_action="continue",
        )

    if decision.label in {"TEST_RESULT", "DOCTOR_SCHEDULE", "DOCTOR_INFO"} and should_keep_appointment_flow_override(user_text):
        return _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.60),
            flags=set(decision.flags) | {"flow_appointment_guard_override"},
            needs_handoff=False,
            context_action="continue",
        )

    return decision


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
    # Backward-compatible wrapper: реализация живет в entity_grounder.
    return await verify_doctor_entities_in_decision(
        decision=decision,
        user_text=user_text,
        services=services,
    )


async def _sanitize_doctor_in_entities(
    entities: dict[str, Any],
    services: Services,
    *,
    label: str,
) -> dict[str, Any]:
    # Backward-compatible wrapper: реализация живет в entity_grounder.
    return await sanitize_doctor_entities(entities=entities, services=services, label=label)


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
    return planner_build_plan(decision, state, user_text, memory)


async def execute_plan(plan: Plan, state: SessionState, services: Services) -> Evidence:
    return await executor_execute_plan(plan, state, services)


async def route_patient_message(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence]:
    # Вежливое переключение на вторичный интент по короткому "да/нет".
    queue = get_secondary_queue(state)
    if (
        state.last_entities.get("_secondary_offer_pending")
        and queue
        and not state.last_entities.get("appointment_confirm_pending")
        and not state.last_entities.get("appointment_flow_active")
    ):
        reply_kind = contextual_reply_kind(user_text)
        if reply_kind == "other" and _is_secondary_soft_yes(user_text):
            reply_kind = "yes"
        if reply_kind == "yes":
            next_label = queue.pop(0)
            set_secondary_queue(state, queue)
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
            set_secondary_queue(state, [])
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
                "source": cand.source,
                "label": cand.label,
                "confidence": cand.confidence,
                "flags": cand.flags[:8],
            }
            for cand in nlu_result.candidates
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

    topic_match = match_topic(user_text)
    if topic_match is not None:
        decision = _apply_topic_registry_override(decision, topic_match)
        nlu_debug["topic_registry"] = {
            "topic_id": topic_match.topic_id,
            "label": topic_match.label,
            "priority": topic_match.priority,
            "score": topic_match.score,
            "matched_keywords": list(topic_match.matched_keywords),
            "matched_regex": list(topic_match.matched_regex),
        }

    decision = await _verify_doctor_entity(decision, services, user_text)
    # В активном APPOINTMENT flow короткий follow-up с датой/временем
    # считаем продолжением записи до применения context_action.
    if (
        state.last_entities.get("appointment_flow_active")
        and _is_appointment_datetime_followup(user_text)
        and decision.label in {"OTHER", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "TEST_RESULT", "ADDRESS", "PRICE"}
    ):
        decision = _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.68),
            flags=set(decision.flags) | {"flow_datetime_appointment_prelock"},
            needs_handoff=False,
            context_action="continue",
        )
    decision = apply_context_action(decision, state, user_text)
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
                svc = nonbookable_service_hint(user_text, merged_ctx)
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

    # Короткий follow-up после PREPARE (например, "вульвоскопия") держим
    # в подготовке, чтобы не сваливаться обратно в TEST_ASSIST.
    if (
        str(state.last_entities.get("_last_label") or "") == "PREPARE"
        and decision.label in {"OTHER", "TEST_ASSIST", "ADDRESS", "PRICE"}
        and is_short_prepare_followup(user_text)
        and not detect_prepare_intent(user_text)
    ):
        decision = _copy_decision(
            decision,
            label="PREPARE",
            confidence=max(decision.confidence, 0.62),
            flags=set(decision.flags) | {"flow_prepare_followup_override"},
            needs_handoff=False,
            context_action="continue",
        )

    decision = _apply_appointment_continuity_overrides(decision, state, user_text)

    if decision.label == "DOCTOR_SCHEDULE":
        # Очищаем хвосты сценария записи, чтобы расписание не фильтровалось
        # старым branch/date/time из предыдущих шагов.
        reset_appointment_runtime_state(state)
        for k in ("service_name", "test_name"):
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
            (is_appointment_waiting_patient_name(pending_now) and looks_like_patient_fio(user_text))
            or (
                bool(state.last_entities.get("appointment_flow_active"))
                and should_keep_appointment_flow_override(user_text)
            )
        )
        if not keep_appointment_flow:
            reset_appointment_runtime_state(state)

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
    sec_now = normalize_secondary_labels(decision.entities.get("secondary_intents"))
    if sec_now:
        existing = get_secondary_queue(state)
        merged = [x for x in existing if x != decision.label]
        for x in sec_now:
            if x != decision.label and x not in merged:
                merged.append(x)
        set_secondary_queue(state, merged)

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

    fill_date_from_schedule_windows(state, decision.label)

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
    return response_build_price_response(flow_label, evidence, state)


def _build_service_bundle_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    return response_build_service_bundle_response(flow_label, evidence, state)


def _build_doctor_info_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    return response_build_doctor_info_response(flow_label, evidence, state)


def _build_test_result_response(flow_label: str, evidence: Evidence) -> ResponseEnvelope | None:
    return response_build_test_result_response(flow_label, evidence)


def _build_doctor_schedule_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    return response_build_doctor_schedule_response(flow_label, evidence, state)


def _build_address_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    memory: MemoryStore,
    decision: RouteDecision,
    user_text: str,
) -> ResponseEnvelope | None:
    return response_build_address_response(flow_label, evidence, state, memory, decision, user_text)


def _build_news_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    return response_build_news_response(flow_label, evidence, state)


def _build_main_index_info_response(evidence: Evidence) -> ResponseEnvelope | None:
    return response_build_main_index_info_response(evidence)


def _build_appointment_schedule_preview_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
) -> ResponseEnvelope | None:
    return response_build_appointment_schedule_preview_response(flow_label, evidence, state)


def _build_appointment_step_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    return response_build_appointment_step_response(flow_label, evidence, state, services, memory)


def _build_first_structured_response(
    *,
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    decision: RouteDecision,
    user_text: str,
) -> ResponseEnvelope | None:
    return response_build_first_structured_response(
        flow_label=flow_label,
        evidence=evidence,
        state=state,
        services=services,
        memory=memory,
        decision=decision,
        user_text=user_text,
    )


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
        clear_appointment_flow_context(state, memory)
        state.last_entities["_nlu_unclear_count"] = 0
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
        clear_appointment_flow_context(state, memory)
        state.last_entities.pop("city", None)
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

    precheck = run_appointment_precheck(
        user_text=user_text,
        state=state,
        memory=memory,
        debug=debug,
        debug_state_update_factory=_early_debug_state_update,
    )
    if precheck is not None:
        yield precheck
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

    structured_response = _build_first_structured_response(
        flow_label=flow_label,
        evidence=evidence,
        state=state,
        services=services,
        memory=memory,
        decision=decision,
        user_text=user_text,
    )
    if structured_response is not None:
        yield structured_response
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

    secondary = get_secondary_queue(state)
    followup = secondary_followup_text(secondary)
    if (
        followup
        and not state.last_entities.get("_secondary_offer_pending")
        and not decision.needs_handoff
        and flow_label not in {"APPOINTMENT", "URGENT", "COMPLAINT", "MEDICAL_ADVICE", "TEST_RESULT"}
    ):
        state.last_entities["_secondary_offer_pending"] = True
        yield ResponseEnvelope(text=followup, attachments=[], handoff=False)
