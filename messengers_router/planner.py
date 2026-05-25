"""Планировщик инструментальных шагов для patient-router.

Отвечает только за построение `Plan` по текущему `RouteDecision` и state,
без выполнения сервисных вызовов.
"""

from __future__ import annotations

from typing import Any

from .flow_policy import apply_pending_override
from .llm_mode_policy import RuntimeOptions
from .memory import MemoryStore
from .mess_types import Plan, PlanStep, RouteDecision, SessionState
from .policies import missing_slots
from .topic_registry import extract_topic_id_from_flags, get_topic as topic_registry_get_topic


def _build_other_plan_from_topic_registry(
    decision: RouteDecision,
    user_text: str,
    entities: dict[str, Any],
) -> list[PlanStep]:
    topic_id = extract_topic_id_from_flags(decision.flags)
    if not topic_id:
        return []
    topic = topic_registry_get_topic(topic_id)
    if not isinstance(topic, dict):
        return []
    route = topic.get("route")
    if not isinstance(route, dict):
        return []
    sources = route.get("sources")
    if not isinstance(sources, list):
        return []

    base_input = {"query": user_text, "entities": dict(entities)}
    for source in sources:
        if not isinstance(source, dict):
            continue
        kind = str(source.get("kind") or "").strip().lower()
        if kind == "meili":
            index = str(source.get("index") or "main_index").strip().lower()
            if index == "news":
                return [PlanStep(tool="news_info", input=base_input)]
            return [PlanStep(tool="main_index_info", input=base_input)]

    return []


def build_plan(
    decision: RouteDecision,
    state: SessionState,
    user_text: str,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> Plan:
    pending = memory.get_pending(state)
    effective_label = apply_pending_override(decision, pending, user_text=user_text)

    entities = state.last_entities
    if runtime_options is not None:
        entities = {**entities, "__runtime_llm_mode": runtime_options.llm_mode}
    missing = missing_slots(effective_label, entities)

    # «во сколько/когда прийти сдать кровь» — это PREPARE без конкретного анализа.
    # Не уходим в clarify-петлю «к какому анализу нужна подготовка?», а планируем
    # test_prepare: он короткозамкнётся на адреса филиалов с графиком. Иначе
    # clarify-gate отвечает раньше, чем тул успевает отработать.
    if effective_label == "PREPARE" and missing:
        from .services.prepare import _is_lab_visit_timing_query

        if _is_lab_visit_timing_query(user_text):
            missing = []

    if missing:
        memory.set_pending(state, label=effective_label, missing_slots=missing)
        return Plan(label=effective_label, steps=[])

    memory.clear_pending(state)

    label = effective_label
    steps: list[PlanStep] = []

    if label == "OTHER":
        topic_steps = _build_other_plan_from_topic_registry(decision, user_text, entities)
        if topic_steps:
            return Plan(label=label, steps=topic_steps)

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
        # Демоция: «Расписание уролог» (без конкретной фамилии) — это
        # запрос списка специалистов, не загрузки 8 расписаний скопом.
        # Прежде такие запросы триггерили `_schedule_by_specialty`,
        # который под нагрузкой Nayka API отвечал 70+ секунд (а часто
        # вообще не успевал). При запросе только специальности
        # переключаем на DOCTOR_INFO — мгновенная выдача списка из
        # JSONL-кэша. После выбора конкретного врача расписание
        # запрашивается прицельно за разумное время.
        has_specific_doctor = any(
            str(entities.get(key) or "").strip()
            for key in ("doctor_id", "doctor_name", "doctor", "fio", "last_name", "doctor_last_name")
        )
        if not has_specific_doctor:
            steps.append(PlanStep(tool="doctors_info", input={"query": user_text, "entities": dict(entities)}))
            return Plan(label="DOCTOR_INFO", steps=steps)
        steps.append(PlanStep(tool="doctors_schedule_week", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_INFO":
        steps.append(PlanStep(tool="doctors_info", input={"query": user_text, "entities": dict(entities)}))
        if entities.get("doctor_id") or entities.get("doctor_name"):
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}, required=False))
        return Plan(label=label, steps=steps)

    if label == "APPOINTMENT":
        action = str(entities.get("appointment_action") or "").strip().lower()
        if action == "cancel":
            return Plan(label=label, steps=steps)
        flow_active = bool(state.last_entities.get("appointment_flow_active"))
        selection_mode = str(state.last_entities.get("appointment_selection_mode") or "").strip().lower()
        if entities.get("doctor_id") or entities.get("doctor_name"):
            has_cached_windows = bool(entities.get("appointment_windows"))
            has_selected_datetime = bool(
                (entities.get("date_from") or entities.get("date_hint"))
                and (entities.get("time_from") or entities.get("time_flexible"))
            )
            if not has_cached_windows and not has_selected_datetime:
                steps.append(
                    PlanStep(
                        tool="doctors_schedule_week",
                        input={"query": user_text, "entities": dict(entities)},
                        required=not flow_active,
                    )
                )
        else:
            if selection_mode == "doctor":
                steps.append(
                    PlanStep(
                        tool="doctors_info",
                        input={"query": user_text, "entities": dict(entities)},
                        required=not flow_active,
                    )
                )
                address_entities = dict(entities)
                address_entities["__appointment_mode"] = True
                steps.append(
                    PlanStep(
                        tool="address_info",
                        input={"query": user_text, "entities": address_entities},
                        required=False,
                    )
                )
            else:
                address_entities = dict(entities)
                address_entities["__appointment_mode"] = True
                steps.append(
                    PlanStep(
                        tool="address_info",
                        input={"query": user_text, "entities": address_entities},
                        required=not flow_active,
                    )
                )
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
