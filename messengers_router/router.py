"""Главный оркестратор мессенджерного диалога.

Выполняет pipeline: classify -> slot filling -> plan -> services -> rendering,
ведет APPOINTMENT flow, pending-уточнения, handoff решения и debug meta.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, Any

from .mess_types import Evidence, Plan, PlanStep, ResponseEnvelope, RouteDecision, SessionState
from .classifier import analyze
from .policies import (
    require_auth_for_test_result,
    missing_slots,
    clarification_question,
    evidence_requires_handoff,
    handoff_message,
    appointment_step_policy,
    APPOINTMENT_STEP_BRANCH,
    APPOINTMENT_STEP_DATETIME,
    APPOINTMENT_STEP_CONFIRM,
    APPOINTMENT_CONFIRM_YES,
    APPOINTMENT_CONFIRM_NO,
    appointment_confirmation_transition,
    extract_price_rub,
    appointment_summary,
    appointment_addresses_for_city,
    appointment_text_branch_prompt,
    appointment_text_datetime_prompt,
    appointment_text_confirm_prompt,
    appointment_text_confirmed_handoff,
    appointment_text_reask_datetime,
    appointment_text_reask_confirm,
    doctor_schedule_text_clarify_doctor,
    decision_handoff_text,
    quick_fill_core_entities,
    extract_branch_hint,
    looks_like_branch_hint,
    branch_options_to_indexable,
    build_branch_index,
    match_branch_hint,
)
from .services import Services
from .renderer import (
    render_urgent,
    render_complaint,
    render_medical_advice,
    render_stream,
    format_doctor_schedule_for_patient,
)
from .memory import MemoryStore
from .city import match_city


def _apply_pending_override(decision_label: str, pending: dict | None) -> str:
    if not pending:
        return decision_label
    pending_label = pending.get("label")
    if not isinstance(pending_label, str):
        return decision_label
    if decision_label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return decision_label
    return pending_label


def _normalize_secondary_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        if isinstance(x, str) and x and x not in out:
            out.append(x)
    return out


def _get_secondary_queue(state: SessionState) -> list[str]:
    return _normalize_secondary_labels(state.last_entities.get("_secondary_queue"))


def _set_secondary_queue(state: SessionState, labels: list[str]) -> None:
    clean = _normalize_secondary_labels(labels)
    if clean:
        state.last_entities["_secondary_queue"] = clean
        state.last_entities["secondary_intents"] = clean
    else:
        state.last_entities.pop("_secondary_queue", None)
        state.last_entities.pop("secondary_intents", None)
        state.last_entities.pop("_secondary_offer_pending", None)


_YES_RE = re.compile(r"^\s*(да|ага|угу|ок|окей|хорошо|давайте|конечно)\s*[!.,?]*\s*$", re.I)
_NO_RE = re.compile(r"^\s*(нет|не надо|не нужно|неа|отмена|не хочу)\s*[!.,?]*\s*$", re.I)
_TIME_FRAGMENT_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_APPOINTMENT_WORD_RE = re.compile(r"\b(запис\w*|перен\w*|отмен\w*|при(е|ё)м\w*)\b", re.I)


def _is_affirmative(text: str) -> bool:
    return bool(_YES_RE.match(text or ""))


def _is_negative(text: str) -> bool:
    return bool(_NO_RE.match(text or ""))


def _secondary_followup_text(labels: list[str]) -> str | None:
    if not labels:
        return None
    first = labels[0]
    mapping = {
        "PRICE": "Также вижу вопрос по стоимости. Могу сразу подсказать цену по услуге.",
        "ADDRESS": "Также могу подсказать адрес и режим работы подходящего филиала.",
        "APPOINTMENT": "Также могу помочь с записью на прием после уточнения текущего вопроса.",
        "TEST_ASSIST": "Также могу помочь с подбором анализов под вашу цель.",
        "DOCTOR_SCHEDULE": "Также могу показать расписание нужного врача.",
        "DOCTOR_INFO": "Также могу подсказать, какие врачи принимают по вашему запросу.",
    }
    return mapping.get(first)


def _safe_get_branches(services: Services) -> list[dict[str, str]]:
    """
    Надежный доступ к справочнику филиалов:
    при ошибке интеграции возвращаем пустой список, а не исключение.
    """
    try:
        branches = services.get_branches()
    except Exception:
        return []
    if not isinstance(branches, list):
        return []
    return [b for b in branches if isinstance(b, dict)]


def _hydrate_appointment_context_from_schedule(state: SessionState, schedule_payload: dict[str, Any]) -> None:
    """
    Переносит минимальный контекст из ответа расписания в сценарий записи.
    Нужен для фраз вида "записаться на 09:30" сразу после показа расписания.
    """
    docs = schedule_payload.get("schedule")
    if not isinstance(docs, list) or not docs:
        return

    doctor_fio = ""
    regions: list[str] = []
    windows: list[dict[str, str]] = []
    for doc in docs[:3]:
        if not isinstance(doc, dict):
            continue
        if not doctor_fio:
            fio = str(doc.get("fio") or "").strip()
            if fio:
                doctor_fio = fio
        raw_regions = doc.get("regions") or []
        if isinstance(raw_regions, list):
            for r in raw_regions:
                if not isinstance(r, str):
                    continue
                addr = r.strip()
                if addr and addr not in regions:
                    regions.append(addr)
        schedule_map = doc.get("schedule") or {}
        if isinstance(schedule_map, dict):
            for region_name, days in schedule_map.items():
                branch = str(region_name or "").strip()
                if not isinstance(days, list):
                    continue
                for day in days:
                    if not isinstance(day, dict):
                        continue
                    day_date = str(day.get("date") or "").strip()
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_date):
                        continue
                    raw_slots = day.get("slots") or []
                    if not isinstance(raw_slots, list):
                        continue
                    for slot in raw_slots:
                        slot_s = str(slot or "").strip()[:5]
                        if not re.fullmatch(r"\d{2}:\d{2}", slot_s):
                            continue
                        windows.append({"date": day_date, "time": slot_s, "branch": branch})

    if doctor_fio:
        state.last_entities["doctor_name"] = doctor_fio
    if regions:
        state.last_entities["appointment_branch_options"] = regions[:10]
        current_branch = str(state.last_entities.get("branch_name") or "").strip()
        if len(regions) == 1:
            state.last_entities["branch_name"] = regions[0]
        elif current_branch and current_branch not in regions:
            state.last_entities.pop("branch_name", None)
    if windows:
        uniq: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for w in windows:
            key = (w.get("date", ""), w.get("time", ""), w.get("branch", ""))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(w)
            if len(uniq) >= 300:
                break
        state.last_entities["appointment_windows"] = uniq


def _fill_date_from_schedule_windows(state: SessionState, label: str) -> None:
    """
    Если пациент после показа расписания написал только время (например, 09:00),
    подставляем дату автоматически, когда она однозначна в показанных окнах.
    """
    if label != "APPOINTMENT":
        return
    entities = state.last_entities
    if entities.get("date_from") or entities.get("date_hint"):
        return
    if not (entities.get("doctor_name") or entities.get("doctor_id")):
        return
    time_from = str(entities.get("time_from") or "").strip()[:5]
    if not re.fullmatch(r"\d{2}:\d{2}", time_from):
        return

    windows = entities.get("appointment_windows")
    if not isinstance(windows, list):
        return

    matches: list[tuple[str, str]] = []
    for w in windows:
        if not isinstance(w, dict):
            continue
        t = str(w.get("time") or "").strip()[:5]
        if t != time_from:
            continue
        d = str(w.get("date") or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            continue
        branch = str(w.get("branch") or "").strip()
        matches.append((d, branch))

    if not matches:
        return

    dates = sorted({d for d, _ in matches})
    if len(dates) != 1:
        return

    chosen_date = dates[0]
    entities["date_from"] = chosen_date
    entities["date_to"] = chosen_date

    if not entities.get("branch_name"):
        branches = sorted({b for _, b in matches if b})
        if len(branches) == 1:
            entities["branch_name"] = branches[0]


def quick_fill_entities_from_text(
    text: str,
    state_entities: dict[str, Any],
    missing_rules: list[str],
    services: Services,
) -> dict[str, Any]:
    """
    Пытаемся заполнить частые слоты из текста без LLM.
    + резолв филиала (branch_id)
    + парсинг даты/времени RU
    """
    t = text.strip()
    out: dict[str, Any] = quick_fill_core_entities(t, state_entities, missing_rules)

    # ----------------------------
    # Branch resolution
    # ----------------------------
    if not state_entities.get("branch_id"):
        branch_hint = extract_branch_hint(t, state_entities)

        if branch_hint:
            # Если мы уже показывали список адресов пользователю, резолвим только внутри него.
            shown_options = branch_options_to_indexable(state_entities.get("appointment_branch_options"))
            branches = shown_options
            if not branches:
                branches = _safe_get_branches(services)
                city_hint = str(out.get("city") or state_entities.get("city") or "").strip().lower()
                if city_hint:
                    city_filtered = []
                    for b in branches:
                        if not isinstance(b, dict):
                            continue
                        hay = f"{b.get('name', '')} {b.get('aliases', '')}".lower()
                        if city_hint in hay:
                            city_filtered.append(b)
                    if city_filtered:
                        branches = city_filtered
            idx = build_branch_index(branches)
            bid, bname = match_branch_hint(branch_hint, idx)
            if bid and not str(bid).startswith("shown_"):
                out["branch_id"] = bid
            if bname and not state_entities.get("branch_name"):
                out["branch_name"] = bname
            if (
                not shown_options
                and not bid
                and not state_entities.get("branch_name")
                and looks_like_branch_hint(branch_hint)
            ):
                out["branch_name"] = branch_hint[:80]

    return out


# ----------------------------
# Planning & execution
# ----------------------------

def build_plan(decision: RouteDecision, state: SessionState, user_text: str, memory: MemoryStore) -> Plan:
    pending = memory.get_pending(state)
    effective_label = _apply_pending_override(decision.label, pending)

    entities = state.last_entities
    missing = missing_slots(effective_label, entities)

    if missing:
        memory.set_pending(state, label=effective_label, missing_slots=missing)
        return Plan(label=effective_label, steps=[])

    memory.clear_pending(state)

    label = effective_label
    steps: list[PlanStep] = []

    if label == "TEST_RESULT":
        steps.append(PlanStep(tool="test_result_status", input={"query": user_text, "entities": dict(entities)}, auth="patient_token"))
        steps.append(PlanStep(tool="test_result_pdf", input={"query": user_text, "entities": dict(entities)}, auth="patient_token", required=False))
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
            steps.append(PlanStep(tool="address_info", input={"query": user_text, "entities": dict(entities)}))
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
            elif tool == "test_result_pdf":
                ev.put("test_result_pdf", await services.test_result_pdf(q, ent))
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
) -> tuple[RouteDecision, Plan, Evidence]:
    # Вежливое переключение на вторичный интент по короткому "да/нет".
    queue = _get_secondary_queue(state)
    if (
        state.last_entities.get("_secondary_offer_pending")
        and queue
        and not state.last_entities.get("appointment_confirm_pending")
        and not state.last_entities.get("appointment_flow_active")
    ):
        if _is_affirmative(user_text):
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
        if _is_negative(user_text):
            _set_secondary_queue(state, [])
            state.last_entities["_secondary_offer_pending"] = False
        else:
            # Пользователь продолжил диалог в другом направлении.
            state.last_entities["_secondary_offer_pending"] = False

    decision = await analyze(user_text, state.last_entities)

    # Сохраняем сценарий записи на операторских уточнениях (ветка "да/нет", короткие ответы и т.п.).
    if decision.label == "OTHER" and state.last_entities.get("appointment_flow_active"):
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            entities=decision.entities,
            flags=set(decision.flags) | {"flow_appointment_override"},
            needs_handoff=False,
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

    # merge entities from LLM+rules
    memory.merge_entities(state, decision.entities, label=decision.label)
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
                memory.merge_entities(state, quick_now, label=decision.label)
        elif decision.label == "APPOINTMENT" and state.last_entities.get("appointment_flow_active"):
            # В активном сценарии записи продолжаем извлекать филиал/дату/время
            # даже если формально required slots уже заполнены.
            quick_flow = quick_fill_entities_from_text(
                user_text,
                state.last_entities,
                ["_any_of:city,branch_name,branch_id", "date_from", "time_from"],
                services,
            )
            if quick_flow:
                memory.merge_entities(state, quick_flow, label=decision.label)

    # if pending exists, try quick fill missing slots (NO LLM)
    pending = memory.get_pending(state)
    if pending:
        pend_label = pending.get("label")
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        if isinstance(pend_label, str) and isinstance(missing, list) and missing:
            quick = quick_fill_entities_from_text(user_text, state.last_entities, missing, services)
            if quick:
                memory.merge_entities(state, quick, label=pend_label)

    _fill_date_from_schedule_windows(state, decision.label)

    plan = build_plan(decision, state, user_text, memory=memory)
    evidence = await execute_plan(plan, state, services)
    return decision, plan, evidence



def _debug_meta(decision: RouteDecision, plan: Plan, evidence: Evidence, state: SessionState, pending: Any) -> dict[str, Any]:
    return {
        "decision": {
            "label": decision.label,
            "confidence": decision.confidence,
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
    }


async def patient_routing_stream(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    debug: bool = False,
) -> AsyncGenerator[ResponseEnvelope, None]:
    try:
        decision, plan, evidence = await route_patient_message(user_text, state, services, memory)
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

    if decision.label == "URGENT":
        yield render_urgent()
        return
    if decision.label == "COMPLAINT":
        yield render_complaint()
        return
    if decision.label == "MEDICAL_ADVICE":
        yield render_medical_advice()
        return

    if "doc_request_handoff" in decision.flags:
        yield ResponseEnvelope(
            text=handoff_message("doc_request_handoff"),
            handoff=True,
        )
        return

    if flow_label == "TEST_RESULT":
        yield ResponseEnvelope(
            text=handoff_message("test_result_fallback"),
            handoff=True,
        )
        return

    # APPOINTMENT confirmation loop: после вопроса "Подтверждаете?"
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

    if flow_label == "DOCTOR_SCHEDULE":
        # есть специальность, но нет врача → уточняем
        doctor_known = bool(
            decision.entities.get("doctor_name")
            or decision.entities.get("doctor_last_name")
            or decision.entities.get("last_name")
            or state.last_entities.get("doctor_name")
            or state.last_entities.get("doctor_id")
        )
        if (
                "specialty" in decision.entities
                and not doctor_known
        ):
            yield ResponseEnvelope(
                text=doctor_schedule_text_clarify_doctor(),
                handoff=False,
            )
            return

    pending = memory.get_pending(state)
    if not plan.steps and pending:
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
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
        service_raw = str(entities.get("service_name") or entities.get("test_name") or "").strip()
        if not service_raw or _APPOINTMENT_WORD_RE.search(service_raw) or _TIME_FRAGMENT_RE.search(service_raw):
            doctor_name = str(entities.get("doctor_name") or "").strip()
            specialty = str(entities.get("specialty") or "").strip()
            if doctor_name:
                service = f"приём к врачу {doctor_name}"
            elif specialty:
                service = f"приём к {specialty}"
            else:
                service = "услугу"
        else:
            service = service_raw
        city = str(entities.get("city") or "").strip()

        if appointment_step == APPOINTMENT_STEP_BRANCH:
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

        if appointment_step == APPOINTMENT_STEP_CONFIRM:
            state.last_entities["appointment_confirm_pending"] = True
            summary = appointment_summary(entities)
            yield ResponseEnvelope(text=appointment_text_confirm_prompt(summary), handoff=False)
            return

    attachments: list[dict[str, Any]] = []
    pdf_payload = evidence.get("test_result_pdf")
    if isinstance(pdf_payload, dict) and pdf_payload.get("pdf"):
        attachments.append({"type": "pdf", "name": "Результаты анализов.pdf", "url": pdf_payload["pdf"]})

    try:
        async for chunk in render_stream(user_text, decision, evidence):
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

    if attachments:
        yield ResponseEnvelope(text="", attachments=attachments, handoff=False)

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
