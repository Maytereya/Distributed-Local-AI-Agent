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
    doctor_schedule_text_clarify_doctor,
    decision_handoff_text,
    quick_fill_core_entities,
    extract_branch_hint,
    looks_like_branch_hint,
    branch_options_to_indexable,
    build_branch_index,
    match_branch_hint,
    detect_schedule_intent,
    apply_verified_doctor_override,
    detect_nonbookable_walkin_intent,
    nonbookable_service_hint,
    is_context_affirmative,
    is_context_negative,
    appointment_service_display,
)
from .services import Services
from .renderer import (
    render_urgent,
    render_complaint,
    render_medical_advice,
    render_stream,
    format_doctor_schedule_for_patient,
    format_doctor_info_for_patient,
    format_address_for_patient,
)
from .memory import MemoryStore
from .city import match_city


def _should_break_pending(decision: RouteDecision, pending_label: str) -> bool:
    if decision.context_action in {"new_topic", "cancel_flow", "overwrite_doctor"}:
        return True
    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return True
    if decision.label == "TEST_RESULT" and "rule_test_result" in decision.flags:
        return True
    if decision.label != pending_label and decision.label != "OTHER":
        if decision.confidence >= 0.70:
            return True
        if any(str(f).startswith("rule_") for f in decision.flags):
            return True
        if "promoted_from_rule_hints" in decision.flags:
            return True
    return False


def _is_appointment_waiting_patient_name(pending: dict | None) -> bool:
    if not isinstance(pending, dict):
        return False
    if pending.get("label") != "APPOINTMENT":
        return False
    missing = pending.get("missing")
    if not isinstance(missing, list):
        return False
    return "patient_name" in missing


def _apply_pending_override(decision: RouteDecision, pending: dict | None, user_text: str = "") -> str:
    if not pending:
        return decision.label
    pending_label = pending.get("label")
    if not isinstance(pending_label, str):
        return decision.label
    # В шаге добора ФИО пациента не даем случайной переклассификации
    # (например, в TEST_RESULT) перебить активный APPOINTMENT flow.
    if (
        _is_appointment_waiting_patient_name(pending)
        and _looks_like_patient_fio(user_text)
        and decision.context_action == "continue"
        and decision.label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}
    ):
        return "APPOINTMENT"
    if _should_break_pending(decision, pending_label):
        return decision.label
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


_PATIENT_FIO_RE = re.compile(r"^[А-ЯЁа-яё\-]+(?:\s+[А-ЯЁа-яё\-]+){1,2}$")
_PATIENT_FIO_STOPWORDS = {
    "анализ", "анализы", "анализов", "результат", "результаты", "тест", "тесты", "тестов",
    "врач", "врача", "доктор", "расписание", "запись", "прием", "приём", "окна", "слоты",
    "оператор", "город", "филиал", "адрес", "цена", "стоимость", "услуга", "услуги",
    "мне", "нужно", "надо", "хочу", "когда", "где", "какой", "какие", "покажи", "покажите",
    "да", "нет",
}
_OPERATOR_REQUEST_RE = re.compile(r"\b(оператор\w*|соедин\w*.*оператор\w*|жив[оы]м?\s+человек\w*)\b", re.I)
_INTRO_TEXT = (
    "Здравствуйте! Это ИИ-помощник клиники «Наука».\n"
    "Через меня вы можете записаться к врачу, перенести или отменить запись, "
    "узнать результаты анализов и получить информацию об услугах."
)
_LOW_CONF_CLARIFY_TEXT = (
    "Уточните, пожалуйста, запрос чуть подробнее, чтобы я не ошибся: "
    "что именно нужно — запись, расписание врача, стоимость, адрес или результаты анализов?"
)


def _looks_like_patient_fio(text: str) -> bool:
    s = str(text or "").strip()
    if not _PATIENT_FIO_RE.fullmatch(s):
        return False
    tokens = [t for t in s.split() if t]
    if len(tokens) < 2:
        return False
    normalized = [t.lower().replace("ё", "е") for t in tokens]
    if any(t in _PATIENT_FIO_STOPWORDS for t in normalized):
        return False
    return True


def _normalize_doctor_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("ё", "е")
    return re.sub(r"\s+", " ", s)


def _clear_flow_state(state: SessionState) -> None:
    for k in (
        "appointment_flow_active",
        "appointment_confirm_pending",
        "appointment_confirmed",
        "appointment_windows",
        "appointment_branch_options",
        "date_from",
        "date_to",
        "time_from",
        "time_to",
        "date_hint",
        "_pending",
    ):
        state.last_entities.pop(k, None)


def _clear_topic_state(state: SessionState) -> None:
    _clear_flow_state(state)
    for k in (
        "doctor_id",
        "doctor_name",
        "specialty",
        "service_name",
        "test_name",
        "branch_id",
        "branch_name",
        "secondary_intents",
        "_secondary_queue",
        "_secondary_offer_pending",
    ):
        state.last_entities.pop(k, None)


def _apply_context_action(decision: RouteDecision, state: SessionState, user_text: str) -> RouteDecision:
    action = decision.context_action
    if action == "cancel_flow":
        _clear_flow_state(state)
        return decision

    if action == "new_topic":
        _clear_topic_state(state)
        return decision

    if action != "overwrite_doctor":
        return decision

    extracted_doctor = str(decision.entities.get("doctor_name") or "").strip()
    if not extracted_doctor:
        return decision

    current = _normalize_doctor_key(state.last_entities.get("doctor_name"))
    target = _normalize_doctor_key(extracted_doctor)
    if current and target and current == target:
        return decision

    _clear_flow_state(state)
    entities = dict(decision.entities)
    entities["doctor_name"] = extracted_doctor
    sanitized_flags = set(decision.flags)
    sanitized_flags.discard("handoff_recommended")
    sanitized_flags.discard("low_confidence")
    sanitized_flags.discard("ollama_timeout")

    # Если пользователь сменил врача в процессе и при этом явно просит расписание
    # (или до этого уже был schedule-контекст), приоритезируем DOCTOR_SCHEDULE.
    prev_label = str(state.last_entities.get("_last_label") or "")
    if decision.label == "OTHER" and (prev_label == "DOCTOR_SCHEDULE" or detect_schedule_intent(user_text or "")):
        return RouteDecision(
            label="DOCTOR_SCHEDULE",
            confidence=max(decision.confidence, 0.74),
            entities=entities,
            flags=sanitized_flags | {"context_action_overwrite_doctor"},
            needs_handoff=False,
            context_action=action,
        )

    return RouteDecision(
        label=decision.label,
        confidence=decision.confidence,
        entities=entities,
        flags=sanitized_flags | {"context_action_overwrite_doctor"},
        needs_handoff=decision.needs_handoff,
        context_action=action,
    )


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
        if is_context_affirmative(user_text):
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
        if is_context_negative(user_text):
            _set_secondary_queue(state, [])
            state.last_entities["_secondary_offer_pending"] = False
        else:
            # Пользователь продолжил диалог в другом направлении.
            state.last_entities["_secondary_offer_pending"] = False

    decision = await analyze(user_text, state.last_entities)
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
    ):
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            entities=decision.entities,
            flags=set(decision.flags) | {"flow_appointment_override"},
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
        keep_appointment_flow = _is_appointment_waiting_patient_name(pending_now) and _looks_like_patient_fio(user_text)
        if not keep_appointment_flow:
            for k in ("appointment_flow_active", "appointment_confirm_pending", "appointment_confirmed"):
                state.last_entities.pop(k, None)

    # merge entities from LLM+rules
    memory.merge_entities(state, decision.entities, label=decision.label)
    # Явный город в текущей реплике должен уметь исправлять/обновлять контекст
    # даже если city уже был заполнен ранее неверно.
    city_hint_now = match_city(user_text)
    if city_hint_now and decision.label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        memory.merge_entities(state, {"city": city_hint_now}, label=decision.label)
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

    # Явный запрос оператора должен иметь абсолютный приоритет
    # над pending/clarify сценариями, чтобы не зацикливать пользователя.
    if _OPERATOR_REQUEST_RE.search(user_text or ""):
        memory.clear_pending(state)
        state.last_entities["_nlu_unclear_count"] = 0
        state.last_entities["appointment_flow_active"] = False
        yield ResponseEnvelope(
            text=handoff_message("manual_operator"),
            attachments=[],
            handoff=True,
        )
        return

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
    if (
        "low_confidence" in decision.flags
        and decision.label in {"OTHER", "TEST_ASSIST"}
        and flow_label == decision.label
        and not pending_now
        and not flow_active
    ):
        if _OPERATOR_REQUEST_RE.search(user_text or ""):
            state.last_entities["_nlu_unclear_count"] = 0
            yield ResponseEnvelope(text=handoff_message("low_confidence"), handoff=True)
            return

        prev = state.last_entities.get("_nlu_unclear_count")
        try:
            n = int(prev) if prev is not None else 0
        except Exception:
            n = 0
        n += 1
        state.last_entities["_nlu_unclear_count"] = n

        if n >= 3:
            state.last_entities["_nlu_unclear_count"] = 0
            yield ResponseEnvelope(text=handoff_message("low_confidence"), handoff=True)
            return

        yield ResponseEnvelope(text=_LOW_CONF_CLARIFY_TEXT, handoff=False)
        return
    else:
        state.last_entities["_nlu_unclear_count"] = 0

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

    # Если пользователь сразу хочет записаться к конкретному врачу, сначала
    # показываем его актуальные окна, а не отправляем в общий сценарий "город -> филиал".
    if (
        flow_label == "APPOINTMENT"
        and isinstance(schedule_payload, dict)
        and isinstance(schedule_payload.get("schedule"), list)
        and bool(schedule_payload.get("schedule"))
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
