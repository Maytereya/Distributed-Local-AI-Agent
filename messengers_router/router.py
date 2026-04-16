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

import logging
import re
from typing import AsyncGenerator, Any

from .mess_types import AppointmentPhase, Evidence, Plan, ResponseEnvelope, RouteDecision, SessionState
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
    is_city_only_reply,
    is_short_prepare_followup,
    looks_like_patient_fio,
    normalize_secondary_labels,
    prelock_active_appointment_turn,
    quick_fill_entities_from_text,
    set_secondary_queue,
)
from .appointment_flow_guard import (
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
from .russian_nlu import normalize_ru
from .policies import (
    missing_slots,
    handoff_message,
    apply_verified_doctor_override,
    detect_nonbookable_walkin_intent,
    detect_appointment_action,
    detect_test_assist_intent,
    detect_test_result_intent,
    detect_prepare_intent,
    detect_price_intent,
    detect_address_intent,
    detect_doc_request_intent,
    detect_schedule_intent,
    detect_doctor_info_intent,
    has_datetime_signal,
    normalize_appointment_action,
    nonbookable_service_hint,
    service_name_conflicts_with_doctor,
)
from .recovery_policy import contextual_reply_kind, explicit_operator_requested
from .services import Services, match_compound_price_service_option
from .memory import MemoryStore
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
logger = logging.getLogger(__name__)
_SECONDARY_SOFT_YES_RE = re.compile(
    r"^\s*(?:(?:да|ок|окей|хорошо|ладно)(?:\s+(?:спасибо|благодарю|благодарствую|спс))?|"
    r"(?:спасибо|благодарю|благодарствую|спс))\s*[!.,?;:]*\s*$",
    re.I,
)
_CATALOG_CONFIRM_STATE_KEY = "_catalog_confirm_pending"
_CATALOG_CONFIRM_REJECTS_KEY = "_catalog_confirm_rejects"
_CATALOG_CONFIRM_MAX_REJECTS = 2
_COMPOUND_PRICE_PENDING_KEY = "_compound_price_pending"
_COMPOUND_PRICE_SOFT_YES_RE = re.compile(
    r"^\s*(?:хочу|можно|давай|давайте|покажи|покажите)\b",
    re.I,
)

# Backward-compat alias for tests/internal callers.
_should_keep_appointment_flow_override = should_keep_appointment_flow_override


def _unsupported_catalog_kind(flags: set[str]) -> str | None:
    """
    Возвращает subtype детерминированного unavailable-кейса из набора флагов.

    :param flags: флаги RouteDecision
    :return: тип unavailable-кейса или None
    """

    for kind in ("unsupported_service", "unsupported_specialist", "unsupported_document_service"):
        if kind in flags:
            return kind
    return None


def _catalog_health_requirements(
    decision: RouteDecision,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[bool, bool]:
    """
    Определяет, какие каталоги обязательны для текущего решения.

    :return: (need_service_catalog, need_doctors_catalog)
    """

    label = str(decision.label or "").strip().upper()
    entities = dict(state.last_entities or {})
    entities.update(decision.entities or {})

    has_doctor_context = any(
        str(entities.get(k) or "").strip()
        for k in (
            "doctor_id",
            "doctor_name",
            "last_name",
            "doctor_last_name",
            "specialty",
            "_catalog_doctor_candidate",
        )
    )
    has_service_context = any(
        str(entities.get(k) or "").strip()
        for k in ("service_name", "test_name", "_catalog_service_candidate")
    )

    if label == "DOCTOR_INFO":
        return False, True

    if label == "DOCTOR_SCHEDULE":
        # Если врач уже однозначно определен, расписание можно пробовать получить
        # без обязательного catalog-lookup.
        return False, not has_doctor_context

    if label == "APPOINTMENT":
        pending = memory.get_pending(state)
        if isinstance(pending, dict) and pending.get("label") == "APPOINTMENT":
            missing = pending.get("missing")
            if isinstance(missing, list):
                core_slots = (
                    "doctor_id",
                    "doctor_name",
                    "last_name",
                    "doctor_last_name",
                    "specialty",
                    "service_name",
                    "test_name",
                )
                needs_core_lookup = any(
                    isinstance(item, str) and any(slot in item for slot in core_slots)
                    for item in missing
                )
                if not needs_core_lookup:
                    return False, False
        need_doctors = has_doctor_context or not has_service_context
        need_service = has_service_context or not has_doctor_context
        return need_service, need_doctors

    if label == "PRICE":
        pending = memory.get_pending(state)
        if isinstance(pending, dict) and pending.get("label") == "PRICE":
            missing = pending.get("missing")
            if isinstance(missing, list):
                core_slots = (
                    "doctor_id",
                    "doctor_name",
                    "last_name",
                    "doctor_last_name",
                    "specialty",
                    "service_name",
                    "test_name",
                )
                needs_core_lookup = any(
                    isinstance(item, str) and any(slot in item for slot in core_slots)
                    for item in missing
                )
                if not needs_core_lookup:
                    return False, False
        if has_doctor_context:
            return False, True
        return True, False

    if label == "ADDRESS":
        # ADDRESS может корректно работать по уже выбранному контексту без
        # catalog-grounding (например, follow-up по филиалам/забору).
        return False, False

    return False, False


def _catalog_health_degraded_text(
    *,
    need_service_catalog: bool,
    need_doctors_catalog: bool,
) -> str:
    if need_service_catalog and need_doctors_catalog:
        return (
            "Сейчас временно недоступен каталог услуг и врачей. "
            "Я не смогу надежно подобрать услугу или врача. "
            "Попробуйте повторить запрос через 5 минут или напишите «оператор»."
        )
    if need_service_catalog:
        return (
            "Сейчас временно недоступен каталог услуг. "
            "Я не смогу надежно подобрать нужную услугу. "
            "Попробуйте повторить запрос через 5 минут или напишите «оператор»."
        )
    if need_doctors_catalog:
        return (
            "Сейчас временно недоступен каталог врачей. "
            "Я не смогу надежно подобрать врача или расписание. "
            "Попробуйте повторить запрос через 5 минут или напишите «оператор»."
        )
    return (
        "Сейчас временно недоступен каталог клиники. "
        "Попробуйте повторить запрос через 5 минут или напишите «оператор»."
    )


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


async def _resolve_secondary_queue_doctor_reply(
    user_text: str,
    next_label: str,
    services: Services,
) -> dict[str, Any]:
    """
    Пытается извлечь выбранного врача из краткого follow-up по secondary-очереди.

    :param user_text: текущая реплика пользователя
    :param next_label: активируемый secondary intent
    :param services: сервисный слой
    :return: сущности для вторичного интента
    """

    if next_label not in {"DOCTOR_SCHEDULE", "DOCTOR_INFO", "APPOINTMENT"}:
        return {}

    raw_text = str(user_text or "").strip()
    probes = [raw_text]
    for sep in (",", "—", "-", ";"):
        if sep in raw_text:
            head = str(raw_text.split(sep, 1)[0] or "").strip()
            if head:
                probes.append(head)

    seen: set[str] = set()
    for probe in probes:
        key = probe.lower()
        if not probe or key in seen:
            continue
        seen.add(key)
        resolved = await services.resolve_doctor_name(probe)
        if resolved:
            return {"doctor_name": resolved}
    return {}


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


def _clear_operator_offer_pending(state: SessionState, memory: MemoryStore) -> None:
    """
    Сбрасывает локальный pending-флаг подтверждения перевода на оператора.

    :param state: текущее состояние сессии
    :param memory: хранилище pending-слотов
    :return: None
    """

    state.last_entities.pop("_operator_offer_pending", None)
    pending = memory.get_pending(state)
    if isinstance(pending, dict):
        missing = pending.get("missing")
        if pending.get("label") == "OTHER" and isinstance(missing, list) and "operator_offer_confirm" in missing:
            memory.clear_pending(state)


def _get_catalog_confirm_pending(state: SessionState) -> dict[str, Any] | None:
    payload = state.last_entities.get(_CATALOG_CONFIRM_STATE_KEY)
    if not isinstance(payload, dict):
        return None
    canonical = str(payload.get("canonical") or "").strip()
    entity_key = str(payload.get("entity_key") or "").strip()
    label = str(payload.get("label") or "").strip()
    if not canonical or entity_key not in {"doctor_name", "service_name"} or not label:
        return None
    kind = str(payload.get("kind") or "").strip().lower()
    if kind not in {"doctor", "service"}:
        return None
    return payload


def _clear_catalog_confirm_pending(state: SessionState, memory: MemoryStore) -> None:
    state.last_entities.pop(_CATALOG_CONFIRM_STATE_KEY, None)
    pending = memory.get_pending(state)
    if not isinstance(pending, dict):
        return
    missing = pending.get("missing")
    if pending.get("label") == "OTHER" and isinstance(missing, list) and "catalog_confirm" in missing:
        memory.clear_pending(state)


def _get_compound_price_pending(state: SessionState) -> dict[str, Any] | None:
    """
    Возвращает валидный pending-контекст compound PRICE-уточнения.

    :param state: состояние сессии
    :return: payload с услугами либо None
    """

    payload = state.last_entities.get(_COMPOUND_PRICE_PENDING_KEY)
    if not isinstance(payload, dict):
        return None
    services = [str(item).strip() for item in (payload.get("services") or []) if str(item).strip()]
    default_service = str(payload.get("default_service") or "").strip()
    if len(services) < 2:
        return None
    if default_service not in services:
        default_service = services[0]
    return {
        "services": services[:4],
        "default_service": default_service,
    }


def _clear_compound_price_pending(state: SessionState) -> None:
    """
    Сбрасывает transient-состояние compound PRICE-уточнения.

    :param state: состояние сессии
    :return: None
    """

    state.last_entities.pop(_COMPOUND_PRICE_PENDING_KEY, None)


def _is_appointment_action_pending(pending: dict[str, Any] | None) -> bool:
    """
    Проверяет, что pending ждет выбор действия записи: отмена или перенос.

    :param pending: pending-объект из memory
    :return: True, если активен шаг выбора appointment_action
    """

    if not isinstance(pending, dict):
        return False
    if pending.get("label") != "APPOINTMENT":
        return False
    missing = pending.get("missing")
    return isinstance(missing, list) and "appointment_action" in missing


async def _handle_appointment_action_pending(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """
    Детерминированно обрабатывает шаг выбора действия записи.

    Это узкий pre-NLU handler для pending `appointment_action`, чтобы короткие
    ответы вроде `нет` или `перенести` не уезжали в общий OTHER-flow.

    :param user_text: текущая реплика пользователя
    :param state: состояние сессии
    :param services: сервисный слой
    :param memory: хранилище pending/state
    :param runtime_options: runtime-параметры роутера
    :return: decision/plan/evidence либо None, если шаг не активен
    """

    pending = memory.get_pending(state)
    if not _is_appointment_action_pending(pending):
        return None

    if explicit_operator_requested(user_text):
        return (
            RouteDecision(
                label="APPOINTMENT",
                confidence=0.95,
                entities={},
                flags={"manual_operator", "appointment_action_pending_operator"},
                needs_handoff=False,
                source="appointment_action_pending",
            ),
            Plan(label="APPOINTMENT"),
            Evidence(
                items={
                    "operator_offer_response": {
                        "text": handoff_message("manual_operator"),
                        "handoff": True,
                    }
                }
            ),
        )

    normalized_action = normalize_appointment_action(detect_appointment_action(user_text), user_text)
    if normalized_action in {"cancel", "reschedule"}:
        memory.merge_entities(state, {"appointment_action": normalized_action}, label="APPOINTMENT")
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.95,
            entities={"appointment_action": normalized_action},
            flags={"appointment_action_selected"},
            needs_handoff=False,
            context_action="continue",
            source="appointment_action_pending",
        )
        plan = build_plan(decision, state, user_text, memory=memory, runtime_options=runtime_options)
        evidence = await execute_plan(plan, state, services)
        return decision, plan, evidence

    if contextual_reply_kind(user_text) == "no":
        return (
            RouteDecision(
                label="APPOINTMENT",
                confidence=0.95,
                entities={},
                flags={"appointment_action_declined"},
                needs_handoff=False,
                source="appointment_action_pending",
            ),
            Plan(label="APPOINTMENT"),
            Evidence(
                items={
                    "operator_offer_response": {
                        "text": handoff_message("manual_operator"),
                        "handoff": True,
                    }
                }
            ),
        )

    return (
        RouteDecision(
            label="APPOINTMENT",
            confidence=0.9,
            entities={},
            flags={"appointment_action_reask"},
            needs_handoff=False,
            source="appointment_action_pending",
        ),
        Plan(label="APPOINTMENT"),
        Evidence(
            items={
                "operator_offer_response": {
                    "text": "Хотите отменить или перенести запись?",
                    "handoff": False,
                }
            }
        ),
    )


async def _handle_compound_price_pending(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """
    Обрабатывает follow-up после compound PRICE-уточнения.

    Поведение узкое и предсказуемое:
    - `да/хочу/можно` -> берем первую услугу из списка;
    - явное название одной из услуг -> берем выбранную услугу;
    - `нет` -> просим назвать услугу явно;
    - любая другая новая тема -> просто снимаем pending и отдаем ход общему NLU.

    :param user_text: текущая реплика пользователя
    :param state: состояние сессии
    :param services: сервисный слой
    :param memory: хранилище state/pending
    :param runtime_options: runtime-параметры роутера
    :return: decision/plan/evidence либо None
    """

    pending = _get_compound_price_pending(state)
    if not pending:
        return None

    if explicit_operator_requested(user_text):
        _clear_compound_price_pending(state)
        return None

    options = list(pending.get("services") or [])
    default_service = str(pending.get("default_service") or options[0]).strip()
    reply_kind = contextual_reply_kind(user_text)
    if reply_kind == "other" and (
        _is_secondary_soft_yes(user_text) or _COMPOUND_PRICE_SOFT_YES_RE.match(str(user_text or ""))
    ):
        reply_kind = "yes"

    if reply_kind == "yes":
        selected_service = default_service
    else:
        selected_service = match_compound_price_service_option(user_text, options)

    if selected_service:
        _clear_compound_price_pending(state)
        set_secondary_queue(state, [])
        state.last_entities["_secondary_offer_pending"] = False
        state.last_entities.pop("secondary_intents", None)
        memory.merge_entities(state, {"service_name": selected_service}, label="PRICE")
        decision = RouteDecision(
            label="PRICE",
            confidence=0.92,
            entities={"service_name": selected_service, "compound_price_selected": True},
            flags={"compound_price_selected"},
            needs_handoff=False,
            context_action="continue",
            source="compound_price",
        )
        plan = build_plan(decision, state, user_text, memory=memory, runtime_options=runtime_options)
        evidence = await execute_plan(plan, state, services)
        return decision, plan, evidence

    if reply_kind == "no":
        _clear_compound_price_pending(state)
        set_secondary_queue(state, [])
        state.last_entities["_secondary_offer_pending"] = False
        options_text = " или ".join(f"«{item}»" for item in options[:3])
        return (
            RouteDecision(
                label="PRICE",
                confidence=0.9,
                entities={},
                flags={"compound_price_declined"},
                needs_handoff=False,
                source="compound_price",
            ),
            Plan(label="PRICE"),
            Evidence(
                items={
                    "service_bundle": {
                        "clarify_text": (
                            f"Хорошо. Тогда напишите, какую из услуг проверить первой: {options_text}."
                        )
                    }
                }
            ),
        )

    _clear_compound_price_pending(state)
    set_secondary_queue(state, [])
    state.last_entities["_secondary_offer_pending"] = False
    return None


def _catalog_confirm_prompt(kind: str, query: str, canonical: str) -> str:
    query_text = str(query or "").strip() or "ваш запрос"
    if kind == "doctor":
        return (
            f"Похоже, вы имели в виду врача «{canonical}» (по запросу «{query_text}»). "
            "Это верно? Ответьте «да» или «нет»."
        )
    return (
        f"Похоже, вы имели в виду услугу «{canonical}» (по запросу «{query_text}»). "
        "Это верно? Ответьте «да» или «нет»."
    )


def _catalog_refine_missing_slots(kind: str, label: str) -> list[str]:
    if kind == "doctor":
        if label == "APPOINTMENT":
            return ["_any_of:doctor_id,doctor_name,specialty,service_name"]
        if label in {"DOCTOR_SCHEDULE", "DOCTOR_INFO"}:
            return ["_any_of:doctor_id,doctor_name,specialty"]
        if label == "PRICE":
            return ["_any_of:doctor_name,service_name"]
        return ["doctor_name"]
    if label == "APPOINTMENT":
        return ["_any_of:doctor_id,doctor_name,specialty,service_name"]
    if label == "PRICE":
        return ["service_name"]
    if label == "ADDRESS":
        return ["_any_of:service_name,branch_name,branch_id,city"]
    return ["service_name"]


def _catalog_refine_prompt(kind: str, label: str) -> str:
    if kind == "doctor":
        if label == "APPOINTMENT":
            return "Хорошо. Уточните, пожалуйста, фамилию врача для записи."
        if label == "DOCTOR_SCHEDULE":
            return "Хорошо. Уточните, пожалуйста, фамилию врача, чтобы показать расписание."
        return "Хорошо. Уточните, пожалуйста, фамилию врача."
    if label == "PRICE":
        return "Хорошо. Уточните, пожалуйста, точное название услуги или анализа для расчёта цены."
    if label == "APPOINTMENT":
        return "Хорошо. Уточните, пожалуйста, точное название услуги для записи."
    return "Хорошо. Уточните, пожалуйста, точное название услуги."


async def _handle_catalog_confirm_pending(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    pending = _get_catalog_confirm_pending(state)
    if not pending:
        return None

    label = str(pending.get("label") or "OTHER").strip()
    kind = str(pending.get("kind") or "").strip().lower()
    entity_key = str(pending.get("entity_key") or "").strip()
    canonical = str(pending.get("canonical") or "").strip()

    if explicit_operator_requested(user_text):
        _clear_catalog_confirm_pending(state, memory)
        state.last_entities.pop(_CATALOG_CONFIRM_REJECTS_KEY, None)
        return (
            RouteDecision(
                label="OTHER",
                confidence=0.95,
                entities={},
                flags={"catalog_confirm_operator"},
                needs_handoff=False,
                source="catalog_confirm",
            ),
            Plan(label="OTHER"),
            Evidence(
                items={
                    "catalog_confirm_response": {
                        "text": handoff_message("manual_operator"),
                        "handoff": True,
                    }
                }
            ),
        )

    reply_kind = contextual_reply_kind(user_text)
    if reply_kind == "yes":
        _clear_catalog_confirm_pending(state, memory)
        state.last_entities.pop(_CATALOG_CONFIRM_REJECTS_KEY, None)
        memory.merge_entities(state, {entity_key: canonical}, label=label)
        if entity_key == "doctor_name":
            service_name = str(state.last_entities.get("service_name") or "").strip()
            if service_name and service_name_conflicts_with_doctor(service_name, canonical):
                state.last_entities.pop("service_name", None)
        decision = RouteDecision(
            label=label,  # type: ignore[arg-type]
            confidence=0.93,
            entities={entity_key: canonical},
            flags={"catalog_confirmed"},
            needs_handoff=False,
            context_action="continue",
            source="catalog_confirm",
        )
        plan = build_plan(decision, state, user_text, memory=memory, runtime_options=runtime_options)
        evidence = await execute_plan(plan, state, services)
        return decision, plan, evidence

    if reply_kind == "no":
        _clear_catalog_confirm_pending(state, memory)
        rejects = int(state.last_entities.get(_CATALOG_CONFIRM_REJECTS_KEY) or 0) + 1
        state.last_entities[_CATALOG_CONFIRM_REJECTS_KEY] = rejects
        if rejects >= _CATALOG_CONFIRM_MAX_REJECTS:
            state.last_entities.pop(_CATALOG_CONFIRM_REJECTS_KEY, None)
            return (
                RouteDecision(
                    label="OTHER",
                    confidence=0.95,
                    entities={},
                    flags={"catalog_confirm_rejected_handoff"},
                    needs_handoff=False,
                    source="catalog_confirm",
                ),
                Plan(label="OTHER"),
                Evidence(
                    items={
                        "catalog_confirm_response": {
                            "text": "Не удалось точно сопоставить запрос с каталогом. Соединяю с оператором.",
                            "handoff": True,
                        }
                    }
                ),
            )
        missing_slots = _catalog_refine_missing_slots(kind, label)
        memory.set_pending(state, label=label, missing_slots=missing_slots)
        return (
            RouteDecision(
                label="OTHER",
                confidence=0.9,
                entities={},
                flags={"catalog_confirm_rejected"},
                needs_handoff=False,
                source="catalog_confirm",
            ),
            Plan(label="OTHER"),
            Evidence(
                items={
                    "catalog_confirm_response": {
                        "text": _catalog_refine_prompt(kind, label),
                        "handoff": False,
                    }
                }
            ),
        )

    return (
        RouteDecision(
            label="OTHER",
            confidence=0.9,
            entities={},
            flags={"catalog_confirm_reask"},
            needs_handoff=False,
            source="catalog_confirm",
        ),
        Plan(label="OTHER"),
        Evidence(
            items={
                "catalog_confirm_response": {
                    "text": _catalog_confirm_prompt(
                        kind=kind,
                        query=str(pending.get("query") or ""),
                        canonical=canonical,
                    ),
                    "handoff": False,
                }
            }
        ),
    )


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
    return normalize_ru(city) == "самара"


def _reset_state_after_handoff(state: SessionState, memory: MemoryStore) -> None:
    """
    Сбрасывает transient/focus state после передачи диалога оператору.
    Сохраняем только устойчивый профильный контекст города (Самара).
    """

    city = str(state.last_entities.get("city") or "").strip()
    keep_city = city if _is_samara_city(city) else ""
    memory.clear_pending(state)
    state.last_entities.clear()
    if keep_city:
        state.last_entities["city"] = keep_city


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


def _is_short_verified_doctor_followup(decision: RouteDecision, user_text: str) -> bool:
    """
    Определяет, что пользователь короткой репликой выбрал конкретного врача из списка.

    :param decision: текущее решение маршрутизатора
    :param user_text: исходный текст пользователя
    :return: True, если это короткий follow-up по врачу
    """

    text = str(user_text or "").strip()
    if not text or len(text) > 64:
        return False
    if "doctor_name_verified" not in set(decision.flags):
        return False
    if not any(decision.entities.get(k) for k in ("doctor_name", "doctor_id", "last_name", "doctor_last_name")):
        return False
    if detect_prepare_intent(text) or detect_price_intent(text) or detect_address_intent(text) or detect_doc_request_intent(text):
        return False
    if detect_test_result_intent(text) or detect_test_assist_intent(text):
        return False
    if has_datetime_signal(text):
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
    if state.dialog.phase not in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM):
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


async def _inject_catalog_candidates(
    decision: RouteDecision,
    *,
    user_text: str,
    state: SessionState,
    services: Services,
) -> RouteDecision:
    entities = dict(decision.entities or {})
    flags = set(decision.flags or set())
    updated = False

    should_try_doctor = (
        decision.label in {"APPOINTMENT", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "PRICE"}
        and not entities.get("doctor_name")
        and (
            "doctor_name_unverified" in flags
            or decision.context_action == "overwrite_doctor"
        )
    )
    if should_try_doctor:
        doctor_match = await services.match_catalog_doctor(
            str(entities.get("doctor_name") or user_text or ""),
        )
        status = str(doctor_match.get("status") or "")
        canonical = str(doctor_match.get("canonical") or "").strip()
        if status == "exact" and canonical:
            entities["doctor_name"] = canonical
            flags.discard("doctor_name_unverified")
            flags.add("doctor_name_verified")
            flags.add("catalog_doctor_exact")
            updated = True
        elif status == "fuzzy" and canonical:
            entities["_catalog_doctor_candidate"] = canonical
            entities["_catalog_doctor_query"] = str(doctor_match.get("query") or "").strip()
            flags.add("catalog_doctor_fuzzy_candidate")
            updated = True

    should_try_service = (
        decision.label in {"APPOINTMENT", "PRICE", "ADDRESS", "TEST_ASSIST"}
        and not entities.get("service_name")
    )
    if should_try_service:
        if (
            decision.label == "PRICE"
            and (entities.get("specialty") or state.last_entities.get("specialty"))
            and not entities.get("doctor_name")
        ):
            should_try_service = False
        if (
            decision.label == "APPOINTMENT"
            and (entities.get("doctor_name") or state.last_entities.get("doctor_name"))
        ):
            should_try_service = False
    if should_try_service:
        service_match = await services.match_catalog_service(
            str(entities.get("service_name") or entities.get("test_name") or user_text or ""),
            current_service_name=str(state.last_entities.get("service_name") or ""),
        )
        status = str(service_match.get("status") or "")
        canonical = str(service_match.get("canonical") or "").strip()
        if status == "exact" and canonical:
            entities["service_name"] = canonical
            entities.pop("test_name", None)
            flags.add("catalog_service_exact")
            updated = True
        elif status == "fuzzy" and canonical:
            entities["_catalog_service_candidate"] = canonical
            entities["_catalog_service_query"] = str(service_match.get("query") or "").strip()
            flags.add("catalog_service_fuzzy_candidate")
            updated = True

    if not updated:
        return decision
    return _copy_decision(
        decision,
        entities=entities,
        flags=flags,
    )


def _maybe_start_catalog_confirm(
    *,
    decision: RouteDecision,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    entities = dict(decision.entities or {})
    doctor_candidate = str(entities.get("_catalog_doctor_candidate") or "").strip()
    service_candidate = str(entities.get("_catalog_service_candidate") or "").strip()
    if not doctor_candidate and not service_candidate:
        return None

    if doctor_candidate:
        kind = "doctor"
        entity_key = "doctor_name"
        canonical = doctor_candidate
        query = str(entities.get("_catalog_doctor_query") or "").strip()
    else:
        kind = "service"
        entity_key = "service_name"
        canonical = service_candidate
        query = str(entities.get("_catalog_service_query") or "").strip()

    state.last_entities[_CATALOG_CONFIRM_STATE_KEY] = {
        "kind": kind,
        "label": decision.label,
        "entity_key": entity_key,
        "canonical": canonical,
        "query": query,
    }
    memory.clear_pending(state)
    memory.set_pending(state, label="OTHER", missing_slots=["catalog_confirm"])

    return (
        _copy_decision(
            decision,
            label="OTHER",
            entities={},
            flags=set(decision.flags) | {"catalog_confirm_requested"},
            source="catalog_confirm",
            confidence=max(decision.confidence, 0.9),
            needs_handoff=False,
        ),
        Plan(label="OTHER"),
        Evidence(
            items={
                "catalog_confirm_response": {
                    "text": _catalog_confirm_prompt(kind=kind, query=query, canonical=canonical),
                    "handoff": False,
                }
            }
        ),
    )


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
        svc_norm = " ".join(normalize_ru(svc).split())
        raw_norm = " ".join(normalize_ru(raw).split())
        resolved_norm = " ".join(normalize_ru(resolved).split())
        if svc_norm in {raw_norm, resolved_norm}:
            state.last_entities.pop("service_name", None)

# ----------------------------
# Planning & execution
# ----------------------------

def build_plan(
    decision: RouteDecision,
    state: SessionState,
    user_text: str,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> Plan:
    return planner_build_plan(decision, state, user_text, memory, runtime_options=runtime_options)


async def execute_plan(plan: Plan, state: SessionState, services: Services) -> Evidence:
    return await executor_execute_plan(plan, state, services)


async def route_patient_message(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence]:
    if state.last_entities.get("_operator_offer_pending"):
        reply_kind = contextual_reply_kind(user_text)
        if explicit_operator_requested(user_text):
            reply_kind = "yes"
        if reply_kind == "yes":
            _clear_operator_offer_pending(state, memory)
            return (
                RouteDecision(
                    label="OTHER",
                    confidence=0.95,
                    entities={},
                    flags={"operator_offer_confirmed"},
                    needs_handoff=False,
                ),
                Plan(label="OTHER"),
                Evidence(items={"operator_offer_response": {"text": handoff_message("manual_operator"), "handoff": True}}),
            )
        if reply_kind == "no":
            _clear_operator_offer_pending(state, memory)
            return (
                RouteDecision(
                    label="OTHER",
                    confidence=0.95,
                    entities={},
                    flags={"operator_offer_declined"},
                    needs_handoff=False,
                ),
                Plan(label="OTHER"),
                Evidence(
                    items={
                        "operator_offer_response": {
                            "text": "Хорошо, продолжаем диалог. Можете задать другой вопрос.",
                            "handoff": False,
                        }
                    }
                ),
            )
        _clear_operator_offer_pending(state, memory)

    catalog_pending_result = await _handle_catalog_confirm_pending(
        user_text=user_text,
        state=state,
        services=services,
        memory=memory,
        runtime_options=runtime_options,
    )
    if catalog_pending_result is not None:
        return catalog_pending_result

    appointment_action_result = await _handle_appointment_action_pending(
        user_text=user_text,
        state=state,
        services=services,
        memory=memory,
        runtime_options=runtime_options,
    )
    if appointment_action_result is not None:
        return appointment_action_result

    compound_price_result = await _handle_compound_price_pending(
        user_text=user_text,
        state=state,
        services=services,
        memory=memory,
        runtime_options=runtime_options,
    )
    if compound_price_result is not None:
        return compound_price_result

    # Вежливое переключение на вторичный интент по короткому "да/нет".
    queue = get_secondary_queue(state)
    if (
        state.last_entities.get("_secondary_offer_pending")
        and queue
        and state.dialog.phase not in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
    ):
        reply_kind = contextual_reply_kind(user_text)
        if reply_kind == "other" and _is_secondary_soft_yes(user_text):
            reply_kind = "yes"
        next_label = queue[0]
        secondary_entities = await _resolve_secondary_queue_doctor_reply(user_text, next_label, services)
        if reply_kind == "other" and secondary_entities:
            reply_kind = "yes"
        if reply_kind == "yes":
            next_label = queue.pop(0)
            set_secondary_queue(state, queue)
            state.last_entities["_secondary_offer_pending"] = False
            if secondary_entities:
                memory.merge_entities(state, secondary_entities, label=next_label)
            decision = RouteDecision(
                label=next_label,  # type: ignore[arg-type]
                confidence=0.9,
                entities={"secondary_intent_from_queue": True, **secondary_entities},
                flags={"secondary_intent_activated"},
                needs_handoff=False,
            )
            plan = build_plan(decision, state, user_text, memory=memory, runtime_options=runtime_options)
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
    pending_before_nlu = memory.get_pending(state)
    appointment_prelock = await prelock_active_appointment_turn(
        user_text,
        state.last_entities,
        pending_before_nlu if isinstance(pending_before_nlu, dict) else None,
        services,
    )
    if appointment_prelock is not None:
        clear_service_name = bool(appointment_prelock.pop("__clear_service_name", False))
        clear_patient_name = bool(appointment_prelock.pop("__clear_patient_name", False))
        if clear_service_name:
            state.last_entities.pop("service_name", None)
            state.last_entities.pop("test_name", None)
        if clear_patient_name:
            state.last_entities.pop("patient_name", None)
            state.last_entities.pop("appointment_confirm_pending", None)
            state.last_entities.pop("appointment_confirmed", None)
            state.dialog.phase = AppointmentPhase.COLLECTING
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.91,
            entities=appointment_prelock,
            flags={"appointment_slot_prelock"},
            needs_handoff=False,
            context_action="continue",
            source="flow_prelock",
        )
        nlu_debug["appointment_prelock"] = {
            "hit": True,
            "entities": dict(appointment_prelock),
            "pending_label": str((pending_before_nlu or {}).get("label") or ""),
        }
    else:
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
    decision = await _inject_catalog_candidates(
        decision,
        user_text=user_text,
        state=state,
        services=services,
    )
    last_label_before = str(state.last_entities.get("_last_label") or "")
    service_context = str(state.last_entities.get("service_name") or state.last_entities.get("test_name") or "").strip()
    if (
        last_label_before == "PRICE"
        and service_context
        and detect_prepare_intent(user_text)
        and decision.label in {"OTHER", "TEST_ASSIST", "ADDRESS", "PRICE"}
    ):
        entities = dict(decision.entities)
        if not entities.get("service_name"):
            entities["service_name"] = service_context
        decision = _copy_decision(
            decision,
            label="PREPARE",
            confidence=max(decision.confidence, 0.72),
            entities=entities,
            flags=set(decision.flags) | {"flow_price_followup_prepare"},
            needs_handoff=False,
            context_action="continue",
        )
    if (
        last_label_before == "PRICE"
        and service_context
        and decision.label in {"OTHER", "TEST_ASSIST", "DOCTOR_INFO", "APPOINTMENT"}
    ):
        merged_ctx = dict(state.last_entities)
        merged_ctx.update(decision.entities or {})
        followup_address_hint = bool(re.search(r"\b(где|адрес|филиал|сдать|сдавать)\b", user_text or "", re.I))
        if followup_address_hint and detect_nonbookable_walkin_intent(user_text, merged_ctx):
            entities = dict(decision.entities)
            if not entities.get("service_name"):
                entities["service_name"] = service_context
            decision = _copy_decision(
                decision,
                label="ADDRESS",
                confidence=max(decision.confidence, 0.76),
                entities=entities,
                flags=set(decision.flags) | {"flow_price_followup_address"},
                needs_handoff=False,
                context_action="continue",
                source="guardrail_post",
            )
    # В активном APPOINTMENT flow короткий follow-up с датой/временем
    # считаем продолжением записи до применения context_action.
    if (
        state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
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

    pending_before_overrides = memory.get_pending(state)
    if (
        isinstance(pending_before_overrides, dict)
        and pending_before_overrides.get("label") == "PRICE"
        and decision.label == "ADDRESS"
        and is_city_only_reply(user_text)
    ):
        decision = _copy_decision(
            decision,
            label="PRICE",
            confidence=max(decision.confidence, 0.72),
            flags=set(decision.flags) | {"flow_price_city_reply_override"},
            needs_handoff=False,
            context_action="continue",
        )

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
        and decision.label in {"OTHER", "TEST_ASSIST", "ADDRESS", "PRICE", "DOCTOR_INFO", "DOCTOR_SCHEDULE", "APPOINTMENT"}
        and is_short_prepare_followup(user_text)
        and not detect_prepare_intent(user_text)
        and not detect_nonbookable_walkin_intent(user_text, state.last_entities)
        and not any(decision.entities.get(k) for k in ("doctor_name", "doctor_id", "last_name", "doctor_last_name"))
    ):
        decision = _copy_decision(
            decision,
            label="PREPARE",
            confidence=max(decision.confidence, 0.62),
            flags=set(decision.flags) | {"flow_prepare_followup_override"},
            needs_handoff=False,
            context_action="continue",
        )

    last_label = str(state.last_entities.get("_last_label") or "")
    if last_label == "DOCTOR_INFO" and decision.label in {"OTHER", "DOCTOR_INFO", "TEST_RESULT"}:
        doctor_followup_entities: dict[str, Any] = {}
        if (
            not detect_schedule_intent(user_text)
            and not has_datetime_signal(user_text)
            and not detect_prepare_intent(user_text)
            and not detect_price_intent(user_text)
            and not detect_address_intent(user_text)
            and not detect_doc_request_intent(user_text)
            and not detect_test_result_intent(user_text)
            and not detect_test_assist_intent(user_text)
        ):
            doctor_followup_entities = await _resolve_secondary_queue_doctor_reply(
                user_text,
                "DOCTOR_SCHEDULE",
                services,
            )
        if doctor_followup_entities or (
            _is_short_verified_doctor_followup(decision, user_text)
            and not detect_schedule_intent(user_text)
        ):
            entities = dict(decision.entities)
            entities.update(doctor_followup_entities)
            state.last_entities["_secondary_offer_pending"] = False
            set_secondary_queue(state, [])
            decision = _copy_decision(
                decision,
                label="DOCTOR_SCHEDULE",
                confidence=max(decision.confidence, 0.74),
                entities=entities,
                flags=set(decision.flags) | {"flow_doctor_info_to_schedule"},
                needs_handoff=False,
                context_action="continue",
            )

    if (
        last_label == "DOCTOR_SCHEDULE"
        and (state.last_entities.get("doctor_name") or state.last_entities.get("doctor_id"))
        and _is_appointment_datetime_followup(user_text)
        and decision.label in {"OTHER", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "TEST_RESULT", "TEST_ASSIST", "ADDRESS", "PRICE"}
    ):
        if state.dialog.phase not in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM) and not looks_like_patient_fio(user_text):
            state.last_entities.pop("patient_name", None)
            state.last_entities.pop("appointment_confirm_pending", None)
            state.last_entities.pop("appointment_confirmed", None)
            state.dialog.phase = AppointmentPhase.COLLECTING
        entities = dict(decision.entities)
        for key in ("doctor_id", "doctor_name", "branch_id", "branch_name"):
            if not entities.get(key) and state.last_entities.get(key):
                entities[key] = state.last_entities.get(key)
        decision = _copy_decision(
            decision,
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.72),
            entities=entities,
            flags=set(decision.flags) | {"flow_schedule_to_appointment"},
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
                state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
                and should_keep_appointment_flow_override(user_text)
            )
        )
        if not keep_appointment_flow:
            reset_appointment_runtime_state(state)

    unsupported_kind = _unsupported_catalog_kind(set(decision.flags))
    if unsupported_kind:
        update_summary(
            state,
            reason="topic_switch" if decision.context_action in {"new_topic", "overwrite_doctor"} else "",
        )
        return (
            decision,
            Plan(label=decision.label),
            Evidence(items={"unsupported_catalog": {"kind": unsupported_kind}}),
        )

    need_service_catalog, need_doctors_catalog = _catalog_health_requirements(
        decision=decision,
        state=state,
        memory=memory,
    )
    if need_service_catalog or need_doctors_catalog:
        health = await services.get_catalog_health()
        service_catalog_ok = bool(health.get("service_catalog_ok"))
        doctors_catalog_ok = bool(health.get("doctors_catalog_ok"))
        is_blocked = (
            (need_service_catalog and not service_catalog_ok)
            or (need_doctors_catalog and not doctors_catalog_ok)
        )
        if is_blocked:
            degraded_flags = set(decision.flags) | {"catalog_health_degraded"}
            if need_service_catalog and not service_catalog_ok:
                degraded_flags.add("catalog_service_unavailable")
            if need_doctors_catalog and not doctors_catalog_ok:
                degraded_flags.add("catalog_doctors_unavailable")
            return (
                _copy_decision(
                    decision,
                    label="OTHER",
                    confidence=max(decision.confidence, 0.92),
                    entities={},
                    flags=degraded_flags,
                    needs_handoff=False,
                    source="catalog_health",
                    context_action="continue",
                ),
                Plan(label="OTHER"),
                Evidence(
                    items={
                        "catalog_health_response": {
                            "text": _catalog_health_degraded_text(
                                need_service_catalog=need_service_catalog,
                                need_doctors_catalog=need_doctors_catalog,
                            ),
                            "handoff": False,
                            "status": str(health.get("status") or "degraded"),
                            "reason": str(health.get("reason") or "").strip(),
                            "need_service_catalog": need_service_catalog,
                            "need_doctors_catalog": need_doctors_catalog,
                            "service_catalog_ok": service_catalog_ok,
                            "doctors_catalog_ok": doctors_catalog_ok,
                        }
                    }
                ),
            )

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

    catalog_confirm = _maybe_start_catalog_confirm(
        decision=decision,
        state=state,
        memory=memory,
    )
    if catalog_confirm is not None:
        return catalog_confirm

    # Служебные каталожные ключи не должны попадать в долгоживущий state.
    if any(k in decision.entities for k in ("_catalog_doctor_candidate", "_catalog_doctor_query", "_catalog_service_candidate", "_catalog_service_query")):
        clean_entities = dict(decision.entities)
        for key in ("_catalog_doctor_candidate", "_catalog_doctor_query", "_catalog_service_candidate", "_catalog_service_query"):
            clean_entities.pop(key, None)
        decision = _copy_decision(decision, entities=clean_entities)

    # merge entities from LLM+rules
    memory.merge_entities(state, decision.entities, label=decision.label)
    # Явный город в текущей реплике должен уметь исправлять/обновлять контекст
    # даже если city уже был заполнен ранее неверно.
    city_hint_now = match_city(user_text)
    if city_hint_now and decision.label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        memory.merge_entities(state, {"city": city_hint_now}, label=decision.label)
    elif decision.label in {"APPOINTMENT", "ADDRESS", "TEST_ASSIST", "DOCTOR_INFO", "DOCTOR_SCHEDULE", "PRICE"}:
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
        elif decision.label == "APPOINTMENT" and state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM):
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

    plan = build_plan(decision, state, user_text, memory=memory, runtime_options=runtime_options)
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


def _build_doctor_schedule_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    return response_build_doctor_schedule_response(flow_label, evidence, state, memory)


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
        logger.warning("background_refresh_start_failed", exc_info=True)

    # Явный запрос оператора должен иметь абсолютный приоритет.
    if explicit_operator_requested(user_text):
        _reset_state_after_handoff(state, memory)
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
        _reset_state_after_handoff(state, memory)
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
        if precheck.handoff:
            _reset_state_after_handoff(state, memory)
        yield precheck
        return

    try:
        from .orchestrator import run_pipeline

        ctx = await run_pipeline(
            user_text,
            state,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
        response = ctx.response
        decision = ctx.decision
        plan = ctx.plan
        evidence = ctx.evidence
        if response is None:
            raise RuntimeError("orchestrator returned no response")
    except Exception as e:
        logger.exception("patient_routing_stream failed for session_id=%s", state.session_id)
        fallback_text = handoff_message("service_error")
        state_update: dict[str, Any] = {}
        if debug:
            state_update = {"debug": {"route_error": str(e)}}
        _reset_state_after_handoff(state, memory)
        yield ResponseEnvelope(
            text=fallback_text,
            attachments=[],
            handoff=True,
            state_update=state_update,
        )
        return

    if debug and decision is not None and plan is not None and evidence is not None:
        pending = memory.get_pending(state)
        yield ResponseEnvelope(
            text="",
            attachments=[],
            handoff=False,
            state_update={"debug": _debug_meta(decision, plan, evidence, state, pending)},
        )

    memory.append_turn(state, role="user", text=user_text)
    memory.append_turn(state, role="assistant", text=response.text)
    update_summary(state, reason="normal")

    if response.handoff:
        _reset_state_after_handoff(state, memory)

    yield response
    return
