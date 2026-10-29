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

import asyncio
import contextlib
import copy
import logging
import re
from typing import AsyncGenerator, Any

from . import evidence_keys as ek
from . import address_flow, booking_slots, button_flow, expects, insurance_topic, llm_runtime, pending_answer, role_pick, service_answer, service_overview, shown_context, test_assist_answer, topic_confirm, topic_policy, turn_names, turn_parse, turn_telemetry, turn_understanding, walkin
from .state_mutations import reactivate_appointment_collecting
from .mess_types import AppointmentPhase, Evidence, Plan, ResponseEnvelope, RouteDecision, SessionState
from .classifier import analyze
from .context_summary import update_summary
from .dialog_graph import GraphEngine
from .entity_grounder import (
    BOOKING_SERVICE_CLARIFY_KEY,
    BOOKING_SERVICE_OPTIONS_KEY,
    ground_decision_entities,
    judge_fuzzy_service_candidate,
    sanitize_doctor_entities,
    service_dropped_by_llm,
    verify_doctor_entities_in_decision,
)
from .doctor_name_port import cached_doctor_fios
from .flow_policy import (
    apply_context_action,
    apply_pending_override,
    clear_on_handoff,
    clear_on_topic_switch,
    fill_date_from_schedule_windows,
    get_secondary_queue,
    is_appointment_waiting_patient_name,
    is_city_only_reply,
    is_short_prepare_followup,
    looks_like_patient_fio,
    normalize_secondary_labels,
    prelock_active_appointment_turn,
    quick_fill_entities_from_text,
    reset_appointment_runtime_state,
    set_secondary_queue,
)
from .appointment_flow_guard import (
    run_appointment_precheck,
    should_keep_appointment_flow_override,
)
from .nlu_pipeline import analyze_with_candidates
from .llm_mode_policy import RuntimeOptions, fail_offer
from .planner import build_plan as planner_build_plan, price_of as planner_price_of
from .executor import execute_plan as executor_execute_plan
from .response_builder import (
    build_first_structured_response as response_build_first_structured_response,
    mark_operator_offer_pending,
)
from .russian_nlu import normalize_ru
from .policies import (
    branch_options_to_indexable,
    build_branch_index,
    match_branch_hint,
    APPOINTMENT_WISH_MAX_LEN,
    looks_like_time_wish,
    missing_slots,
    handoff_message,
    evidence_requires_handoff,
    apply_verified_doctor_override,
    detect_nonbookable_walkin_intent,
    detect_appointment_action,
    detect_test_assist_goal_phrasing,
    detect_test_assist_intent,
    detect_test_result_intent,
    detect_prepare_intent,
    detect_price_intent,
    detect_address_intent,
    detect_doc_request_intent,
    detect_schedule_intent,
    detect_doctor_info_intent,
    detect_appointment_change_without_object,
    detect_existing_appointment_candidate,
    detect_existing_appointment_request,
    detect_urgent,
    extract_specialty,
    has_datetime_signal,
    is_specialty_word,
    is_test_assist_category_term,
    normalize_appointment_action,
    nonbookable_service_hint,
    service_name_conflicts_with_doctor,
    strip_booking_necessity,
)
from .service_phrase import extract_service_phrase
from .recovery_policy import contextual_reply_kind, explicit_operator_requested
from .services import Services, match_compound_price_service_option, resolve_price_service_name_from_catalog
from .services._appointment_change_validator import is_own_appointment_change
from .services._existing_appointment_validator import is_own_existing_appointment
from .services._doctors_helpers import _extract_specialties_from_text, _is_consultation_service_query, _service_name_allows_specialty
from .services._prices_helpers import _PRICE_PROCEDURE_LIKE_RE, consult_price_phrase, price_reply_names_own_service
from .services._readiness_select import readiness_by_choice, readiness_from_catalog
from .services.lab_tests import RESULT_TIMING_OPERATOR_TEXT, RESULTS_PORTAL_URL
from .services.prepare import anesthesia_reply, prepare_variant_payload
from .services._patient_name_validator import is_patient_name_reply
from .services._samara_perimeter import (
    branch_name_for_place,
    is_branch_address_word,
    is_samara_perimeter_text,
    match_perimeter_place,
)
from .services._service_slot_validator import is_service_name_reply
from .memory import MemoryStore
from .city import match_city
from .topic_registry import (
    build_topic_flag,
    extract_topic_id_from_flags,
    match_topic,
)
from .topic_registry import get_topic as topic_registry_get_topic
from .runtime_fingerprint import runtime_fingerprint
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

    refusals = ("topic_other_offer", "role_absent", "role_qualifier", "unsupported_service", "unsupported_specialist")
    for kind in (*refusals, "unsupported_document_service", "llm_fail_offer", "role_fail_offer"):  # сбой LLM в узле — последним
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
    # Ход (д), принцип 3 («уточните у оператора» — только с настоящим переводом): каталог пуст — перевод, а не
    # «попробуйте через 5 минут или напишите «оператор»» (решение владельца 18.10: сбой МИС → оператор).
    # Каталог услуг нужен и цене, и записи — общий текст (не «цены» на вопрос о записи).
    if need_doctors_catalog and not need_service_catalog:
        return handoff_message("service_error_doctors_list")
    return handoff_message("service_error")


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


def _relabel(decision: RouteDecision, policy: str, **overrides: Any) -> RouteDecision:
    """Переклейка метки после NLU — только строкой таблицы политик `topic_policy.TABLE` (ход (б), этап 3, решение
    владельца 20.10 (б) 7): копия решения, как `_copy_decision`, и запись строки в телеметрию хода (`tu_relabel_src`)."""

    relabeled = _copy_decision(decision, **overrides)
    topic_policy.note(policy, decision.label, relabeled.label)
    return relabeled


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


def _topic_would_override(decision: RouteDecision, topic_match: Any) -> bool:
    """Тема реестра сменила бы метку (из OTHER, по allowlist и счёту) — те же условия, что в
    `_apply_topic_registry_override`; только здесь слова реестра меняют маршрут."""

    topic_label = str(getattr(topic_match, "label", "") or "").strip().upper()
    return (
        bool(topic_label and getattr(topic_match, "topic_id", ""))
        and decision.label == "OTHER"
        and topic_label in _TOPIC_OVERRIDE_ALLOW_FROM_OTHER
        and int(getattr(topic_match, "score", 0) or 0) >= _TOPIC_OVERRIDE_MIN_SCORE
    )


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

    return _relabel(
        decision, "topic_registry_override_from_other",
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


# O4 анти-залип: минимальная длина «содержательного» ответа (короткие
# «да/нет/ок» не считаем поводом для детекта повтора).
_REPEAT_GUARD_MIN_LEN = 24
# Пункт 5 принципов (docs/ARCHITECTURE_PRINCIPLES.md, 08.10): шаг дважды без продвижения →
# оператор. Для ответа-ВОПРОСА (бот чего-то ждёт: «Укажите…», «Скажите…», «…?») оффер —
# на первом повторе; раньше был на втором, и петли из двух одинаковых вопросов на трафике
# проходили незамеченными. Ответ с данными (ссылка на результат, цена, список) — на втором,
# как раньше: второй анализ по той же ссылке — законный повтор, а не залип (встречный свип
# 08.10 по 1 613 реальным диалогам).
_REPEAT_GUARD_AFTER_REPEATS_QUESTION = 1
_REPEAT_GUARD_AFTER_REPEATS_ANSWER = 2
_REPEAT_ASKS_RE = re.compile(
    r"\?\s*$|^(?:уточните|укажите|скажите|напишите|выберите|чтобы|для какой цели|из какого города|какой|какая|какие|к какому)",
    re.I,
)
_REPEAT_GUARD_OFFER = "Похоже, мне не удаётся помочь с этим в чате. Перевести на оператора?"


_REPEAT_URL_RE = re.compile(r"https?://\S+")


def _repeat_norm(text: str) -> str:
    """Нормализует текст ответа для сравнения на повтор.

    Вырезает динамические URL: анти-повтор ловит повтор СМЫСЛОВОЙ части ответа,
    а не сбрасывается от меняющейся ссылки. Прод-триаж 15.07 (#659/#673):
    пациент правит данные результата (Иванов↔Иванова, «ву 276»↔«266») —
    текст «Результат по вашим данным доступен по ссылке» один, но result-ссылка
    (getanaliz.php?...nom2=…) зависит от данных; сравнение полного текста с URL
    сбрасывало счётчик → эскалация на оператора не срабатывала.
    """
    stripped = _REPEAT_URL_RE.sub("", str(text or ""))
    return " ".join(stripped.lower().split())


def _capture_appointment_wish(decision: RouteDecision, user_text: str, state: SessionState, memory: MemoryStore) -> None:
    """Вариант A записи (08.10): пожелание по времени сохраняется ДОСЛОВНО — для сводки оператору.

    Источники: ответ на вопрос «Когда вам удобно?» (висящий слот `appointment_wish`) — любой
    текст; или реплика, в которой пациент сам назвал время («на завтра утром»). Дату и время
    не разбираем в графы: разбор ошибался («03.06» → 03:06), а оператор всё равно сверяет с
    расписанием.

    :param decision: решение хода
    :param user_text: реплика пациента
    :param state: состояние сессии
    :param memory: хранилище pending
    """

    if decision.label != "APPOINTMENT":
        return
    if str(state.last_entities.get("appointment_action") or "").strip().lower() == "cancel":
        return
    raw = " ".join(str(user_text or "").split())[:APPOINTMENT_WISH_MAX_LEN]
    if not raw:
        return
    pending = memory.get_pending(state)
    missing = pending.get("missing") if isinstance(pending, dict) else None
    if isinstance(pending, dict) and pending.get("label") == "APPOINTMENT" and isinstance(missing, list) and "appointment_wish" in missing:
        state.last_entities["appointment_wish"] = raw
        memory.clear_pending(state)
        return
    if looks_like_time_wish(raw):
        state.last_entities["appointment_wish"] = raw


def _maybe_offer_operator_on_repeat(
    response: ResponseEnvelope, state: SessionState, memory: MemoryStore, user_text: str = "", *, pressed: bool = False
) -> ResponseEnvelope:
    """Анти-залип: бот второй раз подряд задаёт тот же ВОПРОС (третий раз — тот же ответ с
    данными) на ДРУГУЮ реплику пациента → к ответу добавляется «Перевести на оператора?».

    Пациент повторил тот же вопрос — тот же ответ законен и повтором не считается.

    Нажатие листа меню (`pressed`) — выбор темы, не реплика пациента: повтором не считается, счёт начинается заново с ответа
    на нажатие (BUG-2026-10-28-BUTTON-PRESS-AS-REPEAT: «Сколько стоит анализ» → «Цена анализа» — один вопрос у двух листов —
    давал оффер, и название анализа следующей репликой становилось согласием на перевод). Реплика после нажатия, на которую бот
    снова задал тот же вопрос, — повтор, как без кнопки.

    Переиспользует существующий operator-offer-pending: следующий ход
    обрабатывает ``_handle_operator_offer_pending`` (да→handoff, нет→продолжаем,
    иное→снимает offer и обычный роутинг, без ловушки). Срабатывает только на
    реальном повторе → нормальные диалоги не затрагивает.

    В активном APPOINTMENT-флоу гард намеренно не срабатывает: у записи свои
    счётчики попыток (``_appointment_doctor_lookup_attempts``,
    ``_appointment_datetime_attempts``) и свой порог эскалации, которые
    обработают залип точнее (на конкретном слоте, а не на тексте ответа).

    :param response: финальный ответ текущего хода
    :param state: состояние сессии (счётчик в last_entities)
    :param memory: хранилище pending-слотов
    :param user_text: реплика пациента этого хода
    :return: тот же response (возможно, с offer-вопросом)
    """
    text = (response.text or "").strip()
    norm = _repeat_norm(text)
    if pressed:
        state.last_entities.update({"_last_answer_norm": norm, "_answer_repeat_count": 0, "_last_user_norm": ""})
        return response
    user_norm = " ".join(str(user_text or "").lower().split())
    same_question = bool(user_norm) and user_norm == state.last_entities.get("_last_user_norm")
    state.last_entities["_last_user_norm"] = user_norm
    # Не вмешиваемся в handoff, пустые/короткие ответы, ответы про оператора,
    # уже-висящий offer (напр. кейс «нет слотов» сам ставит operator-offer) и
    # активный appointment-flow (у него свои attempt-счётчики и эскалация).
    if (
        response.handoff
        or len(norm) < _REPEAT_GUARD_MIN_LEN
        or "оператор" in norm
        or state.last_entities.get("_operator_offer_pending")
        or state.last_entities.get("appointment_flow_active")
    ):
        state.last_entities["_last_answer_norm"] = norm
        state.last_entities["_answer_repeat_count"] = 0
        return response

    prev = str(state.last_entities.get("_last_answer_norm") or "")
    repeat_count = int(state.last_entities.get("_answer_repeat_count") or 0)
    if norm != prev:
        repeat_count = 0
    elif not same_question:
        repeat_count += 1
    state.last_entities["_last_answer_norm"] = norm
    state.last_entities["_answer_repeat_count"] = repeat_count

    threshold = (
        _REPEAT_GUARD_AFTER_REPEATS_QUESTION if _REPEAT_ASKS_RE.search(text) else _REPEAT_GUARD_AFTER_REPEATS_ANSWER
    )
    if repeat_count >= threshold:
        state.last_entities["_answer_repeat_count"] = 0
        reset_appointment_runtime_state(state)
        mark_operator_offer_pending(state, memory)
        response.text = f"{text}\n\n{_REPEAT_GUARD_OFFER}"
    return response


def _maybe_offer_operator_on_reask_limit(
    response: ResponseEnvelope, state: SessionState, memory: MemoryStore
) -> ResponseEnvelope:
    """Предел переспросов (D2.4): третьего раза одного вопроса нет — нужный ответ и оффер оператора.

    Под `[MESSENGER_ROUTER] reask_limit = on` (по умолчанию off — ответ тот же). Обработчик висящего вопроса не
    разобрал ответ и спрашивает то же снова (`expects.ask(..., reask=True)`); без счётчика это петля (аудит 14.10,
    пп. 2–3): «Подтвердите запись» → «нет, другое время», «Отменить текущий процесс записи?» → «да, отмените».
    Инвариант `3bacf58`, пункт 5 принципов: шаг дважды без продвижения → назвать недостающее или оператор. Второй
    раз вопрос звучит, как раньше (сразу оператор на первом переспросе — лавина переводов); третьего нет.

    Прежний вопрос снимается вместе с состоянием записи — как у анти-залипа (`_maybe_offer_operator_on_repeat`):
    иначе предпроверка записи раньше обработчика оффера разобрала бы «да» как ответ на снятый вопрос (подтверждение
    заявки), и оператор получил бы заявку, которую пациент не подтверждал (BUG-2026-10-07-CONFIRM-REFUSAL-AS-YES).
    Анти-залип ответ с оффером не трогает (в тексте «оператор»).

    :param response: ответ хода — переспрос обработчика
    :param state: состояние сессии
    :param memory: хранилище pending-слотов
    :return: тот же ответ или оффер оператора вместо третьего вопроса
    """

    limited = expects.over_limit()
    if limited is None or response.handoff:
        return response
    from .response_builder import PREPARE_VARIANT_PENDING_KEY

    reset_appointment_runtime_state(state)
    _clear_catalog_confirm_pending(state, memory)
    state.last_entities.pop(_CATALOG_CONFIRM_REJECTS_KEY, None)
    state.last_entities.pop(PREPARE_VARIANT_PENDING_KEY, None)
    mark_operator_offer_pending(state, memory)
    turn_telemetry.override(answer_source="reask_limit")
    return ResponseEnvelope(text=expects.reask_limit_text(limited), handoff=False, state_update=response.state_update)


# Дедлайн хода истёк (`llm_turn_deadline_s`): LLM не успела там, где была нужна, — честно, без ответа правил с фактом.
DEADLINE_OFFER_TEXT = "Не успел подготовить точный ответ — чтобы не ошибиться, его даст оператор. Перевести на оператора?"


def _offer_operator_on_deadline(response: ResponseEnvelope, state: SessionState, memory: MemoryStore) -> ResponseEnvelope:
    """Дедлайн хода не оставил времени вызову LLM — ответ хода честный оффер оператора, а не ответ правил
    (BUG-2026-10-27-LLM-QUEUE-GIVES-UP, решение владельца 27.10: «истёк дедлайн — уточню у оператора, не правило с ценой»).

    Под выключателем `llm_turn_deadline_s` (0 — дедлайна нет, ответ тот же). Перевод на оператора остаётся переводом.
    Вопросы, которые задал бы заменённый ответ, снимаются — как у предела переспросов: «да» следующего хода — ответ на оффер.
    """

    if response.handoff or not llm_runtime.turn_deadline_hit():
        return response
    from .response_builder import PREPARE_VARIANT_PENDING_KEY

    reset_appointment_runtime_state(state)
    _clear_catalog_confirm_pending(state, memory)
    state.last_entities.pop(PREPARE_VARIANT_PENDING_KEY, None)
    mark_operator_offer_pending(state, memory)
    turn_telemetry.override(answer_source="llm_deadline_offer")
    return ResponseEnvelope(text=DEADLINE_OFFER_TEXT, handoff=False, state_update=response.state_update)


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

    action_q = expects.Question.APPOINTMENT_ACTION
    if pending_answer.passes(action_q) and normalize_appointment_action(detect_appointment_action(user_text), user_text) != "cancel":
        # Другая тема: вопрос снят, ход идёт обычным путём (распознанная отмена — ниже, всегда к оператору).
        memory.clear_pending(state)
        return None

    if pending_answer.operator_requested(action_q, user_text):
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
    # Распознанная отмена правил — вето в сторону оператора при любом решении модели (CANCEL-ALWAYS-OPERATOR);
    # иначе — пункт, выбранный одной функцией («лучше перенести на пятницу» → «перенести»).
    chosen = pending_answer.picked(action_q)
    if normalized_action != "cancel" and chosen in expects.APPOINTMENT_ACTION_OPTIONS:
        normalized_action = "cancel" if chosen == expects.APPOINTMENT_ACTION_OPTIONS[0] else "reschedule"
    if normalized_action == "cancel":
        # BUG-2026-10-14-CANCEL-EXISTING-LOST: выбор отмены существующей записи — сразу оператор.
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.95,
            entities={},
            flags={"appointment_action_selected"},
            needs_handoff=False,
            source="appointment_action_pending",
        )
        return _existing_appointment_cancel_handoff(decision, state, memory)
    if normalized_action == "reschedule":
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

    if pending_answer.kind(action_q, lambda: contextual_reply_kind(user_text)) == "no":
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

    expects.ask(expects.Question.APPOINTMENT_ACTION, kind=expects.PICK, options=expects.APPOINTMENT_ACTION_OPTIONS, reask=True)
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


async def _handle_prepare_variant_pending(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Ответ на «с наркозом или без?» — нужная памятка базы знаний как есть (07.10).

    «с наркозом / во сне» и «без» — своя памятка; «не знаю» — обе; «да / нет» — переспрос
    один раз, потом обе; другая тема — вопрос снимается, ход идёт обычным путём.
    """

    from .response_builder import PREPARE_VARIANT_PENDING_KEY

    pending = state.last_entities.get(PREPARE_VARIANT_PENDING_KEY)
    if not isinstance(pending, dict):
        return None
    anesthesia_q = expects.Question.ANESTHESIA
    if pending_answer.operator_requested(anesthesia_q, user_text) or pending_answer.passes(anesthesia_q):
        state.last_entities.pop(PREPARE_VARIANT_PENDING_KEY, None)
        return None
    answered = pending_answer.current(anesthesia_q)
    if answered is not None and answered.decided:
        # Решение одной функции: только явный пункт — своя памятка; «ещё не решил» — обе; «да / нет» — переспрос
        # один раз, потом обе (решение 07.10); прочее — вопрос снимается, ход идёт обычным путём.
        chosen = pending_answer.picked(anesthesia_q)
        verdict = pending_answer.kind(anesthesia_q, lambda: "other")
        if chosen in expects.ANESTHESIA_OPTIONS:
            choice = ("with", "without")[expects.ANESTHESIA_OPTIONS.index(chosen)]
        elif verdict == "pause":
            choice = "both"
        else:
            choice = None
        reply_is_yes_no = choice is None and verdict in {"yes", "no"}
    else:
        choice = anesthesia_reply(user_text)
        reply_is_yes_no = choice is None and contextual_reply_kind(user_text) in {"yes", "no"}
    if reply_is_yes_no:
        if not pending.get("asked_again"):
            pending["asked_again"] = True
            expects.ask(expects.Question.ANESTHESIA, kind=expects.PICK, options=expects.ANESTHESIA_OPTIONS, reask=True)
            text = "Напишите, пожалуйста: «с наркозом» или «без наркоза»."
            return (
                RouteDecision(label="PREPARE", confidence=0.9, flags={"prepare_variant_reask"}, source="prepare_variant"),
                Plan(label="PREPARE"),
                Evidence(items={ek.PREPARE: {"prepare": text}}),  # вопрос остаётся висеть с asked_again
            )
        choice = "both"
    state.last_entities.pop(PREPARE_VARIANT_PENDING_KEY, None)
    if choice is None:
        return None
    payload = await asyncio.to_thread(prepare_variant_payload, dict(pending.get("titles") or {}), choice)
    if payload is None:
        return None
    return (
        RouteDecision(label="PREPARE", confidence=0.95, flags={"prepare_variant_chosen"}, source="prepare_variant"),
        Plan(label="PREPARE"),
        Evidence(items={ek.PREPARE: payload}),
    )


async def _compound_prices_for_each(
    chosen: list[str],
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """«Обе» на «Вижу в запросе две услуги» (`pending_buttons`, выбор нескольких пунктов одной функцией): цена каждой
    услуги — её обычным путём (как на выбор одного пункта), ответы — подряд. Какая-то не дала ответа или ведёт к переводу —
    None: дальше, как было (первая услуга)."""

    from .shown_context import SHOWN_KEY

    texts: list[str] = []
    shown: list[str] = []
    last: RouteDecision | None = None
    for service_name in chosen:
        memory.merge_entities(state, {"service_name": service_name}, label="PRICE")
        decision = RouteDecision(
            label="PRICE",
            confidence=0.92,
            entities={"service_name": service_name, "compound_price_selected": True},
            flags={"compound_price_selected", "compound_price_all"},
            needs_handoff=False,
            context_action="continue",
            source="compound_price",
        )
        plan = build_plan(decision, state, service_name, memory=memory, runtime_options=runtime_options)
        evidence = await execute_plan(plan, state, services)
        env = response_build_first_structured_response(
            flow_label="PRICE", evidence=evidence, state=state, services=services, memory=memory, decision=decision, user_text=service_name
        )
        if env is None or env.handoff or not str(env.text or "").strip() or memory.get_pending(state) is not None:
            return None  # нет цены, перевод или встречный вопрос (город, уточнение) — по одной, как было
        texts.append(str(env.text).strip())
        shown.extend(str(name) for name in state.last_entities.get(SHOWN_KEY) or () if str(name or "").strip())
        last = decision
    if last is None:
        return None
    _clear_compound_price_pending(state)
    set_secondary_queue(state, [])
    state.last_entities["_secondary_offer_pending"] = False
    state.last_entities.pop("secondary_intents", None)
    # Контекст = показанное (решение владельца 14.10): показаны обе услуги — одной «услуги контекста» нет.
    state.last_entities[SHOWN_KEY] = list(dict.fromkeys(shown))
    state.last_entities.pop("service_name", None)
    state.last_entities.pop("test_name", None)
    return last, Plan(label="PRICE"), Evidence(items={ek.COMPOUND_PRICE_RESPONSE: {"text": "\n\n".join(texts)}})


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

    compound_q = expects.Question.COMPOUND_PRICE
    if pending_answer.operator_requested(compound_q, user_text) or pending_answer.passes(compound_q):
        _clear_compound_price_pending(state)
        return None

    options = list(pending.get("services") or [])
    default_service = str(pending.get("default_service") or options[0]).strip()

    def _rules() -> str:
        reply_kind = contextual_reply_kind(user_text)
        if reply_kind == "other" and (
            _is_secondary_soft_yes(user_text) or _COMPOUND_PRICE_SOFT_YES_RE.match(str(user_text or ""))
        ):
            reply_kind = "yes"
        return reply_kind

    reply_kind = pending_answer.kind(compound_q, _rules)
    answered = pending_answer.current(compound_q)
    every = [item for item in pending_answer.picked_all(compound_q) if item in options]
    if len(every) >= 2:
        both = await _compound_prices_for_each(every, state, services, memory, runtime_options)
        if both is not None:
            return both
    if reply_kind == "yes":
        selected_service = default_service
    elif answered is not None and answered.decided:
        chosen = pending_answer.picked(compound_q)  # только явный пункт показанного списка
        selected_service = chosen if chosen in options else None
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


# Второй отказ на «Похоже, вы имели в виду …?» под `pending_buttons`: перевод — вопросом, решает пациент (решение
# владельца 19.10: автоматически не переводить — участие операторов со временем сводим к минимуму).
CATALOG_REJECTED_OFFER_TEXT = "Не удалось точно сопоставить запрос с каталогом. Перевести на оператора?"


def _catalog_rejected_handoff(state: SessionState, memory: MemoryStore) -> tuple[RouteDecision, Plan, Evidence]:
    """Второй отказ на «Похоже, вы имели в виду …?» — оператор; под `pending_buttons` — оффер оператора вопросом."""

    state.last_entities.pop(_CATALOG_CONFIRM_REJECTS_KEY, None)
    if pending_answer.buttons_on():
        mark_operator_offer_pending(state, memory)
        return (
            RouteDecision(label="OTHER", confidence=0.95, entities={}, flags={"catalog_confirm_rejected_offer"}, needs_handoff=False, source="catalog_confirm"),
            Plan(label="OTHER"),
            Evidence(items={ek.OPERATOR_OFFER_RESPONSE: {"text": CATALOG_REJECTED_OFFER_TEXT, "handoff": False}}),
        )
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


def _catalog_second_no(state: SessionState, memory: MemoryStore) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Ответ «не могу назвать» («нет, не то») на «Уточните название услуги» после первого «нет» на «Похоже, вы имели в виду
    …?» — второй отказ, оффер оператора вопросом (политика `catalog_confirm`: 2-е «нет» → оператор; `pending_buttons`).

    Было (стенд, `pa_catalog_no_twice`): после первого «нет» вопрос уточнения поля ожидания не ставил, второе «нет» читала
    предпроверка записи по своим правилам — «Отменить текущий процесс записи?». Вопрос уточнения — тот, что поставил первый
    отказ: `_pending` с недостающим уточнения (`_catalog_refine_missing_slots`), счётчик отказов ≥ 1. Название услуги в
    ответе («нет, мне массаж спины» — исход «ответ») — обычный путь, как было.
    """

    found = pending_answer.slot_answer()
    if found is None or found.outcome is not pending_answer.Outcome.NO:
        return None
    if int(state.last_entities.get(_CATALOG_CONFIRM_REJECTS_KEY) or 0) < 1:
        return None
    pending = memory.get_pending(state) or {}
    label = str(pending.get("label") or "")
    refine = [_catalog_refine_missing_slots(kind, label) for kind in ("service", "doctor")]
    if not label or pending.get("missing") not in refine:
        return None
    memory.clear_pending(state)
    return _catalog_rejected_handoff(state, memory)


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
        return _catalog_second_no(state, memory)

    label = str(pending.get("label") or "OTHER").strip()
    kind = str(pending.get("kind") or "").strip().lower()
    entity_key = str(pending.get("entity_key") or "").strip()
    canonical = str(pending.get("canonical") or "").strip()

    if pending_answer.passes(expects.Question.CATALOG_CONFIRM):
        # Другая тема или «нет, мне ферритин»: вопрос снят, хвост идёт обычным путём; во «2-е нет» не считается.
        _clear_catalog_confirm_pending(state, memory)
        return None

    if pending_answer.operator_requested(expects.Question.CATALOG_CONFIRM, user_text):
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

    reply_kind = pending_answer.kind(expects.Question.CATALOG_CONFIRM, lambda: contextual_reply_kind(user_text))
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
            return _catalog_rejected_handoff(state, memory)
        missing_slots = _catalog_refine_missing_slots(kind, label)
        memory.set_pending(state, label=label, missing_slots=missing_slots)
        if pending_answer.buttons_on():
            # «Уточните название услуги» — вопрос со свободным ответом: второе «нет» на него («нет, не то») — второй отказ
            # (`_catalog_second_no`), а не «Отменить процесс записи?» (BUG-2026-10-19-PENDING-BUTTONS-CLASS5, п. 4).
            expects.ask(expects.Question.SLOT_VALUE, kind=expects.FREE)
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

    expects.ask(expects.Question.CATALOG_CONFIRM, reask=True)
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

    if decision.label != "APPOINTMENT" and pending_answer.keeps_flow():
        # Ответ на вопрос шага записи решила одна функция (класс 5): «1», «давайте во второй» — ответ, а не новая тема.
        return _relabel(
            decision, "flow_pending_answer",
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.9),
            flags=set(decision.flags) | {"flow_pending_answer"},
            needs_handoff=False,
            context_action="continue",
        )

    if (
        has_datetime_signal(user_text)
        and decision.label in {"OTHER", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "TEST_RESULT", "ADDRESS", "PRICE"}
        and not detect_prepare_intent(user_text)
        and not detect_test_assist_intent(user_text)
        and not detect_test_result_intent(user_text)
        and not detect_doc_request_intent(user_text)
    ):
        return _relabel(
            decision, "flow_datetime_appointment_override",
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
        return _relabel(
            decision, "flow_appointment_override",
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            flags=set(decision.flags) | {"flow_appointment_override"},
            needs_handoff=False,
            context_action="continue",
        )

    if decision.label in {"TEST_RESULT", "DOCTOR_SCHEDULE", "DOCTOR_INFO"} and should_keep_appointment_flow_override(user_text):
        return _relabel(
            decision, "flow_appointment_guard_override",
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.60),
            flags=set(decision.flags) | {"flow_appointment_guard_override"},
            needs_handoff=False,
            context_action="continue",
        )

    return decision


def _note_decision(decision: RouteDecision | None, evidence: Evidence | None, stage_timings: dict[str, float] | None) -> None:
    """Решение хода в телеметрию: метка, подпись NLU, первое правило, пустота данных."""

    if decision is None:
        return
    from .orchestrator import _evidence_is_empty

    rule = next((f for f in sorted(decision.flags or []) if str(f).startswith("rule_")), None)
    turn_telemetry.note(
        label=decision.label,
        tu_label_final=decision.label,
        decision_source=decision.source,
        rule=rule,
        evidence_empty=_evidence_is_empty(evidence) if evidence is not None else None,
        stage_ms={k: int(v) for k, v in (stage_timings or {}).items()},
    )


def _early_debug_state_update(
    debug: bool,
    *,
    label: str,
    handoff: bool,
    flags: set[str] | None = None,
    context_action: str = "continue",
    confidence: float = 0.95,
) -> dict[str, Any]:
    # Все ранние выходы (срочное, оператор, город, предпроверка записи) проходят здесь —
    # одна точка для телеметрии хода, и при debug=false тоже.
    reason = sorted(flags or set())[0] if flags else context_action
    turn_telemetry.note(answer_source=f"precheck:{reason}", label=label, decision_source="router_precheck")
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


def _plan_service_catalog_prefetch(
    decision: RouteDecision,
    state: SessionState,
    user_text: str,
) -> dict[str, str] | None:
    """Part IV Stage 15 (full OPTION B): decide whether a speculative service
    catalog match can be fired in parallel with doctor verification.

    Speculation is only safe when the pre-verify state guarantees the
    post-verify ``_inject_catalog_candidates`` gate fires with an identical
    ``match_catalog_service`` input. ``_verify_doctor_entity`` can only alter
    doctor-related fields (resolve/drop ``doctor_name``, drop ``service_name``
    on conflict, or promote to ``patient_name``); this helper restricts
    speculation to cases where none of those mutations can change the gate:

    - label must be service-eligible
    - no ``service_name`` already present (otherwise gate is False pre-verify
      and speculation is pointless)
    - no ``doctor_name`` anywhere (entities or ``state.last_entities``) —
      absence pre-verify plus ``context_action != overwrite_doctor`` means
      verify cannot introduce a doctor that would flip the APPOINTMENT gate
    - ``context_action`` not ``overwrite_doctor`` — that branch runs
      ``resolve_doctor_name(user_text)`` and may set ``doctor_name``
    - not a PRICE+specialty pre-condition (gate would be False)

    :return: ``{"query": ..., "current_service_name": ...}`` or ``None``
    """

    entities = decision.entities or {}
    last = state.last_entities or {}

    if decision.label not in {"APPOINTMENT", "PRICE", "ADDRESS", "TEST_ASSIST"}:
        return None
    if entities.get("service_name"):
        return None
    if entities.get("doctor_name") or last.get("doctor_name"):
        return None
    if decision.context_action == "overwrite_doctor":
        return None
    if decision.label == "PRICE" and (
        entities.get("specialty") or last.get("specialty")
    ):
        return None
    if shown_context.expands_shown(decision):
        return None  # ход раскрытия семейства: каталог по реплике не ищут (гейт ниже закрыт так же)

    query = str(entities.get("test_name") or user_text or "")
    if not query or is_specialty_word(query):
        return None
    current = str(last.get("service_name") or "")
    if is_specialty_word(current):
        current = ""  # «Терапевту» — врач, не услуга: каталог по буквам дал бы «ТЭС-терапию»
    return {"query": query, "current_service_name": current}


async def _promote_bare_doctor_name(
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    services: Services,
) -> RouteDecision:
    """Голая фамилия врача — расписание этого врача, а не свободный ответ (08.10).

    Живой пациент 08.10: «Бородина» → «у меня нет информации о враче с фамилией Бородина»,
    хотя врач в справочнике есть и со слотами. LLM ставит голой фамилии OTHER, а для OTHER
    сущность врача выбрасывается. Ответ — как на «когда принимает Бородина»: карточка и слоты.
    Только когда сообщение целиком — ФИО врача справочника (`doctor_surname_for_bare_name`).
    Не в отмене и переносе: фамилия там — врач существующей записи, ответ — оператор или
    перенос (разбор переписки 08.10: 4 из 13 голых фамилий шли после «отменить запись»).
    """

    if decision.label != "OTHER" or decision.source == "button":
        return decision
    if str(state.last_entities.get("appointment_action") or "") in {"cancel", "reschedule"}:
        return decision
    surname = await services.doctor_surname_for_bare_name(user_text)
    if not surname:
        return decision
    return _relabel(
        decision, "bare_doctor_name",
        label="DOCTOR_SCHEDULE",
        entities={"doctor_name": surname},
        flags=set(decision.flags) | {"bare_doctor_name"},
        needs_handoff=False,
        source="guardrail_post",
    )


async def _verify_decision_and_prefetch_catalog(
    *,
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    services: Services,
) -> tuple[RouteDecision, dict[str, Any] | None]:
    """Part IV Stage 15 (full OPTION B): parallelize doctor verification with
    a speculative service catalog match when guards permit.

    Falls back to a plain ``_verify_doctor_entity`` await when no speculation
    is safe (which covers the majority of turns that carry a raw doctor_name
    or ``overwrite_doctor`` context). Returns ``(verified_decision, prefetch)``
    where ``prefetch`` is ``{"query": str, "match": dict | None}`` or ``None``.
    """

    decision = await _promote_bare_doctor_name(decision, user_text, state, services)
    plan = _plan_service_catalog_prefetch(decision, state, user_text)
    if plan is None:
        verified = await _verify_doctor_entity(decision, services, user_text)
        return verified, None

    verify_coro = _verify_doctor_entity(decision, services, user_text)
    service_coro = services.match_catalog_service(
        plan["query"],
        current_service_name=plan["current_service_name"],
        context_text=user_text,
    )
    verified, service_match = await asyncio.gather(verify_coro, service_coro)
    return verified, {"query": plan["query"], "match": service_match}


async def _sanitize_doctor_in_entities(
    entities: dict[str, Any],
    services: Services,
    *,
    label: str,
) -> dict[str, Any]:
    # Backward-compatible wrapper: реализация живет в entity_grounder.
    return await sanitize_doctor_entities(entities=entities, services=services, label=label)


# A′ (BUG-2026-06-01-01) корень: quick-fill повторно извлекает сущности из
# СЫРОГО user_text и может воскресить слот, который грундер уже отверг в этом
# же ходу (напр. ФИО, ошибочно попавшее в service_name), в обход
# ground_decision_entities. Карта: слот -> грундер-флаги его отказа.
_GROUNDER_REJECTED_SLOT_FLAGS: dict[str, frozenset[str]] = {
    "service_name": frozenset(
        {
            "entity_dropped_unverified_service_name",
            "entity_dropped_doctor_like_service_name",
            "entity_dropped_stale_service_name_in_reschedule",
        }
    ),
}


def _suppress_grounder_rejected_slots(
    quick: dict[str, Any] | None,
    decision_flags: set[str] | frozenset[str] | None,
) -> dict[str, Any]:
    """Убирает из quick-fill результата слоты, которые грундер отверг в этом ходу.

    Грундер санирует только ``decision.entities``; quick-fill пишет напрямую в
    ``state.last_entities`` (его читает планировщик), поэтому без этого фильтра
    отвергнутая сущность воскресает мимо защиты (корень A′). Инвариант: если у
    текущего решения стоит флаг отказа грундера по слоту — quick-fill НЕ имеет
    права заново подставить ИЗ СЫРОГО ТЕКСТА то же невалидное значение (напр.
    ФИО, ошибочно принятое за услугу).

    Исключение (P3): значение, **подтверждённое каталогом**, остаётся —
    оно grounded-by-source, а не сырой re-extract. Это критично для ответа на
    pending-PRICE clarify: «Общий анализ крови» каталожно резолвится в реальную
    услугу, и срезать её нельзя, хотя у текущего (TEST_ASSIST) решения и стоит
    ``entity_dropped_unverified_service_name``.

    :param quick: сущности, извлечённые quick-fill из user_text
    :param decision_flags: флаги текущего (грундированного) решения
    :return: quick без отвергнутых грундером НЕвалидных слотов
    """
    if not quick:
        return quick or {}
    flags = set(decision_flags or set())
    if not flags:
        return quick
    out = dict(quick)
    for slot, reject_flags in _GROUNDER_REJECTED_SLOT_FLAGS.items():
        if slot not in out or not (flags & reject_flags):
            continue
        value = str(out.get(slot) or "").strip()
        # Каталожно-подтверждённую услугу не трогаем (она grounded, не re-extract). Под `slot_suppress_llm = on` — кроме
        # услуги, которую в этом ходу отвергла LLM: резолвер находит её по буквам («Гастроскопию» → «Гастрин»), а
        # смысл уже решён (принцип 2).
        if (
            slot == "service_name"
            and value
            and not (booking_slots.suppress_enabled() and service_dropped_by_llm(flags))
            and resolve_price_service_name_from_catalog(value)
        ):
            continue
        out.pop(slot, None)
    return out


async def _inject_catalog_candidates(
    decision: RouteDecision,
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    prefetched_service: dict[str, Any] | None = None,
    runtime_llm_mode: str = "",
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
        # «Чекап» — это КАТЕГОРИЯ (линейка пакетов), а не одна услуга.
        # match_catalog_service отдаёт exact на «Ежегодный Чекап» и схлопывает
        # весь список в одну каноническую строку, из-за чего test_assist
        # возвращает единственный пакет вместо всей линейки. Оставляем запрос
        # широким — service_name не пиннится, test_assist ищет по сырому тексту
        # и отдаёт всё семейство чекапов.
        if decision.label == "TEST_ASSIST" and is_test_assist_category_term(user_text):
            should_try_service = False
            flags.add("test_assist_category_kept_broad")
            updated = True
        # Понимание хода (ход (б), шаг 1, `on`): реплика — список услуг. Сопоставление всей
        # реплики с каталогом свело бы его к одной строке («холестерин» → «Липидограмма»),
        # список разберёт корзина цены (PRICE-LIST-COLLAPSES-TO-ONE).
        if should_try_service and decision.label in {"PRICE", "TEST_ASSIST"} and turn_understanding.acting_list():
            should_try_service = False
            flags.add("understanding_list_kept")
            updated = True
            turn_understanding.note_acted("catalog_inject_skipped")
        # Контекст = показанное (L-14): ход раскрытия семейства — ответ на подсказку бота
        # «напишите: "все"», а не название услуги; каталог по реплике свёл бы «все» к «УЗИ … все зоны (детское)»
        # (BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE). Список раскроет инструмент цены из контекста семейства. Так же номер
        # строки показанного списка («1»): строку ответит инструмент цены (BUG-2026-10-24-SHOW-ALL-HINT-ON-PROD).
        if should_try_service and (shown_context.expands_shown(decision) or shown_context.picks_shown_row(decision)):
            should_try_service = False
            flags.add(shown_context.SHOW_ALL_KEPT_FLAG)
            updated = True
        # Ход (б), этап 2, U1 (`turn_understanding_one = on`): реплика — одна услуга, сильная по данным МИС, с одним
        # названием прайса — это название; каталог по буквам реплики не ищем (дизайн 19.10 §3, исход one_strong). У подбора
        # анализов точное совпадение — ещё и довод П1 «это цена»; понимание находит название и в длинном вопросе, где буквы
        # не находили, и П1 без LLM отвечал бы ценой на вопрос о сроке (replay dev 21.10, g0168). Поэтому TEST_ASSIST — только
        # когда тему решила LLM (`topic_by_llm`): свойство вопроса для П1 там решает она же (`_test_assist_facet_by_llm`).
        strong_ok = decision.label == "PRICE" or (decision.label == "TEST_ASSIST" and "topic_by_llm" in flags)
        chosen = turn_understanding.strong_choice(user_text) if should_try_service and strong_ok else None
        if chosen:
            should_try_service = False
            entities["service_name"] = chosen
            entities.pop("test_name", None)
            flags.update({"catalog_service_exact", "understanding_service_chosen"})
            updated = True
            turn_understanding.note_acted("catalog_from_understanding")

    # Part IV Stage 15 (OPTION B, partial): doctor + service catalog lookups
    # are independent — both read post-verify entities and don't depend on
    # each other's result. Gather them into a single await to drop up to
    # ~50-100ms off the turn when both conditions fire. The APPLY half below
    # stays sequential (doctor first, then service) to preserve legacy flag /
    # entity merge order byte-for-byte.
    doctor_coro = (
        services.match_catalog_doctor(
            str(entities.get("doctor_name") or user_text or ""),
        )
        if should_try_doctor
        else None
    )

    # Part IV Stage 15 (full OPTION B): if ``_verify_decision_and_prefetch_catalog``
    # already fired a speculative service match in parallel with verify, and
    # the post-verify gate resolves to the same input, reuse that result
    # instead of re-fetching. ``_plan_service_catalog_prefetch`` guards the
    # speculation such that gate+input can only stay identical or the
    # prefetch is ``None``; the defensive query equality check below protects
    # against drift if those guards are ever loosened.
    service_coro = None
    service_match: dict[str, Any] | None = None
    if should_try_service:
        service_query = str(
            entities.get("service_name") or entities.get("test_name") or user_text or ""
        )
        # Название врача в любом падеже — не услуга: каталог по буквам превращал
        # «Терапевту» в «ТЭС-терапию», и запись уходила не к тому врачу (07.10).
        current_service = str(state.last_entities.get("service_name") or "")
        if is_specialty_word(current_service):
            current_service = ""
        if is_specialty_word(service_query):
            pass
        elif (
            prefetched_service is not None
            and prefetched_service.get("query") == service_query
        ):
            service_match = prefetched_service.get("match")
        else:
            service_coro = services.match_catalog_service(
                service_query,
                current_service_name=current_service,
                context_text=user_text,
            )

    doctor_match: dict[str, Any] | None = None
    if doctor_coro is not None and service_coro is not None:
        doctor_match, service_match = await asyncio.gather(doctor_coro, service_coro)
    elif doctor_coro is not None:
        doctor_match = await doctor_coro
    elif service_coro is not None:
        service_match = await service_coro

    if doctor_match is not None:
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

    if service_match is not None:
        status = str(service_match.get("status") or "")
        canonical = str(service_match.get("canonical") or "").strip()
        if status == "exact" and canonical:
            entities["service_name"] = canonical
            entities.pop("test_name", None)
            flags.add("catalog_service_exact")
            updated = True
        elif status == "fuzzy" and canonical and any(f.startswith("catalog_confirm_judged_") for f in flags):
            # Догадку этого хода уже судила LLM в заземлении: вторая догадка по буквам —
            # с другой строкой запроса — переспрос не получает (g0196: заземление сняло «клещ на энцефалит», роутер
            # предложил «Вирус клещевого энцефалита IgG» после исчерпанного лимита решений хода).
            pass
        elif status == "fuzzy" and canonical:
            await judge_fuzzy_service_candidate(
                label=decision.label,
                candidate=canonical,
                query=str(service_match.get("query") or service_query or "").strip(),
                user_text=user_text,
                services=services,
                runtime_llm_mode=runtime_llm_mode,
                out=entities,
                flags=flags,
                candidate_flag="catalog_service_fuzzy_candidate",
            )
            updated = True

    if not updated:
        return decision
    return _copy_decision(
        decision,
        entities=entities,
        flags=flags,
    )


async def _test_assist_facet_by_llm(decision: RouteDecision, user_text: str) -> str:
    """Ход `topic_by_llm` (тему решила LLM: TEST_ASSIST), каталог нашёл услугу — что пациент о ней спрашивает, решает LLM
    (ход (б), STEP-2026-10-21-B-P1-FACET). Не такой ход — пустая строка: П1 работает как было."""

    flags = set(decision.flags or ())
    if decision.label != "TEST_ASSIST" or "topic_by_llm" not in flags or "catalog_service_exact" not in flags:
        return ""
    facet = await booking_slots.test_assist_facet(user_text, full=_p1_facet_none_offer())
    turn_telemetry.note(tu_p1_facet=facet or "failed")
    return facet


def _p1_facet_none_offer() -> bool:
    """`p1_facet_none_offer = on` (BUG-2026-10-22-P1-FACET-NONE-AS-PRICE): полный набор свойств вопроса и «ни одно» → оффер."""

    return str(getattr(c, "MR_P1_FACET_NONE_OFFER", "off") or "off").strip().lower() == "on"


def _relabel_test_assist_catalog_hit_as_price(
    decision: RouteDecision,
    *,
    user_text: str,
    facet: str = "",
) -> RouteDecision:
    """П1: реплика приземлилась на РЕАЛЬНУЮ услугу каталога → это PRICE.

    Прод 31.07/09.08 (кластер №2 триажа: 17 диалогов, 82% провал): пациент
    называет анализ («С-реактивный белок», «спермограмма», «липидограмма») — и
    получает воронку подбора «Для какой цели хотите подобрать анализы?».
    Корень — МАРШРУТИЗАЦИЯ, не поиск: каталог такие названия резолвит, но голое
    название без ценового глагола (особенно вторым ходом, когда пациент
    продолжает список) размечается как TEST_ASSIST, а `REQUIRED_SLOTS`
    TEST_ASSIST требует `test_goal|test_name` — `service_name` его не закрывает,
    и строится generic clarify.

    Инвариант (класс): воронка подбора существует ровно для случая «услуга НЕ
    найдена». Если exact-матч по каталогу состоялся — цель реплики известна,
    это PRICE. Мутирует ТОЛЬКО метку: сущности каталожная инъекция уже проставила.

    Ставится ПОСЛЕ `_apply_post_nlu_guardrails` намеренно: устоявшиеся политики
    (walk-in «где сдать», prepare, адрес) сильнее — если guardrails увели метку
    из TEST_ASSIST, переклейка не срабатывает вовсе.

    Два исключения, где exact-матч не означает «спросили цену»:
    - категорийный запрос («чекап» — линейка пакетов): до сюда не доходит,
      `should_try_service` для него выключается и exact-матча нет;
    - формулировка ЦЕЛИ подбора («какие анализы сдать для профилактики»):
      матчер приземляет её на случайную строку по одному общему токену
      («…для ПРОФИЛАКТИКИ» → «Здоровая молодость: курс для профилактики и
      восстановления»), а спрашивают не цену этой услуги.

    :param decision: решение после каталожной инъекции и guardrails
    :param user_text: исходная реплика пациента (для гарда формулировки цели)
    :return: решение с меткой `PRICE`, если сработал класс, иначе исходное
    """

    if decision.label != "TEST_ASSIST":
        return decision
    flags = set(decision.flags or set())
    # Понимание хода (ход (б), шаг 1, `on`): список узнанных каталогом услуг — цена, как одна
    # узнанная услуга (П1). Вопрос о сроке сюда не доходит как цена: ему отвечает
    # `_maybe_readiness_reply` — срок из МИС или перевод (READINESS-AS-PRICE).
    understanding = turn_understanding.acting()
    if (
        understanding is not None
        and understanding.is_list
        and understanding.facet in (None, "price")
        and not detect_test_assist_goal_phrasing(user_text)
    ):
        turn_understanding.note_acted("list_is_price")
        return _relabel(decision, "understanding_list_is_price", label="PRICE", flags=flags | {"understanding_list_is_price"})
    if "catalog_service_exact" not in flags:
        return decision
    if not str((decision.entities or {}).get("service_name") or "").strip():
        return decision
    if button_flow.topic_firm(flags):
        # Подбор анализов задала кнопка: совпадение с каталогом по буквам тему не перебивает — названное подбор находит по данным
        # хода (`test_assist_answer`, цена — кодом), а не цена ближайшей строки. Список из разметки хода (`understanding_list_is_price`
        # выше) — решение LLM, не буквы: он остаётся ценой (STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM).
        turn_telemetry.note(button_topic_firm="p1")
        return decision
    if detect_test_assist_goal_phrasing(user_text):
        return _copy_decision(decision, flags=flags | {"test_assist_goal_phrasing_kept"})
    # Тему решила LLM (`topic_by_llm`) и она же решила свойство вопроса: «есть ли / делаете ли» — не голая цена (метка LLM
    # остаётся), «где / когда сдать» — адреса; «цена» и сбой — как было (STEP-2026-10-21-B-P1-FACET, g0811, g0816).
    if facet == "availability":
        return _copy_decision(decision, flags=flags | {"p1_availability_kept"})
    if facet == "where_when":
        return _relabel(decision, "p1_where_when", label="ADDRESS", flags=flags | {"p1_where_when"}, source="guardrail_post")
    if not facet and "topic_by_llm" in flags and fail_offer("facet"):
        # Сбой LLM свойства вопроса — не цена «как было» («есть ли ПЦР на аскаридоз?» → цены антител): честный оффер
        # (`llm_fail_offer` с узлом `facet`; BUG-2026-10-27-LLM-QUEUE-GIVES-UP, шаг 4).
        return _relabel(decision, "p1_facet_fail_offer", label="OTHER", flags=flags | {"llm_fail_offer"})
    if facet == "none" and _p1_facet_none_offer():
        # «Ни одно» от LLM — спрашивают об услуге не цену («со скольки лет берут кровь из вены?», g0577): данных хода на такой
        # вопрос нет — честный оффер оператора тем же путём, что OTHER от LLM, а не цена чужой строки
        # (BUG-2026-10-22-P1-FACET-NONE-AS-PRICE). Сбой LLM свойства (пустая строка) — П1 как было.
        return _relabel(
            decision, "p1_facet_none_offer", label="OTHER", flags=flags | {"p1_facet_none_offer", "topic_other_offer"}
        )
    return _relabel(
        decision, "test_assist_catalog_hit_is_price",
        label="PRICE",
        flags=flags | {"test_assist_catalog_hit_is_price"},
    )


_EXISTING_APPOINTMENT_OFFER: dict[str, str] = {
    "check": (
        "Проверить или подтвердить уже оформленную запись я не могу — эти данные "
        "видит только оператор. Перевести на оператора?"
    ),
    "cancel": (
        "Отменить уже оформленную запись я не могу — это делает оператор. "
        "Перевести на оператора?"
    ),
}


async def _confirm_appointment_change_verb(
    decision: RouteDecision,
    *,
    user_text: str,
    state: SessionState,
) -> RouteDecision:
    """«Отменить / перенести» без слова «запись»: о записи ли речь — решает LLM.

    С 04.10 правило записи срабатывает только на объекте записи; голый глагол — кандидат.
    Классификатор NLU «хочу отменить, заболела» относит к «прочему» (свип 04.10: пациент
    получал «Не совсем понял ваш запрос»), а «нужно ли отменять лекарства перед
    анализом» — верно к подготовке. Различает короткий вопрос к LLM; сбой LLM — как до
    04.10: просьба о записи (отмену увидит оператор).

    :param decision: решение после guardrails
    :param user_text: реплика пациента
    :param state: состояние сессии
    :return: APPOINTMENT с действием записи либо исходное решение
    """

    if decision.label in {"APPOINTMENT", "URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return decision
    if state.last_entities.get("appointment_flow_active"):
        return decision
    if state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM):
        return decision
    action = detect_appointment_change_without_object(user_text)
    if action is None or await is_own_appointment_change(user_text) is False:
        return decision
    return _relabel(
        decision, "appointment_change_verb",
        label="APPOINTMENT",
        entities={"appointment_action": normalize_appointment_action(action, user_text)},
        flags=set(decision.flags) | {"appointment_change_verb"},
        source="appointment_change_verb",
        confidence=max(decision.confidence, 0.9),
        needs_handoff=False,
    )


async def _confirm_existing_appointment_question(
    decision: RouteDecision,
    *,
    user_text: str,
    state: SessionState,
) -> RouteDecision:
    """«Мои записи», «мне подтверждена запись?»: о своей ли оформленной записи речь — решает LLM.

    Узкий детектор пропускал формулировки из реальной переписки, и они уходили в новую
    запись, в «Пока не понял» или в поиск результата (BUG-2026-10-08-EXISTING-APPT-LOOKUP-MISSED).
    Правило отбирает кандидатов, LLM различает «своя / новая / другое». Сбой LLM — как до
    08.10: решение не меняется.

    :param decision: решение после guardrails
    :param user_text: реплика пациента
    :param state: состояние сессии
    :return: решение с пометкой `existing_appointment_llm=check` либо исходное
    """

    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return decision
    if state.last_entities.get("_operator_offer_pending") or state.last_entities.get("appointment_flow_active"):
        return decision
    if state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM):
        return decision
    if decision.entities.get("appointment_action") in {"cancel", "reschedule"}:
        return decision
    if not detect_existing_appointment_candidate(user_text):
        return decision
    if await is_own_existing_appointment(user_text) is not True:
        return decision
    return _copy_decision(
        decision,
        entities={**decision.entities, "existing_appointment_llm": "check"},
        flags=set(decision.flags) | {"existing_appointment_llm"},
    )


def _existing_appointment_cancel_handoff(
    decision: RouteDecision,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence]:
    """Отмена СУЩЕСТВУЮЩЕЙ записи — сразу оператор (решения владельца 25.09, 04.10/08.10).

    Бот в МИС не ходит: запись видит оператор. Состояние оформления записи очищается, висящий
    вопрос снимается. Один путь для детектора/LLM (`_maybe_offer_operator_for_existing_appointment`)
    и для выбора «отменить» на вопрос «Хотите отменить или перенести запись?»
    (`_handle_appointment_action_pending`).
    """

    reset_appointment_runtime_state(state)
    memory.clear_pending(state)
    return (
        _relabel(
            decision, "existing_appointment_operator_handoff",
            label="OTHER",
            entities={},
            flags=set(decision.flags) | {"existing_appointment_operator_handoff", "existing_appointment_cancel"},
            source="existing_appointment",
            confidence=max(decision.confidence, 0.9),
            needs_handoff=False,
        ),
        Plan(label="OTHER"),
        Evidence(
            items={
                ek.OPERATOR_OFFER_RESPONSE: {
                    "text": handoff_message("existing_appointment_change"),
                    "handoff": True,
                }
            }
        ),
    )


def _maybe_offer_operator_for_existing_appointment(
    *,
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """П6-2: обращение по СУЩЕСТВУЮЩЕЙ записи → оператор вопросом.

    Решение владельца (14.08): проверку существующей записи бот не выполняет —
    распознаём намерение и предлагаем оператора, НЕ предлагая создать новую
    запись. Дополнение владельца: отмена записи обрабатывается так же —
    реквизиты отмены бот не собирает.

    Не срабатывает внутри АКТИВНОГО сценария записи: там «отменить» значит
    «прервать текущее оформление» (`cancel_confirm`), а не «отменить
    существующую запись». Перенос (`reschedule`) сюда не относится — у него
    свой путь, решение владельца его не касалось.

    :param decision: решение после guardrails
    :param user_text: реплика пациента
    :param state: состояние сессии
    :param memory: memory-store
    :return: готовый routing triple либо None
    """

    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return None
    if state.last_entities.get("_operator_offer_pending"):
        return None
    if state.last_entities.get("appointment_flow_active"):
        return None
    if state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM):
        return None
    kind = detect_existing_appointment_request(user_text)
    if kind is None and decision.label == "APPOINTMENT" and decision.entities.get("appointment_action") == "cancel":
        # Отмену без слова «запись» распознал не детектор, а LLM («не смогу прийти,
        # отмените») — тоже сразу оператор: реквизиты отмены бот не собирает (25.09).
        kind = "cancel"
    if kind is None and decision.entities.get("existing_appointment_llm") == "check":
        # Вопрос о своей записи распознал не детектор, а LLM («Мои записи»).
        kind = "check"
    if kind is None:
        return None

    if kind == "cancel":
        # Решение владельца 25.09 (заменяет 14.08 для отмены): сразу оператор, без
        # вопроса. Кнопка «Перенести или отменить» с 07.10 — подсказка, и её «отменить
        # запись» приходит сюда.
        return _existing_appointment_cancel_handoff(decision, state, memory)
    reset_appointment_runtime_state(state)
    memory.clear_pending(state)
    mark_operator_offer_pending(state, memory)
    return (
        _relabel(
            decision, "existing_appointment_operator_offer",
            label="OTHER",
            entities={},
            flags=set(decision.flags) | {"existing_appointment_operator_offer", f"existing_appointment_{kind}"},
            source="existing_appointment",
            confidence=max(decision.confidence, 0.9),
            needs_handoff=False,
        ),
        Plan(label="OTHER"),
        Evidence(
            items={
                # Переиспользуем существующий канал оффера оператора — рендерер
                # уже умеет его отдавать, новая плумбинг-ветка не нужна.
                ek.OPERATOR_OFFER_RESPONSE: {
                    "text": _EXISTING_APPOINTMENT_OFFER[kind],
                    "handoff": False,
                }
            }
        ),
    )


def _maybe_topic_handoff(
    *,
    decision: RouteDecision,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Тема со стратегией `handoff` в реестре тем — сразу оператор, текстом темы.

    Какие темы ведёт оператор, решают данные (`route.strategy` и `route.message_key` в
    `topic_registry.yaml`), а не код. Срочные, жалобы и медицинские вопросы тема не
    перехватывает — у них свои шаблоны.
    """

    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return None
    topic_id = extract_topic_id_from_flags(decision.flags)
    topic = topic_registry_get_topic(topic_id) if topic_id else None
    route = topic.get("route") if isinstance(topic, dict) else None
    if not isinstance(route, dict) or str(route.get("strategy") or "").strip().lower() != "handoff":
        return None
    memory.clear_pending(state)
    return (
        _relabel(
            decision, "topic_handoff",
            label="OTHER",
            entities={},
            flags=set(decision.flags) | {"topic_handoff"},
            source="topic_registry",
            needs_handoff=False,
        ),
        Plan(label="OTHER"),
        Evidence(
            items={
                ek.OPERATOR_OFFER_RESPONSE: {
                    "text": handoff_message(str(route.get("message_key") or "")),
                    "handoff": True,
                }
            }
        ),
    )


async def _maybe_insurance_topic(
    *,
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Вопрос об обслуживании по ДМС / ОМС — ответ темы по решению LLM (`insurance_topic = on`, класс К26).

    Стоит сразу за NLU (стадия `nlu_route`): раньше гарда врача, уточнений и инструментов, которые принимали «ДМС» за услугу
    или врача, — и в идущей записи: предпроверка записи (prelock) кладёт «по дмс» в слот услуги («Похоже, вы имели в виду
    «Госпитализация в терапевтический стационар»»), а записать по ДМС может только сотрудник. Памятка ДМС, ОМС, конкретный
    врач, гарантийное письмо — с оффером оператора вопросом («да» — перевод следующим ходом). Срочное, жалобу и медвопрос тема
    не перехватывает: у них свои шаблоны (принцип 4).
    """

    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return None
    reply = await insurance_topic.decide(user_text)
    if reply is None:
        return None
    mark_operator_offer_pending(state, memory)
    # Тему решила LLM: «низкая уверенность» NLU больше не повод переспрашивать (`recovery_policy.evaluate_recovery`).
    flags = (set(decision.flags) - {"low_confidence"}) | {insurance_topic.FLAG, f"{insurance_topic.FLAG}:{reply.kind}"}
    return (
        _relabel(
            decision, "insurance_topic",
            label="OTHER",
            entities={},
            flags=flags,
            source="insurance_topic",
            needs_handoff=False,
            context_action="new_topic",
            clarify_needed=False,
            clarify_reason="",
            clarify_slots=[],
        ),
        Plan(label="OTHER"),
        Evidence(items={ek.OPERATOR_OFFER_RESPONSE: {"text": reply.text, "handoff": False}}),
    )


async def _maybe_readiness_reply(
    *,
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Понимание хода (ход (б), шаг 1, `on`): вопрос о сроке готовности — срок из МИС или перевод.

    Срок печатает код из поля `deadline` строки прайса (принцип 1; решения владельца 10–11.10:
    явная единица — дословно, без единицы — дни). Без кода вопрос получал цену
    (READINESS-AS-PRICE) или свободный текст LLM со сроком, где единицу и строку LLM выбирала сама
    («креатинин и мочевина — 13–14 дней» — срок чужой строки). Услуга не названа уверенно или срока в
    МИС нет — честный перевод тем же текстом, что у вопроса о сроках в пути результатов (прод #675).
    """

    understanding = turn_understanding.acting()
    if (
        understanding is None
        or understanding.facet != "readiness"
        or not turn_understanding.is_price_turn(decision.label, memory.get_pending(state))
    ):
        return None
    memory.clear_pending(state)
    deadlines = await asyncio.to_thread(readiness_from_catalog, user_text)
    if not deadlines:
        # Слабое совпадение: услугу выбирает LLM из кандидатов, срок печатает код (ход (в)).
        deadlines = await readiness_by_choice(user_text)
    if deadlines:
        # срок показан — его услуга и есть контекст (L-14, шаг 3)
        names = list(dict.fromkeys(str(item.get("name") or "").strip() for item in deadlines if item.get("name")))
        shown_context.remember(state, names)
        turn_understanding.note_acted("readiness_from_mis")
        flag, items = "understanding_readiness_from_mis", {ek.READINESS_RESPONSE: deadlines}
    else:
        turn_understanding.note_acted("readiness_to_operator")
        flag, items = "understanding_readiness_handoff", {ek.OPERATOR_OFFER_RESPONSE: {"text": RESULT_TIMING_OPERATOR_TEXT, "handoff": True}}
    return (
        _relabel(decision, "readiness_reply", label="OTHER", entities={}, flags=set(decision.flags) | {flag}, needs_handoff=False),
        Plan(label="OTHER"),
        Evidence(items=items),
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
    elif decision.label == "PRICE":
        # Вопрос о цене: догадку правил по похожему слову не выносим на «да/нет» —
        # «капельница» → «Инфузионная терапия препаратом «Неодолпассе»», хотя в прайсе
        # есть и общая инфузионная терапия. Строку выбирает LLM (`_price_select_llm`)
        # среди всех вариантов. См. BUG-2026-10-02-PRICE-ASKS-NAMED-SERVICE.
        return None
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
    expects.ask(expects.Question.CATALOG_CONFIRM)

    return (
        _relabel(
            decision, "catalog_confirm_requested",
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


def _booking_option_lines(names: list[str]) -> tuple[str, bool]:
    """Строки вариантов записи: под `booking_options_price = on` — «— Название — N руб.» из строки прайса процедуры с тем же
    названием (принцип 1; давность прайса — как у пути цены); строка приёма врача и название без строки — без цены (25.09).

    :return: (строки, напечатана ли хоть одна цена)
    """

    if str(getattr(c, "MR_BOOKING_OPTIONS_PRICE", "off") or "").strip().lower() != "on":
        return "\n".join(f"— {name}" for name in names), False
    from .renderer import _extract_price_amount, _format_rub
    from .services._doctors_helpers import _is_consultation_service_query
    from .services._prices_helpers import _catalog_rows
    from .services.core import prices_checked

    try:
        rows = {" ".join(str(r.get("serviceName") or "").split()).casefold(): r for r in _catalog_rows(None) if isinstance(r, dict)}
    except Exception:  # каталог недоступен — названия без цен, как было
        rows = {}
    amounts: dict[str, int] = {}
    for name in names:
        row = rows.get(" ".join(name.split()).casefold())
        amount = _extract_price_amount(row) if row is not None and not _is_consultation_service_query(name) else None
        if amount is not None:
            amounts[name] = amount
    checked = prices_checked({"prices": list(amounts)}) if amounts else {}
    if not isinstance(checked, dict) or checked.get("handoff_required") or not checked.get("prices"):
        amounts = {}
    turn_telemetry.note(booking_options_priced=len(amounts))
    lines = "\n".join(f"— {name} — {_format_rub(amounts[name])}" if name in amounts else f"— {name}" for name in names)
    return lines, bool(amounts)


def _maybe_ask_booking_service_choice(
    *,
    decision: RouteDecision,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Цель записи не решена — названия услуг на выбор пациенту (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS).

    Грундер кладёт названия (`BOOKING_SERVICE_OPTIONS_KEY`), когда выбор LLM назвал несколько услуг или LLM недоступна
    на слабом совпадении (решение владельца 18.10 — как L-13 у цены). Записи на догадку правил нет; ожидание записи с
    недостающей услугой — следующий ход с названием из списка продолжает запись.
    """

    entities = decision.entities or {}
    names = [str(n).strip() for n in entities.get(BOOKING_SERVICE_OPTIONS_KEY) or () if str(n).strip()]
    general = bool(entities.get(BOOKING_SERVICE_CLARIFY_KEY))
    if decision.label != "APPOINTMENT" or not (names or general):
        return None
    action = str((decision.entities or {}).get("appointment_action") or "").strip()
    if action:
        state.last_entities["appointment_action"] = action
    state.last_entities.pop("service_name", None)
    memory.clear_pending(state)
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["service_name"])
    expects.ask(expects.Question.BOOKING_SERVICE, kind=expects.PICK, options=tuple(names))
    # Без номеров: ответ «1» правила не разбирают (ответ на висящий вопрос силами LLM — подбаг A
    # BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH, решение владельца о `_expects`); название из списка — сильное совпадение.
    lines, priced = _booking_option_lines(names)
    if general:
        # Названа группа услуг («массаж»): названия — примеры, пациент может назвать и свою.
        text = "Уточните, пожалуйста, какую именно услугу записать — как в направлении врача"
        text += f", например:\n{lines}" if names else "."
    else:
        text = f"Уточните, пожалуйста, на какую услугу вас записать:\n{lines}\nНапишите название из списка."
    if priced:
        from .renderer import _bundle_text

        text = _bundle_text([text], "", priced=True)
    clean = {k: v for k, v in entities.items() if k not in {BOOKING_SERVICE_OPTIONS_KEY, BOOKING_SERVICE_CLARIFY_KEY}}
    return (
        _copy_decision(decision, entities=clean, flags=set(decision.flags) | {"booking_service_choice_requested"}),
        Plan(label="APPOINTMENT"),
        Evidence(items={ek.CATALOG_CONFIRM_RESPONSE: {"text": text, "handoff": False}}),
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
    if turn_names.on("names_doctor_followup") and turn_names.name_reply_is_doctor(raw, cached_doctor_fios()) is False:
        # Реплика — ФИО, но не врача справочника целиком: нечёткий `resolve_doctor_name` нашёл бы однофамильца пациента
        # («Сидорова Анна» → врач «Сидоров») и записал бы к нему (R-01, консилиум 25.10).
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
    # Все пути роутера строят план здесь — пожелание по времени (вариант A) не теряется,
    # какой бы веткой ни пришла реплика.
    _capture_appointment_wish(decision, user_text, state, memory)
    turn_understanding.check_planned(decision.label)
    return planner_build_plan(decision, state, user_text, memory, runtime_options=runtime_options)


async def execute_plan(plan: Plan, state: SessionState, services: Services) -> Evidence:
    return await executor_execute_plan(plan, state, services)


_WALKIN_FLAGS = frozenset({"rule_nonbookable_walkin", "policy_nonbookable_walkin"})


# Темы, где реплика — не голое название услуги по смыслу хода: безопасность, запись и результаты решают свои пути.
# Обзор голого названия — и на темах записи, адресов и врачей (решение владельца 26.10, п. 4 топ-5): «УЗИ щитовидки» правило ставит
# в запись, «Фгдс» — во врачей, и вместо цены пациент получал «на какую услугу вас записать» или список врачей. На цене и подборе
# анализов у хода свой ответ по данным (выбор строки цены, ответ подбора) — строже, и обзор его не перебивает: карточка 26.10 —
# «мазок на онкоцитологию» → «Мазок на эозинофилы…», «соскоб на лишай» → «ЗППП: соскоб», «комплекс на ИППП» → «Обязательный комплекс».
_OVERVIEW_SKIP_LABELS = role_pick._SAFETY_LABELS | {"TEST_RESULT", "PRICE", "TEST_ASSIST"}
# На этих темах строки приёма врача — не кандидаты обзора: цену приёма к специальности не называем, пока врач не выбран (25.09).
_OVERVIEW_NO_RECEPTION_LABELS = frozenset({"APPOINTMENT", "DOCTOR_INFO", "DOCTOR_SCHEDULE"})


async def _maybe_service_overview(
    *, decision: RouteDecision, user_text: str, state: SessionState, memory: MemoryStore
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Голое название услуги → максимум по данным (класс 3, решение владельца 21.10, `service_overview.py`): короткая
    реплика без висящего вопроса — кандидат, решает LLM. Не такой ход, выключатель снят, сбой — None (как было)."""

    if (
        not service_overview.enabled()
        or decision.label in _OVERVIEW_SKIP_LABELS
        or decision.source == "button"
        or memory.get_pending(state)
        or state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
        or not service_overview.candidate(user_text)
        # Явный вопрос о цене, сроке, подготовке — не голое название; «запись» и «где» по буквам детектора walk-in не в
        # счёт («Анализ T-SPOT» он считает записью) — их решает LLM (исход `asks`).
        or turn_understanding.price_turn_facet(user_text)[0] in {"price", "readiness", "prepare"}
    ):
        return None
    overview = await service_overview.decide(user_text, without_reception=decision.label in _OVERVIEW_NO_RECEPTION_LABELS)
    if overview is None:
        return None
    return decision, Plan(label=decision.label), Evidence(items={ek.SERVICE_OVERVIEW: overview.payload()})


def _doctors_by_service_not_found(payload: Any) -> bool:
    """Справочник врачей по вопросу об услуге никого не дал (услуга названа, врач по ФИО — нет)."""

    if not isinstance(payload, dict) or payload.get("doctors"):
        return False
    used = payload.get("entities_used") if isinstance(payload.get("entities_used"), dict) else {}
    return bool(str(used.get("service_query") or "").strip()) and not str(used.get("doctor_query") or "").strip()


_TEST_ASSIST_NAME_SLOT = "_any_of:test_goal,test_name"


def _test_assist_guess_as_name(decision: RouteDecision, state: SessionState, memory: MemoryStore, guess: str, user_text: str) -> None:
    """Анализ назван (каталог нашёл догадку), а слота названия у хода подбора нет — слова пациента в слот (`test_assist_from_data`).

    Суд «да/нет» по догадке на этом ходе пропущен — строку выбирает LLM ответа по данным, догадка — его первый кандидат. Но под
    висящим вопросом кнопки «Цена анализа» (ждёт `service_name`) заземление выбрасывает `test_name` хода, и пустой слот подбора
    строил воронку «Для какой цели хотите подобрать анализы?», хотя анализ назван (карточка 26.10, `buttons_v1`: g0812, g0829,
    g0852; до слияния — «Похоже, вы имели в виду …? да/нет»). В слот — реплика, как её дал бы классификатор без кнопки: поиск по
    самой догадке тянул соседние строки, и LLM выбирала их (g0812 — «Расширенная копрология» вместо исследования на паразиты).
    """

    text = " ".join(str(user_text or "").split())
    if not guess or not text or decision.label != "TEST_ASSIST" or _TEST_ASSIST_NAME_SLOT not in missing_slots("TEST_ASSIST", state.last_entities):
        return
    memory.merge_entities(state, {"test_name": text}, label="TEST_ASSIST")
    decision.flags = set(decision.flags) | {"test_assist_guess_as_name"}


async def _walkin_by_llm(decision: RouteDecision, user_text: str, *, site: str) -> tuple[RouteDecision, bool | None]:
    """Детектор «без записи» сработал — кандидат; под `walkin_by_llm` решает LLM по правилу клиники (класс 4,
    решения владельца 21.10, `walkin.py`). (решение с флагом хода, True — без записи, False — по записи или о другом,
    None — выключатель снят или сбой LLM: как было, решает детектор). Решение хода спрашивается один раз."""

    if not walkin.enabled():
        return decision, None
    answer = walkin.decided(decision.flags)
    if answer is None:
        answer = await walkin.service(user_text, site=site)
        if answer is None:
            return decision, None
        decision = _copy_decision(decision, flags=set(decision.flags) | walkin.flags_for(answer))
    return decision, bool(answer)


async def _walkin_sentence_by_llm(decision: RouteDecision, user_text: str) -> RouteDecision:
    """Ответ адресами с фразой «… без записи, в порядке живой очереди» (флаги walk-in): что именно без записи — решение
    LLM, а не буквы реплики (`response_builder.build_address_response`). Не такой ход или выключатель снят — как было."""

    if decision.label != "ADDRESS" or not any("nonbookable" in str(f) for f in decision.flags):
        return decision
    decision, _by_llm = await _walkin_by_llm(decision, user_text, site="address_sentence")
    return decision


async def _guard_walkin_named_service(
    decision: RouteDecision,
    *,
    user_text: str,
    state: SessionState,
    services: Services,
) -> RouteDecision:
    """«Без записи» — только про анализы и ЭКГ, не про услугу, которую пациент назвал сам (08.10).

    Walk-in по контексту (BUG-F: «нужна запись?» в лаб-контексте — живая очередь) не видел,
    что фраза называет другую услугу: после кнопки «Нужно ли записываться» «а на УЗИ
    брюшной полости нужна запись?» → «УЗИ … без записи». Стоит после grounding: только
    здесь известны и решение walk-in, и услуга из реплики.

    - Своя лаб-улика в реплике («сдать кровь», «анализ», «ЭКГ») — решение правила остаётся.
    - Своей услуги в реплике нет — ответ про класс из контекста («анализы»), а не про
      строку каталога, найденную по словам вопроса. Вопрос — о
      показанной прошлым ответом услуге: класс нужен только ответу об адресах, контекст
      после ответа — показанное (флаг `walkin_shown_service`, `shown_context.sync_address`).
    - Услуга названа — вид решает каталог МИС: анализ (есть срок готовности) — без записи;
      иначе или неизвестно — запись, «без записи» не утверждаем. Как и
      `_classify_catalog_service_kind`, анализом не считаем то, что сам пациент назвал
      процедурой: каталог по буквам даёт «гастроскопию» → «Гастрин» (K14).
    """

    if decision.label != "ADDRESS" or decision.source == "button" or not (_WALKIN_FLAGS & set(decision.flags)):
        return decision
    if detect_nonbookable_walkin_intent(user_text):
        return decision
    asked_about = strip_booking_necessity(user_text)
    specialty = extract_specialty(asked_about)

    def to_booking(**named: str) -> RouteDecision:
        # Класс из контекста («анализы» от кнопки) — не услуга записи: иначе запись
        # подхватит его из состояния («Есть возможность записи на анализы», свип 08.10).
        if walkin.is_class(state.last_entities.get("service_name")):
            state.last_entities.pop("service_name", None)
        entities = {k: v for k, v in decision.entities.items() if k != "service_name"}
        return _relabel(
            decision, "walkin_named_bookable_service",
            label="APPOINTMENT",
            entities={**entities, **named},
            flags=(set(decision.flags) - _WALKIN_FLAGS) | {"walkin_named_bookable_service"},
            source="guardrail_post",
        )

    if specialty:
        # «а на приём к неврологу нужна запись?» — врач: выделитель услуги приём не видит.
        return to_booking(specialty=specialty)
    phrase = extract_service_phrase(asked_about)
    if not phrase:
        hint = nonbookable_service_hint(user_text, state.last_entities) or "анализы"
        about_shown = shown_context.walkin_about_shown()  # прошлый ответ ничего не показал — пусто, решение как было
        if decision.entities.get("service_name") == hint:
            return _copy_decision(decision, flags=set(decision.flags) | about_shown) if about_shown else decision
        return _copy_decision(
            decision,
            entities={**decision.entities, "service_name": hint},
            flags=set(decision.flags) | {"walkin_context_class"} | about_shown,
        )
    match = await services.match_catalog_service(phrase, context_text=user_text)
    canonical = str(match.get("canonical") or "").strip() if match.get("status") in {"exact", "fuzzy"} else ""
    is_lab = bool(canonical) and await services.is_lab_catalog_service(canonical)
    named_procedure = bool(_PRICE_PROCEDURE_LIKE_RE.search(phrase))
    if is_lab and not named_procedure:
        return decision
    if canonical and match.get("status") == "exact" and not is_lab:
        return to_booking(service_name=canonical)
    return to_booking()


def _slot_value_keeps_topic(decision: RouteDecision, state: SessionState, memory: MemoryStore) -> RouteDecision:
    """Ответ на вопрос-слот, который одна функция ответа на висящий вопрос признала ответом (`pending_buttons`, решение
    LLM), остаётся в теме вопроса.

    Висит вопрос-слот («К какой процедуре нужна подготовка?», «Для какой цели подобрать анализы?», реквизиты результата):
    LLM решила «это ответ» — метка темы вопроса, `continue` (смена темы классификатором не сбрасывает вопрос), флаг
    `pending_slot_value`: значение слота кладёт блок quick-fill хода. Значение, разобранное правилами (сужение без LLM),
    решение хода не трогает — как было; нет, сбой, срочное — как было (BUG-2026-10-08-SLOT-REPLY-LOST-TO-LLM,
    BUG-2026-10-08-PREPARE-ANESTHESIA-IN-QUERY; один путь вместо `slot_reply_llm`).
    """

    found = pending_answer.slot_answer()
    if found is None or found.outcome is not pending_answer.Outcome.ANSWER or found.source is not pending_answer.Source.LLM:
        return decision
    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"} or decision.needs_handoff:
        return decision
    pend_label = str((memory.get_pending(state) or {}).get("label") or "")
    if pend_label not in pending_answer.slot_topics() or not missing_slots(pend_label, state.last_entities):
        return decision
    return _relabel(decision, "pending_slot_value", label=pend_label, context_action="continue", flags=set(decision.flags) | {"pending_slot_value"})


def _slot_filled_by_rules(user_text: str, state: SessionState, services: Services, memory: MemoryStore) -> bool:
    """Правила (quick-fill) разбирают в реплике значение недостающего слота темы `_pending` — сужение без LLM для вопроса-
    слота («колоноскопия», «Иванов, 1989, Бг, 123»)."""

    pending = memory.get_pending(state)
    label = str((pending or {}).get("label") or "")
    missing = pending.get("missing") if isinstance(pending, dict) else None
    if not label or not isinstance(missing, list) or not missing:
        return False
    quick = quick_fill_entities_from_text(user_text, state.last_entities, missing, services, pending_label=label)
    return bool(quick) and missing_slots(label, {**state.last_entities, **quick}) != missing_slots(label, state.last_entities)


# Ответ «не могу назвать» на вопрос о реквизитах результата анализов (решение владельца 19.10, п. 4): подсказка, где их
# найти, и оффер оператора вопросом; результаты по ФИО не ищем (R-01).
TEST_RESULT_DATA_HINT_TEXT = (
    "Номер анализа указан в чеке или в договоре, которые выдают при сдаче анализов. "
    "Если их нет под рукой, поможет оператор. Перевести на оператора?"
)
# Уточняющий вопрос о вопросе кнопки результатов («где на бланке код?», «как узнать сам результат?», `button_side_answer`):
# где реквизиты и как посмотреть результат — словами подсказок бота (выше и `lab_tests`), вопрос о реквизитах висит дальше.
TEST_RESULT_SIDE_TEXT = (
    "Код и номер анализа указаны в чеке или в договоре, которые выдают при сдаче анализов. "
    f"Посмотреть результат можно на сайте {RESULTS_PORTAL_URL} — вкладка «Результаты анализов» (вверху слева на сайте) — "
    "или здесь: напишите фамилию, год рождения, код анализа и номер анализа, например: «Иванов, 1989, Бг, 123». "
    "Если чека и договора нет под рукой, поможет оператор."
)


async def _handle_slot_value_pending(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """«Не могу ответить» на вопрос-слот реквизитов результата — подсказка и оффер оператора (`pending_buttons`).

    Решение — одной функции ответа на висящий вопрос (исход `no`: «не помню номер анализа», «не знаю», «нет»). Прочие
    темы и исходы — обычный путь, как было.
    """

    found = pending_answer.slot_answer()
    if found is None or found.outcome is not pending_answer.Outcome.NO:
        return None
    if str((memory.get_pending(state) or {}).get("label") or "") != "TEST_RESULT":
        return None
    mark_operator_offer_pending(state, memory)
    return (
        RouteDecision(label="OTHER", confidence=0.9, entities={}, flags={"test_result_data_hint"}, needs_handoff=False),
        Plan(label="OTHER"),
        Evidence(items={ek.OPERATOR_OFFER_RESPONSE: {"text": TEST_RESULT_DATA_HINT_TEXT, "handoff": False}}),
    )


async def _handle_button_reframe(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Кнопка — рамка хода (`button_answer_topic = frame`, STEP-2026-10-29-BUTTON-FRAME): «другую тему» судьи ответа на вопрос
    кнопки проверяет отдельный вопрос LLM (`button_flow.reframe`); сообщение в теме кнопки — ход идёт общим путём в теме кнопки
    (`button_flow.answer_topic`), её не перебивают несрочные правила. Реплика сразу после листа без своего вопроса — тот же
    вопрос рамки (`button_flow.reframe_leaf`). Ход сам не отвечает — всегда None."""

    if runtime_options is not None and runtime_options.llm_mode == "strict":
        return None
    pending = memory.get_pending(state)
    if button_flow.frame_firm() and not button_flow.question_button(pending) and await button_flow.reframe_leaf(user_text):
        return None  # висящий вопрос, если есть, — от того же нажатия листа: реплика в его рамке
    found = pending_answer.slot_answer()
    if found is None or found.outcome is not pending_answer.Outcome.NEW_TOPIC or found.rest:
        return None
    if button_flow.frame_firm() and button_flow.question_button(pending):
        await button_flow.reframe(user_text, pending)
    return None


async def _handle_button_new_topic(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Чужая тема в ответ на вопрос кнопки меню — подсказка нужной кнопки, а не угадывание темы общим путём
    (`button_topic_hint`, BUG-2026-10-25-BUTTON-FOREIGN-TOPIC).

    Одна функция ответа на висящий вопрос решила «другая тема», а тему общий путь угадывал бы LLM-классификатором (правило
    не сработало или его ветка в `topic_by_llm`): LLM выбирает листья меню, где пациент найдёт ответ (`button_flow.hint`).
    Надёжное правило другой темы («а в субботу открыто?» — часы филиалов) отвечает само, как было: прямой верный ответ
    лучше подсказки. Смешанное сообщение — и ответ на вопрос кнопки, и другой вопрос — ход идёт по теме кнопки, подсказка —
    в конце ответа. Подсказки нет (ничего не подошло, только тема кнопки, сбой LLM) — общий путь, как было. Срочное,
    жалоба и медвопрос по правилам отвечены раньше (`orchestrator.early_guards`).
    """

    from .classifier import deterministic_rule_decision
    from .nlu_pipeline import rule_defers_topic

    pending = memory.get_pending(state)
    current = button_flow.question_button(pending)
    found = pending_answer.slot_answer()
    if not button_flow.hint_on() or not current or found is None or found.outcome is not pending_answer.Outcome.NEW_TOPIC:
        return None
    if button_flow.reframed():
        return None  # вопрос рамки вернул реплику в тему кнопки (`frame`)
    if runtime_options is not None and runtime_options.llm_mode == "strict":
        return None
    rule = await deterministic_rule_decision(user_text, state.last_entities, runtime_options=runtime_options, allow_refine=False, attach_secondary=False)
    if rule is not None and not rule_defers_topic(rule, user_text):
        return None
    expected = pending_answer.expected_from(state.last_entities)
    hint = await button_flow.hint(user_text, current, button_flow.asked_with_button(current, expected[1] if expected is not None else ""))
    if hint is None:
        return None
    if hint.mixed:
        button_flow.answer_with_hint(str((pending or {}).get("label") or ""), hint)
        return None
    memory.clear_pending(state)
    return (
        RouteDecision(
            label="OTHER",
            confidence=0.9,
            entities={},
            flags={"button_hint", *(f"button_hint:{leaf}" for leaf in hint.leaves)},
            needs_handoff=False,
            context_action="new_topic",
            source="button_hint",
        ),
        Plan(label="OTHER"),
        Evidence(items={ek.OPERATOR_OFFER_RESPONSE: {"text": button_flow.hint_text(hint.leaves), "handoff": False}}),
    )


async def _handle_address_followup(
    *,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Реплика сразу после «Адреса и часы работы» / «Где и до скольки принимают» — ответ по данным филиалов, а не общий список
    (`address_followup_llm`, BUG-2026-10-25-ADDRESS-FOLLOWUP-TEMPLATE): LLM выбрала, какие филиалы названы и что спрошено, код
    печатает их адрес, телефон, график и что там делают; чего в данных нет — честно и оффер оператора. Общий вопрос, другая тема,
    сбой LLM — общий путь, как было. Срочное и явный оператор отвечены раньше."""

    if not address_flow.followup_on() or not address_flow.followup_turn():
        return None
    if runtime_options is not None and runtime_options.llm_mode == "strict":
        return None
    try:
        # Филиалы — тем же путём, что общий ответ адресами: живой срез, снимок или запасной список — и с флагами.
        payload = await services.address_info("", {})
    except Exception:
        payload = {}
    regions = [b for b in (payload.get("branches") or []) if isinstance(b, dict) and str(b.get("address") or "").strip()]
    if not regions:
        return None
    found = await address_flow.followup(user_text, regions)
    if found is None:
        return None
    text, offer = address_flow.followup_text(found, regions)
    if offer:
        mark_operator_offer_pending(state, memory)
    # С оффером — метка OTHER: ответ с вопросом «Перевести на оператора?» — свой, не переспрос по слоту (`_extract_pending_response`).
    label = "OTHER" if offer else "ADDRESS"
    return (
        RouteDecision(
            label=label,
            confidence=0.9,
            entities={},
            flags={"address_followup", f"address_followup:{found.outcome}"},
            needs_handoff=False,
            context_action="continue",
            source="address_followup",
        ),
        Plan(label=label),
        Evidence(items={ek.OPERATOR_OFFER_RESPONSE: {"text": text, "handoff": False}}),
    )


async def _apply_post_nlu_guardrails(
    decision: RouteDecision,
    state: SessionState,
    user_text: str,
    services: Services,
    memory: MemoryStore,
) -> RouteDecision:
    """Apply post-NLU flow-continuity overrides before plan building.

    Consolidates the cascade of narrow heuristics that sat inline between
    ``_inject_catalog_candidates`` and ``_apply_appointment_continuity_overrides``:

    * PRICE → PREPARE short follow-up
    * PRICE → ADDRESS short follow-up
    * Active APPOINTMENT flow datetime prelock
    * apply_context_action (promote/continue/new_topic)
    * PRICE city-only reply override
    * verified-doctor label promotion
    * clear stale doctor on specialty-only queries (state mutation)
    * non-bookable walk-in → ADDRESS
    * PREPARE short follow-up hold
    * DOCTOR_INFO → DOCTOR_SCHEDULE follow-up (awaits services)
    * DOCTOR_SCHEDULE → APPOINTMENT datetime follow-up (state mutation)
    * _apply_appointment_continuity_overrides

    State is mutated in-place for the blocks that legacy code mutated directly
    (patient_name reset, secondary_queue clear, stale doctor-key cleanup);
    behaviour is byte-for-byte identical to the previous inline form.

    :param decision: routing decision after NLU + catalog candidate injection
    :param state: session state
    :param user_text: current user turn
    :param services: service layer (for doctor-name resolution in follow-ups)
    :param memory: memory store (for pending inspection)
    :return: decision after the full guardrail cascade
    """

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
        decision = _relabel(
            decision, "flow_price_followup_prepare",
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
        by_llm: bool | None = None
        if followup_address_hint and detect_nonbookable_walkin_intent(user_text, merged_ctx):
            decision, by_llm = await _walkin_by_llm(decision, user_text, site="flow_price_followup_address")
        if by_llm is not False and followup_address_hint and detect_nonbookable_walkin_intent(user_text, merged_ctx):
            entities = dict(decision.entities)
            if not entities.get("service_name"):
                entities["service_name"] = service_context
            decision = _relabel(
                decision, "flow_price_followup_address",
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
        decision = _relabel(
            decision, "flow_datetime_appointment_prelock",
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.68),
            flags=set(decision.flags) | {"flow_datetime_appointment_prelock"},
            needs_handoff=False,
            context_action="continue",
        )
    decision = _slot_value_keeps_topic(decision, state, memory)
    decision = apply_context_action(decision, state, user_text, memory)

    pending_before_overrides = memory.get_pending(state)
    if (
        isinstance(pending_before_overrides, dict)
        and pending_before_overrides.get("label") == "PRICE"
        and decision.label == "ADDRESS"
        and is_city_only_reply(user_text)
    ):
        decision = _relabel(
            decision, "flow_price_city_reply_override",
            label="PRICE",
            confidence=max(decision.confidence, 0.72),
            flags=set(decision.flags) | {"flow_price_city_reply_override"},
            needs_handoff=False,
            context_action="continue",
        )

    promoted_label, promoted_flags = apply_verified_doctor_override(decision.label, set(decision.flags), user_text)
    if promoted_label != decision.label or promoted_flags != decision.flags:
        decision = _relabel(
            decision, "verified_doctor_override",
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

    # Тему хода решила LLM (`topic_by_llm`, ход (б), этап 4): детектор «без записи» её не перебивает — метка ADDRESS есть
    # и у LLM (STEP-2026-10-21-B-TOPIC-BY-LLM, g0716: список анализов «можно ли за один визит» уходил в адреса). Так же —
    # тему кнопки под `button_topic_firm` (STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM).
    keeps_button = button_flow.topic_firm(decision.flags)
    if keeps_button and decision.label in {"APPOINTMENT", "TEST_ASSIST"} and detect_nonbookable_walkin_intent(user_text, state.last_entities):
        turn_telemetry.note(button_topic_firm="walkin")
    if decision.label in {"APPOINTMENT", "TEST_ASSIST"} and not detect_prepare_intent(user_text) and "topic_by_llm" not in decision.flags and not keeps_button:
        merged_ctx = dict(state.last_entities)
        merged_ctx.update(decision.entities or {})
        by_llm = None
        if detect_nonbookable_walkin_intent(user_text, merged_ctx):
            decision, by_llm = await _walkin_by_llm(decision, user_text, site="policy_nonbookable_walkin")
        if by_llm is not False and detect_nonbookable_walkin_intent(user_text, merged_ctx):
            entities = dict(decision.entities)
            svc = walkin.decided(decision.flags) if by_llm else nonbookable_service_hint(user_text, merged_ctx)
            # Для walk-in (анализы/ЭКГ без записи) важен КЛАСС услуги, а не строка
            # каталога. Если service_name пуст или был мягко подставлен catalog-
            # инъекцией (флаг catalog_service_exact — напр. ошибочный «профиль 2» на
            # запрос «профиль 1»), заменяем генериком, чтобы не пинить/не утечь в
            # state неверную услугу.
            if svc and (
                not entities.get("service_name")
                or "catalog_service_exact" in set(decision.flags)
            ):
                entities["service_name"] = svc
            decision = _relabel(
                decision, "policy_nonbookable_walkin",
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
        decision = _relabel(
            decision, "flow_prepare_followup_override",
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
            decision = _relabel(
                decision, "flow_doctor_info_to_schedule",
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
            reactivate_appointment_collecting(state)
        entities = dict(decision.entities)
        for key in ("doctor_id", "doctor_name", "branch_id", "branch_name"):
            if not entities.get(key) and state.last_entities.get(key):
                entities[key] = state.last_entities.get(key)
        decision = _relabel(
            decision, "flow_schedule_to_appointment",
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.72),
            entities=entities,
            flags=set(decision.flags) | {"flow_schedule_to_appointment"},
            needs_handoff=False,
            context_action="continue",
        )

    decision = _apply_appointment_continuity_overrides(decision, state, user_text)
    return decision


async def _handle_secondary_queue_pending(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Resolve a previously-offered secondary intent awaiting user ack.

    Mirrors the legacy block that sat inline at the top of the old entry
    point (removed 29.10, К3): if the previous turn offered a queued follow-up
    (``_secondary_offer_pending``) and the current reply is a short "да"/"нет"
    (or a short doctor last-name for doctor-flavoured follow-ups), promote the
    first queued label into a full routing triple. On a "no"/ambiguous reply
    the pending flag is cleared and ``None`` is returned so normal routing
    continues.

    :param user_text: current user turn
    :param state: session state
    :param services: service layer (used to resolve doctor entities)
    :param memory: memory store (for entity merging)
    :param runtime_options: runtime options propagated into plan building
    :return: fully-materialised routing triple when the turn is consumed by the
        secondary-queue flow, else ``None`` so normal routing continues.
    """

    queue = get_secondary_queue(state)
    if not (
        state.last_entities.get("_secondary_offer_pending")
        and queue
        and state.dialog.phase not in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
    ):
        return None

    show_all_q, secondary_q = expects.Question.SHOW_ALL, expects.Question.SECONDARY_OFFER
    if pending_answer.shows_all():
        # Ответ на подсказку «напишите: "все"» («да, давайте все») — «показать все»: вторичный оффер снимается, как на
        # «все», ответ читает путь цены (`_price_show_all_requested`).
        state.last_entities["_secondary_offer_pending"] = False
        return None
    if pending_answer.passes(secondary_q) or pending_answer.passes(show_all_q):
        # Другая тема или хвост («да, как подготовиться?» → «как подготовиться?») — ход идёт обычным путём.
        state.last_entities["_secondary_offer_pending"] = False
        return None

    def _rules() -> str:
        reply_kind = contextual_reply_kind(user_text)
        if reply_kind == "other" and _is_secondary_soft_yes(user_text):
            reply_kind = "yes"
        return reply_kind

    reply_kind = pending_answer.kind(secondary_q, _rules)
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
    return None


def _handle_operator_offer_pending(
    user_text: str,
    state: SessionState,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence] | None:
    """Resolve a previously-offered operator handoff awaiting user ack.

    :param user_text: current user turn
    :param state: session state
    :param memory: memory store (for clearing pending markers)
    :return: fully-materialised routing triple when the turn is consumed by the
        operator-offer flow, else ``None`` so normal routing continues.
    """

    if not state.last_entities.get("_operator_offer_pending"):
        return None

    if pending_answer.passes(expects.Question.OPERATOR_OFFER):
        # «нет, скажите сколько стоит ферритин» — оффер снят, хвост идёт обычным путём.
        _clear_operator_offer_pending(state, memory)
        return None

    def _rules() -> str:
        # Запасной путь одной функции (сбой LLM): правила обработчика.
        return "yes" if explicit_operator_requested(user_text) else contextual_reply_kind(user_text)

    reply_kind = pending_answer.kind(expects.Question.OPERATOR_OFFER, _rules)
    if reply_kind == "pause":
        reply_kind = "no"  # «я подумаю» — оффер снят без перевода и без «не понял»
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
            Evidence(items={ek.OPERATOR_OFFER_RESPONSE: {"text": handoff_message("manual_operator"), "handoff": True}}),
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
                    ek.OPERATOR_OFFER_RESPONSE: {
                        "text": "Хорошо, продолжаем диалог. Можете задать другой вопрос.",
                        "handoff": False,
                    }
                }
            ),
        )
    # Pending cleared; caller proceeds to normal routing.
    _clear_operator_offer_pending(state, memory)
    return None


async def _resolve_nlu_decision_before_doctor_guard(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
    button_topic: str = "",
) -> tuple[RouteDecision, dict[str, Any]]:
    """Строит NLU-решение до doctor verification.

    Хелпер сохраняет legacy-порядок: очистка stale debug-ключей, appointment
    prelock, основной NLU и topic-registry override. Зовёт его
    orchestrator-stage `nlu_route` (прежняя точка входа `route_patient_message`
    снята 29.10, К3).

    :param user_text: текущий текст пользователя
    :param state: состояние сессии
    :param services: сервисный слой для prelock-логики
    :param memory: memory-store для чтения pending-состояния
    :param runtime_options: runtime-настройки LLM/NLU
    :param button_topic: тема нажатой кнопки режима `pass` (`button_menu`); пусто — набранный текст
    :return: решение маршрутизации и NLU debug-метаданные
    """

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
        # #3 (прод #668): ФИО принимается формой `_looks_like_patient_fio`
        # (2+ кириллических слова), поэтому «Время изменени» (коррекция) прошло
        # как имя → «Запись: Время Изменени…». LLM-валидатор отсекает команду/
        # коррекцию ТОЛЬКО на приёме нового ФИО в активной записи; отвергнутое
        # → штатный путь `__clear_patient_name` (сброс + переспрос ФИО).
        # Fail-open внутри валидатора: сбой/таймаут/kill-switch → принять по форме.
        if (
            is_appointment_waiting_patient_name(
                pending_before_nlu if isinstance(pending_before_nlu, dict) else None
            )
            and appointment_prelock.get("patient_name")
            and not await is_patient_name_reply(str(appointment_prelock.get("patient_name")))
        ):
            appointment_prelock.pop("patient_name", None)
            appointment_prelock["__clear_patient_name"] = True
        # #4 (аудит 2026-07-22): service_name-слот принимает командные слова формой
        # `_SERVICE_SINGLE_WORD_RE` (любое слово ≥4 букв) → «Запись: … Поменяй …
        # Подтверждаете?». LLM-валидатор отсекает команду ТОЛЬКО на приёме НОВОЙ
        # услуги (свежий slot-update этого хода) и только для ОДНОСЛОВНОЙ реплики
        # (командо-риск; многословные услуги «УЗИ брюшной полости» LLM не дёргают).
        # Отвергнутое → штатный `__clear_service_name` (сброс + переспрос услуги).
        # Fail-open внутри валидатора: сбой/таймаут/kill-switch → принять по форме.
        _svc_slot = str(appointment_prelock.get("service_name") or "").strip()
        if (
            _svc_slot
            and len(_svc_slot.split()) == 1  # командо-риск: одно слово
            and not await is_service_name_reply(_svc_slot)
        ):
            appointment_prelock.pop("service_name", None)
            appointment_prelock["__clear_service_name"] = True
        clear_service_name = bool(appointment_prelock.pop("__clear_service_name", False))
        clear_patient_name = bool(appointment_prelock.pop("__clear_patient_name", False))
        if clear_service_name:
            state.last_entities.pop("service_name", None)
            state.last_entities.pop("test_name", None)
        if clear_patient_name:
            state.last_entities.pop("patient_name", None)
            reactivate_appointment_collecting(state)
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
            # Ответ на вопрос кнопки, признанный ответом, — тема кнопки (`button_answer_topic`); смешанное сообщение — тоже
            # (`button_topic_hint`: ответ по теме кнопки, подсказка в конце — её не перебивает и надёжное правило). Под
            # `frame` кнопка — рамка хода: тему не перебивают и несрочные правила (STEP-2026-10-29-BUTTON-FRAME). Только
            # когда тема есть.
            answer_topic = button_flow.answer_topic(pending_before_nlu)
            firm_topic = answer_topic and (button_flow.mixed_topic() or button_flow.frame_firm())
            firm = {"answer_topic_firm": True} if firm_topic else {}
            nlu_result = await analyze_with_candidates(
                user_text,
                state,
                runtime_options=runtime_options,
                **({"button_topic": button_topic} if button_topic else {}),
                **({"answer_topic": answer_topic} if answer_topic else {}),
                **firm,
            )
            decision = nlu_result.decision
            turn_telemetry.note(tu_label_nlu=decision.label)
            turn_understanding.observe_intent(decision.label, decision.confidence, (nlu_result.trace or {}).get("topic_source"))
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
        if topic_confirm.enabled() and _topic_would_override(decision, topic_match) and not await topic_confirm.confirms(user_text, topic_match):
            # Слова реестра указали на тему, LLM по смыслу не подтвердила (или недоступна) — метка остаётся,
            # тема — подсказка (решение владельца 18.10: тему решает смысл, не совпадение букв).
            decision = _copy_decision(
                decision,
                flags=set(decision.flags) | {"topic_registry_match", build_topic_flag(topic_match.topic_id), "topic_registry_hint_only", "topic_registry_llm_rejected"},
            )
        else:
            decision = _apply_topic_registry_override(decision, topic_match)
        nlu_debug["topic_registry"] = {
            "topic_id": topic_match.topic_id,
            "label": topic_match.label,
            "priority": topic_match.priority,
            "score": topic_match.score,
            "matched_keywords": list(topic_match.matched_keywords),
            "matched_regex": list(topic_match.matched_regex),
        }

    secondary = [str(x) for x in (decision.entities.get("secondary_intents") or [])]
    if (
        decision.label == "APPOINTMENT"
        and "PREPARE" in secondary
        and await booking_slots.prepare_wins_over_booking(user_text)
    ):
        # Правила нашли и запись, и подготовку; главная тема по смыслу — подготовка (BUG-2026-10-18-PREPARE-UZI-AS-BOOKING).
        # Запись остаётся вторым намерением, только если пациент написал о ней словом.
        rest = [x for x in secondary if x != "PREPARE"]
        if detect_appointment_action(user_text) == "book":
            rest.append("APPOINTMENT")
        entities = {k: v for k, v in decision.entities.items() if k not in ("secondary_intents", "appointment_action")}
        if rest:
            entities["secondary_intents"] = rest
        decision = _relabel(
            decision, "prepare_over_booking_llm", label="PREPARE", entities=entities, flags=set(decision.flags) | {"prepare_over_booking_llm"}
        )

    return decision, nlu_debug


async def _complete_route_after_doctor_guard(
    *,
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
    nlu_debug: dict[str, Any] | None = None,
    catalog_prefetch: dict[str, Any] | None = None,
) -> tuple[RouteDecision, Plan, Evidence]:
    """Завершает legacy post-NLU middleware после doctor verification.

    Хелпер принимает уже готовое `decision` после `_verify_doctor_entity` и
    выполняет оставшуюся часть пути после NLU (стадия `tool_loop`): catalog
    injection, guardrails, entity grounding, планирование, executor и графовый
    переход.

    :param decision: routing-решение после doctor verification
    :param user_text: текущий текст пользователя
    :param state: состояние сессии
    :param services: сервисный слой
    :param memory: memory-store
    :param runtime_options: runtime-настройки LLM/NLU
    :param nlu_debug: отладочные метаданные NLU для debug_trace
    :return: финальный routing triple
    """

    # Понимание хода (ход (б), шаг 1): услуги и свойство ценовой реплики — один раз, до
    # подстановки каталога, которая иначе сводит список к одной услуге. В тени только пишется.
    await turn_understanding.observe_price_turn(decision.label, user_text=user_text, pending=memory.get_pending(state))
    decision = await role_pick.apply(decision, user_text, services)  # ход (г): роль решает LLM по каталогу МИС
    decision = await _inject_catalog_candidates(
        decision,
        user_text=user_text,
        state=state,
        services=services,
        prefetched_service=catalog_prefetch,
        runtime_llm_mode=runtime_options.llm_mode if runtime_options is not None else "",
    )
    decision = await _apply_post_nlu_guardrails(decision, state, user_text, services, memory)
    decision = _relabel_test_assist_catalog_hit_as_price(
        decision, user_text=user_text, facet=await _test_assist_facet_by_llm(decision, user_text)
    )
    decision = await _confirm_appointment_change_verb(decision, user_text=user_text, state=state)
    decision = await _confirm_existing_appointment_question(decision, user_text=user_text, state=state)

    existing_appointment = _maybe_offer_operator_for_existing_appointment(
        decision=decision,
        user_text=user_text,
        state=state,
        memory=memory,
    )
    if existing_appointment is not None:
        return existing_appointment

    topic_handoff = _maybe_topic_handoff(decision=decision, state=state, memory=memory)
    if topic_handoff is not None:
        return topic_handoff

    readiness_reply = await _maybe_readiness_reply(decision=decision, user_text=user_text, state=state, memory=memory)
    if readiness_reply is not None:
        return readiness_reply

    overview = await _maybe_service_overview(decision=decision, user_text=user_text, state=state, memory=memory)
    if overview is not None:
        return overview

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
            pending_answer.keeps_flow()
            or (is_appointment_waiting_patient_name(pending_now) and looks_like_patient_fio(user_text))
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
        # OTHER от LLM («ни одна из наших тем»): до честного оффера — данные клиники (класс 2, решение владельца 21.10:
        # «есть ли / делаете ли X» — сначала прайс и памятки, оператор — только если данных нет; g0277).
        from_data = await service_answer.from_data_first(user_text, site="other_offer") if unsupported_kind == "topic_other_offer" else None
        if from_data is not None:
            return decision, Plan(label=decision.label), Evidence(items={ek.SERVICE_ANSWER: from_data.payload()})
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
                _relabel(
                    decision, "catalog_unavailable",
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
                            "handoff": True,
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
        runtime_llm_mode=runtime_options.llm_mode if runtime_options is not None else "",
    )
    if grounding.entities != decision.entities or grounding.flags:
        decision = _copy_decision(
            decision,
            entities=grounding.entities,
            flags=set(decision.flags) | set(grounding.flags),
        )
    decision = await _walkin_sentence_by_llm(decision, user_text)
    decision = await _guard_walkin_named_service(decision, user_text=user_text, state=state, services=services)

    catalog_confirm = _maybe_start_catalog_confirm(
        decision=decision,
        state=state,
        memory=memory,
    )
    if catalog_confirm is not None:
        return catalog_confirm
    booking_choice = _maybe_ask_booking_service_choice(decision=decision, state=state, memory=memory)
    if booking_choice is not None:
        return booking_choice

    # Служебные каталожные ключи не должны попадать в долгоживущий state. Догадка каталога подбора анализов уходит в данные
    # хода — кандидатом ответа по данным (`test_assist_answer`).
    test_assist_guess = str(decision.entities.get(test_assist_answer.CATALOG_GUESS_KEY) or "").strip()
    if any(
        k in decision.entities
        for k in (
            "_catalog_doctor_candidate",
            "_catalog_doctor_query",
            "_catalog_service_candidate",
            "_catalog_service_query",
            test_assist_answer.CATALOG_GUESS_KEY,
        )
    ):
        clean_entities = dict(decision.entities)
        for key in (
            "_catalog_doctor_candidate",
            "_catalog_doctor_query",
            "_catalog_service_candidate",
            "_catalog_service_query",
            test_assist_answer.CATALOG_GUESS_KEY,
        ):
            clean_entities.pop(key, None)
        decision = _copy_decision(decision, entities=clean_entities)

    # merge entities from LLM+rules
    memory.merge_entities(state, decision.entities, label=decision.label)
    # Явный город в текущей реплике должен уметь исправлять/обновлять контекст
    # даже если city уже был заполнен ранее неверно.
    city_hint_now = match_city(user_text)
    # П4: место самарского периметра — это Самара, а не «другой город». При
    # ЯВНОМ упоминании места со своим филиалом («в южном городе», «Мехзавод»)
    # ещё и подставляем этот филиал: иначе пациент получал дамп восемнадцати
    # адресов Самары (прод 10.08). Спутники своих филиалов не имеют — для них
    # только город.
    perimeter_place = (
        match_perimeter_place(user_text)
        if decision.label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}
        else None
    )
    if perimeter_place:
        city_hint_now = _DEFAULT_CITY
        branch_for_place = _offered_branch_for_place(branch_name_for_place(perimeter_place), state.last_entities)
        if branch_for_place and not str(state.last_entities.get("branch_name") or "").strip():
            memory.merge_entities(state, {"branch_name": branch_for_place}, label=decision.label)
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
                quick_now = _suppress_grounder_rejected_slots(quick_now, decision.flags)
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
                quick_flow = _suppress_grounder_rejected_slots(quick_flow, decision.flags)
                quick_flow = _booking_city_from_reply(quick_flow, user_text)
                memory.merge_entities(state, quick_flow, label=decision.label)
        elif decision.label == "TEST_RESULT":
            # Ре-ввод данных результата с изменённым полем (прод #673): required
            # слоты уже заполнены (missing_now пусто), но пациент прислал полный
            # ordered ре-ввод «фамилия, год, филиал, номер» (смена филиала
            # БН→БР) — переизвлекаем и перезаписываем stale. missing=[] →
            # _fill_test_result_entities перезаписывает ТОЛЬКО на полный ordered,
            # одиночное слово не перетирает. Симметрично APPOINTMENT-пути выше.
            quick_re = quick_fill_entities_from_text(user_text, state.last_entities, [], services)
            result_re = {k: quick_re[k] for k in ("surname", "year", "filial", "number") if k in quick_re}
            if result_re:
                memory.merge_entities(state, result_re, label=decision.label)

    # if pending exists, try quick fill missing slots (NO LLM)
    pending = memory.get_pending(state)
    if pending:
        pend_label = pending.get("label")
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        if isinstance(pend_label, str) and isinstance(missing, list) and missing:
            quick = quick_fill_entities_from_text(user_text, state.last_entities, missing, services, pending_label=pend_label)
            if quick:
                quick = await _sanitize_doctor_in_entities(quick, services, label=pend_label)
                quick = _suppress_grounder_rejected_slots(quick, decision.flags)
                memory.merge_entities(state, quick, label=pend_label)
        if pend_label == "APPOINTMENT":
            await _backfill_appointment_doctor_from_text(user_text, state, services, memory)
        slot = pending_answer.SLOT_TOPICS.get(str(pend_label or ""))
        value = pending_answer.slot_value(user_text)
        if slot and value and "pending_slot_value" in decision.flags and missing_slots(str(pend_label), state.last_entities):
            # Одна функция решила, что реплика — ответ на вопрос-слот (`_slot_value_keeps_topic`): правила её не разобрали —
            # реплика целиком в слот; что это за памятка или анализы, решает инструмент темы.
            memory.merge_entities(state, {slot: value}, label=str(pend_label))

    fill_date_from_schedule_windows(state, decision.label)
    _test_assist_guess_as_name(decision, state, memory, test_assist_guess, user_text)

    if turn_understanding.prepare_by_parse():
        # Подготовка читает фразы разбора (этап 2 «без стоп-слов»): разбор хода — и на ходу подготовки, по теме, по которой
        # строится план (ответ на вопрос подготовки — тоже).
        planned = apply_pending_override(decision, memory.get_pending(state), user_text=user_text)
        await turn_understanding.observe_prepare_turn(planned, user_text=user_text)
    plan = build_plan(decision, state, user_text, memory=memory, runtime_options=runtime_options)
    evidence = await execute_plan(plan, state, services)
    if test_assist_guess and decision.label == "TEST_ASSIST":
        evidence.put(ek.TEST_ASSIST_CATALOG_GUESS, test_assist_guess)
    if _doctors_by_service_not_found(evidence.get(ek.DOCTORS_INFO)):
        # Врачей по услуге не нашли («не нашёл в справочнике», L-06(б); «информация о враче не найдена») — до отказа данные
        # клиники: услуга бывает в прайсе и памятке, а врачей по её словам справочник не даёт (g0277 «делают ли ФКС с
        # биопсией»; класс 2, решение владельца 21.10).
        from_data = await service_answer.from_data_first(user_text, site="doctors_by_service")
        if from_data is not None:
            evidence.put(ek.SERVICE_ANSWER, from_data.payload())
    if nlu_debug:
        evidence.debug_trace.append({"nlu": nlu_debug})
    # Graph state transition (FSM слой)
    pending_after_plan = memory.get_pending(state)
    graph_out = _GRAPH_ENGINE.next(
        session=state,
        decision=decision,
        pending=pending_after_plan,
        handoff_planned=(
            bool(evidence.get(ek.HANDOFF_REQUIRED))
            or evidence_requires_handoff(evidence)[0]
        ),
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


def _debug_meta(
    decision: RouteDecision,
    plan: Plan,
    evidence: Evidence,
    state: SessionState,
    pending: Any,
    stage_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    nlu_trace = _extract_nlu_trace(evidence)
    # stage_timings считаются оркестратором на КАЖДЫЙ запрос (orchestrator._timed_stage),
    # но раньше в debug-ответ не прокидывались. Нужны для p50/p95 пер-стейдж
    # (П3 дорожной карты: baseline до/после кванта ollama). Additive, только debug=true.
    timings = {"stage_timings": dict(stage_timings)} if stage_timings else {}
    return {
        **timings,
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
        "understanding": understanding.as_debug() if (understanding := turn_understanding.current()) else None,
        "intent": topic.as_debug() if (topic := turn_understanding.intent()) else None,
        # Разбор реплики на фразы (L-15b) — на ценовых ходах (единственный режим с T7 шага 3б); на прочих ключа нет.
        **({"turn_parse": parse_debug} if (parse_debug := turn_parse.debug()) else {}),
        # Отпечаток рантайма (DATA-13): сборка, модель, хэши промптов и реестра тем, TZ, срезы — без хостов и секретов.
        "runtime": runtime_fingerprint(),
    }


def _own_place_city(city: str | None, user_text: str) -> str | None:
    """«Город» из слова адреса нашего филиала или места периметра («пр-т Николаевский», «в Южном городе») — город клиники
    по умолчанию; иначе — как есть (охрана города в начале хода, П4)."""

    if city and not _is_samara_city(city) and (is_samara_perimeter_text(user_text) or is_branch_address_word(city)):
        return _DEFAULT_CITY
    return city


def _booking_city_from_reply(entities: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Город записи из реплики — тем же правилом, что охрана города в начале хода (`pending_buttons`): «в Южном городе, на
    Николаевском» давало «записи … в городе Николаевский» (BUG-2026-10-20-BRANCH-PERIMETER-UNOFFERED). Под off — как было."""

    city = str(entities.get("city") or "").strip()
    if not city or not pending_answer.buttons_on():
        return entities
    return {**entities, "city": _own_place_city(city, user_text)}


def _offered_branch_for_place(branch: str | None, entities: dict[str, Any]) -> str | None:
    """Филиал места периметра («в Южном городе») — только из предложенных, пока висит вопрос о филиале (`pending_buttons`).

    Было (карточка деплоя 12, `pa_branch_hours_side_unoffered_gyn`): «давайте в Южном городе, на Николаевском» на список
    филиалов гинеколога без Южного города — «Да, можем записать на приём к гинекологу (Южный г. Николаевский пр. 2)»:
    подстановка места периметра шла мимо списка записи (BUG-2026-10-20-BRANCH-PERIMETER-UNOFFERED). Правило только сужает:
    место есть среди пунктов — пункт списка; нет — филиала нет, вопрос звучит снова. Списка нет или под off — как было.
    """

    offered = entities.get("appointment_branch_options")
    if not branch or not pending_answer.buttons_on() or not isinstance(offered, list) or not offered:
        return branch
    _, picked = match_branch_hint(branch, build_branch_index(branch_options_to_indexable(offered)))
    return picked


# Уточняющий вопрос о вопросе «Какой филиал вам удобен?» отвечается обычным путём, если его тема — справка, которую
# запись не меняет (адреса и часы, цена, врачи и расписание); иначе — как было.
_BRANCH_SIDE_LABELS = frozenset({"ADDRESS", "PRICE", "DOCTOR_INFO", "DOCTOR_SCHEDULE"})


def _booking_price_service(entities: dict[str, Any]) -> tuple[str, str]:
    """Услуга записи для вопроса о цене — (запрос цены, специальность для сверки строк или "").

    Запись к специальности — приём этой специальности тем видом, каким путь цены разбирает «приём уролога»: сырое слово
    записи («Урологу») каталог по буквам уводит в чужую услугу (BUG-2026-09-25-SPECIALTY-HITS-PROCEDURE). Запись на
    услугу — сама услуга. Цели нет — пусто.
    """

    service = str(entities.get("service_name") or entities.get("test_name") or "").strip()
    specialty = str(entities.get("specialty") or "").strip()
    if specialty and (not service or is_specialty_word(service)):
        return consult_price_phrase(specialty), specialty
    return service, ""


def _price_rows_of_specialty(evidence: Evidence | None, specialty: str) -> bool:
    """Все строки, которые показал ответ о цене, — приём этой специальности (правило только сужает: иначе — без цены)."""

    names: list[str] = []
    for key in (ek.SERVICE_BUNDLE, ek.PRICE):
        payload = evidence.get(key) if evidence is not None else None
        if not isinstance(payload, dict):
            continue
        names.append(str(payload.get("service_name") or ""))
        for rows_key in ("retail_prices", "prices", "family_variants"):
            names.extend(str(row.get("serviceName") or "") for row in payload.get(rows_key) or () if isinstance(row, dict))
    names = [name for name in names if name.strip()]
    return bool(names) and all(_service_name_allows_specialty(name, specialty) for name in names)


# Сколько строк приёма показывает побочный ответ о цене: у ЛОР-врача на прайсе 13, у гинеколога — 10.
_CONSULT_ROWS_MAX = 15


def _consult_price_rows(rows: list[dict[str, Any]], specialty: str) -> list[dict[str, Any]]:
    """Строки прайса — приём этой специальности (первичный, повторный, у разных врачей), с ценой.

    Кандидаты — тот же поиск пути цены по «приём <специальность>» (`_select_patient_price_rows`); правило только сужает:
    строка — приём или консультация, цена больше нуля, и каждая специальность её названия — из семейства специальности
    записи (уролог — уролог, андролог; «приём хирурга (эндокринологическое отделение)» на запрос гинеколога — чужая).
    """

    from .services._prices_helpers import _select_patient_price_rows

    family = set(_extract_specialties_from_text(specialty)) or {specialty}
    picked: list[dict[str, Any]] = []
    for row in _select_patient_price_rows(rows, consult_price_phrase(specialty), limit=_CONSULT_ROWS_MAX):
        name = str(row.get("serviceName") or "")
        found = set(_extract_specialties_from_text(name))
        cost = row.get("cost")
        if isinstance(cost, (int, float)) and cost > 0 and _is_consultation_service_query(name) and found and found <= family:
            picked.append(row)
    return picked


async def _booking_consult_price_rows(specialty: str) -> list[dict[str, Any]]:
    """Строки приёма специальности записи из прайса региона; сбой загрузки — пусто (ответ — как был)."""

    from .services._prices_helpers import _catalog_rows

    return _consult_price_rows(await asyncio.to_thread(_catalog_rows, None), specialty)


async def _answer_branch_side_question(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None,
    debug: bool,
) -> ResponseEnvelope | None:
    """Ответ на уточняющий вопрос о вопросе «Какой филиал вам удобен?» и тот же вопрос снова (`pending_buttons`).

    Было (стенд, `pa_branch_which_closer`, `pa_branch_new_topic`): одна функция решала «ничего из этого» (правило 4
    промпта — вопрос о вопросе бота), запись шла дальше, и бот печатал тот же список филиалов, не ответив. Стало
    (принцип 5): вопрос отвечается обычным путём — на копии состояния без флагов оформления (специальность, услуга, город
    — контекст), так что собранное записи не меняется ни при каком ответе; затем звучит вопрос о филиале с прежними
    пунктами. Ответ годится, только если его тема — справка (`_BRANCH_SIDE_LABELS`), он не перевод и сам ничего не
    спросил; иначе None — ход идёт, как было.
    """

    from .orchestrator import run_pipeline

    raw = expects.read(state)
    options = tuple((raw or {}).get("options") or ())
    shown = str((raw or {}).get("question") or "").strip()
    question = shown.split("\n")[-1].strip() if shown else ""
    if not options or not question:
        return None
    # Вопрос о цене без своей услуги («а сколько стоит приём?», «а цена?») — о цене услуги записи, не о словах реплики:
    # по слову «приём» путь цены находил приём хирурга (стенд `pa_branch_new_topic`, BUG-2026-10-20-BRANCH-SIDE-PRICE-FOREIGN).
    booking_price, booking_specialty = ("", "") if price_reply_names_own_service(user_text) else _booking_price_service(state.last_entities)
    scratch = copy.deepcopy(state)
    reset_appointment_runtime_state(scratch)
    memory.clear_pending(scratch)
    scratch.last_entities.pop(expects.EXPECTS_KEY, None)
    if booking_price:
        scratch.last_entities["service_name"] = booking_price
    price_of = planner_price_of(booking_price) if booking_price else contextlib.nullcontext()
    with pending_answer.aside(), expects.aside(), price_of:
        try:
            ctx = await run_pipeline(user_text, scratch, services=services, memory=memory, runtime_options=runtime_options)
        except Exception:  # побочный ответ не ломает ход — как было
            logger.warning("branch_side_question_failed", exc_info=True)
            return None
        asked_aside = expects.asked()
    response, decision = ctx.response, ctx.decision
    if response is None or decision is None or response.handoff or asked_aside or not str(response.text or "").strip():
        return None
    if decision.label not in _BRANCH_SIDE_LABELS:
        return None
    if decision.label == "PRICE" and not price_reply_names_own_service(user_text):
        # Своей услуги реплика не назвала: цена — только услуги записи; цели нет или строки не той специальности — без
        # цены (дезинформация ценой хуже уточнения).
        if not booking_price or (booking_specialty and not _price_rows_of_specialty(ctx.evidence, booking_specialty)):
            return None
    from . import renderer
    from .shown_context import SHOWN_KEY, remember

    answer_text = str(response.text).strip()
    if SHOWN_KEY in scratch.last_entities:  # что пациент увидел в побочном ответе — контекст следующего хода
        state.last_entities[SHOWN_KEY] = scratch.last_entities[SHOWN_KEY]
    if decision.label == "PRICE" and booking_specialty:
        # Несколько строк приёма специальности — все со своими ценами, а не первая строка и врачи (решение 25.09 №2,
        # BUG-2026-10-20-BRANCH-SIDE-PRICE-ONE-ROW): у невролога первичный приём стоит 3 000, 6 500 и 20 000 у разных
        # врачей. Одна строка — ответ пути цены, как был.
        rows = await _booking_consult_price_rows(booking_specialty)
        if len(rows) > 1:
            answer_text = renderer.format_consult_price_rows(rows)
            remember(state, [str(row.get("serviceName") or "") for row in rows])
    # Вопрос о филиале — прежний: в поле ожидания его текст со списком записи, не текст побочного ответа (в ответе о
    # часах — свой нумерованный список филиалов города, BUG-2026-10-20-BRANCH-AFTER-SIDE-FOREIGN).
    expects.keep_asking()
    return ResponseEnvelope(
        text=f"{answer_text}\n\n{question}",
        attachments=list(response.attachments or []),
        handoff=False,
        state_update=_early_debug_state_update(
            debug,
            label=decision.label,
            handoff=False,
            flags={"appointment_branch_side_question"},
            context_action="continue",
        ),
    )


async def patient_routing_stream(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    debug: bool = False,
    runtime_options: RuntimeOptions | None = None,
    button_id: str = "",
) -> AsyncGenerator[ResponseEnvelope, None]:
    try:
        services.ensure_background_refresh_started()
    except Exception:
        logger.warning("background_refresh_start_failed", exc_info=True)

    from . import button_menu

    if button_menu.action_for(button_id) is not None:
        # Нажатие кнопки — явная смена темы. Подпись кнопки не может стать
        # ответом на висящий вопрос: дату записи, «да/нет» оффера оператора,
        # подтверждение услуги. Поэтому сбрасываем тему ДО всех ранних проверок,
        # включая предпроверку активной записи.
        clear_on_topic_switch(state, memory)
    # Метка адресной кнопки (`address_branch_flags`): нажатие ставит, следующая реплика старит — после сброса кнопкой.
    address_flow.track_address_button(state, button_id)
    button_flow.track_leaf(state, button_id)  # лист без своего вопроса — рамка следующей реплики (`frame`)
    # Вопрос, на который отвечает реплика, — после сброса кнопкой (поле ожидания, D2.1; пишет его конец хода).
    expects.begin_turn(state)

    # Неотложное состояние — первым: раньше явного запроса оператора, охраны города и
    # предпроверки записи. Срочность проверялась только внутри пайплайна, и «задыхаюсь»
    # посреди подтверждения записи получало «Подтвердите запись…», из другого города —
    # «только по Самаре», а «оператор! хочу покончить с собой» — голое «Соединяю с
    # оператором» (ночью ещё и часы операторов), всё без совета вызвать скорую
    # (BUG-2026-10-05-URGENT-INSIDE-APPOINTMENT). Здесь — только регулярка срочности, не
    # каскад правил; ответ — тот же шаблон, что у метки URGENT, он и переводит на оператора.
    # Summary не пересобираем: clear_on_handoff его обнуляет, как у явного оператора.
    if detect_urgent(user_text):
        from . import renderer

        clear_on_handoff(state, memory)
        urgent = renderer.render_urgent()
        yield ResponseEnvelope(
            text=urgent.text,
            attachments=list(urgent.attachments),
            handoff=urgent.handoff,
            state_update=_early_debug_state_update(
                debug,
                label="URGENT",
                handoff=urgent.handoff,
                flags={"urgent"},
                context_action="new_topic",
                confidence=1.0,
            ),
        )
        return

    # Ответ на висящий вопрос бота — одна функция, одно решение на ход (класс 5, дизайн
    # `docs/DESIGN_2026-10-18_one_pending_answer.md`): после срочного, до явного запроса оператора — «не надо
    # оператора» на вопрос бота — ответ, не просьба о переводе. Решение читают обработчики вопросов
    # (`pending_answer.kind / passes / picked`); хвост реплики («нет, мне ферритин») идёт обычным путём. Кнопка снимает
    # висящий вопрос — функция не зовётся.
    pending_answer.reset()
    button_flow.reset()
    answered = None
    said = user_text  # реплика как есть — в историю диалога; дальше ход может идти хвостом
    if not button_id:
        expected = pending_answer.expected_from(state.last_entities)
        if expected is not None:
            filled = expected[0] == expects.Question.SLOT_VALUE.value and _slot_filled_by_rules(user_text, state, services, memory)
            spent = llm_runtime.turn_decisions_spent()
            # Вопрос кнопки меню под `button_answer_topic` / `button_topic_hint`: значение, найденное правилами, LLM проверяет на
            # «другую тему» — правила находят его и в реплике о другом («ЭКГ как подготовиться?» на «Цену анализа»).
            # Судья под `button_reply_judge` видит и кнопку с её темой: полный вопрос в теме кнопки — ответ, не «другая тема».
            pending = memory.get_pending(state)
            answered = await pending_answer.answer(
                *expected,
                user_text,
                filled=filled,
                confirm_topic=button_flow.judged_by_llm(pending),
                button=button_flow.judge_context(pending, expected[1]),
            )
            button_flow.refund_topic_judge(answered, pending, spent)
            if answered is not None and answered.decided and answered.rest:
                user_text = answered.rest

    # Явный запрос оператора — приоритет над всем, кроме срочного (оно выше) и ответа на вопрос бота.
    if explicit_operator_requested(user_text) and not pending_answer.answers_question(answered):
        clear_on_handoff(state, memory)
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
    # П4: самарский периметр. Тридцать филиалов узла «Самара» стоят в черте
    # города, три — вне её (пос. Придорожный «Южный город», п. Мехзавод, ЮГ-2),
    # плюс спутники по решению владельца №5 (Новокуйбышевск, Кинель, Смышляевка,
    # Стройкерамика). Их обслуживаем как Самару. Без этого гарда адрес
    # собственного филиала выглядел чужим городом: «пр-т Николаевский»
    # `match_city` резолвит в город «Николаевский» из справочника → пациента
    # уводило к оператору «Сейчас могу помочь только по Самаре».
    city_now = _own_place_city(city_now, user_text)
    if city_now and not _is_samara_city(city_now):
        # Исключение: запрос результатов анализов работает для любого города —
        # портал результатов (naykalab.ru/samara) не привязан к региону, достаточно
        # ФИО/года/филиала/номера. Пускаем такие реплики в TEST_RESULT-флоу.
        if detect_test_result_intent(user_text):
            pass
        else:
            clear_on_handoff(state, memory)
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

    # Уточняющий вопрос о вопросе «Какой филиал вам удобен?» (`pending_buttons`): ответ обычным путём и тот же вопрос
    # снова, собранное записи цело.
    if pending_answer.branch_side_question(answered):
        side = await _answer_branch_side_question(user_text, state, services, memory, runtime_options, debug)
        if side is not None:
            expects.finish_turn(state, side)
            memory.append_turn(state, role="user", text=said)
            memory.append_turn(state, role="assistant", text=side.text)
            update_summary(state, reason="normal")
            yield side
            return

    # Уточняющий вопрос о вопросе кнопки результатов (`button_side_answer`, класс К3): где код и номер и как посмотреть
    # результат — вместо повтора инструкции; вопрос о реквизитах висит дальше, счёт переспросов прежний.
    if button_flow.result_side_question(answered, memory.get_pending(state)):
        turn_telemetry.note(button_side_answer="test_result")
        expects.keep_asking()
        side = ResponseEnvelope(
            text=TEST_RESULT_SIDE_TEXT,
            attachments=[],
            handoff=False,
            state_update=_early_debug_state_update(
                debug, label="TEST_RESULT", handoff=False, flags={"button_side_answer"}, context_action="continue"
            ),
        )
        expects.finish_turn(state, side)
        memory.append_turn(state, role="user", text=said)
        memory.append_turn(state, role="assistant", text=side.text)
        update_summary(state, reason="normal")
        yield side
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
            clear_on_handoff(state, memory)
        precheck = _maybe_offer_operator_on_reask_limit(precheck, state, memory)
        expects.finish_turn(state, precheck)
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
            # Только когда кнопка есть: путь без кнопок не меняется.
            **({"button_id": button_id} if button_id else {}),
        )
        response = ctx.response
        decision = ctx.decision
        plan = ctx.plan
        evidence = ctx.evidence
        if response is None:
            raise RuntimeError("orchestrator returned no response")
    except Exception as e:
        logger.exception("patient_routing_stream failed for session_id=%s", state.session_id)
        turn_telemetry.override(answer_source="route_error", error=type(e).__name__)
        fallback_text = handoff_message("service_error")
        state_update: dict[str, Any] = {}
        if debug:
            state_update = {"debug": {"route_error": str(e)}}
        clear_on_handoff(state, memory)
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
            state_update={"debug": _debug_meta(decision, plan, evidence, state, pending, stage_timings=ctx.stage_timings)},
        )

    _note_decision(decision, evidence, ctx.stage_timings)
    response = _offer_operator_on_deadline(response, state, memory)
    # Предел переспросов (D2.4, под `reask_limit = on`) — раньше анти-залипа: тот не трогает ответ с оффером.
    response = _maybe_offer_operator_on_reask_limit(response, state, memory)
    offered = _maybe_offer_operator_on_repeat(response, state, memory, user_text, pressed=button_menu.action_for(button_id) is not None)
    if offered is not response:
        turn_telemetry.override(answer_source="operator_offer_on_repeat")
    response = offered
    # Смешанное сообщение внутри кнопки (`button_topic_hint`): ответ по теме кнопки — и подсказка кнопки на остальное.
    response = button_flow.with_mixed_hint(response)
    address_flow.finish_turn(state)  # метка адресной кнопки — следующей реплике (`address_branch_flags`)
    button_flow.finish_leaf(state)  # метка листа без своего вопроса — следующей реплике (`frame`)
    # Поле ожидания — по тексту, который увидит пациент (после анти-залипа: он дописывает свой оффер).
    expects.finish_turn(state, response)

    memory.append_turn(state, role="user", text=said)
    memory.append_turn(state, role="assistant", text=response.text)
    update_summary(state, reason="normal")

    if response.handoff:
        clear_on_handoff(state, memory)

    yield response
    return
