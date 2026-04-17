"""Flow-политики для мессенджерного роутера.

Содержит stateful-хелперы и quick-fill логику, вынесенные из router.py,
чтобы роутер оставался оркестратором pipeline.
Ответственность модуля: обработка flow-специфичных переходов и быстрых заполнений слотов.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .mess_types import RouteDecision, SessionState
from .city import match_city
from .policies import (
    branch_options_to_indexable,
    build_branch_index,
    detect_appointment_action,
    detect_address_intent,
    detect_doc_request_intent,
    detect_prepare_intent,
    detect_price_intent,
    detect_schedule_intent,
    extract_specialty,
    has_datetime_signal,
    extract_branch_hint,
    looks_like_branch_hint,
    match_branch_hint,
    normalize_appointment_action,
    quick_fill_core_entities,
    service_name_conflicts_with_doctor,
)
from .russian_nlu import normalize_ru
from .services import Services, resolve_price_service_name_from_catalog

log = logging.getLogger(__name__)


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


_PATIENT_FIO_RE = re.compile(r"^[А-ЯЁа-яё\-]{2,}(?:\s+[А-ЯЁа-яё\-]{2,}){1,2}$")
_PATIENT_FIO_STOPWORDS = {
    "анализ",
    "анализы",
    "анализов",
    "результат",
    "результаты",
    "тест",
    "тесты",
    "тестов",
    "врач",
    "врача",
    "доктор",
    "расписание",
    "запись",
    "прием",
    "приём",
    "окна",
    "слоты",
    "оператор",
    "город",
    "филиал",
    "адрес",
    "цена",
    "стоимость",
    "услуга",
    "услуги",
    "мне",
    "нужно",
    "надо",
    "хочу",
    "когда",
    "где",
    "какой",
    "какие",
    "покажи",
    "покажите",
    "да",
    "нет",
    "подождите",
    "подожди",
    "пока",
    "ладно",
    "извините",
    "не",
    "то",
    "это",
    "другое",
    "другой",
    "хотел",
    "хотела",
    "буду",
    "на",
    "в",
    "во",
    "к",
    "с",
    "со",
    "до",
    "после",
    "сегодня",
    "завтра",
    "послезавтра",
    "утром",
    "днем",
    "днём",
    "вечером",
}


def _looks_like_patient_fio(text: str) -> bool:
    s = str(text or "").strip()
    if not _PATIENT_FIO_RE.fullmatch(s):
        return False
    if has_datetime_signal(s):
        return False
    tokens = [t for t in s.split() if t]
    if len(tokens) < 2:
        return False
    normalized = [normalize_ru(t) for t in tokens]
    if any(t in _PATIENT_FIO_STOPWORDS for t in normalized):
        return False
    return True


_CITY_REPLY_BLOCK_RE = re.compile(
    r"\b(адрес\w*|филиал\w*|цена|стоим\w*|сколько|запис\w*|расписани\w*|врач\w*|доктор\w*|анализ\w*|результат\w*)\b",
    re.I,
)


def _is_city_only_reply(text: str) -> bool:
    s = str(text or "").strip()
    if not s or len(s) > 48:
        return False
    if _CITY_REPLY_BLOCK_RE.search(s):
        return False
    city = match_city(s)
    if not city:
        return False

    s_norm = re.sub(r"[^a-zа-яё0-9]+", " ", s.lower()).strip()
    s_norm = re.sub(r"^(?:г|город|в)\s+", "", s_norm).strip()
    city_norm = re.sub(r"[^a-zа-яё0-9]+", " ", city.lower()).strip()
    if not s_norm or not city_norm:
        return False
    if s_norm == city_norm:
        return True

    city_stem = city_norm.rstrip("аеиоуыяьюй")
    if len(city_stem) < 3:
        city_stem = city_norm
    # "самара", "самаре", "г самара", "в самаре" -> true
    return city_stem in s_norm and len(s_norm.split()) <= 2


def _is_samara_city_value(city: str | None) -> bool:
    if not city:
        return False
    return normalize_ru(city) == "самара"


def _is_appointment_branch_reply(text: str) -> bool:
    s = str(text or "").strip()
    if not s or len(s) > 160:
        return False
    if has_datetime_signal(s):
        return False
    if detect_prepare_intent(s) or detect_price_intent(s) or detect_doc_request_intent(s):
        return False

    city = match_city(s)
    if city and not _is_samara_city_value(city):
        return False

    hint = extract_branch_hint(s, {"city": city or "Самара"})
    if hint:
        hint_norm = re.sub(r"\s+", " ", str(hint).strip())
        if hint_norm and not _is_city_only_reply(hint_norm):
            return True

    if looks_like_branch_hint(s):
        return True

    low = re.sub(r"[\"'`]", "", normalize_ru(s))
    low = re.sub(r"\s+", " ", low).strip()
    # "на победе", "победы 83" без явного префикса "ул."
    return bool(re.fullmatch(r"(?:на\s+)?[а-я\-]{4,40}(?:\s+\d{1,4}[a-zа-я]?)?", low))


def _is_appointment_waiting_branch_or_city(pending: dict | None) -> bool:
    if not isinstance(pending, dict):
        return False
    if pending.get("label") != "APPOINTMENT":
        return False
    missing = pending.get("missing")
    if not isinstance(missing, list):
        return False
    as_text = " ".join(str(x or "").lower() for x in missing)
    return ("branch" in as_text) or ("city" in as_text)


def _is_appointment_waiting_doctor_or_service(pending: dict | None) -> bool:
    if not isinstance(pending, dict):
        return False
    if pending.get("label") != "APPOINTMENT":
        return False
    missing = pending.get("missing")
    if not isinstance(missing, list):
        return False
    as_text = " ".join(str(x or "").lower() for x in missing)
    return (
        "doctor_id" in as_text
        or "doctor_name" in as_text
        or "specialty" in as_text
        or "service_name" in as_text
    )


def _looks_like_appointment_doctor_reply(text: str) -> bool:
    s = str(text or "").strip()
    if not s or len(s) > 64:
        return False
    if _looks_like_patient_fio(s):
        return False
    if _is_city_only_reply(s):
        return False
    # Явные признаки адреса/филиала не считаем ответом с фамилией врача.
    if re.search(r"\b(ул\.?|улиц\w*|пр\.?|просп\w*|дом|д\.)\b", s, re.I):
        return False
    if re.search(r"^\s*на\s+[А-Яа-яЁёA-Za-z\-]{3,}", s):
        return False
    if has_datetime_signal(s):
        return False
    if (
        detect_schedule_intent(s)
        or detect_prepare_intent(s)
        or detect_price_intent(s)
        or detect_address_intent(s)
        or detect_doc_request_intent(s)
    ):
        return False
    tokens = [t for t in re.findall(r"[A-Za-zА-Яа-яЁё\-]+", s) if t]
    return 1 <= len(tokens) <= 3


def _appointment_pending_missing_slots(pending: dict | None) -> list[str]:
    """
    Возвращает список незаполненных слотов активного APPOINTMENT pending.

    :param pending: текущий pending-объект из memory
    :return: нормализованный список missing-slots
    """

    if not isinstance(pending, dict) or pending.get("label") != "APPOINTMENT":
        return []
    raw_missing = pending.get("missing")
    if not isinstance(raw_missing, list):
        return []
    return [str(item).strip() for item in raw_missing if str(item).strip()]


async def _resolve_appointment_doctor_name(text: str, services: Services) -> str | None:
    """
    Пытается безопасно распознать врача из короткой реплики внутри записи.

    :param text: текущая реплика пользователя
    :param services: сервисный слой с doctor catalog
    :return: каноническое ФИО врача либо None
    """

    raw = str(text or "").strip()
    if not raw:
        return None

    probes: list[str] = [raw]
    if raw.lower().startswith("к "):
        tail = raw[2:].strip()
        if tail:
            probes.append(tail)

    for probe in probes:
        resolved = await services.resolve_doctor_name(probe)
        if resolved:
            return resolved
    return None


async def prelock_active_appointment_turn(
    text: str,
    state_entities: dict[str, Any],
    pending: dict | None,
    services: Services,
) -> dict[str, Any] | None:
    """
    Детерминированно распознает слот-ответ внутри активного APPOINTMENT flow.

    Это pre-routing слой: если пациент прислал короткий ответ вроде даты,
    филиала, ФИО пациента или фамилии врача, мы удерживаем APPOINTMENT до
    общего NLU и извлекаем только нужные slot-updates.

    :param text: текущая реплика пользователя
    :param state_entities: текущее состояние сессии
    :param pending: pending-объект из memory
    :param services: сервисный слой
    :return: словарь slot-updates или None, если prelock не нужен
    """

    appointment_flow_active = bool(state_entities.get("appointment_flow_active"))
    appointment_pending = isinstance(pending, dict) and pending.get("label") == "APPOINTMENT"
    schedule_context = (
        str(state_entities.get("_last_label") or "") == "DOCTOR_SCHEDULE"
        and bool(state_entities.get("doctor_name") or state_entities.get("doctor_id"))
        and has_datetime_signal(text)
        and not (
            detect_prepare_intent(text)
            or detect_price_intent(text)
            or detect_address_intent(text)
            or detect_doc_request_intent(text)
            or detect_schedule_intent(text)
        )
    )
    if not appointment_flow_active and not appointment_pending and not schedule_context:
        return None

    if (
        state_entities.get("appointment_confirm_pending")
        or state_entities.get("appointment_cancel_pending")
        or state_entities.get("appointment_topic_switch_pending")
        or state_entities.get("_operator_offer_pending")
    ):
        return None

    missing_rules = _appointment_pending_missing_slots(pending)
    if not missing_rules and appointment_flow_active:
        missing_rules = [
            "_any_of:doctor_id,doctor_name,specialty,service_name",
            "_any_of:city,branch_name,branch_id",
            "date_from",
            "time_from",
            "patient_name",
        ]
    if not missing_rules and schedule_context:
        missing_rules = [
            "_any_of:city,branch_name,branch_id",
            "date_from",
            "time_from",
            "patient_name",
        ]

    if not missing_rules:
        return None

    slot_updates = quick_fill_entities_from_text(text, state_entities, missing_rules, services)

    if (
        _is_appointment_waiting_doctor_or_service(pending)
        and not slot_updates.get("doctor_name")
        and not slot_updates.get("doctor_id")
        and not state_entities.get("doctor_name")
        and not state_entities.get("doctor_id")
        and _looks_like_appointment_doctor_reply(text)
    ):
        resolved_doctor = await _resolve_appointment_doctor_name(text, services)
        if resolved_doctor:
            slot_updates["doctor_name"] = resolved_doctor
            current_service = str(state_entities.get("service_name") or "").strip()
            if current_service and service_name_conflicts_with_doctor(current_service, resolved_doctor):
                slot_updates["__clear_service_name"] = True

    if slot_updates:
        if schedule_context and not _looks_like_patient_fio(text):
            slot_updates["__clear_patient_name"] = True
        return slot_updates

    if _is_appointment_waiting_patient_name(pending) and _looks_like_patient_fio(text):
        return {}
    if _is_appointment_waiting_branch_or_city(pending) and (_is_appointment_branch_reply(text) or _is_city_only_reply(text)):
        return {}
    if _is_appointment_waiting_doctor_or_service(pending) and _looks_like_appointment_doctor_reply(text):
        return {}
    if has_datetime_signal(text) and not (
        detect_prepare_intent(text)
        or detect_price_intent(text)
        or detect_address_intent(text)
        or detect_doc_request_intent(text)
        or detect_schedule_intent(text)
    ):
        if schedule_context and not _looks_like_patient_fio(text):
            return {"__clear_patient_name": True}
        return {}
    return None


def _is_short_prepare_followup(text: str) -> bool:
    s = str(text or "").strip()
    if not s or len(s) > 64:
        return False
    tokens = [t for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]+", s) if t]
    if not tokens or len(tokens) > 5:
        return False
    if detect_price_intent(s) or detect_address_intent(s) or detect_doc_request_intent(s):
        return False
    return True


def _looks_like_price_service_reply(text: str) -> bool:
    """
    Проверяет, что короткая реплика похожа на ответ с названием услуги для PRICE-flow.

    :param text: текущая реплика пользователя
    :return: True, если это похоже на название услуги/анализа, а не новый вопрос
    """

    s = str(text or "").strip()
    if not s or len(s) > 96:
        return False
    if has_datetime_signal(s) or _is_city_only_reply(s):
        return False
    if detect_prepare_intent(s) or detect_address_intent(s) or detect_doc_request_intent(s):
        return False
    if detect_schedule_intent(s):
        return False
    tokens = [t for t in re.findall(r"[A-Za-zА-Яа-яЁё0-9\-]+", s) if t]
    return 1 <= len(tokens) <= 6


def _apply_pending_override(decision: RouteDecision, pending: dict | None, user_text: str = "") -> str:
    if not pending:
        return decision.label
    pending_label = pending.get("label")
    if not isinstance(pending_label, str):
        return decision.label
    # Для добора города не разрешаем случайному ADDRESS-решению
    # (обычно на короткий ответ "Самара") ломать исходный flow.
    if (
        pending_label in {"PRICE", "TEST_ASSIST", "APPOINTMENT"}
        and decision.label == "ADDRESS"
        and _is_city_only_reply(user_text)
    ):
        return pending_label
    if (
        pending_label == "PRICE"
        and decision.label in {"TEST_ASSIST", "OTHER", "APPOINTMENT", "DOCTOR_INFO", "ADDRESS"}
        and any("service_name" in str(item or "") for item in (pending.get("missing") or []))
        and _looks_like_price_service_reply(user_text)
    ):
        return "PRICE"
    if (
        pending_label == "APPOINTMENT"
        and decision.label == "ADDRESS"
        and _is_appointment_waiting_branch_or_city(pending)
        and _is_appointment_branch_reply(user_text)
    ):
        return "APPOINTMENT"
    if (
        pending_label == "APPOINTMENT"
        and _is_appointment_waiting_branch_or_city(pending)
    ):
        quick = quick_fill_core_entities(
            user_text,
            {},
            ["_any_of:city,branch_name,branch_id"],
        )
        if quick.get("appointment_selection_mode") in {"doctor", "branch"}:
            return "APPOINTMENT"
    if (
        pending_label == "APPOINTMENT"
        and _is_appointment_waiting_doctor_or_service(pending)
        and decision.label in {"DOCTOR_SCHEDULE", "DOCTOR_INFO", "OTHER", "TEST_RESULT"}
        and _looks_like_appointment_doctor_reply(user_text)
    ):
        return "APPOINTMENT"
    # В шаге добора ФИО пациента не даем случайной переклассификации
    # (например, в TEST_RESULT) перебить активный APPOINTMENT flow.
    if (
        _is_appointment_waiting_patient_name(pending)
        and _looks_like_patient_fio(user_text)
        and decision.context_action != "cancel_flow"
        and decision.label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}
    ):
        return "APPOINTMENT"
    if pending_label == "PREPARE":
        # Короткий ответ на уточнение подготовки ("вульвоскопия") не должен
        # сбрасываться в PRICE/ADDRESS из-за одиночной переклассификации.
        if detect_prepare_intent(user_text) or _is_short_prepare_followup(user_text):
            return "PREPARE"
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


def _normalize_doctor_key(value: Any) -> str:
    s = normalize_ru(value)
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
        "doc_request_kind",
        "branch_id",
        "branch_name",
        "secondary_intents",
        "_secondary_queue",
        "_secondary_offer_pending",
        "_catalog_confirm_pending",
        "_catalog_confirm_rejects",
    ):
        state.last_entities.pop(k, None)


def _apply_context_action(decision: RouteDecision, state: SessionState, user_text: str) -> RouteDecision:
    action = decision.context_action
    if action == "cancel_flow":
        _clear_flow_state(state)
        return decision

    if action == "new_topic":
        # Если в APPOINTMENT pending явно ждем ФИО пациента и пользователь
        # прислал ФИО, не даем случайному new_topic сбросить сценарий записи.
        pending_now = state.last_entities.get("_pending")
        if _is_appointment_waiting_patient_name(pending_now if isinstance(pending_now, dict) else None) and _looks_like_patient_fio(user_text):
            return RouteDecision(
                label=decision.label,
                confidence=decision.confidence,
                entities=dict(decision.entities),
                flags=set(decision.flags) | {"context_action_new_topic_blocked_patient_name"},
                needs_handoff=decision.needs_handoff,
                context_action="continue",
                source=decision.source,
                clarify_needed=decision.clarify_needed,
                clarify_reason=decision.clarify_reason,
                clarify_slots=list(decision.clarify_slots),
                intent_candidates=list(decision.intent_candidates),
            )
        _clear_topic_state(state)
        return decision

    if (
        action == "overwrite_doctor"
        and state.last_entities.get("appointment_flow_active")
        and has_datetime_signal(user_text or "")
        and not detect_schedule_intent(user_text or "")
    ):
        # Во время активной записи реплика с датой/временем почти всегда
        # является продолжением APPOINTMENT, а не сменой врача.
        return RouteDecision(
            label=decision.label,
            confidence=decision.confidence,
            entities=dict(decision.entities),
            flags=set(decision.flags) | {"context_action_overwrite_doctor_blocked"},
            needs_handoff=decision.needs_handoff,
            context_action="continue",
            source=decision.source,
            clarify_needed=decision.clarify_needed,
            clarify_reason=decision.clarify_reason,
            clarify_slots=list(decision.clarify_slots),
            intent_candidates=list(decision.intent_candidates),
        )

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
            source=decision.source,
            clarify_needed=decision.clarify_needed,
            clarify_reason=decision.clarify_reason,
            clarify_slots=list(decision.clarify_slots),
            intent_candidates=list(decision.intent_candidates),
        )

    return RouteDecision(
        label=decision.label,
        confidence=decision.confidence,
        entities=entities,
        flags=sanitized_flags | {"context_action_overwrite_doctor"},
        needs_handoff=decision.needs_handoff,
        context_action=action,
        source=decision.source,
        clarify_needed=decision.clarify_needed,
        clarify_reason=decision.clarify_reason,
        clarify_slots=list(decision.clarify_slots),
        intent_candidates=list(decision.intent_candidates),
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
        log.warning("branches_fetch_failed", exc_info=True)
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
        branches_from_windows = sorted({str(w.get("branch") or "").strip() for w in uniq if str(w.get("branch") or "").strip()})
        if branches_from_windows:
            state.last_entities["appointment_branch_options"] = branches_from_windows[:10]
        if len(branches_from_windows) == 1:
            state.last_entities["branch_name"] = branches_from_windows[0]
        elif branches_from_windows:
            current_branch = str(state.last_entities.get("branch_name") or "").strip()
            if current_branch and current_branch not in branches_from_windows:
                state.last_entities.pop("branch_name", None)
    elif regions:
        state.last_entities["appointment_branch_options"] = regions[:10]
        current_branch = str(state.last_entities.get("branch_name") or "").strip()
        if len(regions) == 1:
            state.last_entities["branch_name"] = regions[0]
        elif current_branch and current_branch not in regions:
            state.last_entities.pop("branch_name", None)


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

    if (
        any("service_name" in str(rule or "") for rule in missing_rules)
        and (
            detect_price_intent(t)
            or str(state_entities.get("_last_label") or "") == "PRICE"
        )
    ):
        current_service_name = str(state_entities.get("service_name") or state_entities.get("test_name") or "")
        resolved_service_name = resolve_price_service_name_from_catalog(
            t,
            current_service_name=current_service_name,
        )
        if resolved_service_name:
            out["service_name"] = resolved_service_name
        else:
            specialty = extract_specialty(t.lower())
            if specialty:
                existing = _normalize_doctor_key(out.get("service_name"))
                spec_norm = _normalize_doctor_key(specialty)
                if not existing or existing == spec_norm:
                    out["service_name"] = f"прием {specialty}"

    if "appointment_action" in missing_rules and not out.get("appointment_action"):
        appt_action = normalize_appointment_action(detect_appointment_action(t), t)
        if appt_action:
            out["appointment_action"] = appt_action

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


# Public API for other modules. Wrappers preserve current behavior
# while hiding implementation-specific `_...` names.
def apply_context_action(decision: RouteDecision, state: SessionState, user_text: str) -> RouteDecision:
    return _apply_context_action(decision, state, user_text)


def apply_pending_override(decision: RouteDecision, pending: dict | None, user_text: str = "") -> str:
    return _apply_pending_override(decision, pending, user_text=user_text)


def fill_date_from_schedule_windows(state: SessionState, label: str) -> None:
    _fill_date_from_schedule_windows(state, label)


def get_secondary_queue(state: SessionState) -> list[str]:
    return _get_secondary_queue(state)


def hydrate_appointment_context_from_schedule(state: SessionState, schedule_payload: dict[str, Any]) -> None:
    _hydrate_appointment_context_from_schedule(state, schedule_payload)


def is_appointment_waiting_patient_name(pending: dict | None) -> bool:
    return _is_appointment_waiting_patient_name(pending)


def is_city_only_reply(text: str) -> bool:
    return _is_city_only_reply(text)


def is_short_prepare_followup(text: str) -> bool:
    return _is_short_prepare_followup(text)


def looks_like_patient_fio(text: str) -> bool:
    return _looks_like_patient_fio(text)


def normalize_secondary_labels(value: Any) -> list[str]:
    return _normalize_secondary_labels(value)


def safe_get_branches(services: Services) -> list[dict[str, str]]:
    return _safe_get_branches(services)


def secondary_followup_text(labels: list[str]) -> str | None:
    return _secondary_followup_text(labels)


def reset_appointment_runtime_state(state: SessionState) -> None:
    """
    Сбрасывает только runtime-состояние активного процесса записи.

    Используется, когда сценарий записи нужно завершить без полного стирания
    темы разговора: например, после ответа о том, что у врача нет свободных
    слотов, либо при явном переключении на другой поток.

    :param state: состояние текущей сессии
    :return: None
    """

    _clear_flow_state(state)


def set_secondary_queue(state: SessionState, labels: list[str]) -> None:
    _set_secondary_queue(state, labels)
