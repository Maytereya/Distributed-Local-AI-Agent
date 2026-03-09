"""Flow-политики для мессенджерного роутера.

Содержит stateful-хелперы и quick-fill логику, вынесенные из router.py,
чтобы роутер оставался оркестратором pipeline.
Ответственность модуля: обработка flow-специфичных переходов и быстрых заполнений слотов.
"""

from __future__ import annotations

import re
from typing import Any

from .mess_types import RouteDecision, SessionState
from .city import match_city
from .policies import (
    branch_options_to_indexable,
    build_branch_index,
    detect_schedule_intent,
    extract_branch_hint,
    looks_like_branch_hint,
    match_branch_hint,
    quick_fill_core_entities,
)
from .services import Services


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
}


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
    # В шаге добора ФИО пациента не даем случайной переклассификации
    # (например, в TEST_RESULT) перебить активный APPOINTMENT flow.
    if (
        _is_appointment_waiting_patient_name(pending)
        and _looks_like_patient_fio(user_text)
        and decision.context_action != "cancel_flow"
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
