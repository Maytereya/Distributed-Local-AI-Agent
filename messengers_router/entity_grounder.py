"""Entity grounding для NLU-решений роутера.

Задача модуля:
1) Ограничить, какие entities вообще можно принять на текущем шаге.
2) Подтверждать критичные сущности по данным/эвристикам (doctor/city/service).
3) Возвращать только "безопасные" entities для merge в session state.
Ответственность модуля: не допускать попадание в state неподтвержденных сущностей.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from .city import looks_like_address, match_city
from .flow_policy import looks_like_patient_fio
from .mess_types import RouteDecision, SessionState
from .policies import extract_service_phrase, has_datetime_signal, service_name_conflicts_with_doctor
from .services import Services, resolve_price_service_name_from_catalog

_CONTROL_KEYS = {
    "secondary_intents",
    "secondary_intent_from_queue",
    "appointment_action",
    "result_action",
    "query_terms",
    "_catalog_doctor_candidate",
    "_catalog_doctor_query",
    "_catalog_service_candidate",
    "_catalog_service_query",
}

_GENERIC_SERVICE_FALLBACK_RE = re.compile(
    r"\b(запис\w*|врач\w*|доктор\w*|специалист\w*|услуг\w*|хочу|нужно|надо|можно)\b",
    re.I,
)
_SERVICE_ANCHOR_HINT_RE = re.compile(
    r"\b(узи|экг|холтер|мрт|кт|фгдс|фкс|эндоскоп|гастроскоп|колоноскоп|анализ|биопс|рентген|флюорограф)\b",
    re.I,
)
_DOCTOR_LIKE_REQUEST_RE = re.compile(r"\bк\s+[А-Яа-яЁёA-Za-z\-]{3,}\b")
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

_LABEL_ENTITY_WHITELIST: dict[str, set[str]] = {
    "APPOINTMENT": {
        "doctor_id",
        "doctor_name",
        "specialty",
        "service_name",
        "city",
        "branch_id",
        "branch_name",
        "patient_name",
        "date_from",
        "date_to",
        "time_from",
        "time_to",
        "date_hint",
    },
    "DOCTOR_SCHEDULE": {"doctor_id", "doctor_name", "specialty", "branch_name", "city"},
    "DOCTOR_INFO": {"doctor_id", "doctor_name", "specialty"},
    "PRICE": {"service_name", "city", "branch_id", "branch_name", "doctor_name", "doctor_id"},
    "ADDRESS": {"city", "branch_id", "branch_name", "service_name"},
    "TEST_RESULT": {"surname", "year", "filial", "number", "lang", "result_action", "order_id"},
    "TEST_ASSIST": {"test_name", "test_goal", "service_name", "city", "branch_name", "branch_id"},
    "PREPARE": {"test_name", "service_name"},
    "NEWS": set(),
    "OTHER": set(),
}


def _keys_from_missing(missing: list[Any]) -> set[str]:
    out: set[str] = set()
    for item in missing:
        if not isinstance(item, str):
            continue
        if item.startswith("_any_of:"):
            tail = item.split(":", 1)[1]
            for key in tail.split(","):
                k = key.strip()
                if k:
                    out.add(k)
            continue
        out.add(item)
    return out


def _pending_missing(pending: dict[str, Any] | None) -> list[str]:
    if not isinstance(pending, dict):
        return []
    raw = pending.get("missing")
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, str)]


def _is_allowed_key(key: str, label: str, pending: dict[str, Any] | None) -> bool:
    if key == "patient_name":
        missing = _pending_missing(pending)
        need = _keys_from_missing(missing)
        if (
            isinstance(pending, dict)
            and pending.get("label") == "APPOINTMENT"
            and "patient_name" in need
        ):
            return True

    if key in _CONTROL_KEYS:
        return True
    base = _LABEL_ENTITY_WHITELIST.get(label, set())
    if key not in base:
        return False
    missing = _pending_missing(pending)
    if not missing:
        return True
    need = _keys_from_missing(missing)
    if not need:
        return True
    # В pending-режиме принимаем в первую очередь слоты, которые реально ждем.
    if key in need:
        return True
    # Разрешаем city даже если формально не ждём, чтобы пользователь мог исправить город.
    if key == "city":
        return True
    if label == "PRICE" and key in {"doctor_name", "doctor_id"}:
        return True
    return False


@dataclass
class GroundingResult:
    entities: dict[str, Any] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)


def _sanitize_raw_doctor_name(raw: str, flags: set[str], entities: dict[str, Any]) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    norm = " ".join(value.lower().replace("ё", "е").split())
    tokens = [t for t in norm.split(" ") if t]
    if len(tokens) == 1 and tokens[0] in _DOCTOR_NOISE_TOKENS:
        entities.pop("doctor_name", None)
        flags.add("doctor_name_unverified")
        return ""
    return value


def _is_doctor_like_service_collision(
    *,
    label: str,
    user_text: str,
    phrase: str,
    decision_flags: set[str],
) -> bool:
    """
    Отсекает ложный service_name в APPOINTMENT, когда пользователь просит
    запись "к <фамилии>", а врача не удалось подтвердить.
    """
    if label != "APPOINTMENT":
        return False
    has_doctor_like_request = bool(_DOCTOR_LIKE_REQUEST_RE.search(user_text or ""))
    if "doctor_name_unverified" not in decision_flags and not has_doctor_like_request:
        return False
    if _SERVICE_ANCHOR_HINT_RE.search(user_text or ""):
        return False
    if _SERVICE_ANCHOR_HINT_RE.search(phrase or ""):
        return False
    # Для многословных и явно процедурных формулировок риск ниже.
    if len((phrase or "").split()) > 1:
        return False
    return True


def _should_drop_service_name_in_active_reschedule(
    *,
    label: str,
    user_text: str,
    raw_entities: dict[str, Any],
    state: SessionState,
) -> bool:
    """
    В активном cancel/reschedule сценарии не даем случайному service grounding
    перетирать doctor-context на репликах-слотах (адрес/дата/ФИО пациента).
    """

    if label != "APPOINTMENT":
        return False

    action = str(
        raw_entities.get("appointment_action")
        or state.last_entities.get("appointment_action")
        or ""
    ).strip().lower()
    if action not in {"cancel", "reschedule"}:
        return False

    has_doctor_context = bool(
        raw_entities.get("doctor_name")
        or raw_entities.get("doctor_id")
        or state.last_entities.get("doctor_name")
        or state.last_entities.get("doctor_id")
    )
    if not has_doctor_context:
        return False

    text = str(user_text or "").strip()
    if not text:
        return False
    # Если пользователь явно говорит про услугу/процедуру — разрешаем update.
    if _SERVICE_ANCHOR_HINT_RE.search(text):
        return False

    return (
        looks_like_patient_fio(text)
        or has_datetime_signal(text)
        or bool(match_city(text))
        or looks_like_address(text)
    )


async def verify_doctor_entities_in_decision(
    decision: RouteDecision,
    user_text: str,
    services: Services,
) -> RouteDecision:
    """
    Единая точка doctor-name валидации для decision.

    Поведение синхронизировано с историческим router pre-ground шагом:
    - doctor_name подтверждается только через `resolve_doctor_name`;
    - неподтвержденный doctor_name удаляется;
    - для APPOINTMENT возможна конверсия неподтвержденного doctor_name -> patient_name;
    - при `overwrite_doctor` и отсутствии подтверждения снимается переключение контекста.
    """
    entities = dict(decision.entities)
    flags = set(decision.flags)
    needs_doctor_verification = bool(entities.get("doctor_name")) or decision.label in {
        "DOCTOR_SCHEDULE",
        "DOCTOR_INFO",
        "APPOINTMENT",
    } or (decision.context_action == "overwrite_doctor")
    if not needs_doctor_verification:
        return decision

    raw = _sanitize_raw_doctor_name(str(entities.get("doctor_name") or ""), flags, entities)
    resolved: str | None = None
    if raw:
        resolved = await services.resolve_doctor_name(raw)
    if not resolved and decision.context_action == "overwrite_doctor":
        resolved = await services.resolve_doctor_name(user_text)

    context_action = decision.context_action
    if resolved:
        entities["doctor_name"] = resolved
        flags.add("doctor_name_verified")
        service_name = str(entities.get("service_name") or "").strip()
        if service_name and service_name_conflicts_with_doctor(service_name, resolved):
            entities.pop("service_name", None)
            flags.add("entity_dropped_doctor_like_service_name")
    elif raw:
        entities.pop("doctor_name", None)
        flags.add("doctor_name_unverified")
        if decision.label == "APPOINTMENT" and looks_like_patient_fio(raw):
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


async def sanitize_doctor_entities(
    entities: dict[str, Any],
    services: Services,
    *,
    label: str,
) -> dict[str, Any]:
    """
    Санитизация doctor_name в произвольном entities-словаре перед merge в state.

    Используется для quick-fill/pending merge шагов.
    """
    out = dict(entities or {})
    raw = str(out.get("doctor_name") or "").strip()
    if not raw:
        return out

    resolved = await services.resolve_doctor_name(raw)
    if resolved:
        out["doctor_name"] = resolved
        service_name = str(out.get("service_name") or "").strip()
        if service_name and service_name_conflicts_with_doctor(service_name, resolved):
            out.pop("service_name", None)
        return out

    out.pop("doctor_name", None)
    if label == "APPOINTMENT" and looks_like_patient_fio(raw) and not out.get("patient_name"):
        out["patient_name"] = raw
    return out


async def ground_decision_entities(
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    services: Services,
    pending: dict[str, Any] | None = None,
) -> GroundingResult:
    raw = dict(decision.entities or {})
    if not raw:
        return GroundingResult()

    label = str(decision.label or "OTHER")
    out: dict[str, Any] = {}
    flags: set[str] = set()

    for key, value in raw.items():
        if not _is_allowed_key(str(key), label, pending):
            flags.add(f"entity_dropped_not_allowed:{key}")
            continue

        if key == "doctor_name":
            name = str(value or "").strip()
            if not name:
                continue
            resolved = await services.resolve_doctor_name(name)
            if resolved:
                out[key] = resolved
                if resolved != name:
                    flags.add("entity_grounded_doctor_name")
            else:
                flags.add("entity_dropped_unverified_doctor_name")
            continue

        if key == "city":
            city = match_city(str(value or "")) or match_city(user_text)
            if city:
                out[key] = city
                if str(value or "").strip() and city != str(value).strip():
                    flags.add("entity_grounded_city")
            else:
                flags.add("entity_dropped_unknown_city")
            continue

        if key == "service_name":
            if _should_drop_service_name_in_active_reschedule(
                label=label,
                user_text=user_text,
                raw_entities=raw,
                state=state,
            ):
                flags.add("entity_dropped_stale_service_name_in_reschedule")
                continue

            # Стараемся брать услугу из текущей реплики, а не "как есть" из LLM,
            # чтобы не залипали ложные service_name.
            phrase: str | None = None
            if label == "PRICE":
                phrase = await asyncio.to_thread(
                    resolve_price_service_name_from_catalog,
                    user_text,
                    current_service_name=str(value or ""),
                )
            if not phrase:
                phrase = extract_service_phrase(user_text) or extract_service_phrase(str(value or ""))
            strict_catalog_labels = {"APPOINTMENT", "PRICE", "ADDRESS", "TEST_ASSIST"}
            if phrase:
                if _is_doctor_like_service_collision(
                    label=label,
                    user_text=user_text,
                    phrase=phrase,
                    decision_flags=set(decision.flags),
                ):
                    flags.add("entity_dropped_doctor_like_service_name")
                    continue
                if label in strict_catalog_labels:
                    match = await services.match_catalog_service(
                        phrase,
                        current_service_name=str(state.last_entities.get("service_name") or ""),
                    )
                    status = str(match.get("status") or "")
                    nonbookable_keep = "policy_nonbookable_walkin" in set(decision.flags)
                    if status == "exact":
                        canonical = str(match.get("canonical") or "").strip()
                        if canonical:
                            out[key] = canonical
                            if canonical != phrase:
                                flags.add("entity_grounded_service_name")
                            continue
                    if status == "fuzzy":
                        candidate = str(match.get("canonical") or "").strip()
                        query = str(match.get("query") or phrase).strip()
                        if candidate:
                            out["_catalog_service_candidate"] = candidate
                            out["_catalog_service_query"] = query
                            flags.add("entity_catalog_service_fuzzy_candidate")
                        else:
                            flags.add("entity_dropped_unverified_service_name")
                        continue
                    if status == "unavailable":
                        if nonbookable_keep and phrase and not match_city(phrase):
                            out[key] = phrase
                            flags.add("entity_kept_nonbookable_service_name")
                            continue
                        if (
                            phrase
                            and not match_city(phrase)
                            and (
                                _SERVICE_ANCHOR_HINT_RE.search(phrase)
                                or _SERVICE_ANCHOR_HINT_RE.search(user_text or "")
                            )
                        ):
                            out[key] = phrase
                            flags.add("entity_kept_service_without_catalog")
                            continue
                        flags.add("entity_dropped_unverified_service_name")
                        continue
                    if nonbookable_keep and phrase and not match_city(phrase):
                        out[key] = phrase
                        flags.add("entity_kept_nonbookable_service_name")
                        continue
                    flags.add("entity_dropped_unverified_service_name")
                    continue

                out[key] = phrase
                if str(value or "").strip() and phrase != str(value).strip():
                    flags.add("entity_grounded_service_name")
            elif isinstance(value, str):
                fallback = value.strip()
                if label in strict_catalog_labels:
                    if "policy_nonbookable_walkin" in set(decision.flags) and fallback and not match_city(fallback):
                        out[key] = fallback
                        flags.add("entity_kept_nonbookable_service_name")
                    else:
                        flags.add("entity_dropped_unverified_service_name")
                elif fallback and not match_city(fallback) and not _GENERIC_SERVICE_FALLBACK_RE.search(fallback):
                    out[key] = fallback
                    flags.add("entity_kept_llm_service_name")
                else:
                    flags.add("entity_dropped_unverified_service_name")
            else:
                flags.add("entity_dropped_unverified_service_name")
            continue

        if key == "branch_name":
            branch = str(value or "").strip()
            if branch and looks_like_address(branch):
                out[key] = branch
            else:
                flags.add("entity_dropped_non_address_branch")
            continue

        out[key] = value

    # Не позволяем перетирать уже известного врача неподтвержденным значением.
    if (
        not out.get("doctor_name")
        and state.last_entities.get("doctor_name")
        and "doctor_name" in raw
    ):
        flags.add("entity_doctor_kept_from_state")

    return GroundingResult(entities=out, flags=flags)
