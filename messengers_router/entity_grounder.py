"""Entity grounding для NLU-решений роутера.

Задача модуля:
1) Ограничить, какие entities вообще можно принять на текущем шаге.
2) Подтверждать критичные сущности по данным/эвристикам (doctor/city/service).
3) Возвращать только "безопасные" entities для merge в session state.
Ответственность модуля: не допускать попадание в state неподтвержденных сущностей.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .city import looks_like_address, match_city
from .mess_types import RouteDecision, SessionState
from .policies import extract_service_phrase
from .services import Services

_CONTROL_KEYS = {
    "secondary_intents",
    "secondary_intent_from_queue",
    "appointment_action",
    "result_action",
    "query_terms",
}

_GENERIC_SERVICE_FALLBACK_RE = re.compile(
    r"\b(запис\w*|врач\w*|доктор\w*|специалист\w*|услуг\w*|хочу|нужно|надо|можно)\b",
    re.I,
)

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
            # Стараемся брать услугу из текущей реплики, а не "как есть" из LLM,
            # чтобы не залипали ложные service_name.
            phrase = extract_service_phrase(user_text) or extract_service_phrase(str(value or ""))
            if phrase:
                out[key] = phrase
                if str(value or "").strip() and phrase != str(value).strip():
                    flags.add("entity_grounded_service_name")
            elif isinstance(value, str):
                fallback = value.strip()
                if fallback and not match_city(fallback) and not _GENERIC_SERVICE_FALLBACK_RE.search(fallback):
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
