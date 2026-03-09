"""Bounded summary для контекста сессии.

Ответственность модуля:
1) Собирать компактную сводку из последних реплик и критичных слотов.
2) Ограничивать размер контекста, чтобы не раздувать prompt и latency.
3) Отдавать seed-контекст для NLU, не передавая весь хвост истории.

Модуль не содержит бизнес-логики интентов и не принимает routing-решений.
"""

from __future__ import annotations

from typing import Any

from .mess_types import SessionState

_SUMMARY_MAX_CHARS = 1600
_TAIL_TURNS = 10

_CRITICAL_KEYS = (
    "_last_label",
    "doctor_name",
    "doctor_id",
    "specialty",
    "city",
    "branch_name",
    "service_name",
    "test_name",
    "patient_name",
    "date_from",
    "time_from",
    "appointment_flow_active",
    "appointment_confirm_pending",
    "_pending_label",
    "last_question_kind",
    "last_clarify_slots",
)


def _compact_text(s: str, max_len: int = 220) -> str:
    txt = " ".join(str(s or "").split())
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3].rstrip() + "..."


def update_summary(state: SessionState, *, reason: str = "", every_n_user_turns: int = 4) -> str:
    """
    Обновляет bounded summary в SessionState.
    Обновление выполняется периодически и при явной смене темы.
    """
    user_turns = [h for h in (state.history or []) if isinstance(h, dict) and h.get("role") == "user"]
    if not user_turns:
        state.summary = ""
        return state.summary

    must_refresh = bool(reason in {"topic_switch", "handoff"})
    if not must_refresh and (len(user_turns) % max(1, every_n_user_turns) != 0):
        return state.summary or ""

    tail = (state.history or [])[-_TAIL_TURNS:]
    lines: list[str] = []
    for h in tail:
        if not isinstance(h, dict):
            continue
        role = str(h.get("role") or "").strip() or "user"
        text = _compact_text(str(h.get("text") or ""))
        if not text:
            continue
        lines.append(f"{role}: {text}")

    entity_bits: list[str] = []
    ents = state.last_entities or {}
    for k in _CRITICAL_KEYS:
        v = ents.get(k)
        if v in (None, "", [], {}):
            continue
        entity_bits.append(f"{k}={v}")

    parts: list[str] = []
    if lines:
        parts.append("last_turns:\n" + "\n".join(lines))
    if entity_bits:
        parts.append("state:\n" + ", ".join(entity_bits))
    if reason:
        parts.append(f"reason={reason}")

    summary = "\n\n".join(parts).strip()
    if len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[: _SUMMARY_MAX_CHARS - 3].rstrip() + "..."
    state.summary = summary
    return summary


def seeded_context_for_nlu(state: SessionState) -> dict[str, Any]:
    """
    Формирует bounded контекст для классификатора.
    """
    seeded: dict[str, Any] = {}
    for k in _CRITICAL_KEYS:
        v = state.last_entities.get(k)
        if v in (None, "", [], {}):
            continue
        seeded[k] = v
    if state.summary:
        seeded["_summary"] = state.summary
    return seeded
