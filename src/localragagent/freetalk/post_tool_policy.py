"""Post-tool verification policy for FreeTalk."""

from __future__ import annotations

from typing import Any

from .routing_contract import (
    clarify_question_for_slots,
    clarify_type_for_slots,
    normalize_clarify_type,
)
from .contracts import PostToolVerification


_VALID_ANSWER_POLICIES = {"direct", "clarify", "not_found"}


def heuristic_post_tool_verification(
    *,
    intent: str,
    tool_name: str,
    drafted_answer: str,
    tool_payload: dict[str, Any],
    missing_slots: list[str],
) -> PostToolVerification:
    _ = tool_name
    payload = tool_payload if isinstance(tool_payload, dict) else {}
    clarify_text = str(payload.get("clarify_text") or "").strip()
    payload_missing = payload.get("missing_fields")
    has_missing_fields = isinstance(payload_missing, list) and any(str(item or "").strip() for item in payload_missing)
    clarify_slots = list(missing_slots or [])

    if clarify_text or has_missing_fields or clarify_slots:
        question = clarify_text or clarify_question_for_slots(intent, clarify_slots)
        clarify_type = clarify_type_for_slots(intent, clarify_slots)
        return PostToolVerification(
            enough_data=False,
            should_clarify=True,
            clarify_type=clarify_type,
            clarify_question=question,
            answer_policy="clarify",
            source="heuristic",
        )

    if str(drafted_answer or "").strip():
        return PostToolVerification(
            enough_data=True,
            should_clarify=False,
            clarify_type="",
            clarify_question="",
            answer_policy="direct",
            source="heuristic",
        )

    return PostToolVerification(
        enough_data=False,
        should_clarify=False,
        clarify_type="",
        clarify_question="",
        answer_policy="not_found",
        source="heuristic",
    )


def parse_post_tool_verification(
    payload: dict[str, Any],
    *,
    fallback: PostToolVerification,
) -> PostToolVerification:
    data = payload if isinstance(payload, dict) else {}
    enough_data = _coerce_bool(data.get("enough_data"), fallback.enough_data)
    should_clarify = _coerce_bool(data.get("should_clarify"), fallback.should_clarify)
    clarify_type = normalize_clarify_type(data.get("clarify_type"))
    clarify_question = str(data.get("clarify_question") or "").strip()
    answer_policy = str(data.get("answer_policy") or "").strip().lower()
    if answer_policy not in _VALID_ANSWER_POLICIES:
        if should_clarify:
            answer_policy = "clarify"
        elif enough_data:
            answer_policy = "direct"
        else:
            answer_policy = "not_found"
    if answer_policy == "clarify" and not clarify_type:
        clarify_type = str(fallback.clarify_type or "").strip()
    if answer_policy == "clarify" and not clarify_question:
        clarify_question = str(fallback.clarify_question or "").strip()
    return PostToolVerification(
        enough_data=enough_data,
        should_clarify=should_clarify,
        clarify_type=clarify_type,
        clarify_question=clarify_question,
        answer_policy=answer_policy,
        source="llm",
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "да"}:
        return True
    if text in {"0", "false", "no", "n", "нет"}:
        return False
    return default
