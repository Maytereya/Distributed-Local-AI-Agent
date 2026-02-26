"""Self-check для rich-режима рендера ответа пациенту."""

from __future__ import annotations

import json
from typing import Any

from .prompt_contracts import sanitize_self_check_json
from .prompt_registry import load_prompt_text


def build_critic_prompt(
    *,
    user_text: str,
    label: str,
    flags: list[str],
    evidence_items: dict[str, Any],
    candidate_answer: str,
) -> str:
    tmpl = load_prompt_text("renderer_critic_patient_alignment")
    return (
        tmpl.replace("<<USER_TEXT>>", user_text)
        .replace("<<LABEL>>", label)
        .replace("<<FLAGS>>", ", ".join(flags))
        .replace("<<EVIDENCE>>", json.dumps(evidence_items, ensure_ascii=False))
        .replace("<<CANDIDATE_ANSWER>>", candidate_answer)
    ).strip()


def parse_critic_result(raw_text: str) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    text = str(raw_text or "").strip()
    if text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                parsed = json.loads(chunk)
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                obj = {}
    return sanitize_self_check_json(obj)


def should_regenerate(result: dict[str, Any], *, threshold: float) -> tuple[bool, str]:
    aligned = bool(result.get("aligned_with_user_intent"))
    facts = bool(result.get("fact_consistency"))
    state_ok = bool(result.get("state_consistency"))
    unsafe = bool(result.get("unsafe_or_policy_violation"))
    score = float(result.get("score") or 0.0)
    reason = str(result.get("reason_short") or "").strip()
    if unsafe:
        return True, reason or "Обнаружено нарушение политики ответа."
    if (not aligned) or (not facts) or (not state_ok):
        return True, reason or "Ответ не соответствует запросу или фактам."
    if score < threshold:
        return True, reason or "Качество ответа ниже порога."
    return False, reason

