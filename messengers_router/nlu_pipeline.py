"""NLU-пайплайн patient-router с rollout между legacy и LLM-primary.

Ответственность модуля:
1) Выбрать NLU engine по runtime mode и rollout flags.
2) Для legacy-path выполнить deterministic+LLM merge.
3) Для нового path выполнить guardrail -> LLM-primary -> postprocess.

Legacy merge-policy сохранен как fallback/escape hatch.

Модуль не управляет диалоговым состоянием и не рендерит ответы:
он возвращает только NLU-решение и кандидаты для debug/аналитики.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from . import classifier
from .context_summary import seeded_context_for_nlu
from .llm_mode_policy import RuntimeOptions
from .mess_types import RouteDecision, SessionState

_SAFETY_LABELS = {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}


@dataclass
class NLUCandidate:
    source: str
    label: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class NLUResult:
    decision: RouteDecision
    candidates: list[NLUCandidate]
    merged_from: str
    trace: dict[str, Any] = field(default_factory=dict)


async def _rule_decision(
    text: str,
    last_entities: dict[str, Any],
    runtime_options: RuntimeOptions | None = None,
) -> RouteDecision:
    decision = await classifier.deterministic_rule_decision(
        text,
        last_entities,
        runtime_options=runtime_options,
        # В rule-pass NLU v2 не делаем refine, чтобы не добавлять extra LLM-call.
        allow_refine=False,
        # secondary intents для внутреннего merge не нужны.
        attach_secondary=False,
    )
    if decision is not None:
        return decision
    return RouteDecision(label="OTHER", confidence=0.2, flags={"rule_none"}, needs_handoff=False, context_action="continue")


def _candidate_from_decision(source: str, d: RouteDecision) -> NLUCandidate:
    return NLUCandidate(
        source=source,
        label=d.label,
        confidence=float(d.confidence),
        entities=dict(d.entities or {}),
        flags=sorted(list(d.flags)) if isinstance(d.flags, set) else [str(x) for x in (d.flags or [])],
    )


def _merge(rule: RouteDecision, llm: RouteDecision, *, llm_mode: str = "hybrid") -> tuple[RouteDecision, str]:
    # Safety всегда выше.
    if rule.label in _SAFETY_LABELS and llm.label not in _SAFETY_LABELS:
        return rule, "rule_safety"
    if llm.label in _SAFETY_LABELS:
        return llm, "llm_safety"

    # Если LLM не уверен и deterministic видит явный intent — промотируем rule.
    promote_threshold = 0.45 if llm_mode == "rich" else 0.55
    if llm.confidence < promote_threshold and rule.label != "OTHER":
        merged_flags = set(llm.flags) | set(rule.flags) | {"promoted_from_rule_pass"}
        merged_entities = dict(rule.entities or {})
        merged_entities.update(dict(llm.entities or {}))
        promoted = RouteDecision(
            label=rule.label,  # type: ignore[arg-type]
            confidence=max(rule.confidence, llm.confidence, 0.60),
            entities=merged_entities,
            flags=merged_flags,
            needs_handoff=False,
            context_action=llm.context_action,
            source="guardrail_post",
            clarify_needed=llm.clarify_needed,
            clarify_reason=llm.clarify_reason,
            clarify_slots=list(llm.clarify_slots),
            intent_candidates=list(llm.intent_candidates),
        )
        return promoted, "rule_promoted"

    return llm, "llm_primary"


def _engine_from_env() -> str:
    raw = str(os.getenv("MR_NLU_ENGINE", "legacy_v2")).strip().lower()
    if raw in {"legacy_v2", "llm_primary"}:
        return raw
    return "legacy_v2"


async def _analyze_legacy_with_candidates(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions | None = None,
) -> NLUResult:
    # bounded контекст для LLM pass
    seeded = seeded_context_for_nlu(state)
    llm_context = dict(state.last_entities)
    llm_context.update(seeded)

    rule = await _rule_decision(text, state.last_entities, runtime_options=runtime_options)
    llm = await classifier.analyze(text, llm_context, runtime_options=runtime_options)
    llm_mode = runtime_options.llm_mode if runtime_options else "hybrid"
    merged, source = _merge(rule, llm, llm_mode=llm_mode)
    candidates = [_candidate_from_decision("rule", rule), _candidate_from_decision("llm", llm)]
    return NLUResult(
        decision=merged,
        candidates=candidates,
        merged_from=source,
        trace={
            "guardrail_pre": {},
            "llm_primary_raw": "",
            "llm_primary_sanitized": {},
            "guardrail_post": {},
            "final_decision": {
                "label": merged.label,
                "confidence": merged.confidence,
                "source": source,
            },
        },
    )


async def _analyze_llm_primary_with_candidates(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions | None = None,
) -> NLUResult:
    seeded = seeded_context_for_nlu(state)
    llm_context = dict(state.last_entities)
    llm_context.update(seeded)

    decision, trace = await classifier.analyze_llm_primary(text, llm_context, runtime_options=runtime_options)
    candidate = _candidate_from_decision("llm_primary", decision)
    return NLUResult(
        decision=decision,
        candidates=[candidate],
        merged_from=str(decision.source or "llm_primary"),
        trace=trace,
    )


async def analyze_with_candidates(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions | None = None,
) -> NLUResult:
    opts = runtime_options or RuntimeOptions()
    if not opts.uses_llm_primary_nlu:
        return await _analyze_legacy_with_candidates(text, state, runtime_options=runtime_options)
    if _engine_from_env() != "llm_primary":
        return await _analyze_legacy_with_candidates(text, state, runtime_options=runtime_options)
    return await _analyze_llm_primary_with_candidates(text, state, runtime_options=runtime_options)
