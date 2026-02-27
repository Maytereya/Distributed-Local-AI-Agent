"""NLU-пайплайн v2 для роутера пациента (двухпроходная схема).

Ответственность модуля:
1) Выполнить deterministic pass (правила/безопасность/явные паттерны).
2) Выполнить LLM pass через существующий классификатор.
3) Слить результаты в одно решение по фиксированной политике приоритетов.

Принцип merge:
- safety-интенты имеют абсолютный приоритет;
- при слабой уверенности LLM и явном rule-сигнале применяется rule-promote;
- иначе используется решение LLM.

Модуль не управляет диалоговым состоянием и не рендерит ответы:
он возвращает только NLU-решение и кандидаты для debug/аналитики.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import classifier
from .context_summary import seeded_context_for_nlu
from .llm_mode_policy import RuntimeOptions
from .mess_types import RouteDecision, SessionState
from .policies import (
    detect_address_intent,
    detect_appointment_action,
    detect_appointment_intent,
    detect_complaint,
    detect_doc_request_intent,
    detect_doctor_info_intent,
    detect_medical_advice,
    detect_news_intent,
    detect_price_intent,
    detect_schedule_intent,
    detect_test_assist_intent,
    detect_test_result_intent,
    detect_test_interpretation,
    detect_urgent,
    extract_specialty,
    has_nearest_schedule_hint,
)

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


def _rule_decision(text: str) -> RouteDecision:
    specialty = extract_specialty(text or "")
    if detect_urgent(text):
        return RouteDecision(label="URGENT", confidence=1.0, flags={"rule_urgent"}, needs_handoff=True, context_action="new_topic")
    if detect_complaint(text):
        return RouteDecision(label="COMPLAINT", confidence=1.0, flags={"rule_complaint"}, needs_handoff=True, context_action="new_topic")
    if detect_medical_advice(text) or detect_test_interpretation(text):
        return RouteDecision(
            label="MEDICAL_ADVICE",
            confidence=1.0,
            flags={"rule_medical_advice"},
            needs_handoff=True,
            context_action="new_topic",
        )
    if detect_doc_request_intent(text):
        return RouteDecision(
            label="OTHER",
            confidence=0.99,
            flags={"doc_request_handoff"},
            needs_handoff=True,
            context_action="new_topic",
        )
    if detect_test_result_intent(text):
        return RouteDecision(label="TEST_RESULT", confidence=0.85, flags={"rule_test_result"}, needs_handoff=False, context_action="continue")
    appt = detect_appointment_intent(text)
    appt_action = detect_appointment_action(text)
    price = detect_price_intent(text)
    addr = detect_address_intent(text)
    if price and not appt_action:
        return RouteDecision(label="PRICE", confidence=0.72, flags={"rule_price"}, needs_handoff=False, context_action="continue")
    if detect_schedule_intent(text):
        entities = {"specialty": specialty} if specialty else {}
        flags = {"rule_schedule"}
        if specialty:
            flags.add("rule_schedule_with_specialty")
        if specialty and has_nearest_schedule_hint(text):
            flags.add("rule_schedule_nearest")
        return RouteDecision(
            label="DOCTOR_SCHEDULE",
            confidence=0.74,
            entities=entities,
            flags=flags,
            needs_handoff=False,
            context_action="continue",
        )
    if detect_doctor_info_intent(text):
        entities = {"specialty": specialty} if specialty else {}
        flags = {"rule_doctor_info"}
        if specialty:
            flags.add("rule_doctor_info_with_specialty")
        return RouteDecision(
            label="DOCTOR_INFO",
            confidence=0.72,
            entities=entities,
            flags=flags,
            needs_handoff=False,
            context_action="continue",
        )
    if specialty and has_nearest_schedule_hint(text):
        return RouteDecision(
            label="DOCTOR_SCHEDULE",
            confidence=0.72,
            entities={"specialty": specialty},
            flags={"rule_schedule_nearest", "rule_schedule_with_specialty"},
            needs_handoff=False,
            context_action="continue",
        )
    # Короткие запросы по специальности ("урологи", "нужен гастроэнтеролог")
    # трактуем как поиск врачей, а не OTHER.
    if specialty and not appt:
        return RouteDecision(
            label="DOCTOR_INFO",
            confidence=0.70,
            entities={"specialty": specialty},
            flags={"rule_doctor_info_specialty"},
            needs_handoff=False,
            context_action="continue",
        )
    if addr:
        return RouteDecision(label="ADDRESS", confidence=0.72, flags={"rule_address"}, needs_handoff=False, context_action="continue")
    if appt:
        return RouteDecision(label="APPOINTMENT", confidence=0.75, flags={"rule_appointment"}, needs_handoff=False, context_action="continue")
    if detect_news_intent(text):
        return RouteDecision(label="NEWS", confidence=0.72, flags={"rule_news"}, needs_handoff=False, context_action="continue")
    if detect_test_assist_intent(text):
        return RouteDecision(label="TEST_ASSIST", confidence=0.70, flags={"rule_test_assist"}, needs_handoff=False, context_action="continue")
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
        )
        return promoted, "rule_promoted"

    return llm, "llm_primary"


async def analyze_with_candidates(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions | None = None,
) -> NLUResult:
    # bounded контекст для LLM pass
    seeded = seeded_context_for_nlu(state)
    llm_context = dict(state.last_entities)
    llm_context.update(seeded)

    rule = _rule_decision(text)
    llm = await classifier.analyze(text, llm_context, runtime_options=runtime_options)
    llm_mode = runtime_options.llm_mode if runtime_options else "hybrid"
    merged, source = _merge(rule, llm, llm_mode=llm_mode)
    candidates = [_candidate_from_decision("rule", rule), _candidate_from_decision("llm", llm)]
    return NLUResult(decision=merged, candidates=candidates, merged_from=source)
