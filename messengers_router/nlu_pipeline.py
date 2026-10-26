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

from dataclasses import dataclass, field
from typing import Any

from . import classifier, topic_shadow, turn_telemetry, walkin
from .context_summary import seeded_context_for_nlu
from .llm_mode_policy import RuntimeOptions
from .mess_types import RouteDecision, SessionState
from .runtime_config import config as c

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
) -> RouteDecision | None:
    return await classifier.deterministic_rule_decision(
        text,
        last_entities,
        runtime_options=runtime_options,
        # Держим те же настройки, что и в classifier.analyze(),
        # чтобы убрать дубли rule-pass без поведенческого расхождения.
        allow_refine=True,
        attach_secondary=True,
    )


def _candidate_from_decision(source: str, d: RouteDecision) -> NLUCandidate:
    return NLUCandidate(
        source=source,
        label=d.label,
        confidence=float(d.confidence),
        entities=dict(d.entities or {}),
        flags=sorted(list(d.flags)) if isinstance(d.flags, set) else [str(x) for x in (d.flags or [])],
    )


def _merge(rule: RouteDecision, llm: RouteDecision, *, llm_mode: str = "hybrid") -> tuple[RouteDecision, str]:
    """LLM-first merge policy.

    Intent source of truth
    ----------------------
    1. Safety hard-overrides: if rule fired a safety label (URGENT / COMPLAINT /
       MEDICAL_ADVICE), that always wins over LLM intent.
    2. LLM intent wins unconditionally for all non-safety labels regardless of
       confidence — the LLM handles nuance that regex cannot.

    Entity donation
    ---------------
    Deterministic rule entities are merged INTO the LLM decision as supplemental
    signal — they never override LLM-extracted entities.

    Rationale
    ---------
    The old promotion path (`llm.confidence < threshold → promote rule`) caused
    correct LLM classifications to be silently overridden by crude regex when the
    model was only moderately confident. This produced the green-eval / prod-fail
    gap: eval cases were unambiguous, but real user messages contained natural
    variation that lowered LLM confidence below the threshold.
    """
    # --- 1. Safety hard-overrides (bidirectional) ---
    if rule.label in _SAFETY_LABELS and llm.label not in _SAFETY_LABELS:
        return rule, "rule_safety"
    if llm.label in _SAFETY_LABELS:
        return llm, "llm_safety"

    # --- 1b. Strong deterministic PRICE rescue ---
    # Legacy merge — LLM-first, но уверенный rule-сигнал PRICE
    # («стоимость/цена <услуга>») нельзя терять, когда LLM ушёл в OTHER: иначе
    # явный ценовой вопрос утекает в low-confidence → оператор вместо
    # детерминированного прайс-флоу (прайс-данные и матчинг услуги есть).
    # Узко: спасаем ТОЛЬКО при llm.label == "OTHER", чтобы не перехватывать
    # уверенные не-OTHER интенты (APPOINTMENT/ADDRESS/...), упоминающие цену.
    if rule.label == "PRICE" and "rule_price" in (rule.flags or set()) and llm.label == "OTHER":
        donated = {
            k: v
            for k, v in (llm.entities or {}).items()
            if k not in (rule.entities or {}) and v not in (None, "", [])
        }
        rescued_entities = dict(rule.entities or {})
        rescued_entities.update(donated)
        rescued = RouteDecision(
            label="PRICE",
            confidence=rule.confidence,
            entities=rescued_entities,
            flags=set(rule.flags) | {"rule_price_rescue"},
            needs_handoff=False,
            context_action=rule.context_action,
            source="rule_price_rescue",
            clarify_needed=rule.clarify_needed,
            clarify_reason=rule.clarify_reason,
            clarify_slots=list(rule.clarify_slots),
            intent_candidates=list(rule.intent_candidates),
        )
        return rescued, "rule_price_rescue"

    # --- 2. LLM wins on intent; rule donates entities only ---
    donated_entities = {
        k: v for k, v in (rule.entities or {}).items()
        if k not in (llm.entities or {}) and v not in (None, "", [])
    }
    merged_entities = dict(llm.entities or {})
    merged_entities.update(donated_entities)

    merged = RouteDecision(
        label=llm.label,
        confidence=llm.confidence,
        entities=merged_entities,
        flags=set(llm.flags) | set(rule.flags),
        needs_handoff=llm.needs_handoff,
        context_action=llm.context_action,
        source="llm_primary",
        clarify_needed=llm.clarify_needed,
        clarify_reason=llm.clarify_reason,
        clarify_slots=list(llm.clarify_slots),
        intent_candidates=list(llm.intent_candidates),
    )
    return merged, "llm_primary"


def _rule_branch(rule: RouteDecision, text: str) -> str:
    """Флаги, которые поставила сработавшая ветка правила, — без PII и вторичных намерений (те ставит не ветка)."""

    own = set(rule.flags) - classifier.detect_pii(text) - {"multi_intent_detected"}
    return "+".join(sorted(str(f) for f in own)) or "?"


def _note_topic_source(rule: RouteDecision | None, text: str) -> dict[str, str]:
    """Кто решил тему реплики на уровне NLU — замер хода (б), DEBT-1 (STEP-2026-10-20-B-TOPIC-SOURCE).

    `nlu_src = rule` — ветка правила `legacy_v2` (или guardrail `llm_primary`), `nlu_rule` — её флаги; `nlu_src = llm` —
    правило не сработало, решал LLM-классификатор. Подпись решения (`decision_source`) этого не различает: решение правила
    проходит слияние с подписью `llm_primary`. Только запись, на решение не влияет.
    """

    if rule is None:
        turn_telemetry.note(nlu_src="llm")
        return {"src": "llm", "rule": ""}
    branch = _rule_branch(rule, text)
    turn_telemetry.note(nlu_src="rule", nlu_rule=branch)
    return {"src": "rule", "rule": branch}


def _topic_by_llm_rules() -> frozenset[str]:
    """`[MESSENGER_ROUTER] topic_by_llm` — ветки правил через запятую, у которых тему решает LLM (ход (б), этап 4, вариант
    Б, решение владельца 20.10 (б) 1, 5: по одному правилу); пусто (по умолчанию) — как было."""

    raw = str(getattr(c, "MR_TOPIC_BY_LLM", "") or "")
    rules = {part.strip() for part in raw.split(",") if part.strip()}
    if walkin.enabled():
        # Класс 4 (решения владельца 21.10): «без записи» по буквам тему не решает — решает LLM, правило — кандидат.
        rules.add("rule_nonbookable_walkin")
    return frozenset(rules)


def rule_defers_topic(rule: RouteDecision, text: str) -> bool:
    """Ветка сработавшего правила в `topic_by_llm` — тему этого хода решала бы LLM-классификатор, а правило — кандидат."""

    return bool(_topic_by_llm_rules() & set(_rule_branch(rule, text).split("+")))


def _topic_by_llm(
    rule: RouteDecision | None,
    topic_source: dict[str, str],
    runtime_options: RuntimeOptions | None,
    button_topic: str = "",
) -> bool:
    """Тему этого хода решает LLM, а правило — кандидат: ветка правила в `topic_by_llm`, правило не про безопасность
    (принцип 4: срочное LLM не снимает), LLM разрешена режимом хода, ход — не нажатие кнопки с известной темой."""

    if rule is None or rule.label in _SAFETY_LABELS:
        return False
    if runtime_options is not None and runtime_options.llm_mode == "strict":
        return False
    wanted = _topic_by_llm_rules()
    if not (wanted and wanted & set(str(topic_source.get("rule") or "").split("+"))):
        return False
    if button_topic:
        # Тему выбрал пациент нажатием (`button_menu`, режим `pass`), подпись — запрос этой темы: переспрашивать её у LLM —
        # только ждать классификатор (прод 23.10: 11–12 с на «Где и до скольки принимают»). Решает правило, как без списка
        # (BUG-2026-10-24-BUTTON-PASS-TOPIC-LLM; выключатель `button_pass_rules` снят 31.10).
        turn_telemetry.note(nlu_topic_by_llm="button")
        return False
    return True


def _button_answer_result(text: str, rule: RouteDecision | None, topic: str, topic_source: dict[str, str]) -> NLUResult:
    """Ответ на вопрос кнопки, который одна функция признала ответом (`button_answer_topic`,
    BUG-2026-10-25-BUTTON-ANSWER-CLASSIFIER): тему знает кнопка — там, где её решала бы LLM-классификатор (правило не
    сработало или его ветка в `topic_by_llm`). Правило той же темы отдаёт свои сущности, как при согласии LLM; иначе — тема
    кнопки без сущностей (значение слота возьмёт дозаполнение хода)."""

    if rule is not None and rule.label == topic:
        decision = rule
    else:
        decision = RouteDecision(label=topic, confidence=0.9, needs_handoff=False, context_action="continue", source="button_answer")
    decision.flags = set(decision.flags) | classifier.detect_pii(text) | {"button_answer_topic"}
    turn_telemetry.note(nlu_topic_by_llm="button_answer", nlu_src="button")
    return NLUResult(
        decision=decision,
        candidates=[_candidate_from_decision("button", decision)],
        merged_from="button_answer",
        trace={"final_decision": {"label": decision.label, "confidence": decision.confidence, "source": "button_answer"},
               "topic_source": {"src": "button", "rule": topic_source.get("rule", "")}},
    )


def _llm_failed(decision: RouteDecision) -> bool:
    return bool({"ollama_timeout", "ollama_non_json"} & set(decision.flags or ()))


def _engine_from_config() -> str:
    raw = str(c.MR_NLU_ENGINE).strip().lower()
    if raw in {"legacy_v2", "llm_primary"}:
        return raw
    return "legacy_v2"


def _rules_fallback_from_config() -> str:
    """`[MESSENGER_ROUTER] llm_primary_rules_fallback`: `off` (по умолчанию, как было) | `on`; нет ключа или мусор — `off`."""

    raw = str(getattr(c, "MR_LLM_PRIMARY_RULES_FALLBACK", "off") or "").strip().lower()
    return raw if raw in {"off", "on"} else "off"


async def _analyze_legacy_with_candidates(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions | None = None,
    prefetched_llm: dict[str, Any] | None = None,
    button_topic: str = "",
    answer_topic: str = "",
    answer_topic_firm: bool = False,
) -> NLUResult:
    # bounded контекст для LLM pass
    seeded = seeded_context_for_nlu(state)
    llm_context = dict(state.last_entities)
    llm_context.update(seeded)

    prefetched_rule = await _rule_decision(text, state.last_entities, runtime_options=runtime_options)
    topic_source = _note_topic_source(prefetched_rule, text)
    by_llm = _topic_by_llm(prefetched_rule, topic_source, runtime_options, button_topic)
    firm = answer_topic_firm and prefetched_rule is not None and prefetched_rule.label not in _SAFETY_LABELS and not prefetched_rule.needs_handoff
    if answer_topic and (prefetched_rule is None or by_llm or firm):
        return _button_answer_result(text, prefetched_rule, answer_topic, topic_source)
    if prefetched_rule is not None and not by_llm and (runtime_options is None or runtime_options.llm_mode != "strict"):
        # Тема решена правилом без LLM — тень LLM рядом (под `topic_shadow_llm = on`): как часто правило ошибается.
        # Промпт — тот же, что у хода без правила (`classifier.analyze` с тем же контекстом).
        topic_shadow.launch(
            text,
            classifier._seed_entities_from_memory(llm_context),
            prefetched_rule.label,
            queue_timeout_ms=int(runtime_options.queue_timeout_ms) if runtime_options else 30000,
        )
    rule = prefetched_rule or RouteDecision(
        label="OTHER",
        confidence=0.2,
        flags={"rule_none"},
        needs_handoff=False,
        context_action="continue",
    )
    llm = await classifier.analyze(
        text,
        llm_context,
        runtime_options=runtime_options,
        prefetched_rule=None if by_llm else prefetched_rule,
        prefetched_llm=prefetched_llm,
        skip_rule=by_llm,
    )
    if by_llm and _llm_failed(llm):
        # Сбой LLM — решает правило, как без выключателя (политика сбоя этапа 4, как `llm_primary_rules_fallback`).
        turn_telemetry.note(nlu_topic_by_llm="rules_fallback")
        llm = rule
    elif by_llm:
        turn_telemetry.note(nlu_topic_by_llm="llm", nlu_src="llm")
        topic_source = {"src": "llm", "rule": topic_source.get("rule", "")}
    llm_mode = runtime_options.llm_mode if runtime_options else "hybrid"
    merged, source = _merge(rule, llm, llm_mode=llm_mode)
    if by_llm and llm is not rule:
        # Тему решила LLM, правило — кандидат: детекторы того же правила после NLU метку LLM не перебивают
        # (`router._apply_post_nlu_guardrails`, строка `policy_nonbookable_walkin`).
        merged.flags = set(merged.flags) | {"topic_by_llm"}
        if merged.label == "OTHER" and rule.label != "OTHER":
            # OTHER от LLM — тема: вопрос вне наших тем, данных нет — честный оффер оператора, а не ответ правила (оно
            # выдумывало: «да, берём анализы у детей 2 лет», g0509) и не текст о техническом сбое (решение владельца 21.10).
            merged.flags = set(merged.flags) | {"topic_other_offer"}
            turn_telemetry.note(nlu_topic_by_llm="other_offer")
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
            "topic_source": topic_source,
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
    failure = trace.get("llm_primary_failure")
    if failure and _rules_fallback_from_config() == "on":
        # Запасной путь правил (решение владельца 20.10, п. 2; BUG-2026-10-20-LLM-PRIMARY-NO-RULES-FALLBACK): разбор LLM
        # не состоялся — ход решает код `legacy_v2` на том же (сбойном) ответе модели, без второго вызова LLM.
        fallback = await _analyze_legacy_with_candidates(
            text, state, runtime_options=runtime_options, prefetched_llm=dict(trace.get("llm_primary_sanitized") or {})
        )
        turn_telemetry.note(nlu_fallback="rules", nlu_fail=failure)
        fallback.trace = {
            **trace,
            "rules_fallback": failure,
            "final_decision": fallback.trace["final_decision"],
            "topic_source": fallback.trace.get("topic_source"),
        }
        return fallback
    topic_source = _note_topic_source(decision if (trace.get("guardrail_pre") or {}).get("hit") else None, text)
    candidate = _candidate_from_decision("llm_primary", decision)
    return NLUResult(
        decision=decision,
        candidates=[candidate],
        merged_from=str(decision.source or "llm_primary"),
        trace={**trace, "topic_source": topic_source},
    )


async def analyze_with_candidates(
    text: str,
    state: SessionState,
    runtime_options: RuntimeOptions | None = None,
    button_topic: str = "",
    answer_topic: str = "",
    answer_topic_firm: bool = False,
) -> NLUResult:
    """`button_topic` — тема нажатой кнопки режима `pass` (подпись идёт как текст); `answer_topic` — тема кнопки, на чей
    вопрос реплика — ответ (`button_answer_topic`; движок `llm_primary` её не читает), `answer_topic_firm` — её не
    перебивает и надёжное правило другой темы (смешанное сообщение, `button_topic_hint`); пусто — набранный текст."""

    opts = runtime_options or RuntimeOptions()
    button: dict[str, Any] = {"button_topic": button_topic} if button_topic else {}  # только когда кнопка есть: путь текста не меняется
    if answer_topic:
        button["answer_topic"] = answer_topic
    if answer_topic_firm:
        button["answer_topic_firm"] = True
    if not opts.uses_llm_primary_nlu:
        return await _analyze_legacy_with_candidates(text, state, runtime_options=runtime_options, **button)
    if _engine_from_config() != "llm_primary":
        return await _analyze_legacy_with_candidates(text, state, runtime_options=runtime_options, **button)
    return await _analyze_llm_primary_with_candidates(text, state, runtime_options=runtime_options)
