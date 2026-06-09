"""Новый оркестратор пайплайна роутера.

Пока это только каркас из 5 именованных стадий, который будет постепенно
замещать монолитный `route_patient_message()` из `router.py`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, AsyncIterator

from . import evidence_keys as ek
from .mess_types import AppointmentPhase, Evidence, Plan, ResponseEnvelope, RouteDecision, SessionState

_SAFETY_LABELS = {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}


@dataclass
class OrchestratorContext:
    """Контекст, который передаётся через все стадии пайплайна.

    :param text: исходный текст пользователя
    :param state: текущее состояние сессии
    :param decision: текущее routing-решение, если уже вычислено
    :param should_clarify: нужно ли вернуть уточняющий вопрос
    :param clarify_text: текст уточнения для пользователя
    :param nlu_debug: диагностические данные NLU для debug_trace
    :param tool_results: накопленные результаты вызовов инструментов/сервисов
    :param response: подготовленный ответ для пользователя
    :param short_circuit: нужно ли досрочно завершить пайплайн
    :param short_circuit_reason: причина досрочного завершения
    """

    text: str
    state: SessionState
    decision: RouteDecision | None = None
    should_clarify: bool = False
    clarify_text: str = ""
    nlu_debug: dict[str, Any] = field(default_factory=dict)
    tool_results: dict[str, Any] = field(default_factory=dict)
    plan: Plan | None = None
    evidence: Evidence | None = None
    response: ResponseEnvelope | None = None
    short_circuit: bool = False
    short_circuit_reason: str = ""
    # Per-stage wall-clock in milliseconds (filled by ``_timed_stage``). Used
    # by the Part IV perf work as a baseline for optimisation decisions.
    stage_timings: dict[str, float] = field(default_factory=dict)
    # Part IV Stage 15 (full OPTION B): speculative service-catalog match
    # fetched in parallel with doctor verification. Consumed by ``tool_loop``
    # via ``_complete_route_after_doctor_guard`` → ``_inject_catalog_candidates``.
    catalog_prefetch: dict[str, Any] | None = None


@asynccontextmanager
async def _timed_stage(ctx: OrchestratorContext, name: str) -> AsyncIterator[None]:
    """Record wall-clock latency for a pipeline stage into ``ctx.stage_timings``.

    :param ctx: pipeline context
    :param name: stage name used as the dict key (``early_guards``,
        ``nlu_route``, ``clarify_gate``, ``tool_loop``, ``render``)
    """

    t0 = perf_counter()
    try:
        yield
    finally:
        ctx.stage_timings[name] = round((perf_counter() - t0) * 1000, 1)


def _extract_prebuilt_response(
    ctx: OrchestratorContext,
    services: Any | None = None,
    memory: Any | None = None,
) -> ResponseEnvelope | None:
    """Извлекает готовый deterministic-ответ из evidence без вызова LLM.

    :param ctx: контекст пайплайна с decision/plan/evidence
    :param services: сервисный слой для builders, которым нужен доступ к каталогу
    :param memory: memory-store для builders, которые выставляют pending-состояния
    :return: готовый envelope или None, если нужен LLM-рендер
    """

    decision = ctx.decision
    plan = ctx.plan
    evidence = ctx.evidence
    if decision is None or evidence is None:
        return None

    if evidence.get(ek.AUTH_REQUIRED):
        return ResponseEnvelope(text=evidence.get(ek.AUTH_MESSAGE, "Нужна авторизация."), handoff=False)

    from .policies import evidence_requires_handoff, handoff_message

    handoff_required, handoff_msg, handoff_reason = evidence_requires_handoff(evidence)
    if handoff_required:
        return ResponseEnvelope(text=handoff_message(handoff_reason, handoff_msg), handoff=True)

    if plan is None or services is None or memory is None:
        return None

    from .response_builder import build_first_structured_response

    return build_first_structured_response(
        flow_label=plan.label,
        evidence=evidence,
        state=ctx.state,
        services=services,
        memory=memory,
        decision=decision,
        user_text=ctx.text,
    )


def _mark_secondary_offer_pending(ctx: OrchestratorContext) -> None:
    """Помечает, что после ответа можно предложить вторичный intent.

    :param ctx: контекст пайплайна после построения ответа
    :return: None
    """

    decision = ctx.decision
    plan = ctx.plan
    if decision is None or plan is None:
        return

    from .flow_policy import get_secondary_queue, secondary_followup_text

    secondary = get_secondary_queue(ctx.state)
    followup = secondary_followup_text(secondary)
    if (
        followup
        and not ctx.state.last_entities.get("_secondary_offer_pending")
        and not ctx.state.last_entities.get("_compound_price_pending")
        and not decision.needs_handoff
        and plan.label not in {"APPOINTMENT", "URGENT", "COMPLAINT", "MEDICAL_ADVICE", "TEST_RESULT"}
    ):
        ctx.state.last_entities["_secondary_offer_pending"] = True


def _extract_recovery_response(
    ctx: OrchestratorContext,
    memory: Any | None = None,
) -> ResponseEnvelope | None:
    """Строит ответ recovery-политики до deterministic/LLM-рендера.

    :param ctx: контекст пайплайна после tool_loop
    :param memory: хранилище pending-состояния
    :return: envelope recovery-ответа или None
    """

    decision = ctx.decision
    plan = ctx.plan
    if decision is None or plan is None or memory is None:
        return None

    from . import router
    from .policies import handoff_message
    from .recovery_policy import evaluate_recovery
    from .text_templates import LOW_CONF_CLARIFY_TEXT

    pending_now = memory.get_pending(ctx.state)
    flow_active = ctx.state.dialog.phase in (AppointmentPhase.COLLECTING, AppointmentPhase.CONFIRM)
    recovery = evaluate_recovery(
        user_text=ctx.text,
        decision=decision,
        flow_label=plan.label,
        pending_exists=bool(pending_now),
        flow_active=flow_active,
        state_entities=ctx.state.last_entities,
        summary=ctx.state.summary,
        max_unclear=3,
    )
    if recovery.kind == "handoff":
        return ResponseEnvelope(text=recovery.text or handoff_message("low_confidence"), handoff=True)
    if recovery.kind == "clarify":
        router._remember_question(
            ctx.state,
            recovery.reason or "clarify",
            list(decision.clarify_slots) if decision.clarify_slots else [],
        )
        return ResponseEnvelope(text=recovery.text or LOW_CONF_CLARIFY_TEXT, handoff=False)
    return None


def _extract_pending_response(
    ctx: OrchestratorContext,
    memory: Any | None = None,
) -> ResponseEnvelope | None:
    """Возвращает pending-уточнение раньше structured appointment-ответов.

    :param ctx: контекст пайплайна после tool_loop
    :param memory: memory-store с pending-слотами
    :return: envelope pending-ответа или None
    """

    decision = ctx.decision
    plan = ctx.plan
    evidence = ctx.evidence
    if decision is None or plan is None or evidence is None or memory is None:
        return None

    pending = memory.get_pending(ctx.state)
    if plan.steps or not pending:
        return None

    from . import router
    from .policies import clarification_question, handoff_message
    from .recovery_policy import contextual_reply_kind

    missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
    catalog_pending_reply = (
        plan.label == "OTHER"
        and isinstance(missing, list)
        and "catalog_confirm" in missing
        and isinstance(evidence.get(ek.CATALOG_CONFIRM_RESPONSE), dict)
    )
    if catalog_pending_reply:
        return None

    if (
        plan.label == "APPOINTMENT"
        and isinstance(missing, list)
        and "appointment_action" in missing
        and contextual_reply_kind(ctx.text) == "no"
    ):
        return ResponseEnvelope(text=handoff_message("manual_operator"), handoff=True)

    if plan.label == "APPOINTMENT" and isinstance(missing, list):
        action = str(ctx.state.last_entities.get("appointment_action") or "").strip().lower()
        needs_doctor = any(
            str(item).startswith("_any_of:")
            and ("doctor_id" in str(item) or "doctor_name" in str(item))
            for item in missing
        )
        needs_datetime = any(
            "date_from" in str(item) or "time_from" in str(item) or "date_hint" in str(item)
            for item in missing
        )
        if action in {"cancel", "reschedule"} and needs_doctor:
            attempts = int(ctx.state.last_entities.get("_appointment_doctor_lookup_attempts") or 0) + 1
            ctx.state.last_entities["_appointment_doctor_lookup_attempts"] = attempts
            if attempts >= 3:
                ctx.state.last_entities.pop("_appointment_doctor_lookup_attempts", None)
                return ResponseEnvelope(
                    text="Не удалось точно определить врача для этой записи. Соединяю с оператором.",
                    handoff=True,
                )
        else:
            ctx.state.last_entities.pop("_appointment_doctor_lookup_attempts", None)

        if action in {"cancel", "reschedule"} and needs_datetime:
            attempts = int(ctx.state.last_entities.get("_appointment_datetime_attempts") or 0) + 1
            ctx.state.last_entities["_appointment_datetime_attempts"] = attempts
            if attempts >= 3:
                ctx.state.last_entities.pop("_appointment_datetime_attempts", None)
                return ResponseEnvelope(
                    text="Не удалось точно определить дату или время записи. Соединяю с оператором.",
                    handoff=True,
                )
        else:
            ctx.state.last_entities.pop("_appointment_datetime_attempts", None)

    if plan.label == "APPOINTMENT":
        ctx.state.last_entities["appointment_flow_active"] = True
        ctx.state.dialog.phase = AppointmentPhase.COLLECTING
    router._remember_question(
        ctx.state,
        f"pending:{plan.label}",
        missing if isinstance(missing, list) else [],
    )
    return ResponseEnvelope(
        text=clarification_question(plan.label, missing if isinstance(missing, list) else [], ctx.state.last_entities),
        handoff=False,
    )


async def early_guards(
    ctx: OrchestratorContext,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Выполняет ранние safety-проверки до основного NLU.

    :param ctx: контекст пайплайна
    :param runtime_options: runtime-настройки LLM/NLU
    :return: обновлённый контекст
    """

    from . import classifier

    decision = await classifier.deterministic_rule_decision(
        ctx.text,
        ctx.state.last_entities,
        runtime_options=runtime_options,
    )
    if decision and decision.label in _SAFETY_LABELS:
        ctx.decision = decision
        ctx.short_circuit = True
        ctx.short_circuit_reason = "safety"
    return ctx


async def pending_dispatch(
    ctx: OrchestratorContext,
    services: Any | None = None,
    memory: Any | None = None,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Dispatch multi-turn pending handlers before running NLU.

    Matches the legacy ``route_patient_message`` order: operator_offer first,
    then catalog_confirm, appointment_action, compound_price, secondary_queue.
    If any handler fires the rest of the pipeline is short-circuited and the
    canned triple is used directly.

    :param ctx: pipeline context
    :param services: service layer (handlers need it for catalog lookups)
    :param memory: pending/state store
    :param runtime_options: runtime LLM/NLU options
    :return: updated context
    """

    if ctx.short_circuit or services is None or memory is None:
        return ctx

    from . import router

    operator_offer_result = router._handle_operator_offer_pending(ctx.text, ctx.state, memory)
    if operator_offer_result is not None:
        ctx.decision, ctx.plan, ctx.evidence = operator_offer_result
        ctx.short_circuit = True
        ctx.short_circuit_reason = "pending_handler"
        return ctx

    for handler in (
        router._handle_catalog_confirm_pending,
        router._handle_appointment_action_pending,
        router._handle_compound_price_pending,
        router._handle_secondary_queue_pending,
    ):
        result = await handler(
            user_text=ctx.text,
            state=ctx.state,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
        if result is not None:
            ctx.decision, ctx.plan, ctx.evidence = result
            ctx.short_circuit = True
            ctx.short_circuit_reason = "pending_handler"
            return ctx

    return ctx


async def nlu_route(
    ctx: OrchestratorContext,
    services: Any | None = None,
    memory: Any | None = None,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Запускает основной NLU и обновляет typed dialog state.

    :param ctx: контекст пайплайна
    :param runtime_options: runtime-настройки LLM/NLU
    :return: обновлённый контекст
    """

    if services is not None and memory is not None:
        from . import router

        ctx.decision, ctx.nlu_debug = await router._resolve_nlu_decision_before_doctor_guard(
            user_text=ctx.text,
            state=ctx.state,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
    else:
        from . import nlu_pipeline

        result = await nlu_pipeline.analyze_with_candidates(
            ctx.text,
            ctx.state,
            runtime_options,
        )
        ctx.decision = result.decision
        ctx.nlu_debug = {}

    ctx.state.dialog.merge_entities(ctx.decision.entities)
    ctx.state.dialog.label = ctx.decision.label
    ctx.state.dialog.confidence = ctx.decision.confidence
    ctx.state.dialog.missing_slots = list(ctx.decision.clarify_slots or [])
    return ctx


async def doctor_entity_guard(
    ctx: OrchestratorContext,
    services: Any | None = None,
) -> OrchestratorContext:
    """Проверяет и нормализует doctor entity до post-NLU middleware.

    :param ctx: контекст пайплайна после NLU-стадии
    :param services: сервисный слой для разрешения doctor_name
    :return: обновлённый контекст
    """

    if ctx.short_circuit or ctx.decision is None or services is None:
        return ctx

    from . import router

    ctx.decision, ctx.catalog_prefetch = await router._verify_decision_and_prefetch_catalog(
        decision=ctx.decision,
        user_text=ctx.text,
        state=ctx.state,
        services=services,
    )
    ctx.state.dialog.merge_entities(ctx.decision.entities)
    ctx.state.dialog.label = ctx.decision.label
    ctx.state.dialog.confidence = ctx.decision.confidence
    ctx.state.dialog.missing_slots = list(ctx.decision.clarify_slots or [])
    return ctx


async def clarify_gate(ctx: OrchestratorContext) -> OrchestratorContext:
    """Определяет, нужно ли остановиться на уточняющем вопросе.

    :param ctx: контекст пайплайна
    :return: обновлённый контекст
    """

    decision = ctx.decision
    if decision and decision.clarify_needed:
        ctx.should_clarify = True
        ctx.clarify_text = decision.clarify_reason
    if ctx.state.dialog.missing_slots:
        ctx.should_clarify = True
    return ctx


async def tool_loop(
    ctx: OrchestratorContext,
    services: Any | None = None,
    memory: Any | None = None,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Finish post-NLU middleware, plan and execute.

    Pending handlers already ran in ``pending_dispatch`` before NLU, so this
    stage only runs when a live NLU decision is present. It calls the shared
    router helper ``_complete_route_after_doctor_guard`` which finishes
    catalog injection, guardrails, entity grounding, planning, executor and
    graph transition.

    :param ctx: pipeline context
    :param services: service layer
    :param memory: pending/state store
    :param runtime_options: runtime LLM/NLU options
    :return: updated context
    """

    if ctx.short_circuit or ctx.should_clarify:
        return ctx
    if services is None or memory is None:
        ctx.tool_results = {}
        return ctx
    if ctx.decision is None:
        return ctx

    from . import router

    decision, plan, evidence = await router._complete_route_after_doctor_guard(
        decision=ctx.decision,
        user_text=ctx.text,
        state=ctx.state,
        services=services,
        memory=memory,
        runtime_options=runtime_options,
        nlu_debug=ctx.nlu_debug,
        catalog_prefetch=ctx.catalog_prefetch,
    )
    ctx.decision = decision
    ctx.plan = plan
    ctx.evidence = evidence
    ctx.state.dialog.label = ctx.decision.label
    ctx.state.dialog.confidence = ctx.decision.confidence
    ctx.state.dialog.merge_entities(ctx.decision.entities)
    return ctx


async def render(
    ctx: OrchestratorContext,
    runtime_options: Any | None = None,
    services: Any | None = None,
    memory: Any | None = None,
) -> OrchestratorContext:
    """Собирает финальный `ResponseEnvelope`.

    Порядок проверок:
    1. Safety short-circuits (URGENT / COMPLAINT / MEDICAL_ADVICE) — синхронный шаблон.
       ``pending_handler`` short-circuit с другими label'ами падает сюда же, но
       не попадает ни в одну из веток и переходит к обычному render_stream.
    2. Clarify gate — возвращает вопрос уточнения.
    3. Нормальный путь — собирает render_stream в строку, строит ResponseEnvelope.
    4. Defensive fallback (decision или evidence ещё не установлены).

    :param ctx: контекст пайплайна
    :param runtime_options: runtime-настройки LLM/NLU
    :param services: сервисный слой для deterministic-builders
    :param memory: memory-store для deterministic-builders
    :return: обновлённый контекст с заполненным ctx.response
    """

    from . import renderer

    # 1. Safety short-circuits — use sync templates, no LLM needed
    if ctx.short_circuit and ctx.decision is not None:
        if ctx.decision.label == "URGENT":
            ctx.response = renderer.render_urgent()
        elif ctx.decision.label == "COMPLAINT":
            ctx.response = renderer.render_complaint()
        elif ctx.decision.label == "MEDICAL_ADVICE":
            ctx.response = renderer.render_medical_advice()
        if ctx.response is not None:
            return ctx

    # 2. Clarify gate
    if ctx.should_clarify:
        ctx.response = ResponseEnvelope(text=ctx.clarify_text)
        return ctx

    if ctx.decision is not None:
        user_turn_count = sum(
            1 for item in (ctx.state.history or []) if isinstance(item, dict) and item.get("role") == "user"
        )
        if "smalltalk_greeting" in ctx.decision.flags and user_turn_count <= 1:
            from .text_templates import INTRO_TEXT

            ctx.response = ResponseEnvelope(text=INTRO_TEXT, attachments=[], handoff=False)
            return ctx

    recovery_response = _extract_recovery_response(ctx, memory=memory)
    if recovery_response is not None:
        ctx.response = recovery_response
        return ctx

    pending_response = _extract_pending_response(ctx, memory=memory)
    if pending_response is not None:
        ctx.response = pending_response
        return ctx

    # 3. Deterministic/pre-built path — response is already encoded in evidence.
    prebuilt = _extract_prebuilt_response(ctx, services=services, memory=memory)
    if prebuilt is not None:
        ctx.response = prebuilt
        _mark_secondary_offer_pending(ctx)
        return ctx

    # 4. LLM path — collect render_stream into one envelope.
    if ctx.decision is not None and ctx.evidence is not None:
        from .policies import scrub_internal_disclosure

        chunks: list[str] = []
        async for chunk in renderer.render_stream(
            ctx.text,
            ctx.decision,
            ctx.evidence,
            runtime_options=runtime_options,
        ):
            chunks.append(chunk)
        # Output-guard: единственный путь, способный слить системный/renderer-промпт
        # или модель — это свободная LLM-генерация. На ПОЛНОМ тексте (а не per-chunk)
        # ловим сигнатуры утечки и заменяем на безопасный дефлект. Покрывает оба
        # эндпоинта (оба собирают ответ здесь). См. messengers_router_bug_log.md.
        ctx.response = ResponseEnvelope(
            text=scrub_internal_disclosure("".join(chunks)),
            attachments=list(ctx.evidence.items.get(ek.ATTACHMENTS) or []),
            handoff=bool(ctx.decision.needs_handoff),
        )
        _mark_secondary_offer_pending(ctx)
        return ctx

    # 5. Defensive fallback — tool_loop bailed early (no services/memory)
    ctx.response = ResponseEnvelope(text="")
    return ctx


async def run_pipeline(
    text: str,
    state: SessionState,
    services: Any | None = None,
    memory: Any | None = None,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Прогоняет сообщение через 6-stage skeleton оркестратора.

    :param text: текст пользователя
    :param state: текущее состояние сессии
    :param services: сервисный слой для legacy-bridge
    :param memory: хранилище pending/state для legacy-bridge
    :param runtime_options: runtime-настройки LLM/NLU
    :return: финальный контекст после прохождения стадий
    """

    ctx = OrchestratorContext(text=text, state=state)
    async with _timed_stage(ctx, "early_guards"):
        ctx = await early_guards(ctx, runtime_options=runtime_options)
    if ctx.short_circuit:
        async with _timed_stage(ctx, "render"):
            return await render(ctx, runtime_options=runtime_options)
    async with _timed_stage(ctx, "pending_dispatch"):
        ctx = await pending_dispatch(
            ctx,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
    if ctx.short_circuit:
        async with _timed_stage(ctx, "render"):
            return await render(
                ctx,
                runtime_options=runtime_options,
                services=services,
                memory=memory,
            )
    async with _timed_stage(ctx, "nlu_route"):
        ctx = await nlu_route(
            ctx,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
    async with _timed_stage(ctx, "doctor_entity_guard"):
        ctx = await doctor_entity_guard(ctx, services=services)
    async with _timed_stage(ctx, "clarify_gate"):
        ctx = await clarify_gate(ctx)
    async with _timed_stage(ctx, "tool_loop"):
        ctx = await tool_loop(
            ctx,
            services=services,
            memory=memory,
            runtime_options=runtime_options,
        )
    async with _timed_stage(ctx, "render"):
        ctx = await render(
            ctx,
            runtime_options=runtime_options,
            services=services,
            memory=memory,
        )
    return ctx
