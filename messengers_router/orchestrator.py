"""Новый оркестратор пайплайна роутера.

Пока это только каркас из 5 именованных стадий, который будет постепенно
замещать монолитный `route_patient_message()` из `router.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .mess_types import ResponseEnvelope, RouteDecision, SessionState

_SAFETY_LABELS = {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}


@dataclass
class OrchestratorContext:
    """Контекст, который передаётся через все стадии пайплайна.

    :param text: исходный текст пользователя
    :param state: текущее состояние сессии
    :param decision: текущее routing-решение, если уже вычислено
    :param should_clarify: нужно ли вернуть уточняющий вопрос
    :param clarify_text: текст уточнения для пользователя
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
    tool_results: dict[str, Any] = field(default_factory=dict)
    response: ResponseEnvelope | None = None
    short_circuit: bool = False
    short_circuit_reason: str = ""


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


async def nlu_route(
    ctx: OrchestratorContext,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Запускает основной NLU и обновляет typed dialog state.

    :param ctx: контекст пайплайна
    :param runtime_options: runtime-настройки LLM/NLU
    :return: обновлённый контекст
    """

    from . import nlu_pipeline

    result = await nlu_pipeline.analyze_with_candidates(
        ctx.text,
        ctx.state,
        runtime_options,
    )
    ctx.decision = result.decision
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
) -> OrchestratorContext:
    """Заглушка для будущего вызова сервисов и инструментов.

    :param ctx: контекст пайплайна
    :param services: сервисный слой, который будет задействован позже
    :return: обновлённый контекст
    """

    _ = services
    if ctx.should_clarify:
        return ctx
    ctx.tool_results = {}
    return ctx


async def render(ctx: OrchestratorContext) -> OrchestratorContext:
    """Собирает финальный `ResponseEnvelope`.

    :param ctx: контекст пайплайна
    :return: обновлённый контекст
    """

    if ctx.should_clarify:
        ctx.response = ResponseEnvelope(text=ctx.clarify_text)
        return ctx
    ctx.response = ResponseEnvelope(text="")
    return ctx


async def run_pipeline(
    text: str,
    state: SessionState,
    runtime_options: Any | None = None,
) -> OrchestratorContext:
    """Прогоняет сообщение через 5-stage skeleton оркестратора.

    :param text: текст пользователя
    :param state: текущее состояние сессии
    :param runtime_options: runtime-настройки LLM/NLU
    :return: финальный контекст после прохождения стадий
    """

    ctx = OrchestratorContext(text=text, state=state)
    ctx = await early_guards(ctx, runtime_options=runtime_options)
    if ctx.short_circuit:
        return ctx
    ctx = await nlu_route(ctx, runtime_options=runtime_options)
    ctx = await clarify_gate(ctx)
    ctx = await tool_loop(ctx, services=None)
    ctx = await render(ctx)
    return ctx
