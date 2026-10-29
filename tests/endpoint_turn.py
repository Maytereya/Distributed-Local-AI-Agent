"""Ход пациента через обработчик эндпоинта — замена снятой точки входа `router.route_patient_message` (К3, 29.10).

С 15.04 ход идёт только через `endpoint.messenger_generate_once` → `router.patient_routing_stream` →
`orchestrator.run_pipeline`; прежняя точка входа звалась лишь из тестов и расходилась с прод-путём (решение владельца
29.10 — снять, консилиум: тесты не удалять, а перевести на эндпоинт). Тесты, которые разбирали решение, план и данные
хода, получают их из контекста `run_pipeline` ЭТОГО хода — того же, по которому эндпоинт собрал ответ пациенту.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any, Iterator

from messengers_router import orchestrator
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, Plan, RouteDecision, SessionState
from messengers_router.services import Services


@dataclass
class EndpointTurn:
    """Ответ эндпоинта (`body`) и контекст `run_pipeline` хода (`ctx`; None — ход закончился до пайплайна)."""

    body: dict[str, Any]
    ctx: orchestrator.OrchestratorContext | None

    @property
    def text(self) -> str:
        return str(self.body.get("text") or "")

    @property
    def handoff(self) -> bool:
        return bool(self.body.get("handoff"))

    def route(self) -> tuple[RouteDecision, Plan, Evidence]:
        """Решение, план и данные хода — как их отдавала `route_patient_message`."""

        assert self.ctx is not None, f"ход закончился до пайплайна: {self.body!r}"
        assert self.ctx.decision is not None, f"пайплайн не принял решения: {self.body!r}"
        assert self.ctx.plan is not None and self.ctx.evidence is not None, f"ход без плана (уточнение): {self.body!r}"
        return self.ctx.decision, self.ctx.plan, self.ctx.evidence


@contextlib.contextmanager
def _capture_pipeline(state: SessionState) -> Iterator[list[orchestrator.OrchestratorContext]]:
    """Контексты `run_pipeline` по состоянию сессии (побочный ответ о филиале идёт на копии — его не берём)."""

    seen: list[orchestrator.OrchestratorContext] = []
    real = orchestrator.run_pipeline

    async def spy(text: str, st: SessionState, *args: Any, **kwargs: Any) -> orchestrator.OrchestratorContext:
        ctx = await real(text, st, *args, **kwargs)
        if st is state:
            seen.append(ctx)
        return ctx

    orchestrator.run_pipeline = spy
    try:
        yield seen
    finally:
        orchestrator.run_pipeline = real


def endpoint_turn(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    *,
    button_id: str = "",
) -> EndpointTurn:
    """Один ход через `messenger_generate_once` над состоянием `state` (кладётся в `memory` под своим session_id)."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once

    memory.set(state)

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=state.session_id, text=user_text, debug=True, button_id=button_id)
        return json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)

    with _capture_pipeline(state) as seen:
        body = asyncio.run(run())
    return EndpointTurn(body=body, ctx=seen[-1] if seen else None)


def route_via_endpoint(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence]:
    """Решение, план и данные хода, прошедшего через эндпоинт."""

    return endpoint_turn(user_text, state, services, memory).route()
