"""Отказ LLM в свободном ответе — честная передача оператору, а не route_error (стенд 09.10).

Ветка свободного ответа ловила только тайм-аут бюджета. Любой другой отказ LLM — ollama
лежит (`ConnectionError`), очередь не дождалась слота (`LLMQueueTimeoutError`), сервер
ответил ошибкой (`ollama.ResponseError`) — улетал в `patient_routing_stream` и становился
`route_error`: текст пациенту тот же, но ход терял решение, тайминги и телеметрию. На
бейслайне в режиме «LLM недоступна» так 70 ходов из 1118.

Инвариант класса: отказ LLM в свободном ответе даёт тот же ответ, что тайм-аут бюджета
(`handoff_message("service_error")`, перевод на оператора), и ход остаётся целым — решение
в debug есть, `route_error` нет, в телеметрии `answer_source=render_error` и тип ошибки.
Дефект кода (не отказ LLM) и отмена задачи этой веткой НЕ глушатся.
"""

from __future__ import annotations

import asyncio

import httpx
import ollama
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from messengers_router import endpoint as endpoint_mod
from messengers_router import llm_runtime, orchestrator, renderer, turn_telemetry
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.policies import handoff_message
from messengers_router.services import Services

_TEXT = "расскажите о клинике"


def _llm_down(exc: Exception, *, on_await: bool):
    """Поддельный клиент ollama: короткие решения отвечают, поток падает.

    ``on_await`` — сбой при открытии потока; иначе — при чтении первого куска (так
    настоящий клиент ollama с ``stream=True``: генератор отдаётся без обращения к сети).
    """

    class _Down:
        async def generate(self, **kwargs):
            if not kwargs.get("stream"):
                return ollama.GenerateResponse(model="m", response="", done=True)
            if on_await:
                raise exc

            async def _chunks():
                raise exc
                yield  # pragma: no cover — делает функцию генератором

            return _chunks()

    return _Down()


@pytest.fixture
def rows(monkeypatch):
    """Ход доходит до свободного ответа LLM; строки телеметрии собираются."""

    from agent_logic_2.nayka_api import api_nayka
    from messengers_router.services import _empty_evidence_reply

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )
    # Ход без данных с L-05 до свободного текста не доходит — правило снимаем явно.
    monkeypatch.setattr(_empty_evidence_reply, "enabled", lambda: False)
    monkeypatch.setattr(api_nayka, "site_regions", lambda *a, **kw: [])
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])
    captured: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", captured.append)
    return captured


def _once() -> dict:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    resp = TestClient(app).post(
        "/api/messenger-generate-once", json={"session_id": "render-llm-error", "text": _TEXT, "debug": True}
    )
    assert resp.status_code == 200
    return resp.json()


def _debug(body: dict) -> dict:
    return (body.get("state_update") or {}).get("debug") or {}


@pytest.mark.parametrize("on_await", [True, False], ids=["open", "read"])
@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("Failed to connect to Ollama"),
        llm_runtime.LLMQueueTimeoutError("llm_queue_timeout"),
        ollama.ResponseError("model runner has unexpectedly stopped", 500),
        httpx.RemoteProtocolError("peer closed connection"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_llm_failure_in_free_answer_is_an_honest_handoff_not_route_error(rows, monkeypatch, exc, on_await):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _llm_down(exc, on_await=on_await))

    body = _once()

    assert body["text"] == handoff_message("service_error")
    assert body["handoff"] is True
    dbg = _debug(body)
    assert "route_error" not in dbg and "endpoint_error" not in dbg
    assert dbg["decision"]["label"] == "OTHER"
    assert "render" in dbg.get("stage_timings", {})
    assert len(rows) == 1
    row = rows[0]
    assert row["answer_source"] == "render_error"
    assert row["error"] == type(exc).__name__
    assert row["handoff"] is True
    assert _TEXT not in str(row)


def test_render_budget_timeout_keeps_its_own_branch(rows, monkeypatch):
    """Встречный: тайм-аут бюджета — прежняя ветка `render_timeout`, тот же текст."""

    class _Slow:
        async def generate(self, **kwargs):
            if not kwargs.get("stream"):
                return ollama.GenerateResponse(model="m", response="", done=True)

            async def _chunks():
                yield ollama.GenerateResponse(model="m", response="Начинаю", done=False)
                await asyncio.sleep(5)

            return _chunks()

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Slow())
    monkeypatch.setattr(renderer, "RENDER_TIMEOUT_S", 0.05)

    body = _once()

    assert body["text"] == handoff_message("service_error")
    assert body["handoff"] is True
    assert "route_error" not in _debug(body)
    assert rows[0]["answer_source"] == "render_timeout"


def test_working_llm_answer_is_unchanged(rows, monkeypatch):
    """Встречный: LLM отвечает — свободный ответ как раньше, без перевода."""

    class _Up:
        async def generate(self, **kwargs):
            if not kwargs.get("stream"):
                return ollama.GenerateResponse(model="m", response="", done=True)

            async def _chunks():
                yield ollama.GenerateResponse(model="m", response="Клиника «Наука» работает каждый день.", done=True)

            return _chunks()

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Up())

    body = _once()

    assert "работает каждый день" in body["text"]
    assert body["handoff"] is False
    assert rows[0]["answer_source"] == "llm_free"


def test_code_defect_in_free_answer_is_not_masked_as_llm_failure(rows, monkeypatch):
    """Встречный: ошибка кода (не отказ LLM) по-прежнему видна как route_error."""

    async def broken_stream(*_args, **_kwargs):
        raise KeyError("prompt_slot")
        yield  # pragma: no cover

    monkeypatch.setattr(renderer, "render_stream", broken_stream)

    body = _once()

    assert body["text"] == handoff_message("service_error")
    assert "route_error" in _debug(body)
    assert rows[0]["answer_source"] == "route_error"


def test_cancellation_is_not_swallowed(monkeypatch):
    """Отмена задачи (клиент ушёл, сервер гасится) не превращается в ответ пациенту."""

    async def cancelled_stream(*_args, **_kwargs):
        raise asyncio.CancelledError()
        yield  # pragma: no cover

    monkeypatch.setattr(renderer, "render_stream", cancelled_stream)
    from messengers_router.services import _empty_evidence_reply

    monkeypatch.setattr(_empty_evidence_reply, "enabled", lambda: False)
    ctx = orchestrator.OrchestratorContext(text=_TEXT, state=SessionState(session_id="cancel"))
    ctx.decision = RouteDecision(label="OTHER", confidence=0.9, entities={}, flags=set(), source="llm_primary")
    ctx.evidence = Evidence()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(orchestrator.render(ctx))
