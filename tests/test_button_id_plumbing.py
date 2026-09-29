"""Шаг 1 кнопок: бот принимает `button_id` и доносит его до конвейера (29.09).

Меню Telegram живёт у шлюза; нажатие кнопки-листа приходит идентификатором в
поле `button_id` (контракт — «Таблица экранов, версия 1»). На этом шаге поле
только доставляется до конвейера, поведение бота не меняется.

Правило, общее со шлюзом: запрос БЕЗ кнопки обрабатывается ровно как раньше —
`button_id` передаётся дальше только когда он есть, поэтому существующие фейки
`patient_routing_stream` / `run_pipeline` с прежними сигнатурами не ломаются.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from messengers_router import endpoint as endpoint_mod
from messengers_router import orchestrator as orchestrator_mod
from messengers_router import router as router_mod
from messengers_router.endpoint import MessengerGenerateRequest
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import ResponseEnvelope, RouteDecision, SessionState
from messengers_router.orchestrator import OrchestratorContext
from messengers_router.services import Services

# --- модель запроса ------------------------------------------------------------


def test_request_model_accepts_button_id():
    req = MessengerGenerateRequest(session_id="tg_1", text="Цена анализа", button_id="menu.price.test")
    assert req.button_id == "menu.price.test"


def test_request_without_button_id_defaults_to_empty():
    assert MessengerGenerateRequest(session_id="tg_1", text="цена оак").button_id == ""


def test_empty_text_still_rejected_with_button_id():
    # Контракт со шлюзом: подпись кнопки идёт в text, пустой text — 422.
    with pytest.raises(ValidationError):
        MessengerGenerateRequest(session_id="tg_1", text="", button_id="menu.price.test")


# --- эндпоинты доносят button_id до роутера -----------------------------------


class _Memory:
    async def aget(self, session_id):
        return SessionState(session_id=session_id)

    def append_turn(self, state, role, text):
        pass

    async def aset(self, state):
        pass


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = _Memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: object()
    return app


def _fake_stream_recording(observed: dict):
    async def fake(text, state, services, memory, debug=False, runtime_options=None, **kwargs):
        observed.update(kwargs)
        yield ResponseEnvelope(text="ok")

    return fake


@pytest.mark.parametrize("path", ["/api/messenger-generate-once", "/api/messenger-generate"])
def test_endpoint_passes_button_id_to_router(monkeypatch, path):
    observed: dict = {}
    monkeypatch.setattr(endpoint_mod, "patient_routing_stream", _fake_stream_recording(observed))
    app = _app()
    resp = TestClient(app).post(
        path, json={"session_id": "tg_1", "text": "Цена анализа", "button_id": "menu.price.test"}
    )
    assert resp.status_code == 200
    assert observed.get("button_id") == "menu.price.test"


@pytest.mark.parametrize("path", ["/api/messenger-generate-once", "/api/messenger-generate"])
def test_endpoint_without_button_id_calls_router_as_before(monkeypatch, path):
    # Фейк с прежней сигнатурой, без **kwargs: запрос без кнопки обязан пройти.
    async def old_signature(text, state, services, memory, debug=False, runtime_options=None):
        yield ResponseEnvelope(text="ok")

    monkeypatch.setattr(endpoint_mod, "patient_routing_stream", old_signature)
    app = _app()
    resp = TestClient(app).post(path, json={"session_id": "tg_1", "text": "цена оак"})
    assert resp.status_code == 200


# --- роутер доносит button_id до конвейера ------------------------------------


def _run_stream(text, state, services, memory, **kwargs):
    async def collect():
        return [env async for env in router_mod.patient_routing_stream(text, state, services, memory, **kwargs)]

    return asyncio.run(collect())


def _services():
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    return services


def test_router_passes_button_id_to_pipeline(monkeypatch):
    observed: dict = {}

    async def fake_run_pipeline(text, state, services=None, memory=None, runtime_options=None, **kwargs):
        observed.update(kwargs)
        ctx = OrchestratorContext(text=text, state=state)
        ctx.decision = RouteDecision(label="PRICE", confidence=1.0, source="button")
        ctx.response = ResponseEnvelope(text="Скажите название анализа.")
        return ctx

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)
    _run_stream("Цена анализа", SessionState(session_id="b1"), _services(), MemoryStore(), button_id="menu.price.test")
    assert observed.get("button_id") == "menu.price.test"


def test_router_without_button_id_calls_pipeline_as_before(monkeypatch):
    async def old_signature(text, state, services=None, memory=None, runtime_options=None):
        ctx = OrchestratorContext(text=text, state=state)
        ctx.decision = RouteDecision(label="PRICE", confidence=1.0, source="llm_primary")
        ctx.response = ResponseEnvelope(text="ok")
        return ctx

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", old_signature)
    out = _run_stream("цена оак", SessionState(session_id="b2"), _services(), MemoryStore())
    assert out[-1].text == "ok"


# --- конвейер кладёт button_id в контекст -------------------------------------


class _Stop(Exception):
    pass


def test_pipeline_puts_button_id_into_context(monkeypatch):
    seen: dict = {}

    async def spy_early_guards(ctx, runtime_options=None):
        seen["button_id"] = ctx.button_id
        raise _Stop

    monkeypatch.setattr(orchestrator_mod, "early_guards", spy_early_guards)
    with pytest.raises(_Stop):
        asyncio.run(
            orchestrator_mod.run_pipeline("Цена анализа", SessionState(session_id="b3"), button_id="menu.price.test")
        )
    assert seen["button_id"] == "menu.price.test"


def test_context_button_id_defaults_to_empty():
    assert OrchestratorContext(text="цена", state=SessionState(session_id="b4")).button_id == ""
