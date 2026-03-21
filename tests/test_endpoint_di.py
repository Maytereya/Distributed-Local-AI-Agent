import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from messengers_router import endpoint as endpoint_mod


class _DummyMemory:
    def __init__(self):
        self.calls = []

    async def aget(self, session_id: str):
        self.calls.append(("aget", session_id))
        return {"session_id": session_id}

    def append_turn(self, state, role: str, text: str):
        self.calls.append(("append_turn", role, text))

    async def aset(self, state):
        self.calls.append(("aset", state.get("session_id")))


class _DummyServices:
    pass


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    return app


def test_endpoint_uses_dependency_overrides(monkeypatch):
    app = _build_test_app()
    memory = _DummyMemory()
    services = _DummyServices()
    observed = {}

    async def _fake_patient_routing_stream(
        text,
        state,
        svc,
        mem,
        debug=False,
        runtime_options=None,
    ):
        observed["svc"] = svc
        observed["mem"] = mem
        observed["debug"] = debug
        observed["runtime_options"] = runtime_options
        yield SimpleNamespace(
            text="ok",
            attachments=[],
            handoff=False,
            state_update={"debug": {"source": "fake"}},
        )

    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    monkeypatch.setattr(endpoint_mod, "patient_routing_stream", _fake_patient_routing_stream)

    with TestClient(app) as client:
        once_resp = client.post(
            "/api/messenger-generate-once",
            json={"session_id": "s_once", "text": " hello ", "debug": True},
        )
        assert once_resp.status_code == 200
        once_body = once_resp.json()
        assert once_body["text"] == "ok"
        assert once_body["state_update"] == {"debug": {"source": "fake"}}
        assert observed["svc"] is services
        assert observed["mem"] is memory
        assert observed["debug"] is True
        assert ("append_turn", "user", "hello") in memory.calls
        assert ("append_turn", "assistant", "ok") in memory.calls

        stream_resp = client.post(
            "/api/messenger-generate",
            json={"session_id": "s_stream", "text": " ping ", "debug": True},
        )
        assert stream_resp.status_code == 200
        assert observed["debug"] is False
        line = stream_resp.text.strip().splitlines()[0]
        stream_obj = json.loads(line)
        assert stream_obj["text"] == "ok"

