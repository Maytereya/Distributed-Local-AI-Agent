from fastapi.testclient import TestClient

import agent_api as api_mod


def test_freetalk_debug_turn_fields_include_policy_and_memory_keys():
    fields = api_mod._freetalk_debug_turn_fields(
        text="найди в сети актуальные рекомендации",
        dialog_state={
            "intent": "appointment",
            "flow_active": True,
            "flow_kind": "appointment",
            "missing_slots": ["patient_name"],
        },
        session_entity_memory={"doctor_name": "Дразнин Антон Владимирович"},
    )

    assert fields["turn_kind"] == "web"
    assert fields["flow_relation"] == "switch"
    assert fields["source_mode"] == "web"
    assert fields["memory_updates"]["has_entity_memory"] is True
    assert fields["memory_updates"]["entity_keys"] == ["doctor_name"]


def test_freetalk_generate_once_endpoint_uses_api_key_and_helper(monkeypatch):
    observed = {}

    async def _fake_run_freetalk_once(*, text: str, session_id: str, debug: bool):
        observed["text"] = text
        observed["session_id"] = session_id
        observed["debug"] = debug
        return api_mod.FreeTalkResponse(
            text="ok",
            session_id=session_id,
            next_session_id=session_id,
            source="clinic_data",
            tool_name="doctors_info",
            reply_kind="final",
            source_fragments=[api_mod.FreeTalkSourceFragment(text="ok", source="clinic_data")],
            outcome="ok",
            degraded=True,
            handoff=True,
            attachments=[{"type": "pdf", "url": "https://example.org/result.pdf"}],
            debug={"dialog_state": {}},
        )

    monkeypatch.setattr(api_mod, "EXPECTED_API_KEY", "test-key")
    monkeypatch.setattr(api_mod, "_run_freetalk_once", _fake_run_freetalk_once)

    with TestClient(api_mod.app) as client:
        resp = client.post(
            "/v1/freetalk/generate-once",
            headers={"X-API-Key": "test-key"},
            json={"session_id": "ft_case_1", "text": "Привет", "debug": True},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "ok"
    assert body["source"] == "clinic_data"
    assert body["tool_name"] == "doctors_info"
    assert body["reply_kind"] == "final"
    assert body["outcome"] == "ok"
    assert body["degraded"] is True
    assert body["handoff"] is True
    assert body["attachments"] == [{"type": "pdf", "url": "https://example.org/result.pdf"}]
    assert body["debug"] == {"dialog_state": {}}
    assert observed == {
        "text": "Привет",
        "session_id": "ft_case_1",
        "debug": True,
    }


def test_freetalk_generate_once_endpoint_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setattr(api_mod, "EXPECTED_API_KEY", "test-key")

    with TestClient(api_mod.app) as client:
        resp = client.post(
            "/v1/freetalk/generate-once",
            headers={"X-API-Key": "wrong"},
            json={"session_id": "ft_case_2", "text": "Привет"},
        )

    assert resp.status_code == 401
