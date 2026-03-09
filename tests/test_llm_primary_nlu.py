import asyncio

from messengers_router import classifier
from messengers_router.llm_mode_policy import normalize_runtime_options
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult, NLUCandidate, analyze_with_candidates
from messengers_router.recovery_policy import evaluate_recovery
from messengers_router.router import _debug_meta
from messengers_router.mess_types import Evidence, Plan


def run(coro):
    return asyncio.run(coro)


def test_llm_primary_engine_used_when_enabled(monkeypatch):
    async def fake_llm_primary(text, state, runtime_options=None):
        _ = text, state, runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="PRICE",
                confidence=0.83,
                entities={"service_name": "УЗИ брюшной полости"},
                flags={"llm_primary"},
                source="llm_primary",
                clarify_needed=False,
            ),
            candidates=[NLUCandidate(source="llm_primary", label="PRICE", confidence=0.83)],
            merged_from="llm_primary",
            trace={"final_decision": {"label": "PRICE", "source": "llm_primary"}},
        )

    async def fake_legacy(text, state, runtime_options=None):
        _ = text, state, runtime_options
        raise AssertionError("legacy engine should not be used")

    monkeypatch.setenv("MR_NLU_ENGINE", "llm_primary")
    monkeypatch.setattr("messengers_router.nlu_pipeline._analyze_llm_primary_with_candidates", fake_llm_primary)
    monkeypatch.setattr("messengers_router.nlu_pipeline._analyze_legacy_with_candidates", fake_legacy)

    state = SessionState(session_id="llm-primary")
    out = run(analyze_with_candidates("сколько стоит узи", state, normalize_runtime_options(llm_mode="hybrid")))

    assert out.decision.label == "PRICE"
    assert out.decision.source == "llm_primary"


def test_strict_mode_forces_legacy_engine(monkeypatch):
    async def fake_llm_primary(text, state, runtime_options=None):
        _ = text, state, runtime_options
        raise AssertionError("llm_primary engine should not be used in strict mode")

    async def fake_legacy(text, state, runtime_options=None):
        _ = text, state, runtime_options
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.4, source="fallback"),
            candidates=[NLUCandidate(source="legacy", label="OTHER", confidence=0.4)],
            merged_from="legacy",
        )

    monkeypatch.setenv("MR_NLU_ENGINE", "llm_primary")
    monkeypatch.setattr("messengers_router.nlu_pipeline._analyze_llm_primary_with_candidates", fake_llm_primary)
    monkeypatch.setattr("messengers_router.nlu_pipeline._analyze_legacy_with_candidates", fake_legacy)

    state = SessionState(session_id="strict")
    out = run(analyze_with_candidates("привет", state, normalize_runtime_options(llm_mode="strict")))

    assert out.decision.source == "fallback"


def test_structured_slot_request_recovery():
    decision = RouteDecision(
        label="PRICE",
        confidence=0.72,
        source="llm_primary",
        clarify_needed=True,
        clarify_reason="slot_request",
        clarify_slots=["service_name"],
    )

    recovery = evaluate_recovery(
        user_text="цена",
        decision=decision,
        flow_label="PRICE",
        pending_exists=False,
        flow_active=False,
        state_entities={},
        summary="",
        max_unclear=3,
    )

    assert recovery.kind == "clarify"
    assert "название услуги" in recovery.text.lower()


def test_debug_meta_contains_nlu_fields():
    decision = RouteDecision(
        label="PRICE",
        confidence=0.81,
        source="llm_primary",
        clarify_needed=True,
        clarify_reason="slot_request",
        clarify_slots=["service_name"],
        intent_candidates=["PRICE", "APPOINTMENT"],
    )
    evidence = Evidence(debug_trace=[{"nlu": {"nlu_trace": {"final_decision": {"label": "PRICE"}}}}])
    meta = _debug_meta(decision, Plan(label="PRICE"), evidence, SessionState(session_id="dbg"), None)

    assert meta["decision"]["source"] == "llm_primary"
    assert meta["decision"]["clarify_needed"] is True
    assert meta["decision"]["clarify_reason"] == "slot_request"
    assert meta["decision"]["clarify_slots"] == ["service_name"]
    assert meta["decision"]["intent_candidates"] == ["PRICE", "APPOINTMENT"]
    assert meta["nlu_trace"]["final_decision"]["label"] == "PRICE"


def test_secondary_intents_preserve_llm_metadata():
    base = RouteDecision(
        label="PRICE",
        confidence=0.76,
        entities={"service_name": "УЗИ брюшной полости"},
        flags={"llm_primary"},
        source="llm_primary",
        clarify_needed=True,
        clarify_reason="slot_request",
        clarify_slots=["city"],
        intent_candidates=["PRICE", "APPOINTMENT"],
    )

    out = classifier._attach_secondary_intents("сколько стоит узи и как записаться", base, {})

    assert out.source == "llm_primary"
    assert out.clarify_needed is True
    assert out.clarify_reason == "slot_request"
    assert out.clarify_slots == ["city"]
    assert out.intent_candidates == ["PRICE", "APPOINTMENT"]
