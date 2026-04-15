import asyncio

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, Plan, ResponseEnvelope, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUCandidate, NLUResult
from messengers_router.orchestrator import (
    OrchestratorContext,
    clarify_gate,
    early_guards,
    run_pipeline,
)
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


def test_orchestrator_context_defaults():
    state = SessionState(session_id="orchestrator-defaults")

    ctx = OrchestratorContext(text="привет", state=state)

    assert ctx.text == "привет"
    assert ctx.state is state
    assert ctx.decision is None
    assert ctx.should_clarify is False
    assert ctx.clarify_text == ""
    assert ctx.tool_results == {}
    assert ctx.response is None
    assert ctx.short_circuit is False
    assert ctx.short_circuit_reason == ""


def test_early_guards_short_circuits_on_urgent(monkeypatch):
    async def fake_deterministic_rule_decision(*args, **kwargs):
        _ = args, kwargs
        return RouteDecision(
            label="URGENT",
            confidence=1.0,
            flags={"urgent"},
            needs_handoff=True,
            context_action="new_topic",
        )

    monkeypatch.setattr(
        "messengers_router.classifier.deterministic_rule_decision",
        fake_deterministic_rule_decision,
    )
    ctx = OrchestratorContext(text="болит грудь", state=SessionState(session_id="urgent"))

    out = run(early_guards(ctx))

    assert out.short_circuit is True
    assert out.short_circuit_reason == "safety"
    assert out.decision is not None
    assert out.decision.label == "URGENT"


def test_clarify_gate_marks_clarify_when_decision_requests_it():
    ctx = OrchestratorContext(
        text="цена",
        state=SessionState(session_id="clarify"),
        decision=RouteDecision(
            label="PRICE",
            confidence=0.71,
            clarify_needed=True,
            clarify_reason="Уточните услугу",
        ),
    )

    out = run(clarify_gate(ctx))

    assert out.should_clarify is True
    assert out.clarify_text == "Уточните услугу"


def test_run_pipeline_returns_context_with_mocked_nlu(monkeypatch):
    async def fake_deterministic_rule_decision(*args, **kwargs):
        _ = args, kwargs
        return None

    async def fake_analyze_with_candidates(text, state, runtime_options=None):
        _ = text, state, runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="PRICE",
                confidence=0.83,
                entities={"service_name": "УЗИ щитовидной железы"},
                flags={"llm_primary"},
                source="llm_primary",
            ),
            candidates=[NLUCandidate(source="llm_primary", label="PRICE", confidence=0.83)],
            merged_from="llm_primary",
        )

    monkeypatch.setattr(
        "messengers_router.classifier.deterministic_rule_decision",
        fake_deterministic_rule_decision,
    )
    monkeypatch.setattr(
        "messengers_router.nlu_pipeline.analyze_with_candidates",
        fake_analyze_with_candidates,
    )

    state = SessionState(session_id="pipeline")

    out = run(run_pipeline("сколько стоит узи щитовидки", state))

    assert isinstance(out, OrchestratorContext)
    assert out.decision is not None
    assert out.decision.label == "PRICE"
    assert out.response == ResponseEnvelope(text="")
    assert state.dialog.label == "PRICE"
    assert state.dialog.entities == {"service_name": "УЗИ щитовидной железы"}


def test_run_pipeline_delegates_legacy_route_inside_orchestrator(monkeypatch):
    async def fake_deterministic_rule_decision(*args, **kwargs):
        _ = args, kwargs
        return None

    async def fake_analyze_with_candidates(text, state, runtime_options=None):
        _ = text, state, runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="PRICE",
                confidence=0.84,
                flags={"llm_primary"},
                source="llm_primary",
            ),
            candidates=[NLUCandidate(source="llm_primary", label="PRICE", confidence=0.84)],
            merged_from="llm_primary",
        )

    async def fake_route_patient_message(text, state, services, memory, runtime_options=None):
        _ = text, state, services, memory, runtime_options
        return (
            RouteDecision(label="PRICE", confidence=0.9, source="legacy_router"),
            Plan(label="PRICE"),
            Evidence(items={"payload": "ok"}),
        )

    monkeypatch.setattr(
        "messengers_router.classifier.deterministic_rule_decision",
        fake_deterministic_rule_decision,
    )
    monkeypatch.setattr(
        "messengers_router.nlu_pipeline.analyze_with_candidates",
        fake_analyze_with_candidates,
    )
    monkeypatch.setattr(
        "messengers_router.router.route_patient_message",
        fake_route_patient_message,
    )

    state = SessionState(session_id="pipeline-legacy-bridge")
    services = Services()
    memory = MemoryStore()

    out = run(run_pipeline("сколько стоит узи", state, services=services, memory=memory))

    assert out.decision is not None
    assert out.decision.source == "legacy_router"
    assert out.plan == Plan(label="PRICE")
    assert out.evidence == Evidence(items={"payload": "ok"})
    assert out.response is None
