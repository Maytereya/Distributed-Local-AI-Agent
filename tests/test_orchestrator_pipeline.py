import asyncio

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, Plan, ResponseEnvelope, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUCandidate, NLUResult
from messengers_router.orchestrator import (
    OrchestratorContext,
    clarify_gate,
    doctor_entity_guard,
    early_guards,
    render,
    run_pipeline,
    tool_loop,
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


def test_doctor_entity_guard_verifies_decision_before_tool_loop(monkeypatch):
    async def fake_verify(decision, services, user_text):
        assert decision.label == "APPOINTMENT"
        assert user_text == "к дразнину"
        assert services is not None
        return RouteDecision(
            label=decision.label,
            confidence=decision.confidence,
            entities={"doctor_name": "Дразнин Антон Владимирович"},
            flags={"doctor_name_verified"},
            needs_handoff=False,
            context_action=decision.context_action,
            source=decision.source,
        )

    monkeypatch.setattr("messengers_router.router._verify_doctor_entity", fake_verify)

    ctx = OrchestratorContext(
        text="к дразнину",
        state=SessionState(session_id="doctor-guard"),
        decision=RouteDecision(
            label="APPOINTMENT",
            confidence=0.81,
            entities={"doctor_name": "Дразнину"},
            flags={"doctor_name_unverified"},
            source="llm_primary",
        ),
    )

    out = run(doctor_entity_guard(ctx, services=Services()))

    assert out.decision is not None
    assert out.decision.entities["doctor_name"] == "Дразнин Антон Владимирович"
    assert "doctor_name_verified" in out.decision.flags


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


def test_tool_loop_uses_post_nlu_helper_without_legacy_route_call(monkeypatch):
    import messengers_router.router as router_mod

    decision = RouteDecision(label="PRICE", confidence=0.91, source="llm_primary")
    plan = Plan(label="PRICE")
    evidence = Evidence(items={"price": {"prices": []}})

    async def _no_pending(**kw):
        _ = kw
        return None

    async def fake_complete_route(**kwargs):
        assert kwargs["decision"] is decision
        return decision, plan, evidence

    async def fail_route_patient_message(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("tool_loop should use post-nlu helper, not full route_patient_message")

    monkeypatch.setattr(router_mod, "_handle_catalog_confirm_pending", _no_pending)
    monkeypatch.setattr(router_mod, "_handle_appointment_action_pending", _no_pending)
    monkeypatch.setattr(router_mod, "_handle_compound_price_pending", _no_pending)
    monkeypatch.setattr(router_mod, "_complete_route_after_doctor_guard", fake_complete_route)
    monkeypatch.setattr(router_mod, "route_patient_message", fail_route_patient_message)

    ctx = OrchestratorContext(
        text="цена",
        state=SessionState(session_id="tool-loop-post-nlu"),
        decision=decision,
    )

    out = run(tool_loop(ctx, services=Services(), memory=MemoryStore()))

    assert out.decision is decision
    assert out.plan is plan
    assert out.evidence is evidence


def test_tool_loop_pending_handler_short_circuits_before_legacy_route(monkeypatch):
    """When a pending handler fires, tool_loop short-circuits and skips route_patient_message.

    Tests tool_loop directly so we don't go through render() and hit the LLM.
    """

    import messengers_router.router as router_mod

    pending_decision = RouteDecision(
        label="PRICE",
        confidence=0.95,
        entities={"service_name": "ТТГ"},
        source="compound_price",
    )
    pending_plan = Plan(label="PRICE")
    pending_evidence = Evidence(items={"payload": "compound_ok"})

    async def fake_compound_price_handler(**kw):
        return pending_decision, pending_plan, pending_evidence

    async def _no_pending(**kw):
        return None

    legacy_called: list = []

    async def fake_route_patient_message(*args, **kwargs):
        legacy_called.append(True)
        return pending_decision, pending_plan, pending_evidence

    monkeypatch.setattr(router_mod, "_handle_catalog_confirm_pending", _no_pending)
    monkeypatch.setattr(router_mod, "_handle_appointment_action_pending", _no_pending)
    monkeypatch.setattr(router_mod, "_handle_compound_price_pending", fake_compound_price_handler)
    monkeypatch.setattr(router_mod, "route_patient_message", fake_route_patient_message)

    state = SessionState(session_id="pipeline-pending-gate")
    # Prime ctx with an NLU decision (simulates post-nlu_route state)
    ctx = OrchestratorContext(
        text="да",
        state=state,
        decision=RouteDecision(label="PRICE", confidence=0.84, source="llm_primary"),
    )

    out = run(tool_loop(ctx, services=Services(), memory=MemoryStore()))

    # Legacy route must NOT have been called — pending handler short-circuited
    assert legacy_called == []
    assert out.short_circuit is True
    assert out.short_circuit_reason == "pending_handler"
    assert out.decision is pending_decision
    assert out.plan is pending_plan
    assert out.evidence is pending_evidence


def test_render_uses_prebuilt_structured_response_and_sets_secondary_offer_pending():
    state = SessionState(
        session_id="render-prebuilt",
        last_entities={"_secondary_queue": ["DOCTOR_SCHEDULE"]},
    )
    ctx = OrchestratorContext(
        text="Какие кардиологи принимают?",
        state=state,
        decision=RouteDecision(label="DOCTOR_INFO", confidence=0.93),
        plan=Plan(label="DOCTOR_INFO"),
        evidence=Evidence(
            items={
                "doctors_info": {
                    "doctors": [
                        {
                            "fio": "Хальметова Алина Алексеевна",
                            "specialization": "Кардиолог",
                            "regions": ["г. Самара, пр. Ленина, 5"],
                        }
                    ]
                }
            }
        ),
    )

    out = run(render(ctx, services=Services(), memory=MemoryStore()))

    assert out.response is not None
    assert "Хальметова" in out.response.text
    assert state.last_entities.get("_secondary_offer_pending") is True


def test_render_collects_stream_into_response_envelope(monkeypatch):
    async def fake_render_stream(user_text, decision, evidence, runtime_options=None):
        _ = user_text, decision, evidence, runtime_options
        yield "hello "
        yield "world"

    monkeypatch.setattr("messengers_router.renderer.render_stream", fake_render_stream)

    ctx = OrchestratorContext(
        text="цена",
        state=SessionState(session_id="render-stream"),
        decision=RouteDecision(label="PRICE", confidence=0.93, needs_handoff=False),
        plan=Plan(label="PRICE"),
        evidence=Evidence(items={"payload": "ok", "attachments": [{"type": "pdf", "name": "memo"}]}),
    )

    out = run(render(ctx, services=Services(), memory=MemoryStore()))

    assert out.response == ResponseEnvelope(
        text="hello world",
        attachments=[{"type": "pdf", "name": "memo"}],
        handoff=False,
    )


def test_render_prioritizes_pending_clarification_before_appointment_step_response():
    state = SessionState(
        session_id="render-pending-clarify",
        last_entities={"appointment_flow_active": True, "appointment_action": "reschedule"},
    )
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["_any_of:doctor_id,doctor_name"])
    ctx = OrchestratorContext(
        text="не знаю фамилию",
        state=state,
        decision=RouteDecision(
            label="OTHER",
            confidence=0.4,
            flags={"low_confidence"},
            needs_handoff=False,
        ),
        plan=Plan(label="APPOINTMENT"),
        evidence=Evidence(items={}),
    )

    out = run(render(ctx, services=Services(), memory=memory))

    assert out.response is not None
    assert "фио врача" in out.response.text.lower()
