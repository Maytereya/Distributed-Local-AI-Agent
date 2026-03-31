import asyncio

from messengers_router.flow_policy import (
    apply_context_action,
    apply_pending_override,
    hydrate_appointment_context_from_schedule,
)
from messengers_router.mess_types import Evidence, Plan, PlanStep, SessionState
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision
from messengers_router.nlu_pipeline import NLUCandidate, NLUResult
from messengers_router.policies import (
    quick_fill_core_entities,
    extract_branch_hint,
    appointment_service_display,
    appointment_confirmation_transition,
    service_name_conflicts_with_doctor,
    detect_nonbookable_walkin_intent,
    nonbookable_service_hint,
    detect_prepare_intent,
)
from messengers_router.services import Services
from messengers_router import classifier as classifier_mod
from messengers_router.city import match_city
from messengers_router.router import (
    _DEFAULT_CITY,
    _apply_appointment_continuity_overrides,
    _build_first_structured_response,
    _build_address_response,
    _build_appointment_schedule_preview_response,
    _build_appointment_step_response,
    build_plan,
    _build_doctor_info_response,
    _build_doctor_schedule_response,
    _build_price_response,
    _build_test_result_response,
    _should_keep_appointment_flow_override,
    _verify_doctor_entity,
    execute_plan,
)
from messengers_router import router as router_mod


def _run_stream_once(user_text: str, state: SessionState, services: Services, memory: MemoryStore):
    async def _collect():
        out = []
        async for env in router_mod.patient_routing_stream(user_text, state, services, memory):
            out.append(env)
        return out

    return asyncio.run(_collect())


def test_appointment_flow_override_allows_city_datetime_and_fio():
    assert _should_keep_appointment_flow_override("Самара") is True
    assert _should_keep_appointment_flow_override("на 16:30") is True
    assert _should_keep_appointment_flow_override("Рахманов Владимир Александрович") is True


def test_appointment_flow_override_blocks_new_topics():
    assert _should_keep_appointment_flow_override("Как можно сдать анализы") is False
    assert _should_keep_appointment_flow_override("результаты анализов") is False
    assert _should_keep_appointment_flow_override("покажи расписание Казакова") is False


def test_apply_appointment_continuity_overrides_prioritizes_datetime():
    state = SessionState(session_id="appt-override-datetime", last_entities={"appointment_flow_active": True})
    decision = RouteDecision(
        label="DOCTOR_SCHEDULE",
        confidence=0.35,
        entities={},
        flags={"low_confidence"},
        needs_handoff=True,
        context_action="new_topic",
    )

    out = _apply_appointment_continuity_overrides(decision, state, "на завтра на 9:00")

    assert out.label == "APPOINTMENT"
    assert "flow_datetime_appointment_override" in out.flags
    assert out.context_action == "continue"
    assert out.needs_handoff is False


def test_apply_appointment_continuity_overrides_keeps_other_followup():
    state = SessionState(session_id="appt-override-other", last_entities={"appointment_flow_active": True})
    decision = RouteDecision(
        label="OTHER",
        confidence=0.3,
        entities={},
        flags={"low_confidence"},
        needs_handoff=False,
        context_action="continue",
    )

    out = _apply_appointment_continuity_overrides(decision, state, "Самара")

    assert out.label == "APPOINTMENT"
    assert "flow_appointment_override" in out.flags


def test_apply_appointment_continuity_overrides_keeps_address_like_branch_reply():
    state = SessionState(session_id="appt-override-address", last_entities={"appointment_flow_active": True})
    decision = RouteDecision(
        label="ADDRESS",
        confidence=0.72,
        entities={},
        flags={"rule_address"},
        needs_handoff=False,
        context_action="continue",
    )

    out = _apply_appointment_continuity_overrides(decision, state, "г. Самара, ул. Победы, 83")

    assert out.label == "APPOINTMENT"
    assert "flow_appointment_override" in out.flags


def test_apply_appointment_continuity_overrides_allows_explicit_address_topic_switch():
    state = SessionState(session_id="appt-override-address-topic", last_entities={"appointment_flow_active": True})
    decision = RouteDecision(
        label="ADDRESS",
        confidence=0.72,
        entities={},
        flags={"rule_address"},
        needs_handoff=False,
        context_action="continue",
    )

    out = _apply_appointment_continuity_overrides(decision, state, "адрес в Самаре")

    assert out.label == "ADDRESS"


def test_service_name_conflicts_with_doctor_detects_surname_case():
    assert service_name_conflicts_with_doctor("Дразнину", "Дразнин Антон Владимирович") is True
    assert service_name_conflicts_with_doctor("Холтер", "Дразнин Антон Владимирович") is False


def test_appointment_service_display_ignores_doctor_like_service_name():
    text = appointment_service_display(
        {
            "service_name": "Дразнину",
            "doctor_name": "Дразнин Антон Владимирович",
        }
    )

    assert text == "приём к врачу Дразнин Антон Владимирович"


def test_verify_doctor_entity_drops_service_name_that_matches_doctor():
    class _FakeServices:
        async def resolve_doctor_name(self, raw_text_or_name: str) -> str | None:
            low = str(raw_text_or_name or "").lower()
            if "дразнин" in low or "дразнину" in low:
                return "Дразнин Антон Владимирович"
            return None

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.75,
        entities={"doctor_name": "Дразнину", "service_name": "Дразнину"},
        flags={"rule_appointment"},
        needs_handoff=False,
    )

    out = asyncio.run(_verify_doctor_entity(decision, _FakeServices(), "Записаться к Дразнину"))

    assert out.entities.get("doctor_name") == "Дразнин Антон Владимирович"
    assert "service_name" not in out.entities
    assert "entity_dropped_doctor_like_service_name" in out.flags


def test_appointment_confirmation_transition_accepts_common_yes_forms():
    for text in ("да", "Да", "Да,", "Да?", "подтверждаю", "Подтверждаю"):
        assert appointment_confirmation_transition(text) == "yes"


def test_appointment_confirmation_transition_accepts_soft_yes_forms():
    for text in ("хорошо", "Ладно", "хорошо, спасибо"):
        assert appointment_confirmation_transition(text) == "yes"


def test_appointment_confirmation_transition_accepts_common_no_forms():
    for text in ("нет", "Нет", "неа", "не правильно", "неправильно", "не подтверждаю"):
        assert appointment_confirmation_transition(text) == "no"


def test_default_city_is_samara_for_messenger_router():
    assert _DEFAULT_CITY == "Самара"


def test_hydrate_schedule_sets_branch_when_windows_have_single_branch():
    state = SessionState(session_id="t1")
    payload = {
        "schedule": [
            {
                "fio": "Трубин Алексей Юрьевич",
                "regions": ["Ново-Садовая 106", "Ленина 5"],
                "schedule": {
                    "Ленина 5": [
                        {"date": "2026-03-03", "slots": ["13:00", "13:30"]},
                        {"date": "2026-03-04", "slots": ["15:00"]},
                    ]
                },
            }
        ]
    }
    hydrate_appointment_context_from_schedule(state, payload)
    assert state.last_entities.get("branch_name") == "Ленина 5"


def test_hydrate_schedule_prefers_branches_from_actual_windows():
    state = SessionState(session_id="t2")
    payload = {
        "schedule": [
            {
                "fio": "Трубин Алексей Юрьевич",
                "regions": ["Ново-Садовая 106", "Ленина 5", "Победа 83"],
                "schedule": {
                    "Ленина 5": [{"date": "2026-03-03", "slots": ["13:00"]}],
                    "Ново-Садовая 106": [{"date": "2026-03-04", "slots": ["15:00"]}],
                },
            }
        ]
    }
    hydrate_appointment_context_from_schedule(state, payload)
    assert state.last_entities.get("appointment_branch_options") == ["Ленина 5", "Ново-Садовая 106"]
    assert not state.last_entities.get("branch_name")


def test_route_message_keeps_nlu_debug_out_of_state(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(label="OTHER", confidence=0.4, entities={}, flags=set(), needs_handoff=False),
            candidates=[
                NLUCandidate(source="rule", label="OTHER", confidence=0.4, entities={}, flags=["rule_none"]),
                NLUCandidate(source="llm", label="OTHER", confidence=0.2, entities={}, flags=["low_confidence"]),
            ],
            merged_from="rule_promoted",
        )

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="nlu-debug-clean",
        last_entities={
            "_nlu_candidates": [{"stale": True}],
            "_nlu_merged_from": "stale",
            "_nlu_shadow": {"stale": True},
        },
    )
    services = Services()
    memory = MemoryStore()

    _decision, _plan, evidence = asyncio.run(
        router_mod.route_patient_message(
            "привет",
            state,
            services,
            memory,
        )
    )

    assert "_nlu_candidates" not in state.last_entities
    assert "_nlu_merged_from" not in state.last_entities
    assert "_nlu_shadow" not in state.last_entities
    assert any(isinstance(t, dict) and "nlu" in t for t in evidence.debug_trace)


def test_route_message_promotes_profile_followup_to_nonbookable_address(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="TEST_ASSIST",
                confidence=0.72,
                entities={},
                flags={"rule_test_assist"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(
        session_id="walkin-profile-followup",
        last_entities={"test_goal": "Какие анализы сдать на сахарный диабет"},
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Диабетический профиль 1 где можно сдать?",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "ADDRESS"
    assert "policy_nonbookable_walkin" in decision.flags
    assert plan.label == "ADDRESS"
    assert str(state.last_entities.get("service_name") or "").lower() == "анализы"


def test_route_message_secondary_offer_accepts_thanks_as_soft_yes():
    state = SessionState(
        session_id="secondary-thanks-yes",
        last_entities={
            "_secondary_offer_pending": True,
            "_secondary_queue": ["ADDRESS"],
            "secondary_intents": ["ADDRESS"],
        },
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "спасибо",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "ADDRESS"
    assert decision.entities.get("secondary_intent_from_queue") is True
    assert plan.label == "ADDRESS"
    assert state.last_entities.get("_secondary_offer_pending") is False


def test_build_price_response_for_price_flow():
    state = SessionState(session_id="price", last_entities={"service_name": "УЗИ брюшной полости"})
    evidence = Evidence(items={"price": {"prices": [{"serviceName": "УЗИ брюшной полости", "cost": 1500.0}]}})

    env = _build_price_response("PRICE", evidence, state)

    assert env is not None
    assert "1 500 руб." in env.text
    assert env.handoff is False


def test_build_plan_price_service_uses_service_bundle_tool():
    state = SessionState(
        session_id="price-bundle",
        last_entities={"service_name": "УЗИ брюшной полости", "city": "Самара"},
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="PRICE",
        confidence=0.8,
        entities={"service_name": "УЗИ брюшной полости"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "Сколько стоит УЗИ брюшной полости?", memory)

    assert plan.steps
    assert plan.steps[0].tool == "service_bundle_info"


def test_build_plan_price_with_doctor_uses_price_info():
    state = SessionState(
        session_id="price-doctor",
        last_entities={"service_name": "УЗИ брюшной полости", "doctor_name": "Иванов"},
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="PRICE",
        confidence=0.8,
        entities={"service_name": "УЗИ брюшной полости", "doctor_name": "Иванов"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "Цена УЗИ у Иванова", memory)

    assert plan.steps
    assert plan.steps[0].tool == "price_info"


def test_build_plan_doc_request_uses_main_index_info():
    state = SessionState(session_id="doc-request", last_entities={})
    memory = MemoryStore()
    decision = RouteDecision(
        label="OTHER",
        confidence=0.85,
        entities={},
        flags={"doc_request_main_index"},
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "Как получить справку для налоговой?", memory)

    assert plan.steps
    assert plan.steps[0].tool == "main_index_info"


def test_build_first_structured_response_keeps_builder_priority():
    state = SessionState(session_id="builder-priority", last_entities={})
    evidence = Evidence(
        items={
            "main_index_info": {"content": "Справка для налоговой"},
            "price": {"prices": [{"serviceName": "УЗИ", "cost": 1500}]},
        }
    )
    services = Services()
    memory = MemoryStore()
    decision = RouteDecision(label="PRICE", confidence=0.9, entities={}, flags=set(), needs_handoff=False)

    env = _build_first_structured_response(
        flow_label="PRICE",
        evidence=evidence,
        state=state,
        services=services,
        memory=memory,
        decision=decision,
        user_text="Сколько стоит УЗИ?",
    )

    assert env is not None
    assert env.text == "Справка для налоговой"


def test_build_first_structured_response_returns_prepare_payload_for_prepare_flow():
    state = SessionState(session_id="builder-prepare", last_entities={})
    evidence = Evidence(items={"prepare": {"prepare": "Сдавайте анализ натощак, воду пить можно."}})
    services = Services()
    memory = MemoryStore()
    decision = RouteDecision(label="PREPARE", confidence=0.9, entities={}, flags=set(), needs_handoff=False)

    env = _build_first_structured_response(
        flow_label="PREPARE",
        evidence=evidence,
        state=state,
        services=services,
        memory=memory,
        decision=decision,
        user_text="Как подготовиться к анализу на холестерин?",
    )

    assert env is not None
    assert "натощак" in env.text.lower()
    assert env.handoff is False


def test_build_plan_legacy_doc_request_handoff_flag_also_uses_main_index_info():
    state = SessionState(session_id="doc-request-legacy", last_entities={})
    memory = MemoryStore()
    decision = RouteDecision(
        label="OTHER",
        confidence=0.85,
        entities={},
        flags={"doc_request_handoff"},
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "Нужна справка для ФНС", memory)

    assert plan.steps
    assert plan.steps[0].tool == "main_index_info"


def test_build_plan_appointment_with_cached_windows_skips_realtime_schedule():
    state = SessionState(
        session_id="appt-cached-windows",
        last_entities={
            "doctor_name": "Дразнин",
            "appointment_windows": [{"branch": "Ленина 5", "date": "2026-03-21", "time": "16:30"}],
            "branch_name": "Ленина 5",
        },
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"doctor_name": "Дразнин"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "на завтра на 16:30", memory)

    assert plan.steps == []


def test_build_plan_appointment_with_selected_datetime_skips_realtime_schedule():
    state = SessionState(
        session_id="appt-selected-datetime",
        last_entities={
            "doctor_name": "Дразнин",
            "branch_name": "Ленина 5",
            "date_from": "2026-03-21",
            "time_from": "16:30",
        },
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"doctor_name": "Дразнин"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "Иванов Иван Иванович", memory)

    assert plan.steps == []


def test_build_plan_appointment_active_flow_makes_schedule_optional():
    state = SessionState(
        session_id="appt-flow-schedule-optional",
        last_entities={
            "doctor_name": "Дразнин",
            "appointment_flow_active": True,
        },
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"doctor_name": "Дразнин"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "на завтра после 16:00", memory)

    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "doctors_schedule_week"
    assert plan.steps[0].required is False


def test_build_plan_appointment_active_flow_makes_address_optional():
    state = SessionState(
        session_id="appt-flow-address-optional",
        last_entities={
            "appointment_flow_active": True,
            "city": "Самара",
            "service_name": "холтер",
        },
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "холтер"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "Самара", memory)

    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "address_info"
    assert plan.steps[0].required is False
    assert plan.steps[1].tool == "price_info"
    assert plan.steps[1].required is False


def test_execute_plan_optional_step_handoff_is_suppressed():
    class _Svc:
        async def price_info(self, _q, _e):
            return {
                "handoff_required": True,
                "handoff_reason": "service_error",
                "handoff_message": "fallback",
                "note": "price unavailable",
                "prices": [],
            }

    plan = Plan(
        label="APPOINTMENT",
        steps=[PlanStep(tool="price_info", input={"query": "Самара", "entities": {}}, required=False)],
    )

    ev = asyncio.run(execute_plan(plan, SessionState(session_id="opt-suppress"), _Svc()))

    assert ev.get("handoff_required") is None
    assert ev.get("price") is None
    suppressed = ev.get("price_info_optional_suppressed")
    assert isinstance(suppressed, dict)
    assert suppressed.get("handoff_reason") == "service_error"


def test_execute_plan_optional_step_exception_is_suppressed():
    class _Svc:
        async def price_info(self, _q, _e):
            raise RuntimeError("boom")

    plan = Plan(
        label="APPOINTMENT",
        steps=[PlanStep(tool="price_info", input={"query": "Самара", "entities": {}}, required=False)],
    )

    ev = asyncio.run(execute_plan(plan, SessionState(session_id="opt-exc"), _Svc()))

    assert ev.get("handoff_required") is None
    optional_err = ev.get("price_info_optional_error")
    assert isinstance(optional_err, dict)
    assert "boom" in str(optional_err.get("message"))


def test_merge_entities_keeps_appointment_context_for_same_doctor_short_name():
    memory = MemoryStore()
    state = SessionState(
        session_id="merge-doctor-same",
        last_entities={
            "doctor_name": "Дразнин Антон Владимирович",
            "appointment_flow_active": True,
            "appointment_windows": [{"date": "2026-03-20", "time": "16:30", "branch": "Ленина 5"}],
            "branch_name": "Ленина 5",
        },
    )

    memory.merge_entities(state, {"doctor_name": "Дразнин"}, label="APPOINTMENT")

    assert state.last_entities.get("doctor_name") == "Дразнин Антон Владимирович"
    assert state.last_entities.get("appointment_flow_active") is True
    assert state.last_entities.get("appointment_windows")


def test_merge_entities_resets_appointment_context_for_different_doctor():
    memory = MemoryStore()
    state = SessionState(
        session_id="merge-doctor-different",
        last_entities={
            "doctor_name": "Дразнин Антон Владимирович",
            "appointment_flow_active": True,
            "appointment_windows": [{"date": "2026-03-20", "time": "16:30", "branch": "Ленина 5"}],
            "branch_name": "Ленина 5",
        },
    )

    memory.merge_entities(state, {"doctor_name": "Хальметова Алина Алексеевна"}, label="APPOINTMENT")

    assert state.last_entities.get("doctor_name") == "Хальметова Алина Алексеевна"
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("appointment_windows") is None


def test_quick_fill_patient_name_skips_datetime_phrase():
    out = quick_fill_core_entities("на завтра на 9:00", {}, ["patient_name"])
    assert "patient_name" not in out
    assert out.get("time_from") == "09:00"


def test_quick_fill_patient_name_accepts_real_fio():
    out = quick_fill_core_entities("Рахманов Владимир", {}, ["patient_name"])
    assert out.get("patient_name") == "Рахманов Владимир"


def test_build_doctor_info_response_for_doctor_info_flow():
    state = SessionState(session_id="doc-info", last_entities={})
    evidence = Evidence(
        items={
            "doctors_info": {
                "doctors": [
                    {
                        "fio": "Иванов Иван Иванович",
                        "specialization": "Терапевт",
                        "regions": ["г. Самара, пр. Ленина, 5"],
                    }
                ]
            }
        }
    )

    env = _build_doctor_info_response("DOCTOR_INFO", evidence, state)

    assert env is not None
    assert "Иванов Иван Иванович" in env.text
    assert env.handoff is False


def test_build_doctor_info_response_appends_price_block():
    state = SessionState(session_id="doc-info-price", last_entities={"doctor_name": "Иванов"})
    evidence = Evidence(
        items={
            "doctors_info": {
                "doctors": [
                    {
                        "fio": "Иванов Иван Иванович",
                        "specialization": "Терапевт",
                        "regions": ["г. Самара, пр. Ленина, 5"],
                    }
                ]
            },
            "price": {
                "prices": [
                    {
                        "doctorId": 11,
                        "fio": "Иванов Иван Иванович",
                        "serviceName": "Прием врача",
                        "cost": 1200,
                    }
                ]
            },
        }
    )

    env = _build_doctor_info_response("DOCTOR_INFO", evidence, state)

    assert env is not None
    assert "Примеры стоимости услуг этого врача" in env.text
    assert "1 200 руб." in env.text


def test_build_test_result_response_with_ready_link():
    evidence = Evidence(
        items={
            "test_result_status": {
                "ready": True,
                "note": "result_link_constructed",
                "result_links": ["https://example.com/result.pdf"],
            }
        }
    )
    env = _build_test_result_response("TEST_RESULT", evidence)
    assert env is not None
    assert "Ссылка на результат" in env.text
    assert "https://example.com/result.pdf" in env.text


def test_build_doctor_schedule_response_sets_flow_active():
    state = SessionState(session_id="doc-schedule", last_entities={})
    evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [
                    {
                        "fio": "Трубин Алексей Юрьевич",
                        "regions": ["Ленина 5"],
                        "schedule": {"Ленина 5": [{"date": "2026-03-09", "slots": ["10:00"]}]},
                    }
                ]
            }
        }
    )
    env = _build_doctor_schedule_response("DOCTOR_SCHEDULE", evidence, state)
    assert env is not None
    assert state.last_entities.get("appointment_flow_active") is True
    assert "Трубин Алексей Юрьевич" in env.text


def test_build_address_response_sets_pending_when_empty():
    state = SessionState(session_id="addr", last_entities={"city": "Самара"})
    memory = MemoryStore()
    decision = RouteDecision(label="ADDRESS", flags=set())
    evidence = Evidence(items={"address": {"branches": [], "addresses": []}})

    env = _build_address_response("ADDRESS", evidence, state, memory, decision, "адрес")

    assert env is not None
    assert "Не нашёл филиалы" in env.text
    assert state.last_entities.get("city") is None
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "ADDRESS"


def test_build_appointment_schedule_preview_response():
    state = SessionState(session_id="appt-preview", last_entities={"doctor_name": "Трубин"})
    evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [
                    {
                        "fio": "Трубин Алексей Юрьевич",
                        "regions": ["Ленина 5"],
                        "schedule": {"Ленина 5": [{"date": "2026-03-09", "slots": ["10:00"]}]},
                    }
                ]
            }
        }
    )
    env = _build_appointment_schedule_preview_response("APPOINTMENT", evidence, state)
    assert env is not None
    assert "Трубин Алексей Юрьевич" in env.text


def test_apply_pending_override_keeps_price_flow_on_city_reply():
    decision = RouteDecision(label="ADDRESS", confidence=0.72, flags={"rule_address"})
    pending = {"label": "PRICE", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="Самара")

    assert label == "PRICE"


def test_apply_pending_override_allows_address_switch_on_explicit_address_request():
    decision = RouteDecision(label="ADDRESS", confidence=0.72, flags={"rule_address"})
    pending = {"label": "PRICE", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="адрес в Самаре")

    assert label == "ADDRESS"


def test_apply_pending_override_keeps_appointment_on_full_branch_address_reply():
    decision = RouteDecision(label="ADDRESS", confidence=0.78, flags={"rule_nonbookable_walkin"})
    pending = {"label": "APPOINTMENT", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="г. Самара, ул. Победы, 83")

    assert label == "APPOINTMENT"


def test_apply_pending_override_keeps_appointment_on_street_without_house():
    decision = RouteDecision(label="ADDRESS", confidence=0.78, flags={"rule_nonbookable_walkin"})
    pending = {"label": "APPOINTMENT", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="на победе")

    assert label == "APPOINTMENT"


def test_apply_pending_override_allows_non_samara_city_switch():
    decision = RouteDecision(label="ADDRESS", confidence=0.78, flags={"rule_nonbookable_walkin"})
    pending = {"label": "APPOINTMENT", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(
        decision,
        pending,
        user_text="мне вообще не в Самаре а в Сызрани надо!!! Сызрань! Слышите?",
    )

    assert label == "ADDRESS"


def test_detect_nonbookable_walkin_intent_uses_test_context_for_profile_followup():
    ok = detect_nonbookable_walkin_intent(
        "Диабетический профиль 1 где можно сдать?",
        {"test_goal": "Какие анализы сдать на сахарный диабет"},
    )

    assert ok is True


def test_detect_nonbookable_walkin_intent_supports_typo_zdat_with_context():
    ok = detect_nonbookable_walkin_intent(
        "хорошо, где здать?",
        {"test_goal": "Какие анализы сдать на сахарный диабет"},
    )

    assert ok is True


def test_nonbookable_service_hint_uses_test_context_for_profile_followup():
    hint = nonbookable_service_hint(
        "Диабетический профиль 1 где можно сдать?",
        {"test_goal": "Какие анализы сдать на сахарный диабет"},
    )

    assert hint == "анализы"


def test_detect_prepare_intent_covers_time_of_day_questions():
    assert detect_prepare_intent("обязательно ли в первую половину дня сдавать анализ?") is True
    assert detect_prepare_intent("а можно вечером сдать анализ?") is True


def test_detect_nonbookable_walkin_intent_covers_how_to_submit_with_context():
    ok = detect_nonbookable_walkin_intent(
        "как сдать?",
        {"test_goal": "Какие анализы сдать на сахарный диабет"},
    )

    assert ok is True


def test_detect_nonbookable_walkin_intent_covers_where_and_how_to_submit_with_context():
    ok = detect_nonbookable_walkin_intent(
        "где и как сдать?",
        {"test_goal": "Какие анализы сдать на сахарный диабет"},
    )

    assert ok is True


def test_detect_prepare_intent_covers_fasting_and_prepare_questions():
    assert detect_prepare_intent("нужно ли натощак?") is True
    assert detect_prepare_intent("как подготовиться к анализу?") is True


def test_match_city_prefers_city_after_negation_switch():
    city = match_city("мне вообще не в Самаре а в Сызрани надо!!! Сызрань! Слышите?")
    assert str(city or "").lower().replace("ё", "е").startswith("сызран")


def test_extract_branch_hint_parses_full_address_reply():
    hint = extract_branch_hint("г. Самара, ул. Победы, 83", {"city": "Самара"})
    low = str(hint or "").lower()
    assert "побед" in low
    assert "83" in low


def test_deterministic_rule_uses_patient_name_when_pending_appointment():
    decision = asyncio.run(
        classifier_mod.deterministic_rule_decision(
            "Рахманов Владимир",
            {
                "appointment_flow_active": True,
                "_pending": {"label": "APPOINTMENT", "missing": ["patient_name"]},
            },
            allow_refine=False,
        )
    )

    assert decision is not None
    assert decision.label == "APPOINTMENT"
    assert decision.entities.get("patient_name") == "Рахманов Владимир"
    assert "rule_appointment_patient_name" in decision.flags


def test_apply_context_action_blocks_new_topic_on_patient_name_step():
    state = SessionState(
        session_id="appt-new-topic-block",
        last_entities={
            "appointment_flow_active": True,
            "_pending": {"label": "APPOINTMENT", "missing": ["patient_name"]},
            "doctor_name": "Хальметова Алина Алексеевна",
        },
    )
    decision = RouteDecision(
        label="OTHER",
        confidence=0.3,
        entities={"patient_name": "Рахманов Владимир"},
        flags={"low_confidence", "new_topic"},
        needs_handoff=False,
        context_action="new_topic",
    )

    out = apply_context_action(decision, state, "Рахманов Владимир")

    assert out.context_action == "continue"
    assert "context_action_new_topic_blocked_patient_name" in out.flags
    assert state.last_entities.get("_pending") is not None


def test_build_appointment_step_response_patient_step_sets_pending():
    state = SessionState(
        session_id="appt-step",
        last_entities={
            "branch_name": "Ленина 5",
            "date_from": "2026-03-09",
            "time_from": "10:00",
        },
    )
    evidence = Evidence(items={})
    memory = MemoryStore()
    services = Services()

    env = _build_appointment_step_response("APPOINTMENT", evidence, state, services, memory)

    assert env is not None
    assert "фио пациента" in env.text.lower()
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "APPOINTMENT"


def test_patient_routing_stream_requests_cancel_confirmation_for_active_appointment_flow():
    state = SessionState(session_id="appt-cancel-confirm", last_entities={"appointment_flow_active": True})
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["date_from", "time_from"])

    out = _run_stream_once("передумал, отменить запись", state, services, memory)

    assert len(out) == 1
    assert "Отменить текущий процесс записи" in out[0].text
    assert state.last_entities.get("appointment_cancel_pending") is True


def test_patient_routing_stream_cancel_rejected_resumes_appointment_flow():
    state = SessionState(
        session_id="appt-cancel-no",
        last_entities={"appointment_flow_active": True, "appointment_cancel_pending": True},
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    out = _run_stream_once("нет", state, services, memory)

    assert len(out) == 1
    assert "фио пациента" in out[0].text.lower()
    assert state.last_entities.get("appointment_cancel_pending") is None
    assert state.last_entities.get("appointment_flow_active") is True


def test_patient_routing_stream_topic_switch_requests_confirmation():
    state = SessionState(session_id="appt-topic-switch", last_entities={"appointment_flow_active": True})
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["date_from", "time_from"])

    out = _run_stream_once("Сколько стоит общий анализ крови?", state, services, memory)

    assert len(out) == 1
    assert "Отменить этот процесс и перейти к новому вопросу" in out[0].text
    assert state.last_entities.get("appointment_topic_switch_pending") is True


def test_patient_routing_stream_topic_switch_confirm_yes_clears_appointment_flow():
    state = SessionState(
        session_id="appt-topic-switch-yes",
        last_entities={
            "appointment_flow_active": True,
            "appointment_topic_switch_pending": True,
            "doctor_name": "Ким Татьяна Александровна",
            "date_from": "2026-03-20",
            "time_from": "09:00",
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    out = _run_stream_once("да", state, services, memory)

    assert len(out) == 1
    assert "Процесс записи отменён" in out[0].text
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("doctor_name") is None
    assert memory.get_pending(state) is None


def test_patient_routing_stream_manual_operator_clears_appointment_context():
    state = SessionState(
        session_id="appt-manual-operator",
        last_entities={
            "appointment_flow_active": True,
            "appointment_confirm_pending": True,
            "appointment_cancel_pending": True,
            "doctor_name": "Ким Татьяна Александровна",
            "service_name": "Прием врача",
            "date_from": "2026-03-20",
            "time_from": "09:00",
            "city": "Самара",
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    out = _run_stream_once("Соедините с оператором", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is True
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("appointment_confirm_pending") is None
    assert state.last_entities.get("appointment_cancel_pending") is None
    assert state.last_entities.get("doctor_name") is None
    assert state.last_entities.get("service_name") is None
    assert state.last_entities.get("date_from") is None
    assert state.last_entities.get("time_from") is None
    assert state.last_entities.get("city") == "Самара"
    assert memory.get_pending(state) is None
