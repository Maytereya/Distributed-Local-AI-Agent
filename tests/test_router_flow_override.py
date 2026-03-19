import asyncio

from messengers_router.flow_policy import _apply_pending_override, _hydrate_appointment_context_from_schedule
from messengers_router.mess_types import Evidence, Plan, PlanStep, SessionState
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision
from messengers_router.nlu_pipeline import NLUCandidate, NLUResult
from messengers_router.services import Services
from messengers_router.router import (
    _DEFAULT_CITY,
    _build_address_response,
    _build_appointment_schedule_preview_response,
    _build_appointment_step_response,
    build_plan,
    _build_doctor_info_response,
    _build_doctor_schedule_response,
    _build_price_response,
    _build_test_result_response,
    _should_keep_appointment_flow_override,
    execute_plan,
)
from messengers_router import router as router_mod


def test_appointment_flow_override_allows_city_datetime_and_fio():
    assert _should_keep_appointment_flow_override("Самара") is True
    assert _should_keep_appointment_flow_override("на 16:30") is True
    assert _should_keep_appointment_flow_override("Рахманов Владимир Александрович") is True


def test_appointment_flow_override_blocks_new_topics():
    assert _should_keep_appointment_flow_override("Как можно сдать анализы") is False
    assert _should_keep_appointment_flow_override("результаты анализов") is False
    assert _should_keep_appointment_flow_override("покажи расписание Казакова") is False


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
    _hydrate_appointment_context_from_schedule(state, payload)
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
    _hydrate_appointment_context_from_schedule(state, payload)
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

    label = _apply_pending_override(decision, pending, user_text="Самара")

    assert label == "PRICE"


def test_apply_pending_override_allows_address_switch_on_explicit_address_request():
    decision = RouteDecision(label="ADDRESS", confidence=0.72, flags={"rule_address"})
    pending = {"label": "PRICE", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = _apply_pending_override(decision, pending, user_text="адрес в Самаре")

    assert label == "ADDRESS"


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
