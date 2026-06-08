import asyncio
import logging
import pytest

from messengers_router.flow_policy import (
    apply_context_action,
    apply_pending_override,
    clear_on_appointment_end,
    clear_on_handoff,
    clear_on_topic_switch,
    hydrate_appointment_context_from_schedule,
    quick_fill_entities_from_text,
)
from messengers_router.mess_types import (
    AppointmentPhase,
    DialogState,
    Evidence,
    Plan,
    PlanStep,
    ResponseEnvelope,
    SessionState,
)
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision
from messengers_router.appointment_flow_guard import is_new_topic_while_confirm_pending
from messengers_router.flow_policy import reset_appointment_runtime_state
from messengers_router.orchestrator import OrchestratorContext
from messengers_router.nlu_pipeline import NLUCandidate, NLUResult
from messengers_router.policies import (
    quick_fill_core_entities,
    extract_branch_hint,
    appointment_service_display,
    appointment_step_policy,
    appointment_summary,
    appointment_confirmation_transition,
    clarification_question,
    missing_slots,
    service_name_conflicts_with_doctor,
    detect_nonbookable_walkin_intent,
    nonbookable_service_hint,
    detect_prepare_intent,
    detect_unsupported_catalog,
    is_test_assist_category_term,
)
from messengers_router.services import Services
from messengers_router.entity_grounder import ground_decision_entities
from messengers_router import classifier as classifier_mod
from messengers_router import doctor_name_port as doctor_name_port_mod
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
from messengers_router import flow_policy as flow_policy_mod


def _run_stream_once(user_text: str, state: SessionState, services: Services, memory: MemoryStore):
    async def _collect():
        out = []
        async for env in router_mod.patient_routing_stream(user_text, state, services, memory):
            out.append(env)
        return out

    return asyncio.run(_collect())


def _run_stream_once_debug(user_text: str, state: SessionState, services: Services, memory: MemoryStore):
    async def _collect():
        out = []
        async for env in router_mod.patient_routing_stream(user_text, state, services, memory, debug=True):
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


def test_is_new_topic_while_confirm_pending_detects_news_intent():
    assert is_new_topic_while_confirm_pending("какие скидки?") is True


def test_patient_routing_stream_logs_background_refresh_start_failure(monkeypatch, caplog):
    class _FailingRefreshServices:
        def ensure_background_refresh_started(self):
            raise RuntimeError("background refresh failed")

    monkeypatch.setattr(router_mod, "explicit_operator_requested", lambda text: True)

    state = SessionState(session_id="router-refresh-log")
    memory = MemoryStore()

    with caplog.at_level(logging.WARNING):
        out = _run_stream_once("оператор", state, _FailingRefreshServices(), memory)

    assert out
    assert out[0].handoff is True
    assert "background_refresh_start_failed" in caplog.text


def test_safe_get_branches_logs_failure(caplog):
    class _FailingServices:
        def get_branches(self):
            raise RuntimeError("branches unavailable")

    with caplog.at_level(logging.WARNING):
        branches = flow_policy_mod._safe_get_branches(_FailingServices())

    assert branches == []
    assert "branches_fetch_failed" in caplog.text


def test_reset_appointment_runtime_state_clears_active_dialog_state():
    state = SessionState(
        session_id="appt-dialog-reset",
        last_entities={"appointment_flow_active": True, "doctor_name": "Трубин Алексей Юрьевич"},
        dialog=DialogState(
            label="APPOINTMENT",
            phase=AppointmentPhase.CONFIRM,
            entities={"doctor_name": "Трубин Алексей Юрьевич"},
            missing_slots=["patient_name"],
            confidence=0.82,
        ),
    )

    reset_appointment_runtime_state(state)

    assert state.last_entities == {"doctor_name": "Трубин Алексей Юрьевич"}
    assert state.dialog.label == "OTHER"
    assert state.dialog.phase == ""
    assert state.dialog.entities == {}


def test_clear_on_handoff_preserves_samara_city_and_clears_dialog_and_pending():
    state = SessionState(
        session_id="handoff-clear",
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "doctor_name": "Трубин Алексей Юрьевич",
        },
        dialog=DialogState(
            label="APPOINTMENT",
            phase=AppointmentPhase.CONFIRM,
            entities={"doctor_name": "Трубин Алексей Юрьевич"},
        ),
    )
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    clear_on_handoff(state, memory)

    assert state.last_entities == {"city": "Самара"}
    assert memory.get_pending(state) is None
    assert state.dialog.label == "OTHER"
    assert state.dialog.phase == ""


def test_clear_on_topic_switch_clears_topic_context_and_pending():
    state = SessionState(
        session_id="topic-switch-clear",
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "appointment_action": "book",
            "doctor_id": 7,
            "doctor_name": "Ким Татьяна Александровна",
            "specialty": "уролог",
            "service_name": "Прием врача",
            "test_name": "ОАК",
            "doc_request_kind": "tax",
            "secondary_intents": ["DOCTOR_SCHEDULE"],
            "_secondary_queue": ["DOCTOR_SCHEDULE"],
            "_secondary_offer_pending": True,
            "_catalog_confirm_pending": {"canonical": "оак"},
            "_catalog_confirm_rejects": 1,
        },
        dialog=DialogState(label="PRICE", entities={"service_name": "Прием врача"}),
    )
    memory = MemoryStore()
    memory.set_pending(state, label="PRICE", missing_slots=["service_name"])

    clear_on_topic_switch(state, memory)

    assert state.last_entities == {"city": "Самара"}
    assert memory.get_pending(state) is None
    assert state.dialog.label == "OTHER"
    assert state.dialog.entities == {}


def test_clear_on_appointment_end_clears_only_appointment_context():
    state = SessionState(
        session_id="appointment-end-clear",
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "appointment_confirm_pending": True,
            "doctor_id": 3,
            "doctor_name": "Хальметова Алина Алексеевна",
            "service_name": "Прием врача",
            "test_name": "ОАК",
            "patient_name": "Иван Иванов",
            "doc_request_kind": "tax",
        },
        dialog=DialogState(label="APPOINTMENT", phase=AppointmentPhase.CONFIRM),
    )
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    clear_on_appointment_end(state, memory)

    assert state.last_entities == {"city": "Самара", "doc_request_kind": "tax"}
    assert memory.get_pending(state) is None
    assert state.dialog.label == "OTHER"
    assert state.dialog.phase == ""


def test_patient_routing_stream_uses_orchestrator_by_default(monkeypatch):
    async def fake_run_pipeline(text, state, services=None, memory=None, runtime_options=None):
        _ = services, memory, runtime_options
        ctx = OrchestratorContext(text=text, state=state)
        ctx.decision = RouteDecision(label="PRICE", confidence=0.81, source="llm_primary")
        ctx.response = ResponseEnvelope(text="Уточните, пожалуйста, услугу.")
        return ctx

    async def fail_route_patient_message(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("legacy route_patient_message should not be used when orchestrator is the default path")

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(router_mod, "route_patient_message", fail_route_patient_message)

    state = SessionState(session_id="orchestrator-stream")
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("цена", state, services, memory)

    assert len(out) == 1
    assert out[0].text == "Уточните, пожалуйста, услугу."
    assert out[0].handoff is False


def test_patient_routing_stream_uses_orchestrator_outputs_without_legacy_route_call(monkeypatch):
    async def fake_run_pipeline(text, state, services=None, memory=None, runtime_options=None):
        _ = services, memory, runtime_options
        ctx = OrchestratorContext(text=text, state=state)
        ctx.decision = RouteDecision(label="PRICE", confidence=0.91, source="llm_primary")
        ctx.plan = Plan(label="PRICE")
        ctx.evidence = Evidence(items={"payload": "stub"})
        ctx.response = ResponseEnvelope(text="Ответ из orchestrator", handoff=False)
        return ctx

    async def fail_route_patient_message(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("patient_routing_stream should use decision/plan/evidence from orchestrator")

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(router_mod, "route_patient_message", fail_route_patient_message)

    state = SessionState(session_id="orchestrator-outputs")
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("цена", state, services, memory)

    assert len(out) == 1
    assert out[0].text == "Ответ из orchestrator"
    assert out[0].handoff is False
    assert state.history[-2:] == [
        {"role": "user", "text": "цена"},
        {"role": "assistant", "text": "Ответ из orchestrator"},
    ]


def test_patient_routing_stream_logs_traceback_when_orchestrator_crashes(monkeypatch):
    async def fake_run_pipeline(*args, **kwargs):
        _ = args, kwargs
        raise RuntimeError("boom")

    logged: list[tuple[tuple, dict]] = []

    def fake_logger_exception(*args, **kwargs):
        logged.append((args, kwargs))

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(router_mod.logger, "exception", fake_logger_exception)

    state = SessionState(session_id="orchestrator-crash")
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once_debug("расписание Хальметовой", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is True
    assert out[0].state_update == {"debug": {"route_error": "boom"}}
    assert logged


def test_patient_routing_stream_allows_test_result_for_non_samara_city(monkeypatch):
    """TEST_RESULT работает для любого города: публичная ссылка результата
    не привязана к региону, поэтому city-gate не должен блокировать handoff'ом
    запросы вроде «нужен результат анализов Оренбург»."""
    captured: dict[str, str] = {}

    async def fake_run_pipeline(text, state, services=None, memory=None, runtime_options=None):
        _ = services, memory, runtime_options
        captured["text"] = text
        ctx = OrchestratorContext(text=text, state=state)
        ctx.decision = RouteDecision(label="TEST_RESULT", confidence=0.9, source="llm_primary")
        ctx.plan = Plan(label="TEST_RESULT")
        ctx.evidence = Evidence(items={})
        ctx.response = ResponseEnvelope(
            text="Пришлите, пожалуйста, фамилию, год рождения, филиал и номер заказа.",
            handoff=False,
        )
        return ctx

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fake_run_pipeline)

    state = SessionState(session_id="test-result-orenburg")
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("нужен результат анализов Оренбург", state, services, memory)

    assert captured.get("text") == "нужен результат анализов Оренбург", (
        "TEST_RESULT-реплика с не-самарским городом обязана дойти до оркестратора, "
        "а не блокироваться city-gate"
    )
    assert len(out) == 1
    assert out[0].handoff is False
    assert "только по Самаре" not in out[0].text


def test_patient_routing_stream_still_blocks_non_test_result_for_non_samara_city(monkeypatch):
    """Регрессия: city-gate должен продолжать блокировать ВСЁ, что НЕ TEST_RESULT,
    при упоминании не-самарского города (например, запись/цены в Оренбурге)."""

    async def fail_run_pipeline(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError(
            "city-gate должен сработать ДО оркестратора для не-TEST_RESULT интентов"
        )

    monkeypatch.setattr("messengers_router.orchestrator.run_pipeline", fail_run_pipeline)

    state = SessionState(session_id="appointment-orenburg")
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("хочу записаться в Оренбурге", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is True
    assert out[0].text == "Сейчас могу помочь только по Самаре. Соединяю с оператором."


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


def test_resolve_cached_doctor_name_candidate_uses_local_doctors_cache(monkeypatch):
    doctor_name_port_mod._DOCTOR_CACHE_SIGNATURE = None
    doctor_name_port_mod._DOCTOR_SURNAMES_MAP = {}
    doctor_name_port_mod._DOCTOR_FIO_MAP = {}

    monkeypatch.setattr(doctor_name_port_mod, "_find_existing_doctors_file", lambda: object())
    monkeypatch.setattr(
        doctor_name_port_mod,
        "_doctor_cache_signature",
        lambda _file: ("fake-doctors.jsonl", 1),
    )
    monkeypatch.setattr(
        doctor_name_port_mod,
        "_load_doctors_data",
        lambda _file: [
            {"fio": "Дразнин Антон Владимирович"},
            {"fio": "Карасев Виталий Валерьевич"},
        ],
    )

    assert doctor_name_port_mod.resolve_cached_doctor_name_candidate("к Дразнину") == "Дразнин"
    assert doctor_name_port_mod.resolve_cached_doctor_name_candidate("подскажите к кому записаться") is None


def test_classifier_does_not_extract_unmatched_question_word_as_doctor_name(monkeypatch):
    monkeypatch.setattr(
        classifier_mod,
        "resolve_cached_doctor_name_candidate",
        lambda _text, prefer_schedule=False: None,
    )

    assert classifier_mod._extract_doctor_name("подскажите к кому записаться", mode="appointment") is None


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


def test_route_message_starts_catalog_confirm_for_fuzzy_doctor(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="APPOINTMENT",
                confidence=0.74,
                entities={},
                flags={"rule_appointment", "doctor_name_unverified"},
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

    async def fake_match_catalog_doctor(self, raw_text_or_name: str):
        _ = self, raw_text_or_name
        return {"status": "fuzzy", "query": "евграфова", "canonical": "Евграфова"}

    async def fake_match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = ""):
        _ = self, raw_text_or_name, current_service_name
        return {"status": "miss", "query": "", "canonical": ""}

    async def fake_execute_plan(_plan, _state, _services):
        raise AssertionError("execute_plan must not run while waiting catalog confirmation")

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(Services, "match_catalog_doctor", fake_match_catalog_doctor)
    monkeypatch.setattr(Services, "match_catalog_service", fake_match_catalog_service)

    state = SessionState(session_id="catalog-confirm-fuzzy-doctor")
    services = Services()
    memory = MemoryStore()

    decision, plan, evidence = asyncio.run(
        router_mod.route_patient_message(
            "Запишите к Евграфовой",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "OTHER"
    assert plan.label == "OTHER"
    payload = evidence.get("catalog_confirm_response")
    assert isinstance(payload, dict)
    assert "Евграфова" in str(payload.get("text") or "")
    pending = state.last_entities.get("_catalog_confirm_pending")
    assert isinstance(pending, dict)
    assert pending.get("canonical") == "Евграфова"


def test_route_message_keeps_checkup_category_broad(monkeypatch):
    """Регрессия (Bug #2): «чекап» — это КАТЕГОРИЯ (линейка пакетов), а не одна
    услуга. Каталог отдаёт exact на «Ежегодный Чекап»; если запиннить это имя,
    test_assist схлопнет всю линейку в один пакет. Гард в
    _inject_catalog_candidates обязан оставить запрос широким: service_name НЕ
    пиннится, выставляется флаг test_assist_category_kept_broad."""

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="TEST_ASSIST",
                confidence=0.9,
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

    # Каталог ВСЕГДА вернул бы exact на «Ежегодный Чекап» — без гарда это
    # запиннило бы service_name и схлопнуло линейку. Гард должен не дать
    # применить этот результат для категорийного запроса.
    async def fake_match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = ""):
        _ = self, current_service_name
        if "чекап" in str(raw_text_or_name or "").lower():
            return {"status": "exact", "query": raw_text_or_name, "canonical": "Ежегодный Чекап"}
        return {"status": "miss", "query": "", "canonical": ""}

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(Services, "match_catalog_service", fake_match_catalog_service)

    state = SessionState(session_id="checkup-category-broad")
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "чекап",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "TEST_ASSIST"
    assert plan.label == "TEST_ASSIST"
    assert "test_assist_category_kept_broad" in decision.flags
    # Главное: имя НЕ запиннилось в одну каноническую строку.
    assert not (decision.entities or {}).get("service_name")
    assert "catalog_service_exact" not in decision.flags
    assert str(state.last_entities.get("service_name") or "").lower() != "ежегодный чекап"


def test_route_message_accepts_catalog_confirm_yes_and_continues_flow(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.2,
                entities={},
                flags={"low_confidence"},
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
        return Evidence(items={"doctor_schedule": {"schedule": []}})

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(
        session_id="catalog-confirm-yes",
        last_entities={
            "_catalog_confirm_pending": {
                "kind": "doctor",
                "label": "DOCTOR_SCHEDULE",
                "entity_key": "doctor_name",
                "canonical": "Евграфова",
                "query": "евграфова",
            }
        },
    )
    services = Services()
    memory = MemoryStore()
    memory.set_pending(state, label="OTHER", missing_slots=["catalog_confirm"])

    decision, plan, evidence = asyncio.run(
        router_mod.route_patient_message(
            "да",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "DOCTOR_SCHEDULE"
    assert plan.label == "DOCTOR_SCHEDULE"
    assert state.last_entities.get("doctor_name") == "Евграфова"
    assert state.last_entities.get("_catalog_confirm_pending") is None
    assert evidence.get("doctor_schedule") is not None


def test_patient_routing_stream_renders_catalog_confirm_response(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="APPOINTMENT",
                confidence=0.74,
                entities={},
                flags={"rule_appointment", "doctor_name_unverified"},
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

    async def fake_match_catalog_doctor(self, raw_text_or_name: str):
        _ = self, raw_text_or_name
        return {"status": "fuzzy", "query": "евграфова", "canonical": "Евграфова"}

    async def fake_match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = ""):
        _ = self, raw_text_or_name, current_service_name
        return {"status": "miss", "query": "", "canonical": ""}

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(Services, "match_catalog_doctor", fake_match_catalog_doctor)
    monkeypatch.setattr(Services, "match_catalog_service", fake_match_catalog_service)

    state = SessionState(session_id="catalog-confirm-stream")
    services = Services()
    memory = MemoryStore()
    out = _run_stream_once("Запишите к Евграфовой", state, services, memory)

    assert out
    assert "Это верно? Ответьте «да» или «нет»." in out[0].text
    assert out[0].handoff is False


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("Где сделать МРТ?", "unsupported_service"),
        ("А КТ у вас есть?", "unsupported_service"),
        ("Можно сделать рентген?", "unsupported_service"),
        ("Есть прививки?", "unsupported_service"),
        ("Нужно сделать АКДС-М ребенку", "unsupported_service"),
        ("Можно поставить прививку от полиомиелита?", "unsupported_service"),
        ("Есть Пентаксим?", "unsupported_service"),
        ("Можно записаться к детскому урологу?", "unsupported_specialist"),
        ("Здравствуйте! К детскому кардиологу можно попасть?", "unsupported_specialist"),
        ("Нужен психиатр", "unsupported_specialist"),
        ("У вас работает косметолог?", "unsupported_specialist"),
        ("Нужна справка в ГИБДД", "unsupported_document_service"),
        ("Нужна медкомиссия для спортсменов", "unsupported_document_service"),
        ("Заказать справку для спортсменов", "unsupported_document_service"),
    ],
)
def test_detect_unsupported_catalog_matches_catalog_entries(text, expected_kind):
    match = detect_unsupported_catalog(text)
    assert match is not None
    assert match.kind == expected_kind


def test_patient_routing_stream_short_circuits_unsupported_catalog(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="PRICE",
                confidence=0.95,
                entities={"service_name": "мрт"},
                flags={"unsupported_catalog", "unsupported_service"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    async def fake_execute_plan(_plan, _state, _services):
        raise AssertionError("execute_plan must not run for unsupported catalog")

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(session_id="unsupported-short-circuit")
    services = Services()
    memory = MemoryStore()

    out = _run_stream_once("Где сделать МРТ?", state, services, memory)

    assert out
    assert out[0].text == "К сожалению, в данный момент клиника не оказывает данную услугу. Приносим извинения за неудобства."
    assert out[0].handoff is False


def test_patient_routing_stream_unsupported_specialist_does_not_pollute_state(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="DOCTOR_INFO",
                confidence=0.95,
                entities={"specialty": "детский кардиолог"},
                flags={"unsupported_catalog", "unsupported_specialist"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    async def fake_execute_plan(_plan, _state, _services):
        raise AssertionError("execute_plan must not run for unsupported specialist")

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(session_id="unsupported-specialist-clean-state")
    services = Services()
    memory = MemoryStore()

    out = _run_stream_once("Здравствуйте! К детскому кардиологу можно попасть?", state, services, memory)

    assert out
    assert out[0].text == "Данные врачи не ведут прием."
    assert out[0].handoff is False
    assert state.last_entities.get("specialty") is None
    assert state.last_entities.get("_last_label") is None


def test_route_message_short_circuits_on_catalog_health_degraded(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="PRICE",
                confidence=0.91,
                entities={"service_name": "Прием уролога"},
                flags={"rule_price"},
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

    async def fake_verify_doctor(decision, _services, _user_text):
        return decision

    async def fake_inject_catalog(
        decision, *, user_text, state, services, prefetched_service=None
    ):
        _ = user_text, state, services, prefetched_service
        return decision

    async def fake_get_catalog_health(self):
        _ = self
        return {
            "ok": False,
            "status": "degraded",
            "service_catalog_ok": False,
            "doctors_catalog_ok": True,
            "reason": "service_catalog_empty",
        }

    async def fake_execute_plan(_plan, _state, _services):
        raise AssertionError("execute_plan must not run when catalog health is degraded")

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "_verify_doctor_entity", fake_verify_doctor)
    monkeypatch.setattr(router_mod, "_inject_catalog_candidates", fake_inject_catalog)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(Services, "get_catalog_health", fake_get_catalog_health)

    state = SessionState(session_id="catalog-health-degraded")
    services = Services()
    memory = MemoryStore()

    decision, plan, evidence = asyncio.run(
        router_mod.route_patient_message(
            "Сколько стоит прием уролога?",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "OTHER"
    assert plan.label == "OTHER"
    assert "catalog_health_degraded" in decision.flags
    payload = evidence.get("catalog_health_response")
    assert isinstance(payload, dict)
    assert "каталог услуг" in str(payload.get("text") or "").lower()
    assert payload.get("handoff") is False


def test_patient_routing_stream_renders_catalog_health_response(monkeypatch):
    async def fake_complete_route(
        *,
        decision,
        user_text,
        state,
        services,
        memory,
        runtime_options=None,
        nlu_debug=None,
        catalog_prefetch=None,
    ):
        _ = decision, user_text, state, services, memory, runtime_options, nlu_debug, catalog_prefetch
        return (
            RouteDecision(
                label="OTHER",
                confidence=0.95,
                entities={},
                flags={"catalog_health_degraded"},
                needs_handoff=False,
            ),
            Plan(label="OTHER"),
            Evidence(
                items={
                    "catalog_health_response": {
                        "text": (
                            "Сейчас временно недоступен каталог услуг. "
                            "Попробуйте повторить запрос через 5 минут или напишите «оператор»."
                        ),
                        "handoff": False,
                    }
                }
            ),
        )

    monkeypatch.setattr(router_mod, "_complete_route_after_doctor_guard", fake_complete_route)

    state = SessionState(session_id="catalog-health-stream")
    services = Services()
    memory = MemoryStore()

    out = _run_stream_once("Сколько стоит прием уролога?", state, services, memory)

    assert out
    assert out[0].handoff is False
    assert "временно недоступен каталог услуг" in out[0].text.lower()


def test_catalog_health_requirements_doctor_schedule_with_selected_doctor():
    state = SessionState(
        session_id="catalog-health-schedule-selected-doctor",
        last_entities={"doctor_name": "Хальметова"},
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="DOCTOR_SCHEDULE",
        confidence=0.8,
        entities={"doctor_name": "Хальметова"},
        flags={"rule_schedule"},
        needs_handoff=False,
    )

    need_service, need_doctors = router_mod._catalog_health_requirements(decision, state, memory)

    assert need_service is False
    assert need_doctors is False


def test_route_message_prepare_followup_to_blooddraw_keeps_address(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="ADDRESS",
                confidence=0.82,
                entities={"service_name": "Анализы"},
                flags={"rule_nonbookable_walkin"},
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
        session_id="prepare-followup-address",
        last_entities={"_last_label": "PREPARE", "service_name": "Анализ крови на холестерин"},
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Где можно сдать кровь?",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "ADDRESS"
    assert "flow_prepare_followup_override" not in decision.flags
    assert plan.label == "ADDRESS"


def test_route_message_prepare_short_followup_overrides_doctor_info(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="DOCTOR_INFO",
                confidence=0.7,
                entities={"specialty": "вульвоскопия"},
                flags={"rule_doctor_info"},
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
        session_id="prepare-followup-keep-prepare",
        last_entities={"_last_label": "PREPARE", "service_name": "ФГДС"},
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "А к вульвоскопии?",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "PREPARE"
    assert "flow_prepare_followup_override" in decision.flags
    assert plan.label == "PREPARE"


def test_route_message_prepare_short_followup_overrides_appointment(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="APPOINTMENT",
                confidence=0.8,
                entities={"service_name": "Вульвоскопия"},
                flags={"rule_appointment"},
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
        session_id="prepare-followup-appointment-to-prepare",
        last_entities={"_last_label": "PREPARE", "service_name": "Вульвоскопия"},
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Вульвоскопия",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "PREPARE"
    assert "flow_prepare_followup_override" in decision.flags
    assert plan.label == "PREPARE"


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


def test_build_plan_appointment_selection_mode_doctor_uses_doctors_info_first():
    state = SessionState(
        session_id="appt-doctor-mode",
        last_entities={
            "city": "Самара",
            "service_name": "УЗИ брюшной полости",
            "appointment_selection_mode": "doctor",
        },
    )
    memory = MemoryStore()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "УЗИ брюшной полости"},
        flags=set(),
        needs_handoff=False,
    )

    plan = build_plan(decision, state, "покажите врачей", memory)

    assert plan.steps
    assert plan.steps[0].tool == "doctors_info"
    assert any(step.tool == "address_info" for step in plan.steps)


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


def test_quick_fill_patient_name_rejects_soft_pause_phrase():
    out = quick_fill_core_entities("подождите пока", {}, ["patient_name"])
    assert "patient_name" not in out


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
    assert state.last_entities.get("doctor_name") == "Иванов Иван Иванович"


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


def test_build_doctor_info_response_clears_stale_doctor_for_multi_specialty_list():
    state = SessionState(
        session_id="doc-info-multi",
        last_entities={"doctor_name": "Рязанова Валерия Владимировна", "doctor_id": 999},
    )
    evidence = Evidence(
        items={
            "doctors_info": {
                "doctors": [
                    {"id": 1, "fio": "Просвиров Евгений Юрьевич", "specialization": "Ревматолог"},
                    {"id": 2, "fio": "Другой Врач", "specialization": "Ревматолог"},
                ],
                "entities_used": {"specialty_query": "ревматолог"},
            }
        }
    )

    env = _build_doctor_info_response("DOCTOR_INFO", evidence, state)

    assert env is not None
    assert state.last_entities.get("doctor_name") is None
    assert state.last_entities.get("doctor_id") is None


def test_build_test_result_response_ready_delivers_pdf_link_and_attachment():
    # BUG-2026-06-08-01: результат готов → рендер отдаёт реальную ссылку на PDF + вложение
    # (bot-constructed deep-link getanaliz удалён). Контракт: result_links/result_attachments
    # от сервиса доходят до пациента; портал в этом исходе не показываем.
    pdf = "https://naykalab.ru/result/blank.pdf"
    evidence = Evidence(
        items={
            "test_result_status": {
                "ready": True,
                "note": "result_ready_pdf",
                "result_preview": "Ваш результат готов.",
                "result_links": [pdf],
                "result_attachments": [{"type": "pdf", "name": "Результат анализа", "url": pdf}],
            }
        }
    )
    env = _build_test_result_response("TEST_RESULT", evidence)
    assert env is not None
    assert pdf in env.text
    assert "getanaliz" not in env.text.lower()
    assert "naykalab.ru/samara" not in env.text
    assert any(a.get("type") == "pdf" and a.get("url") == pdf for a in env.attachments)


def test_build_doctor_schedule_response_hydrates_context_without_forcing_flow_active():
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
    env = _build_doctor_schedule_response("DOCTOR_SCHEDULE", evidence, state, MemoryStore())
    assert env is not None
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("doctor_name") == "Трубин Алексей Юрьевич"
    assert "Трубин Алексей Юрьевич" in env.text


def test_build_doctor_schedule_response_offers_operator_when_no_slots_for_two_weeks():
    state = SessionState(
        session_id="doc-schedule-no-slots",
        last_entities={
            "appointment_flow_active": True,
            "doctor_name": "Ким Татьяна Александровна",
            "date_from": "2026-04-15",
            "time_from": "09:00",
        },
    )
    memory = MemoryStore()
    evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [],
                "schedule_unavailable_reason": "no_free_slots_2_weeks",
            }
        }
    )

    env = _build_doctor_schedule_response("DOCTOR_SCHEDULE", evidence, state, memory)

    assert env is not None
    assert "Врач найден, но свободных слотов нет в ближайшие 2 недели." in env.text
    assert "Перевести на оператора?" in env.text
    assert env.handoff is False
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("date_from") is None
    assert state.last_entities.get("time_from") is None
    assert state.last_entities.get("_operator_offer_pending") is True
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "OTHER"
    assert "operator_offer_confirm" in (pending.get("missing") or [])


def test_route_message_datetime_after_no_slots_does_not_reactivate_appointment_prelock(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.2,
                entities={},
                flags={"low_confidence"},
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

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="kim-no-slots-followup",
        last_entities={
            "appointment_flow_active": True,
            "doctor_name": "Ким Татьяна Александровна",
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    no_slots_evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [],
                "schedule_unavailable_reason": "no_free_slots_2_weeks",
            }
        }
    )

    first = _build_doctor_schedule_response("DOCTOR_SCHEDULE", no_slots_evidence, state, memory)
    assert first is not None
    assert state.last_entities.get("appointment_flow_active") is None
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "OTHER"

    out = _run_stream_once("на завтра на 9:00", state, services, memory)

    assert len(out) == 1
    assert "в ближайшие 2 недели" not in out[0].text.lower() or "перевести на оператора" in first.text.lower()
    assert "фио пациента" not in out[0].text.lower()
    assert "подтверждаете" not in out[0].text.lower()
    assert "если нужно записаться" not in out[0].text.lower()


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
    env = _build_appointment_schedule_preview_response("APPOINTMENT", evidence, state, MemoryStore())
    assert env is not None
    assert "Трубин Алексей Юрьевич" in env.text


def test_build_appointment_schedule_preview_response_offers_operator_when_no_slots_for_two_weeks():
    state = SessionState(
        session_id="appt-preview-no-slots",
        last_entities={
            "appointment_flow_active": True,
            "doctor_name": "Паничева Ольга",
        },
    )
    memory = MemoryStore()
    evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [],
                "schedule_unavailable_reason": "no_free_slots_2_weeks",
            }
        }
    )

    env = _build_appointment_schedule_preview_response("APPOINTMENT", evidence, state, memory)

    assert env is not None
    assert "Врач найден, но свободных слотов нет в ближайшие 2 недели." in env.text
    assert "Перевести на оператора?" in env.text
    assert "расписание не найдено" not in env.text.lower()
    assert env.handoff is False
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("_operator_offer_pending") is True
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "OTHER"
    assert "operator_offer_confirm" in (pending.get("missing") or [])


def test_build_appointment_schedule_preview_response_offers_operator_when_doctor_not_in_samara():
    # Назван конкретный врач, которого нет в самарском каталоге онлайн-записи
    # (doctor_lookup=unresolved). Вместо «расписание не найдено» и слепого
    # списка самарских филиалов — честный отказ «только по Самаре» + оператор.
    state = SessionState(
        session_id="appt-preview-non-samara-doctor",
        last_entities={
            "appointment_flow_active": True,
            "doctor_name": "Турмухамбетова Балслу Турмурадовна",
        },
    )
    memory = MemoryStore()
    evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [],
                "doctor_lookup": "unresolved",
            }
        }
    )

    env = _build_appointment_schedule_preview_response("APPOINTMENT", evidence, state, memory)

    assert env is not None
    assert "только по филиалам в Самаре" in env.text
    assert "Перевести на оператора?" in env.text
    assert "расписание не найдено" not in env.text.lower()
    assert env.handoff is False
    assert state.last_entities.get("appointment_flow_active") is None
    assert state.last_entities.get("_operator_offer_pending") is True
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "OTHER"
    assert "operator_offer_confirm" in (pending.get("missing") or [])


def test_build_appointment_step_response_offers_operator_when_doctor_not_in_samara():
    # «Записаться к <ФИО> на завтра»: дата задана → preview-ветка пропущена,
    # доходим до branch-шага; врача нет в самарском каталоге → честный отказ
    # (а не слепой список 5 филиалов Самары).
    state = SessionState(
        session_id="appt-step-non-samara-doctor",
        last_entities={
            "doctor_name": "Турмухамбетова Балслу Турмурадовна",
            "date_from": "2026-06-02",
            "time_from": "10:00",
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    evidence = Evidence(
        items={
            "doctor_schedule": {
                "schedule": [],
                "doctor_lookup": "unresolved",
            }
        }
    )

    env = _build_appointment_step_response("APPOINTMENT", evidence, state, services, memory)

    assert env is not None
    assert "только по филиалам в Самаре" in env.text
    assert "Перевести на оператора?" in env.text
    assert env.handoff is False
    assert state.last_entities.get("_operator_offer_pending") is True
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "OTHER"
    assert "operator_offer_confirm" in (pending.get("missing") or [])


def test_doctors_schedule_week_flags_unresolved_doctor_when_not_in_samara_directory():
    # Имя врача названо, но фамилия не резолвится в самарском каталоге →
    # doctor_lookup=unresolved (без сетевого запроса расписания). Каталог
    # непустой и не содержит запрошенного врача (как Турмухамбетова —
    # только Оренбург, исключён из кэша через EXCLUDED_REGION_ROOTS).
    services = Services()

    samara_doctors = [
        {"fio": "Иванова Мария Петровна", "regions": ["Ленина 5"]},
        {"fio": "Петров Сергей Иванович", "regions": ["Самара, ул. Гагарина, 1"]},
        {"fio": "Сидорова Анна Олеговна", "regions": ["Самара, Московское шоссе, 10"]},
    ]

    async def _cache():
        return samara_doctors

    services._ensure_doctors_cache_loaded = _cache

    async def _run():
        return await services.doctors_schedule_week(
            "Записаться к Турмухамбетова Балслу Турмурадовна",
            {"doctor_name": "Турмухамбетова Балслу Турмурадовна"},
        )

    payload = asyncio.run(_run())
    assert payload.get("schedule") == []
    assert payload.get("doctor_lookup") == "unresolved"


def test_build_appointment_schedule_preview_response_skips_for_reschedule_action():
    state = SessionState(
        session_id="appt-preview-reschedule-skip",
        last_entities={"doctor_name": "Трубин", "appointment_action": "reschedule"},
    )
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

    env = _build_appointment_schedule_preview_response("APPOINTMENT", evidence, state, MemoryStore())

    assert env is None


def test_apply_pending_override_keeps_price_flow_on_city_reply():
    decision = RouteDecision(label="ADDRESS", confidence=0.72, flags={"rule_address"})
    pending = {"label": "PRICE", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="Самара")

    assert label == "PRICE"


def test_apply_pending_override_keeps_price_flow_on_catalog_service_reply(monkeypatch):
    decision = RouteDecision(label="TEST_ASSIST", confidence=0.71, flags={"rule_test_assist"})
    pending = {"label": "PRICE", "missing": ["service_name"]}

    monkeypatch.setattr(
        flow_policy_mod,
        "resolve_price_service_name_from_catalog",
        lambda text, current_service_name="": "Биохимия крови" if "биохим" in text.lower() else None,
    )

    label = apply_pending_override(decision, pending, user_text="биохимия крови")

    assert label == "PRICE"


def test_apply_pending_override_keeps_price_flow_on_short_specialty_reply_without_catalog_match(monkeypatch):
    decision = RouteDecision(label="DOCTOR_INFO", confidence=0.74, flags={"rule_doctor_info"})
    pending = {"label": "PRICE", "missing": ["service_name"]}

    monkeypatch.setattr(
        flow_policy_mod,
        "resolve_price_service_name_from_catalog",
        lambda text, current_service_name="": None,
    )

    label = apply_pending_override(decision, pending, user_text="терапевт")

    assert label == "PRICE"


def test_apply_pending_override_allows_address_switch_on_explicit_address_request():
    decision = RouteDecision(label="ADDRESS", confidence=0.72, flags={"rule_address"})
    pending = {"label": "PRICE", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="адрес в Самаре")

    assert label == "ADDRESS"


def test_quick_fill_entities_from_text_resolves_catalog_service_for_price_followup(monkeypatch):
    monkeypatch.setattr(
        flow_policy_mod,
        "resolve_price_service_name_from_catalog",
        lambda text, current_service_name="": "Биохимия крови" if "биохим" in text.lower() else None,
    )

    out = quick_fill_entities_from_text(
        "биохимия крови",
        {"_last_label": "PRICE"},
        ["service_name"],
        Services(),
    )

    assert out.get("service_name") == "Биохимия крови"


def test_quick_fill_entities_from_text_builds_consult_service_from_specialty_when_catalog_miss(monkeypatch):
    monkeypatch.setattr(
        flow_policy_mod,
        "resolve_price_service_name_from_catalog",
        lambda text, current_service_name="": None,
    )

    out = quick_fill_entities_from_text(
        "терапевт",
        {"_last_label": "PRICE"},
        ["service_name"],
        Services(),
    )

    assert out.get("service_name") == "прием терапевт"


def test_missing_slots_appointment_unknown_action_requests_action_first():
    missing = missing_slots("APPOINTMENT", {"appointment_action": "unknown"})
    assert missing == ["appointment_action"]
    assert clarification_question("APPOINTMENT", missing, {"appointment_action": "unknown"}).lower().startswith(
        "хотите отменить или перенести"
    )


def test_clarification_question_reschedule_anyof_datetime_prefers_datetime_prompt():
    missing = ["_any_of:date_from,time_from,date_hint", "patient_name"]
    text = clarification_question("APPOINTMENT", missing, {"appointment_action": "reschedule"}).lower()
    assert "дат" in text and "время" in text


def test_clarification_question_cancel_patient_name_is_action_specific():
    text = clarification_question("APPOINTMENT", ["patient_name"], {"appointment_action": "cancel"}).lower()
    assert "для отмены записи" in text
    assert "фио пациента" in text


def test_clarification_question_reschedule_doctor_prompt_has_no_service_word():
    text = clarification_question(
        "APPOINTMENT",
        ["_any_of:doctor_id,doctor_name"],
        {"appointment_action": "reschedule"},
    ).lower()
    assert "фио врача" in text
    assert "услуг" not in text


def test_missing_slots_reschedule_requires_concrete_doctor_even_with_specialty():
    missing = missing_slots(
        "APPOINTMENT",
        {"appointment_action": "reschedule", "specialty": "кардиолог"},
    )
    assert "_any_of:doctor_id,doctor_name" in missing
    assert "patient_name" in missing


def test_memory_merge_entities_reschedule_clears_stale_specialty_context():
    state = SessionState(
        session_id="appt-reschedule-stale-specialty",
        last_entities={
            "specialty": "кардиолог",
            "service_name": "прием кардиолога",
            "test_name": "общий анализ крови",
            "appointment_selection_mode": "doctor",
            "appointment_branch_options": ["г. Самара, пр. Ленина, 5"],
        },
    )
    memory = MemoryStore()

    memory.merge_entities(state, {"appointment_action": "reschedule"}, label="APPOINTMENT")

    assert state.last_entities.get("specialty") is None
    assert state.last_entities.get("service_name") is None
    assert state.last_entities.get("test_name") is None
    assert state.last_entities.get("appointment_selection_mode") is None
    assert state.last_entities.get("appointment_branch_options") is None


def test_memory_merge_entities_specialty_change_drops_stale_doctor_context():
    state = SessionState(
        session_id="mem-specialty-change-drop-doctor",
        last_entities={
            "doctor_name": "Трубин Алексей Юрьевич",
            "doctor_id": 1001,
            "specialty": "уролог",
            "branch_name": "г. Самара, ул. Победы, 83",
            "date_from": "2026-04-20",
            "time_from": "11:00",
        },
    )
    memory = MemoryStore()

    memory.merge_entities(state, {"specialty": "кардиолог"}, label="PRICE")

    assert state.last_entities.get("doctor_name") is None
    assert state.last_entities.get("doctor_id") is None
    assert state.last_entities.get("branch_name") is None
    assert state.last_entities.get("date_from") is None
    assert state.last_entities.get("time_from") is None
    assert state.last_entities.get("specialty") == "кардиолог"


def test_memory_merge_entities_service_change_drops_stale_doctor_context():
    state = SessionState(
        session_id="mem-service-change-drop-doctor",
        last_entities={
            "doctor_name": "Трубин Алексей Юрьевич",
            "doctor_id": 1001,
            "service_name": "Прием врача-уролога",
            "branch_name": "г. Самара, ул. Победы, 83",
        },
    )
    memory = MemoryStore()

    memory.merge_entities(state, {"service_name": "Прием врача-кардиолога первичный"}, label="PRICE")

    assert state.last_entities.get("doctor_name") is None
    assert state.last_entities.get("doctor_id") is None
    assert state.last_entities.get("branch_name") is None
    assert state.last_entities.get("service_name") == "Прием врача-кардиолога первичный"


def test_memory_merge_entities_reschedule_service_noise_keeps_doctor_context():
    state = SessionState(
        session_id="mem-reschedule-service-noise-keeps-doctor",
        last_entities={
            "appointment_action": "reschedule",
            "doctor_name": "Трубин Алексей Юрьевич",
            "doctor_id": 1001,
            "service_name": "Суточное мониторирование ЭКГ (по Холтеру1/по Холтеру2)",
            "branch_name": "г. Самара, ул. Победы, 83",
        },
    )
    memory = MemoryStore()

    memory.merge_entities(state, {"service_name": "ЭКГ"}, label="APPOINTMENT")

    assert state.last_entities.get("doctor_name") == "Трубин Алексей Юрьевич"
    assert state.last_entities.get("doctor_id") == 1001
    assert state.last_entities.get("service_name") == "Суточное мониторирование ЭКГ (по Холтеру1/по Холтеру2)"


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


def test_apply_pending_override_keeps_appointment_on_selection_mode_reply():
    decision = RouteDecision(label="DOCTOR_INFO", confidence=0.74, flags={"rule_doctor_info"})
    pending = {"label": "APPOINTMENT", "missing": ["_any_of:city,branch_name,branch_id"]}

    label = apply_pending_override(decision, pending, user_text="врачи")

    assert label == "APPOINTMENT"


def test_apply_pending_override_keeps_appointment_on_doctor_reply_when_waiting_doctor():
    decision = RouteDecision(label="DOCTOR_SCHEDULE", confidence=0.82, flags={"rule_schedule_doctor_followup"})
    pending = {"label": "APPOINTMENT", "missing": ["_any_of:doctor_id,doctor_name,specialty,service_name"]}

    label = apply_pending_override(decision, pending, user_text="Трубин")

    assert label == "APPOINTMENT"


def test_apply_pending_override_keeps_appointment_on_doctor_reply_when_misclassified_as_test_result():
    decision = RouteDecision(label="TEST_RESULT", confidence=0.82, flags={"rule_test_result"})
    pending = {"label": "APPOINTMENT", "missing": ["_any_of:doctor_id,doctor_name,specialty,service_name"]}

    label = apply_pending_override(decision, pending, user_text="Белохвостикова")

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


def test_is_test_assist_category_term_distinguishes_category_from_service():
    # Категория (линейка пакетов) — запрос держим широким.
    assert is_test_assist_category_term("чекап") is True
    assert is_test_assist_category_term("Чекап") is True
    assert is_test_assist_category_term("мужской чекап") is True
    assert is_test_assist_category_term("женский чекап") is True
    assert is_test_assist_category_term("анализы чекап мужской") is True
    assert is_test_assist_category_term("чек-ап") is True
    assert is_test_assist_category_term("чек ап") is True
    # Не категория — обычные услуги/реплики не должны триггерить гард.
    assert is_test_assist_category_term("оак") is False
    assert is_test_assist_category_term("ферритин") is False
    assert is_test_assist_category_term("подскажите пожалуйста") is False
    assert is_test_assist_category_term("") is False


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


def test_appointment_step_policy_accepts_time_flexible_as_datetime():
    step = appointment_step_policy(
        {
            "branch_name": "Ленина 5",
            "date_hint": "tomorrow",
            "time_flexible": True,
            "patient_name": "Иванов Иван Иванович",
        }
    )
    assert step == "confirm"


def test_appointment_summary_renders_time_flexible():
    summary = appointment_summary(
        {
            "service_name": "Прием кардиолога",
            "branch_name": "Ленина 5",
            "date_hint": "tomorrow",
            "time_flexible": True,
            "patient_name": "Иванов Иван Иванович",
        }
    )
    assert "любое время" in summary.lower()


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


def test_refine_skips_ambiguous_appointment_action(monkeypatch):
    async def _must_not_call(*_args, **_kwargs):
        raise AssertionError("ollama refine should be skipped for ambiguous appointment action")

    monkeypatch.setattr(classifier_mod, "ollama_classify_json", _must_not_call)
    base = RouteDecision(
        label="APPOINTMENT",
        confidence=0.75,
        entities={"appointment_action": "unknown"},
        flags={"rule_appointment"},
        needs_handoff=False,
        context_action="continue",
    )

    out = asyncio.run(
        classifier_mod._maybe_refine_live_intent(
            "нужно отменить или перенести запись",
            {},
            base,
        )
    )

    assert out.entities.get("appointment_action") == "unknown"
    assert "llm_refine_used" not in out.flags


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


def test_apply_context_action_new_topic_clears_pending_and_dialog():
    state = SessionState(
        session_id="appt-new-topic-clear",
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "doctor_name": "Хальметова Алина Алексеевна",
            "_pending": {"label": "APPOINTMENT", "missing": ["date_from"]},
            "_pending_label": "APPOINTMENT",
        },
        dialog=DialogState(
            label="APPOINTMENT",
            phase=AppointmentPhase.COLLECTING,
            entities={"doctor_name": "Хальметова Алина Алексеевна"},
        ),
    )
    decision = RouteDecision(
        label="PRICE",
        confidence=0.82,
        entities={"service_name": "ОАК"},
        flags={"rule_price"},
        needs_handoff=False,
        context_action="new_topic",
    )
    memory = MemoryStore()

    out = apply_context_action(decision, state, "Сколько стоит ОАК?", memory)

    assert out.context_action == "new_topic"
    assert state.last_entities == {"city": "Самара"}
    assert memory.get_pending(state) is None
    assert state.dialog.label == "OTHER"
    assert state.dialog.phase == ""


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


def test_build_appointment_step_response_doctor_selection_mode_renders_doctors():
    state = SessionState(
        session_id="appt-step-doctor-mode",
        last_entities={
            "city": "Самара",
            "service_name": "УЗИ брюшной полости",
            "appointment_selection_mode": "doctor",
        },
    )
    evidence = Evidence(
        items={
            "doctors_info": {
                "doctors": [
                    {
                        "id": 1,
                        "fio": "Иванов Иван Иванович",
                        "specialization": "УЗИ",
                        "regions": ["г. Самара, пр. Ленина, 5"],
                    }
                ]
            }
        }
    )
    memory = MemoryStore()
    services = Services()

    env = _build_appointment_step_response("APPOINTMENT", evidence, state, services, memory)

    assert env is not None
    assert "Иванов Иван Иванович" in env.text
    assert "расписание" in env.text.lower()


def test_build_appointment_step_response_reschedule_full_data_requests_confirmation():
    state = SessionState(
        session_id="appt-step-reschedule-confirm",
        last_entities={
            "appointment_action": "reschedule",
            "service_name": "Холтер",
            "branch_name": "г. Самара, ул. Победы, 83",
            "date_hint": "tomorrow",
            "time_from": "16:00",
            "patient_name": "Петров Петр Петрович",
        },
    )
    evidence = Evidence(items={})
    memory = MemoryStore()
    services = Services()

    env = _build_appointment_step_response("APPOINTMENT", evidence, state, services, memory)

    assert env is not None
    assert env.handoff is False
    assert "подтверждаете" in env.text.lower()
    assert state.last_entities.get("appointment_confirm_pending") is True


def test_build_appointment_step_response_cancel_renders_russian_date_hint():
    state = SessionState(
        session_id="appt-step-cancel-russian-date",
        last_entities={
            "appointment_action": "cancel",
            "service_name": "Холтер",
            "date_hint": "tomorrow",
            "time_from": "16:00",
            "patient_name": "Петров Петр Петрович",
        },
    )
    evidence = Evidence(items={})
    memory = MemoryStore()
    services = Services()

    env = _build_appointment_step_response("APPOINTMENT", evidence, state, services, memory)

    assert env is not None
    assert env.handoff is True
    assert "завтра" in env.text.lower()
    assert "tomorrow" not in env.text.lower()


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


@pytest.mark.parametrize(
    "phrase",
    [
        "Нет",
        "подождите",
        "пока нет",
        "пока не буду",
        "ладно",
        "извините",
        "не то",
        "не это",
        "я другое хотел",
        "подождите пока",
    ],
)
def test_patient_routing_stream_soft_pause_requests_cancel_confirmation(phrase: str):
    state = SessionState(session_id=f"appt-soft-pause-{phrase}", last_entities={"appointment_flow_active": True})
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    out = _run_stream_once(phrase, state, services, memory)

    assert len(out) == 1
    assert "Отменить текущий процесс записи" in out[0].text
    assert state.last_entities.get("appointment_cancel_pending") is True


def test_patient_routing_stream_waiting_action_no_handoffs_to_operator(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.4,
                entities={},
                flags={"low_confidence"},
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

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="appt-wait-action-no",
        last_entities={"appointment_flow_active": True, "appointment_action": "unknown"},
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["appointment_action"])

    out = _run_stream_once("нет", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is True
    assert "оператор" in out[0].text.lower()


def test_patient_routing_stream_waiting_action_no_handoffs_without_flow_active(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.4,
                entities={},
                flags={"low_confidence"},
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

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(session_id="appt-wait-action-no-no-flow", last_entities={})
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["appointment_action"])

    out = _run_stream_once("нет", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is True
    assert "оператор" in out[0].text.lower()


def test_route_message_waiting_action_reschedule_continues_flow(monkeypatch):
    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(items={"appointment_schedule_preview": {"text": "preview"}})

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(session_id="appt-action-reschedule", last_entities={})
    services = Services()
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["appointment_action"])

    decision, plan, evidence = asyncio.run(
        router_mod.route_patient_message("перенести", state, services, memory)
    )

    assert decision.label == "APPOINTMENT"
    assert decision.entities.get("appointment_action") == "reschedule"
    assert plan.label == "APPOINTMENT"
    assert state.last_entities.get("appointment_action") == "reschedule"
    assert evidence.get("appointment_schedule_preview") is not None


def test_patient_routing_stream_reschedule_unknown_doctor_handoffs_after_repeat_attempts(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.4,
                entities={},
                flags={"low_confidence"},
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

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="appt-reschedule-unknown-doctor",
        last_entities={"appointment_flow_active": True, "appointment_action": "reschedule"},
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["_any_of:doctor_id,doctor_name"])

    first = _run_stream_once("не знаю фамилию", state, services, memory)
    assert len(first) == 1
    assert first[0].handoff is False
    assert "фио врача" in first[0].text.lower()

    second = _run_stream_once("все равно не помню", state, services, memory)
    assert len(second) == 1
    assert second[0].handoff is False
    assert "фио врача" in second[0].text.lower()

    third = _run_stream_once("все равно не помню", state, services, memory)
    assert len(third) == 1
    assert third[0].handoff is True
    assert "оператор" in third[0].text.lower()


def test_patient_routing_stream_reschedule_with_stale_specialty_requests_doctor(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="APPOINTMENT",
                confidence=0.8,
                entities={"appointment_action": "reschedule"},
                flags={"rule_appointment"},
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

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="appt-reschedule-stale-specialty-flow",
        last_entities={"specialty": "кардиолог", "_last_label": "DOCTOR_INFO"},
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("Перенесите запись, пожалуйста", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is False
    assert "фио врача" in out[0].text.lower()
    pending = memory.get_pending(state)
    assert isinstance(pending, dict)
    assert pending.get("label") == "APPOINTMENT"
    assert "_any_of:doctor_id,doctor_name" in (pending.get("missing") or [])


def test_patient_routing_stream_prelocks_reschedule_doctor_reply_before_nlu(monkeypatch):
    async def _must_not_call(*_args, **_kwargs):
        raise AssertionError("primary NLU should be skipped for slot reply inside APPOINTMENT flow")

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    async def fake_resolve_doctor_name(_text: str):
        return "Трубин Алексей Юрьевич"

    monkeypatch.setattr(router_mod, "analyze_with_candidates", _must_not_call)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="appt-reschedule-prelock-doctor",
        last_entities={"appointment_flow_active": True, "appointment_action": "reschedule"},
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    services.resolve_doctor_name = fake_resolve_doctor_name  # type: ignore[method-assign]
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["_any_of:doctor_id,doctor_name", "patient_name"])

    out = _run_stream_once("Трубин", state, services, memory)

    assert len(out) == 1
    assert "фио пациента" in out[0].text.lower()
    assert state.last_entities.get("doctor_name") == "Трубин Алексей Юрьевич"


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


def test_patient_routing_stream_confirm_pending_no_keeps_confirm_transition():
    state = SessionState(
        session_id="appt-confirm-no",
        last_entities={
            "appointment_flow_active": True,
            "appointment_confirm_pending": True,
            "doctor_name": "Хальметова Алина Алексеевна",
            "date_from": "2026-03-19",
            "time_from": "12:00",
            "patient_name": "Рахманов Владимир",
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("нет", state, services, memory)

    assert len(out) == 1
    assert "уточните новую дату" in out[0].text.lower()
    assert state.last_entities.get("appointment_cancel_pending") is None
    assert state.last_entities.get("appointment_confirm_pending") is None
    assert state.last_entities.get("appointment_flow_active") is True


def test_patient_routing_stream_confirm_pending_yes_handoff_resets_state():
    state = SessionState(
        session_id="appt-confirm-yes-reset",
        last_entities={
            "appointment_flow_active": True,
            "appointment_confirm_pending": True,
            "appointment_action": "book",
            "doctor_name": "Трубин Алексей Юрьевич",
            "service_name": "Прием врача-уролога",
            "specialty": "уролог",
            "branch_name": "г. Самара, ул. Победы, 83",
            "date_hint": "next_week",
            "time_flexible": True,
            "patient_name": "Игорь Трубник",
            "city": "Самара",
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])

    out = _run_stream_once("да", state, services, memory)

    assert len(out) == 1
    assert out[0].handoff is True
    assert "передаю заявку оператору" in out[0].text.lower()
    assert memory.get_pending(state) is None
    assert state.last_entities == {"city": "Самара"}


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


def test_patient_routing_stream_datetime_after_schedule_resets_stale_patient_name(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.82,
                entities={},
                flags={"low_confidence"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="llm",
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
        session_id="schedule-to-appointment-stale-patient",
        last_entities={
            "_last_label": "DOCTOR_SCHEDULE",
            "doctor_name": "Хальметова Алина Алексеевна",
            "branch_name": "г. Самара, пр. Ленина, 5",
            "patient_name": "Старый Пациент",
            "appointment_windows": [
                {"date": "2026-03-19", "time": "12:00", "branch": "г. Самара, пр. Ленина, 5"},
            ],
        },
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()

    out = _run_stream_once("на завтра на 12:00", state, services, memory)

    assert len(out) == 1
    assert "фио пациента" in out[0].text.lower()
    assert state.last_entities.get("patient_name") is None


@pytest.mark.parametrize(
    "phrase",
    [
        "не туда",
        "ты несешь бред",
        "остановись",
        "не так",
        "бред",
        "ошибка",
    ],
)
def test_patient_routing_stream_soft_pause_triggers_cancel_confirm(phrase: str):
    state = SessionState(
        session_id=f"appt-soft-pause-{phrase}",
        last_entities={"appointment_flow_active": True},
    )
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["date_from", "time_from"])

    out = _run_stream_once(phrase, state, services, memory)

    assert len(out) == 1
    assert "отменить текущий процесс записи" in out[0].text.lower()
    assert state.last_entities.get("appointment_cancel_pending") is True


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
        dialog=DialogState(
            label="APPOINTMENT",
            phase=AppointmentPhase.CONFIRM,
            entities={"doctor_name": "Ким Татьяна Александровна"},
        ),
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
    assert state.dialog.label == "OTHER"
    assert state.dialog.phase == ""


def test_patient_routing_stream_structured_doctor_info_sets_secondary_offer_pending(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="DOCTOR_INFO",
                confidence=0.85,
                entities={"secondary_intents": ["DOCTOR_SCHEDULE"]},
                flags={"rule_doctor_info"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(
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
        )

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(session_id="secondary-structured", last_entities={})
    services = Services()
    memory = MemoryStore()

    out = _run_stream_once("Какие кардиологи принимают?", state, services, memory)

    assert out
    assert "Хальметова" in out[0].text
    assert state.last_entities.get("_secondary_offer_pending") is True


def test_route_message_secondary_offer_doctor_reply_activates_schedule(monkeypatch):
    async def fake_resolve_doctor_name(_text: str):
        return "Хальметова Алина Алексеевна"

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(
            items={
                "doctor_schedule": {
                    "schedule": [
                        {
                            "fio": "Хальметова Алина Алексеевна",
                            "regions": ["Ленина 5"],
                            "schedule": {"Ленина 5": [{"date": "2026-03-19", "slots": ["12:00"]}]},
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(
        session_id="secondary-doctor-reply",
        last_entities={
            "_secondary_offer_pending": True,
            "_secondary_queue": ["DOCTOR_SCHEDULE"],
            "secondary_intents": ["DOCTOR_SCHEDULE"],
        },
    )
    services = Services()
    services.resolve_doctor_name = fake_resolve_doctor_name
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message("Хальметова, да", state, services, memory)
    )

    assert decision.label == "DOCTOR_SCHEDULE"
    assert decision.entities.get("doctor_name") == "Хальметова Алина Алексеевна"
    assert plan.label == "DOCTOR_SCHEDULE"
    assert state.last_entities.get("_secondary_offer_pending") is False


def test_route_message_verified_doctor_reply_after_doctor_info_promotes_schedule(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="TEST_RESULT",
                confidence=0.62,
                entities={"doctor_name": "Хальметова Алина Алексеевна"},
                flags={"doctor_name_verified"},
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
        session_id="doctor-info-to-schedule",
        last_entities={"_last_label": "DOCTOR_INFO"},
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Хальметова, да",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "DOCTOR_SCHEDULE"
    assert "flow_doctor_info_to_schedule" in decision.flags
    assert plan.label == "DOCTOR_SCHEDULE"


def test_route_message_resolves_short_doctor_reply_after_doctor_info(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.9,
                entities={},
                flags=set(),
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="llm",
        )

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    class _FakeServices(Services):
        async def resolve_doctor_name(self, raw_text_or_name: str) -> str | None:
            if "хальметова" in str(raw_text_or_name or "").lower():
                return "Хальметова Алина Алексеевна"
            return None

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(
        session_id="doctor-info-short-reply-resolve",
        last_entities={"_last_label": "DOCTOR_INFO"},
    )
    services = _FakeServices()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Хальметова, да",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "DOCTOR_SCHEDULE"
    assert decision.entities.get("doctor_name") == "Хальметова Алина Алексеевна"
    assert "flow_doctor_info_to_schedule" in decision.flags
    assert plan.label == "DOCTOR_SCHEDULE"


def test_route_message_datetime_after_doctor_schedule_promotes_appointment(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="OTHER",
                confidence=0.9,
                entities={},
                flags=set(),
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="llm",
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
        session_id="schedule-to-appointment-datetime",
        last_entities={
            "_last_label": "DOCTOR_SCHEDULE",
            "appointment_flow_active": True,
            "doctor_name": "Хальметова Алина Алексеевна",
            "branch_name": "г. Самара, пр. Ленина, 5",
            "appointment_windows": [
                {"date": "2026-03-19", "time": "12:00", "branch": "г. Самара, пр. Ленина, 5"},
            ],
        },
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "2026-03-19 в 12:00!",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "APPOINTMENT"
    assert (
        "flow_schedule_to_appointment" in decision.flags
        or "flow_datetime_appointment_override" in decision.flags
        or "flow_datetime_appointment_prelock" in decision.flags
        or "appointment_slot_prelock" in decision.flags
    )
    assert state.last_entities.get("doctor_name") == "Хальметова Алина Алексеевна"
    assert state.last_entities.get("date_from") == "2026-03-19"
    assert state.last_entities.get("time_from") == "12:00"
    assert plan.label == "APPOINTMENT"


def test_route_message_datetime_after_doctor_schedule_promotes_from_test_assist(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="TEST_ASSIST",
                confidence=0.88,
                entities={},
                flags={"rule_test_assist"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="llm",
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
        session_id="schedule-to-appointment-datetime-test-assist",
        last_entities={
            "_last_label": "DOCTOR_SCHEDULE",
            "doctor_name": "Дразнин Антон Владимирович",
            "branch_name": "г. Самара, пр. Ленина, 5",
        },
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Да, мне удобно на 10 апреля, на 16:00",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "APPOINTMENT"
    assert (
        "flow_schedule_to_appointment" in decision.flags
        or "appointment_slot_prelock" in decision.flags
    )
    assert plan.label == "APPOINTMENT"


def test_entity_grounder_drops_doctor_like_service_on_unverified_doctor():
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "Евграфову"},
        flags={"doctor_name_unverified"},
        needs_handoff=False,
    )
    state = SessionState(session_id="eg_doctor_like_service_drop", last_entities={})
    services = Services()

    grounded = asyncio.run(
        ground_decision_entities(
            decision=decision,
            user_text="Запишите к Евграфову",
            state=state,
            services=services,
            pending=None,
        )
    )

    assert grounded.entities.get("service_name") is None
    assert "entity_dropped_doctor_like_service_name" in grounded.flags


def test_entity_grounder_drops_doctor_like_service_without_doctor_unverified_flag():
    class _NeverCalledServices:
        async def match_catalog_service(self, *_args, **_kwargs):  # pragma: no cover - should not be called
            raise AssertionError("catalog lookup must not run for doctor-like service collision")

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "Кузнецову"},
        flags={"rule_appointment"},
        needs_handoff=False,
    )
    state = SessionState(session_id="eg_doctor_like_service_drop_no_flag", last_entities={})
    services = _NeverCalledServices()

    grounded = asyncio.run(
        ground_decision_entities(
            decision=decision,
            user_text="Можно записаться к Кузнецову на завтра?",
            state=state,
            services=services,  # type: ignore[arg-type]
            pending=None,
        )
    )

    assert grounded.entities.get("service_name") is None
    assert "entity_dropped_doctor_like_service_name" in grounded.flags


def test_entity_grounder_drops_stale_service_name_in_reschedule_branch_reply():
    class _NeverCalledServices:
        async def match_catalog_service(self, *_args, **_kwargs):  # pragma: no cover - should not be called
            raise AssertionError("catalog lookup must not run for stale service in reschedule")

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "ЭКГ"},
        flags={"flow_appointment_override"},
        needs_handoff=False,
    )
    state = SessionState(
        session_id="eg_drop_stale_service_reschedule",
        last_entities={
            "appointment_action": "reschedule",
            "doctor_name": "Трубин",
            "service_name": "Суточное мониторирование ЭКГ (по Холтеру1/по Холтеру2)",
        },
    )
    services = _NeverCalledServices()

    grounded = asyncio.run(
        ground_decision_entities(
            decision=decision,
            user_text="г. Самара, ул. Победы, 83",
            state=state,
            services=services,  # type: ignore[arg-type]
            pending=None,
        )
    )

    assert grounded.entities.get("service_name") is None
    assert "entity_dropped_stale_service_name_in_reschedule" in grounded.flags


def test_entity_grounder_keeps_service_with_explicit_service_anchor_even_if_doctor_unverified():
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "Холтер"},
        flags={"doctor_name_unverified"},
        needs_handoff=False,
    )
    state = SessionState(session_id="eg_keep_service_with_anchor", last_entities={})
    services = Services()

    grounded = asyncio.run(
        ground_decision_entities(
            decision=decision,
            user_text="Запишите к Евграфову на холтер",
            state=state,
            services=services,
            pending=None,
        )
    )

    assert grounded.entities.get("service_name") == "Холтер"


def test_route_message_city_only_reply_keeps_price_label(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="ADDRESS",
                confidence=0.9,
                entities={},
                flags={"rule_address"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="llm",
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
        session_id="price-city-followup",
        last_entities={"service_name": "ЭКГ"},
    )
    services = Services()
    memory = MemoryStore()
    memory.set_pending(state, label="PRICE", missing_slots=["_any_of:city,branch_name,branch_id"])

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message(
            "Самара",
            state,
            services,
            memory,
        )
    )

    assert decision.label == "PRICE"
    assert "flow_price_city_reply_override" in decision.flags
    assert plan.label == "PRICE"


def test_route_message_operator_offer_yes_handoffs():
    state = SessionState(session_id="operator-offer-yes", last_entities={"_operator_offer_pending": True})
    services = Services()
    memory = MemoryStore()
    memory.set_pending(state, label="OTHER", missing_slots=["operator_offer_confirm"])

    decision, plan, evidence = asyncio.run(router_mod.route_patient_message("да", state, services, memory))

    assert decision.label == "OTHER"
    assert "operator_offer_confirmed" in decision.flags
    assert plan.label == "OTHER"
    payload = evidence.get("operator_offer_response")
    assert isinstance(payload, dict)
    assert payload.get("handoff") is True
    assert "Соединяю с оператором" in str(payload.get("text") or "")
    assert state.last_entities.get("_operator_offer_pending") is None
    assert memory.get_pending(state) is None


def test_route_message_operator_offer_no_keeps_dialog_without_handoff():
    state = SessionState(session_id="operator-offer-no", last_entities={"_operator_offer_pending": True})
    services = Services()
    memory = MemoryStore()
    memory.set_pending(state, label="OTHER", missing_slots=["operator_offer_confirm"])

    decision, plan, evidence = asyncio.run(router_mod.route_patient_message("нет", state, services, memory))

    assert decision.label == "OTHER"
    assert "operator_offer_declined" in decision.flags
    assert plan.label == "OTHER"
    payload = evidence.get("operator_offer_response")
    assert isinstance(payload, dict)
    assert payload.get("handoff") is False
    assert "Хорошо, продолжаем диалог." in str(payload.get("text") or "")
    assert state.last_entities.get("_operator_offer_pending") is None
    assert memory.get_pending(state) is None


def test_build_first_structured_response_compound_price_sets_pending():
    state = SessionState(session_id="compound-price-builder", last_entities={})
    evidence = Evidence(
        items={
            "service_bundle": {
                "clarify_text": (
                    "Вижу в запросе две услуги:\n"
                    "1. УЗДГ сосудов шеи\n"
                    "2. ЛПНП\n\n"
                    "Если хотите, сначала покажу по УЗДГ сосудов шеи."
                ),
                "compound_price_services": ["УЗДГ сосудов шеи", "ЛПНП"],
                "compound_price_default_service": "УЗДГ сосудов шеи",
            }
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
        user_text="Сколько стоит УЗДГ и ЛПНП?",
    )

    assert env is not None
    pending = state.last_entities.get("_compound_price_pending")
    assert isinstance(pending, dict)
    assert pending.get("default_service") == "УЗДГ сосудов шеи"
    assert pending.get("services") == ["УЗДГ сосудов шеи", "ЛПНП"]


def test_route_message_compound_price_pending_yes_selects_default_service(monkeypatch):
    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(items={"service_bundle": {"service_name": "УЗДГ сосудов шеи"}})

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(
        session_id="compound-price-yes",
        last_entities={
            "_compound_price_pending": {
                "services": ["УЗДГ сосудов шеи", "ЛПНП"],
                "default_service": "УЗДГ сосудов шеи",
            },
            "city": "Самара",
            "_secondary_queue": ["TEST_ASSIST", "ADDRESS"],
            "_secondary_offer_pending": True,
        },
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message("можно", state, services, memory)
    )

    assert decision.label == "PRICE"
    assert decision.entities.get("service_name") == "УЗДГ сосудов шеи"
    assert plan.label == "PRICE"
    assert plan.steps and plan.steps[0].tool == "service_bundle_info"
    assert state.last_entities.get("service_name") == "УЗДГ сосудов шеи"
    assert state.last_entities.get("_compound_price_pending") is None
    assert state.last_entities.get("_secondary_queue") is None


def test_route_message_compound_price_pending_specific_service_reply_selects_that_service(monkeypatch):
    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(items={"service_bundle": {"service_name": "ЛПНП"}})

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)

    state = SessionState(
        session_id="compound-price-lpnp",
        last_entities={
            "_compound_price_pending": {
                "services": ["УЗДГ сосудов шеи", "ЛПНП"],
                "default_service": "УЗДГ сосудов шеи",
            },
            "city": "Самара",
        },
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message("ЛПНП", state, services, memory)
    )

    assert decision.label == "PRICE"
    assert decision.entities.get("service_name") == "ЛПНП"
    assert plan.label == "PRICE"
    assert state.last_entities.get("service_name") == "ЛПНП"
    assert state.last_entities.get("_compound_price_pending") is None


def test_route_message_compound_price_pending_other_question_clears_and_routes_normally(monkeypatch):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label="NEWS",
                confidence=0.85,
                entities={},
                flags={"rule_news"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(items={"news": {"items": []}})

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)

    state = SessionState(
        session_id="compound-price-other",
        last_entities={
            "_compound_price_pending": {
                "services": ["УЗДГ сосудов шеи", "ЛПНП"],
                "default_service": "УЗДГ сосудов шеи",
            },
            "_secondary_queue": ["TEST_ASSIST", "ADDRESS"],
        },
    )
    services = Services()
    memory = MemoryStore()

    decision, plan, _evidence = asyncio.run(
        router_mod.route_patient_message("Какие сейчас акции?", state, services, memory)
    )

    assert decision.label == "NEWS"
    assert plan.label == "NEWS"
    assert state.last_entities.get("_compound_price_pending") is None
    assert state.last_entities.get("_secondary_queue") is None


def test_appointment_dropped_unverified_target_does_not_offer_branches():
    # BUG-2026-06-01-01 (class invariant, B′): when the APPOINTMENT target was
    # dropped by the grounder as unverified (entity_dropped_unverified_service_name)
    # and there is no valid doctor/specialty, the planner must NOT route to
    # address_info (the blind 5-branch list). Flag-based → covers ANY junk target:
    # full ФИО, frequent 2-word name, bare surname.
    for leaked in ("Турмухамбетова Балслу Турмурадовна", "Иванов Пётр", "Сидоренко"):
        state = SessionState(
            session_id="dropped-target",
            last_entities={"appointment_action": "book", "service_name": leaked, "city": "Самара"},
        )
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.75,
            entities={"appointment_action": "book"},
            flags={"entity_dropped_unverified_service_name", "rule_appointment"},
            needs_handoff=False,
        )
        plan = build_plan(decision, state, f"Записаться к {leaked}", memory=MemoryStore())
        assert "address_info" not in [s.tool for s in plan.steps], leaked
        assert state.last_entities.get("_appointment_unbookable_target") is True, leaked


def test_appointment_unbookable_target_marker_yields_honest_refusal():
    # The marker set by the planner makes the appointment step honestly refuse
    # (booking only for Samara branches → operator), not list branches blindly.
    state = SessionState(
        session_id="unbookable",
        last_entities={"_appointment_unbookable_target": True, "appointment_action": "book"},
    )
    res = _build_appointment_step_response("APPOINTMENT", Evidence(), state, Services(), MemoryStore())
    assert res is not None
    assert "оператор" in res.text.lower()
    assert "по адресам" not in res.text.lower()
    # marker is popped → does not leak into the next turn
    assert state.last_entities.get("_appointment_unbookable_target") is None


def test_quickfill_does_not_resurrect_grounder_rejected_service_name():
    # A′ root (BUG-2026-06-01-01): the grounder sanitizes decision.entities, but
    # quick-fill writes straight into state.last_entities (which the planner reads).
    # When the grounder rejected service_name THIS turn (any *_service_name drop
    # flag), quick-fill must NOT re-inject it from the raw user text — otherwise
    # the dropped value bypasses the grounder. Class-level: parametrized over the
    # rejection flags AND over junk values (full FIO / 2-word name / bare surname).
    reject_flags = [
        "entity_dropped_unverified_service_name",
        "entity_dropped_doctor_like_service_name",
        "entity_dropped_stale_service_name_in_reschedule",
    ]
    leaked_values = ["Турмухамбетова Балслу Турмурадовна", "Иванов Пётр", "Сидоренко"]
    for flag in reject_flags:
        for leaked in leaked_values:
            quick = {"service_name": leaked, "appointment_action": "book"}
            out = router_mod._suppress_grounder_rejected_slots(quick, {flag, "rule_appointment"})
            assert "service_name" not in out, (flag, leaked)
            # неотвергнутые слоты не трогаем
            assert out.get("appointment_action") == "book", (flag, leaked)
    # БЕЗ флага отказа грундера — service_name из quick-fill сохраняется (не пере-подавляем)
    keep = router_mod._suppress_grounder_rejected_slots(
        {"service_name": "УЗИ щитовидной железы"}, {"rule_appointment"}
    )
    assert keep.get("service_name") == "УЗИ щитовидной железы"


def test_quickfill_reextracts_fio_as_service_then_suppressed():
    # Ties the suppressor to the REAL leak primitive: on "Записаться к <ФИО>" the
    # raw text is mis-extracted as a service phrase (what _fill_appointment_entities
    # merges into state.last_entities). With the grounder's drop flag present, the
    # suppressor removes it so it never reaches the planner.
    from messengers_router.policies import extract_service_phrase

    phrase = extract_service_phrase("Записаться к Турмухамбетова Балслу Турмурадовна")
    assert phrase, "precondition: raw FIO is mis-extracted as a service phrase (the leak source)"
    quick = {"service_name": phrase, "appointment_action": "book"}
    out = router_mod._suppress_grounder_rejected_slots(quick, {"entity_dropped_unverified_service_name"})
    assert "service_name" not in out


def test_appointment_dropped_target_with_stale_specialty_still_refuses():
    # A′-2 (BUG-2026-06-01-01): a specialty left over in state.last_entities from a
    # PRIOR topic must NOT satisfy the unbookable-target guard when THIS turn's
    # grounded decision dropped the booking target (no active flow). Otherwise a
    # stale specialty silently substitutes the named (non-bookable) target → blind
    # branch offer, re-opening the bug through the specialty door. The guard now
    # reads the GROUNDED decision.entities, not stale state. Class-level: many specialties.
    for stale_specialty in ("уролог", "кардиолог", "эндокринолог"):
        state = SessionState(
            session_id="stale-spec",
            last_entities={"appointment_action": "book", "specialty": stale_specialty, "city": "Самара"},
        )
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.75,
            entities={"appointment_action": "book"},  # grounded this turn: NO target
            flags={"entity_dropped_unverified_service_name", "rule_appointment"},
            needs_handoff=False,
        )
        plan = build_plan(decision, state, "Записаться к Несуществующему Врачу Ивановичу", memory=MemoryStore())
        assert "address_info" not in [s.tool for s in plan.steps], stale_specialty
        assert state.last_entities.get("_appointment_unbookable_target") is True, stale_specialty


def test_appointment_dropped_target_clean_state_refuses_not_clarifies():
    # With the A′ leak fixed there is no stale service_name in state either; a
    # dropped target must still yield the honest Samara-only refusal (early guard
    # fires BEFORE missing-slots), not a generic "who/what?" clarify pending.
    state = SessionState(
        session_id="dropped-clean",
        last_entities={"appointment_action": "book", "city": "Самара"},
    )
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.75,
        entities={"appointment_action": "book"},
        flags={"entity_dropped_unverified_service_name", "rule_appointment"},
        needs_handoff=False,
    )
    mem = MemoryStore()
    plan = build_plan(decision, state, "Записаться к Несуществующему Врачу", memory=mem)
    assert plan.steps == []
    assert state.last_entities.get("_appointment_unbookable_target") is True
    assert mem.get_pending(state) is None  # honest refuse, NOT a clarify loop


def test_appointment_active_flow_specialty_survives_dropped_service():
    # Regression guard: inside an ACTIVE appointment flow a specialty/doctor held
    # in state MUST keep the booking alive even if THIS turn's service was dropped —
    # we must not over-refuse a live flow.
    state = SessionState(
        session_id="active-spec",
        last_entities={
            "appointment_action": "book",
            "specialty": "уролог",
            "city": "Самара",
            "appointment_flow_active": True,
        },
    )
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.75,
        entities={"appointment_action": "book"},
        flags={"entity_dropped_unverified_service_name", "rule_appointment"},
        needs_handoff=False,
    )
    build_plan(decision, state, "на анализ", memory=MemoryStore())
    assert state.last_entities.get("_appointment_unbookable_target") is not True


def test_pending_price_fill_resolves_service_via_pending_label():
    # P2 (BUG-2026-06-02-02): a bare service reply to a PRICE clarify must resolve
    # even when the current turn's label (and thus _last_label, clobbered by the
    # merge of the current decision) is NOT PRICE. Drive the catalog resolution by
    # the pending label, not by the heuristic gate. Class-level: several services.
    for query, hint in [
        ("Общий анализ крови", "анализ крови"),
        ("УЗИ щитовидной железы", "щитовид"),
    ]:
        st = {"_last_label": "TEST_ASSIST", "city": "Самара"}  # _last_label clobbered this turn
        q = quick_fill_entities_from_text(query, st, ["service_name"], Services(), pending_label="PRICE")
        assert q.get("service_name"), (query, "should resolve via pending_label=PRICE")
        assert hint in q["service_name"].lower(), (query, q.get("service_name"))
    # без pending_label и не-PRICE контекста бот не резолвит (поведение не расширяем)
    st = {"_last_label": "TEST_ASSIST"}
    q = quick_fill_entities_from_text("Общий анализ крови", st, ["service_name"], Services())
    assert not q.get("service_name")


def test_quickfill_keeps_catalog_valid_service_despite_drop_flag():
    # P3 (BUG-2026-06-02-02): the A′-1 suppressor must KEEP a catalog-resolved
    # service_name (grounded-by-source — e.g. the legit answer to a pending PRICE
    # clarify) even when the current decision carries a service drop flag. It may
    # only strip the ungrounded raw re-extract (a FIO). Regression of f12096d.
    keep = router_mod._suppress_grounder_rejected_slots(
        {"service_name": "Общий анализ крови (Le, Er, Hb, СОЭ)"},
        {"entity_dropped_unverified_service_name", "rule_test_assist"},
    )
    assert keep.get("service_name") == "Общий анализ крови (Le, Er, Hb, СОЭ)"
    # ...а ФИО-мусор по-прежнему срезается (каталог его не подтверждает)
    for junk in ("Турмухамбетова Балслу Турмурадовна", "Иванов Пётр", "Сидоренко"):
        out = router_mod._suppress_grounder_rejected_slots(
            {"service_name": junk, "appointment_action": "book"},
            {"entity_dropped_unverified_service_name"},
        )
        assert "service_name" not in out, junk
        assert out.get("appointment_action") == "book", junk


@pytest.mark.parametrize(
    "text,svc,expected",
    [
        ("узи жтк + почки", "", True),
        ("Можно записаться на узи жтк + почки", "", True),
        ("УЗИ печени + поджелудочная + почки", "", True),
        ("А плюс поджелудочная", "Ультразвуковое исследование печени и желчного пузыря", True),
        ("И плюс почки", "Ультразвуковое исследование печени и желчного пузыря", True),
        ("УЗИ печени и желчного пузыря", "", False),  # одна каталожная позиция
        ("УЗИ почек", "", False),
        ("записаться к кардиологу", "", False),
        ("плюс поджелудочная", "прием уролога", False),  # активная запись не УЗИ
    ],
)
def test_is_compound_uzi_request(text, svc, expected):
    # BUG-2026-06-02-08: детектор УЗИ-набора (>1 органа через явное сложение).
    from messengers_router.policies import is_compound_uzi_request

    assert is_compound_uzi_request(text, svc) is expected


def test_compound_uzi_appointment_offers_operator_not_single_service():
    # BUG-2026-06-02-08: УЗИ-запись с >1 органом → честно к оператору (по решению
    # владельца), а НЕ подбор одной услуги / слепые филиалы. Класс-инвариант:
    # planner ставит маркер + пустой план → builder отдаёт оффер оператора.
    cases = [
        ("Можно записаться на узи жтк + почки", {"appointment_action": "book", "city": "Самара"}),
        (
            "А плюс поджелудочная",
            {
                "appointment_action": "book",
                "city": "Самара",
                "service_name": "Ультразвуковое исследование печени и желчного пузыря",
                "appointment_flow_active": True,
            },
        ),
    ]
    for text, state_ent in cases:
        state = SessionState(session_id="cmp-uzi", last_entities=dict(state_ent))
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.8,
            entities={"appointment_action": "book"},
            flags={"rule_appointment"},
            needs_handoff=False,
        )
        plan = build_plan(decision, state, text, memory=MemoryStore())
        assert plan.steps == [], text
        assert state.last_entities.get("_appointment_compound_uzi") is True, text
        res = _build_appointment_step_response("APPOINTMENT", Evidence(), state, Services(), MemoryStore())
        assert res is not None, text
        assert "оператор" in res.text.lower(), (text, res.text)
        assert "по адресам" not in res.text.lower(), (text, res.text)
        # маркер popped → не утечёт в следующий ход
        assert state.last_entities.get("_appointment_compound_uzi") is None, text
