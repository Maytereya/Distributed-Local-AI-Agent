import asyncio

from messengers_router.entity_grounder import ground_decision_entities
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


def test_grounder_drops_unverified_doctor_name(monkeypatch):
    svc = Services()

    async def fake_resolve(_raw: str):
        return None

    monkeypatch.setattr(svc, "resolve_doctor_name", fake_resolve)

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"doctor_name": "Хочу"},
        flags=set(),
    )
    out = run(
        ground_decision_entities(
            decision=decision,
            user_text="хочу записаться",
            state=SessionState(session_id="s1"),
            services=svc,
            pending={"label": "APPOINTMENT", "missing": ["_any_of:doctor_id,doctor_name,specialty,service_name"]},
        )
    )

    assert "doctor_name" not in out.entities
    assert "entity_dropped_unverified_doctor_name" in out.flags


def test_grounder_canonicalizes_doctor_name(monkeypatch):
    svc = Services()

    async def fake_resolve(_raw: str):
        return "Дразнин Антон Владимирович"

    monkeypatch.setattr(svc, "resolve_doctor_name", fake_resolve)

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"doctor_name": "Дразнину"},
        flags=set(),
    )
    out = run(
        ground_decision_entities(
            decision=decision,
            user_text="к Дразнину",
            state=SessionState(session_id="s2"),
            services=svc,
            pending={"label": "APPOINTMENT", "missing": ["_any_of:doctor_id,doctor_name,specialty,service_name"]},
        )
    )

    assert out.entities.get("doctor_name") == "Дразнин Антон Владимирович"


def test_grounder_drops_generic_service_name():
    svc = Services()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "Можно записаться к врачу"},
        flags=set(),
    )
    out = run(
        ground_decision_entities(
            decision=decision,
            user_text="можно записаться к врачу",
            state=SessionState(session_id="s3"),
            services=svc,
            pending={"label": "APPOINTMENT", "missing": ["_any_of:doctor_id,doctor_name,specialty,service_name"]},
        )
    )
    assert "service_name" not in out.entities
    assert "entity_dropped_unverified_service_name" in out.flags


def test_grounder_extracts_service_from_user_text():
    svc = Services()
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.8,
        entities={"service_name": "some_llm_noise"},
        flags=set(),
    )
    out = run(
        ground_decision_entities(
            decision=decision,
            user_text="на торакоцентез",
            state=SessionState(session_id="s4"),
            services=svc,
            pending={"label": "APPOINTMENT", "missing": ["_any_of:doctor_id,doctor_name,specialty,service_name"]},
        )
    )
    assert out.entities.get("service_name") == "Торакоцентез"


def test_grounder_normalizes_city_typo():
    svc = Services()
    decision = RouteDecision(
        label="ADDRESS",
        confidence=0.8,
        entities={"city": "Самраа"},
        flags=set(),
    )
    out = run(
        ground_decision_entities(
            decision=decision,
            user_text="Самраа",
            state=SessionState(session_id="s5"),
            services=svc,
            pending={"label": "ADDRESS", "missing": ["_any_of:city,branch_name,branch_id"]},
        )
    )
    assert out.entities.get("city") == "Самара"


def test_grounder_keeps_patient_name_for_other_when_pending_appointment():
    svc = Services()
    decision = RouteDecision(
        label="OTHER",
        confidence=0.3,
        entities={"patient_name": "Рахманов Владимир"},
        flags=set(),
    )
    out = run(
        ground_decision_entities(
            decision=decision,
            user_text="Рахманов Владимир",
            state=SessionState(session_id="s6"),
            services=svc,
            pending={"label": "APPOINTMENT", "missing": ["patient_name"]},
        )
    )
    assert out.entities.get("patient_name") == "Рахманов Владимир"


def test_keep_nonbookable_generic_hint_overrides_prefix_guard():
    # Regression: a deterministic generic class hint from nonbookable_service_hint
    # («анализы»/«ЭКГ») must be kept verbatim under the walk-in flag even though it
    # shares no token-prefix with the specific instance the patient named — otherwise
    # the grounder fuzzy-matches the text to a wrong catalog row («профиль 2»).
    from messengers_router.entity_grounder import (
        _should_keep_nonbookable_service_name_as_is,
    )

    walkin = {"policy_nonbookable_walkin"}
    user_text = "Диабетический профиль 1 где можно сдать?"
    assert _should_keep_nonbookable_service_name_as_is(
        raw_value="анализы", decision_flags=walkin, user_text=user_text
    )
    assert _should_keep_nonbookable_service_name_as_is(
        raw_value="ЭКГ", decision_flags=walkin, user_text="где сделать ЭКГ?"
    )
    # Anti-hallucination guard for free-form values stays intact.
    assert not _should_keep_nonbookable_service_name_as_is(
        raw_value="Маммопластика увеличение груди",
        decision_flags=walkin,
        user_text="Сдать витамин Д где?",
    )
    # The generic hint is kept only inside the walk-in flow.
    assert not _should_keep_nonbookable_service_name_as_is(
        raw_value="анализы", decision_flags=set(), user_text=user_text
    )
