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
