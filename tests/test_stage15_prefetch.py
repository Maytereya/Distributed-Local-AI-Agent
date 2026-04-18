"""Stage 15 (full OPTION B) — parallel doctor verify + speculative service
catalog match.

Covers:
- ``_plan_service_catalog_prefetch`` guard matrix
- ``_verify_decision_and_prefetch_catalog`` parallel fast-path
- ``_inject_catalog_candidates`` prefetch reuse vs re-fetch fallback
"""

from __future__ import annotations

import asyncio

from messengers_router import router as router_mod
from messengers_router.mess_types import RouteDecision, SessionState


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _plan_service_catalog_prefetch — pure guard matrix


def _mk_state(**kwargs) -> SessionState:
    s = SessionState(session_id="stage15")
    s.last_entities.update(kwargs)
    return s


def test_plan_prefetch_returns_query_on_safe_gate():
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={},  # no service_name, no doctor_name
        source="llm_primary",
    )
    plan = router_mod._plan_service_catalog_prefetch(
        decision, _mk_state(), "хочу записаться на холтер"
    )
    assert plan == {"query": "хочу записаться на холтер", "current_service_name": ""}


def test_plan_prefetch_none_when_label_not_service_eligible():
    decision = RouteDecision(label="DOCTOR_SCHEDULE", confidence=0.9, entities={})
    assert router_mod._plan_service_catalog_prefetch(decision, _mk_state(), "график") is None


def test_plan_prefetch_none_when_service_name_present():
    decision = RouteDecision(
        label="APPOINTMENT", confidence=0.9, entities={"service_name": "УЗИ"}
    )
    assert router_mod._plan_service_catalog_prefetch(decision, _mk_state(), "уzi") is None


def test_plan_prefetch_none_when_doctor_name_in_decision():
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={"doctor_name": "Дразнину"},
    )
    # raw doctor_name — verify may resolve it and flip the APPOINTMENT gate;
    # speculation is unsafe.
    assert router_mod._plan_service_catalog_prefetch(decision, _mk_state(), "текст") is None


def test_plan_prefetch_none_when_doctor_name_in_state():
    decision = RouteDecision(label="APPOINTMENT", confidence=0.9, entities={})
    state = _mk_state(doctor_name="Дразнин Антон Владимирович")
    assert router_mod._plan_service_catalog_prefetch(decision, state, "ещё вопрос") is None


def test_plan_prefetch_none_on_overwrite_doctor():
    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={},
        context_action="overwrite_doctor",
    )
    assert router_mod._plan_service_catalog_prefetch(decision, _mk_state(), "нужен Ким") is None


def test_plan_prefetch_none_on_price_with_specialty():
    decision = RouteDecision(
        label="PRICE", confidence=0.9, entities={"specialty": "кардиолог"}
    )
    assert router_mod._plan_service_catalog_prefetch(decision, _mk_state(), "цена") is None


def test_plan_prefetch_uses_test_name_when_present():
    decision = RouteDecision(
        label="TEST_ASSIST",
        confidence=0.9,
        entities={"test_name": "ТТГ"},
    )
    plan = router_mod._plan_service_catalog_prefetch(decision, _mk_state(), "подскажите")
    assert plan == {"query": "ТТГ", "current_service_name": ""}


def test_plan_prefetch_uses_state_current_service_name():
    decision = RouteDecision(label="APPOINTMENT", confidence=0.9, entities={})
    state = _mk_state(service_name="Анализ крови (ОАК)")
    plan = router_mod._plan_service_catalog_prefetch(decision, state, "да")
    assert plan == {"query": "да", "current_service_name": "Анализ крови (ОАК)"}


# ---------------------------------------------------------------------------
# _verify_decision_and_prefetch_catalog — parallel fast-path


def test_verify_and_prefetch_runs_both_in_parallel_when_safe(monkeypatch):
    calls: list[str] = []

    async def fake_verify(decision, services, user_text):
        _ = services, user_text
        calls.append("verify")
        return decision

    class FakeServices:
        async def match_catalog_service(self, query, *, current_service_name=""):
            _ = current_service_name
            calls.append("service")
            return {"status": "exact", "canonical": query.upper(), "query": query}

    monkeypatch.setattr(router_mod, "_verify_doctor_entity", fake_verify)

    decision = RouteDecision(label="APPOINTMENT", confidence=0.9, entities={})
    state = _mk_state()
    verified, prefetch = run(
        router_mod._verify_decision_and_prefetch_catalog(
            decision=decision,
            user_text="холтер",
            state=state,
            services=FakeServices(),
        )
    )
    assert verified is decision
    assert prefetch is not None
    assert prefetch["query"] == "холтер"
    assert prefetch["match"] == {"status": "exact", "canonical": "ХОЛТЕР", "query": "холтер"}
    assert set(calls) == {"verify", "service"}  # both ran


def test_verify_and_prefetch_skips_service_fetch_when_guard_blocks(monkeypatch):
    calls: list[str] = []

    async def fake_verify(decision, services, user_text):
        _ = services, user_text
        calls.append("verify")
        return decision

    class FakeServices:
        async def match_catalog_service(self, *_a, **_k):  # pragma: no cover
            calls.append("service")
            raise AssertionError("service fetch must not fire when guard blocks")

    monkeypatch.setattr(router_mod, "_verify_doctor_entity", fake_verify)

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={"doctor_name": "Дразнину"},
    )
    verified, prefetch = run(
        router_mod._verify_decision_and_prefetch_catalog(
            decision=decision,
            user_text="к дразнину",
            state=_mk_state(),
            services=FakeServices(),
        )
    )
    assert verified is decision
    assert prefetch is None
    assert calls == ["verify"]


# ---------------------------------------------------------------------------
# _inject_catalog_candidates prefetch reuse vs re-fetch fallback


def test_inject_reuses_prefetch_when_query_matches(monkeypatch):
    refetch_calls: list[str] = []

    class FakeServices:
        async def match_catalog_doctor(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("doctor match must not fire for this case")

        async def match_catalog_service(self, query, *, current_service_name=""):
            _ = current_service_name
            refetch_calls.append(query)
            raise AssertionError("service match must reuse prefetch, not re-fetch")

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={},
        source="llm_primary",
    )
    state = _mk_state()
    prefetch = {
        "query": "хочу на холтер",
        "match": {"status": "exact", "canonical": "Холтер-24", "query": "хочу на холтер"},
    }
    out = run(
        router_mod._inject_catalog_candidates(
            decision,
            user_text="хочу на холтер",
            state=state,
            services=FakeServices(),
            prefetched_service=prefetch,
        )
    )
    assert out.entities.get("service_name") == "Холтер-24"
    assert "catalog_service_exact" in out.flags
    assert refetch_calls == []


def test_inject_refetches_when_prefetch_query_differs(monkeypatch):
    observed_query: list[str] = []

    class FakeServices:
        async def match_catalog_service(self, query, *, current_service_name=""):
            _ = current_service_name
            observed_query.append(query)
            return {"status": "fuzzy", "canonical": "Холтер", "query": query}

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={},  # post-verify: no service_name, no doctor
    )
    state = _mk_state()
    # prefetch was done for a stale query that no longer matches post-verify input.
    stale_prefetch = {
        "query": "какой-то старый текст",
        "match": {"status": "exact", "canonical": "WRONG", "query": "какой-то старый текст"},
    }
    out = run(
        router_mod._inject_catalog_candidates(
            decision,
            user_text="хочу на холтер",
            state=state,
            services=FakeServices(),
            prefetched_service=stale_prefetch,
        )
    )
    assert observed_query == ["хочу на холтер"]
    # fuzzy candidate applied from the re-fetch, not from the stale prefetch
    assert out.entities.get("_catalog_service_candidate") == "Холтер"
    assert "catalog_service_fuzzy_candidate" in out.flags


def test_inject_ignores_prefetch_when_post_verify_gate_closed():
    class FakeServices:
        async def match_catalog_service(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("service fetch must be skipped when gate is closed")

        async def match_catalog_doctor(self, *_a, **_k):  # pragma: no cover
            raise AssertionError("doctor fetch must be skipped for this case")

    decision = RouteDecision(
        label="APPOINTMENT",
        confidence=0.9,
        entities={"service_name": "УЗИ"},  # post-verify gate closed
    )
    prefetch = {
        "query": "узи",
        "match": {"status": "exact", "canonical": "УЗИ-щитовидки", "query": "узи"},
    }
    out = run(
        router_mod._inject_catalog_candidates(
            decision,
            user_text="узи",
            state=_mk_state(),
            services=FakeServices(),
            prefetched_service=prefetch,
        )
    )
    # Gate was closed: prefetch result must be ignored and decision unchanged.
    assert out.entities.get("service_name") == "УЗИ"
    assert "catalog_service_exact" not in out.flags
