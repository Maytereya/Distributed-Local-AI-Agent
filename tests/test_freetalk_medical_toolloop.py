import asyncio
from types import SimpleNamespace

from localragagent.freetalk.medical_toolloop import (
    MedicalToolLoopContext,
    MedicalToolLoopRuntime,
    execute_medical_tool_loop,
)
from localragagent.freetalk.contracts import ToolCallResult


async def _noop_async(*_args, **_kwargs):
    return None


async def _dispatcher_not_found(*_args, **_kwargs):
    return SimpleNamespace(error="", found=False, payload={})


def _public_entities(entities):
    return dict(entities or {})


def _merge_entities(left, right):
    out = dict(left or {})
    out.update(dict(right or {}))
    return out


def _missing_slots_from_payload(_tool_name, _payload):
    return []


def _merge_missing_slots(base, extra):
    seen = set()
    out = []
    for slot in [*(base or []), *(extra or [])]:
        name = str(slot or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _filter_missing_slots_by_entities(missing_slots, entities):
    entities = dict(entities or {})
    out = []
    for slot in missing_slots or []:
        name = str(slot or "").strip()
        if not name:
            continue
        if name == "doctor_name" and str(entities.get("doctor_name") or "").strip():
            continue
        if name == "specialty" and str(entities.get("specialty") or "").strip():
            continue
        out.append(name)
    return out


def test_medical_toolloop_restores_plan_based_clarify_slots_when_not_found():
    saved = {}

    runtime = MedicalToolLoopRuntime(
        adapter=SimpleNamespace(
            prepare_tool_call=lambda **kwargs: SimpleNamespace(
                tool_name=kwargs["tool_name"],
                backend_query=kwargs["user_message"],
                backend_entities=kwargs["entities"],
            )
        ),
        dispatcher=SimpleNamespace(call=_dispatcher_not_found),
        max_tool_steps=3,
        public_entities=_public_entities,
        schedule_payload_stats=lambda _payload: {},
        render_tool_reply=_noop_async,
        post_tool_verify=_noop_async,
        remember_doctor_from_tool_result=_noop_async,
        save_session_entity_memory=_noop_async,
        save_dialog_state=lambda session_id, state: _save_state(saved, session_id, state),
        clear_dialog_state=_noop_async,
        merge_entities=_merge_entities,
        tool_payload_memory_entities=lambda _tool_name, _payload: {},
        missing_slots_from_tool_payload=_missing_slots_from_payload,
        merge_missing_slots=_merge_missing_slots,
        filter_missing_slots_by_entities=_filter_missing_slots_by_entities,
    )
    context = MedicalToolLoopContext(
        session_id="medical-not-found-specialty",
        user_message="Привет! Дай информацию по кардиологам клиники",
        intent="doctor_info",
        confidence=0.91,
        clarify_count=0,
        phase="",
        tool_plan=["doctors_info", "doctors_schedule_week"],
        missing_slots=[],
        entities={},
        candidate_entities={},
    )

    reply = asyncio.run(execute_medical_tool_loop(context=context, runtime=runtime))

    assert "фамилию врача или специальность" in reply.text.lower()
    state = saved["state"]
    assert state.flow_kind == "clarify"
    assert set(state.expected_slots) == {"doctor_name", "specialty"}


def test_medical_toolloop_returns_tech_unavailable_without_trying_next_tool():
    calls = []
    cleared = {"value": False}

    async def dispatcher_call(tool_name, _query, entities=None):
        _ = entities
        calls.append(tool_name)
        return ToolCallResult(
            tool_name=tool_name,
            payload={
                "ready": False,
                "note": "result_tech_unavailable",
                "result_preview": "По техническим причинам сейчас не удаётся загрузить результаты анализов.",
                "degraded": True,
            },
            found=True,
            outcome="tech_unavailable",
            degraded=True,
        )

    async def render_tool_reply(*, user_message, tool_name, tool_payload):
        _ = user_message, tool_name
        return str(tool_payload.get("result_preview") or "")

    async def post_tool_verify(*_args, **_kwargs):
        raise AssertionError("post-tool verifier must not run for tech_unavailable")

    async def clear_state(_session_id):
        cleared["value"] = True

    runtime = MedicalToolLoopRuntime(
        adapter=SimpleNamespace(
            prepare_tool_call=lambda **kwargs: SimpleNamespace(
                tool_name=kwargs["tool_name"],
                backend_query=kwargs["user_message"],
                backend_entities=kwargs["entities"],
            ),
            normalize_tool_payload=lambda **kwargs: SimpleNamespace(ft_payload=kwargs["payload"]),
        ),
        dispatcher=SimpleNamespace(call=dispatcher_call),
        max_tool_steps=3,
        public_entities=_public_entities,
        schedule_payload_stats=lambda _payload: {},
        render_tool_reply=render_tool_reply,
        post_tool_verify=post_tool_verify,
        remember_doctor_from_tool_result=_noop_async,
        save_session_entity_memory=_noop_async,
        save_dialog_state=_noop_async,
        clear_dialog_state=clear_state,
        merge_entities=_merge_entities,
        tool_payload_memory_entities=lambda _tool_name, _payload: {},
        missing_slots_from_tool_payload=_missing_slots_from_payload,
        merge_missing_slots=_merge_missing_slots,
        filter_missing_slots_by_entities=_filter_missing_slots_by_entities,
    )
    context = MedicalToolLoopContext(
        session_id="medical-tech-unavailable",
        user_message="Проверь результат анализа",
        intent="test_result",
        confidence=0.91,
        clarify_count=0,
        phase="",
        tool_plan=["test_result_status", "test_assist"],
        missing_slots=[],
        entities={
            "result_surname": "Иванов",
            "result_year_of_birth": "1990",
            "result_analysis_code": "Бг",
            "result_analysis_number": "12345",
        },
        candidate_entities={},
    )

    reply = asyncio.run(execute_medical_tool_loop(context=context, runtime=runtime))

    assert calls == ["test_result_status"]
    assert "техническим причинам" in reply.text.lower()
    assert reply.outcome == "tech_unavailable"
    assert reply.degraded is True
    assert cleared["value"] is True


async def _save_state(store, session_id, state):
    store["session_id"] = session_id
    store["state"] = state
