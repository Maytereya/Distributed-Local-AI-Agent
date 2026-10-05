"""Срочное посреди записи и из другого города — шаблон скорой и оператор (05.10).

Ревью DLA (находка LOGIC-1, BUG-2026-10-05-URGENT-INSIDE-APPOINTMENT): срочность
проверялась только внутри пайплайна (`early_guards`), а охрана города и предпроверка
записи (`run_appointment_precheck`) стояли раньше. «задыхаюсь» на подтверждении записи
получало «Подтвердите запись…», фраза с «?», «извините» или «стоп» на сборе данных —
вопрос о смене темы или «Отменить текущий процесс записи?», всё с handoff=false;
«кровотечение, я в Тольятти» — «только по Самаре» без совета вызвать скорую;
«оператор! хочу покончить с собой» — голое «Соединяю с оператором по вашему запросу.».

Инвариант: реплика, которую узнаёт регулярка срочности, получает шаблон скорой и
оператора в любой фазе записи, из любого города и вместе с просьбой позвать оператора;
сценарий записи сбрасывается, summary пуст (как после любого перевода на оператора).
Встречные проверки: обычный ответ на подтверждении записи, обычный вопрос из другого
города и явный запрос оператора без срочности ведут себя как раньше.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from messengers_router import endpoint as endpoint_mod
from messengers_router import renderer
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import AppointmentPhase, DialogState, Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.policies import handoff_message
from messengers_router.recovery_policy import explicit_operator_requested
from messengers_router.services import Services

URGENT_PHRASES = [
    "задыхаюсь",
    "у меня кровотечение, что делать?",
    "извините, сильная боль в груди",
    "стоп, у меня инфаркт",
    "не хочу жить, что мне делать?",
    "выпила много таблеток снотворного",
]


@pytest.fixture(autouse=True)
def _llm_would_continue_the_appointment(monkeypatch):
    """Если реплика дойдёт до NLU, LLM продолжит запись: срочность — только от гарда."""

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(
            decision=RouteDecision(label="APPOINTMENT", confidence=0.9, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )


def _say(text: str, state: SessionState, memory: MemoryStore) -> tuple[str, bool]:
    async def go():
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return asyncio.run(go())


def _idle() -> tuple[SessionState, MemoryStore]:
    return SessionState(session_id="urgent-before-guards"), MemoryStore()


# Прежние реплики: summary, пересобранный из них после перевода на оператора, сбивал бы NLU
# следующего вопроса (clear_on_handoff, 2026-05-05).
_HISTORY = [
    {"role": "user", "text": "запишите к терапевту на 7 октября в 10"},
    {"role": "assistant", "text": "Подтвердите запись, пожалуйста: ответьте «да» или «нет»."},
]


def _collecting() -> tuple[SessionState, MemoryStore]:
    state = SessionState(
        session_id="urgent-before-guards",
        history=list(_HISTORY),
        last_entities={"city": "Самара", "appointment_flow_active": True, "specialty": "терапевт"},
        dialog=DialogState(label="APPOINTMENT", phase=AppointmentPhase.COLLECTING),
    )
    memory = MemoryStore()
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])
    return state, memory


def _confirm(session_id: str = "urgent-before-guards") -> tuple[SessionState, MemoryStore]:
    state = SessionState(
        session_id=session_id,
        history=list(_HISTORY),
        summary="last_turns: запись к терапевту",
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "appointment_confirm_pending": True,
            "specialty": "терапевт",
            "branch_name": "пр. Ленина, 5",
            "date_from": "2026-10-07",
            "time_from": "10:00",
            "patient_name": "Иванов Иван",
        },
        dialog=DialogState(label="APPOINTMENT", phase=AppointmentPhase.CONFIRM),
    )
    return state, MemoryStore()


def _cancel_pending() -> tuple[SessionState, MemoryStore]:
    state, memory = _collecting()
    state.last_entities["appointment_cancel_pending"] = True
    return state, memory


def _topic_switch_pending() -> tuple[SessionState, MemoryStore]:
    state, memory = _collecting()
    state.last_entities["appointment_topic_switch_pending"] = True
    return state, memory


@pytest.mark.parametrize("text", URGENT_PHRASES)
@pytest.mark.parametrize(
    "phase",
    [_collecting, _confirm, _cancel_pending, _topic_switch_pending],
    ids=["collecting", "confirm", "cancel_pending", "topic_switch_pending"],
)
def test_urgent_inside_appointment_flow_gets_urgent_template(phase, text):
    state, memory = phase()

    answer, handoff = _say(text, state, memory)

    assert answer == renderer.render_urgent().text
    assert handoff is True
    # Сценарий записи сброшен: следующее «да» не оформит заявку молча.
    assert state.dialog.phase == AppointmentPhase.IDLE
    assert memory.get_pending(state) is None
    for flag in (
        "appointment_flow_active",
        "appointment_confirm_pending",
        "appointment_cancel_pending",
        "appointment_topic_switch_pending",
    ):
        assert flag not in state.last_entities
    assert state.summary == ""


@pytest.mark.parametrize(
    "text",
    ["у меня кровотечение, я в Тольятти", "я в Сызрани, сильная боль в груди", "не хочу жить, я в Тольятти"],
)
def test_urgent_from_other_city_gets_urgent_template(text):
    state, memory = _idle()
    state.history = list(_HISTORY)

    answer, handoff = _say(text, state, memory)

    assert answer == renderer.render_urgent().text
    assert handoff is True
    assert state.summary == ""


@pytest.mark.parametrize(
    "text",
    [
        "оператор! хочу покончить с собой",
        "соедините с оператором, у ребенка судороги",
        "позовите оператора, я наглоталась таблеток",
        "мне нужен живой человек, не хочу жить",
        "оператор! задыхаюсь",
    ],
)
@pytest.mark.parametrize("phase", [_idle, _confirm], ids=["idle", "confirm"])
def test_urgent_with_explicit_operator_request_gets_urgent_template(phase, text):
    # Обе ветки переводят на оператора, но только шаблон срочной помощи говорит про скорую.
    assert explicit_operator_requested(text)
    state, memory = phase()

    answer, handoff = _say(text, state, memory)

    assert answer == renderer.render_urgent().text
    assert handoff is True
    assert "appointment_flow_active" not in state.last_entities


@pytest.mark.parametrize("text", ["соедините с оператором", "позовите живого человека"])
def test_explicit_operator_without_urgency_is_unchanged(text):
    state, memory = _confirm()

    answer, handoff = _say(text, state, memory)

    assert answer == handoff_message("manual_operator")
    assert handoff is True
    assert "appointment_flow_active" not in state.last_entities


def test_other_city_without_urgency_still_goes_to_operator():
    answer, handoff = _say("сколько стоит прием терапевта в Тольятти", *_idle())

    assert answer == router_mod._SAMARA_ONLY_OPERATOR_TEXT
    assert handoff is True


def test_ordinary_reply_inside_confirm_keeps_appointment():
    state, memory = _confirm()

    answer, handoff = _say("а можно попозже, часов в 12?", state, memory)

    assert answer != renderer.render_urgent().text
    assert handoff is False
    assert "appointment_flow_active" in state.last_entities


def test_endpoint_once_debug_urgent_inside_confirm():
    # Сквозь обработчик /api/messenger-generate-once: отладочный ответ, запись хода в
    # историю и сброс записи в сохранённой сессии.
    memory = MemoryStore()
    state, _ = _confirm(session_id="tg_urgent_once")
    memory.set(state)
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services

    resp = TestClient(app).post(
        "/api/messenger-generate-once",
        json={"session_id": "tg_urgent_once", "text": "задыхаюсь", "debug": True},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["handoff"] is True
    # ночью к ответу с переводом дописываются часы операторов — шаблон всё равно первым
    assert body["text"].startswith(renderer.render_urgent().text)
    assert body["state_update"]["debug"]["decision"]["label"] == "URGENT"
    saved = memory.get("tg_urgent_once")
    assert saved.dialog.phase == AppointmentPhase.IDLE
    assert "appointment_confirm_pending" not in saved.last_entities
    assert saved.summary == ""
    assert saved.history[-2:] == [
        {"role": "user", "text": "задыхаюсь"},
        {"role": "assistant", "text": body["text"]},
    ]
