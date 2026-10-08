"""Вариант A записи (решение владельца 08.10, повестка рефакторинга №6; BUG-2026-10-08-BOOKING-VARIANT-A).

Разбор реальной переписки 08.10: в теме «запись» бот провалился в 67% из 193 диалогов, до заявки
дошли 12%, медиана — 6 реплик пациента. Главный класс — «слоты и состояние»: обрывки фраз в
графе ФИО («Завтра С Утра», «Номер Администратора»), вопрос о дате по 3–5 раз, «нет» на карточке
понято как перенос, правка на карточке — «ответьте да или нет». Оператор после полной заявки всё
равно переспрашивал данные.

Инварианты класса (всё — по полному тексту ответа):
- после цели и филиала бот ОДИН раз спрашивает пожелание по времени и переводит на оператора
  со сводкой: цель, филиал, слова пациента в кавычках;
- ФИО, дату, время в графы бот не собирает, карточки «Подтверждаете?» нет;
- назвал время сам — второй раз не спрашиваем;
- отказ или новая тема на шаге пожелания — не заявка (прежний гард отмены);
- перенос: врач + пожелание → оператор, без филиала и ФИО.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from messengers_router import endpoint as endpoint_mod
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import AppointmentPhase, Evidence, SessionState
from messengers_router.policies import appointment_step_policy, looks_like_time_wish
from messengers_router.services import Services

_WISH_PROMPT = "Когда вам удобно прийти?"
_HANDOFF_TAIL = "Передаю оператору — он подберёт время по расписанию и подтвердит запись."
_NEVER_IN_BOOKING = ("Подтверждаете", "ФИО", "На какую дату", "ответьте «да» или «нет»")


@pytest.fixture(autouse=True)
def _offline_plan(monkeypatch):
    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )


def _services() -> Services:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    return services


def _booking_at_branch(session_id: str) -> SessionState:
    state = SessionState(
        session_id=session_id,
        last_entities={
            "specialty": "терапевт",
            "city": "Самара",
            "branch_name": "пр.Ленина, 5",
            "appointment_flow_active": True,
            "__appointment_mode": True,
        },
    )
    state.dialog.phase = AppointmentPhase.COLLECTING
    return state


def _say(text: str, state: SessionState, memory: MemoryStore, services: Services) -> tuple[str, bool]:
    async def go():
        out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return asyncio.run(go())


# --- основной путь: филиал выбран → пожелание → оператор со сводкой ----------------------

@pytest.mark.parametrize("wish", ["в субботу утром", "Завтра с утра", "любой день после 17:00", "не знаю"])
def test_after_branch_one_wish_question_then_operator_with_summary(wish):
    state, memory, services = _booking_at_branch(f"va-{wish}"), MemoryStore(), _services()

    asked, asked_handoff = _say("записаться к терапевту", state, memory, services)
    assert _WISH_PROMPT in asked and asked_handoff is False
    for banned in _NEVER_IN_BOOKING:
        assert banned not in asked, (banned, asked)

    summary, handoff = _say(wish, state, memory, services)
    assert handoff is True
    assert summary.startswith(f"Заявка на запись: приём к терапевту, пр.Ленина, 5. Пожелание по времени: «{wish}».")
    assert summary.endswith(_HANDOFF_TAIL)
    for banned in _NEVER_IN_BOOKING:
        assert banned not in summary, (banned, summary)


def test_phrase_never_becomes_patient_name():
    # Трафик: «Завтра с утра» в графе ФИО карточки как «Завтра С Утра». Теперь — пожелание.
    state, memory, services = _booking_at_branch("va-no-fio"), MemoryStore(), _services()
    _say("записаться к терапевту", state, memory, services)

    summary, _ = _say("Завтра с утра", state, memory, services)

    assert "Завтра С Утра" not in summary
    assert "«Завтра с утра»" in summary


def test_time_named_up_front_is_not_asked_again():
    state, memory, services = _booking_at_branch("va-upfront"), MemoryStore(), _services()

    summary, handoff = _say("записаться к терапевту на субботу утром", state, memory, services)

    assert handoff is True
    assert _WISH_PROMPT not in summary
    assert "Пожелание по времени: «записаться к терапевту на субботу утром»." in summary


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("ладно, не надо", "Отменить текущий процесс записи?"),
        ("отмена", "Отменить текущий процесс записи?"),
        ("а сколько стоит приём?", "Отменить этот процесс и перейти к новому вопросу?"),
    ],
    ids=["refusal", "cancel", "topic_switch"],
)
def test_refusal_or_new_topic_at_wish_step_is_not_a_booking(reply, expected):
    # Класс CONFIRM-REFUSAL-AS-YES: отказ не превращается в заявку оператору.
    state, memory, services = _booking_at_branch(f"va-{reply}"), MemoryStore(), _services()
    _say("записаться к терапевту", state, memory, services)

    answer, handoff = _say(reply, state, memory, services)

    assert handoff is False
    assert expected in answer
    assert "Заявка на запись" not in answer


def test_wish_is_asked_only_once():
    # Пункт 5 принципов: ответа на вопрос о времени нет (pending потерян) — всё равно перевод.
    assert appointment_step_policy({"branch_name": "Ленина 5", "_appointment_wish_asked": True}) == "handoff_with_summary"


# --- перенос: врач + пожелание → оператор ------------------------------------------------

def test_reschedule_with_doctor_and_time_hands_off_with_summary():
    state, memory, services = SessionState(session_id="va-reschedule"), MemoryStore(), _services()
    state.last_entities.update({"appointment_action": "reschedule", "doctor_name": "Дразнин"})

    summary, handoff = _say("перенесите на пятницу", state, memory, services)

    assert handoff is True
    assert summary.startswith("Перенос записи: приём к врачу Дразнин. Пожелание по времени: «перенесите на пятницу».")
    assert "ФИО" not in summary


# --- признак пожелания и сквозной вызов эндпоинта ----------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("в субботу утром", True),
        ("на пятницу", True),
        ("завтра после 17", True),
        ("на следующей неделе", True),
        ("в выходные", True),
        ("15.10 в 9:30", True),
        ("к терапевту", False),
        ("на Ленина 5", False),
        ("Иванова Мария", False),
    ],
)
def test_time_wish_detector(text, expected):
    assert looks_like_time_wish(text) is expected


def test_endpoint_once_booking_wish_hands_off():
    memory, services = MemoryStore(), _services()
    state = _booking_at_branch("tg_va_once")
    memory.set(state)
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    client = TestClient(app)

    first = client.post("/api/messenger-generate-once", json={"session_id": "tg_va_once", "text": "записаться к терапевту"}).json()
    second = client.post("/api/messenger-generate-once", json={"session_id": "tg_va_once", "text": "в субботу утром"}).json()

    assert _WISH_PROMPT in first["text"] and first["handoff"] is False
    assert second["handoff"] is True
    assert second["text"].startswith("Заявка на запись: приём к терапевту, пр.Ленина, 5. Пожелание по времени: «в субботу утром».")


def test_cancel_then_no_resumes_with_wish_question_not_confirm_card():
    # «отмена» → «Отменить текущий процесс записи?» → «нет»: возврат к записи — снова вопрос о
    # времени; старой карточки «Подтверждаете?» нет.
    state, memory, services = _booking_at_branch("va-resume"), MemoryStore(), _services()
    _say("записаться к терапевту", state, memory, services)
    _say("отмена", state, memory, services)

    answer, handoff = _say("нет", state, memory, services)

    assert handoff is False
    assert _WISH_PROMPT in answer, answer
    assert "Подтверждаете" not in answer
    summary, handoff = _say("в пятницу вечером", state, memory, services)
    assert handoff is True
    assert "Пожелание по времени: «в пятницу вечером»." in summary
