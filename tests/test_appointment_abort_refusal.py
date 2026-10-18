"""Отказ или оговорка на вопрос записи не исполняются как «да» (07.10; с 18.10 — на вопросе об отмене оформления).

Ревью DLA 04–07.10: на «Подтвердите запись» ответы «ладно, не надо», «да нет», «хорошо, подумаю», «хорошо, я
перезвоню» получали подтверждённую заявку оператору — согласием считалось всё, что начинается с «да », «ладно »,
«хорошо » (BUG-2026-10-07-CONFIRM-REFUSAL-AS-YES). Шага «Подтвердите запись» нет с 08.10 (вариант A), его обработчик
снят 18.10 (решение владельца Q2, класс 5). Необратимое в записи теперь одно — «Отменить текущий процесс записи?»: «да»
стирает собранное. Инвариант того же класса: стирает только согласие; отказ ведёт обратно в запись, оговорка — в
переспрос, собранное цело. Срочное — первым. Здесь — правила (выключатель `pending_reply` по умолчанию off и запасной
путь одной функции при сбое LLM); решение модели с вето — `test_pending_answer_class.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import AppointmentPhase, DialogState, SessionState
from messengers_router.recovery_policy import contextual_reply_kind
from messengers_router.services import Services

_CANCELLED = "Процесс записи отменён"
AGREE = ["да", "Да!", "ок", "конечно", "давайте", "ага"]
REFUSE = ["нет", "не надо", "ладно, не надо", "да нет", "да нет, наверное не надо"]
HEDGE = ["хорошо, подумаю", "хорошо, я перезвоню", "не уверена", "давайте потом", "позже"]


def _abort_step() -> tuple[SessionState, MemoryStore]:
    state = SessionState(
        session_id="abort-refusal",
        history=[
            {"role": "user", "text": "запишите к терапевту"},
            {"role": "assistant", "text": "Отменить текущий процесс записи? Ответьте «да» или «нет»."},
        ],
        last_entities={
            "city": "Самара",
            "appointment_flow_active": True,
            "appointment_cancel_pending": True,
            "specialty": "терапевт",
            "branch_name": "пр. Ленина, 5",
        },
        dialog=DialogState(label="APPOINTMENT", phase=AppointmentPhase.COLLECTING),
    )
    return state, MemoryStore()


def _answer(text: str) -> tuple[str, bool, SessionState]:
    state, memory = _abort_step()

    async def go():
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    answer, handoff = asyncio.run(go())
    return answer, handoff, state


@pytest.mark.parametrize("text", AGREE)
def test_agreement_cancels_the_booking(text):
    answer, handoff, state = _answer(text)
    assert _CANCELLED in answer and handoff is False
    assert not state.last_entities.get("specialty")


@pytest.mark.parametrize("text", REFUSE + HEDGE)
def test_refusal_or_hedge_keeps_what_was_collected(text):
    answer, handoff, state = _answer(text)
    assert _CANCELLED not in answer and handoff is False
    assert (state.last_entities.get("specialty"), state.last_entities.get("branch_name")) == ("терапевт", "пр. Ленина, 5")


def test_urgent_still_comes_first_on_the_abort_question():
    answer, handoff, _state = _answer("задыхаюсь")
    assert _CANCELLED not in answer
    assert handoff is True  # срочное — оператор (f0c6165), а не отмена записи


@pytest.mark.parametrize("text", ["ладно, не надо", "да нет", "хорошо, я перезвоню", "да отмените лучше"])
def test_refusal_is_not_yes_to_any_bot_question(text):
    # Та же проверка отвечает на «Перевести на оператора?», «Это верно?» и другие вопросы.
    assert contextual_reply_kind(text) != "yes"


@pytest.mark.parametrize("text", ["да, не против", "хорошо, без проблем", "почему бы и нет"])
def test_affirmative_idioms_with_negation_stay_yes(text):
    assert contextual_reply_kind(text) == "yes"
