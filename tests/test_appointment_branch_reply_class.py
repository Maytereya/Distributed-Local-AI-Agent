"""Класс 5 («бот знает, чего ждёт»): ответ на вопрос бота «Какой филиал вам удобен?» — не смена темы.

BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH (ОТКРЫТО), найден стендом сбоев 17.10: «хочу записаться к урологу» → бот
перечисляет филиалы и спрашивает «Какой филиал вам удобен? Или написать список врачей?» → «в любой филиал» (или
«какие адреса?») → «Сейчас идет оформление записи. Отменить этот процесс и перейти к новому вопросу?».

Корень — предпроверка записи (`appointment_flow_guard.run_appointment_precheck`): реплику, на которой срабатывает
любой детектор другой темы (здесь адрес — «филиал», «адреса»), или вопрос из 4+ слов она считает уходом из записи, не
глядя, о чём спросил бот. Вопрос о филиале поля ожидания `_expects` не пишет, поэтому и ответ силами LLM
(`pending_reply`, до предпроверки) его не видит: баг держится и при живой LLM. Через эндпоинт, мини-прод, день, LLM
недоступна (мир гейта).

Подбаг B того же класса: в записи на УСЛУГУ («запишите на узи щитовидки») даже дословно названный филиал из списка даёт
«Пока не понял, к какому врачу или на какую услугу вас записать» — путь `flow_prelock` (NLU и LLM не участвуют): грундер
переизвлекает «услугу» из реплики-адреса, роняет её как неподтверждённую, а гард B′ планировщика в активной записи
считает целью только врача или специальность.

Красное — `xfail(strict=True)`: позеленевший случай краснеет (XPASS), отметку снимает коммит фикса. Зелёное —
характеризация: названный филиал при записи к специальности ведёт к вопросу о времени (вариант A), явная новая тема —
к вопросу об отмене. Мир — гейт (локальные срезы врачей и филиалов: в мини-проде филиалов нет).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from messengers_router.policies import appointment_text_topic_switch_confirm

_BRANCH_QUESTION = "Какой филиал вам удобен?"
_UNKNOWN_TARGET = "Пока не понял, к какому врачу или на какую услугу вас записать"
_WISH_QUESTION = "Когда вам удобно прийти?"

# Цель записи: специальность и услуга (у услуги своя ветка «цель не подтверждена» в планировщике — B′).
OPENINGS = {
    "specialty": "хочу записаться к урологу",
    "service": "запишите на узи щитовидки",
}

# Ответы на вопрос «какой филиал» — о филиалах, адресах и врачах из того же вопроса.
BRANCH_REPLIES = {
    "any_branch": "в любой филиал",
    "which_addresses": "какие адреса?",
    "no_matter": "без разницы какой филиал",
    "closer": "а какой филиал ближе к центру?",
    "compare": "где удобнее, на Ленина или на Победы?",
}

_REASON = (
    "BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH: предпроверка записи считает ответ о филиале сменой темы — вопрос "
    "«какой филиал» не пишет `_expects`, детекторы других тем его не учитывают"
)


def _dialog(turns: tuple[str, ...], session_id: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            entities = dict(memory.get(session_id).last_entities or {})
            out.append({"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff")), "entities": entities})
        return out

    return asyncio.run(run())


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_opening_asks_for_a_branch(opening):
    first = _dialog((OPENINGS[opening],), f"branch-reply-open-{opening}")[0]
    assert _BRANCH_QUESTION in first["text"], first["text"]
    assert first["entities"].get("appointment_branch_options"), first["entities"]


@pytest.mark.parametrize("reply", sorted(BRANCH_REPLIES))
@pytest.mark.parametrize("opening", sorted(OPENINGS))
@pytest.mark.xfail(strict=True, reason=_REASON)
def test_branch_reply_is_not_a_topic_switch(opening, reply):
    first, second = _dialog((OPENINGS[opening], BRANCH_REPLIES[reply]), f"branch-reply-{opening}-{reply}")
    assert _BRANCH_QUESTION in first["text"], first["text"]
    assert second["text"] != appointment_text_topic_switch_confirm(), second["text"]
    assert not second["entities"].get("appointment_topic_switch_pending"), second["entities"]
    # Запись продолжается с той же целью: не «не понял, куда записать», не перевод.
    assert _UNKNOWN_TARGET not in second["text"], second["text"]
    assert not second["handoff"], second["text"]
    assert second["entities"].get("service_name") == first["entities"].get("service_name"), second["entities"]
    assert second["entities"].get("appointment_flow_active"), second["entities"]


def _named_branch_answers(opening: str) -> dict[str, str]:
    """Каждый показанный филиал, названный дословно, — ответ второго хода (сессия на филиал)."""

    first = _dialog((OPENINGS[opening],), f"branch-named-probe-{opening}")[0]
    options = first["entities"].get("appointment_branch_options") or []
    assert options, first["text"]
    return {
        option: _dialog((OPENINGS[opening], option), f"branch-named-{opening}-{i}")[1]["text"]
        for i, option in enumerate(options)
    }


def test_named_branch_moves_to_the_time_wish():
    answers = _named_branch_answers("specialty")
    assert all(_WISH_QUESTION in text for text in answers.values()), answers


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH, подбаг B: в записи на услугу ответ адресом переизвлекается "
        "грундером как «услуга», роняется как неподтверждённая, и гард B′ планировщика (цель — только врач или "
        "специальность) отвечает «Пока не понял, к какому врачу»"
    ),
)
def test_named_branch_moves_to_the_time_wish_for_a_service():
    answers = _named_branch_answers("service")
    assert all(_WISH_QUESTION in text for text in answers.values()), answers


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_new_topic_still_asks_to_leave_the_booking(opening):
    _, second = _dialog((OPENINGS[opening], "какие у вас сейчас акции?"), f"branch-new-topic-{opening}")
    assert second["text"] == appointment_text_topic_switch_confirm(), second["text"]
