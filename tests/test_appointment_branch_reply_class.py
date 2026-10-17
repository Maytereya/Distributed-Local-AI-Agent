"""Класс 5 («бот знает, чего ждёт»): ответ на вопрос бота «Какой филиал вам удобен?» — не смена темы.

BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH (A — ОТКРЫТО, B — исправлен), найден стендом сбоев 17.10: «хочу записаться к урологу» → бот
перечисляет филиалы и спрашивает «Какой филиал вам удобен? Или написать список врачей?» → «в любой филиал» (или
«какие адреса?») → «Сейчас идет оформление записи. Отменить этот процесс и перейти к новому вопросу?».

Корень — предпроверка записи (`appointment_flow_guard.run_appointment_precheck`): реплику, на которой срабатывает
любой детектор другой темы (здесь адрес — «филиал», «адреса»), или вопрос из 4+ слов она считает уходом из записи, не
глядя, о чём спросил бот. Вопрос о филиале поля ожидания `_expects` не пишет, поэтому и ответ силами LLM
(`pending_reply`, до предпроверки) его не видит: баг держится и при живой LLM. Через эндпоинт, мини-прод, день, LLM
недоступна (мир гейта).

Подбаг B того же класса — ИСПРАВЛЕН 18.10: в записи на УСЛУГУ («запишите на узи щитовидки») даже дословно названный
филиал из списка давал «Пока не понял, к какому врачу или на какую услугу вас записать» — путь `flow_prelock` (NLU и LLM
не участвуют): грундер получал услугу записи каноническим названием каталога, `extract_service_phrase` его не узнавал,
значение падало как «неподтверждённое», гард B′ планировщика отказывал. Инвариант: в активной записи цель-услуга
переживает ход-ответ на шаг записи так же, как специальность, — ответы записи на услугу и к специальности на одни и те же
реплики одного рода (`entity_grounder._is_active_booking_target`).

Красное — `xfail(strict=True)`: позеленевший случай краснеет (XPASS), отметку снимает коммит фикса. Зелёное —
названный филиал ведёт к вопросу о времени и заявке оператору (вариант A), явная новая тема — к вопросу об отмене,
названная новая услуга сменяет цель. Мир — гейт (локальные срезы врачей и филиалов: в мини-проде филиалов нет).
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


# Записи на услугу: цель — каноническое название каталога (у специальности — «приём к …»).
SERVICE_OPENINGS = ("запишите на узи щитовидки", "запишите на узи почек", "хочу записаться на узи брюшной полости")
# Ответы на шаг записи, которые новую цель не называют.
SLOT_ANSWERS = ("завтра после 17", "Иванов Иван")


def _named_branch_dialogs(opening: str) -> dict[str, list[dict[str, Any]]]:
    """Каждый показанный филиал, названный дословно, — второй ход; третий — пожелание по времени (сессия на филиал)."""

    first = _dialog((opening,), f"branch-named-probe-{opening}")[0]
    options = first["entities"].get("appointment_branch_options") or []
    assert options, first["text"]
    return {
        option: _dialog((opening, option, "в субботу утром"), f"branch-named-{opening}-{i}")
        for i, option in enumerate(options)
    }


@pytest.mark.parametrize("opening", (OPENINGS["specialty"], *SERVICE_OPENINGS))
def test_named_branch_moves_to_the_time_wish_and_the_operator(opening):
    for option, (first, second, third) in _named_branch_dialogs(opening).items():
        assert _WISH_QUESTION in second["text"], (option, second["text"])
        assert second["entities"].get("service_name") == first["entities"].get("service_name"), (option, second)
        assert third["handoff"], (option, third["text"])
        assert option in third["text"] and "«в субботу утром»" in third["text"], (option, third["text"])


@pytest.mark.parametrize("answer", SLOT_ANSWERS)
@pytest.mark.parametrize("opening", SERVICE_OPENINGS)
def test_slot_answer_keeps_the_service_target(opening, answer):
    # Паритет со специальностью: ответ без новой цели — запись продолжается с той же услугой, не «не понял, куда».
    first, second = _dialog((opening, answer), f"branch-slot-{opening}-{answer}")
    assert _UNKNOWN_TARGET not in second["text"], second["text"]
    assert not second["handoff"], second["text"]
    assert second["entities"].get("service_name") == first["entities"].get("service_name"), second["entities"]
    assert second["entities"].get("appointment_flow_active"), second["entities"]


@pytest.mark.parametrize("opening", (OPENINGS["specialty"], *SERVICE_OPENINGS))
def test_named_new_service_replaces_the_target(opening):
    # Встречный случай: правило держит только цель, которую реплика не меняет; новая услуга сменяет цель.
    first, second = _dialog((opening, "лучше на узи сердца"), f"branch-new-service-{opening}")
    assert second["entities"].get("service_name") != first["entities"].get("service_name"), second["entities"]
    assert "сердца" in second["text"], second["text"]


@pytest.mark.parametrize("opening", sorted(OPENINGS))
def test_new_topic_still_asks_to_leave_the_booking(opening):
    _, second = _dialog((OPENINGS[opening], "какие у вас сейчас акции?"), f"branch-new-topic-{opening}")
    assert second["text"] == appointment_text_topic_switch_confirm(), second["text"]
