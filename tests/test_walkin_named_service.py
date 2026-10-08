"""«Без записи» говорим только про анализы и ЭКГ — не про услугу, которую пациент назвал сам (08.10).

Проба прода после деплоя 08.10: кнопка «Нужно ли записываться» → «а на УЗИ брюшной
полости нужна запись?» → «Ультразвуковое исследование … выполняются без записи, в порядке
живой очереди». Кнопка кладёт в контекст `service_name=анализы`, и правило walk-in
(BUG-F: вопрос о записи в лаб-контексте — живая очередь) не смотрело, что фраза называет
другую, записываемую услугу. Пациент пришёл бы на УЗИ без записи. Офлайн так же:
«а на гастроскопию надо записываться?» → «Анализы выполняются без записи», а
«нужно ли записываться?» → «Запись фотоизображения на флеш-карту выполняются без записи».

Инвариант класса: ответ «без записи» про услугу, названную в реплике, допустим, только
если каталог МИС считает её анализом (у строк прайса есть срок готовности `deadline`) или
это ЭКГ. Без своей услуги в реплике ответ — про класс из контекста («Анализы …»), а не
про строку каталога, найденную по словам вопроса.
"""

import asyncio

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState
from messengers_router.services import Services
from tests.test_button_actions import GATEWAY_V1_TITLES

WALK_IN = "без записи"

# Контекст анализов: кнопкой из раздела «Сдать анализы» или обычной фразой.
STARTS = {
    "button": ("menu.tests.booking_needed", GATEWAY_V1_TITLES["menu.tests.booking_needed"]),
    "text": ("", "нужно ли записываться чтобы сдать кровь"),
}

# Записываемые услуги из каталога: не анализы, срока готовности у строк нет.
BOOKABLE = [
    "а на УЗИ брюшной полости нужна запись?",
    "а на гастроскопию надо записываться?",
    "а на колоноскопию нужно записываться?",
    "а на маммографию нужна запись?",
    # Врач: выделитель услуги приём не видит («фраза пустая»), а это не вопрос без объекта.
    "а на приём к неврологу нужна запись?",
    "а к офтальмологу надо записываться?",
]

# Анализы и ЭКГ — живая очередь, как и раньше.
WALK_IN_SERVICES = [
    "а на общий анализ крови нужна запись?",
    "а на ферритин нужна запись?",
    "а на ТТГ надо записываться?",
    "а на ЭКГ нужна запись?",
]


def _dialog(start: str, reply: str) -> tuple[str, str]:
    button_id, first = STARTS[start]
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    state, memory = SessionState(session_id=f"walkin-{start}-{abs(hash(reply))}"), MemoryStore()

    async def run():
        async def say(text, **kwargs):
            out = [e async for e in router_mod.patient_routing_stream(text, state, services, memory, **kwargs)]
            return "".join(e.text for e in out if e.text)

        pressed = await say(first, **({"button_id": button_id} if button_id else {}))
        return pressed, await say(reply)

    return asyncio.run(run())


@pytest.mark.parametrize("start", sorted(STARTS))
@pytest.mark.parametrize("reply", BOOKABLE)
def test_named_bookable_service_is_not_answered_walk_in(start, reply):
    first, answer = _dialog(start, reply)
    assert WALK_IN in first  # контекст анализов задан
    assert WALK_IN not in answer.lower(), answer
    assert "живой очереди" not in answer.lower(), answer
    # Класс из контекста («анализы» от кнопки) не становится услугой записи (свип 08.10).
    assert "на анализы" not in answer.lower(), answer


@pytest.mark.parametrize("start", sorted(STARTS))
@pytest.mark.parametrize("reply", WALK_IN_SERVICES)
def test_named_lab_service_stays_walk_in(start, reply):
    _, answer = _dialog(start, reply)
    assert WALK_IN in answer.lower(), answer


def test_bare_question_answers_about_context_class_not_a_catalog_row():
    _, answer = _dialog("button", "нужно ли записываться?")
    assert answer.startswith("Анализы выполняются без записи"), answer
