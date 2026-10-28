"""Ответ на вопрос кнопки «Записаться на приём» — ответ, а не смена темы (BUG-2026-10-25-BOOK-BUTTON-REPLY-AS-TOPIC-SWITCH).

Замер «внутри кнопок» 25.10 (`buttons_v1`, STEP-2026-10-25-BUTTONS-METRIC): после нажатия бот спрашивает «К какому врачу или на
какую услугу записать?», пациент отвечает услугой, специальностью или вопросом о записи («Как записаться на УЗИ?») — и 29 из 107
получили «Сейчас идет оформление записи. Отменить этот процесс…?». Корень: вопрос цели записи не ставил поле ожидания (в темах
вопросов-слотов `pending_answer` записи не было), функция ответа на висящий вопрос не звалась, и реплику судили правила смены
темы предпроверки записи.

Выключатель `[MESSENGER_ROUTER] booking_target_slot_llm = off | on` (по умолчанию off — как было).
Инвариант: под on ответ, который одна функция (LLM) признала ответом на вопрос цели записи, идёт в запись и вопроса об отмене не
получает; ответ с условием, которого бот не выполняет (ДМС, ребёнок, онлайн), и другая тема, пока в записи ничего не выбрано, — тот же
ответ, что на ту же реплику текстом (оффер оператора, без повтора «к какому врачу?» и без вопроса об отмене); сбой LLM и off — как
было. Тесты — через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import pytest

from messengers_router import expects
from messengers_router.policies import appointment_text_topic_switch_confirm
from tests.test_booking_slots_by_meaning import _LLM, booking_world  # noqa: F401 — booking_world — фикстура
from tests.test_pending_buttons_class import _FREE_MARK, _free, _in_world

_BOOK_BUTTON = ("menu.appointment.book", "Записаться на приём")
_BASE = {"MR_CHOOSE_CACHE_TTL_S": 0}
_ON = {**_BASE, "MR_BOOKING_TARGET_SLOT_LLM": "on"}
_OFF = {**_BASE, "MR_BOOKING_TARGET_SLOT_LLM": "off"}

# Ответы на вопрос цели записи, которые правила смены темы считали другой темой (вопрос, цена, адрес, подготовка, врачи).
# Вторая часть кортежа — ответ остаётся в теме записи (на «где сделать …» и услугу без записи бот вправе ответить адресами).
_REPLIES = (
    ("Как записаться на УЗИ?", True),
    ("Можно записаться на узи брюшной полости и почек?", True),
    ("Хотела бы записаться на ЭКГ с расшифровкой", False),  # ЭКГ — без записи, живая очередь (штатный ответ)
    ("Кардиолог когда принимает?", False),  # вопрос расписания — врачи и расписание, как текстом
    ("Где можно сделать УЗИ щитовидной железы?", False),
)
# Реплика без врача, специальности и услуги: значение не разбирают правила — решает LLM.
_NO_VALUE = "А маленьких детей принимаете?"


def _abort_asked(turn: dict) -> bool:
    return appointment_text_topic_switch_confirm() in turn["text"] or "appointment_flow_topic_switch_requested" in turn["flags"]


def test_book_button_question_is_a_free_answer_expectation(booking_world, monkeypatch):  # noqa: F811
    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON], switches=_ON, llm=_LLM({}))[-1]
    off = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON], switches=_OFF, llm=_LLM({}))[-1]

    assert on["text"] == off["text"]  # подсказка кнопки та же
    assert on["state"][expects.EXPECTS_KEY]["type"] == expects.Question.SLOT_VALUE.value, on["state"].get(expects.EXPECTS_KEY)
    assert expects.EXPECTS_KEY not in off["state"]  # как было: вопрос цели записи поля ожидания не ставил


@pytest.mark.parametrize("reply,booking", _REPLIES)
def test_answer_to_the_book_button_question_stays_in_booking(reply, booking, booking_world, monkeypatch):  # noqa: F811
    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, reply], switches=_ON, llm=_LLM({_FREE_MARK: _free("answer")}))[-1]

    assert not _abort_asked(on), on["text"]
    assert not on["handoff"] and (on["label"] == "APPOINTMENT" or not booking), on


def test_value_the_rules_understand_needs_no_llm(booking_world, monkeypatch):  # noqa: F811
    """Услугу или специальность в реплике узнали правила — «это ответ» без вызова LLM (сужение `pending_buttons`): и при
    недоступной LLM ответ на вопрос кнопки идёт в запись."""

    llm = _LLM({})
    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _REPLIES[0][0]], switches=_ON, llm=llm)[-1]
    assert _FREE_MARK not in llm.calls and not _abort_asked(on) and on["label"] == "APPOINTMENT", on


def test_llm_decides_a_reply_without_a_value(booking_world, monkeypatch):  # noqa: F811
    llm = _LLM({_FREE_MARK: _free("answer")})
    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _NO_VALUE], switches=_ON, llm=llm)[-1]
    assert _FREE_MARK in llm.calls and not _abort_asked(on), on["text"]


def test_off_keeps_the_old_abort_question(booking_world, monkeypatch):  # noqa: F811
    """Прод до фикса: тот же ход получает вопрос об отмене — класс воспроизводится, выключатель его закрывает."""

    off = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _REPLIES[0][0]], switches=_OFF, llm=_LLM({_FREE_MARK: _free("answer")}))[-1]
    assert _abort_asked(off), off["text"]


def test_other_topic_before_anything_is_chosen_is_answered_without_abort_question(booking_world, monkeypatch):  # noqa: F811
    """Другая тема на вопрос цели записи (решение LLM), а в записи ещё ничего не выбрано: отменять нечего — бот не спрашивает
    «Отменить оформление?», реплика идёт обычным путём (`buttons_v1`: 7 из 30 после первой части фикса)."""

    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _NO_VALUE], switches=_ON, llm=_LLM({_FREE_MARK: _free("new_topic")}))[-1]
    assert not _abort_asked(on) and "appointment_topic_switch_pending" not in on["state"], on["text"]


@pytest.mark.parametrize("answer", ["не json", None], ids=["invalid", "down"])
def test_failure_is_as_before(answer, booking_world, monkeypatch):  # noqa: F811
    answers = {_FREE_MARK: answer} if answer is not None else {}
    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _NO_VALUE], switches=_ON, llm=_LLM(answers))[-1]
    off = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _NO_VALUE], switches=_OFF, llm=_LLM(answers))[-1]

    assert (on["text"], on["label"], on["handoff"]) == (off["text"], off["label"], off["handoff"])


# --------------------------------------------------------------------------- класс E: условие, которое бот не выполняет

class _QualifierLLM:
    """Модель прода по месту вызова: выбор роли по каталогу (`role_pick`) и задача специальности
    (`booking_specialty_llm = on`) называют уточнение «по ДМС» / «детский» исходом `qualifier` (на проде на пути кнопки его
    ставил выбор роли — `bkt_on4`, флаг `role_qualifier`); вопрос цели записи — исход `free`; прочие места — недоступны."""

    def __init__(self, free: str):
        from tests.test_booking_specialty_qualifier_class import _SchemaLLM
        from tests.test_role_pick_shadow_class import _MARK, _picks

        self.free = free
        self.inner = _SchemaLLM("qualifier")
        self.role_mark, self.role_answer = _MARK, _picks(None, "qualifier")
        self.calls: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        from ollama import GenerateResponse

        seen = f"{system}\n\n{prompt}" if system else prompt
        if _FREE_MARK in seen:
            self.calls.append(_FREE_MARK)
            return GenerateResponse(model="m", response=_free(self.free), done=True)
        if self.role_mark in seen:
            self.calls.append(self.role_mark)
            return GenerateResponse(model="m", response=self.role_answer(prompt), done=True)
        return await self.inner.generate(prompt=prompt, system=system, **kwargs)


_PROD = {**_ON, "MR_BOOKING_SPECIALTY_LLM": "on"}
_SPECIALTY_REPRASK = "к какому врачу/специалисту или на какую услугу"
# Реплика с условием, которое бот не выполняет · решение функции о вопросе цели записи.
_CONDITIONED = (
    ("Гастроэнтеролог ведет прием по ДМС?", "answer"),
    ("можно записаться к кардиологу по ДМС?", "answer"),
    ("запишите к детскому неврологу", "answer"),
    ("А маленьких детей принимаете?", "new_topic"),
)


@pytest.mark.parametrize("reply,free", _CONDITIONED, ids=[r for r, _ in _CONDITIONED])
def test_condition_the_bot_cannot_meet_gets_the_same_answer_as_typed_text(reply, free, booking_world, monkeypatch):  # noqa: F811
    """BUG-2026-10-25-BOOK-BUTTON-REPLY-AS-TOPIC-SWITCH, класс E (`bkt_on4`: g0302, g0337, g0456, g0494, g0548): внутри кнопки
    ответ с ДМС / ребёнком / онлайн-приёмом — тот же, что на ту же реплику текстом (оффер оператора), а не повторный вопрос
    «к какому врачу?»: готовый ответ обработчика не перебивается висящим вопросом записи."""

    button = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, reply], switches=_PROD, llm=_QualifierLLM(free))[-1]
    typed = _in_world(booking_world, monkeypatch, [reply], switches=_PROD, llm=_QualifierLLM(free))[-1]

    assert _SPECIALTY_REPRASK not in button["text"] and not _abort_asked(button), button["text"]
    assert button["text"] == typed["text"], (button["text"], typed["text"])
