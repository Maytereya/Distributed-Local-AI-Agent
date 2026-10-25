"""Ответ на вопрос кнопки «Записаться на приём» — ответ, а не смена темы (BUG-2026-10-25-BOOK-BUTTON-REPLY-AS-TOPIC-SWITCH).

Замер «внутри кнопок» 25.10 (`buttons_v1`, STEP-2026-10-25-BUTTONS-METRIC): после нажатия бот спрашивает «К какому врачу или на
какую услугу записать?», пациент отвечает услугой, специальностью или вопросом о записи («Как записаться на УЗИ?») — и 29 из 107
получили «Сейчас идет оформление записи. Отменить этот процесс…?». Корень: вопрос цели записи не ставил поле ожидания (в темах
вопросов-слотов `pending_answer` записи не было), функция ответа на висящий вопрос не звалась, и реплику судили правила смены
темы предпроверки записи.

Выключатель `[MESSENGER_ROUTER] booking_target_slot_llm = off | on` (по умолчанию off — как было) поверх `pending_buttons = on`.
Инвариант: под on ответ, который одна функция (LLM) признала ответом на вопрос цели записи, идёт в запись и вопроса об отмене не
получает; другая тема, пока в записи ничего не выбрано, — обычный путь без вопроса об отмене (отменять нечего; запись не снимается); сбой LLM и off — как
было. Тесты — через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import pytest

from messengers_router import expects
from messengers_router.policies import appointment_text_topic_switch_confirm
from tests.test_booking_slots_by_meaning import _LLM, booking_world  # noqa: F401 — booking_world — фикстура
from tests.test_pending_buttons_class import _FREE_MARK, _free, _in_world

_BOOK_BUTTON = ("menu.appointment.book", "Записаться на приём")
_BASE = {"MR_PENDING_BUTTONS": "on", "MR_CHOOSE_CACHE_TTL_S": 0}
_ON = {**_BASE, "MR_BOOKING_TARGET_SLOT_LLM": "on"}
_OFF = {**_BASE, "MR_BOOKING_TARGET_SLOT_LLM": "off"}

# Ответы на вопрос цели записи, которые правила смены темы считали другой темой (вопрос, цена, адрес, подготовка, врачи).
# Вторая часть кортежа — ответ остаётся в теме записи (на «где сделать …» и услугу без записи бот вправе ответить адресами).
_REPLIES = (
    ("Как записаться на УЗИ?", True),
    ("Можно записаться на узи брюшной полости и почек?", True),
    ("Хотела бы записаться на ЭКГ с расшифровкой", False),  # ЭКГ — без записи, живая очередь (штатный ответ)
    ("Кардиолог когда принимает?", True),
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


def test_wrong_new_topic_keeps_the_booking(booking_world, monkeypatch):  # noqa: F811
    """Модель ошиблась («другая тема» на просьбу записаться с филиалом в реплике): запись не снимается — обычный путь
    продолжает запись, а не теряет контекст до «Не совсем понял…» (карточка `bkt_on3`, r0142: потерян названный филиал)."""

    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, "хочу к врачу на ленина 5"], switches=_ON, llm=_LLM({_FREE_MARK: _free("new_topic")}))[-1]
    assert not _abort_asked(on) and on["label"] == "APPOINTMENT" and not on["text"].startswith("Не совсем понял"), on


@pytest.mark.parametrize("answer", ["не json", None], ids=["invalid", "down"])
def test_failure_is_as_before(answer, booking_world, monkeypatch):  # noqa: F811
    answers = {_FREE_MARK: answer} if answer is not None else {}
    on = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _NO_VALUE], switches=_ON, llm=_LLM(answers))[-1]
    off = _in_world(booking_world, monkeypatch, [_BOOK_BUTTON, _NO_VALUE], switches=_OFF, llm=_LLM(answers))[-1]

    assert (on["text"], on["label"], on["handoff"]) == (off["text"], off["label"], off["handoff"])
