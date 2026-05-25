from messengers_router.mess_types import ResponseEnvelope, SessionState
from messengers_router.router import (
    _REPEAT_GUARD_HINT,
    _maybe_append_operator_hint_on_repeat,
)


def _state():
    return SessionState(session_id="t")


def _run(state, text, handoff=False):
    env = ResponseEnvelope(text=text, handoff=handoff)
    return _maybe_append_operator_hint_on_repeat(env, state)


def test_hint_appears_on_third_identical_answer():
    st = _state()
    msg = "На какую дату и время вам удобно записаться на приём?"
    assert _REPEAT_GUARD_HINT not in _run(st, msg).text  # original
    assert _REPEAT_GUARD_HINT not in _run(st, msg).text  # 1-й повтор
    r3 = _run(st, msg)  # 2-й повтор → подсказка
    assert _REPEAT_GUARD_HINT in r3.text
    assert msg in r3.text  # исходный текст сохранён


def test_counter_resets_on_different_answer():
    st = _state()
    a = "На какую дату и время вам удобно записаться на приём?"
    b = "Уточните, пожалуйста, фамилию врача или город приёма."
    _run(st, a)
    _run(st, a)  # count=1
    _run(st, b)  # сброс
    assert _REPEAT_GUARD_HINT not in _run(st, b).text  # count=1, без подсказки


def test_short_answers_not_guarded():
    st = _state()
    r = None
    for _ in range(4):
        r = _run(st, "Хорошо.")
    assert _REPEAT_GUARD_HINT not in r.text


def test_handoff_answer_not_guarded():
    st = _state()
    r = None
    for _ in range(4):
        r = _run(st, "Передаю заявку в обработку, ожидайте подтверждения записи.", handoff=True)
    assert _REPEAT_GUARD_HINT not in r.text


def test_answer_mentioning_operator_not_guarded():
    st = _state()
    r = None
    for _ in range(4):
        r = _run(st, "Для уточнения могу перевести на оператора. Перевести на оператора?")
    assert _REPEAT_GUARD_HINT not in r.text
