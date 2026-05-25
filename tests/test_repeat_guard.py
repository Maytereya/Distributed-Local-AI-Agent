from messengers_router.memory import MemoryStore
from messengers_router.mess_types import ResponseEnvelope, SessionState
from messengers_router.router import (
    _REPEAT_GUARD_OFFER,
    _maybe_offer_operator_on_repeat,
)


def _state():
    return SessionState(session_id="t")


def _run(state, memory, text, handoff=False):
    env = ResponseEnvelope(text=text, handoff=handoff)
    return _maybe_offer_operator_on_repeat(env, state, memory)


def test_offer_appears_on_third_identical_answer():
    st, mem = _state(), MemoryStore()
    msg = "На какую дату и время вам удобно записаться на приём?"
    assert _REPEAT_GUARD_OFFER not in _run(st, mem, msg).text  # original
    assert _REPEAT_GUARD_OFFER not in _run(st, mem, msg).text  # 1-й повтор
    r3 = _run(st, mem, msg)  # 2-й повтор → offer
    assert _REPEAT_GUARD_OFFER in r3.text
    assert msg in r3.text  # исходный текст сохранён
    assert st.last_entities.get("_operator_offer_pending") is True


def test_counter_resets_on_different_answer():
    st, mem = _state(), MemoryStore()
    a = "На какую дату и время вам удобно записаться на приём?"
    b = "Уточните, пожалуйста, фамилию врача или город приёма."
    _run(st, mem, a)
    _run(st, mem, a)  # count=1
    _run(st, mem, b)  # сброс
    r = _run(st, mem, b)  # count=1, без offer
    assert _REPEAT_GUARD_OFFER not in r.text
    assert not st.last_entities.get("_operator_offer_pending")


def test_short_answers_not_guarded():
    st, mem = _state(), MemoryStore()
    r = None
    for _ in range(4):
        r = _run(st, mem, "Хорошо.")
    assert _REPEAT_GUARD_OFFER not in r.text


def test_handoff_answer_not_guarded():
    st, mem = _state(), MemoryStore()
    r = None
    for _ in range(4):
        r = _run(st, mem, "Передаю заявку в обработку, ожидайте подтверждения записи.", handoff=True)
    assert _REPEAT_GUARD_OFFER not in r.text


def test_answer_mentioning_operator_not_guarded():
    st, mem = _state(), MemoryStore()
    r = None
    for _ in range(4):
        r = _run(st, mem, "Для уточнения могу перевести на оператора. Перевести на оператора?")
    assert _REPEAT_GUARD_OFFER not in r.text


def test_already_pending_offer_is_skipped():
    st, mem = _state(), MemoryStore()
    st.last_entities["_operator_offer_pending"] = True
    msg = "Уточните, пожалуйста, фамилию врача или город приёма для записи."
    r = None
    for _ in range(4):
        r = _run(st, mem, msg)
    assert _REPEAT_GUARD_OFFER not in r.text
