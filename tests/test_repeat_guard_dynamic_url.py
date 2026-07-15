"""Анти-повтор гард ловит СМЫСЛОВОЙ повтор, а не байты с динамическим URL.

Прод-триаж 15.07 (#659, #673): пациент правит данные результата анализа
(Дементьев↔Дементьева, «ву 276»↔«ву 266») — смысловой текст ответа один
(«Результат по вашим данным доступен по ссылке ниже»), но result-ссылка
(getanaliz.php?...nom2=276) зависит от данных. `_repeat_norm` сравнивал ВЕСЬ
текст включая URL → norm разный каждый ход → счётчик повтора не копился →
эскалация на оператора НЕ срабатывала → пациент застревал (5+ идентичных
ответов без предложения оператора).

Инвариант (класс): анти-повтор меряет смысловую часть ответа; меняющийся
динамический хвост (URL результата/заявки) не сбрасывает детектор повтора.
"""

from __future__ import annotations

from messengers_router.mess_types import ResponseEnvelope, SessionState
from messengers_router.memory import MemoryStore
from messengers_router.router import _maybe_offer_operator_on_repeat

_PREVIEW = (
    "Результат по вашим данным доступен по ссылке ниже. "
    "Если анализ ещё не готов, сайт сообщит об этом."
)


def _answer_with_url(number: str) -> str:
    return f"{_PREVIEW}\n\nОткрыть результат: https://naykalab.ru/getanaliz.php?fam=%c4&nom2={number}&fast=1"


def _turn(state: SessionState, memory: MemoryStore, text: str) -> bool:
    out = _maybe_offer_operator_on_repeat(ResponseEnvelope(text=text), state, memory)
    return "перевести на оператора" in out.text.lower()


def test_repeat_escalates_when_only_result_url_changes():
    """Пациент правит данные → URL меняется, смысл тот же. На 3-м смысловом
    повторе гард обязан предложить оператора (в проде — не срабатывал)."""
    memory = MemoryStore()
    state = SessionState(session_id="dyn-url")
    offers = [
        _turn(state, memory, _answer_with_url("276")),
        _turn(state, memory, _answer_with_url("266")),
        _turn(state, memory, _answer_with_url("276")),
    ]
    assert offers == [False, False, True], f"эскалация не сработала при смене URL: {offers}"


def test_identical_answer_still_escalates():
    """Анти-регресс: полностью идентичный ответ (без URL-динамики) по-прежнему
    эскалирует на 3-м повторе."""
    memory = MemoryStore()
    state = SessionState(session_id="identical")
    same = _answer_with_url("276")
    offers = [_turn(state, memory, same) for _ in range(3)]
    assert offers == [False, False, True]


def test_distinct_answers_do_not_escalate():
    """Анти-over-trigger: РАЗНЫЕ по смыслу ответы не считаются повтором."""
    memory = MemoryStore()
    state = SessionState(session_id="distinct")
    o1 = _turn(state, memory, "По услуге «ОАК» нашёл: 350 рублей.")
    o2 = _turn(state, memory, "Приём кардиолога стоит 2700 рублей.")
    o3 = _turn(state, memory, "В городе Самара доступны филиалы: пр.Ленина, 5.")
    assert [o1, o2, o3] == [False, False, False]
