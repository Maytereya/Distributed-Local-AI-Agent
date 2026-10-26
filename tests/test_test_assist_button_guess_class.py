"""Ответ на вопрос кнопки «Цена анализа» догадкой каталога — строка прайса, не воронка цели подбора (п. 1 топ-5 разбора 95 цен).

Класс (общая карточка 26.10 после слияния `claude/test-assist-decisions`, набор `buttons_v1`: g0812, g0829, g0852): после
«Цена анализа» → «Какой анализ интересует? Напишите название» пациент называет анализ, классификатор ставит TEST_ASSIST
(«продолжение»), каталог находит строку нечётко. Под `test_assist_from_data = on` суд «да/нет» по догадке пропущен (строку
выбирает ответ хода), а висящий вопрос кнопки ждёт `service_name` — заземление выбрасывает `test_name` хода, и пустой слот
подбора строил воронку «Для какой цели хотите подобрать анализы?», хотя анализ назван. До слияния — «Похоже, вы имели в виду
…? да/нет».

Инвариант: ход TEST_ASSIST с догадкой каталога и пустым слотом названия анализа получает в слот слова пациента — как без
кнопки (догадка — первый кандидат ответа, что показать, решает LLM ответа по данным); своё название, цель подбора, другая тема,
нет догадки — как было. Живой путь (кнопка → ответ по данным) — карточка `buttons_v1`.
"""

from __future__ import annotations

import pytest

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.policies import missing_slots
from messengers_router.router import _test_assist_guess_as_name

_GUESS = "Исследование кала на паразиты"
_REPLY = "Есть ли анализ кала на паразитов методом обогащения?"
_NAME_SLOT = "_any_of:test_goal,test_name"


def _turn(label: str, entities: dict, guess: str, reply: str = _REPLY) -> tuple[RouteDecision, SessionState]:
    memory, state = MemoryStore(), SessionState(session_id="ta-guess")
    if entities:
        memory.merge_entities(state, dict(entities), label=label)
    decision = RouteDecision(label=label, confidence=0.9)
    _test_assist_guess_as_name(decision, state, memory, guess, reply)
    return decision, state


@pytest.mark.parametrize(
    "entities",
    [
        {},
        # Как на стенде: висящий вопрос кнопки ждал услугу — в состоянии она есть, `test_name` хода выброшен.
        {"service_name": "Паразитологию методом формалин"},
        {"city": "Самара"},
    ],
)
def test_guess_fills_the_empty_name_slot(entities):
    decision, state = _turn("TEST_ASSIST", entities, _GUESS)

    assert state.last_entities.get("test_name") == _REPLY  # слова пациента, как без кнопки; догадка — кандидат ответа
    assert _NAME_SLOT not in missing_slots("TEST_ASSIST", state.last_entities)  # воронки цели не будет
    assert "test_assist_guess_as_name" in decision.flags


@pytest.mark.parametrize("entities", [{"test_name": "анализ кала на паразитов"}, {"test_goal": "проверить щитовидку"}])
def test_own_name_or_goal_is_kept(entities):
    decision, state = _turn("TEST_ASSIST", entities, _GUESS)

    assert {k: state.last_entities.get(k) for k in entities} == entities
    assert state.last_entities.get("test_name") != _REPLY
    assert "test_assist_guess_as_name" not in decision.flags


@pytest.mark.parametrize(("label", "guess"), [("PRICE", _GUESS), ("ADDRESS", _GUESS), ("TEST_ASSIST", "")])
def test_other_topic_or_no_guess_as_before(label, guess):
    decision, state = _turn(label, {}, guess)

    assert "test_name" not in state.last_entities
    assert "test_assist_guess_as_name" not in decision.flags
