"""Реквизиты результатов анализов до правки — как решают сейчас (BUG-2026-10-24-RESULT-SLOT-COMMAND).

Характеристика при `result_slots_llm` = off: быстрое заполнение «фамилия, год, код, номер» (`policies._fill_test_result_entities`)
и переспрос (`policies.clarification_question`). Тесты не говорят «так правильно» — они держат поведение, от которого меряется
замена; известные дефекты отмечены (класс «слот берёт команду как данные», FEAT-2026-07-15-SERVICE-SLOT-LLM — отложенные сиблинги).
Код анализа в состоянии — поле `filial` (на сайте `nom`). ПДн — только выдуманные (репо публичный).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import policies
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState
from messengers_router.services import Services

_ALL = ["surname", "year", "filial", "number"]
_WHOLE_QUESTION = "фамилия, год рождения, код анализа, номер анализа"


def _fields(text: str, missing: list[str]) -> dict:
    out: dict = {}
    policies._fill_test_result_entities(text, missing, out)
    return out


@pytest.mark.parametrize("text", ["Сидорова", "сидорова", "Результаты", "да", "Бг"])
def test_off_any_short_reply_is_also_the_code_known_defect(text: str) -> None:
    """Известный дефект: 1–3 слова без цифр при недостающем коде — код анализа, что бы это ни было (фамилия, команда, «да»)."""

    assert _fields(text, _ALL)["filial"] == text


@pytest.mark.parametrize(("text", "city"), [("Хочу результаты, я из Самары", "Самара"), ("в Тольятти", "Тольятти")])
def test_off_city_becomes_the_code_known_defect(text: str, city: str) -> None:
    """Известный дефект: город реплики — код анализа (у кода и города нет ничего общего, ссылка будет пустой)."""

    assert _fields(text, _ALL)["filial"] == city


def test_off_code_keeps_the_marker_word_known_defect() -> None:
    """Известный дефект: «код Бг» — код анализа «код Бг» (маркер в значении)."""

    assert _fields("код Бг", _ALL)["filial"] == "код Бг"
    assert _fields("Код анализа: бг", ["filial"])["filial"] == "Код анализа: бг"


def test_off_branch_marker_takes_rest_of_reply_known_defect() -> None:
    assert _fields("филиал Ленина, хочу узнать результат", _ALL)["filial"] == "Ленина, хочу узнать результат"


def test_off_bare_year_is_also_the_number_known_defect() -> None:
    """Известный дефект: одно число «1985» — и год рождения, и номер анализа."""

    assert _fields("1985", _ALL) == {"year": 1985, "number": 1985}


def test_off_booking_words_are_not_the_code() -> None:
    """Слова записи (`_QF_APPOINTMENT_WORD_RE`) держат реплику вне кода."""

    assert "filial" not in _fields("Перенести запись", _ALL)
    assert "filial" not in _fields("Отмена", ["filial"])


def test_off_rest_of_the_form_loses_the_code() -> None:
    """Остаток формы «год, код, номер» (фамилия уже есть): год и номер — да, кода нет; «ву 276» (прод #659) — только номер."""

    assert _fields("1985, Бг, 123", ["year", "filial", "number"]) == {"year": 1985, "number": 123}
    assert _fields("ву 276", ["filial", "number"]) == {"number": 276}


def test_off_full_form_fills_all_four() -> None:
    out = _fields("Сидорова, 1985, Бг, 123", _ALL)
    assert out == {"surname": "Сидорова", "year": 1985, "filial": "Бг", "number": 123}


def test_off_partial_clarify_is_the_whole_question() -> None:
    """Принят хотя бы один реквизит — переспрос всё равно просит все четыре (тот же текст, что и до продвижения)."""

    asked = policies.clarification_question("TEST_RESULT", ["year", "filial", "number"])
    assert _WHOLE_QUESTION in asked
    assert asked == policies.clarification_question("TEST_RESULT", list(_ALL))
    assert policies.clarification_question("TEST_RESULT", ["filial"]) == "Укажите код анализа (например: Бг)."


def _say(texts: list[str]) -> tuple[list[str], dict]:
    state = SessionState(session_id="result-slots-characterization")
    memory = MemoryStore()
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    async def go() -> list[str]:
        answers = []
        for text in texts:
            out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
            answers.append("".join(env.text for env in out if env.text))
        return answers

    return asyncio.run(go()), dict(state.last_entities)


def test_off_progress_still_repeats_question_and_offers_operator_known_defect() -> None:
    """Известный дефект (пункт 5 принципов): пациент назвал год — продвижение есть, а бот задаёт тот же вопрос, и анти-залип
    сразу предлагает оператора. LLM в тестах нет (сеть закрыта)."""

    (ask, again), entities = _say(["Хочу получить результаты анализов", "Сидорова 1985"])
    assert entities.get("year") == 1985
    assert _WHOLE_QUESTION in ask and _WHOLE_QUESTION in again
    assert "Перевести на оператора?" in again
