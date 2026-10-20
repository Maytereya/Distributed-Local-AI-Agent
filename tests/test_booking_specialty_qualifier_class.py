"""Запись к специальности не теряет уточнение пациента и не принимает процедуру за врача (`booking_specialty_llm = on`).

DEBT-2026-10-20-BOOKING-SPECIALTY-QUALIFIER (совместная карточка 20.10): задача `booking_specialty` получала один вариант —
кандидата правил (`extract_specialty`) — и отвечала «да»: «Записаться к детскому аллергологу» → запись к аллергологу
(взрослому), «Записаться на узи кисти рук» → «запись на приём к узи», «Гастроэнтеролог ведет прием по ДМС?» → список
филиалов. Под off все три — честный отказ с оффером оператора.

Корень: подтверждение одного кандидата — вопрос «да/нет», и сказать «есть уточнение» или «это процедура» модели было
некуда. Фикс: у задачи исходы `qualifier` и `procedure`; исход побеждает выбор.

Инвариант класса (принципы 1 и 2): на реплике с уточнением, которого нет в названии специальности (детский/взрослый,
ДМС/ОМС/полис), и на записи на процедуру (УЗИ, МРТ, ЭКГ) ответ под on — тот же, что под off; записи к кандидату нет.
Выигрыши выключателя («на какое время запись к неврологу» и набор `sw_booking_specialty_llm`) — вопрос о филиале нужной
специальности.

Подделка LLM — настоящего типа (`GenerateResponse` клиента ollama), читает схему ответа из запроса, как модель со
структурным выводом: кандидата она «подтверждает» всегда (так отвечала прод-модель на карточке), а уточнение или
процедуру называет, только если схема даёт для этого место. Мир — `mini_prod` и выдержка врачей (как
`test_booking_slots_by_meaning.py`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import booking_slots, choice
from messengers_router.choice import Option, Status
from tests.test_booking_slots_by_meaning import _BRANCH_QUESTION, _NOT_A_SERVICE, _PRICE_MARK, _SPECIALTY_MARK, _dialog

_TESTS = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def booking_world(tmp_path_factory) -> Path:
    """Мир: прайс `mini_prod` и врачи выдержки — запись к специальности доходит до вопроса о филиале."""

    src = tmp_path_factory.mktemp("qualifier_src")
    shutil.copytree(_TESTS / "fixtures" / "mini_prod" / "snapshots", src, dirs_exist_ok=True)
    shutil.copy(_TESTS / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl", src / "doctors_20261008.jsonl")
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("qualifier") / "world")

_ON = {"MR_BOOKING_SPECIALTY_LLM": "on"}
_OFF = {"MR_BOOKING_SPECIALTY_LLM": "off"}


class _SchemaLLM:
    """Модель со структурным выводом: на задачу специальности всегда выбирает кандидата; исход — `honest`, если схема
    ответа его допускает (иначе исхода в ответе нет: грамматика схемы не даст лишнего поля). Цена — «не услуга»."""

    def __init__(self, honest: str = "none"):
        self.honest = honest
        self.calls: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, format: Any = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _PRICE_MARK in seen:
            self.calls.append(_PRICE_MARK)
            return GenerateResponse(model="m", response=_NOT_A_SERVICE, done=True)
        if _SPECIALTY_MARK in seen:
            self.calls.append(_SPECIALTY_MARK)
            answer: dict[str, Any] = {"picks": [1]}
            outcome = (format or {}).get("properties", {}).get("outcome") if isinstance(format, dict) else None
            if outcome and self.honest in outcome.get("enum", []):
                answer["outcome"] = self.honest
            return GenerateResponse(model="m", response=json.dumps(answer), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


# Реплика · исход, который назвала бы честная модель. Первые три — совместная карточка 20.10.
_QUALIFIED = [
    ("Записаться к детскому аллергологу, Самара", "qualifier"),
    ("Гастроэнтеролог ведет прием по ДМС?", "qualifier"),
    ("запишите к детскому неврологу", "qualifier"),
    ("хочу записаться к детскому эндокринологу", "qualifier"),
    ("можно записаться к кардиологу по ДМС?", "qualifier"),
    ("запись к гинекологу по полису ОМС", "qualifier"),
    ("Гастроэнтеролог принимает по ОМС?", "qualifier"),
    ("взрослый уролог есть? хочу записаться", "qualifier"),
]
_PROCEDURES = [
    ("Записаться на узи кисти рук", "procedure"),
    ("хочу записаться на узи щитовидки", "procedure"),
    ("записаться на экг к кардиологу", "procedure"),
    ("записаться на мрт к неврологу", "procedure"),
]
# Карточка 20.10: эти реплики доходят до задачи специальности (специальность разбор хода не нашёл).
_CARD = ("Записаться к детскому аллергологу, Самара", "Записаться на узи кисти рук", "Гастроэнтеролог ведет прием по ДМС?")


@pytest.mark.parametrize(("question", "honest"), _QUALIFIED + _PROCEDURES, ids=[q for q, _ in _QUALIFIED + _PROCEDURES])
def test_qualified_or_procedure_booking_is_as_without_the_switch(question, honest, booking_world, monkeypatch):
    llm = _SchemaLLM(honest)
    [on] = _dialog(booking_world, monkeypatch, [question], switches=_ON, llm=llm)
    [off] = _dialog(booking_world, monkeypatch, [question], switches=_OFF, llm=_SchemaLLM(honest))

    assert on["text"] == off["text"], (on["text"], off["text"])
    assert "booking_specialty_llm" not in on["flags"], on["flags"]
    if question in _CARD:
        assert _SPECIALTY_MARK in llm.calls, llm.calls  # путь задачи пройден, а не обойдён
        assert f"booking_specialty_llm_{honest}" in on["flags"], on["flags"]


_WINS = [
    ("На какое время запись к неврологу", "невролог"),
    ("На какое время можно записаться к урологу", "уролог"),
    ("на какое время есть запись к гинекологу?", "гинеколог"),
    ("когда можно записаться к кардиологу", "кардиолог"),
    ("на какое число можно к эндокринологу записаться?", "эндокринолог"),
]


@pytest.mark.parametrize(("question", "specialty"), _WINS, ids=[q for q, _ in _WINS])
def test_plain_booking_question_keeps_the_switch_win(question, specialty, booking_world, monkeypatch):
    [on] = _dialog(booking_world, monkeypatch, [question], switches=_ON, llm=_SchemaLLM("none"))

    assert _BRANCH_QUESTION in on["text"], on["text"]
    assert on["entities"].get("specialty") == specialty, on["entities"]


def test_outcome_beats_the_pick():
    options = [Option(id="аллерголог", text="аллерголог")]
    for outcome in ("qualifier", "procedure"):
        verdict = choice.parse_answer(booking_slots.BOOKING_SPECIALTY_TASK, options, json.dumps({"picks": [1], "outcome": outcome}))
        assert verdict.status is Status.OUTCOME and not booking_slots.confirms(verdict, "аллерголог"), outcome
    plain = choice.parse_answer(booking_slots.BOOKING_SPECIALTY_TASK, options, json.dumps({"picks": [1], "outcome": "none"}))
    assert booking_slots.confirms(plain, "аллерголог")


def test_schema_gives_the_model_a_place_for_the_qualifier():
    options = [Option(id="невролог", text="невролог")]
    schema = choice.answer_schema(booking_slots.BOOKING_SPECIALTY_TASK, options)
    assert set(schema["properties"]["outcome"]["enum"]) == {"none", "qualifier", "procedure"}
    system, _user = choice.render(booking_slots.BOOKING_SPECIALTY_TASK, "запишите к детскому неврологу", options)
    assert '"qualifier"' in system and '"procedure"' in system
