"""Класс «просим уточнить — уточнение не принимаем» — BUG-2026-10-26-CLARIFY-REPLY-NOT-ACCEPTED.

Живой замер врачей 26.10 (`doctors_by_parse = on`): «Кто делает массаж?» → «Уточните, пожалуйста, полное название услуги или
специальность врача… Или перевести на оператора?» → «массаж спины» → «По этому вопросу у меня нет данных… Перевести на
оператора?» (тема OTHER). На пути прода деплоя 18 тот же класс — у просьб инструмента цены назвать услугу: «сколько стоит узи» →
«Введите конкретное название процедуры…» → «узи щитовидной железы» → врачи УЗИ (правило врачей по процедуре); разбор реплики не
решён → «Уточните, пожалуйста, название услуги или анализа» → то же; «Не нашёл… Уточните название услуги» → то же.

Корень: просьбу назвать услугу печатает инструмент темы (`clarify_text` цены; под `doctors_by_parse` — `service_clarify` врачей), а
висящего вопроса она не ставит — ни `_pending` темы, ни поля ожидания `_expects`. Следующую реплику маршрутизирует NLU так, будто
вопроса не было: правила и LLM темы видят только её слова («узи щитовидной железы» — правило врачей по процедуре). Вопрос того же
смысла от планировщика («Скажите название услуги», «Какого врача вы ищете?») ставит `_pending` темы — ответ на него в теме
остаётся (контраст ниже зелёный и до фикса).

Инвариант класса: после просьбы бота назвать услугу в теме T реплика, которая называет услугу, отвечается в теме T — план хода
T, цена названной услуги; не другая тема, не «не понял», не оператор, не та же просьба снова. Мир — прайс мини-прод и выдержка
врачей репозитория; оракул настоящего типа: разметка реплики (`turn_spans`) отмечает фразы случая (в реплике без них —
недоступна: разбор не решён), выбор строки — строки с заданным началом; прочие места LLM недоступны. Подготовка в класс не
входит: её ответ держит тему правилом продолжения подготовки (`flow_prepare_followup_override`, проверено зондом 26.10). Наборы
выключателей — прод деплоя 18; `clarify_reply_topic` — off (как на проде: красное, `xfail strict`) и on (фикс).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_stage2_turns import PROD_D17
from tests.test_price_turns_golden import FIXTURE, run_dialog

# Прод деплоя 18 поверх деплоя 17 (STEP-2026-10-25-DEPLOY18-SET): выключатели, которые читают путь цены, подготовки и NLU.
PROD_D18: dict[str, Any] = {
    **PROD_D17,
    "MR_PRICE_BY_PARSE": "on",
    "MR_BOOKING_TARGET_SLOT_LLM": "on",
    "MR_BUTTON_PRICE_SCOPE": "doctor",
    "MR_BASKET_CALC": "on",
    "MR_ADDRESS_BRANCH_FLAGS": "on",
    "MR_ADDRESS_FOLLOWUP_LLM": "on",
    "MR_LLM_DECISION_TURN_SECONDS": 25,
}

_SPANS_MARK = "Слова сообщения с номерами:"
_ROWS_MARKS = ("ответить на вопрос о ценах нескольких услуг", "ответить пациенту на вопрос о цене")
_MESSAGE_MARK = "Сообщение пациента: «"
_FREE_MARK = "Бот попросил назвать что-то своими словами"
_REPLY_ANSWER_MARK = "Ответ пациента: «"
_REPLY_MARK = "понять, что пациент ответил на вопрос"
_THYROID = "узи щитовидной железы"
_THYROID_ROWS = ("Ультразвуковое исследование щитовидной", "Узи щитовидной")
_TESTS = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Case:
    turns: tuple[str, str]
    topic: str  # тема просьбы назвать услугу (план второго хода)
    asked: str  # слова просьбы в первом ответе — тест проверяет, что пациент её видел
    phrases: tuple[str, ...] = (_THYROID,)  # что отмечает разметка; реплика без них — разметка недоступна
    rows: tuple[str, ...] = _THYROID_ROWS  # строки прайса, которые выбирает LLM


CASES: dict[str, Case] = {
    # «Введите конкретное название процедуры…» — правило общего «узи» (на проде при любом состоянии LLM)
    "price_generic_uzi": Case(("сколько стоит узи", _THYROID), "PRICE", "конкретное название процедуры"),
    # разбор первой реплики не решён (LLM разметки не ответила) — «Уточните, пожалуйста, название услуги или анализа»
    "price_parse_failed": Case(("сколько стоит анализ", _THYROID), "PRICE", "название услуги или анализа"),
    # разметка нашла фразу, LLM выбора строки не выбрала ни одной — строк нет, рендер цены: «Уточните, пожалуйста, название…»
    "price_none_picked": Case(("сколько стоит узи сердца плода", _THYROID), "PRICE", "название услуги или анализа", ("узи сердца плода", _THYROID)),
}


@pytest.fixture(scope="module")
def clarify_world(tmp_path_factory) -> Path:
    """Мир: прайс мини-прод и выдержка врачей — правило врачей по процедуре ведёт «узи щитовидной железы» к врачам, как на проде
    (без среза врачей тема уходила бы в «каталог недоступен»)."""

    src = tmp_path_factory.mktemp("clarify_src")
    shutil.copytree(FIXTURE, src, dirs_exist_ok=True)
    shutil.copy(_TESTS / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl", src / "doctors_20261008.jsonl")
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("clarify") / "world")


class _Oracle:
    """Разметка — фразы случая (в реплике без них — недоступна: разбор не решён); выбор строки — строки случая, если реплика их
    назвала (иначе «ни одна»); ответ на вопрос со свободным ответом — «ответ»; на вопрос «да/нет» — «ничего из этого»; прочее —
    недоступна."""

    def __init__(self, case: Case, judged: dict[str, str] | None = None) -> None:
        self.case = case
        self.judged = judged or {}  # ответ пациента → исход вопроса со свободным ответом (нет в словаре — «ответ»)

    def _spans(self, prompt: str) -> dict[str, Any]:
        numbered = prompt.split(_SPANS_MARK, 1)[1].strip().splitlines()[0].split()
        words = [_normalise_catalog_text(item.partition(":")[2]) for item in numbered]
        for phrase in self.case.phrases:
            want = _normalise_catalog_text(phrase).split()
            for k in range(len(words) - len(want) + 1):
                if words[k : k + len(want)] == want:
                    return {"phrases": [[k + 1, k + len(want)]]}
        raise ConnectionError("тест: разметка этой реплики недоступна")

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _SPANS_MARK in prompt:
            return GenerateResponse(model="m", response=json.dumps(self._spans(prompt)), done=True)
        if any(mark in seen for mark in _ROWS_MARKS):
            options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
            message = seen.split(_MESSAGE_MARK, 1)[-1].split("»", 1)[0]
            named = _normalise_catalog_text(_THYROID) in _normalise_catalog_text(message)  # строки — только к реплике, где их назвали
            picks = [n for n, text in options if named and text.split(" → ", 1)[-1].startswith(self.case.rows)]
            answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks}
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        if _FREE_MARK in seen:
            reply = seen.split(_REPLY_ANSWER_MARK, 1)[-1].split("»", 1)[0]
            outcome = self.judged.get(reply, "answer")
            return GenerateResponse(model="m", response=json.dumps({"picks": [], "outcome": outcome}), done=True)
        if _REPLY_MARK in seen:
            return GenerateResponse(model="m", response=json.dumps({"picks": [], "outcome": "none"}), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _dialog(
    name: str, switch: str, world_dir, monkeypatch, turns: tuple[str, ...] | None = None, judged: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    from messengers_router.runtime_config import config as cfg

    for key, value in {**PROD_D18, "MR_CLARIFY_REPLY_TOPIC": switch}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Oracle(CASES[name], judged))
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    with ev_world.FrozenWorld(world_dir, "day") as world:
        records = run_dialog(list(turns or CASES[name].turns), f"clarify-{name}-{switch}-{len(turns or ())}")
        assert list(world.net.attempts) == []
    return records


def _violations_after(second: dict[str, Any], case: Case, asked: str) -> list[str]:
    """Нарушения инварианта во втором ходе: тема, оператор, та же просьба, цена названной услуги."""

    found = []
    if not second["plan"] or second["plan"][0] != case.topic:
        found.append(f"тема ответа {second['plan']} вместо {case.topic}")
    if second["handoff"] or "Перевести на оператора" in second["text"]:
        found.append("оператор вместо ответа")
    if asked in second["text"]:
        found.append("повторная просьба назвать услугу")
    if not any(row.casefold() in second["text"].casefold() for row in case.rows):
        found.append("нет цены названной услуги")
    return found


def _violations(name: str, records: list[dict[str, Any]]) -> list[str]:
    case = CASES[name]
    first, second = records
    assert case.asked in first["text"], first["text"]  # пациент видел просьбу назвать услугу
    return _violations_after(second, case, case.asked)


@pytest.mark.parametrize("name", sorted(CASES))
def test_reply_to_name_request_stays_in_topic_under_on(name, clarify_world, monkeypatch):
    assert _violations(name, _dialog(name, "on", clarify_world, monkeypatch)) == []


@pytest.mark.xfail(strict=True, reason="BUG-2026-10-26-CLARIFY-REPLY-NOT-ACCEPTED: путь прода (`clarify_reply_topic = off`)")
@pytest.mark.parametrize("name", sorted(CASES))
def test_reply_to_name_request_stays_in_topic_on_prod_path(name, clarify_world, monkeypatch):
    assert _violations(name, _dialog(name, "off", clarify_world, monkeypatch)) == []


def test_planner_question_keeps_topic_without_switch(clarify_world, monkeypatch):
    """Контраст: тот же по смыслу вопрос планировщика («Скажите, пожалуйста, название услуги/анализа») ставит `_pending` темы —
    ответ на него остаётся в теме и под off. Фикс делает просьбу инструмента таким же вопросом."""

    first, second = _dialog("price_generic_uzi", "off", clarify_world, monkeypatch, turns=("сколько стоит?", _THYROID))
    assert first["pending"] == {"label": "PRICE", "missing": ["service_name"]}, first
    assert _violations_after(second, CASES["price_generic_uzi"], "название услуги") == []


@pytest.mark.parametrize("switch", ["off", "on"])
def test_switch_does_not_touch_answers_without_name_request(switch, clarify_world, monkeypatch):
    """Ответ с ценой — не просьба назвать услугу: висящего вопроса нет при любом положении выключателя."""

    first, _second = _dialog("price_generic_uzi", switch, clarify_world, monkeypatch, turns=(f"сколько стоит {_THYROID}", "спасибо"))
    assert first["plan"][0] == "PRICE" and first["pending"] is None, first


# Встречный свип (anti-over-block): реплика, которую LLM не признала ответом на просьбу, идёт тем же путём, что без выключателя.
# Правило «похоже на название услуги» (`flow_policy._looks_like_price_service_reply`, до 6 слов) приняло бы их все: «спасибо» —
# «По услуге «Спасибо»…», «хочу записаться к кардиологу» — снова «Уточните название» (зонд 26.10 на первом варианте фикса).
_NOT_ANSWERS = {
    "спасибо": "none",
    "ок": "none",
    "нет, не надо": "no",
    "хочу записаться к кардиологу": "new_topic",
    "а где вы находитесь?": "new_topic",
    "подумаю": "pause",
}


@pytest.mark.parametrize("reply", sorted(_NOT_ANSWERS))
def test_reply_not_judged_an_answer_goes_as_without_switch(reply, clarify_world, monkeypatch):
    turns = ("сколько стоит узи", reply)
    off = _dialog("price_generic_uzi", "off", clarify_world, monkeypatch, turns=turns, judged=_NOT_ANSWERS)[1]
    on = _dialog("price_generic_uzi", "on", clarify_world, monkeypatch, turns=turns, judged=_NOT_ANSWERS)[1]
    assert (on["plan"], on["text"], on["handoff"]) == (off["plan"], off["text"], off["handoff"])


def test_llm_failure_on_the_answer_goes_as_without_switch(clarify_world, monkeypatch):
    """Ответ LLM на разборе ответа не по контракту (исход не из списка — как сбой) — как без выключателя (запасной путь — прежний)."""

    turns = ("сколько стоит узи", _THYROID)
    off = _dialog("price_generic_uzi", "off", clarify_world, monkeypatch, turns=turns)[1]
    on = _dialog("price_generic_uzi", "on", clarify_world, monkeypatch, turns=turns, judged={_THYROID: "не по контракту"})[1]
    assert (on["plan"], on["text"]) == (off["plan"], off["text"])
