"""Слоты записи по смыслу: LLM решает, правила дают кандидатов (облако 19.10, задача C; `messengers_router/booking_slots.py`).

Каждый раздел — свой пункт задачи и свой выключатель (по умолчанию off — как было). Мир — `mini_prod` (прайс) и
выдержка врачей репозитория (`tests/fixtures/doctors_excerpt`); LLM — подделка настоящего типа
(`llm_runtime._OLLAMA_CLIENT`), отвечает по метке промпта места вызова, прочие места — «недоступна».

C1, `booking_specialty_llm` (карточка test 18.10, g0792; BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, «Открыто»): «На какое
время запись к неврологу» — разбор хода брал «время» за услугу, LLM выбора строки её отвергала (`not_a_service`) или
каталог её не находил, а специальности в решении хода не было: честный отказ гарда B′ «эту услугу я не нашёл…» вместо вопроса о филиале.
Инвариант: специальность, названная в реплике о записи, — цель записи, если запись к ней подтвердила LLM; без
подтверждения (сбой, «нет», off) — как было.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router.entity_grounder import service_dropped_by_llm
from messengers_router import booking_slots, choice, llm_runtime
from messengers_router.choice import Option

_TESTS = Path(__file__).resolve().parent
_PRICE_MARK = "ответить пациенту на вопрос о цене"
_SPECIALTY_MARK = "к какому врачу пациент хочет записаться"
_REFUSAL = "эту услугу я не нашёл в системе онлайн-записи"
_BRANCH_QUESTION = "Какой филиал вам удобен?"


@pytest.fixture(scope="module")
def booking_world(tmp_path_factory) -> Path:
    """Мир: прайс `mini_prod` и врачи выдержки — запись к специальности доходит до вопроса о филиале."""

    src = tmp_path_factory.mktemp("booking_src")
    shutil.copytree(_TESTS / "fixtures" / "mini_prod" / "snapshots", src, dirs_exist_ok=True)
    shutil.copy(_TESTS / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl", src / "doctors_20261008.jsonl")
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("booking") / "world")


class _LLM:
    """Отвечает по метке промпта: `answers[метка] = ответ` (JSON строкой) или вызываемое (данные хода → ответ)."""

    def __init__(self, answers: dict[str, Any]):
        self.answers = answers
        self.calls: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        for mark, answer in self.answers.items():
            if mark in seen:
                self.calls.append(mark)
                text = answer(prompt) if callable(answer) else answer
                return GenerateResponse(model="m", response=text, done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


_NOT_A_SERVICE = json.dumps({"picks": [], "reason": "not_a_service", "search_terms": []})
_PICK_FIRST = json.dumps({"picks": [1]})
_PICK_NONE = json.dumps({"picks": []})


def _dialog(world_dir, monkeypatch, turns: list[str], *, switches: dict[str, str], llm: _LLM | None) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"slots-{json.dumps(switches)}-{turns[0]}"

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            dbg = (body.get("state_update") or {}).get("debug") or {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "label": (dbg.get("decision") or {}).get("label"),
                    "flags": (dbg.get("decision") or {}).get("flags") or [],
                    "entities": dict(memory.get(session).last_entities or {}),
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        for key, value in switches.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        if llm is not None:
            patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        choice._CACHE.clear()
        return asyncio.run(run())


# --- C1: специальность записи из реплики — по смыслу ---------------------------------------------------------------------

# Выключатель `booking_specialty_llm` снят 22.10: база «как было» — тот же ход без ответа задачи специальности (LLM недоступна).
_SPECIALTY_ON: dict[str, str] = {}
_TIME_QUESTIONS = [
    ("На какое время запись к неврологу", "невролог"),
    ("На какое время можно записаться к урологу", "уролог"),
    ("на какое время есть запись к гинекологу?", "гинеколог"),
]


@pytest.mark.parametrize(("question", "specialty"), _TIME_QUESTIONS)
def test_specialty_confirmed_by_llm_is_the_booking_target(question, specialty, booking_world, monkeypatch):
    llm = _LLM({_PRICE_MARK: _NOT_A_SERVICE, _SPECIALTY_MARK: _PICK_FIRST})
    [turn] = _dialog(booking_world, monkeypatch, [question], switches=_SPECIALTY_ON, llm=llm)

    assert _SPECIALTY_MARK in llm.calls, llm.calls
    assert _BRANCH_QUESTION in turn["text"] and _REFUSAL not in turn["text"], turn["text"]
    assert turn["entities"].get("specialty") == specialty, turn["entities"]
    assert "booking_specialty_llm" in turn["flags"]


@pytest.mark.parametrize(("question", "specialty"), _TIME_QUESTIONS)
@pytest.mark.parametrize(
    ("switches", "specialty_answer"),
    [(_SPECIALTY_ON, _PICK_NONE), (_SPECIALTY_ON, None)],
    ids=["llm-no", "llm-down"],
)
def test_without_confirmation_it_is_as_before(question, specialty, switches, specialty_answer, booking_world, monkeypatch):
    answers: dict[str, Any] = {_PRICE_MARK: _NOT_A_SERVICE}
    if specialty_answer is not None:
        answers[_SPECIALTY_MARK] = specialty_answer
    [turn] = _dialog(booking_world, monkeypatch, [question], switches=switches, llm=_LLM(answers))
    [before] = _dialog(booking_world, monkeypatch, [question], switches={}, llm=_LLM({_PRICE_MARK: _NOT_A_SERVICE}))

    assert turn["text"] == before["text"]
    assert _REFUSAL in turn["text"], turn["text"]  # как было: честный отказ с оффером оператора


def test_booking_with_a_named_specialty_does_not_call_the_confirmation(booking_world, monkeypatch):
    # Специальность разбор хода нашёл сам — подтверждать нечего; услуги нет — выбора строки нет.
    llm = _LLM({_PRICE_MARK: _NOT_A_SERVICE, _SPECIALTY_MARK: _PICK_FIRST})
    [turn] = _dialog(booking_world, monkeypatch, ["запишите к неврологу"], switches=_SPECIALTY_ON, llm=llm)

    assert _SPECIALTY_MARK not in llm.calls
    assert _BRANCH_QUESTION in turn["text"], turn["text"]


def test_specialty_prompt_renders_question_and_candidate():
    system, user = choice.render(booking_slots.BOOKING_SPECIALTY_TASK, "На какое время запись к неврологу", [Option(id="невролог", text="невролог")])

    assert "Совпадение слова — ещё не запись" in system
    assert "«На какое время запись к неврологу»" in user and "1. невролог" in user


def test_service_rejected_by_llm_needs_a_verdict():
    assert service_dropped_by_llm({"entity_dropped_unverified_service_name", "booking_service_not_a_service"})
    assert service_dropped_by_llm({"entity_dropped_unverified_service_name", "booking_service_none_fit"})
    assert not service_dropped_by_llm({"entity_dropped_unverified_service_name"})
    assert not service_dropped_by_llm({"entity_dropped_unverified_service_name", "booking_service_unavailable"})


# --- C2: подавитель A′-1 — отвергнутую LLM услугу буквы не воскрешают --------------------------------------------------
#
# BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, попутно (3): «запишите на гастроскопию» → LLM строки не нашла → честный отказ,
# но quick-fill клал «Гастроскопию» в состояние снова, а подавитель её держал: резолвер прайса находит по ней «Гастрин».
# Инвариант (2 BUG-2026-06-02-02 под `slot_suppress_llm = on`): услугу, отвергнутую LLM этого хода, буквы в состоянии
# не воскрешают; снятие без суда LLM — как было (ответ на уточнение цены, P3).

_NONE_FIT = json.dumps({"picks": [], "reason": "other_service", "search_terms": []})


@pytest.mark.parametrize("question", ["запишите на гастроскопию", "хочу записаться на гастроскопию"])
def test_service_rejected_by_llm_is_not_resurrected(question, booking_world, monkeypatch):
    on = _dialog(booking_world, monkeypatch, [question], switches={"MR_SLOT_SUPPRESS_LLM": "on"}, llm=_LLM({_PRICE_MARK: _NONE_FIT}))
    off = _dialog(booking_world, monkeypatch, [question], switches={"MR_SLOT_SUPPRESS_LLM": "off"}, llm=_LLM({_PRICE_MARK: _NONE_FIT}))

    assert service_dropped_by_llm(on[0]["flags"]), on[0]["flags"]
    assert not on[0]["entities"].get("service_name"), on[0]["entities"]
    assert off[0]["entities"].get("service_name") == "Гастроскопию"  # как было
    assert on[0]["text"] == off[0]["text"]  # ответ этого хода тот же — честный отказ


def test_without_llm_verdict_the_catalog_resolved_value_is_kept(monkeypatch):
    # P3 (BUG-2026-06-02-02): ответ на уточнение цены «Общий анализ крови» в ходе TEST_ASSIST — флаг снятия есть,
    # суда LLM нет: каталожное значение остаётся и под on.
    from messengers_router import router
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(router, "resolve_price_service_name_from_catalog", lambda value, **_kw: "Общий анализ крови")
    for switch in ("off", "on"):
        monkeypatch.setattr(cfg, "MR_SLOT_SUPPRESS_LLM", switch, raising=False)
        kept = router._suppress_grounder_rejected_slots({"service_name": "Общий анализ крови"}, {"entity_dropped_unverified_service_name"})
        assert kept == {"service_name": "Общий анализ крови"}, switch
    monkeypatch.setattr(cfg, "MR_SLOT_SUPPRESS_LLM", "on", raising=False)
    judged = {"entity_dropped_unverified_service_name", "booking_service_none_fit"}
    assert router._suppress_grounder_rejected_slots({"service_name": "Гастроскопию", "specialty": "эндоскопист"}, judged) == {"specialty": "эндоскопист"}


# --- C3a: подготовка или запись — главную тему выбирает LLM ----------------------------------------------------------------
#
# BUG-2026-10-18-PREPARE-UZI-AS-BOOKING: «как подготовиться к узи, нужно ли не есть» — «узи» + «нужно» правило записи на
# диагностику считает записью, вопрос о подготовке получал отказ записи «эту услугу я не нашёл…». Инвариант: вопрос «как
# подготовиться к <услуге>» не уходит в запись, если LLM выбрала подготовку; «запись», «ни одна», сбой — как было (база
# «как было» — тот же ход с недоступной LLM; выключатель `prepare_vs_booking_llm` снят 22.10).

_PREPARE_OR_BOOKING_MARK = "есть слова и о записи, и о подготовке"
_PREPARE_ON: dict[str, str] = {}
_PREPARE_QUESTIONS = ["как подготовиться к узи, нужно ли не есть", "нужно ли не есть перед узи", "нужно ли голодать перед узи брюшной полости"]


@pytest.mark.parametrize("question", _PREPARE_QUESTIONS)
def test_preparation_question_is_not_a_booking(question, booking_world, monkeypatch):
    llm = _LLM({_PREPARE_OR_BOOKING_MARK: _PICK_FIRST})
    [turn] = _dialog(booking_world, monkeypatch, [question], switches=_PREPARE_ON, llm=llm)
    [before] = _dialog(booking_world, monkeypatch, [question], switches={}, llm=_LLM({}))  # LLM недоступна — как было

    assert before["label"] == "APPOINTMENT", before  # как было: запись (отказ или «на какую услугу вас записать»)
    assert llm.calls == [_PREPARE_OR_BOOKING_MARK]
    assert turn["label"] == "PREPARE" and "prepare_over_booking_llm" in turn["flags"], turn
    assert "онлайн-записи" not in turn["text"] and "вас записать" not in turn["text"], turn["text"]


@pytest.mark.parametrize("question", _PREPARE_QUESTIONS[:2])
@pytest.mark.parametrize("answer", [json.dumps({"picks": [2]}), _PICK_NONE, None], ids=["booking", "none", "down"])
def test_booking_answer_none_or_failure_is_as_before(question, answer, booking_world, monkeypatch):
    llm = _LLM({_PREPARE_OR_BOOKING_MARK: answer} if answer is not None else {})
    [turn] = _dialog(booking_world, monkeypatch, [question], switches=_PREPARE_ON, llm=llm)
    [before] = _dialog(booking_world, monkeypatch, [question], switches={}, llm=_LLM({}))  # LLM недоступна — как было

    assert (turn["label"], turn["text"]) == (before["label"], before["text"])


@pytest.mark.parametrize("question", ["как подготовиться к узи", "запишите на узи брюшной полости"])
def test_no_conflict_no_llm_call(question, booking_world, monkeypatch):
    llm = _LLM({_PREPARE_OR_BOOKING_MARK: _PICK_FIRST})
    _dialog(booking_world, monkeypatch, [question], switches=_PREPARE_ON, llm=llm)

    assert _PREPARE_OR_BOOKING_MARK not in llm.calls


def test_prepare_or_booking_prompt_renders_both_options():
    system, user = choice.render(booking_slots.PREPARE_OR_BOOKING_TASK, "нужно ли не есть перед узи", booking_slots._PREPARE_OR_BOOKING_OPTIONS)

    assert "можно ли есть или пить" in system
    assert "1. Узнать, как подготовиться" in user and "2. Записаться" in user


# --- C3b/C3c: ответ на вопрос-слот кнопки — с 19.10 решает одна функция ответа на висящий вопрос ----------------------------
#
# Отдельный путь `slot_reply_llm` снят (решение владельца 19.10, п. 5): тесты класса — `tests/test_pending_buttons_class.py`
# (вопрос `slot_value`, выключатель `pending_buttons`).
