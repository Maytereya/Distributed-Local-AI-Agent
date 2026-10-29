"""Свой вопрос пациента в теме кнопки — исход `in_frame`, а не ответ на вопрос-слот (Н-1 карты «Понимания»,
STEP-2026-10-30-PENDING-IN-FRAME, выключатель `pending_in_frame`).

Стенд деплоя 21 (`nason_btn`, вариант Б рамки кнопки): судья ответа на вопрос кнопки звал свой вопрос пациента в теме кнопки
(«domain») ответом на слот, и блок дозаполнения клал **всю реплику** в `service_name`: «Как подготовиться к анализу» → «Откуда
берут биоматериал для ВПЧ?» → в состоянии `service_name = «Откуда берут биоматериал для ВПЧ»`. Ответ этого хода верный
(подготовка читает разбор хода), вред — на следующем: «а сколько стоит?» искал цену вопроса пациента (класс L-13).

Инвариант класса: под `pending_in_frame = on` реплика целиком значением слота-услуги не становится — ни у «domain», ни у «ответа»
судьи; значение — позиции из разбора хода, разбор пуст или упал — слот не пишется; тема кнопки держится; тот же переспрос на ходу
`in_frame` — повтор для анти-залипа, как у ответа. Цель подбора (`test_goal`) — свободные слова, как у правил. off — байт в байт как было. Через обработчик
эндпоинта, на два-три хода; LLM — подделки настоящего типа (судья, классификатор, оракул разметки хода `tests.spans_oracle`).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime, pending_answer
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_FRAME_JUDGE_MARK = "пациент спрашивает о другом, но в теме нажатой кнопки"  # судья варианта Б (`choose_pending_free_reply_button_frame`)
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_PREPARE = "menu.tests.prepare"
_SUGGEST = "menu.tests.suggest"
_OWN_QUESTION = "Откуда берут биоматериал для ВПЧ?"
_PRICE_FOLLOWUP = "а сколько стоит?"


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def _judge(outcomes: dict[str, str]):
    """Судья по реплике пациента в данных хода: первая найденная реплика — её исход, иначе «другая тема»."""

    def answer(prompt: str) -> str:
        outcome = next((value for reply, value in outcomes.items() if reply in prompt), "new_topic")
        return json.dumps({"picks": [], "outcome": outcome})

    return answer


def _dialog(
    world_dir,
    monkeypatch,
    turns: list[Any],
    *,
    in_frame: str,
    judge: dict[str, str],
    phrases: tuple[str, ...] | None = ("ВПЧ",),
    label: str = "PRICE",
) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"in-frame-{in_frame}-{json.dumps(turns, ensure_ascii=False)}-{json.dumps(judge, ensure_ascii=False)}-{phrases}"
    classifier = json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})
    inner = _LLM({_FRAME_JUDGE_MARK: _judge(judge), _CLASSIFIER_MARK: classifier, "Сообщение пациента": json.dumps({"picks": []})})

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "label": decision.get("label"),
                    "flags": decision.get("flags") or [],
                    "state": dict(memory.get(session).last_entities or {}),
                    "telemetry": world.take_telemetry() or {},
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {
            "MR_PENDING_BUTTONS": "on",
            "MR_CHOOSE_CACHE_TTL_S": 0,
            "MR_BUTTON_ANSWER_TOPIC": "frame",
            "MR_BUTTON_FRAME_CHECK": "judge",
            "MR_BUTTON_REPLY_JUDGE": "on",
            "MR_PENDING_IN_FRAME": in_frame,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        keep_llm_breaker_closed(patch)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(phrases, inner=inner))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _service(turn: dict[str, Any]) -> str:
    return str(turn["state"].get("service_name") or "")


# --------------------------------------------------------------------------- слот-услуга: позиция из разбора, не реплика


@pytest.mark.parametrize("outcome", ["domain", "answer"])
def test_own_question_is_not_the_service_on_the_next_turn(outcome, mini_prod_world, monkeypatch):  # noqa: F811
    """«Как подготовиться к анализу» → свой вопрос → «а сколько стоит?»: в состоянии — позиция разбора («ВПЧ»), цена — ВПЧ, а не
    «По услуге «Откуда берут биоматериал…»». И у «domain», и у «ответа» судьи — реплика целиком слотом-услугой не становится."""

    turns = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), _OWN_QUESTION, _PRICE_FOLLOWUP], in_frame="on", judge={_OWN_QUESTION: outcome})
    assert turns[1]["telemetry"].get("tu_pending_outcome") == ("in_frame" if outcome == "domain" else "answer"), turns[1]["telemetry"]
    assert turns[1]["label"] == "PREPARE" and turns[1]["telemetry"].get("nlu_topic_by_llm") == "button_answer", turns[1]
    assert _service(turns[1]) == "ВПЧ", turns[1]["state"]
    assert "Откуда" not in turns[2]["text"] and "ВПЧ" in turns[2]["text"], turns[2]["text"]
    assert all("Откуда" not in str(value) for value in turns[2]["state"].values()), turns[2]["state"]


def test_off_keeps_the_reply_in_the_slot_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    """off — как было (прод деплоя 21): «domain» — исход «ответ», реплика целиком в `service_name`, следующий ход ищет её цену."""

    turns = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), _OWN_QUESTION, _PRICE_FOLLOWUP], in_frame="off", judge={_OWN_QUESTION: "domain"})
    assert turns[1]["telemetry"].get("tu_pending_outcome") == "answer", turns[1]["telemetry"]
    assert _service(turns[1]) == _OWN_QUESTION.rstrip("?"), turns[1]["state"]
    assert "Откуда берут биоматериал для ВПЧ" in turns[2]["text"], turns[2]["text"]


@pytest.mark.parametrize("phrases", [(), None], ids=["empty", "down"])
def test_no_parse_writes_no_slot(phrases, mini_prod_world, monkeypatch):  # noqa: F811
    """Разбор пуст (услуги в реплике нет) или разметка упала — слот не пишется: следующий ход спрашивает название, а не ищет
    цену вопроса пациента. Тема кнопки держится и без слота."""

    turns = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), _OWN_QUESTION, _PRICE_FOLLOWUP], in_frame="on", judge={_OWN_QUESTION: "domain"}, phrases=phrases)
    assert turns[1]["label"] == "PREPARE" and turns[1]["telemetry"].get("tu_pending_outcome") == "in_frame", turns[1]
    assert not _service(turns[1]), turns[1]["state"]
    assert "Откуда" not in turns[2]["text"] and "Откуда" not in _service(turns[2]), turns[2]


@pytest.mark.parametrize("outcome", ["domain", "answer"])
def test_list_is_the_positions_not_one_of_them(outcome, mini_prod_world, monkeypatch):  # noqa: F811
    """Две услуги в реплике — обе позиции разбора (не одна из них и не реплика): слот закрыт, и инструмент подготовки зовут, а не
    переспрашивают название (стенд 30.10, g0363: «Можно ли сдать ОАК ребенку … и ОАМ?» — памятка, а не «Напишите название»)."""

    reply = "Можно ли сдать ОАК ребенку в год из пальца и ОАМ?"
    turns = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), reply], in_frame="on", judge={reply: outcome}, phrases=("ОАК", "ОАМ"))
    assert turns[1]["telemetry"].get("tu_parse_status") == "picked" and turns[1]["telemetry"].get("tu_parse_n") == 2, turns[1]["telemetry"]
    assert _service(turns[1]) == "ОАК, ОАМ", turns[1]["state"]
    assert turns[1]["telemetry"].get("prepare_subject") == "parse", turns[1]["telemetry"]


def test_goal_slot_keeps_free_words(mini_prod_world, monkeypatch):  # noqa: F811
    """«Подобрать анализы» → свой вопрос → следующий ход: цель подбора — свободные слова пациента, как у правил дозаполнения
    (её никто не читает названием услуги); `service_name` вопросом пациента не становится; следующий ход — как off."""

    reply = "Можно ли сдать мочу на ВПЧ, а не делать мазок?"
    on = _dialog(mini_prod_world, monkeypatch, [_press(_SUGGEST), reply, _PRICE_FOLLOWUP], in_frame="on", judge={reply: "domain"})
    off = _dialog(mini_prod_world, monkeypatch, [_press(_SUGGEST), reply, _PRICE_FOLLOWUP], in_frame="off", judge={reply: "domain"})
    assert on[1]["telemetry"].get("tu_pending_outcome") == "in_frame", on[1]["telemetry"]
    assert on[1]["state"].get("test_goal") == off[1]["state"].get("test_goal") == reply.rstrip("?")
    assert "Можно ли" not in _service(on[2]) and "Можно ли" not in on[2]["text"], on[2]
    assert (on[2]["text"], on[2]["label"]) == (off[2]["text"], off[2]["label"])


# --------------------------------------------------------------------------- тема, лимит решений, анти-залип


def test_in_frame_spends_the_turn_like_an_answer(mini_prod_world, monkeypatch):  # noqa: F811
    """Решение судьи — шаг понимания хода: `in_frame` возвращает ходу решение, как «ответ», и расход решений хода тот же."""

    dialog = [_press(_PREPARE), _OWN_QUESTION]
    on = _dialog(mini_prod_world, monkeypatch, dialog, in_frame="on", judge={_OWN_QUESTION: "domain"})[1]["telemetry"]
    off = _dialog(mini_prod_world, monkeypatch, dialog, in_frame="off", judge={_OWN_QUESTION: "domain"})[1]["telemetry"]
    assert on.get("tu_button_refund") == off.get("tu_button_refund") == 1, (on, off)
    assert on.get("tu_decisions_spent") == off.get("tu_decisions_spent"), (on, off)
    assert "tu_pending_refund" not in on, on


_NO_SERVICE = ("а натощак нужно приходить?", "а натощак нужно приходить утром?")
_REPEAT_OFFER = "Похоже, мне не удаётся помочь с этим в чате"


@pytest.mark.parametrize("outcome", ["domain", "answer"])
def test_same_reask_is_a_repeat_as_before(outcome, mini_prod_world, monkeypatch):  # noqa: F811
    """Свой вопрос в теме кнопки, на который бот снова задал тот же вопрос, — повтор, как у ответа: оффер оператора анти-залипа
    остаётся (стенд 30.10: без него результаты «из архива», «вся история» теряли честный перевод — 678 → 676 / 886; пункт
    консилиума «`in_frame` — продвижение» снят из шага 1а, STEP-2026-10-30-PENDING-IN-FRAME)."""

    turns = _dialog(mini_prod_world, monkeypatch, [_press(_PREPARE), *_NO_SERVICE], in_frame="on", judge={"натощак": outcome}, phrases=())
    assert turns[1]["telemetry"].get("tu_pending_outcome") == ("in_frame" if outcome == "domain" else "answer"), turns[1]["telemetry"]
    assert turns[1]["text"] in turns[2]["text"] and _REPEAT_OFFER in turns[2]["text"], turns[2]["text"]


# --------------------------------------------------------------------------- одно определение «в теме вопроса»


def test_in_frame_is_in_topic_and_closes_like_an_answer():
    found = pending_answer.Answer(pending_answer.Question.SLOT_VALUE.value, pending_answer.Outcome.IN_FRAME, text="x")
    assert found.in_topic and found.closes
    assert not pending_answer.answers_question(found)  # явный запрос оператора в своём вопросе — просьба о переводе
    assert not pending_answer.Answer(found.type, pending_answer.Outcome.NEW_TOPIC, text="x").in_topic


def test_readers_ask_in_topic_not_the_answer_outcome():
    """Читатели решения «реплика в теме вопроса» (тема кнопки, лимит решений, запись, слот) — через `Answer.in_topic`: исход «ответ»
    по имени вне функции ответа не читает никто, иначе `in_frame` выпал бы из одного из них."""

    package = Path(pending_answer.__file__).resolve().parent
    offenders = [
        f"{path.relative_to(package)}:{n}"
        for path in sorted(package.rglob("*.py"))
        if path.name != "pending_answer.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"Outcome\.(ANSWER|IN_FRAME)\b", line)
    ]
    assert not offenders, offenders
