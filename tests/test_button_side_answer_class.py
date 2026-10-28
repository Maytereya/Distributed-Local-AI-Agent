"""Уточняющий вопрос о вопросе кнопки результатов получает ответ, а не повтор инструкции (BUG-2026-10-27-RESULT-BUTTON-SIDE-QUESTION,
класс К3 карты провалов деплоя 18, выключатель `button_side_answer`).

Набор кнопок `buttons_v1` (деплой 18, набор 19): после «Ввести данные» («укажите фамилию, год рождения, код анализа, номер
анализа») пациент спрашивает о самом вопросе — «Где посмотреть код бланка?», «Где найти номер анализа?», «А как узнать сам
результат?» — и получает ту же инструкцию и «Похоже, мне не удаётся помочь… Перевести на оператора?».

Инвариант класса: под `button_side_answer = on` реплика на вопрос кнопки результатов, которую
судья одной функции назвал вопросом о вопросе бота (исход «ничего из этого», без хвоста), получает ответ, где код и номер и
как посмотреть результат, без оффера оператора вопросом; вопрос о реквизитах висит дальше, и следующая реплика с данными —
ответ на него. Встречные: off, другие исходы судьи, вопрос другой кнопки и вопрос не кнопки — как было. Через обработчик
эндпоинта; LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import button_flow, choice, expects, llm_runtime, router
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_button_answer_topic_class import _CLASSIFIER_MARK, _FREE_MARK, _PICK_NONE, _free
from tests.test_button_reply_judge_class import _BUTTON_MARK
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_RESULT_BUTTONS = ("menu.results.lookup", "menu.results.missing")
_SIDE = "Где посмотреть код бланка?"
_DATA = "Иванов, 1989, Бг, 123"
_OTHER = json.dumps({"label": "OTHER", "confidence": 0.6, "context_action": "new_topic", "entities": {}})


def _dialog(world_dir, monkeypatch, turns, *, side: str, judge: str = "none") -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"side-{side}-{judge}-{json.dumps(turns, ensure_ascii=False)}"
    answers: dict[str, Any] = {_BUTTON_MARK: _free(judge), _FREE_MARK: _free(judge), _CLASSIFIER_MARK: _OTHER, "Сообщение пациента": _PICK_NONE}

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
                    "handoff": bool(body.get("handoff")),
                    "state": dict(memory.get(session).last_entities or {}),
                    "telemetry": world.take_telemetry() or {},
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {
            "MR_CHOOSE_CACHE_TTL_S": 0,
            "MR_BUTTON_REPLY_JUDGE": "on",
            "MR_BUTTON_SIDE_ANSWER": side,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


@pytest.mark.parametrize("button_id", _RESULT_BUTTONS)
def test_side_question_gets_answer_and_question_stays(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _SIDE, _DATA], side="on")
    press, side, data = on
    assert side["text"] == router.TEST_RESULT_SIDE_TEXT and not side["handoff"], side["text"]
    assert side["text"] not in press["text"] and "Перевести на оператора?" not in side["text"]
    assert side["telemetry"].get("button_side_answer") == "test_result", side["telemetry"]
    # Вопрос о реквизитах висит дальше: тот же вопрос кнопки, следующая реплика с данными — ответ на него.
    assert (side["state"].get(expects.EXPECTS_KEY) or {}).get("type") == expects.Question.SLOT_VALUE.value, side["state"]
    assert button_flow.question_button(side["state"].get("_pending")) == button_id
    # Реплика с данными после побочного ответа — тот же ход, что сразу после нажатия.
    direct = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _DATA], side="on")[1]
    assert data["telemetry"].get("tu_pending_type") == "slot_value", data["telemetry"]
    assert (data["text"], data["label"]) == (direct["text"], direct["label"])


@pytest.mark.parametrize("button_id", _RESULT_BUTTONS)
def test_off_keeps_the_old_reply(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _SIDE], side="off")
    assert off[1]["text"] != router.TEST_RESULT_SIDE_TEXT
    assert "button_side_answer" not in off[1]["telemetry"]


@pytest.mark.parametrize("judge", ["answer", "new_topic", "no"])
def test_other_judge_outcomes_keep_the_old_path(judge, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press("menu.results.lookup"), _SIDE]
    on = _dialog(mini_prod_world, monkeypatch, turns, side="on", judge=judge)
    off = _dialog(mini_prod_world, monkeypatch, turns, side="off", judge=judge)
    assert "button_side_answer" not in on[1]["telemetry"]
    assert (on[1]["text"], on[1]["label"]) == (off[1]["text"], off[1]["label"])


def test_other_button_question_keeps_the_old_path(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: «ничего из этого» на вопрос кнопки подготовки — ход как был (ответ — только о вопросе результатов)."""

    turns = [_press("menu.tests.prepare"), "а что писать?"]
    on = _dialog(mini_prod_world, monkeypatch, turns, side="on")
    off = _dialog(mini_prod_world, monkeypatch, turns, side="off")
    assert "button_side_answer" not in on[1]["telemetry"]
    assert (on[1]["text"], on[1]["label"]) == (off[1]["text"], off[1]["label"])


def test_typed_results_question_keeps_the_old_path(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: вопрос о реквизитах задан на набранный текст, не кнопкой — ход как был."""

    turns = ["хочу узнать результаты анализов", _SIDE]
    on = _dialog(mini_prod_world, monkeypatch, turns, side="on")
    assert not button_flow.question_button(on[0]["state"].get("_pending"))
    assert on[1]["text"] != router.TEST_RESULT_SIDE_TEXT
