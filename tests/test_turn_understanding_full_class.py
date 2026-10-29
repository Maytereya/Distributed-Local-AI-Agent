"""Полное понимание хода — ход (б), шаг 0 (STEP-2026-10-30-B-STEP0-FULL, консилиум 30.10): `turn_understanding_full = shadow`.

На каждом ходе одна точка (`router.build_plan`, план построен) собирает объект понимания: тема и кто её решил, рамка кнопки,
висящий вопрос и ответ на него, сущности — копиями уже принятых решений. Инварианты наблюдателя (консилиум): не пересчитывает
решений и не зовёт LLM; невидим для ответа (текст, метка, состояние — как без тени); сбой гасится со следом `tu_full_error`.
Через обработчик эндпоинта; LLM — подделка настоящего типа (харнесс `test_button_answer_topic_class`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime, turn_understanding
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_answer_topic_class import _CLASSIFIER_MARK, _FOREIGN, _FREE_MARK, _PICK_NONE, _free, _press
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура


def _dialog(world_dir, monkeypatch, turns, *, full: str, free: str = "answer") -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"tu-full-{full}-{free}-{json.dumps(turns, ensure_ascii=False)}"
    answers = {_FREE_MARK: _free(free), _CLASSIFIER_MARK: _FOREIGN, "Сообщение пациента": _PICK_NONE}

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            debug = (body.get("state_update") or {}).get("debug") or {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "label": (debug.get("decision") or {}).get("label"),
                    "state": {k: v for k, v in dict(memory.get(session).last_entities or {}).items() if k != "_pending"},
                    "full": debug.get("understanding_full"),
                    "telemetry": world.take_telemetry() or {},
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {
            "MR_PENDING_BUTTONS": "on",
            "MR_CHOOSE_CACHE_TTL_S": 0,
            "MR_BUTTON_ANSWER_TOPIC": "frame",
            "MR_TURN_UNDERSTANDING_FULL": full,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


_DIALOGS = (
    ["сколько стоит ферритин"],
    [_press("menu.price.test"), "ферритин"],
    [_press("menu.tests.prepare"), "к общему анализу крови"],
    [_press("menu.appointment.book"), "к терапевту"],
    [_press("menu.address"), "а на Победы до скольки?"],
)


def _llm_calls(telemetry: dict[str, Any]) -> int:
    return len(telemetry.get("choose") or []) + len(telemetry.get("llm") or [])


@pytest.mark.parametrize("turns", _DIALOGS, ids=lambda t: str(t[-1]))
def test_shadow_is_invisible_and_calls_no_llm(turns, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, turns, full="off")
    shadow = _dialog(mini_prod_world, monkeypatch, turns, full="shadow")
    for before, after in zip(off, shadow):
        assert (after["text"], after["label"], after["state"]) == (before["text"], before["label"], before["state"])
        assert _llm_calls(after["telemetry"]) == _llm_calls(before["telemetry"]), (before["telemetry"], after["telemetry"])
        assert before["full"] is None and "tu_full_intent_src" not in before["telemetry"]


def test_every_turn_gets_the_full_understanding(mini_prod_world, monkeypatch):  # noqa: F811
    turns = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), "ферритин", "а где сдать?"], full="shadow")
    for turn in turns:
        assert turn["full"] is not None and turn["full"]["intent"]["final"], turn
        assert turn["telemetry"].get("tu_full_intent_src"), turn["telemetry"]


def test_press_and_reply_to_the_button_question_are_framed(mini_prod_world, monkeypatch):  # noqa: F811
    press, reply = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), "ферритин"], full="shadow")
    assert press["full"]["frame"] == {"button": "menu.price.test", "kind": "press", "topic": "PRICE", "held": True}
    assert reply["full"]["frame"]["kind"] == "question" and reply["full"]["frame"]["button"] == "menu.price.test", reply["full"]
    assert reply["full"]["frame"]["held"] is True, reply["full"]
    # «ферритин» — значение слота разобрали правила (сужение без LLM): источник `fast`.
    assert reply["full"]["expects"] == {"type": "slot_value", "outcome": "answer", "source": "fast"}, reply["full"]
    assert reply["telemetry"].get("tu_expects") == "slot_value:answer", reply["telemetry"]


def test_typed_question_has_no_frame_and_copies_the_entities(mini_prod_world, monkeypatch):  # noqa: F811
    (turn,) = _dialog(mini_prod_world, monkeypatch, ["сколько стоит ферритин"], full="shadow")
    assert turn["full"]["frame"] is None and "tu_frame" not in turn["telemetry"], turn["full"]
    assert turn["full"]["intent"]["final"] == turn["label"], turn["full"]


def test_observer_failure_is_swallowed_with_a_trace(mini_prod_world, monkeypatch):  # noqa: F811
    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    off = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), "ферритин"], full="off")
    monkeypatch.setattr(turn_understanding, "_frame_of", _boom)
    broken = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), "ферритин"], full="shadow")
    assert [t["text"] for t in broken] == [t["text"] for t in off]
    assert broken[1]["telemetry"].get("tu_full_error") == "RuntimeError", broken[1]["telemetry"]
    assert broken[1]["full"] is None


def test_frame_held_is_the_same_decision_the_policy_table_reads(mini_prod_world, monkeypatch):  # noqa: F811
    """«Держится ли рамка» — одно решение (`button_flow.note_frame`): его читают и таблица политик, и полное понимание хода."""

    from messengers_router import button_flow

    seen: list[object] = []
    real = button_flow.turn_frame
    monkeypatch.setattr(button_flow, "turn_frame", lambda: seen.append(real()) or seen[-1])
    reply = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), "а вы делаете узи щитовидки?"], full="shadow", free="new_topic")[1]
    decided = seen[-1]
    assert decided is not None and reply["full"]["frame"]["held"] is decided.held, (decided, reply["full"])
    assert reply["full"]["frame"]["held"] is False, reply["full"]
