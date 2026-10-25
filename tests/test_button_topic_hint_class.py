"""Чужая тема в ответ на вопрос кнопки меню — подсказка нужной кнопки (BUG-2026-10-25-BUTTON-FOREIGN-TOPIC, выключатель
`button_topic_hint`).

Предложение владельца 25.10: ~90 % пациентов придут через меню-кнопки шлюза; в ответ на вопрос кнопки («Цена анализа» →
«Какой анализ интересует?») пациент бывает пишет о другом («а где вы находитесь?»). Было — тему угадывал общий путь
(LLM-классификатор, p50 3,7 с, и его ошибки); стало — LLM выбирает из закрытого списка листьев меню, бот подсказывает кнопку.

Инвариант класса: под `button_topic_hint = on` (поверх `pending_buttons = on`) ответ на вопрос любой кнопки-темы, который
одна функция признала другой темой и тему которого общий путь угадывал бы LLM-классификатором, а выбор LLM указал лист
другой темы, — подсказка этого листа (подпись, раздел, о чём), без классификатора; вопрос кнопки снят. Смешанное сообщение
(лист кнопки + лист другой темы) — ответ по теме кнопки и подсказка в конце. Встречные: off; надёжное правило другой темы
отвечает само, как было; лист той же темы, «ничего», сбой LLM подсказки — общий путь, как без выключателя, и решение
подсказки лимит хода не тратит; «это ответ» и тот же вопрос без кнопки — без подсказки. Через обработчик эндпоинта;
LLM — подделка настоящего типа, классификатор отвечает чужой темой, чтобы его вызов был виден.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import button_flow, button_menu, choice, llm_runtime
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_button_answer_topic_class import _JUDGED
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_FREE_MARK = "Бот попросил назвать что-то своими словами"
_HINT_MARK = "Пациент нажал кнопку в меню бота"
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_FOREIGN = json.dumps({"label": "APPOINTMENT", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
_PICK_NONE = json.dumps({"picks": []})
_REPLY = "а можно прийти с собакой?"  # без своего правила: тему угадывал бы классификатор
_RULED = "а у вас в субботу открыто?"  # надёжное правило часов филиалов
_OPTIONS = [o.id for o in button_flow.menu_options()]


def _picks(*leaves: str) -> str:
    return json.dumps({"picks": [_OPTIONS.index(leaf) + 1 for leaf in leaves]})


def _other(button_id: str) -> str:
    """Лист другой темы, чем у кнопки: адреса, а для адресной кнопки — цена анализа."""

    label = button_menu.BUTTONS[button_id].label
    return "menu.address" if label != "ADDRESS" else "menu.price.test"


def _dialog(world_dir, monkeypatch, turns, *, switch: str, hint: str | None, free: str = "new_topic") -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"hint-{switch}-{hint}-{free}-{json.dumps(turns, ensure_ascii=False)}"
    answers: dict[str, Any] = {_FREE_MARK: json.dumps({"picks": [], "outcome": free})}
    if hint is not None:
        answers[_HINT_MARK] = hint
    answers[_CLASSIFIER_MARK] = _FOREIGN
    answers["Сообщение пациента"] = _PICK_NONE  # прочие места выбора — «ничего не подходит»

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
            "MR_NLU_ENGINE": "legacy_v2",
            "MR_PENDING_BUTTONS": "on",
            "MR_CHOOSE_CACHE_TTL_S": 0,
            "MR_BUTTON_ANSWER_TOPIC": "on",
            "MR_BUTTON_TOPIC_HINT": switch,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def test_menu_options_cover_every_leaf_but_operator():
    assert set(_OPTIONS) == set(button_menu.BUTTONS) - {"menu.operator"}
    for leaf in _OPTIONS:
        action = button_menu.BUTTONS[leaf]
        assert action.title and action.about, leaf
        text = button_flow.hint_text((leaf,))
        assert f"«{action.title}»" in text and action.about in text, text
        assert (f"раздела «{action.section}»" in text) if action.section else ("главного меню" in text), text


@pytest.mark.parametrize("button_id", _JUDGED)
def test_foreign_topic_gets_the_button_hint(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    leaf = _other(button_id)
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLY], switch="on", hint=_picks(leaf))
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLY], switch="off", hint=_picks(leaf))
    assert on[0]["text"] == off[0]["text"], "нажатие отвечает как было"
    tel = on[1]["telemetry"]
    assert on[1]["text"] == button_flow.hint_text((leaf,)) and not on[1]["handoff"], on[1]["text"]
    assert tel.get("button_hint") == leaf and tel.get("nlu_llm") is not True, tel
    assert "_pending" not in on[1]["state"], "вопрос кнопки снят — пациент идёт в подсказанную кнопку"
    assert off[1]["text"] != on[1]["text"] and "button_hint" not in off[1]["telemetry"]
    assert off[1]["telemetry"].get("nlu_llm") is True, "off — тему угадывает общий путь, как было"


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.prepare", "menu.results.lookup"])
def test_two_foreign_topics_get_both_buttons(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    leaves = ("menu.address", "menu.price.promo") if button_menu.BUTTONS[button_id].label != "NEWS" else ("menu.address",)
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLY], switch="on", hint=_picks(*leaves))
    assert on[1]["text"] == button_flow.hint_text(leaves), on[1]["text"]


@pytest.mark.parametrize(
    ("button_id", "pick"),
    [("menu.price.test", "menu.tests.price"), ("menu.price.doctor", "menu.price.test"), ("menu.tests.prepare", "menu.appointment.prepare")],
)
def test_same_topic_leaf_is_not_a_hint(button_id, pick, mini_prod_world, monkeypatch):  # noqa: F811
    """Лист той же темы, что у кнопки, — не подсказка: ход как без выключателя, решение подсказки лимит не тратит."""

    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLY], switch="on", hint=_picks(pick))
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLY], switch="off", hint=_picks(pick))
    assert (on[1]["text"], on[1]["label"]) == (off[1]["text"], off[1]["label"])
    assert on[1]["telemetry"].get("button_hint") == "none" and on[1]["telemetry"].get("tu_button_hint_refund") == 1


@pytest.mark.parametrize("hint", [_PICK_NONE, None, "не JSON"])
def test_no_hint_keeps_general_path(hint, mini_prod_world, monkeypatch):  # noqa: F811
    """«Ничего не подходит», LLM подсказки недоступна или ответила не по контракту — общий путь, как без выключателя."""

    turns = [_press("menu.price.test"), _REPLY]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", hint=hint)
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", hint=hint)
    assert (on[1]["text"], on[1]["label"], on[1]["handoff"]) == (off[1]["text"], off[1]["label"], off[1]["handoff"])
    assert on[1]["telemetry"].get("button_hint") == "none"


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.prepare"])
def test_mixed_message_answers_in_topic_and_hints_the_rest(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """В сообщении и ответ на вопрос кнопки, и другой вопрос: ответ по теме кнопки, подсказка — в конце."""

    topic = button_menu.BUTTONS[button_id].label
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLY], switch="on", hint=_picks(button_id, "menu.address"))
    tel = on[1]["telemetry"]
    assert tel.get("button_hint") == "mixed:menu.address" and tel.get("nlu_llm") is not True, tel
    assert tel.get("tu_label_nlu") == topic, tel
    tail = button_flow.hint_text(("menu.address",), tail=True)
    assert on[1]["text"].endswith(tail) and on[1]["text"].strip() != tail, on[1]["text"]


def test_answer_outcome_gets_no_hint(mini_prod_world, monkeypatch):  # noqa: F811
    on = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), _REPLY], switch="on", hint=_picks("menu.address"), free="answer")
    assert "button_hint" not in on[1]["telemetry"] and button_flow.hint_text(("menu.address",)) not in on[1]["text"]


def test_typed_question_without_button_gets_no_hint(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: тот же вопрос набран руками — висящий вопрос не кнопки, подсказки нет."""

    on = _dialog(mini_prod_world, monkeypatch, ["как подготовиться к анализу", _REPLY], switch="on", hint=_picks("menu.address"))
    assert "button_hint" not in on[1]["telemetry"], on[1]["telemetry"]


def test_reliable_rule_of_other_topic_answers_itself(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: надёжное правило другой темы («в субботу открыто?» — часы) отвечает само, как без выключателя."""

    turns = [_press("menu.price.test"), _RULED]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", hint=_picks("menu.address"))
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", hint=_picks("menu.address"))
    assert off[1]["telemetry"].get("nlu_src") == "rule", off[1]["telemetry"]
    assert (on[1]["text"], on[1]["label"]) == (off[1]["text"], off[1]["label"])
    assert "button_hint" not in on[1]["telemetry"], on[1]["telemetry"]
