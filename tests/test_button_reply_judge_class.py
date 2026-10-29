"""Судья ответа на вопрос кнопки меню видит нажатую кнопку и её тему (BUG-2026-10-27-BUTTON-REPLY-JUDGE-FOREIGN, выключатель
`button_reply_judge`).

Карточка 25.10 и карта провалов деплоя 18: у 54 % провалов набора кнопок тема ответа не совпала с темой нажатого листа.
Судья одной функции (`choose_pending_free_reply`) видел только вопрос бота («Какой анализ интересует?») и полный вопрос по
теме кнопки («Сколько стоит общий анализ крови?») звал «другой темой» — на «Цене анализа» 214 из 343, — и тему снова решал
общий путь (правила → классификатор).

Инвариант класса: под `button_reply_judge = on` (поверх `pending_buttons = on`) ответ на вопрос любой кнопки-темы судит
судья, которому даны подпись, раздел и тема нажатой кнопки (задача `pending_free_reply_button`); его «ответ» — решение
одной функции, и тема кнопки держится, как у любого ответа на её вопрос (`button_answer_topic`). Встречные: off и вопрос не
кнопки — прежний судья (задача `pending_free_reply`, данные — только вопрос бота); «другая тема» этого судьи и его сбой —
как было. Через обработчик эндпоинта; LLM — подделка настоящего типа: судья с кнопкой отвечает «ответ», прежний судья —
«другая тема» (как на карточке), классификатор — чужой темой, чтобы его вызов был виден.
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
from tests.test_button_answer_topic_class import _CLASSIFIER_MARK, _FOREIGN, _FREE_MARK, _JUDGED, _PICK_NONE, _free
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_BUTTON_MARK = "написал ли пациент в теме нажатой кнопки"
# Полный вопрос в теме кнопки — по одному на тему (что ответит путь темы, здесь не важно: важно, чей судья и что решил).
_IN_TOPIC = {
    "PRICE": "Сколько стоит общий анализ крови?",
    "PREPARE": "Как подготовиться к УЗИ брюшной полости?",
    "TEST_ASSIST": "Какие анализы сдать при выпадении волос?",
    "TEST_RESULT": "Как узнать результат, если номер заказа потерялся?",
}


def _dialog(
    world_dir, monkeypatch, turns, *, judge: str, topic: str = "on", button_judge: str | None = "answer", old_judge: str | None = "new_topic"
) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"reply-judge-{judge}-{topic}-{button_judge}-{old_judge}-{json.dumps(turns, ensure_ascii=False)}"
    seen: list[str] = []
    answers: dict[str, Any] = {}
    if button_judge is not None:
        answers[_BUTTON_MARK] = lambda prompt: (seen.append(prompt), _free(button_judge))[1]
    if old_judge is not None:
        answers[_FREE_MARK] = lambda prompt: (seen.append(prompt), _free(old_judge))[1]
    answers[_CLASSIFIER_MARK] = _FOREIGN
    answers["Сообщение пациента"] = _PICK_NONE  # прочие места выбора — «ничего не подходит»

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            del seen[:]
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "label": decision.get("label"),
                    "state": dict(memory.get(session).last_entities or {}),
                    "telemetry": world.take_telemetry() or {},
                    "judge_data": list(seen),
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {
            "MR_PENDING_BUTTONS": "on",
            "MR_CHOOSE_CACHE_TTL_S": 0,
            "MR_TOPIC_BY_LLM": "",
            "MR_BUTTON_ANSWER_TOPIC": topic,
            "MR_BUTTON_REPLY_JUDGE": judge,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def _judges(tel: dict[str, Any]) -> list[str]:
    return [c.get("task") for c in tel.get("choose") or [] if str(c.get("task") or "").startswith("pending_free_reply")]


@pytest.mark.parametrize("button_id", _JUDGED)
def test_button_question_is_judged_seeing_the_button(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Под on судья ответа на вопрос кнопки — с кнопкой: в его данных подпись и тема нажатого листа, полный вопрос в теме
    кнопки — «ответ»; off — прежний судья по одному вопросу бота сказал «другая тема» (как на карточке 25.10)."""

    action = button_menu.BUTTONS[button_id]
    turns = [_press(button_id), _IN_TOPIC[action.label]]
    on = _dialog(mini_prod_world, monkeypatch, turns, judge="on")
    off = _dialog(mini_prod_world, monkeypatch, turns, judge="off")
    assert on[0]["text"] == off[0]["text"], "нажатие отвечает как было"
    tel_on, tel_off = on[1]["telemetry"], off[1]["telemetry"]
    assert _judges(tel_on) == ["pending_free_reply_button"], tel_on
    assert action.title in on[1]["judge_data"][0] and action.about in on[1]["judge_data"][0], on[1]["judge_data"]
    # Значение нашли правила — судья проверяет только «другую тему», и «ответ» остаётся решением правил (`fast`).
    assert tel_on.get("tu_pending_outcome") == "answer" and tel_on.get("tu_pending_src") in {"llm", "fast"}, tel_on
    assert _judges(tel_off) == ["pending_free_reply"] and tel_off.get("tu_pending_outcome") == "new_topic", tel_off
    assert action.title not in off[1]["judge_data"][0], "off — прежние данные судьи байт в байт"


@pytest.mark.parametrize("button_id", _JUDGED)
def test_answer_of_button_judge_keeps_button_topic(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс до конца хода: «ответ» судьи с кнопкой — тема кнопки там, где тему решал бы классификатор (`button_answer_topic`);
    при «другой теме» прежнего судьи тему решал классификатор (подделка — чужая тема)."""

    topic = button_menu.BUTTONS[button_id].label
    turns = [_press(button_id), "для мамы, ей 60 лет"]  # своего правила нет — тему решал бы классификатор
    on = _dialog(mini_prod_world, monkeypatch, turns, judge="on")
    off = _dialog(mini_prod_world, monkeypatch, turns, judge="off")
    tel_on, tel_off = on[1]["telemetry"], off[1]["telemetry"]
    assert tel_off.get("tu_pending_outcome") == "new_topic" and tel_off.get("nlu_llm") is True, tel_off
    assert tel_on.get("tu_pending_outcome") == "answer", tel_on
    assert tel_on.get("nlu_llm") is False and tel_on.get("nlu_topic_by_llm") == "button_answer", tel_on
    assert tel_on.get("tu_label_nlu") == topic, tel_on


def test_typed_question_keeps_the_old_judge(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: тот же вопрос подготовки набран руками — висящий вопрос не кнопки, судья прежний."""

    on = _dialog(mini_prod_world, monkeypatch, ["как подготовиться к процедуре", "для мамы, ей 60 лет"], judge="on")
    assert not button_flow.question_button(on[0]["state"].get("_pending"))
    tel = on[1]["telemetry"]
    assert "pending_free_reply_button" not in _judges(tel), tel


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.prepare"])
def test_new_topic_by_button_judge_keeps_general_path(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: судья с кнопкой сказал «другая тема» — тему решает общий путь (классификатор), как было."""

    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), "для мамы, ей 60 лет"], judge="on", button_judge="new_topic")
    tel = on[1]["telemetry"]
    assert tel.get("tu_pending_outcome") == "new_topic" and tel.get("nlu_llm") is True, tel
    assert tel.get("nlu_topic_by_llm") != "button_answer", tel


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.suggest"])
def test_button_judge_failure_is_the_old_fallback(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: LLM судьи недоступна — запасной путь, ответ тот же, что без выключателя при той же недоступности."""

    turns = [_press(button_id), "для мамы, ей 60 лет"]
    on = _dialog(mini_prod_world, monkeypatch, turns, judge="on", button_judge=None, old_judge=None)
    off = _dialog(mini_prod_world, monkeypatch, turns, judge="off", button_judge=None, old_judge=None)
    assert on[1]["telemetry"].get("tu_pending_src") in {"fallback", None}, on[1]["telemetry"]
    assert (on[1]["text"], on[1]["label"]) == (off[1]["text"], off[1]["label"])


def test_judge_switch_alone_marks_the_button_question(mini_prod_world, monkeypatch):  # noqa: F811
    """Метка листа у висящего вопроса — и когда её читает только судья (прочие читатели off): иначе судья кнопку не увидит."""

    with monkeypatch.context() as patch:
        from messengers_router.runtime_config import config as cfg

        patch.setattr(cfg, "MR_BUTTON_PRICE_SCOPE", "off", raising=False)
        on = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.prepare"), "для мамы, ей 60 лет"], judge="on", topic="off")
    assert button_flow.question_button(on[0]["state"].get("_pending")) == "menu.tests.prepare"
    assert _judges(on[1]["telemetry"]) == ["pending_free_reply_button"], on[1]["telemetry"]
