"""Лист без своего вопроса проверяет тот же судья, что ответ на вопрос кнопки (Р2 консилиума 30.10, STEP-2026-10-30-R2-LEAF-JUDGE).

Три определения «в теме кнопки» (карта «Понимания» 30.10, П-4): судья ответа с исходом «domain» (листы с вопросом, вариант Б),
отдельный вопрос рамки `choose_button_frame` (лист без вопроса — «Нужно ли записываться»), сравнение меток подсказки. Под
`button_frame_check = judge_leaf` лист без вопроса судит тот же судья (`pending_answer.FRAME_BUTTON_TASK`, тот же промпт) по
метке нажатия: в данных — лист и «бот ответил на нажатие, своего вопроса не задавал».

Инвариант класса: под `judge_leaf` «в теме листа» решает судья ответа («answer» / «domain» — рамка листа, «new_topic» и сбой —
общий путь), вопроса рамки нет; листы с вопросом — как под `judge`; `judge` и `question` — как было. Через обработчик эндпоинта,
LLM — подделка настоящего типа, ключи кнопок деплоя 21.
"""

from __future__ import annotations

import pytest

from messengers_router import button_flow, button_menu
from tests.test_frame_policy_class import _dialog, _press
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_LEAF = "menu.tests.booking_needed"
_LEAF_REPLY = "Записаться на хгч"
_JUDGE_LEAF = {"MR_BUTTON_FRAME_CHECK": "judge_leaf"}


def _leaf(world_dir, monkeypatch, *, judge: str, frame: str | None, switches: dict | None = None) -> dict:
    return _dialog(
        world_dir, monkeypatch, [_press(_LEAF), _LEAF_REPLY], policy="off", classifier="APPOINTMENT", judge=judge, frame=frame, switches=switches
    )[1]


@pytest.mark.parametrize("judge", ["domain", "answer"])
def test_judge_leaf_holds_the_leaf_frame_by_the_judge(judge, mini_prod_world, monkeypatch):  # noqa: F811
    # Вопрос рамки сказал бы «другое» — значит, рамку держит судья, а вопроса рамки не было.
    turn = _leaf(mini_prod_world, monkeypatch, judge=judge, frame="other", switches=_JUDGE_LEAF)
    tel = turn["telemetry"]
    assert tel.get("button_frame") == f"judge:{judge}", tel
    assert tel.get("nlu_topic_by_llm") == "button_answer" and tel.get("tu_label_nlu") == "ADDRESS", tel


@pytest.mark.parametrize("judge", ["new_topic", "broken"])
def test_judge_leaf_new_topic_or_failure_takes_the_general_path(judge, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _leaf(mini_prod_world, monkeypatch, judge=judge, frame="same", switches=_JUDGE_LEAF)
    tel = turn["telemetry"]
    assert tel.get("button_frame") == f"judge:{'none' if judge == 'broken' else judge}", tel
    assert tel.get("nlu_topic_by_llm") != "button_answer", tel


@pytest.mark.parametrize("check", ["judge", "question"])
def test_other_checks_keep_the_frame_question_on_the_leaf(check, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _leaf(mini_prod_world, monkeypatch, judge="new_topic", frame="same", switches={"MR_BUTTON_FRAME_CHECK": check})
    tel = turn["telemetry"]
    assert tel.get("button_frame") == "same" and tel.get("nlu_topic_by_llm") == "button_answer", tel


@pytest.mark.parametrize("judge", ["domain", "new_topic"])
def test_question_leaves_under_judge_leaf_are_judge(judge, mini_prod_world, monkeypatch):  # noqa: F811
    turns = [_press("menu.price.test"), "есть ли у вас узи щитовидки"]
    b = _dialog(mini_prod_world, monkeypatch, turns, policy="off", judge=judge)[1]
    leaf = _dialog(mini_prod_world, monkeypatch, turns, policy="off", judge=judge, switches=_JUDGE_LEAF)[1]
    assert (leaf["text"], leaf["label"]) == (b["text"], b["label"])
    assert leaf["telemetry"].get("button_frame") == b["telemetry"].get("button_frame")


def test_leaf_context_says_no_question_was_asked():
    context = button_flow.leaf_judge_context(button_menu.action_for(_LEAF))
    assert "Нужно ли записываться" in context and "своего вопроса не задавал" in context and "вопрос бота" not in context
