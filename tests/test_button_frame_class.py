"""Кнопка задаёт рамку хода: тему ответа на вопрос кнопки не перебивают несрочные правила NLU (STEP-2026-10-29-BUTTON-FRAME,
`button_answer_topic = frame`).

Стенд деплоя 20 (`s20con_btn`): у 86 из 236 провалов кнопок тема ответа ≠ теме нажатого листа, 50 из них — «Цена анализа». На
кнопках цены рамки не было (`slot`), а под `on` тему кнопки, признанную судьёй ответа, перебивало любое сработавшее правило не из
`topic_by_llm`: «Можно ли сделать фгдс и колоноскопию вместе под наркозом» → `rule_appointment` («какой филиал вам удобен?»),
«справка 079у» → `doc_request_*` («Соединяю с оператором»), «кт легких» → `unsupported_*` («Уточните детали запроса»).

Инвариант класса: под `frame` ответ на вопрос кнопки, который судья признал ответом, — в теме кнопки и тогда, когда сработало
несрочное правило другой темы (`nlu_topic_by_llm = button_answer`); под `on` — как было (правило решает). Встречные: «другая тема»
по решению судьи, медвопрос по правилу, та же реплика без кнопки — как без выключателя. Через обработчик эндпоинта; LLM — подделка
настоящего типа (харнесс `test_button_answer_topic_class`).
"""

from __future__ import annotations

import pytest

from tests.test_button_answer_topic_class import _dialog, _press
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_BUTTONS = ("menu.price.test", "menu.price.doctor")
# Реплики в теме цены, на которые срабатывает несрочное правило другой темы (запись).
_RULED = ("Можно ли сделать фгдс и колоноскопию вместе под наркозом", "Можно ли пройти 3д узи плода?")


def _rule(tel: dict) -> str:
    return str(tel.get("nlu_rule") or "")


@pytest.mark.parametrize("button_id", _BUTTONS)
@pytest.mark.parametrize("reply", _RULED)
def test_on_lets_the_rule_take_the_topic_as_before(button_id, reply, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _dialog(mini_prod_world, monkeypatch, [_press(button_id), reply], switch="on")[1]
    assert "rule_appointment" in _rule(turn["telemetry"]), turn["telemetry"]
    assert turn["label"] == "APPOINTMENT", turn


@pytest.mark.parametrize("button_id", _BUTTONS)
@pytest.mark.parametrize("reply", _RULED)
def test_frame_keeps_the_button_topic_against_the_rule(button_id, reply, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _dialog(mini_prod_world, monkeypatch, [_press(button_id), reply], switch="frame")[1]
    assert turn["telemetry"].get("nlu_topic_by_llm") == "button_answer", turn["telemetry"]
    assert turn["label"] == "PRICE", turn


@pytest.mark.parametrize("reply", _RULED)
def test_frame_new_topic_by_the_judge_keeps_the_rule(reply, mini_prod_world, monkeypatch):  # noqa: F811
    turn = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), reply], switch="frame", free="new_topic")[1]
    assert turn["telemetry"].get("nlu_topic_by_llm") != "button_answer", turn["telemetry"]
    assert turn["label"] == "APPOINTMENT", turn


def test_frame_medical_advice_rule_still_wins(mini_prod_world, monkeypatch):  # noqa: F811
    turn = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), "что со мной, болит голова третий день"], switch="frame")[1]
    assert turn["telemetry"].get("answer_source") == "safety:medical_advice", turn["telemetry"]


@pytest.mark.parametrize("reply", _RULED)
def test_typed_question_without_button_is_untouched(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [reply], switch="off")[0]
    frame = _dialog(mini_prod_world, monkeypatch, [reply], switch="frame")[0]
    assert frame["text"] == off["text"] and frame["label"] == off["label"]


@pytest.mark.parametrize("button_id", _BUTTONS)
def test_price_leaf_domain_names_availability_for_the_judge(button_id, monkeypatch):
    """Судья ответа видит домен листа цены — каталог услуг: «есть ли услуга / врач» — вопрос в теме кнопки."""

    from messengers_router import button_flow
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_BUTTON_REPLY_JUDGE", "on", raising=False)
    context = button_flow.judge_context({"button": button_id}, "Какой анализ интересует?")
    assert "есть ли" in context and "цену" in context, context
