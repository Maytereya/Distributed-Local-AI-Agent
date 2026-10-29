"""Шаг 3 деплоя 22: «продолжает ли реплика висящий вопрос» решает одно решение хода; Д2 звено 2 — вариант (А) консилиума
(`[MESSENGER_ROUTER] expects_by_understanding = off | on`, умолчание off; STEP-2026-10-30-EXPECTS-BY-UNDERSTANDING).

Звено 2 BUG-2026-10-28-BUTTON-PRESS-AS-REPEAT: анти-залип дописывал «Перевести на оператора?» к вопросу бота — в сообщении два
вопроса, а `_expects` — оффер; ответ на вопрос бота («ферритин» на «Какой анализ интересует?») судья читал согласием на перевод.
Вариант (А), решение владельца 29.10: первый повтор — вопрос слота единственный висящий, оффер — строкой без «?»; второй шаг без
продвижения — один вопрос-оффер с причиной.

Инварианты класса (под on): И-1 — в сообщении бота не больше одного висящего вопроса, и `_expects` — ровно он (проверяет и
`tests/conftest.py::_one_hanging_question` на каждом ходе гейта); И-2 — перевод только при `_expects = operator_offer` или явной
просьбе; И-3 — тот же вопрос слота подряд не больше двух раз. Off — как было.

Через обработчик эндпоинта на 2–5 ходов (харнесс `tests/understanding_contract.py`, мир `d21_world`, конфиг деплоя 21).
"""

from __future__ import annotations

import pytest

from messengers_router import expects
from messengers_router.mess_types import RouteDecision
from messengers_router.recovery_policy import OPERATOR_HINT_TEXT, evaluate_recovery, operator_offer_with_reason
from messengers_router.router import _REPEAT_GUARD_OFFER
from tests import understanding_contract as uc
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.understanding_contract import d21_world  # noqa: F401 — фикстура модуля

ON = {"MR_EXPECTS_BY_UNDERSTANDING": "on"}
OFFER = expects.OPERATOR_OFFER_QUESTION

# Вид вопроса слота → диалог, где бот повторяет его на не-ответ: (ходы до повтора, реплика-не-ответ, вид `_expects`, причина).
# Ступени (А): ход №k — первый повтор (строка без «?»), ход №k+1 — один вопрос-оффер с причиной.
_PREPARE = [uc.press("menu.appointment.prepare"), uc.say("не знаю", pending="no"), uc.say("без понятия", pending="no")]
_SUGGEST = [uc.press("menu.tests.suggest"), uc.say("хм", pending="none")]
_RESULTS = [uc.press("menu.results.lookup"), uc.say("не помню номер", pending="answer")]
_BOOK = [uc.press("menu.appointment.book"), uc.say("хм", pending="none"), uc.say("ну хм", pending="none")]
_BRANCH = [uc.say("запишите к кардиологу", topic="APPOINTMENT"), uc.say("хм"), uc.say("ну хм"), uc.say("ещё хм")]
SLOTS = {
    "procedure": (_PREPARE, uc.say("совсем не знаю", pending="no"), ("slot_value", "free"), "название анализа или услуги"),
    "test_goal": (_SUGGEST, uc.say("ну хм", pending="none"), ("slot_value", "free"), "цель подбора анализов"),
    "result_data": (_RESULTS, uc.say("не помню номер вообще", pending="answer"), ("slot_value", "free"), "данные для поиска результата"),
    "booking_target": (_BOOK, uc.say("ещё раз хм", pending="none"), ("slot_value", "free"), "врача или услугу для записи"),
    "branch_pick": (_BRANCH, uc.say("опять хм"), ("appointment_branch", "pick"), "ваш выбор из списка"),
}


def _hint_turn(record: list[dict]) -> dict:
    return record[-1]


def _consecutive_same(record: list[dict]) -> int:
    """Сколько раз подряд (максимум) бот прислал одно и то же сообщение."""

    best = run = 0
    prev = None
    for turn in record:
        run = run + 1 if turn["text"] == prev else 1
        best, prev = max(best, run), turn["text"]
    return best


# --------------------------------------------------------------------------- ступени (А) на всех видах вопроса слота


@pytest.mark.parametrize("slot", list(SLOTS))
def test_first_repeat_keeps_the_slot_question_with_an_offer_line_without_question(slot, d21_world, monkeypatch):  # noqa: F811
    turns, _non_answer, kind, _reason = SLOTS[slot]
    record = uc.run(d21_world, monkeypatch, turns, cfg=ON)
    hint = _hint_turn(record)
    assert hint["text"].endswith(OPERATOR_HINT_TEXT) and "?" not in OPERATOR_HINT_TEXT, hint["text"]
    assert OFFER not in hint["text"] and not hint["handoff"], hint["text"]
    assert ((hint["expects"] or {}).get("type"), (hint["expects"] or {}).get("kind")) == kind, hint["expects"]


@pytest.mark.parametrize("slot", list(SLOTS))
def test_second_step_without_progress_is_one_offer_question_with_a_reason(slot, d21_world, monkeypatch):  # noqa: F811
    turns, non_answer, _kind, reason = SLOTS[slot]
    record = uc.run(d21_world, monkeypatch, [*turns, non_answer], cfg=ON)
    offer = record[-1]
    assert offer["text"] == operator_offer_with_reason(reason) and offer["text"].count("?") == 1, offer["text"]
    assert (offer["expects"] or {}).get("type") == "operator_offer" and not offer["handoff"], offer["expects"]
    assert _consecutive_same(record) <= 2, [t["text"] for t in record]  # И-3


@pytest.mark.parametrize("slot", ["procedure", "test_goal", "result_data", "booking_target"])
def test_yes_after_the_offer_line_is_not_a_handoff(slot, d21_world, monkeypatch):  # noqa: F811
    """«да» после необязывающей строки — ответ на вопрос слота (его и разбирает судья), не согласие на перевод (И-2)."""

    turns, *_ = SLOTS[slot]
    for judged in ("answer", "none"):
        record = uc.run(d21_world, monkeypatch, [*turns, uc.say("да", pending=judged)], cfg=ON, session=f"{slot}-{judged}")
        assert not record[-1]["handoff"] and "Соединяю" not in record[-1]["text"], (judged, record[-1]["text"])


def test_answer_after_the_offer_line_goes_to_the_slot(d21_world, monkeypatch):  # noqa: F811
    """Ответ-название после строки — в обработчик слота (реквизиты → поиск результата; процедура → путь подготовки)."""

    record = uc.run(d21_world, monkeypatch, [*_RESULTS, uc.say("Иванов, 1989, Бг, 123", pending="answer")], cfg=ON)
    assert record[-1]["label"] == "TEST_RESULT" and not record[-1]["handoff"], (record[-1]["label"], record[-1]["text"])
    record = uc.run(d21_world, monkeypatch, [*_PREPARE, uc.say("гастроскопия", pending="answer", services=("гастроскопия",))], cfg=ON)
    assert record[-1]["label"] == "PREPARE" and not record[-1]["handoff"], (record[-1]["label"], record[-1]["text"])


@pytest.mark.parametrize("step", [1, 2, 3])
def test_explicit_operator_on_any_step_is_a_handoff(step, d21_world, monkeypatch):  # noqa: F811
    """Явное «оператор» — перевод на любом шаге: после вопроса, после строки, после вопроса-оффера."""

    turns = [*_SUGGEST, uc.say("ну хм", pending="none")][:step]
    record = uc.run(d21_world, monkeypatch, [*turns, uc.say("оператор", pending="new_topic")], cfg=ON)
    assert record[-1]["handoff"], record[-1]["text"]


def test_offer_question_takes_yes_as_a_handoff(d21_world, monkeypatch):  # noqa: F811
    """Висит только вопрос-оффер — «да» на него законно переводит."""

    record = uc.run(d21_world, monkeypatch, [*_SUGGEST, uc.say("ну хм", pending="none"), uc.say("да")], cfg=ON)
    assert record[-1]["handoff"], record[-1]["text"]


# --------------------------------------------------------------------------- генеративно: все листья с вопросом × три не-ответа


@pytest.mark.parametrize("leaf", sorted(GATEWAY_V1_TITLES))
def test_every_leaf_with_non_answers_keeps_one_hanging_question(leaf, d21_world, monkeypatch):  # noqa: F811
    """И-1 на каждом ходе (проверяет и автофикстура гейта) и И-3 — у всех листьев меню на три не-ответа подряд."""

    turns = [uc.press(leaf), *(uc.say(text, pending="none") for text in ("хм", "ну хм", "ещё хм"))]
    record = uc.run(d21_world, monkeypatch, turns, cfg=ON)
    for turn in record:
        assert not (OFFER in turn["text"] and "?" in turn["text"].split(OFFER)[0]), turn["text"]
        if OFFER in turn["text"] and not turn["handoff"]:
            assert (turn["expects"] or {}).get("type") == "operator_offer", turn["text"]
    assert _consecutive_same(record) <= 2, [t["text"] for t in record]


# --------------------------------------------------------------------------- off — как было


def test_off_puts_the_offer_question_on_top_of_the_slot_question_as_before(d21_world, monkeypatch):  # noqa: F811
    record = uc.run(d21_world, monkeypatch, _RESULTS)
    assert record[-1]["text"].endswith(_REPEAT_GUARD_OFFER) and (record[-1]["expects"] or {}).get("type") == "operator_offer"


@pytest.mark.parametrize("slot", list(SLOTS))
def test_off_is_the_same_as_without_the_key(slot, d21_world, monkeypatch):  # noqa: F811
    turns, non_answer, *_ = SLOTS[slot]
    dialog = [*turns, non_answer]
    off = uc.run(d21_world, monkeypatch, dialog, cfg={"MR_EXPECTS_BY_UNDERSTANDING": "off"}, session=f"off-{slot}")
    absent = uc.run(d21_world, monkeypatch, dialog, session=f"absent-{slot}")
    assert [(t["text"], t["handoff"], t["expects"]) for t in off] == [(t["text"], t["handoff"], t["expects"]) for t in absent]


# --------------------------------------------------------------------------- восстановление «не понял» и предел переспросов


def _unclear(**kwargs) -> list[str]:
    entities: dict = {}
    decision = RouteDecision(label="OTHER", confidence=0.2, flags={"low_confidence"})
    return [
        evaluate_recovery(user_text="хм", decision=decision, flow_label="OTHER", pending_exists=False, flow_active=False, state_entities=entities, **kwargs).text
        for _ in range(3)
    ]


def test_recovery_on_gives_a_line_then_one_offer_question():
    first, second, third = _unclear(one_question=True)
    assert OFFER not in first and second.endswith(OPERATOR_HINT_TEXT) and OFFER not in second
    assert third == operator_offer_with_reason("ваш вопрос")


def test_recovery_off_is_as_before():
    _first, second, _third = _unclear()
    assert second.endswith(f"\n\n{_REPEAT_GUARD_OFFER}")


def test_handler_reask_is_limited_under_on(monkeypatch):
    """Переспрос обработчика («Это верно? Ответьте «да» или «нет»») — не больше двух раз подряд и без `reask_limit` (И-3)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_REASK_LIMIT", "off", raising=False)
    monkeypatch.setattr(cfg, "MR_EXPECTS_BY_UNDERSTANDING", "off", raising=False)
    token = expects._ASKED.set((expects.Asked(expects.Question.CATALOG_CONFIRM, expects.YES_NO, (), True, 3),))
    try:
        assert expects.over_limit() is None
        monkeypatch.setattr(cfg, "MR_EXPECTS_BY_UNDERSTANDING", "on", raising=False)
        assert expects.over_limit() is not None
    finally:
        expects._ASKED.reset(token)


# --------------------------------------------------------------------------- И-1: проверка сама


@pytest.mark.parametrize(
    ("text", "field", "asked", "violation"),
    [
        ("Какой анализ интересует?\n\nПохоже, мне не удаётся помочь с этим в чате. Перевести на оператора?", {"type": "operator_offer"}, (), "two_questions"),
        ("Укажите фамилию, год, код и номер.\n\n" + OFFER, {"type": "operator_offer"}, (expects.Question.SLOT_VALUE,), "two_questions"),
        ("Нет данных — подскажет оператор. " + OFFER, {"type": "slot_value"}, (), "offer_not_expected"),
        ("Нет данных — подскажет оператор. " + OFFER, {"type": "operator_offer"}, (expects.Question.OPERATOR_OFFER,), ""),
        ("Какой анализ интересует?\n\n" + OPERATOR_HINT_TEXT, {"type": "slot_value"}, (expects.Question.SLOT_VALUE,), ""),
        ("Соединяю с оператором.", None, (), ""),
        ("Результат по ссылке: https://naykalab.ru/getanaliz.php?fam=X\n\n" + OFFER, {"type": "operator_offer"}, (expects.Question.OPERATOR_OFFER,), ""),
    ],
)
def test_one_question_check(text, field, asked, violation):
    token = expects._ASKED.set(tuple(expects.Asked(q, expects.FREE, (), False, 0) for q in asked))
    try:
        assert expects.one_question_violation(text, field) == violation
    finally:
        expects._ASKED.reset(token)


# --------------------------------------------------------------------------- предпроверка записи читает одно решение (коммит 2)

_SWITCH = "перейти к новому вопросу"
_TO_CARDIO = uc.say("запишите к кардиологу", topic="APPOINTMENT")


@pytest.mark.parametrize("reply", ["расскажите про вашу клинику", "а маме моей можно сделать массаж спины"])
def test_new_topic_judged_by_llm_asks_the_switch_question_where_rules_did_not(reply, d21_world, monkeypatch):  # noqa: F811
    """«Другая тема» модели на вопрос шага записи — вердикт: вопрос «перейти к новому вопросу?», а не запись дальше (off:
    детекторы тем реплику не видели — «массаж спины» становился подтверждением чужой услуги)."""

    dialog = [_TO_CARDIO, uc.say(reply, pending="new_topic", topic="OTHER")]
    off = uc.run(d21_world, monkeypatch, dialog, session=f"off-{reply}")
    on = uc.run(d21_world, monkeypatch, dialog, cfg=ON, session=f"on-{reply}")
    assert _SWITCH not in off[1]["text"] and _SWITCH in on[1]["text"], (off[1]["text"], on[1]["text"])
    assert on[1]["telemetry"].get("tu_expects_acted") == ["appointment_precheck"]


def test_not_a_new_topic_judged_by_llm_keeps_the_booking_where_rules_switched(d21_world, monkeypatch):  # noqa: F811
    """«Ничего из этого» модели — не смена темы: правило «`?` и ≥ 4 слов» больше не спрашивает «перейти к новому вопросу?»."""

    dialog = [uc.press("menu.appointment.book"), uc.say("а у вас есть парковка возле клиники?", pending="none")]
    off = uc.run(d21_world, monkeypatch, dialog, session="off-parking")
    on = uc.run(d21_world, monkeypatch, dialog, cfg=ON, session="on-parking")
    assert _SWITCH in off[1]["text"] and _SWITCH not in on[1]["text"], (off[1]["text"], on[1]["text"])
    assert (on[1]["expects"] or {}).get("type") == "slot_value" and not on[1]["handoff"], on[1]["text"]


@pytest.mark.parametrize("reply", ["а у вас есть парковка возле клиники?", "расскажите про вашу клинику"])
def test_rules_decide_the_switch_when_the_llm_is_down(reply, d21_world, monkeypatch):  # noqa: F811
    """Сбой LLM — решения нет: предпроверка решает своими правилами, как было (on = off)."""

    for dialog in ([uc.press("menu.appointment.book"), uc.say(reply)], [_TO_CARDIO, uc.say(reply)]):
        off = uc.run(d21_world, monkeypatch, dialog, session=f"off-down-{reply}-{dialog[0].text}")
        on = uc.run(d21_world, monkeypatch, dialog, cfg=ON, session=f"on-down-{reply}-{dialog[0].text}")
        assert [(t["text"], t["handoff"]) for t in on] == [(t["text"], t["handoff"]) for t in off]
        assert not on[1]["telemetry"].get("tu_expects_acted"), on[1]["telemetry"]


# --------------------------------------------------------------------------- тема висящего вопроса в планировщике и подготовке (коммит 3)


@pytest.mark.parametrize(
    ("dialog", "site"),
    [
        ([uc.press("menu.price.test"), uc.say("а запишите меня к терапевту", pending="new_topic", topic="APPOINTMENT")], "plan_pending_override"),
        ([uc.press("menu.appointment.prepare"), uc.say("а запишите меня", pending="new_topic", topic="APPOINTMENT")], "flow_prepare_followup_override"),
    ],
    ids=["price_button_looks_like_a_name", "prepare_button_short_reply"],
)
def test_new_topic_judged_by_llm_is_not_glued_back_by_the_form_of_the_reply(dialog, site, d21_world, monkeypatch):  # noqa: F811
    """«Другая тема» модели на вопрос-слот: тема хода — решение NLU (запись), без переклейки в тему вопроса по форме реплики
    («похоже на название услуги» — цена, «короткий ответ» — подготовка). Off — переклейка, как было."""

    off = uc.run(d21_world, monkeypatch, dialog, session=f"off-{site}")
    on = uc.run(d21_world, monkeypatch, dialog, cfg=ON, session=f"on-{site}")
    assert off[1]["plan"][0] != "APPOINTMENT" and on[1]["plan"][0] == "APPOINTMENT", (off[1]["plan"], on[1]["plan"])
    assert site in on[1]["telemetry"].get("tu_expects_acted", []), on[1]["telemetry"]


def test_answer_judged_by_llm_keeps_the_topic_of_the_question(d21_world, monkeypatch):  # noqa: F811
    """Встречный: ответ на вопрос-слот («гастроскопия» на вопрос подготовки) — тема вопроса, как было (on = off)."""

    dialog = [uc.press("menu.appointment.prepare"), uc.say("гастроскопия", pending="answer", services=("гастроскопия",), topic="PRICE")]
    off = uc.run(d21_world, monkeypatch, dialog, session="off-gastro")
    on = uc.run(d21_world, monkeypatch, dialog, cfg=ON, session="on-gastro")
    assert [(t["text"], t["plan"]) for t in on] == [(t["text"], t["plan"]) for t in off] and on[1]["plan"][0] == "PREPARE"
