"""Ответ на вопрос кнопки меню тему у LLM-классификатора не переспрашивает (BUG-2026-10-25-BUTTON-ANSWER-CLASSIFIER,
выключатель `button_answer_topic`).

Журнал хода прода 14–24.10: после нажатия кнопки-темы (`button_menu`, режим `slot`: «Цена анализа» → «Какой анализ
интересует?») следующая реплика ждала LLM-классификатор в 35 % ходов, с 21.10 — в 59 % (p50 3,7 с, до 12 с), и в половине
из них он возвращал ту же тему. Тему ответа пациент уже выбрал нажатием.

Инвариант класса: под `button_answer_topic = on` (поверх `pending_buttons = on`) ответ на вопрос любой кнопки-темы, который
одна функция ответа на висящий вопрос признала ответом, — без вызова классификатора и в теме кнопки, и тогда, когда правило
не сработало, и тогда, когда ветка правила в `topic_by_llm`; off — как было (классификатор решает). Встречные: та же
реплика без кнопки, другая тема по решению функции, сбой LLM функции и медвопрос по правилу — как было; решение судьи темы
у кнопок цены лимит решений хода не тратит. Через обработчик эндпоинта; LLM — подделка настоящего типа, классификатор
отвечает чужой темой (APPOINTMENT), чтобы его вызов был виден.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import button_flow, button_menu, choice, llm_runtime, pending_answer
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_FREE_MARK = "Бот попросил назвать что-то своими словами"
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_FOREIGN = json.dumps({"label": "APPOINTMENT", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
_PICK_NONE = json.dumps({"picks": []})

# Кнопки-темы, ответ на чей вопрос судит одна функция: темы вопросов-слотов и кнопки цены (под выключателем).
_JUDGED = sorted(
    b
    for b, a in button_menu.BUTTONS.items()
    if a.mode == "slot" and not a.entities and a.label in {*pending_answer.SLOT_TOPICS, "PRICE"}
)
# Реплика без своего правила и реплика, на которую срабатывает правило (его ветку кладём в `topic_by_llm`).
_REPLIES = ("для мамы, ей 60 лет", "сколько стоит анализ на ферритин")


def _free(outcome: str) -> str:
    return json.dumps({"picks": [], "outcome": outcome})


def _dialog(world_dir, monkeypatch, turns, *, switch: str, rules: str = "", free: str | None = "answer", cache_ttl: int = 0) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"answer-topic-{switch}-{rules}-{free}-{json.dumps(turns, ensure_ascii=False)}"
    answers: dict[str, Any] = {_CLASSIFIER_MARK: _FOREIGN}
    if free is not None:
        answers = {_FREE_MARK: _free(free), **answers}
    # Прочие места выбора (цена, разбор реплики) — «ничего не подходит»: виден только вызов классификатора.
    answers["Сообщение пациента"] = _PICK_NONE

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
                    "state": dict(memory.get(session).last_entities or {}),
                    "telemetry": world.take_telemetry() or {},
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {
            "MR_NLU_ENGINE": "legacy_v2",
            "MR_PENDING_BUTTONS": "on",
            "MR_CHOOSE_CACHE_TTL_S": cache_ttl,
            "MR_TOPIC_BY_LLM": rules,
            "MR_BUTTON_ANSWER_TOPIC": switch,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


def _branch(world_dir, monkeypatch, button_id: str, reply: str) -> str:
    """Ветки правила, которые ставит реплика (`nlu_rule`), — их и кладём в `topic_by_llm`."""

    tel = _dialog(world_dir, monkeypatch, [_press(button_id), reply], switch="off")[1]["telemetry"]
    return ",".join(part for part in str(tel.get("nlu_rule") or "").split("+") if part and part != "?")


def test_judged_buttons_cover_the_menu():
    """Кнопки-темы без своих сущностей, чей вопрос судит одна функция: подготовка, подбор, результаты, цены."""

    assert {"menu.price.test", "menu.price.doctor", "menu.tests.price", "menu.tests.prepare", "menu.tests.suggest"} <= set(_JUDGED)


@pytest.mark.parametrize("reply", _REPLIES)
@pytest.mark.parametrize("button_id", _JUDGED)
def test_answer_to_button_question_skips_classifier(button_id, reply, mini_prod_world, monkeypatch):  # noqa: F811
    topic = button_menu.BUTTONS[button_id].label
    rules = _branch(mini_prod_world, monkeypatch, button_id, reply) or "rule_test_assist"
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), reply], switch="off", rules=rules)
    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), reply], switch="on", rules=rules)
    assert on[0]["text"] == off[0]["text"], "нажатие отвечает как было"
    assert button_flow.question_button(on[0]["state"].get("_pending")) == button_id
    tel = on[1]["telemetry"]
    if off[1]["telemetry"].get("nlu_llm") is not True:
        pytest.skip(f"{button_id}: «{reply}» и без выключателя решает правило — классификатора нет")
    assert tel.get("nlu_llm") is False and tel.get("nlu_topic_by_llm") == "button_answer", tel
    # Тема NLU — тема кнопки (как если бы её вернул классификатор); переклейки после NLU — прежние (`topic_policy`):
    # «сколько стоит…» в ответ на «Подобрать анализы» по-прежнему уходит в цену.
    assert tel.get("tu_label_nlu") == topic, tel
    if reply == _REPLIES[0]:
        assert on[1]["label"] == topic, (on[1]["label"], tel)


@pytest.mark.parametrize("button_id", _JUDGED)
def test_off_keeps_classifier_like_prod(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLIES[0]], switch="off")
    assert off[1]["telemetry"].get("nlu_llm") is True, off[1]["telemetry"]
    # Метка листа у висящего вопроса — только когда её кто-то читает: на проде её читает `button_price_scope = doctor` (26.10).
    marked = button_flow.QUESTION_KEY in (off[0]["state"].get("_pending") or {})
    assert marked is button_flow.marks_questions(), "off — метка вопроса только для читателей"


def test_typed_question_without_button_keeps_classifier(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: тот же вопрос цены набран руками — висящий вопрос не кнопки, тему решает NLU, как было."""

    on = _dialog(mini_prod_world, monkeypatch, ["сколько стоит анализ", _REPLIES[0]], switch="on")
    assert not button_flow.question_button(on[0]["state"].get("_pending"))
    assert on[1]["telemetry"].get("nlu_topic_by_llm") != "button_answer", on[1]["telemetry"]


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.prepare"])
def test_new_topic_by_the_function_keeps_classifier(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: одна функция решила «другая тема» — тему решает NLU (классификатор), как было."""

    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLIES[0]], switch="on", free="new_topic")
    tel = on[1]["telemetry"]
    assert tel.get("tu_pending_outcome") == "new_topic" and tel.get("nlu_llm") is True, tel


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.suggest"])
def test_judge_failure_keeps_classifier(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: LLM функции недоступна — решения нет, тему решает NLU, как было (без выключателя)."""

    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLIES[0]], switch="on", free=None)
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLIES[0]], switch="off", free=None)
    assert on[1]["telemetry"].get("tu_pending_src") in {"fallback", None}
    assert (on[1]["text"], on[1]["label"]) == (off[1]["text"], off[1]["label"])
    assert on[1]["telemetry"].get("nlu_topic_by_llm") != "button_answer"


@pytest.mark.parametrize("button_id", ["menu.price.test", "menu.tests.prepare"])
def test_medical_advice_rule_still_wins(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """Принцип 4: медвопрос по правилу — шаблоном медвопроса, даже если функция (подделка) назвала реплику ответом."""

    on = _dialog(mini_prod_world, monkeypatch, [_press(button_id), "что со мной, болит голова третий день"], switch="on")
    assert on[1]["telemetry"].get("answer_source") == "safety:medical_advice", on[1]["telemetry"]


def test_price_judge_does_not_spend_turn_decision(mini_prod_world, monkeypatch):  # noqa: F811
    """Решение судьи темы у кнопки цены — шаг NLU вместо классификатора: лимит решений хода не тратит."""

    on = _dialog(mini_prod_world, monkeypatch, [_press("menu.price.test"), _REPLIES[0]], switch="on")
    tel = on[1]["telemetry"]
    assert tel.get("tu_pending_type") == "slot_value" and tel.get("tu_pending_src") == "llm", tel
    assert tel.get("tu_button_refund") == 1, tel
    judged = [c for c in tel.get("choose") or [] if c.get("task") == "pending_free_reply"]
    assert len(judged) == 1 and tel.get("tu_decisions_spent", 0) == len(tel.get("choose") or []) - 1, tel



def test_cached_judge_refunds_nothing(mini_prod_world, monkeypatch):  # noqa: F811
    """Ответ судьи из кэша решения хода не тратил — возвращать нечего (иначе путь цены получил бы лишнее решение)."""

    turns = [_press("menu.price.test"), _REPLIES[0], _press("menu.price.test"), _REPLIES[0]]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", cache_ttl=600)
    assert on[1]["telemetry"].get("tu_button_refund") == 1, on[1]["telemetry"]
    judged = [c for c in on[3]["telemetry"].get("choose") or [] if c.get("task") == "pending_free_reply"]
    assert judged and judged[0].get("cached") is True, on[3]["telemetry"]
    assert "tu_button_refund" not in on[3]["telemetry"], on[3]["telemetry"]


def test_value_found_by_rules_in_foreign_reply_is_judged_by_llm(mini_prod_world, monkeypatch):  # noqa: F811
    """Карточка 25.10 (набор «чужая тема»): правила дозаполнения находят значение слота и в реплике о другом («где сдать общий
    анализ крови?», «ЭКГ как подготовиться?» на «Цену анализа»), и сужение без LLM навязывало тему кнопки 31 чужому вопросу из 32. Под on вопрос кнопки
    судит LLM — она сказала «другая тема», и тему решает NLU. Встречный: без этого (сужение правил) — тема кнопки навязана."""

    turns = [_press("menu.price.test"), "ЭКГ как подготовиться?"]  # правила находят «ЭКГ», ветка подготовки — в `topic_by_llm`
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", rules="rule_prepare", free="new_topic")
    tel = on[1]["telemetry"]
    assert tel.get("tu_pending_src") == "llm" and tel.get("tu_pending_outcome") == "new_topic", tel
    assert tel.get("nlu_topic_by_llm") != "button_answer", tel
    with monkeypatch.context() as patch:
        patch.setattr(button_flow, "judged_by_llm", lambda pending: False)
        rules = _dialog(mini_prod_world, monkeypatch, turns, switch="on", rules="rule_prepare", free="new_topic")[1]["telemetry"]
    assert rules.get("tu_pending_src") == "fast" and rules.get("nlu_topic_by_llm") == "button_answer", rules


def test_slot_judge_does_not_spend_turn_decision(mini_prod_world, monkeypatch):  # noqa: F811
    """Решение судьи темы у вопроса-слота кнопки — тоже шаг NLU вместо классификатора: лимит решений хода не тратит."""

    on = _dialog(mini_prod_world, monkeypatch, [_press("menu.tests.prepare"), "колоноскопия"], switch="on")
    tel = on[1]["telemetry"]
    judged = [c for c in tel.get("choose") or [] if c.get("task") == "pending_free_reply"]
    assert judged and tel.get("tu_pending_outcome") == "answer" and tel.get("tu_button_refund") == 1, tel


def test_rules_value_keeps_its_answer_unless_llm_says_other_topic(mini_prod_world, monkeypatch):  # noqa: F811
    """Карточка 25.10: значение, найденное правилами, LLM проверяет только на «другую тему» — её «не могу назвать» на
    «Результаты» после «Ввести данные» (32 из 89) делало из инструкции оффер оператора. Правила нашли значение и LLM сказала
    «нет» — ответ правил, как без выключателя."""

    turns = [_press("menu.tests.prepare"), "колоноскопия"]
    on = _dialog(mini_prod_world, monkeypatch, turns, switch="on", free="no")
    off = _dialog(mini_prod_world, monkeypatch, turns, switch="off", free="no")
    tel = on[1]["telemetry"]
    assert tel.get("tu_pending_outcome") == "answer" and tel.get("tu_pending_src") == "fast", tel
    assert on[1]["text"] == off[1]["text"]


@pytest.mark.parametrize("button_id", _JUDGED)
def test_slot_mode_keeps_price_buttons_like_before(button_id, mini_prod_world, monkeypatch):  # noqa: F811
    """`slot` (карточка 25.10: на кнопках цены +3 чужие цены выбора строки — блок): тема без классификатора — у вопросов-слотов
    (подготовка, подбор, результаты); вопрос кнопок цены свободного ответа функции не получает, тему решает NLU, как было."""

    topic = button_menu.BUTTONS[button_id].label
    slot = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLIES[0]], switch="slot")
    off = _dialog(mini_prod_world, monkeypatch, [_press(button_id), _REPLIES[0]], switch="off")
    tel = slot[1]["telemetry"]
    if topic == "PRICE":
        assert "tu_pending_type" not in tel and tel.get("nlu_llm") is True, tel
        assert (slot[1]["text"], slot[1]["label"]) == (off[1]["text"], off[1]["label"])
    else:
        assert tel.get("nlu_topic_by_llm") == "button_answer" and tel.get("nlu_llm") is False, tel
