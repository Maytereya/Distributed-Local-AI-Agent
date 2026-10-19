"""Вопросы-слоты кнопок и остатки класса 5 — через одну функцию ответа на висящий вопрос (`pending_buttons`, облако 19.10).

Решение владельца 19.10, п. 5: вопросы кнопок не должны быть вторым путём рядом с `pending_answer`. Выключатель
`[MESSENGER_ROUTER] pending_buttons = off | on` (по умолчанию off) действует поверх `pending_reply = on` (прод, деплой 10);
под off — ответы прежние. Тесты — через обработчик эндпоинта, LLM — подделка настоящего типа (`llm_runtime._OLLAMA_CLIENT`),
отвечает по метке промпта места вызова, прочие места — «недоступна», как в мире гейта.

П. 1 — вопрос-слот со свободным ответом (`expects.Question.SLOT_VALUE`): BUG-2026-10-08-PREPARE-ANESTHESIA-IN-QUERY,
BUG-2026-10-08-SLOT-REPLY-LOST-TO-LLM. Инвариант: ответ на вопрос-слот, который одна функция признала ответом, разбирается в
теме вопроса (модификатор процедуры не заменяет её и сразу выбирает вариант памятки); значение, которое разобрали правила,
LLM не зовёт; «не помню номер анализа» на реквизиты результата — подсказка, где их найти, и оффер оператора вопросом
(результаты по ФИО не ищем, R-01); другая тема, сбой, off — как было; срочное — раньше функции.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, expects, llm_runtime, pending_answer
from messengers_router.policies import handoff_message
from messengers_router.router import TEST_RESULT_DATA_HINT_TEXT
from tests.test_booking_slots_by_meaning import _LLM, booking_world  # noqa: F401 — booking_world — фикстура
from tests.test_price_turns_golden import mini_prod_world, synonym_fork  # noqa: F401 — фикстуры

# Метки промптов мест вызова: свободный ответ — раньше общего (у них общее начало инструкции).
_FREE_MARK = "Бот попросил назвать что-то своими словами"
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_CLASSIFIER_NEW_TOPIC = json.dumps({"label": "APPOINTMENT", "confidence": 0.9, "context_action": "new_topic", "entities": {}})

_ON = {"MR_PENDING_REPLY": "on", "MR_PENDING_BUTTONS": "on", "MR_CHOOSE_CACHE_TTL_S": 0}
_OFF = {"MR_PENDING_REPLY": "on", "MR_PENDING_BUTTONS": "off", "MR_CHOOSE_CACHE_TTL_S": 0}  # прод: деплой 10

_PREPARE_BUTTON = ("menu.appointment.prepare", "Подготовка к процедуре")
_SUGGEST_BUTTON = ("menu.tests.suggest", "Подобрать анализы")
_RESULTS_BUTTON = ("menu.results.lookup", "Ввести данные")
_PREPARE_REASK = "К какому анализу или исследованию нужна подготовка?"


def _free(outcome: str) -> str:
    return json.dumps({"picks": [], "outcome": outcome})


def _dialog(services, monkeypatch, turns, *, switches: dict[str, Any], llm: _LLM) -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта: ход — реплика или (кнопка, подпись); запись хода — ответ, метка, флаги, перевод
    и состояние сессии после хода."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg

    session = f"buttons-{json.dumps(switches)}-{json.dumps(turns, ensure_ascii=False)}"

    async def run() -> list[dict[str, Any]]:
        memory, out = MemoryStore(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            dbg = (body.get("state_update") or {}).get("debug") or {}
            decision = dbg.get("decision") or {}
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "label": decision.get("label"),
                    "flags": decision.get("flags") or [],
                    "handoff": bool(body.get("handoff")),
                    "state": dict(memory.get(session).last_entities or {}),
                }
            )
        return out

    with monkeypatch.context() as patch:
        for key, value in switches.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        choice._CACHE.clear()
        return asyncio.run(run())


def _in_world(world_dir, monkeypatch, turns, *, switches, llm) -> list[dict[str, Any]]:
    from messengers_router.services import Services

    with ev_world.FrozenWorld(world_dir, "day"):
        return _dialog(Services(), monkeypatch, turns, switches=switches, llm=llm)


@pytest.fixture
def memo_services(monkeypatch):
    """Памятки ФКС с наркозом и без (как в `test_prepare_kb_memo_variants`): ответ — текст памятки клиники как есть."""

    from tests.test_prepare_kb_memo_variants import services as memo_fixture

    return memo_fixture.__wrapped__(monkeypatch)


# --------------------------------------------------------------------------- п. 1: вопрос-слот — свободный ответ


@pytest.mark.parametrize("button", [_PREPARE_BUTTON, _SUGGEST_BUTTON, _RESULTS_BUTTON], ids=["prepare", "suggest", "results"])
def test_slot_question_of_a_button_is_a_free_answer_expectation(button, mini_prod_world, monkeypatch):  # noqa: F811
    on = _in_world(mini_prod_world, monkeypatch, [button], switches=_ON, llm=_LLM({}))[-1]
    off = _in_world(mini_prod_world, monkeypatch, [button], switches=_OFF, llm=_LLM({}))[-1]

    assert on["text"] == off["text"]
    assert on["state"][expects.EXPECTS_KEY]["type"] == expects.Question.SLOT_VALUE.value, on["state"].get(expects.EXPECTS_KEY)
    assert on["state"][expects.EXPECTS_KEY]["kind"] == expects.FREE
    assert expects.EXPECTS_KEY not in off["state"]  # как было: вопрос кнопки поля ожидания не ставил


@pytest.mark.parametrize("reply", ["колоноскопия с наркозом", "колоноскопия во сне", "колоноскопию под наркозом"])
def test_procedure_with_modifier_gets_the_memo_variant_at_once(reply, memo_services, monkeypatch):
    from tests.test_prepare_kb_memo_variants import _WITH, _WITHOUT

    llm = _LLM({_FREE_MARK: _free("answer")})
    on = _dialog(memo_services, monkeypatch, [_PREPARE_BUTTON, reply], switches=_ON, llm=llm)[-1]
    off = _dialog(memo_services, monkeypatch, [_PREPARE_BUTTON, reply], switches=_OFF, llm=_LLM({}))[-1]

    assert off["text"].startswith(_PREPARE_REASK), off  # как было: переспрос
    assert llm.calls == [_FREE_MARK]
    assert on["label"] == "PREPARE" and "pending_slot_value" in on["flags"], on
    assert _WITH.text in on["text"] and _WITHOUT.text not in on["text"], on["text"]


def test_name_the_rules_understand_does_not_call_the_llm(memo_services, monkeypatch):
    llm = _LLM({_FREE_MARK: _free("answer")})
    on = _dialog(memo_services, monkeypatch, [_PREPARE_BUTTON, "колоноскопия"], switches=_ON, llm=llm)[-1]
    off = _dialog(memo_services, monkeypatch, [_PREPARE_BUTTON, "колоноскопия"], switches=_OFF, llm=_LLM({}))[-1]

    assert llm.calls == [] and on["text"] == off["text"]  # прежний вопрос «с наркозом или без?»


@pytest.mark.parametrize("answer", [_free("new_topic"), _free("none"), _free("pause"), "не json", None], ids=["new_topic", "none", "pause", "invalid", "down"])
def test_other_topic_none_or_failure_is_as_before(answer, memo_services, monkeypatch):
    answers = {_FREE_MARK: answer} if answer is not None else {}
    on = _dialog(memo_services, monkeypatch, [_PREPARE_BUTTON, "колоноскопия с наркозом"], switches=_ON, llm=_LLM(answers))[-1]
    off = _dialog(memo_services, monkeypatch, [_PREPARE_BUTTON, "колоноскопия с наркозом"], switches=_OFF, llm=_LLM({}))[-1]

    assert on["text"] == off["text"]


def test_example_from_the_button_hint_stays_in_the_button_topic(booking_world, monkeypatch):  # noqa: F811
    llm = _LLM({_FREE_MARK: _free("answer"), _CLASSIFIER_MARK: _CLASSIFIER_NEW_TOPIC})
    on = _in_world(booking_world, monkeypatch, [_SUGGEST_BUTTON, "проверить щитовидку"], switches=_ON, llm=llm)[-1]
    off = _in_world(booking_world, monkeypatch, [_SUGGEST_BUTTON, "проверить щитовидку"], switches=_OFF, llm=_LLM({_CLASSIFIER_MARK: _CLASSIFIER_NEW_TOPIC}))[-1]

    assert off["label"] == "APPOINTMENT" and "к какому врачу" in off["text"].lower(), off  # как было: чужая тема
    assert on["label"] == "TEST_ASSIST" and "pending_slot_value" in on["flags"], on
    assert "к какому врачу" not in on["text"].lower(), on["text"]


@pytest.mark.parametrize("button", [_PREPARE_BUTTON, _RESULTS_BUTTON], ids=["prepare", "results"])
def test_urgent_reply_comes_before_the_one_function(button, memo_services, monkeypatch):
    llm = _LLM({_FREE_MARK: _free("answer")})
    turn = _dialog(memo_services, monkeypatch, [button, "у меня сильная боль в груди и не могу дышать"], switches=_ON, llm=llm)[-1]

    assert turn["label"] == "URGENT" and turn["handoff"] and _FREE_MARK not in llm.calls, turn


# «Ввести данные» → «не помню номер анализа» (BUG-2026-10-08-SLOT-REPLY-LOST-TO-LLM, второй пример; решение владельца 19.10,
# п. 4): прод — «Для какой цели хотите подобрать анализы?». Класс — «не могу назвать реквизиты» любыми словами.
_CANT_NAME = ["не помню номер анализа", "не знаю код анализа", "а где взять номер? не помню", "чек потеряла, номера нет"]


@pytest.mark.parametrize("reply", _CANT_NAME)
def test_cant_name_the_result_data_gets_where_to_find_it_and_an_operator_offer(reply, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM({_FREE_MARK: _free("no")})
    on = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, reply, "да"], switches=_ON, llm=llm)
    off = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, reply], switches=_OFF, llm=_LLM({}))[-1]

    assert off["text"] != TEST_RESULT_DATA_HINT_TEXT  # как было: подсказки не было
    hint, agreed = on[1], on[2]
    assert hint["text"] == TEST_RESULT_DATA_HINT_TEXT and not hint["handoff"], hint
    assert "чеке" in hint["text"] and "договоре" in hint["text"] and hint["text"].endswith("Перевести на оператора?")
    assert hint["state"][expects.EXPECTS_KEY]["type"] == expects.Question.OPERATOR_OFFER.value  # оффер — вопросом
    assert agreed["handoff"] and agreed["text"] == handoff_message("manual_operator"), agreed


def test_result_data_the_rules_understand_does_not_call_the_llm(mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM({_FREE_MARK: _free("no")})
    on = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, "Иванов, 1989, Бг, 123"], switches=_ON, llm=llm)[-1]
    off = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, "Иванов, 1989, Бг, 123"], switches=_OFF, llm=_LLM({}))[-1]

    assert llm.calls == [] and on["text"] == off["text"], on


def test_a_bare_no_to_the_result_data_needs_no_llm(mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM({_FREE_MARK: _free("answer")})
    turn = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, "нет"], switches=_ON, llm=llm)[-1]

    assert llm.calls == [] and turn["text"] == TEST_RESULT_DATA_HINT_TEXT, turn


@pytest.mark.parametrize("answer", [_free("new_topic"), None], ids=["new_topic", "down"])
def test_result_data_other_topic_or_failure_is_as_before(answer, mini_prod_world, monkeypatch):  # noqa: F811
    answers = {_FREE_MARK: answer} if answer is not None else {}
    on = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, "не помню номер анализа"], switches=_ON, llm=_LLM(answers))[-1]
    off = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, "не помню номер анализа"], switches=_OFF, llm=_LLM({}))[-1]

    assert on["text"] == off["text"]


def test_explicit_operator_request_in_a_free_answer_is_a_transfer(mini_prod_world, monkeypatch):  # noqa: F811
    llm = _LLM({_FREE_MARK: _free("no")})
    turn = _in_world(mini_prod_world, monkeypatch, [_RESULTS_BUTTON, "не помню номер, соедините с оператором"], switches=_ON, llm=llm)[-1]

    assert turn["handoff"] and turn["text"] == handoff_message("manual_operator"), turn


def test_free_answer_prompt_renders_the_bot_question_without_options():
    system, user = choice.render(pending_answer.FREE_TASK, "не помню номер анализа", [], asked="Укажите номер анализа.")

    assert _FREE_MARK in system and "«не помню номер анализа»" in system
    assert "Вопрос бота: «Укажите номер анализа.»" in user and "Ответ пациента: «не помню номер анализа»" in user
    assert pending_answer.FREE_TASK.outcomes == ("answer", "no", "pause", "new_topic")


def test_slot_value_is_decided_only_under_pending_buttons(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PENDING_REPLY", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_PENDING_BUTTONS", "off", raising=False)
    pending_answer.reset()
    assert asyncio.run(pending_answer.answer("slot_value", "Укажите номер анализа.", (), "нет")) is None
    monkeypatch.setattr(cfg, "MR_PENDING_BUTTONS", "on", raising=False)
    found = asyncio.run(pending_answer.answer("slot_value", "Укажите номер анализа.", (), "нет"))
    assert found is not None and found.outcome is pending_answer.Outcome.NO and found.source is pending_answer.Source.FAST
