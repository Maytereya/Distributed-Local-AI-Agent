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

_ON = {"MR_PENDING_BUTTONS": "on", "MR_CHOOSE_CACHE_TTL_S": 0}
_OFF = {"MR_PENDING_BUTTONS": "off", "MR_CHOOSE_CACHE_TTL_S": 0}  # прод: деплой 10

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
    assert "Номер анализа" in hint["text"] and "чеке" in hint["text"] and "договоре" in hint["text"] and hint["text"].endswith("Перевести на оператора?")
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

    monkeypatch.setattr(cfg, "MR_PENDING_BUTTONS", "off", raising=False)
    pending_answer.reset()
    assert asyncio.run(pending_answer.answer("slot_value", "Укажите номер анализа.", (), "нет")) is None
    monkeypatch.setattr(cfg, "MR_PENDING_BUTTONS", "on", raising=False)
    found = asyncio.run(pending_answer.answer("slot_value", "Укажите номер анализа.", (), "нет"))
    assert found is not None and found.outcome is pending_answer.Outcome.NO and found.source is pending_answer.Source.FAST


# --------------------------------------------------------------------------- п. 2: уточняющий вопрос о филиале
#
# Стенд `pa_branch_which_closer`, `pa_branch_new_topic`: на «Какой филиал вам удобен?» — «а какой ближе к …?», «а сколько
# стоит приём?». Модель: «ничего из этого» (правило 4 промпта — вопрос о вопросе бота); прод печатал тот же список
# филиалов, не ответив. Инвариант: уточняющий вопрос о вопросе записи получает ответ обычным путём, и вопрос о филиале
# звучит снова с прежними пунктами; собранное записи цело при любом ответе; не справка, сбой, off — как было.
# Вопрос о цене («а сколько стоит приём?») — `tests/test_branch_side_price_class.py`: цена услуги записи, не слов реплики
# (BUG-2026-10-20-BRANCH-SIDE-PRICE-FOREIGN; здесь «уролог» в ответе был удачей мира — в прайсе `mini_prod` нет приёма хирурга).

_PENDING_MARK = "понять, что пациент ответил на вопрос"
_BOOK = "хочу записаться к урологу"
_BRANCH_TAIL = "Какой филиал вам удобен? Или написать список врачей по этой услуге?"
_COLLECTED = ("specialty", "appointment_flow_active", "_pending")


def _classifier(label: str) -> str:
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": {}})


def _branch_llm(reply_label: str | None) -> _LLM:
    answers: dict[str, Any] = {_PENDING_MARK: json.dumps({"picks": [], "outcome": "none"})}
    if reply_label is not None:
        answers[_CLASSIFIER_MARK] = lambda prompt: _classifier(reply_label) if _BOOK not in prompt else _classifier("APPOINTMENT")
    return _LLM(answers)


def _collected(state: dict[str, Any]) -> dict[str, Any]:
    pending = dict(state.get("_pending") or {})
    pending.pop("ts", None)
    return {key: (pending if key == "_pending" else state.get(key)) for key in _COLLECTED}


@pytest.mark.parametrize(
    ("reply", "label", "shows"),
    [
        ("а какой ближе к Безымянке?", "ADDRESS", "пр.Кирова"),  # адреса филиалов — ответ справочника, не список записи
        ("а во сколько работает филиал на Победы?", "ADDRESS", "Победы"),
    ],
)
def test_side_question_about_the_branch_question_is_answered_and_asked_again(reply, label, shows, booking_world, monkeypatch):  # noqa: F811
    asked, side, picked = _in_world(booking_world, monkeypatch, [_BOOK, reply, "1"], switches=_ON, llm=_branch_llm(label))
    off = _in_world(booking_world, monkeypatch, [_BOOK, reply], switches=_OFF, llm=_branch_llm(label))[-1]

    assert asked["text"].endswith(_BRANCH_TAIL), asked["text"]
    assert side["label"] == label and "appointment_branch_side_question" in side["flags"], side
    assert shows in side["text"] and side["text"] != asked["text"] and side["text"].endswith("\n\n" + _BRANCH_TAIL), side["text"]
    assert _collected(side["state"]) == _collected(asked["state"])  # собранное записи цело
    assert side["state"][expects.EXPECTS_KEY]["type"] == expects.Question.APPOINTMENT_BRANCH.value
    assert side["state"][expects.EXPECTS_KEY]["options"] == asked["state"][expects.EXPECTS_KEY]["options"]
    first = asked["state"][expects.EXPECTS_KEY]["options"][0]
    assert picked["state"].get("branch_name") == first, picked  # «1» после побочного ответа — первый филиал списка
    assert "appointment_branch_side_question" not in off["flags"]  # как было


@pytest.mark.parametrize("label", ["OTHER", None], ids=["not-a-reference-topic", "classifier-down"])
def test_side_question_without_a_reference_answer_is_as_before(label, booking_world, monkeypatch):  # noqa: F811
    on = _in_world(booking_world, monkeypatch, [_BOOK, "а какой ближе к Безымянке?"], switches=_ON, llm=_branch_llm(label))[-1]
    off = _in_world(booking_world, monkeypatch, [_BOOK, "а какой ближе к Безымянке?"], switches=_OFF, llm=_branch_llm(label))[-1]

    assert on["text"] == off["text"] and "appointment_branch_side_question" not in on["flags"]


def test_side_question_needs_the_model_decision(booking_world, monkeypatch):  # noqa: F811
    llm = _LLM({_CLASSIFIER_MARK: lambda prompt: _classifier("ADDRESS") if _BOOK not in prompt else _classifier("APPOINTMENT")})  # ответ — сбой
    on = _in_world(booking_world, monkeypatch, [_BOOK, "а какой ближе к Безымянке?"], switches=_ON, llm=llm)[-1]
    off = _in_world(booking_world, monkeypatch, [_BOOK, "а какой ближе к Безымянке?"], switches=_OFF, llm=llm)[-1]

    assert on["text"] == off["text"] and "appointment_branch_side_question" not in on["flags"]


# --------------------------------------------------------------------------- п. 3: «обе» на составную цену
#
# Стенд `pa_compound_both`: на «Вижу в запросе две услуги: 1. … 2. …» ответ «обе, пожалуйста» уходил в чужую тему (подбор
# анализов). Инвариант: выбор нескольких показанных пунктов (решение модели, `MANY_TASK`) — цены каждой услуги её обычным
# путём, подряд; один пункт, сбой, off — как было. Пункты — только из показанного списка.

_PAIR = ("ТТГ (TSH) тиреотропный гормон", "Глюкоза")
_PAIR_PRICES = ("400 руб", "190 руб")


def _compound(world_dir, monkeypatch, reply: str, *, switches, llm) -> dict[str, Any]:
    """Бот уже задал вопрос составной цены (триггер через эндпоинт не найден — дизайн 18.10, §6): состояние сессии
    готовится при первом чтении."""

    from messengers_router.memory import MemoryStore
    from tests.test_pending_reply import _expects

    real_get = MemoryStore.get

    def get(self, session_id):
        state = real_get(self, session_id)
        if not state.last_entities.get("_prepared"):
            state.last_entities.update(
                {
                    "_prepared": True,
                    "city": "Самара",
                    "_compound_price_pending": {"services": list(_PAIR), "default_service": _PAIR[0]},
                    expects.EXPECTS_KEY: _expects("Вижу в запросе две услуги: 1. ТТГ 2. Глюкоза", "compound_price", kind="pick", options=_PAIR),
                }
            )
        return state

    monkeypatch.setattr(MemoryStore, "get", get)
    return _in_world(world_dir, monkeypatch, [reply], switches=switches, llm=llm)[-1]


def _picks(*numbers: int) -> _LLM:
    return _LLM({_PENDING_MARK: json.dumps({"picks": list(numbers), "outcome": "none"})})


@pytest.mark.parametrize("reply", ["обе, пожалуйста", "давайте обе", "и то и другое", "оба анализа"])
def test_both_on_a_compound_price_gives_the_price_of_each(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on = _compound(mini_prod_world, monkeypatch, reply, switches=_ON, llm=_picks(1, 2))
    off = _compound(mini_prod_world, monkeypatch, reply, switches=_OFF, llm=_picks(1, 2))

    assert on["label"] == "PRICE" and "compound_price_all" in on["flags"], on
    assert all(price in on["text"] for price in _PAIR_PRICES), on["text"]
    assert on["text"].index(_PAIR_PRICES[0]) < on["text"].index(_PAIR_PRICES[1])  # в порядке списка
    assert on["state"].get("_shown_services") == list(_PAIR) and not on["state"].get("service_name")  # контекст = показанное
    assert "_compound_price_pending" not in on["state"]
    assert not all(price in off["text"] for price in _PAIR_PRICES), off["text"]  # как было


@pytest.mark.parametrize("llm", [_picks(2), _picks(), None], ids=["one-pick", "none", "down"])
def test_one_pick_none_or_failure_on_a_compound_price_is_as_before(llm, mini_prod_world, monkeypatch):  # noqa: F811
    llm = llm if llm is not None else _LLM({})
    on = _compound(mini_prod_world, monkeypatch, "вторую", switches=_ON, llm=llm)
    off = _compound(mini_prod_world, monkeypatch, "вторую", switches=_OFF, llm=llm)

    assert on["text"] == off["text"] and "compound_price_all" not in on["flags"]


def test_many_picks_only_where_the_policy_allows():
    many = [qtype for qtype, policy in pending_answer.POLICY.items() if policy.many]
    schema = choice.answer_schema(pending_answer.MANY_TASK, [choice.Option("1", _PAIR[0]), choice.Option("2", _PAIR[1])])
    system, _user = choice.render(pending_answer.MANY_TASK, "обе", [choice.Option("1", _PAIR[0]), choice.Option("2", _PAIR[1])], asked="Вижу в запросе две услуги")

    assert many == [expects.Question.COMPOUND_PRICE.value]
    assert schema["properties"]["picks"]["maxItems"] == 2
    assert "«обе», «оба», «все» — все показанные варианты" in system


# --------------------------------------------------------------------------- п. 4: второе «нет» на «Похоже, вы имели в виду …?»
#
# Стенд `pa_catalog_no_twice` (кандидат `pending_reply = on`: единственный неверный из 33): после первого «нет» бот просит
# уточнить, но поля ожидания нет — второе «нет» читает предпроверка записи и спрашивает «Отменить текущий процесс записи?».
# Инвариант (политика `catalog_confirm`): второй отказ — оператор; названная в ответе услуга — обычный путь; первый отказ —
# свой вопрос уточнения; сбой LLM, off — как было.

_CANONICAL = "Женский чекап Базовый"
_CATALOG_QUESTION = f"Похоже, вы имели в виду услугу «{_CANONICAL}» (по запросу «женский чекап»). Это верно? Ответьте «да» или «нет»."
_CATALOG_HANDOFF = "Не удалось точно сопоставить запрос с каталогом. Соединяю с оператором."
_CATALOG_OFFER = "Не удалось точно сопоставить запрос с каталогом. Перевести на оператора?"
_ABORT = "Отменить текущий процесс записи?"


def _catalog(world_dir, monkeypatch, turns, *, label: str, switches, llm) -> list[dict[str, Any]]:
    """Бот уже спросил «Похоже, вы имели в виду …?» (состояние готовится при первом чтении сессии)."""

    from messengers_router.memory import MemoryStore
    from tests.test_pending_reply import _expects

    real_get = MemoryStore.get

    def get(self, session_id):
        state = real_get(self, session_id)
        if not state.last_entities.get("_prepared"):
            pending = {"kind": "service", "label": label, "entity_key": "service_name", "canonical": _CANONICAL, "query": "женский чекап"}
            state.last_entities.update({"_prepared": True, "_catalog_confirm_pending": pending, expects.EXPECTS_KEY: _expects(_CATALOG_QUESTION, "catalog_confirm")})
            if label == "APPOINTMENT":
                state.last_entities["appointment_flow_active"] = True
        return state

    monkeypatch.setattr(MemoryStore, "get", get)
    return _in_world(world_dir, monkeypatch, turns, switches=switches, llm=llm)


def _no_llm(outcome: str = "no") -> _LLM:
    return _LLM({_FREE_MARK: _free(outcome), _PENDING_MARK: json.dumps({"picks": [], "outcome": "no"})})


# Решение владельца 19.10 (вопрос 7 итога): после второго отказа не переводить автоматически — спросить; перевести или
# нет, решает пациент (участие операторов со временем сводим к минимуму).
@pytest.mark.parametrize("second", ["нет, не то", "нет", "не то", "нет, это не та услуга"])
@pytest.mark.parametrize("label", ["APPOINTMENT", "PRICE"])
def test_second_refusal_to_did_you_mean_offers_the_operator(second, label, mini_prod_world, monkeypatch):  # noqa: F811
    first, again, agreed = _catalog(mini_prod_world, monkeypatch, ["нет", second, "да"], label=label, switches=_ON, llm=_no_llm())
    off = _catalog(mini_prod_world, monkeypatch, ["нет", second], label=label, switches=_OFF, llm=_no_llm())

    assert first["text"].startswith("Хорошо. Уточните, пожалуйста, точное название") and not first["handoff"], first
    assert first["state"][expects.EXPECTS_KEY]["type"] == expects.Question.SLOT_VALUE.value
    assert again["text"] == _CATALOG_OFFER and not again["handoff"], again  # перевод — вопросом, не сразу
    assert again["state"][expects.EXPECTS_KEY]["type"] == expects.Question.OPERATOR_OFFER.value
    assert _ABORT not in again["text"]
    assert agreed["handoff"] and agreed["text"] == handoff_message("manual_operator"), agreed
    assert off[1]["text"] != _CATALOG_HANDOFF  # как было
    if label == "APPOINTMENT" and second == "нет, не то":
        assert off[1]["text"].startswith(_ABORT), off[1]  # стенд `pa_catalog_no_twice`: петля «Отменить?»


def test_declined_operator_offer_after_the_second_refusal_is_not_a_transfer(mini_prod_world, monkeypatch):  # noqa: F811
    declined = _catalog(mini_prod_world, monkeypatch, ["нет", "нет, не то", "нет"], label="APPOINTMENT", switches=_ON, llm=_no_llm())[-1]

    assert not declined["handoff"] and _ABORT not in declined["text"], declined


def test_service_named_after_the_first_refusal_goes_the_usual_way(mini_prod_world, monkeypatch):  # noqa: F811
    on = _catalog(mini_prod_world, monkeypatch, ["нет", "нет, мне массаж спины"], label="APPOINTMENT", switches=_ON, llm=_no_llm("answer"))[-1]
    off = _catalog(mini_prod_world, monkeypatch, ["нет", "нет, мне массаж спины"], label="APPOINTMENT", switches=_OFF, llm=_no_llm("answer"))[-1]

    assert not on["handoff"] and _ABORT not in on["text"] and on["text"] == off["text"], on


def _down(_prompt: str) -> str:
    raise ConnectionError("тест: LLM недоступна")


@pytest.mark.parametrize("answer", [_down, "не json", _free("new_topic")], ids=["down", "invalid", "new_topic"])
def test_second_turn_without_a_refusal_decision_is_as_before(answer, mini_prod_world, monkeypatch):  # noqa: F811
    answers = {_FREE_MARK: answer, _PENDING_MARK: json.dumps({"picks": [], "outcome": "no"})}
    on = _catalog(mini_prod_world, monkeypatch, ["нет", "нет, не то"], label="APPOINTMENT", switches=_ON, llm=_LLM(answers))[-1]
    off = _catalog(mini_prod_world, monkeypatch, ["нет", "нет, не то"], label="APPOINTMENT", switches=_OFF, llm=_LLM(answers))[-1]

    assert on["text"] == off["text"] and not on["handoff"]
