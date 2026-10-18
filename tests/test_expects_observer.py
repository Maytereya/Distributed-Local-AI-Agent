"""Поле ожидания `_expects` — D2.1, наблюдатель: бот знает, какой ответ ждёт на свой вопрос (пункт 5 принципов).

Было: висящий вопрос жил флагами в разных местах состояния, текста вопроса, который увидел пациент, в состоянии не
было — ни клиенту «ответ на висящий вопрос силами LLM» (`pending_reply`, D2.3), ни пределу переспросов (D2.4)
опереться не на что (`~/Dev/handoffs/2026-10-14_pending_reply_design.md`).

Стало: после хода, задавшего вопрос, `state.last_entities["_expects"]` = `{kind, type, question, options, asks}`;
ответ принят, тема сменилась, нажата кнопка, перевод — поля нет; тот же вопрос снова без продвижения — `asks + 1`.

Инвариант класса: поле есть ровно тогда, когда последний ответ бота задал вопрос и обработчик ответа на него взведён
(на начало следующего хода — тоже: истёкший флаг снимает поле до тени `pending_reply`); `question` — текст, который
увидел пациент, без ПДн (ссылки, ФИО из состояния); `options` — слова самого вопроса; ответы и решения те же, что без
поля; поле не доходит ни до инструментов, ни до промптов LLM; телеметрия — ключи и число, без текста.

Ходы — через обработчик эндпоинта (`messenger_generate_once`) в мире стенда на мини-проде (сеть закрыта, LLM гейта
недоступна), диалоги из двух ходов и больше. Запись в мини-проде без врачей не собрать — её состояние задаётся явно,
как в `test_appointment_confirm_refusal.py`.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from evalbench import world as ev_world
from messengers_router import expects
from messengers_router.mess_types import AppointmentPhase, DialogState, ResponseEnvelope, SessionState
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

_OFFER_CHECK = "Проверить мою запись"
_ABORT_TEXT = "Отменить текущий процесс записи? Ответьте «да» или «нет»."


@dataclass
class _Turn:
    text: str
    handoff: bool
    expects: dict[str, Any] | None
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    body: dict[str, Any] = field(default_factory=dict)


def _play(
    turns: list[str | tuple[str, str]],
    world_dir: Path,
    *,
    seed: Callable[[SessionState, Any], None] | None = None,
    services: Any = None,
    session: str = "expects",
) -> list[_Turn]:
    """Диалог через обработчик эндпоинта в одной сессии; после хода — поле ожидания и строки телеметрии хода."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    with ev_world.FrozenWorld(world_dir, "day") as world:

        async def run() -> list[_Turn]:
            memory, svc = MemoryStore(), services or Services()
            if seed is not None:
                seed(memory.get(session), memory)
            out: list[_Turn] = []
            for turn in turns:
                button_id, text = turn if isinstance(turn, tuple) else ("", turn)
                before = len(world.telemetry)
                req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
                body = json.loads((await messenger_generate_once(req, memory=memory, services=svc)).body)
                state = memory.get(session)
                out.append(
                    _Turn(
                        text=body.get("text", ""),
                        handoff=bool(body.get("handoff")),
                        expects=copy.deepcopy(state.last_entities.get(expects.EXPECTS_KEY)),
                        telemetry=list(world.telemetry[before:]),
                        body=body,
                    )
                )
            return out

        result = asyncio.run(run())
        assert list(world.net.attempts) == []
    return result


def _seed_collecting(state: SessionState, memory: Any) -> None:
    """Запись собрана до вопроса о пожелании по времени (вариант A)."""

    state.last_entities.update(
        {
            "city": "Самара",
            "appointment_flow_active": True,
            "specialty": "терапевт",
            "service_name": "Прием терапевта",
            "branch_name": "пр. Ленина, 5",
            "_appointment_wish_asked": True,
        }
    )
    state.dialog = DialogState(label="APPOINTMENT", phase=AppointmentPhase.COLLECTING)
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["appointment_wish"])


def _seed_action(state: SessionState, memory: Any) -> None:
    """Бот спросил «Хотите отменить или перенести запись?» (действие с записью не ясно)."""

    state.last_entities.update({"appointment_action": "unknown"})
    memory.set_pending(state, label="APPOINTMENT", missing_slots=["appointment_action"])


def _expected(kind: str, type_: str, question: str, asks: int, options: list[str] | None = None) -> dict[str, Any]:
    return {"kind": kind, "type": type_, "question": question, "options": options or [], "asks": asks}


# --------------------------------------------------------------------------- модуль


def _asked_turn(state: SessionState, text: str, *calls: tuple[expects.Question, dict[str, Any]], handoff: bool = False) -> dict | None:
    expects.begin_turn(state)
    for question, kwargs in calls:
        expects.ask(question, **kwargs)
    return expects.finish_turn(state, ResponseEnvelope(text=text, handoff=handoff))


def test_question_is_what_the_patient_saw_and_a_long_one_keeps_its_tail():
    short = "Перевести на оператора?"
    assert expects.visible_question(f"  {short}\n") == short
    long = "цена " * 200 + "Перевести на оператора?"
    shown = expects.visible_question(long)
    assert len(shown) <= expects.QUESTION_MAX_CHARS and shown.startswith("…") and shown.endswith("Перевести на оператора?")


def test_asks_counts_the_same_question_in_a_row_and_restarts_on_another():
    state = SessionState(session_id="unit-asks")
    state.last_entities["_operator_offer_pending"] = True
    offer = (expects.Question.OPERATOR_OFFER, {})
    assert _asked_turn(state, "Перевести на оператора?", offer)["asks"] == 1
    assert _asked_turn(state, "Перевести на оператора?", offer)["asks"] == 2  # тот же вопрос снова
    assert _asked_turn(state, "Другой текст. Перевести на оператора?", offer)["asks"] == 1  # другой вопрос
    state.last_entities["_catalog_confirm_pending"] = {"canonical": "X"}
    reask = (expects.Question.CATALOG_CONFIRM, {"reask": True})
    assert _asked_turn(state, "Это верно?", reask)["asks"] == 2  # переспрос первым не бывает
    assert _asked_turn(state, "Это верно?", reask)["asks"] == 3


def test_no_question_handoff_or_disarmed_flag_leaves_no_expectation():
    state = SessionState(session_id="unit-clear", last_entities={"_operator_offer_pending": True})
    offer = (expects.Question.OPERATOR_OFFER, {})
    assert _asked_turn(state, "Перевести на оператора?", offer)
    assert _asked_turn(state, "Цена: 100 руб.") is None and expects.EXPECTS_KEY not in state.last_entities  # вопроса нет
    assert _asked_turn(state, "Перевести на оператора?", offer, handoff=True) is None  # перевод
    state.last_entities.pop("_operator_offer_pending")
    assert _asked_turn(state, "Перевести на оператора?", offer) is None  # флаг снят — читать ответ некому


def test_two_questions_in_a_turn_keep_the_one_answered_first():
    state = SessionState(session_id="unit-order", last_entities={"_operator_offer_pending": True, "_secondary_offer_pending": True})
    got = _asked_turn(state, "Памятки нет. Перевести на оператора?", (expects.Question.OPERATOR_OFFER, {}), (expects.Question.SECONDARY_OFFER, {}))
    assert got["type"] == "operator_offer"  # следующий ход первым разбирает оффер оператора


def test_tools_never_see_the_expectation():
    entities = {"city": "Самара"}
    assert expects.without(entities) is entities
    with_field = {**entities, expects.EXPECTS_KEY: {"type": "operator_offer"}}
    assert expects.without(with_field) == entities


@pytest.mark.parametrize("raw", [None, [], {"type": "x", "kind": "yes_no", "asks": 1}, {"type": "operator_offer", "kind": "yes_no", "asks": 0}, {"type": "operator_offer", "kind": "maybe", "asks": 1}])
def test_malformed_field_is_not_a_previous_question(raw):
    state = SessionState(session_id="unit-read", last_entities={expects.EXPECTS_KEY: raw})
    assert expects.read(state) is None


def test_question_carries_no_personal_data():
    """Ссылка на результат несёт фамилию, год рождения и номер анализа; сводка записи — ФИО пациента."""

    text = "Открыть результат: https://naykalab.ru/getanaliz.php?fam=X&year=1985\n\nЗапись: Иванов Иван. Перевести на оператора?"
    shown = expects.visible_question(text, {"patient_name": "Иванов Иван", "surname": "Иванов"})
    assert shown == f"Открыть результат: {expects.LINK_MASK}\n\nЗапись: {expects.NAME_MASK}. Перевести на оператора?"


_FIELD = {"kind": "yes_no", "type": "appointment_action", "question": "Хотите отменить или перенести запись?", "options": [], "asks": 1}


@pytest.mark.parametrize(
    ("pending", "kept"),
    [
        ({"label": "APPOINTMENT", "missing": ["appointment_action"], "ts": 0.0, "ttl": 900}, False),  # срок истёк
        ({"label": "OTHER", "missing": ["operator_offer_confirm"], "ts": 1e12, "ttl": 900}, False),  # флаг другого вопроса
        ({"label": "APPOINTMENT", "missing": ["appointment_action"], "ts": 1e12, "ttl": 900}, True),
    ],
    ids=["expired", "other_flag", "armed"],
)
def test_stale_field_is_dropped_before_anyone_reads_it(pending, kept):
    """Флаг выбора действия живёт в `_pending` со сроком: истёк между ходами — поле снимается до тени `pending_reply`."""

    state = SessionState(session_id="unit-stale", last_entities={expects.EXPECTS_KEY: dict(_FIELD), "_pending": pending})
    expects.begin_turn(state)
    assert (expects.EXPECTS_KEY in state.last_entities) is kept
    assert (expects.previous() is not None) is kept


def test_flags_without_a_term_are_reconciled_by_the_end_of_the_turn():
    """У прочих флагов срока нет: начало хода поле не трогает (его согласовал конец прошлого хода)."""

    field_ = {"kind": "yes_no", "type": "operator_offer", "question": "Перевести на оператора?", "options": [], "asks": 1}
    state = SessionState(session_id="unit-no-term", last_entities={expects.EXPECTS_KEY: dict(field_)})
    expects.begin_turn(state)
    assert state.last_entities[expects.EXPECTS_KEY] == field_ and expects.previous() == field_


# --------------------------------------------------------------------------- места вопросов через эндпоинт


@pytest.mark.parametrize(("reply", "handoff"), [("нет", False), ("да", True)])
def test_operator_offer_is_the_expectation_and_the_answer_clears_it(reply, handoff, mini_prod_world):  # noqa: F811
    first, second = _play([_OFFER_CHECK, reply], mini_prod_world)
    assert first.expects == _expected("yes_no", "operator_offer", first.text, 1)
    assert first.text.endswith("Перевести на оператора?")
    assert second.handoff is handoff and second.expects is None


def test_another_question_instead_of_an_answer_drops_the_offer(mini_prod_world):  # noqa: F811
    first, second = _play([_OFFER_CHECK, "сколько стоит ферритин"], mini_prod_world)
    assert first.expects["type"] == "operator_offer"
    assert "Ферритин" in second.text
    # Оффер снят обработчиком; у ответа о цене своё (скрытое) ожидание второй темы — с текстом этого ответа.
    assert second.expects is None or (second.expects["type"], second.expects["question"]) == ("secondary_offer", second.text)


def test_appointment_abort_counts_the_same_question_and_the_answer_clears_it(mini_prod_world):  # noqa: F811
    turns = _play(["нет", "да, отмените", "да, отмените же", "нет"], mini_prod_world, seed=_seed_collecting)
    assert [t.text for t in turns[:3]] == [_ABORT_TEXT] * 3  # как было: переспрос без предела (D2.4 — под выключателем)
    assert [t.expects for t in turns[:3]] == [_expected("yes_no", "appointment_abort", _ABORT_TEXT, n) for n in (1, 2, 3)]
    assert turns[3].expects is None and "Когда вам удобно" in turns[3].text  # «не отменять» — вернулись к записи


def test_topic_switch_question_is_an_abort_question_too(mini_prod_world):  # noqa: F811
    first, second = _play(["а где вы находитесь и до скольки работаете?", "да"], mini_prod_world, seed=_seed_collecting)
    assert first.expects == _expected("yes_no", "appointment_abort", first.text, 1)
    assert "Отменить этот процесс и перейти к новому вопросу?" in first.text
    assert second.expects is None


def test_offer_after_repeated_result_link_keeps_no_personal_data(mini_prod_world):  # noqa: F811
    """Анти-залип дописывает оффер к ответу со ссылкой на результат (третий одинаковый ответ): в поле ссылки нет."""

    *_, third = _play(["Иванова 1985 Бг 123", "Иванов 1985 Бг 124", "Иванов 1985 Бг 125"], mini_prod_world)
    assert "getanaliz" in third.text and third.text.endswith("Перевести на оператора?")
    assert third.expects["type"] == "operator_offer"
    assert expects.LINK_MASK in third.expects["question"]
    for secret in ("http", "getanaliz", "Иванов", "1985", "125"):
        assert secret not in third.expects["question"]


def test_during_the_turn_the_field_is_the_question_being_answered(mini_prod_world, monkeypatch):  # noqa: F811
    """Тень `pending_reply` читает поле до предпроверки записи: там — вопрос прошлого хода, не устаревший."""

    from messengers_router import router as router_mod

    seen: list[Any] = []
    real = router_mod.run_appointment_precheck

    def spy(**kwargs):
        seen.append(copy.deepcopy(kwargs["state"].last_entities.get(expects.EXPECTS_KEY)))
        return real(**kwargs)

    monkeypatch.setattr(router_mod, "run_appointment_precheck", spy)
    first, second, third = _play([_OFFER_CHECK, "нет", "сколько стоит ферритин"], mini_prod_world)
    assert seen == [None, first.expects, None]  # ответ на оффер принят в прошлом ходе — поля уже нет


def test_appointment_action_is_a_pick_of_the_words_of_the_question(mini_prod_world):  # noqa: F811
    first, second = _play(["не знаю", "перенести"], mini_prod_world, seed=_seed_action)
    assert first.text == "Хотите отменить или перенести запись?"
    assert first.expects == _expected("pick", "appointment_action", first.text, 2, list(expects.APPOINTMENT_ACTION_OPTIONS))
    assert all(option in first.text for option in first.expects["options"])
    assert second.expects is None or second.expects["type"] != "appointment_action"


def _fake_nlu(monkeypatch, decision_factory) -> None:
    """Решение NLU задано тестом; каталоги «здоровы» (в мини-проде нет врачей — запись упёрлась бы в «каталог
    врачей недоступен» раньше вопроса)."""

    from messengers_router import router as router_mod
    from messengers_router.nlu_pipeline import NLUResult

    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        return NLUResult(decision=decision_factory(_text), candidates=[], merged_from="rule")

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default))
    monkeypatch.setattr(router_mod, "_catalog_health_requirements", lambda *_args, **_kwargs: (False, False))


def test_appointment_action_question_from_the_pipeline_is_asked_once(mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.mess_types import RouteDecision

    _fake_nlu(monkeypatch, lambda _t: RouteDecision(label="APPOINTMENT", confidence=0.9, entities={"appointment_action": "unknown"}, flags={"rule_appointment"}))
    first, second = _play(["что-то с визитом к врачу", "не знаю"], mini_prod_world)
    assert first.text == "Хотите отменить или перенести запись?"
    assert first.expects == _expected("pick", "appointment_action", first.text, 1, list(expects.APPOINTMENT_ACTION_OPTIONS))
    assert second.text == first.text and second.expects["asks"] == 2


def _catalog_doctor(monkeypatch) -> None:
    from messengers_router.mess_types import RouteDecision
    from messengers_router.services import Services

    _fake_nlu(monkeypatch, lambda _t: RouteDecision(label="APPOINTMENT", confidence=0.74, flags={"rule_appointment", "doctor_name_unverified"}))

    async def fuzzy_doctor(self, raw_text_or_name: str):
        return {"status": "fuzzy", "query": "евграфова", "canonical": "Евграфова"}

    async def no_service(self, raw_text_or_name: str, *, current_service_name: str = "", context_text: str = ""):
        return {"status": "miss", "query": "", "canonical": ""}

    monkeypatch.setattr(Services, "match_catalog_doctor", fuzzy_doctor)
    monkeypatch.setattr(Services, "match_catalog_service", no_service)


def test_catalog_confirm_question_reask_and_refusal(mini_prod_world, monkeypatch):  # noqa: F811
    _catalog_doctor(monkeypatch)
    first, second, third = _play(["Запишите к Евграфовой", "именно", "нет"], mini_prod_world)
    assert "Похоже, вы имели в виду врача «Евграфова»" in first.text
    assert first.expects == _expected("yes_no", "catalog_confirm", first.text, 1)
    assert second.text == first.text and second.expects == _expected("yes_no", "catalog_confirm", first.text, 2)
    assert third.expects is None  # «нет» принят: дальше — вопрос о фамилии, не «да/нет»


@pytest.fixture
def anesthesia_services(monkeypatch):
    from tests.test_prepare_kb_memo_variants import _KB, _WITHOUT
    from messengers_router.services import Services
    from messengers_router.services import prepare as prepare_mod

    async def select(_question, memos, **_kwargs):
        return (_WITHOUT,)

    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: _KB)
    monkeypatch.setattr(prepare_mod, "select_prepare_memos", select)
    return Services()


def test_anesthesia_question_is_a_pick_and_the_reask_counts(anesthesia_services, mini_prod_world):  # noqa: F811
    from tests.test_prepare_kb_memo_variants import _WITH, _WITHOUT

    first, second, third = _play(["как подготовиться к колоноскопии", "да", "нет"], mini_prod_world, services=anesthesia_services)
    assert first.expects == _expected("pick", "anesthesia", first.text, 1, list(expects.ANESTHESIA_OPTIONS))
    assert all(f"«{option}»" in first.text for option in expects.ANESTHESIA_OPTIONS)
    assert second.expects == _expected("pick", "anesthesia", second.text, 2, list(expects.ANESTHESIA_OPTIONS))
    assert _WITH.text in third.text and _WITHOUT.text in third.text and third.expects is None


_COMPOUND = {
    "service_name": "УЗДГ сосудов шеи",
    "clarify_text": "Вижу в запросе две услуги:\n1. УЗДГ сосудов шеи\n2. ЛПНП\n\nЕсли хотите, сначала покажу по УЗДГ сосудов шеи.",
    "compound_price_services": ["УЗДГ сосудов шеи", "ЛПНП"],
    "compound_price_default_service": "УЗДГ сосудов шеи",
    "service_kind": "compound_clarify",
}


def test_compound_price_question_is_a_pick_of_the_listed_services(mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router import router as router_mod
    from messengers_router.mess_types import Evidence, RouteDecision

    _fake_nlu(monkeypatch, lambda _t: RouteDecision(label="PRICE", confidence=0.9, entities={"service_name": "УЗДГ сосудов шеи"}, flags={"rule_price"}))
    answers = iter([{"service_bundle": dict(_COMPOUND)}, {"service_bundle": {"service_name": "УЗДГ сосудов шеи", "retail_prices": [{"serviceName": "УЗДГ сосудов шеи", "price": 1500}]}}])

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence(items=next(answers))

    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    first, second = _play(["сколько стоит УЗДГ шеи и ЛПНП", "можно"], mini_prod_world)
    assert first.expects == _expected("pick", "compound_price", first.text, 1, ["УЗДГ сосудов шеи", "ЛПНП"])
    assert all(option in first.text for option in first.expects["options"])
    assert second.expects is None


def test_secondary_offer_is_recorded_though_the_patient_does_not_see_it(mini_prod_world):  # noqa: F811
    """Вторичный оффер ждёт «да» молча (текст оффера не выводится с fec9770): в поле — ответ, который видел пациент."""

    first, second = _play(["сколько стоит ферритин", "а как подготовиться?"], mini_prod_world)
    assert first.expects == _expected("yes_no", "secondary_offer", first.text, 1)
    assert "Также" not in first.text  # вопроса о второй теме пациент не видел
    # Второй ход задал оффер оператора поверх скрытого вторичного: в поле — тот, что следующий ход разберёт первым.
    assert second.text.endswith("Перевести на оператора?")
    assert second.expects == _expected("yes_no", "operator_offer", second.text, 1)


def test_button_press_drops_the_question(mini_prod_world):  # noqa: F811
    button = ("menu.price.test", GATEWAY_V1_TITLES["menu.price.test"])
    first, second = _play([_OFFER_CHECK, button], mini_prod_world)
    assert first.expects is not None
    assert second.expects is None and not second.handoff  # подпись кнопки — не ответ на вопрос


def test_new_question_after_a_button_starts_its_count_again(mini_prod_world):  # noqa: F811
    button = ("menu.appointment.change", GATEWAY_V1_TITLES["menu.appointment.change"])
    turns = _play(["нет", "да, отмените", button], mini_prod_world, seed=_seed_collecting)
    assert turns[1].expects["asks"] == 2
    assert turns[2].expects is None or turns[2].expects["asks"] == 1


# --------------------------------------------------------------------------- наблюдатель невидим


def test_telemetry_has_the_type_kind_and_count_without_text(mini_prod_world):  # noqa: F811
    turns = _play(["нет", "да, отмените"], mini_prod_world, seed=_seed_collecting)
    rows = [row for turn in turns for row in turn.telemetry]
    assert [(r.get("tu_expects_type"), r.get("tu_expects_kind"), r.get("tu_expects_asks")) for r in rows] == [
        ("appointment_abort", "yes_no", 1),
        ("appointment_abort", "yes_no", 2),
    ]
    dumped = json.dumps(rows, ensure_ascii=False)
    assert "отмените" not in dumped and "Отменить текущий процесс" not in dumped


def test_expectation_never_reaches_tools_or_llm_prompts(mini_prod_world, monkeypatch):  # noqa: F811
    """Планировщик отдаёт инструментам всё состояние, их `entities_used` — в промпт свободного ответа LLM."""

    from messengers_router import llm_runtime

    prompts: list[str] = []

    class _Recorder:
        async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
            prompts.append(f"{system or ''}\n{prompt}")
            raise ConnectionError("тест: LLM недоступна")

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Recorder())
    first, second = _play([_OFFER_CHECK, "какие анализы сдать для профилактики"], mini_prod_world)
    assert first.expects is not None
    items = second.body["state_update"]["debug"]["evidence"]["items"]
    used = [payload["entities_used"] for payload in items.values() if isinstance(payload, dict) and isinstance(payload.get("entities_used"), dict)]
    assert used, "ход ушёл в инструмент с сущностями"
    assert all(expects.EXPECTS_KEY not in entities for entities in used)
    assert prompts and all(expects.EXPECTS_KEY not in prompt for prompt in prompts)


_DIALOGS: dict[str, tuple[list[Any], Callable | None]] = {
    "offer_no": ([_OFFER_CHECK, "нет"], None),
    "offer_then_question": ([_OFFER_CHECK, "какие анализы сдать для профилактики"], None),
    "abort_loop": (["нет", "да, отмените", "да, отмените же", "нет"], _seed_collecting),
    "action": (["не знаю", "перенести"], _seed_action),
    "price_then_prepare": (["сколько стоит ферритин", "а как подготовиться?", "да"], None),
}


@pytest.mark.parametrize("name", sorted(_DIALOGS))
def test_answers_are_the_same_without_the_observer(name, mini_prod_world, monkeypatch):  # noqa: F811
    turns, seed = _DIALOGS[name]
    observed = _play(turns, mini_prod_world, seed=seed, session=f"obs-{name}")
    with monkeypatch.context() as off:
        off.setattr(expects, "ask", lambda *_a, **_kw: None)
        off.setattr(expects, "finish_turn", lambda state, _response: state.last_entities.pop(expects.EXPECTS_KEY, None))
        plain = _play(turns, mini_prod_world, seed=seed, session=f"plain-{name}")
    assert [(t.text, t.handoff) for t in observed] == [(t.text, t.handoff) for t in plain]
    assert all(t.expects is None for t in plain)
