"""Одна функция ответа на висящий вопрос — инварианты класса через обработчик эндпоинта (класс 5, `pending_reply = on`).

Дизайн `docs/DESIGN_2026-10-18_one_pending_answer.md` §3, консилиум 18.10 (МЕГА п. 3), журнал
BUG-2026-10-18-PENDING-ANSWER-CLASS5. Мини-прод, оракул LLM нарочно отвечает «да» там, где пациент отказался, —
проверяется, что вето правил внутри функции не даёт исполнить «да»:

1. REFUSAL-AS-YES (BUG-2026-10-07-CONFIRM-REFUSAL-AS-YES): ни на один вопрос «да/нет» отказ или оговорка не исполняются
   как «да» — ни перевода на оффере, ни сущности на «Похоже, вы имели в виду …?», ни стёртой записи на «Отменить?»;
2. CANCEL-ALWAYS-OPERATOR (BUG-2026-10-07-CANCEL-LOW-CONF-CLARIFY): распознанная отмена существующей записи — оператор
   при любом решении и состоянии LLM;
3. URGENT-BEFORE-PENDING (f0c6165): срочное — раньше функции на любом из 10 типов, LLM не зовётся.
"""

from __future__ import annotations

import time

import pytest

from messengers_router.policies import handoff_message
from tests.test_pending_reply import (  # noqa: F401 — mini_prod_world, synonym_fork — фикстуры
    _ABORT_QUESTION,
    _OFFER_QUESTION,
    _PendingOracle,
    _answer,
    _expects,
    _play,
    _prepared,
    mini_prod_world,
    synonym_fork,
)

_CANONICAL = "Женский чекап Базовый"
_CATALOG_QUESTION = f"Похоже, вы имели в виду услугу «{_CANONICAL}» (по запросу «женский чекап»). Это верно? Ответьте «да» или «нет»."
_COLLECTED = {"specialty": "невролог", "branch_name": "пр. Ленина, 5", "city": "Самара"}

# Отказы и оговорки из BUG-2026-10-07 (test_appointment_confirm_refusal.py) — на вопросы «да/нет».
REFUSALS = ["ладно, не надо", "да нет, наверное не надо", "хорошо, я перезвоню", "хорошо, подумаю", "давайте потом", "не уверена"]
# На «Отменить текущий процесс записи?» «отмените» — согласие; отказ здесь — «не отменяйте», оговорка — «подумаю».
ABORT_REFUSALS = ["нет-нет, продолжаем", "не отменяйте, просто уточните адрес", "не надо отменять", "хорошо, подумаю", "давайте потом"]

_OFFER = ({"_operator_offer_pending": True}, _expects(_OFFER_QUESTION))
_CATALOG = (
    {"_catalog_confirm_pending": {"kind": "service", "label": "PRICE", "entity_key": "service_name", "canonical": _CANONICAL, "query": "женский чекап"}},
    _expects(_CATALOG_QUESTION, "catalog_confirm"),
)
_ABORT = ({"appointment_cancel_pending": True, "appointment_flow_active": True, **_COLLECTED}, _expects(_ABORT_QUESTION, "appointment_abort"))


def _one_turn(reply: str, prepared, name: str, world, monkeypatch, oracle):
    entities, expects = prepared
    return _play([reply], name, "on", world, monkeypatch, before=_prepared(entities, expects), oracle=oracle)


@pytest.mark.parametrize("reply", REFUSALS)
def test_refusal_to_the_operator_offer_is_not_a_transfer(reply, mini_prod_world, monkeypatch):  # noqa: F811
    records, *_ = _one_turn(reply, _OFFER, "cls-offer", mini_prod_world, monkeypatch, _PendingOracle(default=_answer("yes")))
    assert not records[0]["handoff"], records[0]["text"]


@pytest.mark.parametrize("reply", REFUSALS)
def test_refusal_to_did_you_mean_does_not_take_the_service(reply, mini_prod_world, monkeypatch):  # noqa: F811
    _records, states, *_ = _one_turn(reply, _CATALOG, "cls-catalog", mini_prod_world, monkeypatch, _PendingOracle(default=_answer("yes")))
    assert states[0].get("service_name") != _CANONICAL, states[0]


@pytest.mark.parametrize("reply", ABORT_REFUSALS)
def test_refusal_to_cancel_keeps_what_was_collected(reply, mini_prod_world, monkeypatch):  # noqa: F811
    records, states, *_ = _one_turn(reply, _ABORT, "cls-abort", mini_prod_world, monkeypatch, _PendingOracle(default=_answer("yes")))
    assert "отменён" not in records[0]["text"], records[0]["text"]
    assert (states[0].get("specialty"), states[0].get("branch_name")) == ("невролог", "пр. Ленина, 5"), states[0]


@pytest.mark.parametrize("reply", ["да, давайте отменим", "да, отмените"])
def test_yes_to_cancel_clears_the_booking(reply, mini_prod_world, monkeypatch):  # noqa: F811
    records, states, *_ = _one_turn(reply, _ABORT, "cls-abort-yes", mini_prod_world, monkeypatch, _PendingOracle(default=_answer("yes")))
    assert "отменён" in records[0]["text"], records[0]["text"]
    assert not states[0].get("specialty"), states[0]


def _action_pending() -> tuple[dict, dict]:
    pending = {"label": "APPOINTMENT", "missing": ["appointment_action"], "ts": time.time(), "ttl": 900}
    return (
        {"_pending": pending, "_pending_label": "APPOINTMENT", "appointment_flow_active": True},
        _expects("Хотите отменить или перенести запись?", "appointment_action", kind="pick", options=("отменить", "перенести")),
    )


@pytest.mark.parametrize("reply", ["отменить", "да, отмените", "отмените, не приду"])
@pytest.mark.parametrize(
    "llm",
    [_answer("no"), _answer("pause"), _answer(picks=(2,)), ConnectionError("ollama лежит"), "не json"],
    ids=["no", "pause", "pick-move", "down", "invalid"],
)
def test_recognised_cancel_always_goes_to_the_operator(reply, llm, mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _PendingOracle(exc=llm) if isinstance(llm, BaseException) else _PendingOracle(default=llm)
    records, *_ = _one_turn(reply, _action_pending(), "cls-cancel", mini_prod_world, monkeypatch, oracle)
    assert records[0]["handoff"], records[0]["text"]
    assert records[0]["text"] == handoff_message("existing_appointment_change"), records[0]["text"]


_PREPARED_BY_TYPE = {
    "operator_offer": _OFFER,
    "catalog_confirm": _CATALOG,
    "appointment_abort": _ABORT,
    "appointment_action": _action_pending(),
    "secondary_offer": ({"_secondary_offer_pending": True, "secondary_intents": ["PREPARE"]}, _expects("Подсказать подготовку?", "secondary_offer")),
    "anesthesia": (
        {"_prepare_variant_pending": {"titles": {"with": "ФКС с наркозом", "without": "ФКС без наркоза"}}},
        _expects("С наркозом или без?", "anesthesia", kind="pick", options=("с наркозом", "без наркоза")),
    ),
    "compound_price": (
        {"_compound_price_pending": {"services": ["УЗДГ сосудов шеи", "ЛПНП"], "default_service": "УЗДГ сосудов шеи"}},
        _expects("Вижу в запросе две услуги", "compound_price", kind="pick", options=("УЗДГ сосудов шеи", "ЛПНП")),
    ),
    "show_all": ({}, _expects('Чтобы показать их, напишите: "все"', "show_all", kind="pick", options=("все",))),
    "booking_service": ({}, _expects("Уточните, какую услугу записать", "booking_service", kind="pick", options=("Массаж спины", "Массаж шеи"))),
    "appointment_branch": (
        {"appointment_branch_options": ["пр. Ленина, 5"], "appointment_flow_active": True},
        _expects("Какой филиал вам удобен?", "appointment_branch", kind="pick", options=("пр. Ленина, 5", "любой филиал")),
    ),
}


@pytest.mark.parametrize("qtype", sorted(_PREPARED_BY_TYPE))
def test_urgent_comes_before_the_answer_to_any_question(qtype, mini_prod_world, monkeypatch):  # noqa: F811
    reply = "нет, мне плохо, задыхаюсь"
    records, _states, rows, oracle = _one_turn(reply, _PREPARED_BY_TYPE[qtype], f"cls-urgent-{qtype}", mini_prod_world, monkeypatch, _PendingOracle(default=_answer("no")))
    assert records[0]["handoff"], records[0]["text"]
    assert oracle.calls == [] and "tu_pending_outcome" not in rows[0]
