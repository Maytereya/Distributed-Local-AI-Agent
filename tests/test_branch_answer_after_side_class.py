"""Ответ на вопрос о филиале после побочного ответа — только из предложенных филиалов (BUG-2026-10-20-BRANCH-AFTER-SIDE-FOREIGN).

Смоук прода 20.10 (деплой 11, `b6b168f`, `pending_buttons = on`): у уролога два филиала (пр.Ленина, 5 и ул.Ново-Садовая,
106, корпус 82). «хочу записаться к урологу» → список, «Какой филиал вам удобен?» → «а во сколько работает филиал на
Ленина?» (или «а сколько стоит приём?») → побочный ответ и тот же вопрос → «давайте на Победы» → «Да, можем записать на
приём к урологу (пр.Ленина, 5)…» — молча другой филиал; «на Аминева» → «…(любой филиал)…». Без побочного вопроса бот
переспрашивал список.

Корень: побочный ответ ставил вопрос о филиале заново, и в поле ожидания вместо вопроса со списком записи попадал текст
побочного ответа — в ответе о часах это свой нумерованный список всех филиалов города («1. пр.Ленина, 5 2. ул. Победы, 83
…»), а пункты оставались от записи («1. пр.Ленина 2. ул.Ново-Садовая 3. список врачей 4. любой филиал»). Модель следующего
хода видит вопрос бота и пункты и берёт номер из того списка, который видит в вопросе, — он указывает на чужой пункт.

Инвариант класса: ответ на вопрос о филиале, называющий филиал, которого нет среди предложенных, — после любого побочного
ответа — не записывает в другой филиал и не превращается в «любой филиал»: бот отвечает, как без побочного вопроса
(снова список). «1», «первый», «на Ленина», «на Ново-Садовую» — тот филиал из списка записи; «любой, без разницы» —
«любой филиал» (решение 18.10). Через обработчик эндпоинта; LLM — подделка настоящего типа: одна функция отвечает, как
модель, которая сверяет слова ответа сначала со списком в вопросе бота, затем с пунктами, а без списка в вопросе
угадывает первый пункт (так ошибалась прод-LLM: после часов — номер из списка города, после цены — первый филиал).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evalbench import world as ev_world
from messengers_router import expects
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_branch_side_price_class import _PRICE_MARK, side_price_source
from tests.test_pending_buttons_class import _CLASSIFIER_MARK, _OFF, _ON, _PENDING_MARK, _classifier, _in_world

_TWO_BRANCHES = ("пр.Ленина, 5", "ул.Ново-Садовая, 106, корпус 82")
_STEMS = ("уролог", "гинеколог", "невролог")
_BOOKINGS = {"уролог": "хочу записаться к урологу", "гинеколог": "хочу записаться к гинекологу", "невролог": "хочу записаться к неврологу"}
# Побочные вопросы о вопросе о филиале и тема их ответа (её решает классификатор).
_SIDES = {
    "price": ("а сколько стоит приём?", "PRICE"),
    "hours": ("а во сколько работает филиал на Ленина?", "ADDRESS"),
    "address": ("а где находятся ваши филиалы?", "ADDRESS"),
}
# Филиал, которого нет среди предложенных: есть в городе и в ответе о часах (у ЛОР-врачей), нет вовсе, место периметра
# со своим филиалом (Южный город — карточка деплоя 12, BUG-2026-10-20-BRANCH-PERIMETER-UNOFFERED).
_NOT_OFFERED = ("давайте на Победы", "на Аминева", "давайте в Южном городе, на Николаевском")
_PICKS = {"1": 0, "первый": 0, "на Ленина": 0, "давайте на Ново-Садовую": 1}
_ANY = "любой, без разницы"


@pytest.fixture(scope="module")
def two_branch_world(tmp_path_factory) -> Path:
    """Мир `side_price_world` (приём своей специальности и цены врачей), где у уролога, гинеколога и невролога ровно два
    филиала — пр.Ленина и ул.Ново-Садовая, 106; ул. Победы, 83 в городе есть (ЛОР-врачи)."""

    def two_branches(doctor: dict) -> dict:
        units = " ".join(str(unit) for unit in doctor.get("units") or ()).lower()
        if not any(stem in units for stem in _STEMS):
            return doctor
        kept = [(name, rid) for name, rid in zip(doctor["regions"], doctor["region_ids"]) if name in _TWO_BRANCHES]
        kept = kept or [(_TWO_BRANCHES[0], 8882)]
        return {**doctor, "regions": [name for name, _ in kept], "region_ids": [rid for _, rid in kept]}

    src = side_price_source(tmp_path_factory.mktemp("two_branch_src"), doctor=two_branches)
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("two_branch") / "world")


def _field(prompt: str, start: str, end: str) -> str:
    at = prompt.rfind(start)
    return prompt[at + len(start) : prompt.find(end, at + len(start))] if at >= 0 else ""


def _numbered(text: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2)) for m in re.finditer(r"(\d+)\.\s+(.+?)(?=\s+\d+\.\s|\n|$)", text)]


def _model_reads_the_question(prompt: str) -> str:
    """Одна функция как модель: вопрос пациента о вопросе — «ничего из этого»; номер, «первый», «любой» — пункт; адрес —
    номер первого совпадения в списке вопроса бота, затем в пунктах; иначе — «ничего из этого»."""

    asked = _field(prompt, "Вопрос бота: «", "»\n")
    reply = _field(prompt, "Ответ пациента: «", "»").lower()
    options = _numbered(prompt.rsplit("(номер. вариант):", 1)[-1])
    none = json.dumps({"picks": [], "outcome": "none"})
    if "?" in reply:
        return none
    if "первый" in reply:
        return json.dumps({"picks": [1], "outcome": "none"})
    if "любой" in reply:
        return json.dumps({"picks": [n for n, text in options if "любой" in text][:1], "outcome": "none"})
    words = [word[:5] for word in re.findall(r"[а-яё\-]{4,}", reply) if word != "давайте"]
    for listed in (_numbered(asked), options):
        for number, text in listed:
            if any(word in text.lower() for word in words):
                return json.dumps({"picks": [number], "outcome": "none"})
    if words and not any(text in asked for _, text in options if text not in (expects.DOCTOR_LIST_OPTION, expects.ANY_BRANCH_OPTION)):
        # В вопросе бота нет списка, к которому относятся пункты, — модель угадывает (прод: после цены — первый филиал).
        return json.dumps({"picks": [1], "outcome": "none"})
    return none


def _llm(side: str, label: str) -> _LLM:
    answers = (*_NOT_OFFERED, *_PICKS, _ANY)

    def classify(prompt: str) -> str:
        return _classifier(label) if side in prompt and not any(answer in prompt for answer in answers) else _classifier("APPOINTMENT")

    return _LLM({_PENDING_MARK: _model_reads_the_question, _CLASSIFIER_MARK: classify, _PRICE_MARK: json.dumps({"picks": [1]})})


def _after_side(world, monkeypatch, specialty: str, side: str, reply: str) -> tuple[dict, dict, dict]:
    text, label = _SIDES[side]
    return tuple(_in_world(world, monkeypatch, [_BOOKINGS[specialty], text, reply], switches=_ON, llm=_llm(text, label)))


@pytest.mark.parametrize("reply", _NOT_OFFERED)
@pytest.mark.parametrize("side", sorted(_SIDES))
@pytest.mark.parametrize("specialty", _STEMS)
def test_branch_not_offered_after_a_side_answer_is_never_booked(specialty, side, reply, two_branch_world, monkeypatch):
    asked, answered, after = _after_side(two_branch_world, monkeypatch, specialty, side, reply)
    text, label = _SIDES[side]
    without_side = _in_world(two_branch_world, monkeypatch, [_BOOKINGS[specialty], reply], switches=_ON, llm=_llm(text, label))[-1]

    assert list(asked["state"][expects.EXPECTS_KEY]["options"][:2]) == list(_TWO_BRANCHES), asked["text"]
    assert "appointment_branch_side_question" in answered["flags"], answered  # побочный ответ был
    assert not after["state"].get("branch_name") and not after["handoff"], after["text"]  # ни чужого, ни «любого» филиала
    assert after["state"].get("city") == "Самара", after["state"].get("city")  # слово адреса — не город
    assert after["text"] == without_side["text"], after["text"]  # как без побочного вопроса — снова список
    assert after["state"][expects.EXPECTS_KEY]["options"] == asked["state"][expects.EXPECTS_KEY]["options"]


@pytest.mark.parametrize(("reply", "index"), sorted(_PICKS.items()))
@pytest.mark.parametrize("side", sorted(_SIDES))
@pytest.mark.parametrize("specialty", _STEMS)
def test_offered_branch_after_a_side_answer_is_that_branch(specialty, side, reply, index, two_branch_world, monkeypatch):
    asked, answered, after = _after_side(two_branch_world, monkeypatch, specialty, side, reply)

    assert "appointment_branch_side_question" in answered["flags"], answered
    assert after["state"].get("branch_name") == _TWO_BRANCHES[index], after["text"]
    assert _TWO_BRANCHES[index] in after["text"], after["text"]


@pytest.mark.parametrize("side", sorted(_SIDES))
def test_any_branch_after_a_side_answer_is_any_branch(side, two_branch_world, monkeypatch):
    _asked, answered, after = _after_side(two_branch_world, monkeypatch, "уролог", side, _ANY)

    assert "appointment_branch_side_question" in answered["flags"], answered
    assert after["state"].get("branch_name") == expects.ANY_BRANCH_OPTION, after["text"]


@pytest.mark.parametrize("side", sorted(_SIDES))
def test_side_answer_keeps_the_branch_question_as_it_was(side, two_branch_world, monkeypatch):
    """Поле ожидания после побочного ответа — прежний вопрос: его текст со списком записи, пункты и счёт переспросов."""

    text, label = _SIDES[side]
    asked, answered = _in_world(two_branch_world, monkeypatch, [_BOOKINGS["уролог"], text], switches=_ON, llm=_llm(text, label))

    assert "appointment_branch_side_question" in answered["flags"], answered
    assert answered["state"][expects.EXPECTS_KEY] == asked["state"][expects.EXPECTS_KEY]


def test_without_pending_buttons_there_is_no_side_answer(two_branch_world, monkeypatch):
    text, label = _SIDES["hours"]
    off = _in_world(two_branch_world, monkeypatch, [_BOOKINGS["уролог"], text], switches=_OFF, llm=_llm(text, label))[-1]

    assert "appointment_branch_side_question" not in off["flags"]  # как было
