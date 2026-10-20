"""Побочный вопрос о цене внутри записи — все строки приёма специальности, не одна цена (BUG-2026-10-20-BRANCH-SIDE-PRICE-ONE-ROW).

Решение владельца 25.09 №2 (подтверждено 20.10, деплой 12): на вопрос о цене посреди записи — список цен приёма этой
специальности. Было (после BUG-2026-10-20-BRANCH-SIDE-PRICE-FOREIGN): ответ — рендер `doctor_consult`, первая строка
прайса и врачи: у невролога «Прием … врача- невролога — 2 700 руб.», хотя первичный приём у разных врачей стоит 3 000,
6 500 и 20 000.

Инвариант класса: побочный ценовой вопрос внутри записи к специальности без своей услуги показывает каждую строку приёма
этой специальности из прайса со своей ценой (первичный, повторный, у разных врачей); строк чужих специальностей и не
приёма в списке нет; одна строка приёма — ответ как был; своя услуга в реплике — по её словам, как было. Через обработчик
эндпоинта; LLM — подделка настоящего типа (как в `test_branch_side_price_class.py`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evalbench import world as ev_world
from tests.test_branch_side_price_class import _BRANCH_TAIL, _llm, side_price_source
from tests.test_pending_buttons_class import _ON, _in_world

_SPECIALTIES = {"урологу": "уролог", "гинекологу": "гинеколог", "неврологу": "невролог"}
_REPLIES = ("а сколько стоит приём?", "а цена?")
_HEADER = "Стоимость приёма по прайсу клиники:"
# Чужие строки прайса `mini_prod` рядом с приёмами этих специальностей: приём терапевта эндокринологического отделения
# (поиск «приём гинеколог» его находит), приёмы хирурга мира, «Невролиз», «Курс гинекологический».
_FOREIGN = ("терапевт", "хирург", "невролиз", "курс гинекологический")
_ROW = re.compile(r"^- (.+) — ([\d ]+) руб\.", re.M)


@pytest.fixture(scope="module")
def one_row_world(tmp_path_factory) -> Path:
    """Мир `side_price_world`, где у уролога в прайсе одна строка приёма — «Прием (осмотр, консультация) врача-уролога»."""

    def one_urology_row(rows: list[dict]) -> list[dict]:
        base = "Прием (осмотр, консультация) врача-уролога"
        return [row for row in rows if "уролог" not in row["serviceName"].lower() or row["serviceName"] == base]

    src = side_price_source(tmp_path_factory.mktemp("one_row_src"), price=one_urology_row)
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("one_row") / "world")


@pytest.fixture(scope="module")
def many_rows_world(tmp_path_factory) -> Path:
    src = side_price_source(tmp_path_factory.mktemp("many_rows_src"))
    return ev_world.build_world(src, "20261008", tmp_path_factory.mktemp("many_rows") / "world")


@pytest.mark.parametrize("reply", _REPLIES)
@pytest.mark.parametrize("target", sorted(_SPECIALTIES))
def test_side_price_question_lists_every_consult_row_of_the_specialty(target, reply, many_rows_world, monkeypatch):
    own, book = _SPECIALTIES[target], f"хочу записаться к {target}"
    asked, side, picked = _in_world(many_rows_world, monkeypatch, [book, reply, "1"], switches=_ON, llm=_llm(book))

    assert "appointment_branch_side_question" in side["flags"], side
    answer = side["text"].removesuffix("\n\n" + _BRANCH_TAIL)
    assert answer != side["text"] and answer.startswith(_HEADER), side["text"]
    rows = _ROW.findall(answer)
    assert len(rows) >= 2, answer
    assert all(own in name.lower() and re.search(r"при[её]м|консультац", name.lower()) for name, _ in rows), rows
    assert not [word for word in _FOREIGN if word in answer.lower()], answer  # ни одной чужой строки
    assert picked["state"].get("branch_name") == asked["state"]["_expects"]["options"][0], picked  # запись идёт дальше


def test_neurologist_prices_of_different_doctors_are_all_shown(many_rows_world, monkeypatch):
    """Пример владельца: первичный приём невролога — 20 000 у одного врача и 6 500 у другого; видны оба, не одна цена."""

    book = "хочу записаться к неврологу"
    side = _in_world(many_rows_world, monkeypatch, [book, "а сколько стоит приём?"], switches=_ON, llm=_llm(book))[-1]

    for price in ("20 000 руб.", "6 500 руб.", "3 000 руб.", "2 700 руб."):
        assert price in side["text"], side["text"]


def test_one_consult_row_is_answered_as_before(one_row_world, monkeypatch):
    book = "хочу записаться к урологу"
    side = _in_world(one_row_world, monkeypatch, [book, "а сколько стоит приём?"], switches=_ON, llm=_llm(book))[-1]

    assert "appointment_branch_side_question" in side["flags"], side
    assert _HEADER not in side["text"] and "Прием (осмотр, консультация) врача-уролога — 3 100 руб." in side["text"], side["text"]


def test_own_service_in_the_reply_is_searched_by_its_words(many_rows_world, monkeypatch):
    book = "хочу записаться к урологу"
    side = _in_world(many_rows_world, monkeypatch, [book, "а сколько стоит приём гинеколога?"], switches=_ON, llm=_llm(book))[-1]

    assert "appointment_branch_side_question" in side["flags"], side
    assert _HEADER not in side["text"] and "гинеколог" in side["text"].lower(), side["text"]


def test_consult_rows_of_a_specialty_from_the_catalog():
    """Правило отбора только сужает: приём своей специальности с ценой; чужая специальность, не приём, ноль — нет."""

    from messengers_router.router import _consult_price_rows

    rows = [
        {"serviceName": "Прием (осмотр, консультация) врача-акушера-гинеколога", "cost": 3000.0},
        {"serviceName": "Прием (осмотр, консультация) врача-акушера-гинеколога  к.м.н.", "cost": 4000.0},
        {"serviceName": "Прием (осмотр, консультация) врача-терапевта (эндокринологическое отделение)", "cost": 3950.0},
        {"serviceName": "Прием (осмотр, консультация) врача-акушера-гинеколога повторный (второе посещение)", "cost": 0.0},
        {"serviceName": "Курс гинекологический (хронические состояния)", "cost": 39600.0},
    ]
    names = [row["serviceName"] for row in _consult_price_rows(rows, "гинеколог")]

    assert sorted(names) == sorted(row["serviceName"] for row in rows[:2]), names
