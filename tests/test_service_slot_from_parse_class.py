"""Слот-услуга после памятки — понятое, не слова вопроса пациента (Д3-слот, STEP-2026-10-30-D3-SLOT-FROM-PARSE, `pending_in_frame`).

Совместная карточка набора 22 (`s22_dlgnew_on`, диалог `d22_d3_letters_book`): «Цена анализа» → «ферритин» → «а как к нему
подготовиться?» (памятка ферритина — верно) → «а запишите меня» → «Есть возможность записи на Нему подготовиться…». Под on тип
вопроса (шаг 2) верно не уводит запись обратно в памятку, и ход читает мусор, который лежал в слоте и раньше: на ходу подготовки
грундер (`entity_grounder.ground_decision_entities`) клал в `service_name` фразу реплики по буквам (`extract_service_phrase`) —
у темы без проверки каталогом (PREPARE) без проверки вовсе. Разбор хода (LLM) этой реплики пуст — «к нему» услугу не называет.

Инвариант класса (тот же, что Н-1): под `pending_in_frame = on` значение слота-услуги темы без проверки каталогом — позиции разбора
хода этой реплики; разбор решил «услуги нет» — показанная прошлым ответом услуга (одна строка), сбой — слот не пишется. Названная услуга («как подготовиться к узи
брюшной полости») — в слоте, как была. off — как было. Через обработчик эндпоинта, на три-четыре хода; LLM — подделки настоящего
типа (судья, классификатор, оракул разметки хода).
"""

from __future__ import annotations

import pytest

from tests.test_pending_in_frame_class import _dialog, _press, _service
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_PRICE = "menu.price.test"
# Вопросы о подготовке к только что показанной услуге — местоимением, без названия: разбор хода пуст.
_PRONOUN = ["а как к нему подготовиться?", "как подготовиться к этому анализу?", "а что нужно перед ним?"]


def _after_memo(world_dir, monkeypatch, followup: str, last: str, *, in_frame: str) -> list[dict]:
    turns = [_press(_PRICE), "ферритин", followup, last]
    return _dialog(world_dir, monkeypatch, turns, in_frame=in_frame, judge={"ферритин": "answer"}, phrases=("ферритин",), label="PREPARE")


@pytest.mark.parametrize("followup", _PRONOUN)
def test_pronoun_prepare_question_leaves_the_shown_service_in_the_slot(followup, mini_prod_world, monkeypatch):  # noqa: F811
    turns = _after_memo(mini_prod_world, monkeypatch, followup, "а запишите меня", in_frame="on")
    assert _service(turns[1]) == "Ферритин", turns[1]["state"]
    assert turns[2]["label"] == "PREPARE", turns[2]
    assert _service(turns[2]) == "Ферритин", turns[2]["state"]
    words = {w for w in followup.lower().strip("?").split() if len(w) > 3}
    assert not any(w in turns[3]["text"].lower() for w in words), turns[3]["text"]


def test_off_puts_the_question_words_in_the_slot_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    turns = _after_memo(mini_prod_world, monkeypatch, _PRONOUN[0], "а запишите меня", in_frame="off")
    assert _service(turns[2]) == "Нему подготовиться", turns[2]["state"]


def test_named_service_still_fills_the_slot(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречная сторона: услуга названа — позиция разбора в слоте, как была (следующий ход знает её)."""

    on = _dialog(
        mini_prod_world, monkeypatch, ["как подготовиться к узи брюшной полости", "а сколько стоит?"],
        in_frame="on", judge={}, phrases=("узи брюшной полости",), label="PREPARE",
    )
    assert _service(on[0]).lower() == "узи брюшной полости", on[0]["state"]
    assert "entity_dropped_service_not_in_parse" not in on[0]["flags"], on[0]["flags"]


@pytest.mark.parametrize("phrases", [(), None], ids=["empty", "down"])
def test_no_parse_writes_no_slot_on_a_fresh_prepare_question(phrases, mini_prod_world, monkeypatch):  # noqa: F811
    """Разбор пуст или разметка упала — слот не пишется (Н-1): реплика целиком значением услуги не становится."""

    turns = _dialog(mini_prod_world, monkeypatch, ["а как к нему подготовиться?"], in_frame="on", judge={}, phrases=phrases, label="PREPARE")
    assert not _service(turns[0]), turns[0]["state"]
