"""Проверка выбора строки — один механизм для цены и подбора анализов (STEP-2026-10-28-POSITION-QUALIFIERS-8; `qualifier_marks = on`).

Класс: «Точно в таком виде в прайсе нет» пишется без сведений клиники о выбранной строке или тем же вызовом, что выбирал строки.
Карточка С3 28.10: проверка выбора цены не видела описания МИС («на яйца глистов» — «нет» в «Соскоб на энтеробиоз», хотя описание
о глистной инвазии) и не доверяла другому названию справочника («на флору» — «нет» при пометке «другое название — мазок на флору»);
ярлык `partly` подбора анализов (`test_assist_answer`) был ложным в 4 из 5 случайных ответов: метод — в описании МИС («определяется
чувствительность к антибиотикам»), «остальное» — станция метро, тот самый женский чекап.

Механизм: после выбора — отдельный вопрос LLM (`_qualifier_pool.check_rows`): какие из слов пациента (название услуги и уточнения
единственной позиции разбора хода) есть в выбранных строках; под каждой строкой — что о ней знает справочник МИС и назвал пациент
(другое название, материал, предложения описания). Заголовок «в таком виде нет» пишет код по этому ответу.

Инварианты класса (подбор анализов, `test_assist_from_data` и `test_assist_price` — on, как на проде):
- уточнение, которого проверка не нашла, — заголовок «Точно в таком виде в прайсе нет. Есть:», строки и цены выбора, оффер;
- проверка нашла всё — заголовок «В прайсе клиники есть:», без оффера, даже если выбор сказал `partly`;
- самой услуги в строках нет (выбор и проверка разошлись) — «нет данных», цены нет (принцип 1);
- уточнений у позиции нет, а выбор сказал `partly` — заголовок «В прайсе клиники есть:» и «остального нет — уточнит оператор»;
- проверка не ответила — решение выбора как есть; off — проверки нет, ответ как был.
Данные справочника под строкой проверки — только совпавшие со словами пациента: прочие названия и предложения описания судить не
помогают, а длинный запрос — время LLM.
"""

from __future__ import annotations

import json
from typing import Any

from messengers_router import test_assist_answer
from messengers_router.service_answer import NO_DATA_TEXT
from messengers_router.services import _biomaterial, _qualifier_pool
from tests.spans_oracle import SpansOracle
from tests.test_test_assist_answer_class import _PRICE, CLASS_REPLIES, OFFER, _answers, _pick, ta_world, turn  # noqa: F401 — фикстура

_CHECK = "Слова пациента (номер. слова):"
_REPLY = CLASS_REPLIES[5]  # «Есть ли у вас анализ на кальпротектин в жидком кале?»
_PHRASE = "кальпротектин в жидком кале"
_QUALIFIER = "в жидком кале"
_SERVICE = "кальпротектин"
_ROW = "Кальпротектин в кале"
_MARKS = {**_PRICE, "MR_POSITION_QUALIFIERS": "on", "MR_QUALIFIER_MARKS": "on"}


def _check(*present: str):
    """Ответ проверки выбора: номера слов пациента из `present`."""

    def answer(prompt: str) -> str:
        lines = (line.partition(". ") for line in prompt.split(_CHECK, 1)[1].splitlines())
        return json.dumps({"picks": [int(head) for head, dot, rest in lines if dot and head.isdigit() and rest in present]})

    return answer


def _turn(world, monkeypatch, *, outcome: str, check: Any, qualifiers=((_PHRASE, _QUALIFIER, "material"),), marks: str = "on") -> dict[str, Any]:
    answers = {_CHECK: check, **_answers(_pick(_ROW, outcome=outcome))}
    cfg = {**_MARKS, "MR_QUALIFIER_MARKS": marks}
    return turn(
        world, monkeypatch, _REPLY, switch="on", answers=answers, cfg_extra=cfg,
        wrap=lambda fake: SpansOracle((_PHRASE,), inner=fake, qualifiers=list(qualifiers)),
    )


def _priced(text: str) -> bool:
    return f"— {_ROW} — 2 730 руб." in text


def test_check_found_everything_is_a_plain_answer_even_if_the_choice_said_partly(ta_world, monkeypatch):  # noqa: F811
    out = _turn(ta_world, monkeypatch, outcome="partly", check=_check(_SERVICE, _QUALIFIER))

    assert out["text"].startswith(test_assist_answer.PICKED_HEADING) and _priced(out["text"]), out["text"]
    assert test_assist_answer.PRICED_PARTLY_HEADING not in out["text"] and not out["pending_offer"]
    assert out["telemetry"]["ta_answer"] == "picked" and out["telemetry"]["price_qual_check"] == "found"


def test_qualifier_the_check_did_not_find_gets_the_partly_heading(ta_world, monkeypatch):  # noqa: F811
    out = _turn(ta_world, monkeypatch, outcome="none", check=_check(_SERVICE))

    assert out["text"].startswith(test_assist_answer.PRICED_PARTLY_HEADING) and _priced(out["text"]), out["text"]
    assert out["text"].endswith(OFFER) and out["pending_offer"]
    assert out["telemetry"]["ta_answer"] == "partly" and out["telemetry"]["price_qual_missing"] == 1
    prompt = out["prompts"][_CHECK]
    assert f"1. {_ROW}" in prompt.split(_CHECK)[0] and f"1. {_SERVICE}" in prompt and f"2. {_QUALIFIER}" in prompt


def test_service_the_check_did_not_find_is_no_data(ta_world, monkeypatch):  # noqa: F811
    out = _turn(ta_world, monkeypatch, outcome="partly", check=_check(_QUALIFIER))

    assert NO_DATA_TEXT in out["text"] and not _priced(out["text"]), out["text"]
    assert out["telemetry"]["ta_answer"] == "no_data" and out["telemetry"]["price_qual_check"] == "service_absent"


def test_check_down_keeps_the_choice(ta_world, monkeypatch):  # noqa: F811
    out = _turn(ta_world, monkeypatch, outcome="partly", check=ConnectionError("тест: проверка недоступна"))

    assert out["text"].startswith(test_assist_answer.PRICED_PARTLY_HEADING) and _priced(out["text"]), out["text"]
    assert out["telemetry"]["ta_answer"] == "partly" and out["telemetry"]["price_qual_check"] == "unavailable"


def test_partly_without_qualifiers_keeps_the_plain_heading_and_offers_the_rest(ta_world, monkeypatch):  # noqa: F811
    """Уточнений у позиции нет — спрошено не об анализе (состав, время, место) или нашлась часть перечня: «в таком виде нет» о
    строке было бы ложью (g0015 «район метро», g0313 «что входит»); остальное — у оператора."""

    out = _turn(ta_world, monkeypatch, outcome="partly", check=_check(_SERVICE, _QUALIFIER), qualifiers=())

    assert out["text"].startswith(test_assist_answer.PICKED_HEADING) and _priced(out["text"]), out["text"]
    assert test_assist_answer.PRICED_PARTLY_TAIL in out["text"] and out["text"].endswith(OFFER) and out["pending_offer"]
    assert out["telemetry"]["ta_answer"] == "partly_rest" and _CHECK not in out["calls"]


def test_off_keeps_the_choice_label(ta_world, monkeypatch):  # noqa: F811
    out = _turn(ta_world, monkeypatch, outcome="partly", check=_check(_SERVICE, _QUALIFIER), marks="off")

    assert out["text"].startswith(test_assist_answer.PRICED_PARTLY_HEADING) and _priced(out["text"]), out["text"]
    assert out["telemetry"]["ta_answer"] == "partly" and _CHECK not in out["calls"]


# --------------------------------------------------------------------------- данные справочника под строкой проверки

_DIRECTORY = [
    {
        "serviceName": "Посев на микрофлору",
        "serviceSynonyms": "посев на флору, бакпосев",
        "biomatNames": "мокрота, кровь из вены, моча",
        "description": "<h1>Описание исследования</h1><p>Исследование выделяет стрептококки и стафилококки.</p><p>Результат &mdash; через 5 дней.</p>",
    },
    {
        "serviceName": "Длинное описание",
        "serviceSynonyms": None,
        "biomatNames": None,
        "description": "<p>" + " ".join(f"слово{k}" for k in range(100)) + " выделяет стрептококки " + " ".join(f"слово{k}" for k in range(100, 200)) + "</p>",
    },
    {"serviceName": "Глюкоза", "serviceSynonyms": None, "biomatNames": "кровь из вены", "description": ""},
]


def _context(monkeypatch, rows: list[dict[str, Any]], words: list[str]) -> str:
    monkeypatch.setattr(_biomaterial, "_vocabularies", lambda: _biomaterial._build_vocabularies(_DIRECTORY))
    return _qualifier_pool.rows_context(rows, words)


def test_the_check_sees_what_the_directory_knows_about_the_row(monkeypatch):
    text = _context(monkeypatch, [{"serviceName": "Посев на микрофлору"}], ["посев мокроты", "на стрептококк", "на флору"])

    assert text.splitlines()[0] == "1. Посев на микрофлору"
    assert "- другое название в справочнике клиники: посев на флору" in text, text
    assert "- материал по справочнику клиники: мокрота" in text, text
    assert "- из описания исследования в справочнике клиники: Исследование выделяет стрептококки и стафилококки. Результат" in text, text
    assert "бакпосев" not in text and "кровь из вены" not in text, text  # слов пациента в них нет


def test_without_directory_data_the_row_is_its_name(monkeypatch):
    text = _context(monkeypatch, [{"serviceName": "Глюкоза", "price": 190}, {"serviceName": "Глюкоза", "price": 210}], ["глюкоза натощак"])

    assert text == "1. Глюкоза"  # одна услуга разных цен — одной строкой; материала «натощак» в справочнике нет


def test_description_is_text_without_markup_and_heading():
    assert _biomaterial._description_text(_DIRECTORY[0]["description"]) == "Исследование выделяет стрептококки и стафилококки. Результат — через 5 дней."


def test_long_description_gives_a_window_around_the_patient_words(monkeypatch):
    text = _context(monkeypatch, [{"serviceName": "Длинное описание"}], ["на стрептококк"])

    excerpt = text.splitlines()[1]
    assert "выделяет стрептококки" in excerpt and excerpt.count("слово") == 30 and excerpt.endswith(" …") and ": … слово91 " in excerpt, excerpt  # окно — за 10 слов до совпадения
