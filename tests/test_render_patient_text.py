"""Класс «служебный текст в ответе пациенту» (BUG-2026-10-14-RENDER-FORMAT-LABEL).

Смоук прода 14.10: в ответе на список «Формат: По услуге «ТТГ…»:» и «400 руб..». Заголовок группы корзины
(«По услуге «X»») лежал в поле формата оказания (`care_setting_label`), и рендер дописывал «Формат:»; сумма
кончается на «руб.», а шаблон строки добавлял ещё точку. На стенде (dev 867) — 72 и 194 ответа.

Инвариант класса: в тексте пациенту нет служебных подписей рендера и удвоенной точки после суммы — во всех
эталонных диалогах цены (все режимы выключателей), а настоящий формат оказания («поликлиника») печатается как был.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from messengers_router import renderer

_GOLDEN_DIR = Path(__file__).parent / "golden"
_GOLDENS = sorted(_GOLDEN_DIR.glob("price_turns*.json"))
_ARTIFACTS = ("Формат: По услуге", "руб..")


def _texts(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for case, turns in data.items():
        for turn in turns if isinstance(turns, list) else []:
            if isinstance(turn, dict) and isinstance(turn.get("text"), str):
                out.append((case, turn["text"]))
    return out


@pytest.mark.parametrize("golden", _GOLDENS, ids=[p.stem for p in _GOLDENS])
def test_no_service_text_in_patient_answers(golden):
    assert _GOLDENS, "нет эталонов цены"
    bad = [(case, artifact) for case, text in _texts(golden) for artifact in _ARTIFACTS if artifact in text]
    assert not bad, f"служебный текст в ответе пациенту: {bad[:5]}"


def test_basket_group_heading_is_printed_as_is():
    heading = renderer._format_family_care_group_heading("", "", group_heading="По услуге «Ферритин»")
    assert heading == "По услуге «Ферритин»:"


def test_real_care_setting_keeps_its_heading():
    assert renderer._format_family_care_group_heading("поликлиника", "") == "В поликлинике:"
    assert renderer._format_family_care_group_heading("выездная бригада", "") == "Формат: выездная бригада:"


def test_family_answer_groups_basket_positions_without_format_label():
    rows = [
        {"serviceName": "Ферритин", "price": 530, "price_group_heading": "По услуге «Ферритин»"},
        {"serviceName": "ТТГ (TSH) тиреотропный гормон", "price": 400, "price_group_heading": "По услуге «ТТГ»"},
    ]
    text = renderer._format_price_family_variants({"family_variants": rows, "service_name": "ферритин, ттг"}, "")
    assert "Формат:" not in text and "руб.." not in text
    assert "По услуге «Ферритин»:\n1. Ферритин — 530 руб." in text
    assert "По услуге «ТТГ»:\n1. ТТГ (TSH) тиреотропный гормон — 400 руб." in text


@pytest.mark.parametrize(("line", "expected"), [("1. Ферритин — 530 руб.", "1. Ферритин — 530 руб."),
                                                  ("1. Ферритин — цена по запросу", "1. Ферритин — цена по запросу.")])
def test_sentence_ends_with_one_period(line, expected):
    assert renderer._close_sentence(line) == expected
