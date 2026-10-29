"""Малый герметичный набор по рамке кнопки и висящему вопросу — эталон деплоя 21 (консилиум 30.10, Q6: 30–50 фраз в гейте).

Все 20 листьев `button_menu` (таблица шлюза v1): нажатие → одна реплика пациента. Виды реплики: ответ на вопрос кнопки, свой вопрос
в теме кнопки (`domain` варианта Б), другая тема, срочное, просьба оператора; у листа без вопроса — рамка `same` / `other`; у
адресных листьев и листьев без рамки — обычный уточняющий вопрос. Что поняла LLM, задаёт тест (харнесс `understanding_contract`,
подделка настоящего типа, конфиг деплоя 21).

Эталон — понимание и решение хода, а не текст: тема, перевод, кто дал тему (`nlu_topic_by_llm`), исход судьи висящего вопроса,
что легло в слот (`service_name` / `test_goal`), какой вопрос висит после хода, оффер оператора, срочное. Он фиксирует поведение
деплоя 21 вместе с известными дефектами (Н-1: реплика целиком в слоте после `domain` — видно в `slot`). Шаги 0–5 «Понимания»
меняют эти строки только намеренно: перезапись `GOLDEN_UPDATE=1 pytest tests/test_button_frame_phrases_d21.py`, затем дифф
`tests/golden/button_frame_d21.json` — в коммит с объяснением, какие строки и почему.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from messengers_router import button_menu
from tests import understanding_contract as uc
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.understanding_contract import d21_world  # noqa: F401 — фикстура

GOLDEN = Path(__file__).resolve().parent / "golden" / "button_frame_d21.json"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"

_URGENT = "у меня сильная боль в груди и трудно дышать"
_OPERATOR = "соедините с оператором"
_ELSEWHERE = uc.Understood(pending="new_topic", services=(), topic="ADDRESS")

# (лист, вид, реплика, что поняла LLM)
CASES: list[tuple[str, str, str, uc.Understood]] = [
    # --- листья с вопросом-слотом и `_expects` (рамку держит судья): ответ / свой вопрос в рамке / другая тема
    ("menu.appointment.book", "answer", "к кардиологу", uc.Understood(pending="answer", services=())),
    ("menu.appointment.book", "domain", "а кардиолог в субботу принимает?", uc.Understood(pending="domain", services=())),
    ("menu.appointment.book", "new_topic", "где ближайший филиал?", _ELSEWHERE),
    ("menu.appointment.prepare", "answer", "гастроскопия", uc.Understood(pending="answer", services=("гастроскопия",))),
    ("menu.appointment.prepare", "domain", "а гастроскопию под наркозом делаете?", uc.Understood(pending="domain", services=("гастроскопию",))),
    ("menu.tests.prepare", "answer", "ферритин", uc.Understood(pending="answer", services=("ферритин",))),
    ("menu.tests.prepare", "domain", "а у вас делают ВПЧ?", uc.Understood(pending="domain", services=("ВПЧ",))),
    ("menu.tests.prepare", "new_topic", "а сколько стоит ТТГ?", uc.Understood(pending="new_topic", services=("ТТГ",), topic="PRICE")),
    ("menu.tests.suggest", "answer", "для мамы, ей 60 лет", uc.Understood(pending="answer", services=())),
    ("menu.tests.suggest", "domain", "а какие анализы на щитовидку бывают?", uc.Understood(pending="domain", services=())),
    ("menu.tests.suggest", "new_topic", "где ближайший филиал?", _ELSEWHERE),
    ("menu.tests.price", "answer", "ферритин", uc.Understood(pending="answer", services=("ферритин",))),
    ("menu.tests.price", "domain", "есть ли у вас узи щитовидки", uc.Understood(pending="domain", services=("узи щитовидки",))),
    ("menu.price.test", "answer", "глюкоза", uc.Understood(pending="answer", services=("глюкоза",))),
    ("menu.price.test", "domain", "а ферритин у вас делают?", uc.Understood(pending="domain", services=("ферритин",))),
    ("menu.price.test", "new_topic", "где ближайший филиал?", _ELSEWHERE),
    ("menu.price.doctor", "answer", "невролог", uc.Understood(pending="answer", services=("невролог",))),
    ("menu.price.doctor", "domain", "а невролог у вас есть?", uc.Understood(pending="domain", services=("невролог",))),
    ("menu.results.lookup", "answer", "Иванов, 1989, Бг, 123", uc.Understood(pending="answer", services=())),
    ("menu.results.lookup", "domain", "а сколько ждать результат?", uc.Understood(pending="domain", services=())),
    ("menu.results.missing", "answer", "не помню номер", uc.Understood(pending="answer", services=())),
    ("menu.results.missing", "new_topic", "где ближайший филиал?", _ELSEWHERE),
    # --- срочное и оператор поверх рамки — решение судьи не важно (принцип 4)
    ("menu.price.test", "urgent", _URGENT, uc.Understood(pending="answer", services=())),
    ("menu.tests.prepare", "urgent", _URGENT, uc.Understood(pending="domain", services=())),
    ("menu.tests.suggest", "operator", _OPERATOR, uc.Understood(pending="answer", services=())),
    ("menu.price.doctor", "operator", _OPERATOR, uc.Understood(pending="new_topic", services=())),
    # --- лист без вопроса с рамкой листа («Нужно ли записываться»): вопрос рамки same / other
    ("menu.tests.booking_needed", "same", "Записаться на хгч", uc.Understood(frame="same", services=("хгч",))),
    ("menu.tests.booking_needed", "other", "Записаться на хгч", uc.Understood(frame="other", services=("хгч",), topic="APPOINTMENT")),
    ("menu.tests.booking_needed", "same", "а ферритин тоже без записи?", uc.Understood(frame="same", services=("ферритин",))),
    # --- адресные листья (метка `_address_button`) и листья без рамки
    ("menu.tests.where", "followup", "а на Ленина есть?", uc.Understood(services=(), topic="ADDRESS")),
    ("menu.address", "followup", "а до скольки работаете?", uc.Understood(services=(), topic="ADDRESS")),
    ("menu.appointment.schedule", "answer", "кардиолог", uc.Understood(services=(), topic="DOCTOR_SCHEDULE")),
    ("menu.appointment.change", "answer", "на пятницу", uc.Understood(services=(), topic="APPOINTMENT")),
    ("menu.price.promo", "followup", "а на анализы есть скидки?", uc.Understood(services=(), topic="NEWS")),
    ("menu.docs.tax", "followup", "за прошлый год можно?", uc.Understood(services=(), topic="OTHER")),
    # --- листья-переводы: следующая реплика — общий путь
    ("menu.results.eta", "followup", "да", uc.Understood(pending="yes", services=())),
    ("menu.docs.contract", "followup", "спасибо", uc.Understood(services=(), topic="OTHER")),
    ("menu.docs.sick_leave", "followup", "сколько стоит больничный?", uc.Understood(services=(), topic="PRICE")),
    ("menu.operator", "followup", "сколько стоит ферритин", uc.Understood(services=("ферритин",), topic="PRICE")),
    ("menu.operator", "urgent", _URGENT, uc.Understood(services=())),
]


def _case_id(case: tuple[str, str, str, uc.Understood]) -> str:
    return f"{case[0]}|{case[1]}|{case[2]}"


def _row(press: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    tel = turn["telemetry"]
    expects = turn["expects"] if isinstance(turn["expects"], dict) else {}
    return {
        "press": {"label": press["label"], "handoff": press["handoff"], "expects": (press["expects"] or {}).get("type")},
        "label": turn["label"],
        "handoff": turn["handoff"],
        "topic_by": tel.get("nlu_topic_by_llm"),
        "pending": [tel.get("tu_pending_outcome"), tel.get("tu_pending_src")],
        "frame": tel.get("button_frame"),
        "slot": {k: turn[k] for k in ("service_name", "test_goal") if turn[k]},
        "expects": expects.get("type"),
        "offer": "Перевести на оператора" in turn["text"],
        "urgent": str(tel.get("answer_source") or "").startswith(("precheck:urgent", "safety")),
    }


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}


def test_set_covers_every_leaf_and_stays_small():
    leaves = {case[0] for case in CASES}
    assert leaves == set(button_menu.BUTTONS) == set(GATEWAY_V1_TITLES), set(button_menu.BUTTONS) ^ leaves
    assert 30 <= len(CASES) <= 50 and len({_case_id(c) for c in CASES}) == len(CASES)


def test_golden_has_no_orphans():
    if UPDATE:
        pytest.skip("перезапись эталона")
    assert set(_golden()) == {_case_id(c) for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_button_frame_phrase_matches_d21(case, d21_world, monkeypatch):  # noqa: F811
    button_id, _kind, reply, understood = case
    record = uc.run(d21_world, monkeypatch, [uc.press(button_id), uc.Turn(text=reply, understood=understood)])
    row = _row(record[0], record[1])
    if UPDATE:
        data = _golden()
        data[_case_id(case)] = row
        GOLDEN.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    expected = _golden().get(_case_id(case))
    assert expected is not None, f"нет эталона: GOLDEN_UPDATE=1 pytest {__file__}"
    assert row == expected
