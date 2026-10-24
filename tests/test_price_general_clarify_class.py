"""Класс: код после выбора строки цены LLM не подменяет её решение ценой (BUG-2026-10-24-FOREIGN-PRICE-D17, класс 3).

Карточка 24.10 (31 вариант семи вопросов с чужой ценой набора деплоя 17): «Сколько стоит гастропанель» — LLM дважды ответила
«слишком общее», а путь цены отдал ответ правил «Гастрин, Гастрин-17»; «стоимость консультации врача» (4 варианта из 4) —
LLM выбрала приёмы разных специальностей, а путь «услуга с врачами» показал первую строку — приём хирурга (против решения
25.09: цену приёма без специальности не называть).

Инварианты под `price_general_clarify = on`:
- «слишком общее» / «не услуга» на пути ответа о цене — уточнение без цен, в обоих инструментах цены;
- несколько разных услуг, выбранных LLM, — список на выбор без врачей одной из них; варианты одной услуги (одно название,
  разные цены) — путь с врачами, как был;
- каждый вызов выбора строки на пути цены передаёт выключатель (новый путь не забудет его);
- off — как было.
Промпты `*_v2` (`price_select_v2`): приём врача без специальности — «слишком общее»; прежние промпты не меняются.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from messengers_router.services import _price_select_llm as PS
from tests.test_price_llm_select import _HYBRID, _fake_llm, _services, run

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_GASTRIN = {"serviceName": "Гастрин", "serviceHomecode": "1001", "cost": 750}
_GASTRIN17 = {"serviceName": "Гастрин-17 стимулированный", "serviceHomecode": "1002", "cost": 1950}
_SURGEON = {"serviceName": "Прием (осмотр, консультация) врача-хирурга", "serviceHomecode": "15.1.1", "cost": 3000}
_UROLOGIST = {"serviceName": "Прием (осмотр, консультация) врача-уролога", "serviceHomecode": "16.1.1", "cost": 3100}
_THERAPIST = {"serviceName": "Прием (осмотр, консультация) врача-терапевта", "serviceHomecode": "13.1.1", "cost": 2700}
_THERAPIST_2 = {"serviceName": "Прием (осмотр, консультация) врача-терапевта", "serviceHomecode": "13.1.2", "cost": 3500}
_FERRITIN = {"serviceName": "Ферритин", "serviceHomecode": "3001", "cost": 530}
RETAIL = [_GASTRIN, _GASTRIN17, _SURGEON, _UROLOGIST, _THERAPIST, _THERAPIST_2, _FERRITIN]

_SURGEON_DOC = {"id": 7, "fio": "Хирург Иван", "ord": 1, "regions": ["г. Самара, пр. Ленина, 5"], "main_units": ["Врач-хирург"], "units": ["Врач-хирург"]}
_THERAPIST_DOC = {"id": 8, "fio": "Терапевт Анна", "ord": 1, "regions": ["г. Самара, пр. Ленина, 5"], "main_units": ["Врач-терапевт"], "units": ["Врач-терапевт"]}
_DOCTOR_PRICES = [
    {"doctorId": 7, "fio": "Хирург Иван", "serviceName": _SURGEON["serviceName"], "serviceHomecode": "15.1.1", "cost": 3000},
    {"doctorId": 8, "fio": "Терапевт Анна", "serviceName": _THERAPIST["serviceName"], "serviceHomecode": "13.1.1", "cost": 2700},
]


@pytest.fixture(autouse=True)
def _enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(PS, "_ENABLED", True)
    monkeypatch.setattr(PS, "_journal_path", lambda: tmp_path / "price_synonym_suggestions.jsonl")


def _switch(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_GENERAL_CLARIFY", mode, raising=False)


def _too_general(prompt, rows):
    return {"match": [], "reason": "too_general", "search_terms": []}


def _prices(payload: dict) -> list:
    return [*(payload.get("prices") or []), *(payload.get("retail_prices") or []), *(payload.get("family_variants") or [])]


def _ask(monkeypatch, mode: str, tool: str, question: str, decide) -> dict:
    _switch(monkeypatch, mode)
    _fake_llm(monkeypatch, decide)
    svc = _services(monkeypatch, RETAIL, doctors=[_SURGEON_DOC, _THERAPIST_DOC], doctor_prices=_DOCTOR_PRICES)
    entities = {"service_name": question, **_HYBRID}
    call = svc.price_info if tool == "price_info" else svc.service_bundle_info
    return run(call(question, entities))


@pytest.mark.parametrize("tool", ["price_info", "service_bundle_info"])
@pytest.mark.parametrize("question", ["сколько стоит гастропанель", "Гастропанель цена"])
def test_too_general_is_a_clarification_not_prices(tool, question, monkeypatch) -> None:
    on = _ask(monkeypatch, "on", tool, question, _too_general)
    assert not _prices(on), on
    assert on.get("clarify_text"), on
    off = _ask(monkeypatch, "off", tool, question, _too_general)
    assert _prices(off), "off — как было: ответ правил (тот самый дефект g0340)"


def _pick_consultations(prompt, rows):
    return {"match": [num for name, num in rows.items() if name.startswith("Прием")], "search_terms": []}


def test_different_services_picked_are_a_list_not_the_first_with_its_doctors(monkeypatch) -> None:
    question = "стоимость консультации врача"
    on = _ask(monkeypatch, "on", "service_bundle_info", question, _pick_consultations)
    names = {row.get("serviceName") for row in on.get("family_variants") or []}
    assert {_SURGEON["serviceName"], _UROLOGIST["serviceName"], _THERAPIST["serviceName"]} <= names, on
    assert not on.get("doctors"), "у списка разных услуг нет врачей одной из них"
    off = _ask(monkeypatch, "off", "service_bundle_info", question, _pick_consultations)
    shown = {row.get("serviceName") for row in off.get("family_variants") or off.get("retail_prices")[:1]}
    assert len(shown) == 1 or names <= shown, "off — как было"


def test_variants_of_one_service_keep_the_doctor_path(monkeypatch) -> None:
    def pick_therapist(prompt, rows):
        return {"match": [num for name, num in rows.items() if name == _THERAPIST["serviceName"]], "search_terms": []}

    for mode in ("off", "on"):
        payload = _ask(monkeypatch, mode, "service_bundle_info", "сколько стоит приём терапевта", pick_therapist)
        assert not payload.get("family_variants"), (mode, payload)
        assert {row.get("serviceName") for row in payload.get("retail_prices") or []} == {_THERAPIST["serviceName"]}, (mode, payload)


@pytest.mark.parametrize(
    ("rows", "distinct"),
    [
        ([_THERAPIST, _THERAPIST_2], False),
        ([_SURGEON, _UROLOGIST], True),
        ([_THERAPIST, {**_THERAPIST_2, "serviceName": "Прием (осмотр консультация) врача-терапевта"}], False),
        ([_FERRITIN], False),
    ],
)
def test_distinct_services_is_by_name_not_by_price(rows, distinct) -> None:
    assert PS.distinct_services(rows) is distinct


def test_every_price_path_choice_passes_the_switch() -> None:
    tree = ast.parse((_ROOT / "messengers_router/services/prices.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "select_price_rows"]
    assert len(calls) >= 2
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        judge = kw.get("judge_qualifier")
        if isinstance(judge, ast.Constant) and judge.value is True:
            continue  # судьи условия (П2 по сущности, гард сильной фразы): «слишком общее» — None, решает основной выбор ниже
        value = kw.get("general_as_none_fit")
        assert isinstance(value, ast.Call) and getattr(value.func, "id", "") == "general_clarify_on", ast.dump(call)[:200]


@pytest.mark.parametrize(("mode", "judge", "task"), [
    ("off", False, "price_service"),
    ("off", True, "price_service_qualifier"),
    ("on", False, "price_service_v2"),
    ("on", True, "price_service_qualifier_v2"),
])
def test_v2_prompts_only_under_the_switch(mode, judge, task, monkeypatch) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_SELECT_V2", mode, raising=False)
    monkeypatch.setattr(cfg, "MR_PRICE_CANDIDATES_BY_MEANING", "on", raising=False)
    assert PS.price_task(judge_qualifier=judge).name == task


@pytest.mark.parametrize("base", ["choose_price_service", "choose_price_service_qualifier"])
def test_v2_prompt_calls_a_doctor_visit_without_specialty_too_general(base) -> None:
    old = (_ROOT / f"messengers_router/prompts/{base}.txt").read_text(encoding="utf-8")
    new = (_ROOT / f"messengers_router/prompts/{base}_v2.txt").read_text(encoding="utf-8")
    assert "без специальности" not in old
    assert "консультация врача" in new and "приём врача без специальности" in new
    assert new == (_ROOT / f"app_data/prompts/mr_{base}_v2.txt").read_text(encoding="utf-8")
