"""Кто делает услугу — по данным МИС (`doctor_prices`), а не по правилу «процедура → специальность» (ход (г), L-06).

Инвариант класса (принцип 1 «факты — из данных МИС» и «знания о клинике — данные, не списки в коде»): врачей услуги
называет связь «врач ↔ код услуги» в МИС. Правило кода («цистоскопия → уролог», «эндоскопия → эндоскопист») — кандидат:
сужает выдачу, только если среди врачей услуги есть врачи этой роли; само врачей не добавляет. Врач другого города по
коду не появляется; нет данных или среза — ответ как был (решать «врачей нет» по пустым данным нельзя).
Выключатель `doctors_by_service_data`: off — как было (и ни строки телеметрии), shadow — ответ как при off + согласие
выдачи с данными в ход, on — врачи услуги из данных. Журнал: STEP-2026-10-21-G-L06-DOCTORS-BY-SERVICE.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services import Services
from messengers_router.services import doctors as doctors_module
from messengers_router.services._service_doctors import agreement, build_service_doctor_index

_HERE, _AWAY = "пр.Ленина, 5", "г. Сызрань, ул. Кирова, 1"


_UNITS = {"Врач-уролог": 17, "Врач-хирург": 40, "Врач-эндоскопист": 30, "Врач-оториноларинголог": 55}


def _doctor(doctor_id: int, unit: str, text: str, region: str = _HERE) -> dict:
    return {
        "id": doctor_id,
        "fio": f"Врач{doctor_id} Тестовый Тестович",
        "specialization": text,
        "regions": [region],
        "units": [unit],
        "main_units": [unit],
        "region_ids": [101 if region == _HERE else 909],
        "unit_links": [{"company_unit_id": _UNITS[unit], "company_unit_name": unit, "main": True, "specialization": text}],
        "main_specializations": [text],
    }


_DOCTORS = [
    _doctor(1, "Врач-уролог", "Приём взрослых пациентов"),
    _doctor(2, "Врач-уролог", "Приём взрослых пациентов"),
    _doctor(3, "Врач-хирург", "Амбулаторная хирургия"),
    _doctor(4, "Врач-эндоскопист", "• Эндоскопия полости носа и носоглотки\n• ФГДС"),
    _doctor(5, "Врач-оториноларинголог", "Заболевания уха, горла, носа"),
    _doctor(6, "Врач-уролог", "Приём взрослых пациентов", region=_AWAY),
]
_CYSTO, _NOSE, _BIOPSY = "Цистоскопия", "Эндоскопия полости носа и носоглотки с видеозаписью", "Гистологическое исследование эндоскопических биоптатов"


def _price(doctor_id: int, name: str, code: str, unit: int, branch: str = _HERE) -> dict:
    return {"doctorId": doctor_id, "serviceName": name, "serviceHomecode": code, "companyUnitId": unit, "regionName": branch}


_DOCTOR_PRICES = [
    _price(1, _CYSTO, "4.2.1", 17),  # цистоскопию делает один из двух урологов
    _price(3, _CYSTO, "4.2.1", 40),  # и хирург (другая роль приёма)
    _price(6, _CYSTO, "4.2.1", 17, branch=_AWAY),  # врач другого города
    _price(5, _NOSE, "7.3.1", 55),  # эндоскопию носа делает ЛОР, не эндоскопист
    _price(1, "Массаж предстательной железы (1 сеанс)", "4.9.1", 17),  # единственный «массаж» с врачом в МИС
]


@pytest.fixture
def services(monkeypatch):
    svc = Services()
    svc.ensure_background_refresh_started = lambda: None

    async def doctors():
        return [dict(d) for d in _DOCTORS]

    async def tokens():
        return {"пр ленина 5", "ленина"}

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", doctors)
    monkeypatch.setattr(svc, "_samara_region_tokens", tokens)
    monkeypatch.setattr(api_price, "load_doctor_prices", lambda *a, **k: list(_DOCTOR_PRICES))
    return svc


@pytest.fixture
def notes(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(doctors_module.turn_telemetry, "note", lambda **kw: seen.update(kw))
    return seen


def _ask(services, monkeypatch, mode: str, query: str, entities: dict) -> tuple[list[int], dict]:
    monkeypatch.setattr(doctors_module._cfg, "MR_DOCTORS_BY_SERVICE_DATA", mode, raising=False)
    payload = asyncio.run(services.doctors_info(query, dict(entities)))
    return sorted(d["id"] for d in payload["doctors"]), payload


_CASES = {
    # правило «цистоскопия → уролог» показывает всех урологов; МИС: цистоскопию делают уролог 1 и хирург 3
    "rule_role": ("кто делает цистоскопию", {"service_name": "цистоскопия"}),
    # пациент назвал роль сам — роль сужает врачей услуги
    "named_role_narrows": ("уролог делает цистоскопию?", {"service_name": "цистоскопия", "specialty": "уролог"}),
    # текстовый фильтр нашёл эндоскописта по описанию; МИС: услугу делает ЛОР
    "text_filter_other_role": (_NOSE, {"service_name": _NOSE}),
    # МИС услугу ни с кем не связывает — сравнивать не с чем, ответ как был
    "no_data": (_BIOPSY, {"service_name": _BIOPSY}),
    # общее слово: все слова запроса есть только в названии чужой услуги — это буквы, не опознание (свип 21.10)
    "words_endoscopy": ("кто делает эндоскопию", {"service_name": "эндоскопия"}),
    "words_massage": ("массаж", {"service_name": "массаж"}),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_shadow_never_changes_the_answer(services, monkeypatch, notes, case):
    query, entities = _CASES[case]
    off, off_payload = _ask(services, monkeypatch, "off", query, entities)
    assert notes == {}, "off — ни строки телеметрии"
    shadow, shadow_payload = _ask(services, monkeypatch, "shadow", query, entities)

    assert shadow == off
    assert shadow_payload["service_filter_emptied"] == off_payload["service_filter_emptied"]
    assert notes["svc_doc_src"] == "rules"
    assert "fio" not in str(notes).lower() and "Тестов" not in str(notes)  # в журнале хода — счётчики, не ФИО


def test_shadow_measures_agreement_of_current_answer_with_mis(services, monkeypatch, notes):
    # «цистоскопия»: показаны оба уролога (уролог 2 услугу не делает), хирург 3 потерян — частичное согласие
    expected = {"rule_role": "shown_partial", "text_filter_other_role": "shown_other", "no_data": "no_data"}
    for case, label in expected.items():
        notes.clear()
        _ask(services, monkeypatch, "shadow", *_CASES[case])
        assert notes["svc_doc_agree"] == label, case


def test_on_shows_doctors_linked_to_the_service_not_the_whole_rule_role(services, monkeypatch, notes):
    off, _ = _ask(services, monkeypatch, "off", *_CASES["rule_role"])
    on, payload = _ask(services, monkeypatch, "on", *_CASES["rule_role"])

    assert off == [1, 2]  # дефект: все урологи, хотя уролог 2 цистоскопию не делает
    assert on == [1, 3]  # врачи услуги по МИС; роль, выведенная правилом из процедуры, не сужает: хирург тоже делает
    assert payload["doctors_source"] == "data" and notes["svc_doc_src"] == "data"


def test_on_role_named_by_patient_narrows_service_doctors(services, monkeypatch, notes):
    on, payload = _ask(services, monkeypatch, "on", *_CASES["named_role_narrows"])

    assert on == [1]
    assert payload["doctors_source"] == "data_role"


def test_on_service_done_by_other_role_shows_that_role(services, monkeypatch, notes):
    off, _ = _ask(services, monkeypatch, "off", *_CASES["text_filter_other_role"])
    on, _ = _ask(services, monkeypatch, "on", *_CASES["text_filter_other_role"])

    assert off == [4]  # дефект: эндоскопист по словам описания
    assert on == [5]  # МИС: эту эндоскопию делает ЛОР


def test_on_without_data_answers_as_before(services, monkeypatch, notes):
    off, off_payload = _ask(services, monkeypatch, "off", *_CASES["no_data"])
    on, on_payload = _ask(services, monkeypatch, "on", *_CASES["no_data"])

    assert on == off
    assert on_payload["service_filter_emptied"] == off_payload["service_filter_emptied"]
    assert notes["svc_doc_src"] == "rules"


@pytest.mark.parametrize("case", ["words_endoscopy", "words_massage"])
def test_on_service_matched_only_by_words_answers_as_before(services, monkeypatch, notes, case):
    """«Эндоскопия» — не «Эндоскопия полости носа» (ЛОР), «массаж» — не «Массаж предстательной железы» (уролог):
    подмножество слов в названии строки не опознаёт услугу (принцип 2), данные МИС по такой строке не действуют."""

    off, _ = _ask(services, monkeypatch, "off", *_CASES[case])
    on, payload = _ask(services, monkeypatch, "on", *_CASES[case])

    assert on == off
    assert notes["svc_doc_ident"] == "words" and notes["svc_doc_src"] == "rules"
    assert payload["doctors_source"] == "rules"


def test_on_doctor_of_other_city_linked_by_code_is_not_shown(services, monkeypatch, notes):
    on, _ = _ask(services, monkeypatch, "on", *_CASES["rule_role"])

    assert 6 not in on


def test_slice_unavailable_answers_as_before(services, monkeypatch, notes):
    def broken(*_a, **_k):
        raise OSError("срез цен врачей недоступен")

    monkeypatch.setattr(api_price, "load_doctor_prices", broken)
    off, _ = _ask(services, monkeypatch, "off", *_CASES["rule_role"])
    on, _ = _ask(services, monkeypatch, "on", *_CASES["rule_role"])

    assert on == off
    assert notes["svc_doc_agree"] == "no_slice" and notes["svc_doc_src"] == "rules"


# --- чистая функция среза --------------------------------------------------------------------------------------------


def _here_row(row: dict) -> bool:
    return row.get("regionName") != _AWAY


def test_new_service_and_role_in_mis_are_known_without_code_change():
    """AUT-5 для услуг: процедуры нет ни в одном списке кода — достаточно связи в МИС."""

    index = build_service_doctor_index([_price(77, "Полисомнография ночная", "99.1", 777)], in_region=_here_row)

    item = index.by_code("99.1")
    assert (item.doctor_ids, item.unit_ids, item.name) == ((77,), (777,), "Полисомнография ночная")
    assert index.codes_where(lambda name: "полисомнография" in name.lower()) == ("99.1",)


def test_index_keeps_region_rows_only_and_merges_by_code():
    index = build_service_doctor_index(_DOCTOR_PRICES + [_price(2, _CYSTO, "4.2.1", 17)], in_region=_here_row)

    assert index.by_code("4.2.1").doctor_ids == (1, 2, 3)
    assert index.by_code("4.2.1").unit_ids == (17, 40)
    assert index.doctors_for(["7.3.1", "4.2.1", "нет-кода"]) == (5, 1, 2, 3)


@pytest.mark.parametrize("rows", [[], [{"doctorId": 1}], [{"serviceHomecode": "1"}], ["мусор"]])
def test_empty_or_broken_slice_is_empty_index(rows):
    assert build_service_doctor_index(rows, in_region=lambda _row: True).services == {}


@pytest.mark.parametrize(
    "data, shown, label",
    [
        ((), (1,), "no_data"),
        ((1, 2), (), "shown_none"),
        ((1, 2), (2, 1), "same"),
        ((1,), (1, 2), "shown_wider"),
        ((1, 2), (1,), "shown_partial"),
        ((1,), (2,), "shown_other"),
    ],
)
def test_agreement_labels(data, shown, label):
    assert agreement(data, shown) == label
