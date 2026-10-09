"""ДМС и возраст пациентов врача — из МИС в срез и в карточку (решение владельца 09.10).

`/site/doctors` отдаёт `inaccessibilityDms`, `minAgePatient`, `maxAgePatient`, а сборщик среза брал
только id, fio, ord. Инварианты: сборщик сохраняет поля как есть (None = МИС не заполнила);
карточка показывает возраст и «по ДМС не ведёт» только по данным — пустое поле не превращается
в «не принимает».
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_nayka
from messengers_router.renderer import doctor_audience_lines, format_doctor_info_for_patient


@pytest.fixture
def fake_mis(monkeypatch):
    doctors = [
        {"id": 1, "fio": "Детская Анна Ивановна", "ord": 1, "inaccessibilityDms": False, "minAgePatient": 0, "maxAgePatient": 18},
        {"id": 2, "fio": "Взрослый Иван Петрович", "ord": 2, "inaccessibilityDms": True, "minAgePatient": 18, "maxAgePatient": None},
        {"id": 3, "fio": "Неизвестный Пётр", "ord": 3, "inaccessibilityDms": False, "minAgePatient": None, "maxAgePatient": None},
    ]
    monkeypatch.setattr(api_nayka, "site_doctors", lambda: doctors)
    monkeypatch.setattr(api_nayka, "site_company_units", lambda: [{"id": 10, "name": "Врач-педиатр"}])
    monkeypatch.setattr(api_nayka, "site_doctor_company_units", lambda: [{"worker": d["id"], "companyUnit": 10, "main": True, "specialization": "Приём"} for d in doctors])
    monkeypatch.setattr(api_nayka, "site_doctor_regions", lambda: [{"worker": d["id"], "region": 100, "companyUnit": 10, "hasSchedule": True} for d in doctors])
    monkeypatch.setattr(api_nayka, "site_regions", lambda *a, **k: [{"id": 100, "name": "пр.Ленина, 5"}])
    monkeypatch.setattr(api_nayka, "_filter_region_entries_with_schedule", lambda _id, links, *_a: links)


def test_builder_keeps_dms_and_age_fields(fake_mis):
    built = {d["id"]: d for d in api_nayka.get_all_doctors()}

    assert (built[1]["inaccessible_dms"], built[1]["min_age_patient"], built[1]["max_age_patient"]) == (False, 0, 18)
    assert (built[2]["inaccessible_dms"], built[2]["min_age_patient"]) == (True, 18)
    assert (built[3]["min_age_patient"], built[3]["max_age_patient"]) == (None, None)


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ({"min_age_patient": 0, "max_age_patient": 18}, ["Принимает детей до 18 лет"]),
        ({"min_age_patient": 18}, ["Принимает взрослых (с 18 лет)"]),
        ({"min_age_patient": 14}, ["Принимает пациентов с 14 лет"]),
        ({"min_age_patient": 0}, ["Принимает взрослых и детей"]),
        ({"max_age_patient": 18}, ["Принимает детей до 18 лет"]),
        ({"inaccessible_dms": True}, ["Приём по ДМС не ведёт"]),
        ({"min_age_patient": None, "max_age_patient": None, "inaccessible_dms": False}, []),
        ({"inaccessible_dms": None}, []),
    ],
    ids=["child_range", "adults", "from_14", "all_ages", "children_only", "no_dms", "unknown", "dms_unknown"],
)
def test_audience_lines_only_from_data(doc, expected):
    assert doctor_audience_lines(doc) == expected


def test_card_shows_audience_lines(fake_mis):
    doctors = api_nayka.get_all_doctors()

    text = format_doctor_info_for_patient({"doctors": doctors}, {})

    assert "Принимает детей до 18 лет" in text
    assert "Приём по ДМС не ведёт" in text
    assert text.count("Принимает") == 2  # у «Неизвестного» строк о возрасте нет
