"""Текст записи называет врача в дательном падеже, в какой бы форме его ни написал пациент.

Зонд кнопок 04.10: «Есть возможность записи на Кардиолог», «…на Неврологу» — слово
пациента («кардиолог», «к неврологу») подставлялось в шаблон «записи на {услуга}» как
есть. Теперь врач — «на приём к кардиологу»; услуги («УЗИ брюшной полости») и запись к
врачу по фамилии — без изменений. Начальная форма — существующий разбор специальностей
(`extract_specialty`), дательный падеж — одно правило для названий врачей.
"""

from __future__ import annotations

import pytest

from messengers_router.policies import (
    _specialty_dative,
    appointment_service_display,
    appointment_summary,
    appointment_text_branch_prompt,
    appointment_text_datetime_prompt,
)
from messengers_router.specialty_parser import SPECIALTY_CANONICAL

_DOCTORS = [s for s in SPECIALTY_CANONICAL if _specialty_dative(s)]


@pytest.mark.parametrize("specialty", _DOCTORS)
@pytest.mark.parametrize("form", ["как в справочнике", "с заглавной", "дательный"])
def test_doctor_named_as_service_reads_in_dative(specialty, form):
    dative = _specialty_dative(specialty)
    raw = {"как в справочнике": specialty, "с заглавной": specialty.capitalize(), "дательный": dative}[form]
    service = appointment_service_display({"service_name": raw})

    assert service == f"приём к {dative}"
    assert f"записи на приём к {dative} в городе Самара" in appointment_text_branch_prompt(service, "Самара", ["пр.Ленина, 5"])
    assert f"записать на приём к {dative} (" in appointment_text_datetime_prompt(service, "пр.Ленина, 5", None)


@pytest.mark.parametrize(
    ("specialty", "dative"),
    [("кардиолог", "кардиологу"), ("пластический хирург", "пластическому хирургу"), ("травматолог-ортопед", "травматологу-ортопеду"), ("лор", "лору")],
)
def test_specialty_entity_reads_in_dative(specialty, dative):
    assert appointment_service_display({"specialty": specialty}) == f"приём к {dative}"


@pytest.mark.parametrize("service", ["УЗИ брюшной полости", "УЗИ", "Холтер", "ЭКГ", "Общий анализ крови"])
def test_services_keep_their_names(service):
    assert appointment_service_display({"service_name": service}) == service


def test_doctor_by_surname_and_summary_are_unchanged():
    assert appointment_service_display({"doctor_name": "Дразнин Антон Владимирович"}) == "приём к врачу Дразнин Антон Владимирович"
    summary = appointment_summary({"patient_name": "Иванов Иван", "service_name": "Неврологу", "branch_name": "пр.Ленина, 5"})
    assert "приём к неврологу" in summary


@pytest.mark.parametrize("word", ["эндоскопия", "функциональная диагностика", "узи", ""])
def test_non_doctor_words_have_no_dative(word):
    assert _specialty_dative(word) is None
