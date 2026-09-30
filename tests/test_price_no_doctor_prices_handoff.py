"""Нет ни одного врача с ценой по врачебной услуге — перевод на оператора (решение владельца 30.09).

Раньше ответ о цене приёма в этом случае писал «Подходящих врачей по этой услуге
сейчас не нашёл» и предлагал оператора «если нужно». Это ложное отсутствие: у
рефлексотерапевта и офтальмолога врачи в клинике есть, в МИС нет только их цен
(BUG-2026-09-25-NO-DOCTORS-FOUND-FALSE). Инвариант: такой фразы бот не пишет
никогда, а ответ без врачей с ценой — перевод на оператора; розничная цена, если
она есть, остаётся в ответе.
"""

from __future__ import annotations

import pytest

from messengers_router import evidence_keys as ek
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.policies import handoff_message
from messengers_router.renderer import format_service_bundle_for_patient
from messengers_router.response_builder import build_service_bundle_response

_RETAIL = [{"serviceName": "Прием (осмотр, консультация) врача-рефлексотерапевта первичный", "cost": 2000}]
_DOCTOR = {"id": 1, "fio": "Врач Тестовый", "specialty_label": "Кардиолог", "service_price": 3000}


def _respond(payload: dict):
    evidence = Evidence()
    evidence.put(ek.SERVICE_BUNDLE, payload)
    return build_service_bundle_response("PRICE", evidence, SessionState(session_id="t"))


@pytest.mark.parametrize("service_kind", ["doctor_consult", "procedure_with_doctor", ""])
@pytest.mark.parametrize("retail", [_RETAIL, []])
def test_doctor_service_without_doctor_prices_goes_to_operator(service_kind, retail):
    payload = {"service_name": "прием рефлексотерапевт", "service_kind": service_kind, "retail_prices": retail, "doctors": []}

    envelope = _respond(payload)

    assert envelope.handoff is True
    assert handoff_message("price_doctors_via_operator") in envelope.text
    assert "не нашёл" not in envelope.text
    if retail:
        assert "2 000 руб" in envelope.text  # верная розничная цена остаётся


@pytest.mark.parametrize(
    "payload",
    [
        {"service_name": "прием кардиолог", "service_kind": "doctor_consult", "retail_prices": _RETAIL, "doctors": [_DOCTOR]},
        {"service_name": "ферритин", "service_kind": "lab", "retail_prices": [{"serviceName": "Ферритин", "cost": 530}], "doctors": []},
        {"service_name": "узи", "service_kind": "diagnostic_no_doctor", "retail_prices": [{"serviceName": "УЗИ", "cost": 1800}], "doctors": []},
        {"service_name": "узи", "clarify_text": "Введите конкретное название процедуры."},
    ],
    ids=["doctors_listed", "lab", "diagnostic_no_doctor", "clarify"],
)
def test_answers_that_need_no_doctor_do_not_go_to_operator(payload):
    envelope = _respond(payload)

    assert envelope.handoff is False
    assert "не нашёл" not in format_service_bundle_for_patient(payload, {})
