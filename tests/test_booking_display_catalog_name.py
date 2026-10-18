"""Название услуги в записи — как в каталоге, а не «запись на услугу» (BUG-2026-10-18-BOOKING-DISPLAY-PRIEM).

«запишите на консультацию гинеколога» → «Есть возможность записи на услугу…», а оператор получал «Заявка на запись:
услугу, …»: `policies.appointment_service_display` отбрасывал `service_name` со словом «приём» — гард против утёкшей
фразы записи пациента («записаться на прием»), а «Прием (осмотр, консультация) врача-акушера-гинеколога» — название
строки каталога. Гард теперь — по глаголу записи (`detect_appointment_action`), не по слову «приём».

Инварианты класса: ни одно название строки каталога со словом «приём» не показывается как «услугу»; фраза записи
пациента вместо услуги — по-прежнему не название; через эндпоинт — название в вопросе о филиале и в заявке оператору.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from messengers_router.policies import appointment_service_display


def _catalog_names_with_priem() -> list[str]:
    from messengers_router.services import Services

    rows = asyncio.run(Services()._ensure_service_catalog_rows_loaded())
    names = {str(r.get("serviceName") or r.get("name") or "").strip() for r in rows if isinstance(r, dict)}
    return sorted(n for n in names if re.search(r"при[её]м", n, re.I))


def test_catalog_priem_names_are_shown_as_names():
    names = _catalog_names_with_priem()
    assert len(names) >= 50, len(names)  # сторож: каталог гейта не пуст
    hidden = [name for name in names if appointment_service_display({"service_name": name}) == "услугу"]
    assert hidden == [], hidden[:10]


@pytest.mark.parametrize("phrase", ["записаться на прием", "хочу записаться к врачу", "перенести прием", "отменить запись"])
def test_booking_phrase_is_not_a_service_name(phrase):
    assert appointment_service_display({"service_name": phrase}) == "услугу"
    assert appointment_service_display({"service_name": phrase, "specialty": "терапевт"}) == "приём к терапевту"


def test_gynecologist_booking_names_the_service_to_patient_and_operator():
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[str]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in ("запишите на консультацию гинеколога", "г. Самара, пр. Ленина, 5", "в субботу утром"):
            req = MessengerGenerateRequest(session_id="display-gyn", text=text, debug=True)
            out.append(str(json.loads((await messenger_generate_once(req, memory=memory, services=services)).body).get("text") or ""))
        return out

    branch, wish, summary = asyncio.run(run())
    name = "Прием (осмотр, консультация) врача-акушера-гинеколога"
    assert f"записи на {name}" in branch, branch
    assert name in wish, wish
    assert summary.startswith(f"Заявка на запись: {name}, "), summary
