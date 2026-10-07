"""Запись к специальности: врач остаётся врачом, и цену по буквам бот не называет (07.10).

BUG-2026-09-25-SPECIALTY-HITS-PROCEDURE, ревью DLA 04–07.10: «к неврологу» → «стоимость
30000» («Невролиз и декомпрессия нерва»), «к терапевту» → «можем записать на ТЭС-терапия,
1000 руб.», «к гинекологу» → «на УЗИ гинекологическое». Каталог услуг по буквам подменял
и цену, и саму услугу записи: слово пациента «Терапевту» уходило в каталог как услуга.

Инварианты:
- название врача в любом падеже — не услуга и в каталог не идёт: запись «приём к …»;
- в записи к специальности, пока врач не выбран, цену не называем (решение владельца 25.09,
  подтверждено 07.10): у специальности десятки строк приёма;
- запись на услугу («УЗИ брюшной полости») называет цену, как раньше.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState
from messengers_router.planner import _appointment_price_steps
from messengers_router.policies import _specialty_dative, is_specialty_word
from messengers_router.services import Services
from messengers_router.specialty_parser import SPECIALTY_CANONICAL


@pytest.mark.parametrize("specialty", [s for s in SPECIALTY_CANONICAL if _specialty_dative(s)])
def test_doctor_word_in_any_case_is_never_a_catalog_service(specialty):
    for form in (specialty, specialty.capitalize(), _specialty_dative(specialty)):
        assert is_specialty_word(form), form


@pytest.mark.parametrize("service", ["УЗИ брюшной полости", "Холтер", "ЭКГ", "Общий анализ крови", "дерматоскопия"])
def test_services_are_not_doctor_words(service):
    assert not is_specialty_word(service)


@pytest.mark.parametrize(
    ("entities", "tools"),
    [
        ({"specialty": "невролог", "service_name": "Неврологу", "branch_name": "пр.Ленина, 5"}, []),
        ({"specialty": "терапевт"}, []),
        ({"service_name": "УЗИ брюшной полости", "branch_name": "пр.Ленина, 5"}, ["price_info"]),
    ],
    ids=["specialty_with_branch", "specialty_only", "service_booking"],
)
def test_price_step_only_for_service_booking(entities, tools):
    assert [s.tool for s in _appointment_price_steps("пр.Ленина, 5", entities)] == tools


def _book(word: str) -> str:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    state, memory = SessionState(session_id=f"book-{word}"), MemoryStore()

    async def say(text):
        out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
        return "".join(env.text for env in out if env.text)

    async def dialog():
        first = await say(f"хочу записаться к {word}")
        branch = next((line[2:].strip() for line in first.splitlines() if line.startswith("- ")), "пр.Ленина, 5")
        return await say(branch)

    return asyncio.run(dialog())


@pytest.mark.parametrize("word", ["терапевту", "неврологу", "гинекологу", "хирургу", "урологу", "педиатру"])
def test_booking_keeps_the_doctor_and_names_no_price(word):
    answer = _book(word)
    assert f"на приём к {word}" in answer
    assert "стоимость" not in answer
    for wrong in ("ТЭС", "УЗИ", "Невролиз", "стерилизац", "RIDA"):
        assert wrong not in answer
