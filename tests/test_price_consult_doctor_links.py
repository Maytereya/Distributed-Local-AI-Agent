"""Врачи в ответе о цене приёма — только те, кто оказывает ЭТУ услугу (30.09).

С 08.06 срез doctor_prices был пустым, и в ответе о цене приёма стояло «врачей не
нашёл» (BUG-2026-09-25-NO-DOCTORS-FOUND-FALSE). Когда срез починили, встречный свип
по 27 специальностям прайса вскрыл то, что пустота прятала: если специальность в
вопросе не распознана («рефлексотерапевта», «репродуктолога»), фильтра по
специальности нет, и в список шли любые врачи, у которых в прайсе есть какой-нибудь
«приём». На «приём рефлексотерапевта» бот называл колопроктолога и лимфолога с их
ценами — дезинформация ценой хуже, чем «не нашёл».

Инвариант `no_foreign_doctors_in_consult_price`: врач в списке либо оказывает ровно
эту услугу (тот же код услуги МИС, что у розничной строки), либо его специальность
распознана и совпала. Иначе — не показываем.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services import core as svc_mod

_SAMARA = "г. Самара, пр. Ленина, 5"

_DOCTORS = [
    {"id": 10, "fio": "Проктолог Иван", "ord": 1, "regions": [_SAMARA], "units": ["Врач-колопроктолог"]},
    {"id": 20, "fio": "Лимфолог Пётр", "ord": 2, "regions": [_SAMARA], "units": ["Врач-сердечно-сосудистый хирург, флеболог, лимфолог"]},
    {"id": 30, "fio": "Репродуктолог Анна", "ord": 3, "regions": [_SAMARA], "units": ["Врач ультразвуковой диагностики"]},
]

# У каждого врача в прайсе — свой приём со своим кодом. Ни один не рефлексотерапевт.
_DOCTOR_PRICES = [
    {"doctorId": 10, "fio": "Проктолог Иван", "serviceName": "Прием (осмотр, консультация) врача-колопроктолога", "serviceHomecode": "26.1.1", "cost": 3500},
    {"doctorId": 20, "fio": "Лимфолог Пётр", "serviceName": "Прием (осмотр, консультация) врача-сердечно-сосудистого хирурга", "serviceHomecode": "15.2.7.4", "cost": 2700},
    {"doctorId": 30, "fio": "Репродуктолог Анна", "serviceName": "Прием (осмотр, консультация) врача-репродуктолога", "serviceHomecode": "3.1.1.26", "cost": 3000},
]

_RETAIL = [
    {"serviceName": "Прием (осмотр, консультация) врача-рефлексотерапевта первичный — Заров В.Г.", "serviceHomecode": "16.1.1.9", "cost": 2000},
    {"serviceName": "Прием (осмотр, консультация) врача-репродуктолога", "serviceHomecode": "3.1.1.26", "cost": 3000},
]


def _bundle(monkeypatch, query: str) -> dict:
    svc = Services()

    async def fake_regions():
        return [{"id": 1, "addressForSite": _SAMARA, "city": "Самара"}]

    async def fake_doctors():
        return [dict(d) for d in _DOCTORS]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _region_id: [dict(r) for r in _RETAIL])
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", lambda: [dict(r) for r in _DOCTOR_PRICES])
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", lambda *args, **kwargs: [])
    return asyncio.run(svc.service_bundle_info(query, {}))


@pytest.mark.parametrize(
    "query, expected",
    [
        # Рефлексотерапевт в клинике есть, но цены «по врачу» у МИС нет ни у кого.
        ("сколько стоит приём рефлексотерапевта", []),
        # Точная связь по коду услуги: оказывает именно эту услугу.
        ("сколько стоит приём репродуктолога", ["Репродуктолог Анна"]),
    ],
)
def test_consult_price_lists_only_doctors_who_provide_this_service(monkeypatch, query, expected):
    payload = _bundle(monkeypatch, query)

    assert [d["fio"] for d in payload.get("doctors") or []] == expected


def test_doctor_who_provides_this_exact_service_is_not_cut_before_specialty_filter(monkeypatch):
    # Свип 30.09: «приём инфекциониста» → врачей 0, хотя по коду МИС приём
    # инфекциониста ведёт врач, у которой ОСНОВНОЕ подразделение — гепатолог.
    # Два дефекта: список обрезался до N по порядку сортировки ДО фильтра по
    # специальности (и нужного врача вытесняли терапевты со словом «приём»), а
    # фильтр по основной специальности отсекал врача с точной связью по коду.
    # С переводом на оператора при пустом списке каждый такой случай — ложный перевод.
    therapists = [
        {"id": 100 + i, "fio": f"Терапевт {i}", "ord": i, "regions": [_SAMARA], "main_units": ["Врач терапевт"], "units": ["Врач терапевт"]}
        for i in range(1, 6)
    ]
    infectionist = {
        "id": 200, "fio": "Инфекционист Анна", "ord": 99, "regions": [_SAMARA],
        "main_units": ["Врач-гепатолог"], "units": ["Врач-инфекционист", "Врач-гепатолог"],
    }
    doctor_prices = [
        {"doctorId": d["id"], "fio": d["fio"], "serviceName": "Прием (осмотр, консультация) врача-терапевта", "serviceHomecode": "13.1.1", "cost": 2700}
        for d in therapists
    ] + [
        {"doctorId": 200, "fio": "Инфекционист Анна", "serviceName": "Прием (осмотр, консультация) врача-инфекциониста", "serviceHomecode": "33.1.1", "cost": 2700},
    ]
    retail = [{"serviceName": "Прием (осмотр, консультация) врача-инфекциониста", "serviceHomecode": "33.1.1", "cost": 2700}]
    svc = Services()

    async def fake_regions():
        return [{"id": 1, "addressForSite": _SAMARA, "city": "Самара"}]

    async def fake_doctors():
        return [dict(d) for d in therapists + [infectionist]]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _region_id: [dict(r) for r in retail])
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", lambda: [dict(r) for r in doctor_prices])
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", lambda *args, **kwargs: [])

    payload = asyncio.run(svc.service_bundle_info("сколько стоит приём инфекциониста", {}))

    assert [d["fio"] for d in payload.get("doctors") or []] == ["Инфекционист Анна"]
    # Подпись — специальность из вопроса, которая у врача есть, а не основная
    # («Гепатолог»): у терапевта-гирудотерапевта в ответе про терапевта eval
    # запрещает «гирудотерап» (CRIT_PRICE_CONSULT_THERAPIST_NO_HIRUDO_001).
    assert payload["doctors"][0]["specialty_label"] == "Инфекционист"
