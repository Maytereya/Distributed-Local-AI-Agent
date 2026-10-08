"""Врачи по услуге — по связи в МИС, а «нет» только по данным (L-06; BUG-2026-10-05-FGDS-DOCTORS-NOT-FOUND).

«ФГДС» → «информация о враче не найдена»: текстовый фильтр искал «фгдс» в описании врача, где
написано «эзофагогастродуоденоскопия», и отсеивал всех эндоскопистов, хотя срез `doctor_prices`
связывает ФГДС (код 30.1.1.1) с врачами напрямую.

Инварианты класса:
- МИС связывает услугу с врачами (сильное совпадение названия) → бот не отвечает «не найден»;
- слабое совпадение («гастроскопия» ≈ «Гастрин») по коду НЕ соединяется — чужих врачей не показываем;
- фильтр по словам услуги опустошил непустой список специальности → оффер оператора, не «не найден»;
- врачи другого города по коду не появляются.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router import evidence_keys as ek
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.response_builder import DOCTORS_BY_SERVICE_NOT_FOUND_OFFER, build_doctor_info_response
from messengers_router.services import Services

def _doctor(doctor_id: int, fio: str, unit: str, text: str, region: str) -> dict:
    # Форма — как у настоящих записей среза врачей: роль приёма в units / unit_links.
    return {
        "id": doctor_id,
        "fio": fio,
        "specialization": text,
        "regions": [region],
        "units": [unit],
        "main_units": [unit],
        "unit_links": [{"company_unit_name": unit, "main": True, "specialization": text}],
        "main_specializations": [text],
    }


_DOCTORS = [
    _doctor(1, "Эндоскопов Иван Петрович", "Врач-эндоскопист", "• Эзофагогастродуоденоскопия\n• Колоноскопия", "пр.Ленина, 5"),
    _doctor(2, "Колоносков Пётр Ильич", "Врач-эндоскопист", "• Эзофагогастродуденоскопия (с опечаткой)", "ул. Победы, 83"),
    _doctor(3, "Терапевтова Анна Сергеевна", "Врач-терапевт", "Приём взрослых", "пр.Ленина, 5"),
    _doctor(4, "Чужой Город Врач", "Врач-эндоскопист", "• Эзофагогастродуоденоскопия", "г. Сызрань, ул. Кирова, 1"),
]
_DOCTOR_PRICES = [
    {"doctorId": 1, "serviceName": "Эзофагогастродуоденоскопия (ФГДС)", "serviceHomecode": "30.1.1.1", "cost": 5100.0},
    {"doctorId": 2, "serviceName": "Эзофагогастродуоденоскопия (ФГДС)", "serviceHomecode": "30.1.1.1", "cost": 5100.0},
    {"doctorId": 4, "serviceName": "Эзофагогастродуоденоскопия (ФГДС)", "serviceHomecode": "30.1.1.1", "cost": 4000.0},
    {"doctorId": 3, "serviceName": "Приём (осмотр, консультация) врача-терапевта первичный", "serviceHomecode": "1.1", "cost": 2700.0},
]


@pytest.fixture
def services(monkeypatch):
    svc = Services()
    svc.ensure_background_refresh_started = lambda: None

    async def doctors():
        return [dict(d) for d in _DOCTORS]

    async def samara_tokens():
        return {"ленина", "победы"}

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", doctors)
    monkeypatch.setattr(svc, "_samara_region_tokens", samara_tokens)
    monkeypatch.setattr(api_price, "load_doctor_prices", lambda *a, **k: list(_DOCTOR_PRICES))
    return svc


def _fio(payload):
    return sorted(d["fio"] for d in payload["doctors"])


@pytest.mark.parametrize(
    ("query", "entities"),
    [
        ("ФГДС", {"service_name": "фгдс"}),
        ("кто делает ФГДС", {}),
        ("Эзофагогастродуоденоскопия (ФГДС)", {"service_name": "Эзофагогастродуоденоскопия (ФГДС)"}),
        ("ФГДС", {"service_name": "фгдс", "specialty": "эндоскопист"}),
    ],
    ids=["abbr", "who_does", "full_name", "with_rule_specialty"],
)
def test_doctors_linked_by_service_code_are_found(services, query, entities):
    payload = asyncio.run(services.doctors_info(query, entities))

    # Оба самарских эндоскописта; врач другого города по коду не появляется.
    assert _fio(payload) == ["Колоносков Пётр Ильич", "Эндоскопов Иван Петрович"]


def test_weak_name_match_is_not_joined_by_code(services):
    # «гастроскопия» не входит словами в «Эзофагогастродуоденоскопия (ФГДС)» — соединять по коду
    # нельзя: слабое совпадение решает LLM-выбор (ход (в)), иначе были бы чужие врачи.
    payload = asyncio.run(services.doctors_info("гастроскопия", {"service_name": "гастроскопия"}))

    assert payload["doctors"] == []
    # Роль «эндоскопист» выводит правило «процедура → роль», эндоскописты есть — значит, это
    # не «врача нет», а повод спросить оператора (проверено ниже на ответе пациенту).
    assert payload["service_filter_emptied"] is True


def test_service_filter_emptying_specialty_list_offers_operator(services):
    payload = asyncio.run(
        services.doctors_info("гастроскопия", {"service_name": "гастроскопия", "specialty": "эндоскопист"})
    )
    assert payload["doctors"] == []
    assert payload["service_filter_emptied"] is True

    state, memory = SessionState(session_id="l06-offer"), MemoryStore()
    env = build_doctor_info_response("DOCTOR_INFO", Evidence(items={ek.DOCTORS_INFO: payload}), state, memory)

    assert env is not None and env.handoff is False
    assert env.text == DOCTORS_BY_SERVICE_NOT_FOUND_OFFER.format(service="гастроскопия")
    assert "не найдена" not in env.text
    assert state.last_entities.get("_operator_offer_pending") is True


def test_doctor_prices_unavailable_keeps_text_filter(services, monkeypatch):
    def broken(*_a, **_k):
        raise OSError("срез цен врачей недоступен")

    monkeypatch.setattr(api_price, "load_doctor_prices", broken)

    payload = asyncio.run(services.doctors_info("ФГДС", {"service_name": "фгдс"}))

    assert payload["doctors"] == []  # без данных МИС — не выдумываем; «нет» не утверждаем данными
