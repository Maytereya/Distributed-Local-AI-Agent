"""Кейс 08.07 «УЗИ органов мошонки»: простыня всех УЗИ-врачей вместо процедуры.

Прод-диалог: пациент спросил «УЗИ органов мошонки» → бот отдал ВСЕХ 4 УЗИ-врачей
с полными CRM-описаниями (~12К символов, телеграм резал на 2 сообщения), включая
врачей, НЕ делающих процедуру; «Кого рекомендуешь» повторял тот же список до
срабатывания анти-луп защиты.

Инварианты (класс):
1. Роутинг: специальность + процедурная конкретика («УЗИ органов мошонки») →
   DOCTOR_INFO с service_name (включает фильтр по процедуре), а не только
   specialty. Голая специальность («невролог») — прежнее поведение.
2. Фильтр: врачи без процедуры в описании не попадают в выдачу.
3. Компактность: в СПИСКЕ описание врача обрезано (~4 строки); полное описание —
   только при запросе конкретного врача.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.classifier import deterministic_rule_decision
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


# --- 1. роутинг ------------------------------------------------------------------

def test_procedure_detail_routes_with_service_name():
    d = run(deterministic_rule_decision("УЗИ органов мошонки", {}))
    assert d is not None and d.label == "DOCTOR_INFO"
    assert d.entities.get("specialty") == "узи"
    assert "мошонки" in str(d.entities.get("service_name") or "").lower()
    assert "rule_doctor_info_procedure_detail" in d.flags


@pytest.mark.parametrize("text,spec", [("невролог", "невролог"), ("уролог", "уролог")])
def test_bare_specialty_unchanged(text, spec):
    d = run(deterministic_rule_decision(text, {}))
    assert d is not None and d.label == "DOCTOR_INFO"
    assert d.entities.get("specialty") == spec
    assert "service_name" not in d.entities, "голая специальность — без service-фильтра"


@pytest.mark.parametrize("text", [
    "Скажите, кто из кардиологов принимает и по какому адресу?",  # eval-регресс 13.07
    "какие неврологи принимают",
    "кто из хирургов ведёт приём",
])
def test_non_modality_specialty_never_gets_noise_service(text):
    """Регресс eval 13.07: у НЕ-диагностической специальности процедурная ветка
    не должна цеплять шум _extract_service_keyword («Какому адресу») —
    иначе _doctor_matches_service отсекает всех врачей специальности."""
    d = run(deterministic_rule_decision(text, {}))
    assert d is not None and d.label == "DOCTOR_INFO"
    assert "service_name" not in d.entities, f"ложный service_name: {d.entities.get('service_name')!r}"
    assert "rule_doctor_info_procedure_detail" not in d.flags


def test_modality_specialty_keeps_procedure_detail():
    """Другая модальность (ЭКГ/УЗИ) с конкретикой — детализация сохраняется."""
    d = run(deterministic_rule_decision("узи брюшной полости", {}))
    assert d is not None and d.entities.get("specialty") == "узи"
    assert "брюшной" in str(d.entities.get("service_name") or "").lower()


# --- 2-3. фильтр + компактность ----------------------------------------------------

_LONG_DESC_WITH = (
    "• УЗИ брюшной полости:\n- печени\n- желчного пузыря\n"
    "• УЗИ мочеполовой системы:\n- почек\n- предстательной железы и органов мошонки (с ЦДК сосудов)\n"
    "• Гинекологическое УЗИ:\n- органов малого таза\n" + "\n".join(f"- пункт {i}" for i in range(20))
)
_LONG_DESC_WITHOUT = (
    # «органов малого таза» — приманка для generic-матча: живой кейс 08.07 —
    # Ларионова матчилась на «УЗИ ОРГАНОВ мошонки» по «узи»+«органов»,
    # хотя мошонку не делает (различающий токен обязателен).
    "• УЗИ брюшной полости:\n- печени\n- селезенки\n"
    "• Гинекологическое УЗИ:\n- органов малого таза\n"
    "• УЗИ беременных:\n- при сроке до 11 недель\n" + "\n".join(f"- раздел {i}" for i in range(20))
)

_FAKE_DOCTORS = [
    {"id": 1, "fio": "Казакова Ирина Михайловна", "ord": 1, "specialization": _LONG_DESC_WITH,
     "regions": ["пр.Ленина, 5"], "units": ["Врач ультразвуковой диагностики"]},
    {"id": 2, "fio": "Ларионова Диана Александровна", "ord": 2, "specialization": _LONG_DESC_WITHOUT,
     "regions": ["пр.Ленина, 5"], "units": ["Врач ультразвуковой диагностики"]},
    {"id": 3, "fio": "Портянникова Наталия Петровна", "ord": 3, "specialization": _LONG_DESC_WITH,
     "regions": ["пр.Ленина, 5"], "units": ["Врач ультразвуковой диагностики"]},
    {"id": 4, "fio": "Свиридова Елена Александровна", "ord": 4, "specialization": _LONG_DESC_WITHOUT,
     "regions": ["пр.Ленина, 5"], "units": ["Врач ультразвуковой диагностики"]},
]


@pytest.fixture()
def svc(monkeypatch):
    service = Services()

    async def fake_cache():
        return [dict(d) for d in _FAKE_DOCTORS]

    async def fake_tokens():
        return set()  # без samara-фильтрации в этом тесте

    monkeypatch.setattr(service, "_ensure_doctors_cache_loaded", fake_cache)
    monkeypatch.setattr(service, "_samara_region_tokens", fake_tokens)
    return service


def test_service_filter_keeps_only_matching_doctors(svc):
    p = run(svc.doctors_info("УЗИ органов мошонки", {"specialty": "узи", "service_name": "УЗИ органов мошонки"}))
    fios = [d["fio"] for d in p["doctors"]]
    assert "Казакова Ирина Михайловна" in fios
    assert "Портянникова Наталия Петровна" in fios
    assert "Ларионова Диана Александровна" not in fios, "не делает процедуру — не показываем"
    assert "Свиридова Елена Александровна" not in fios


def test_list_mode_descriptions_are_compact(svc):
    p = run(svc.doctors_info("узи", {"specialty": "узи"}))
    docs = p["doctors"]
    assert len(docs) >= 2
    for d in docs:
        assert len(d["specialization"]) <= 300, f"{d['fio']}: описание в списке не обрезано"


def test_specific_doctor_keeps_full_description(svc):
    p = run(svc.doctors_info("Казакова", {"doctor_name": "Казакова"}))
    docs = p["doctors"]
    assert len(docs) == 1
    assert len(docs[0]["specialization"]) > 300, "конкретный врач — полное описание"


def test_full_pipeline_filters_and_compacts(svc):
    """End-to-end регресс: service_name обязан ДОЖИТЬ до doctors_info через
    граундер (DOCTOR_INFO-whitelist молча дропал его — из-за этого фильтр
    не включался в проде, хотя правило сущность передавало)."""
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import SessionState
    from messengers_router.orchestrator import run_pipeline

    state = SessionState(session_id="uzi-pipeline")
    ctx = run(run_pipeline("УЗИ органов мошонки", state, services=svc, memory=MemoryStore()))
    text = ctx.response.text if ctx.response else ""
    assert "Казакова" in text
    assert "Портянникова" in text
    assert "Ларионова" not in text, "врач без процедуры в выдаче — фильтр не дожил до сервиса"
    assert len(text) < 2500, f"ответ не компактен: {len(text)} симв."
