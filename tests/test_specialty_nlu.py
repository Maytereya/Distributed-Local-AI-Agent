import asyncio

from messengers_router.mess_types import SessionState
from messengers_router.classifier import deterministic_rule_decision
from messengers_router.nlu_pipeline import analyze_with_candidates
from messengers_router.policies import missing_slots
from messengers_router.renderer import format_address_for_patient, format_doctor_schedule_for_patient, format_price_for_patient
from messengers_router.router import _is_samara_city


def run(coro):
    return asyncio.run(coro)


def test_specialty_plural_routes_to_doctor_info():
    state = SessionState(session_id="nlu1")
    out = run(analyze_with_candidates("урологи клиники", state))
    assert out.decision.label == "DOCTOR_INFO"
    assert out.decision.entities.get("specialty") == "уролог"


def test_nearest_specialty_routes_to_schedule():
    state = SessionState(session_id="nlu2")
    out = run(analyze_with_candidates("гастроэнтеролог ближайший", state))
    assert out.decision.label == "DOCTOR_SCHEDULE"
    assert out.decision.entities.get("specialty") == "гастроэнтеролог"


def test_diagnostic_make_request_routes_to_appointment():
    state = SessionState(session_id="nlu3")
    out = run(analyze_with_candidates("Хочу сделать узи брюшной полости", state))
    assert out.decision.label == "APPOINTMENT"


def test_doctor_schedule_specialty_satisfies_required_slots():
    miss = missing_slots("DOCTOR_SCHEDULE", {"specialty": "гастроэнтеролог"})
    assert miss == []


def test_samara_city_helper():
    assert _is_samara_city("Самара") is True
    assert _is_samara_city("Оренбург") is False


def test_address_renderer_has_no_nearest_hint_tail():
    payload = {
        "branches": [
            {"address": "г.Самара, пр.Ленина, 5", "phone": "+7 000 000 00 00", "work_time": "ПН-ПТ 7.00-20.00"}
        ]
    }
    text = format_address_for_patient(payload, {"city": "Самара"})
    assert "Если нужно, подскажу ближайший филиал" not in text


def test_doctor_schedule_renderer_explains_visible_schedule_branches():
    payload = {
        "schedule": [
            {
                "fio": "Трубин Алексей Юрьевич",
                "specialization": "Уролог",
                "regions": ["Ленина 5", "Ново-Садовая 106 к.82"],
                "schedule": {
                    "Ленина 5": [{"date": "2026-03-03", "slots": ["13:00"]}],
                },
            }
        ]
    }
    text = format_doctor_schedule_for_patient(payload, {})
    assert "Адреса приема: Ленина 5, Ново-Садовая 106 к.82" in text
    assert "Ближайшее актуальное расписание сейчас есть в филиале: Ленина 5" in text


def test_price_renderer_returns_cost_for_single_match():
    payload = {
        "note": "price_info: priceByRegion(3)",
        "prices": [
            {"serviceName": "УЗИ брюшной полости", "cost": 1500.0},
        ]
    }
    text = format_price_for_patient(payload, {"service_name": "УЗИ брюшной полости"})
    assert "УЗИ брюшной полости" in text
    assert "1 500 руб." in text
    assert "Источник цены: розничный прайс Самары." in text


def test_price_renderer_doctor_context_single_match():
    payload = {
        "note": "price_info: doctorServicePricesByRegion (branch-level regionId)",
        "prices": [
            {
                "serviceName": "УЗИ брюшной полости",
                "cost": 1500.0,
                "fio": "Иванов Иван Иванович",
                "regionName": "г. Самара, пр. Ленина, 5",
            },
        ]
    }
    text = format_price_for_patient(payload, {"doctor_name": "Иванов", "service_name": "УЗИ брюшной полости"})
    assert "У врача Иванов" in text
    assert "УЗИ брюшной полости" in text
    assert "1 500 руб." in text
    assert "Источник цены: врачебный прайс." in text


def test_price_renderer_handles_empty_result():
    payload = {"prices": []}
    text = format_price_for_patient(payload, {"service_name": "УЗИ брюшной полости"})
    assert "Не нашёл актуальную стоимость" in text


def test_deterministic_rule_decision_price():
    out = run(
        deterministic_rule_decision(
            "Сколько стоит ЭКГ в Самаре?",
            {},
            allow_refine=False,
            attach_secondary=False,
        )
    )
    assert out is not None
    assert out.label == "PRICE"


def test_deterministic_rule_decision_price_with_doctor_name():
    out = run(
        deterministic_rule_decision(
            "Сколько стоит УЗИ брюшной полости у Иванова?",
            {},
            allow_refine=False,
            attach_secondary=False,
        )
    )
    assert out is not None
    assert out.label == "PRICE"
    assert out.entities.get("doctor_name")


def test_price_slots_are_not_required_when_doctor_is_known():
    miss = missing_slots("PRICE", {"doctor_name": "Иванов"})
    assert miss == []
