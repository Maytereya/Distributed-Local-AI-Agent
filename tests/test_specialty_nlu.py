import asyncio

from messengers_router.mess_types import SessionState
from messengers_router import classifier as classifier_mod
from messengers_router.classifier import deterministic_rule_decision
from messengers_router.nlu_pipeline import analyze_with_candidates
from messengers_router.policies import missing_slots
from messengers_router.renderer import (
    format_address_for_patient,
    format_doctor_info_for_patient,
    format_doctor_schedule_for_patient,
    format_price_for_patient,
    format_service_bundle_for_patient,
)
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


def test_doctor_schedule_renderer_compact_header_without_specialization_dump():
    payload = {
        "schedule": [
            {
                "fio": "Дразнин Антон Владимирович",
                "specialization": "Очень длинное описание специализации",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {
                    "г. Самара, пр. Ленина, 5": [{"date": "2026-03-23", "slots": ["15:30"]}],
                },
            }
        ]
    }
    text = format_doctor_schedule_for_patient(payload, {})
    first_line = text.splitlines()[0]
    assert first_line == "1. Дразнин Антон Владимирович"
    assert "Очень длинное описание" not in text


def test_doctor_schedule_renderer_no_free_slots_uses_alternative_cta():
    payload = {
        "schedule": [
            {
                "fio": "Вахобов Абдуджалол Нозимович",
                "specialization": "Уролог",
                "regions": ["г.Самара, ул.Ново-Садовая, 106, кор. 82"],
                "schedule": {
                    "г.Самара, ул.Ново-Садовая, 106, кор. 82": [
                        {"date": "2026-03-23", "start": "09:00:00", "end": "13:00:00", "slots": []}
                    ],
                },
            }
        ]
    }
    text = format_doctor_schedule_for_patient(payload, {})
    assert "Свободных окон в ближайшие дни не найдено." in text
    assert "Могу подобрать другого врача или передать диалог оператору." in text
    assert "Если нужно записаться —" not in text


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


def test_doctor_info_renderer_single_selected_doctor_compact_followup():
    payload = {
        "doctors": [
            {
                "fio": "Вахобов Абдуджалол Нозимович",
                "specialization": "Длинный блок услуг врача",
                "regions": ["г.Самара, ул.Ново-Садовая, 106, кор. 82"],
            }
        ]
    }
    text = format_doctor_info_for_patient(payload, {"doctor_name": "Вахобов"})
    assert "1. Вахобов Абдуджалол Нозимович" in text
    assert "Адреса приема: г.Самара, ул.Ново-Садовая, 106, кор. 82" in text
    assert "Длинный блок услуг врача" not in text
    assert "Хотите записаться к этому врачу? Напишите «расписание» или «запись»." in text


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


def test_deterministic_rule_decision_price_with_doctor_name(monkeypatch):
    monkeypatch.setattr(
        classifier_mod,
        "resolve_cached_doctor_name_candidate",
        lambda text, prefer_schedule=False: "Иванова" if "иванов" in text.lower() else None,
    )

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


def test_deterministic_rule_decision_doc_request_uses_main_index_flag():
    out = run(
        deterministic_rule_decision(
            "Как получить справку для налоговой?",
            {},
            allow_refine=False,
            attach_secondary=False,
        )
    )
    assert out is not None
    assert out.label == "OTHER"
    assert "doc_request_main_index" in out.flags
    assert out.needs_handoff is False


def test_deterministic_rule_decision_prepare_question_routes_to_prepare():
    out = run(
        deterministic_rule_decision(
            "Как подготовиться к анализу крови?",
            {},
            allow_refine=False,
            attach_secondary=False,
        )
    )
    assert out is not None
    assert out.label == "PREPARE"
    assert "rule_prepare" in out.flags


def test_price_slots_are_not_required_when_doctor_is_known():
    miss = missing_slots("PRICE", {"doctor_name": "Иванов"})
    assert miss == []


def test_service_bundle_renderer_availability_checked_with_nearest_slot():
    payload = {
        "service_name": "УЗИ брюшной полости",
        "retail_prices": [{"serviceName": "УЗИ брюшной полости", "cost": 1500}],
        "doctors": [
            {
                "fio": "Иванов Иван Иванович",
                "ord": 10,
                "service_price": 1400,
                "available": True,
                "nearest_slot": "2026-03-14T09:30",
                "availability_note": "availability_checked",
            }
        ],
        "prepare": "Натощак 6 часов.",
    }
    text = format_service_bundle_for_patient(payload, {})
    assert "доступен, ближайшее окно 14.03 09:30" in text


def test_service_bundle_renderer_availability_checked_without_slots():
    payload = {
        "service_name": "УЗИ брюшной полости",
        "doctors": [
            {
                "fio": "Иванов Иван Иванович",
                "service_price": 1400,
                "available": False,
                "nearest_slot": "",
                "availability_note": "availability_checked",
            }
        ],
    }
    text = format_service_bundle_for_patient(payload, {})
    assert "сейчас без свободных окон" in text


def test_service_bundle_renderer_availability_source_unavailable():
    payload = {
        "service_name": "УЗИ брюшной полости",
        "doctors": [
            {
                "fio": "Иванов Иван Иванович",
                "service_price": 1400,
                "available": False,
                "availability_note": "availability_source_unavailable",
            }
        ],
    }
    text = format_service_bundle_for_patient(payload, {})
    assert "расписание временно недоступно" in text
