import asyncio

from messengers_router.mess_types import SessionState
from messengers_router import classifier as classifier_mod
from messengers_router.classifier import deterministic_rule_decision
from messengers_router.nlu_pipeline import analyze_with_candidates
from messengers_router.policies import missing_slots
from messengers_router.policies import appointment_summary
from messengers_router.policies import detect_clinic_hours_intent
from messengers_router.policies import detect_tax_doc_request_intent
from messengers_router.policies import detect_ambiguous_analysis_lookup
from messengers_router.policies import AMBIGUOUS_ANALYSIS_CLARIFY_TEXT
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


def test_procedure_urethroscopy_routes_to_urologist():
    state = SessionState(session_id="nlu_urethroscopy")
    out = run(analyze_with_candidates("какой врач выполняет уретроскопию", state))
    assert out.decision.label == "DOCTOR_INFO"
    assert out.decision.entities.get("specialty") == "уролог"


def test_procedure_anoscopy_routes_to_coloproctologist():
    state = SessionState(session_id="nlu_anoscopy")
    out = run(analyze_with_candidates("какой врач выполняет аноскопию", state))
    assert out.decision.label == "DOCTOR_INFO"
    assert out.decision.entities.get("specialty") == "колопроктолог"


def test_procedure_colposcopy_routes_to_obgyn():
    state = SessionState(session_id="nlu_colposcopy")
    out = run(analyze_with_candidates("кто делает кольпоскопию", state))
    assert out.decision.label == "DOCTOR_INFO"
    assert out.decision.entities.get("specialty") == "акушер-гинеколог"


def test_procedure_dermatoscopy_routes_to_dermatovenerologist():
    state = SessionState(session_id="nlu_dermatoscopy")
    out = run(analyze_with_candidates("какой врач делает дерматоскопию", state))
    assert out.decision.label == "DOCTOR_INFO"
    assert out.decision.entities.get("specialty") == "дерматовенеролог"


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


def test_price_renderer_shows_care_setting_and_address_when_present():
    payload = {
        "note": "price_info: priceByRegion(3)",
        "prices": [
            {
                "serviceName": "Тонзиллотомия",
                "cost": 20000.0,
                "care_setting_label": "дневной стационар",
                "care_setting_address": "г. Самара, пр. Ленина, 5",
            },
        ],
    }
    text = format_price_for_patient(payload, {"service_name": "Тонзиллотомия"})
    assert "Тонзиллотомия" in text
    assert "Формат: дневной стационар." in text
    assert "Адрес: г. Самара, пр. Ленина, 5." in text


def test_price_renderer_groups_family_variants_by_care_setting():
    payload = {
        "service_kind": "family_query",
        "service_name": "массаж",
        "family_variants": [
            {
                "serviceName": "Медицинский массаж 30 мин",
                "cost": 2500,
                "care_setting_label": "поликлиника",
                "care_setting_address": "г. Самара, пр. Ленина, 5",
            },
            {
                "serviceName": "Медицинский массаж 60 мин",
                "cost": 4000,
                "care_setting_label": "поликлиника",
                "care_setting_address": "г. Самара, пр. Ленина, 5",
            },
            {
                "serviceName": "Переливание эритроцитарной массы",
                "cost": 6000,
                "care_setting_label": "круглосуточный стационар",
                "care_setting_address": "г. Самара, ул. Ново-Садовая, 106, кор. 82",
            },
            {
                "serviceName": "Хромато масс-спектрометрия микробных маркеров",
                "cost": 4950,
            },
        ],
    }

    text = format_price_for_patient(payload, {"service_name": "массаж"})

    assert "В поликлинике по адресу: г. Самара, пр. Ленина, 5:" in text
    assert "В круглосуточном стационаре по адресу: г. Самара, ул. Ново-Садовая, 106, кор. 82:" in text
    assert "Другие варианты, формат и адрес нужно уточнить:" in text
    assert "1. Медицинский массаж 30 мин — 2 500 руб." in text
    assert "2. Медицинский массаж 60 мин — 4 000 руб." in text
    assert "1. Переливание эритроцитарной массы — 6 000 руб." in text
    assert "1. Хромато масс-спектрометрия микробных маркеров — 4 950 руб." in text
    assert "Медицинский массаж 30 мин — 2 500 руб.. Формат: поликлиника." not in text


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


def test_doctor_info_renderer_plural_followup_for_multiple_doctors():
    payload = {
        "doctors": [
            {"fio": "Иванов Иван Иванович", "specialization": "Кардиолог", "regions": ["Ленина 5"]},
            {"fio": "Петров Петр Петрович", "specialization": "Кардиолог", "regions": ["Ленина 5"]},
        ]
    }
    text = format_doctor_info_for_patient(payload, {"specialty": "кардиолог"})
    assert "Если нужно — могу показать расписание любого из этих врачей или помочь с записью." in text
    assert "этого врача" not in text


def test_doctor_schedule_renderer_russian_date_and_no_extra_clarifications_when_context_fixed():
    payload = {
        "schedule": [
            {
                "fio": "Хальметова Алина Алексеевна",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {
                    "г. Самара, пр. Ленина, 5": [
                        {"date": "2026-04-11", "slots": ["12:30", "13:30", "15:00"]},
                    ]
                },
            }
        ]
    }
    entities = {"doctor_name": "Хальметова", "branch_name": "г. Самара, пр. Ленина, 5"}
    text = format_doctor_schedule_for_patient(payload, entities)
    assert "• 11 апреля: свободно в 12:30, 13:30, 15:00" in text
    assert "Если нужно записаться — напишите удобное время." in text
    assert "уточните врача" not in text
    assert "уточните филиал" not in text


def test_doctor_schedule_renderer_asks_to_clarify_both_when_not_fixed():
    payload = {
        "schedule": [
            {
                "fio": "Иванов Иван Иванович",
                "regions": ["Ленина 5", "Победы 83"],
                "schedule": {
                    "Ленина 5": [{"date": "2026-04-11", "slots": ["10:00"]}],
                },
            },
            {
                "fio": "Петров Петр Петрович",
                "regions": ["Ленина 5", "Победы 83"],
                "schedule": {
                    "Победы 83": [{"date": "2026-04-11", "slots": ["11:00"]}],
                },
            },
        ]
    }
    text = format_doctor_schedule_for_patient(payload, {})
    assert "Если нужно записаться — напишите удобное время или уточните врача/филиал." in text


def test_appointment_summary_translates_relative_date_hints_to_russian():
    summary = appointment_summary(
        {
            "patient_name": "Петров Петр Петрович",
            "service_name": "Холтер",
            "branch_name": "ул. Победы, 83",
            "date_hint": "tomorrow",
            "time_from": "11:00",
        }
    )
    assert "tomorrow" not in summary
    assert ", завтра, 11:00." in summary


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


def test_deterministic_rule_decision_branch_hours_routes_to_address():
    out = run(
        deterministic_rule_decision(
            "Здравствуйте, вы завтра работаете?",
            {},
            allow_refine=False,
            attach_secondary=False,
        )
    )
    assert out is not None
    assert out.label == "ADDRESS"
    assert "rule_address_work_hours" in out.flags


def test_clinic_hours_intent_class_invariant():
    # КЛАСС: «график/режим работы» клиники/филиала без врача — это часы работы.
    clinic_hours = [
        "график работы филиалов",
        "график работы филиала на Победы 83",
        "режим работы филиалов",
        "во сколько работают филиалы",
        "график работы клиники",
        "вы завтра работаете?",
    ]
    for t in clinic_hours:
        assert detect_clinic_hours_intent(t) is True, t
    # НЕ клиника: расписание/график конкретного врача, цена, нейтральное.
    not_clinic_hours = [
        "расписание Трубина",
        "график работы врача Иванова",
        "когда принимает кардиолог",
        "сколько стоит приём врача",
        "запишите к кардиологу",
    ]
    for t in not_clinic_hours:
        assert detect_clinic_hours_intent(t) is False, t


def test_deterministic_rule_decision_grafik_filial_routes_to_address():
    # BUG-2026-06-04-01: голый «график» в SCHEDULE_PATTERNS перехватывал
    # «график работы филиалов» в DOCTOR_SCHEDULE без ФИО. Класс-инвариант:
    # любой «график/режим работы» клиники/филиала без врача → ADDRESS.
    for t in [
        "график работы филиалов",
        "график работы филиала на Победы 83",
        "режим работы филиалов",
        "график работы клиники",
    ]:
        out = run(
            deterministic_rule_decision(t, {}, allow_refine=False, attach_secondary=False)
        )
        assert out is not None
        assert out.label == "ADDRESS", f"{t!r} -> {out.label}"
        assert "rule_address_work_hours" in out.flags, t


def test_deterministic_rule_decision_doctor_schedule_not_stolen_by_clinic_hours():
    # Анти-over-trigger: запрос про график/расписание ВРАЧА остаётся DOCTOR_SCHEDULE.
    for t in [
        "расписание Трубина",
        "график работы врача Иванова",
        "расписание уролога Дразнина",
    ]:
        out = run(
            deterministic_rule_decision(t, {}, allow_refine=False, attach_secondary=False)
        )
        assert out is not None
        assert out.label == "DOCTOR_SCHEDULE", f"{t!r} -> {out.label}"


def test_ambiguous_analysis_lookup_class_invariant():
    # КЛАСС: «узнать анализы» (глагол «узнать» + «анализ», без уточнения) —
    # двусмысленно (результаты vs где сдать) → нужен уточняющий вопрос.
    ambiguous = [
        "узнать анализы",
        "хочу узнать анализы",
        "узнать свои анализы",
        "узнать про анализы",
    ]
    for t in ambiguous:
        assert detect_ambiguous_analysis_lookup(t) is True, t
    # НЕ двусмысленно: явный сигнал снимает неопределённость.
    not_ambiguous = [
        "результаты анализов",          # результаты → TEST_RESULT
        "узнать результаты анализов",    # результаты → TEST_RESULT
        "сдать анализы",                 # сдать → ADDRESS (живая очередь)
        "где сдать анализы",             # где/сдать → ADDRESS
        "какие анализы сдать",           # какие → TEST_ASSIST
        "узнать стоимость анализов",     # цена → PRICE
        "записаться к кардиологу",       # вообще не про анализы
    ]
    for t in not_ambiguous:
        assert detect_ambiguous_analysis_lookup(t) is False, t


def test_deterministic_rule_decision_ambiguous_analysis_clarifies():
    # BUG-2026-06-04-02: «узнать анализы» уходило в ADDRESS (живая очередь,
    # список филиалов) из-за шортката ≤2 слов в walk-in. Теперь — уточняющий
    # вопрос вместо угадывания.
    for t in ["узнать анализы", "хочу узнать анализы", "узнать свои анализы"]:
        out = run(
            deterministic_rule_decision(t, {}, allow_refine=False, attach_secondary=False)
        )
        assert out is not None
        assert out.clarify_needed is True, t
        assert out.clarify_reason == AMBIGUOUS_ANALYSIS_CLARIFY_TEXT, t
        assert "rule_analysis_lookup_ambiguous" in out.flags, t
        assert out.needs_handoff is False, t
    # Анти-регресс: однозначные запросы про анализы НЕ уводятся в clarify.
    explicit = {
        "результаты анализов": "TEST_RESULT",
        "сдать анализы": "ADDRESS",
        "где сдать анализы": "ADDRESS",
    }
    for t, label in explicit.items():
        out = run(
            deterministic_rule_decision(t, {}, allow_refine=False, attach_secondary=False)
        )
        assert out is not None
        assert out.clarify_needed is False, t
        assert out.label == label, f"{t!r} -> {out.label}"


def test_tax_doc_request_reverse_word_order():
    # «налоговая справка» (обратный порядок слов) = справка для вычета → tax,
    # чтобы отдать детерминированный tax-ответ, а не зависеть от meili-поиска.
    for t in ["налоговая справка", "нужна налоговая справка", "налоговую справку"]:
        assert detect_tax_doc_request_intent(t) is True, t
        out = run(
            deterministic_rule_decision(t, {}, allow_refine=False, attach_secondary=False)
        )
        assert out is not None
        assert out.entities.get("doc_request_kind") == "tax", t
        assert "doc_request_tax" in out.flags, t


def test_deterministic_rule_decision_service_location_query_routes_to_address():
    out = run(
        deterministic_rule_decision(
            "Где можно сделать септопластику?",
            {},
            allow_refine=False,
            attach_secondary=False,
        )
    )
    assert out is not None
    assert out.label == "ADDRESS"
    assert "rule_address" in out.flags
    assert "септопласт" in str(out.entities.get("service_name", "")).lower()


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


def test_service_bundle_renderer_consultation_hides_prepare_and_internal_ord():
    payload = {
        "service_name": "Прием (осмотр, консультация) врача-уролога первичный",
        "retail_prices": [{"serviceName": "Прием (осмотр, консультация) врача-уролога первичный", "cost": 2000}],
        "doctors": [
            {
                "fio": "Иванов Иван Иванович",
                "ord": 1,
                "service_price": 1900,
                "available": False,
                "availability_note": "availability_unmatched",
            }
        ],
        "prepare": "Служебный блок не должен выводиться для консультации",
    }
    text = format_service_bundle_for_patient(payload, {})
    assert "ord=" not in text
    assert "Подготовка:" not in text


def test_service_bundle_renderer_hides_prepare_without_explicit_flag():
    payload = {
        "service_name": "УЗИ брюшной полости",
        "retail_prices": [{"serviceName": "УЗИ брюшной полости", "cost": 1500}],
        "doctors": [],
        "prepare": "Натощак 6 часов.",
        "show_prepare": False,
    }
    text = format_service_bundle_for_patient(payload, {})
    assert "Подготовка:" not in text


def test_service_bundle_renderer_no_doctors_uses_operator_handoff_hint():
    payload = {
        "service_name": "Шунтирование желудка",
        "retail_prices": [{"serviceName": "Шунтирование желудка", "cost": 199000}],
        "doctors": [],
        "show_prepare": False,
    }
    text = format_service_bundle_for_patient(payload, {})
    assert "передам запрос оператору" in text.lower()
    assert "подробное расписание выбранного врача" not in text.lower()


def test_service_bundle_renderer_lab_hides_doctor_schedule_suffix():
    payload = {
        "service_name": "Общий анализ крови",
        "service_kind": "lab",
        "retail_prices": [{"serviceName": "Общий анализ крови", "cost": 490}],
        "doctors": [],
        "show_prepare": False,
    }
    text = format_service_bundle_for_patient(payload, {})
    low = text.lower()
    assert "запись к конкретному врачу обычно не требуется" in low
    assert "подробное расписание" not in low


def test_service_bundle_renderer_shows_doctor_specialty_label():
    payload = {
        "service_name": "Прием (осмотр, консультация) врача-хирурга первичный",
        "retail_prices": [{"serviceName": "Прием (осмотр, консультация) врача-хирурга первичный", "cost": 2500}],
        "doctors": [
            {
                "fio": "Дурасов Владимир Владимирович",
                "specialty_label": "Хирург",
                "service_price": 2500,
                "available": False,
                "availability_note": "availability_unmatched",
            }
        ],
    }

    text = format_service_bundle_for_patient(payload, {})

    assert "Дурасов Владимир Владимирович (Хирург)" in text
