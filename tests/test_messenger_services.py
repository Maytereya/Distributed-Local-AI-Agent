import asyncio
import importlib

import pytest

from messengers_router import llm_runtime as llm_runtime_mod
from messengers_router.services import _common as _common_mod
from messengers_router.services import lab_tests as lab_tests_mod
from messengers_router.services._common import _normalise_input as _svc_normalise_input
from messengers_router.services._doctors_helpers import (
    _extract_specialty_from_text as _svc_extract_specialty_from_text,
    _select_effective_price_service_name as _svc_select_effective_price_service_name,
    _service_name_matches_specialty as _svc_service_name_matches_specialty,
)
from messengers_router.services._prepare import (
    _prepare_relevance_gate as _svc_prepare_relevance_gate,
    _prepare_roots_match as _svc_prepare_roots_match,
    _prepare_subject_hint as _svc_prepare_subject_hint,
)
from messengers_router.services._prices_helpers import (
    _extract_price_service_from_query as _svc_extract_price_service_from_query,
    _resolve_ambiguous_price_kind_with_llm as _svc_resolve_ambiguous_price_kind_with_llm,
)
from messengers_router import classifier as classifier_mod
from messengers_router import policies as policies_mod
from messengers_router import services as svc_mod
from messengers_router.city import match_city
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.renderer import format_price_for_patient, format_service_bundle_for_patient
from messengers_router.response_builder import build_price_response
from messengers_router.services import Services, resolve_price_service_name_from_catalog
from messengers_router.policies import (
    build_branch_index,
    extract_specialty,
    extract_service_phrase,
    handoff_message,
    match_branch_hint,
    quick_fill_core_entities,
)


def run(coro):
    return asyncio.run(coro)


def test_services_package_facade_exports_legacy_api():
    assert svc_mod.Services is Services
    assert svc_mod.resolve_price_service_name_from_catalog is resolve_price_service_name_from_catalog


def test_services_placeholder_submodules_are_importable():
    for module_name in (
        "messengers_router.services.doctors",
        "messengers_router.services.appointments",
        "messengers_router.services.lab_tests",
        "messengers_router.services.addresses",
        "messengers_router.services.prices",
    ):
        module = importlib.import_module(module_name)
        assert module is not None


def test_services_doctor_methods_are_sourced_from_doctors_module():
    assert Services.resolve_doctor_name.__module__ == "messengers_router.services.doctors"
    assert Services.doctors_info.__module__ == "messengers_router.services.doctors"
    assert Services.doctors_schedule_week.__module__ == "messengers_router.services.doctors"


def test_services_lab_methods_are_sourced_from_lab_tests_module():
    assert Services.test_assist.__module__ == "messengers_router.services.lab_tests"
    assert Services.test_result_status.__module__ == "messengers_router.services.lab_tests"


def test_services_normalise_input_normalizes_yo_characters():
    assert _svc_normalise_input("  Ёжик   в Тумане  ") == "ежик в тумане"


def test_lab_tests_extract_result_query_fields_uses_order_id_fallback():
    fields = lab_tests_mod._extract_result_query_fields(
        {
            "result_surname": "Иванов",
            "result_filial": "Бг",
            "year": "1990",
            "order_id": "12345",
        },
        "результат анализа",
    )

    assert fields == {
        "surname": "Иванов",
        "year": 1990,
        "filial": "Бг",
        "number": 12345,
        "lang": "ru",
    }


def test_lab_tests_build_public_result_link_keeps_cp1251_contract():
    link = lab_tests_mod._build_public_result_link(
        {
            "surname": "Иванов",
            "year": 1990,
            "filial": "Бг",
            "number": 12345,
            "lang": "ru",
        }
    )

    assert link == (
        "https://naykalab.ru/getanaliz.php"
        "?fam=%C8%E2%E0%ED%EE%E2&year=1990&nom=%C1%E3&nom2=12345&fast=1"
    )


def test_lab_tests_test_assist_clarify_response_stays_non_handoff():
    res = lab_tests_mod._test_assist_clarify_response({"test_goal": "щитовидка"}, note="stage8")

    assert res.get("handoff_required") is not True
    assert res["tests"] == []
    assert "подобрать анализы" in str(res.get("message") or "").lower()
    assert res["note"] == "stage8"


def test_doctors_info_filters_by_name(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван",
                "specialization": "терапевт",
                "regions": ["Ленина 5"],
                "units": ["Терапия"],
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)

    res = run(svc.doctors_info("Иванов", {"doctor_name": "Иванов"}))

    assert res["doctors"], "Expected matched doctors"
    assert res["doctors"][0]["fio"] == "Иванов Иван"
    assert not res.get("handoff_required", False)


def test_doctors_info_exact_surname_does_not_pull_female_variant(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван Иванович",
                "specialization": "Онколог",
                "regions": ["Ленина 5"],
                "units": ["Онкология"],
            },
            {
                "id": 2,
                "fio": "Иванова Анна Сергеевна",
                "specialization": "Терапевт",
                "regions": ["Ленина 5"],
                "units": ["Терапия"],
            },
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)

    res = run(svc.doctors_info("Иванов", {"doctor_name": "Иванов"}))

    assert [row["fio"] for row in res["doctors"]] == ["Иванов Иван Иванович"]


def test_doctors_info_sorts_by_ord(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Врач Второй",
                "ord": 20,
                "specialization": "терапевт",
                "regions": ["Ленина 5"],
                "units": ["Терапевт"],
            },
            {
                "id": 2,
                "fio": "Врач Первый",
                "ord": 5,
                "specialization": "терапевт",
                "regions": ["Ленина 5"],
                "units": ["Терапевт"],
            },
        ]

    async def fake_samara_tokens():
        return set()

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)

    res = run(svc.doctors_info("терапевт", {"specialty": "терапевт"}))

    assert [row["fio"] for row in res["doctors"][:2]] == ["Врач Первый", "Врач Второй"]


def test_doctors_info_procedure_query_falls_back_to_specialty_when_service_text_missing(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Уролог Первый",
                "ord": 2,
                "specialization": "Уролог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Врач-уролог"],
                "unit_links": [{"company_unit_name": "Врач-уролог", "main": True, "specialization": "Уролог"}],
                "main_units": ["Врач-уролог"],
            },
            {
                "id": 2,
                "fio": "Уролог Второй",
                "ord": 5,
                "specialization": "Уролог",
                "regions": ["г. Самара, ул. Победы, 83"],
                "units": ["Врач-уролог"],
                "unit_links": [{"company_unit_name": "Врач-уролог", "main": True, "specialization": "Уролог"}],
                "main_units": ["Врач-уролог"],
            },
            {
                "id": 3,
                "fio": "Хирург Лишний",
                "ord": 1,
                "specialization": "Хирург",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Врач-хирург"],
                "unit_links": [{"company_unit_name": "Врач-хирург", "main": True, "specialization": "Хирург"}],
                "main_units": ["Врач-хирург"],
            },
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5", "г. самара, ул. победы, 83"}

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)

    res = run(
        svc.doctors_info(
            "какой врач выполняет уретроскопию",
            {"service_name": "Уретроскопию", "specialty": "уролог"},
        )
    )

    assert [row["fio"] for row in res["doctors"]] == ["Уролог Первый", "Уролог Второй"]


def test_doctors_info_uzi_query_filters_to_real_uzi_doctors(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Настоящий Узист",
                "specialization": "Врач ультразвуковой диагностики\nУЗИ брюшной полости\nУЗИ щитовидной железы",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["УЗИ"],
            },
            {
                "id": 2,
                "fio": "Терапевт С Упоминанием Узи",
                "specialization": "Терапевт\nРазъяснение в рамках приема данных исследований (рентген, УЗИ, эндоскопия)",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Терапия"],
            },
            {
                "id": 3,
                "fio": "Арцыбашева Олеся Сергеевна",
                "specialization": "Онколог\nУЗИ молочных желез во время консультативного приема",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Онкология"],
            },
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)

    res = run(svc.doctors_info("Врач узи", {}))

    names = [str(x.get("fio") or "") for x in res["doctors"]]
    assert "Настоящий Узист" in names
    assert "Терапевт С Упоминанием Узи" not in names
    assert "Арцыбашева Олеся Сергеевна" not in names


def test_doctors_info_excludes_explicit_non_samara_doctors(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Самарский Онколог",
                "ord": 2,
                "specialization": "онколог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Онкология"],
            },
            {
                "id": 2,
                "fio": "Губский Иван Иванович",
                "ord": 1,
                "specialization": "онколог",
                "regions": ["г. Оренбург, ул. Пушкинская, 10"],
                "units": ["Онкология"],
            },
        ]

    async def fake_samara_tokens():
        # Имитируем недоступность live /regions: фильтр должен все равно отсечь Оренбург.
        return set()

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)

    res = run(svc.doctors_info("онколог", {"specialty": "онколог"}))
    names = [str(x.get("fio") or "") for x in res["doctors"]]
    assert "Самарский Онколог" in names
    assert "Губский Иван Иванович" not in names


def test_doctors_info_empty_cache_returns_fallback(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return []

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)

    res = run(svc.doctors_info("Петров", {}))

    assert res.get("handoff_required") is True
    assert res.get("doctors") == []
    assert res.get("handoff_reason") == "service_error"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("ambiguous_price_service", "Сейчас по этой услуге безопаснее уточнить у оператора. Соединяю с оператором."),
        ("city_not_supported", "Сейчас могу помочь только по Самаре. Соединяю с оператором."),
        ("knowledge_not_found", "В моей базе данных информации недостаточно, перевожу на оператора."),
        ("service_error_doctors_list", "Сейчас не удалось получить список врачей автоматически. Соединяю с оператором."),
        ("service_error_schedule", "Сейчас не удалось получить расписание автоматически. Соединяю с оператором."),
        ("service_error_doctor_info", "Сейчас не удалось найти информацию автоматически. Соединяю с оператором."),
        ("service_error_appointments", "Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором."),
        ("service_error_results", "Сейчас не удалось получить результаты автоматически. Соединяю с оператором."),
        ("service_error_result_link", "Сейчас не удалось сформировать ссылку на результат автоматически. Соединяю с оператором."),
        ("service_error_prices", "Сейчас не удалось получить цены автоматически. Соединяю с оператором."),
    ],
)
def test_handoff_message_supports_domain_specific_service_texts(reason, expected):
    assert handoff_message(reason) == expected


def test_doctors_info_empty_cache_uses_domain_specific_handoff_message(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return []

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)

    res = run(svc.doctors_info("Петров", {}))

    assert res.get("handoff_message") == handoff_message("service_error_doctors_list")


@pytest.mark.parametrize(
    "query",
    [
        "расписание флюорографии",
        "когда можно сделать маммографию",
        "запишите на флюорографию",
        "расписание маммографа",
    ],
)
def test_doctors_schedule_week_diagnostic_fixed_equipment_handoff(monkeypatch, query):
    """Запрос расписания флюорографии/маммографии должен сразу уходить в handoff
    с телефоном регистратуры филиала на Ленина 5, потому что у клиники нет
    приёмного врача-радиолога с расписанием в Naika (рентгенолог только пишет
    заключения). См. кейс «Тагирова», 22.04.2026."""

    svc = Services()

    async def fake_regions():
        return [
            {
                "id": 100,
                "city": "Самара",
                "name": "Поликлиника №1",
                "addressForSite": "г. Самара, пр. Ленина, 5",
                "phone": "+7 (846) 123-45-67",
            },
            {
                "id": 101,
                "city": "Самара",
                "name": "Стационар",
                "addressForSite": "г. Самара, ул. Ново-Садовая, 106",
                "phone": "+7 (846) 999-99-99",
            },
        ]

    async def fail_doctors():
        raise AssertionError("doctors cache must NOT be touched on fixed-equipment short-circuit")

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fail_doctors)

    res = run(svc.doctors_schedule_week(query, {}))

    assert res["schedule"] == []
    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "schedule_via_registry_fixed_equipment"
    msg = str(res.get("handoff_message") or "")
    assert "Ленина, 5" in msg
    assert "846" in msg  # подставлен телефон Ленина 5, а не Ново-Садовой
    assert "999-99-99" not in msg
    branch = res.get("fixed_equipment_branch") or {}
    assert branch.get("address") == "г. Самара, пр. Ленина, 5"
    assert "123-45-67" in branch.get("phone", "")


def test_doctors_schedule_week_diagnostic_handoff_works_without_phone(monkeypatch):
    """Если /regions не отдаёт телефон Ленина 5 — handoff всё равно отрабатывает,
    просто без номера в тексте."""

    svc = Services()

    async def fake_regions():
        return []

    async def fail_doctors():
        raise AssertionError("doctors cache must NOT be touched on fixed-equipment short-circuit")

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fail_doctors)

    res = run(svc.doctors_schedule_week("расписание флюорография", {}))

    assert res["schedule"] == []
    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "schedule_via_registry_fixed_equipment"
    msg = str(res.get("handoff_message") or "")
    assert "Ленина, 5" in msg
    branch = res.get("fixed_equipment_branch") or {}
    assert branch.get("phone") == ""


def test_doctors_schedule_week_non_diagnostic_query_does_not_short_circuit(monkeypatch):
    """Регулярный запрос расписания (без слов флюоро/маммо) НЕ должен попадать
    в fixed-equipment handoff — нужно идти в обычный flow."""

    svc = Services()

    async def fake_doctors():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван Иванович",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    async def fake_regions():
        return [
            {
                "id": 100,
                "city": "Самара",
                "addressForSite": "г. Самара, пр. Ленина, 5",
                "phone": "+7 (846) 000-00-00",
            }
        ]

    async def fake_schedule_payload(_candidate, _region_name=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_get_schedule_payload_cached", fake_schedule_payload)

    res = run(svc.doctors_schedule_week("расписание Иванова", {"doctor_name": "Иванов"}))

    # должен попасть в обычный fallback service_error_schedule (а не наш новый reason)
    assert res.get("handoff_reason") != "schedule_via_registry_fixed_equipment"


def test_doctors_schedule_week_source_unavailable_uses_domain_handoff_message(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван Иванович",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    async def fake_schedule_payload(_candidate, _region_name):
        raise RuntimeError("nayka unavailable")

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_get_schedule_payload_cached", fake_schedule_payload)

    res = run(svc.doctors_schedule_week("расписание Иванова", {"doctor_name": "Иванов"}))

    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "service_error"
    assert res.get("handoff_message") == handoff_message("service_error_schedule")


def test_doctors_schedule_week_api_error_string_triggers_honest_handoff(monkeypatch):
    """find_doctor_schedule вернул строку-ошибку CRM («Не удалось получить…»):
    это СБОЙ источника, а не «врача нет». Ожидаем честный handoff
    service_error_schedule, а не вводящее в заблуждение «расписание не найдено»
    (регресс кейса Паничевой 28.05)."""
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Паничева Анна Сергеевна",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    calls = {"count": 0}

    def fake_schedule(_name, _branch=None):
        calls["count"] += 1
        return "Не удалось получить список врачей: HTTPSConnectionPool read timed out"

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res = run(svc.doctors_schedule_week("расписание Паничевой", {"doctor_name": "Паничева"}))

    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "service_error"
    assert res.get("handoff_message") == handoff_message("service_error_schedule")
    # HTTP-слой (urllib3 Retry) уже исчерпал ретраи — _fetch_schedule_source
    # НЕ должен повторять запрос на api_error (в отличие от raised-exception
    # пути, где retry-петля делает 2 попытки). Один вызов на первый кандидат.
    assert calls["count"] == 1


def test_doctors_schedule_week_doctor_not_found_string_yields_empty_no_handoff(monkeypatch):
    """find_doctor_schedule вернул «врач не найден» — это валидный негатив, а
    НЕ сбой. Ожидаем пустое расписание без handoff (renderer выдаст
    «расписание не найдено»). Различие с api_error не должно стираться."""
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван Иванович",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    def fake_schedule(_name, _branch=None):
        return "Врач с фамилией (или частью ФИО) 'Несуществующий' не найден."

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res = run(svc.doctors_schedule_week("расписание Иванова", {"doctor_name": "Иванов"}))

    assert res.get("schedule") == []
    assert not res.get("handoff_required", False)
    assert res.get("schedule_unavailable_reason") is None


def test_doctors_schedule_week_clears_stale_no_slots_reason_on_later_match(monkeypatch):
    """Regression (H2): an early surname variant returns "no free slots" (sets
    schedule_unavailable_reason), then a LATER variant matches a real schedule.
    The reason must be cleared on the successful match — otherwise the patient
    is wrongly told «слотов нет → оператор» while real slots exist."""
    svc = Services()

    async def fake_doctors():
        return [
            {
                "id": 1,
                "fio": "Кузнецова Анна Ивановна",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    async def fake_regions():
        return [
            {"id": 100, "city": "Самара", "addressForSite": "г. Самара, пр. Ленина, 5"}
        ]

    async def fake_payload(candidate, _region_name=None):
        # surname_variants("Кузнецова") == ["Кузнецова", "Кузнецов"]: the feminine
        # variant is tried first and reports no free slots; the masculine variant
        # resolves to a real Samara schedule with an open slot.
        if str(candidate).strip().lower().endswith("а"):
            return [{"_no_free_slots": True, "fio": str(candidate)}]
        return [
            {
                "fio": "Кузнецов Иван Петрович",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": ["2026-06-10 10:00"]},
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_get_schedule_payload_cached", fake_payload)

    res = run(svc.doctors_schedule_week("расписание", {"doctor_name": "Кузнецова"}))

    assert res.get("schedule"), f"real schedule with a slot must survive, got {res!r}"
    assert res.get("schedule_unavailable_reason") is None, (
        f"stale no-slots reason leaked despite a real match: "
        f"{res.get('schedule_unavailable_reason')!r}"
    )


def test_doctors_schedule_week_api_error_string_serves_stale(monkeypatch):
    """При строке-ошибке CRM отдаём последний валидный (positive) ответ из
    stale-окна, если он есть — и без удвоения вызовов retry-петлёй (api_error
    re-raise'ится сразу, в отличие от raised exception)."""
    svc = Services(
        schedule_fresh_ttl_seconds=10,
        schedule_stale_ttl_seconds=120,
        schedule_negative_ttl_seconds=5,
        schedule_cache_max_keys=100,
    )
    calls = {"count": 0}
    clock = {"ts": 5000.0}
    fail = {"enabled": False}

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Тестов Тест",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        calls["count"] += 1
        if fail["enabled"]:
            return "Не удалось получить связи врача: HTTPSConnectionPool read timed out"
        return [
            {
                "fio": "Тестов Тест",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-20", "slots": ["09:00"]}]},
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)
    monkeypatch.setattr(svc_mod.time, "time", lambda: clock["ts"])

    first = run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert first["schedule"]
    assert calls["count"] == 1

    fail["enabled"] = True
    clock["ts"] += 11
    second = run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))

    # api_error НЕ ретраится на уровне _fetch_schedule_source: +1 вызов (не +2),
    # затем отдаётся stale из кэша.
    assert calls["count"] == 2
    assert second["schedule"], "Expected stale schedule when source returns api error"
    assert not second.get("handoff_required", False)


def test_doctors_schedule_week(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван",
                "specialization": "терапевт",
                "regions": ["Ленина 5"],
                "units": ["Терапия"],
            }
        ]

    def fake_schedule(_name, _branch=None):
        return [{"fio": "Иванов Иван", "schedule": {"Ленина 5": []}}]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res = run(svc.doctors_schedule_week("Иванов", {"doctor_name": "Иванов"}))

    assert res["schedule"], "Expected schedule list"
    assert "entities_used" in res
    assert str(res["entities_used"].get("last_name") or "").lower().startswith("иванов")


def test_doctors_schedule_week_ignores_unrelated_payload_rows_for_requested_doctor(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Просвиров Евгений Юрьевич",
                "specialization": "ревматолог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Врач-ревматолог"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    async def fake_schedule_payload(last_name, _region_name=None):
        # Симулируем некачественный ответ API:
        # запросили Просвирова, а вернулась Рязанова.
        if str(last_name).lower().startswith("просвиров"):
            return [
                {
                    "fio": "Рязанова Валерия Владимировна",
                    "regions": ["г. Самара, пр. Ленина, 5"],
                    "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-04-06", "slots": ["14:30"]}]},
                }
            ]
        return []

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc, "_get_schedule_payload_cached", fake_schedule_payload)

    res = run(svc.doctors_schedule_week("Покажите расписание Просвирова", {"doctor_name": "Просвиров"}))

    assert res["schedule"] == []
    assert "doctors_schedule_week" in str(res.get("note") or "")


def test_doctors_schedule_week_keeps_samara_branch_for_multi_city_doctor(monkeypatch):
    """Регрессия: врач с практикой в нескольких городах (Самара + Оренбург,
    напр. Лунев — «Ленина 5» + Оренбург) не должен выпадать целиком.

    Должны остаться только самарский филиал и его расписание; оренбургские
    адрес и окна — отсечься. Раньше `_has_explicit_non_samara_regions`
    отбрасывал такого врача полностью → «расписание не найдено»."""
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 2738,
                "fio": "Лунев Андрей Владимирович",
                "specialization": "уролог",
                "regions": ["г. Оренбург, ул. Чкалова, 51/1, пом.6.", "Ленина 5"],
            }
        ]

    async def fake_samara_tokens():
        return {"ленина 5"}

    async def fake_schedule_payload(_last_name, _region_name=None):
        return [
            {
                "fio": "Лунев Андрей Владимирович",
                "regions": ["г. Оренбург, ул. Чкалова, 51/1, пом.6.", "Ленина 5"],
                "schedule": {
                    "г. Оренбург, ул. Чкалова, 51/1, пом.6.": [
                        {"date": "2026-05-24", "slots": ["10:00"]}
                    ],
                    "Ленина 5": [{"date": "2026-05-24", "slots": ["12:00"]}],
                },
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc, "_get_schedule_payload_cached", fake_schedule_payload)

    res = run(svc.doctors_schedule_week("Лунев", {"doctor_name": "Лунев"}))

    assert res["schedule"], "Расписание не должно быть пустым для multi-city врача"
    row = res["schedule"][0]
    assert list(row.get("schedule", {}).keys()) == ["Ленина 5"]
    assert row.get("regions") == ["Ленина 5"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("расписание нейрохирурга", "нейрохирург"),
        ("расписание нейрохирургов", "нейрохирург"),
        ("расписание флеболога", "флеболог"),
        ("расписание флебологов", "флеболог"),
        ("нужен прием у уролога", "уролог"),
        ("нужен прием у лора", "лор"),
        ("нужен уролог андролог", "уролог-андролог"),
    ],
)
def test_extract_specialty_supports_inflected_neurosurgeon_and_phlebologist(text, expected):
    assert extract_specialty(text) == expected


@pytest.mark.parametrize(
    ("query", "unit_name", "expected_specialty"),
    [
        ("расписание нейрохирург", "Врач-нейрохирург", "нейрохирург"),
        ("расписание флеболог", "Врач-флеболог", "флеболог"),
    ],
)
def test_doctors_schedule_week_by_expanded_specialty_catalog(monkeypatch, query, unit_name, expected_specialty):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Тестов Тест",
                "ord": 1,
                "specialization": expected_specialty,
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": [unit_name],
                "unit_links": [
                    {
                        "company_unit_name": unit_name,
                        "main": True,
                        "specialization": expected_specialty,
                    }
                ],
                "main_units": [unit_name],
                "main_specializations": [expected_specialty],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        return [
            {
                "fio": "Тестов Тест",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {
                    "г. Самара, пр. Ленина, 5": [
                        {"date": "2026-03-20", "slots": ["09:00"]}
                    ]
                },
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res = run(svc.doctors_schedule_week(query, {}))

    assert res["note"] == "doctors_schedule_week: by specialty"
    assert res["schedule"], "Expected schedule rows for specialty query"
    assert res["entities_used"].get("specialty") == expected_specialty


def test_doctors_schedule_week_excludes_explicit_non_samara_rows(monkeypatch):
    svc = Services()

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван",
                "specialization": "онколог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Онкология"],
            }
        ]

    def fake_schedule(_name, _branch=None):
        return [
            {
                "fio": "Иванов Иван",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-09", "slots": ["09:00"]}]},
            },
            {
                "fio": "Иванов Иван",
                "regions": ["г. Оренбург, ул. Пушкинская, 10"],
                "schedule": {"г. Оренбург, ул. Пушкинская, 10": [{"date": "2026-03-09", "slots": ["10:00"]}]},
            },
        ]

    async def fake_samara_tokens():
        return set()

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res = run(svc.doctors_schedule_week("Иванов", {"doctor_name": "Иванов"}))

    assert len(res["schedule"]) == 1
    assert "Оренбург" not in str(res["schedule"][0].get("regions"))


def test_doctors_schedule_week_city_samara_calls_unfiltered_region(monkeypatch):
    svc = Services()
    calls: list[object] = []

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Иванов Иван",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Терапия"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        calls.append(_branch)
        return [
            {
                "fio": "Иванов Иван",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-09", "slots": ["09:00"]}]},
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res = run(svc.doctors_schedule_week("Иванов", {"doctor_name": "Иванов", "region": "Самара"}))

    assert res["schedule"], "Expected schedule list"
    assert calls, "Expected at least one schedule source call"
    assert all(call is None for call in calls), "Samara city should be normalized to unfiltered region call"
    assert res["entities_used"].get("region_name") is None


def test_doctors_schedule_week_cache_hit(monkeypatch):
    svc = Services(
        schedule_fresh_ttl_seconds=30,
        schedule_stale_ttl_seconds=600,
        schedule_negative_ttl_seconds=15,
        schedule_cache_max_keys=100,
    )
    calls = {"count": 0}

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Тестов Тест",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Терапия"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        calls["count"] += 1
        return [
            {
                "fio": "Тестов Тест",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-20", "slots": ["09:00"]}]},
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)

    res1 = run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    res2 = run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))

    assert calls["count"] == 1
    assert res1["schedule"]
    assert res2["schedule"]


def test_doctors_schedule_week_cache_miss_after_fresh_ttl(monkeypatch):
    svc = Services(
        schedule_fresh_ttl_seconds=10,
        schedule_stale_ttl_seconds=600,
        schedule_negative_ttl_seconds=5,
        schedule_cache_max_keys=100,
    )
    calls = {"count": 0}
    clock = {"ts": 1000.0}

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Тестов Тест",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Терапия"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        calls["count"] += 1
        return [
            {
                "fio": "Тестов Тест",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-20", "slots": ["09:00"]}]},
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)
    monkeypatch.setattr(svc_mod.time, "time", lambda: clock["ts"])

    run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert calls["count"] == 1

    clock["ts"] += 11
    run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert calls["count"] == 2


def test_doctors_schedule_week_returns_stale_on_source_error(monkeypatch):
    svc = Services(
        schedule_fresh_ttl_seconds=10,
        schedule_stale_ttl_seconds=120,
        schedule_negative_ttl_seconds=5,
        schedule_cache_max_keys=100,
    )
    calls = {"count": 0}
    clock = {"ts": 2000.0}
    fail = {"enabled": False}

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Тестов Тест",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Терапия"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        calls["count"] += 1
        if fail["enabled"]:
            raise RuntimeError("source unavailable")
        return [
            {
                "fio": "Тестов Тест",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-20", "slots": ["09:00"]}]},
            }
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)
    monkeypatch.setattr(svc_mod.time, "time", lambda: clock["ts"])

    first = run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert first["schedule"]
    assert calls["count"] == 1

    fail["enabled"] = True
    clock["ts"] += 11
    second = run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))

    # Вторая проверка делает 2 попытки запроса (retry), затем отдает stale из кэша.
    assert calls["count"] == 3
    assert second["schedule"], "Expected stale schedule when source is unavailable"
    assert not second.get("handoff_required", False)


def test_doctors_schedule_week_negative_cache_ttl(monkeypatch):
    svc = Services(
        schedule_fresh_ttl_seconds=30,
        schedule_stale_ttl_seconds=120,
        schedule_negative_ttl_seconds=5,
        schedule_cache_max_keys=100,
    )
    calls = {"count": 0}
    clock = {"ts": 3000.0}

    async def fake_ensure_cache():
        return [
            {
                "id": 1,
                "fio": "Тестов Тест",
                "specialization": "терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Терапия"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_schedule(_name, _branch=None):
        calls["count"] += 1
        return []

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)
    monkeypatch.setattr(svc_mod.time, "time", lambda: clock["ts"])

    run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert calls["count"] == 1

    clock["ts"] += 2
    run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert calls["count"] == 1, "Expected negative cache hit within negative TTL"

    clock["ts"] += 4
    run(svc.doctors_schedule_week("расписание тестова", {"doctor_name": "Тестов"}))
    assert calls["count"] == 2, "Expected cache miss after negative TTL expiry"


def test_appointment_help_meili(monkeypatch):
    svc = Services()
    captured: dict[str, object] = {}

    def fake_search(_index, _query, *args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return "<b>info</b>"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: "info")

    res = run(svc.appointment_help("запись", {}))

    assert res["instructions"] == "info"
    assert captured.get("kwargs") == {"output_mode": "content_only", "max_chars": 12000}


def test_main_index_info_success(monkeypatch):
    svc = Services()
    captured: dict[str, object] = {}

    def fake_search(_index, _query, *args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return "<b>Копия договора с печатью</b>"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: "Копия договора с печатью")

    res = run(svc.main_index_info("Как получить копию договора с печатью?", {}))

    assert res["content"] == "Копия договора с печатью"
    assert str(res["note"]).startswith("main_index_info: main_index")
    assert res.get("handoff_required") is not True
    assert captured.get("kwargs") == {"output_mode": "content_only", "max_chars": 12000}


def test_main_index_info_no_matches(monkeypatch):
    svc = Services()

    def fake_search(_index, _query, *args, **kwargs):
        return "Совпадений не найдено, cформулируйте запрос иначе"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(svc.main_index_info("какой-то редкий запрос", {}))

    assert res["content"] == ""
    assert res["note"] == "main_index_info: no matches"
    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "knowledge_not_found"
    assert "в моей базе данных информации недостаточно" in str(res.get("handoff_message") or "").lower()


def test_main_index_info_source_unavailable_returns_handoff(monkeypatch):
    svc = Services()

    def fake_search(_index, _query, *args, **kwargs):
        raise RuntimeError("meili unavailable")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)

    res = run(svc.main_index_info("копия договора с печатью", {}))

    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "service_error"
    assert res.get("handoff_message") == handoff_message("service_error_doctor_info")


def test_main_index_info_tax_source_unavailable_returns_guidance_without_handoff(monkeypatch):
    svc = Services()

    def fake_search(_index, _query, *args, **kwargs):
        raise RuntimeError("meili unavailable")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)

    res = run(svc.main_index_info("справка для ФНС", {}))

    assert res.get("handoff_required") is not True
    assert str(res.get("content") or "") == "Заказ справки на налоговый вычет осуществляется на сайте https://naykalab.ru/spravka-nalogoviy-vichet"


def test_main_index_info_tax_returns_direct_link_without_meili(monkeypatch):
    svc = Services()

    def fake_search(_index, _query, *args, **kwargs):
        raise AssertionError("tax doc request must not call main_index search")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)

    res = run(svc.main_index_info("Как получить справку для налогового вычета?", {}))

    assert res.get("handoff_required") is not True
    assert res["note"] == "main_index_info: tax direct link"
    assert str(res.get("content") or "") == "Заказ справки на налоговый вычет осуществляется на сайте https://naykalab.ru/spravka-nalogoviy-vichet"


def test_test_assist_price_by_region(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [{"serviceName": "Анализ крови общий", "cost": 500}]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.test_assist("анализ крови", {}))

    assert res["tests"], "Expected matches in priceByRegion"


def test_test_prepare_meili(monkeypatch):
    svc = Services()
    captured: dict[str, object] = {}

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    def fake_search(_index, _query, *args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return "<i>подготовка к анализу крови: натощак</i>"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: "подготовка к анализу крови: натощак")

    res = run(svc.test_prepare("анализ крови", {}))

    assert res["prepare"] == "подготовка к анализу крови: натощак"
    assert captured.get("kwargs") == {"output_mode": "content_only", "max_chars": 12000}


def test_test_prepare_prefers_service_info_preparation(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Анализ крови на холестерин",
                "preparation": "Кровь сдаётся натощак, желательно утром.",
            }
        ],
    )

    def fail_meili(*_args, **_kwargs):
        raise AssertionError("Meili fallback must not run when serviceInfoAll matched")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fail_meili)

    res = run(svc.test_prepare("Как подготовиться к анализу на холестерин?", {"service_name": "Холестерин"}))

    assert "натощак" in str(res.get("prepare") or "").lower()
    assert res["note"] == "prepare: serviceInfoAll"


def test_test_prepare_prefers_service_info_for_analysis_name_query(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Анализ крови на холестерин",
                "preparation": "Кровь сдаётся натощак, желательно утром.",
            }
        ],
    )

    def fail_meili(*_args, **_kwargs):
        raise AssertionError("Meili fallback must not run when serviceInfoAll matched")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fail_meili)

    res = run(svc.test_prepare("Анализ крови на холестерин", {"service_name": "Холестерин"}))

    assert "натощак" in str(res.get("prepare") or "").lower()
    assert res["note"] == "prepare: serviceInfoAll"


def test_test_prepare_falls_back_to_meili_when_service_info_prepare_is_generic_heading(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Пайпель-биопсия",
                "preparation": "<h1>Подготовка к исследованию</h1>",
            }
        ],
    )

    def fake_search(_index, _query, *args, **kwargs):
        return "Подготовка к пайпель-биопсии: забор проводится на 7-11 день цикла."

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(
        svc_mod.html_cleaner,
        "strip_html",
        lambda s: str(s).replace("<h1>", "").replace("</h1>", "").strip(),
    )

    res = run(svc.test_prepare("Как подготовиться к пайпель-биопсии?", {"service_name": "Пайпель-биопсия"}))

    assert "пайпель-биопс" in str(res.get("prepare") or "").lower()
    assert res.get("note") != "prepare: serviceInfoAll"


def test_test_prepare_does_not_match_unrelated_service_info_by_generic_prepare_token(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "ЭЛИ-В-6-Тест (общее состояние иммунной системы, подготовка к вакцинации, 6 антигенов)",
                "preparation": "Специальной подготовки не требуется. Взятие крови производится натощак.",
            }
        ],
    )

    def fake_search(_index, _query, *args, **kwargs):
        return "Подготовка к ЭКГ: специальной подготовки не требуется."

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(svc.test_prepare("ЭКГ подскажите", {"service_name": "ЭКГ"}))

    assert "к экг" in str(res.get("prepare") or "").lower()
    assert "взятие крови производится натощак" not in str(res.get("prepare") or "").lower()


def test_test_prepare_prefers_exact_service_info_row_over_generic_similar_name(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "ЭЛИ-В-6-Тест (общее состояние иммунной системы, подготовка к вакцинации, 6 антигенов)",
                "preparation": "Подготовка к анализу крови натощак.",
            },
            {
                "serviceName": "ЭКГ",
                "preparation": "Подготовка к ЭКГ: специальной подготовки не требуется.",
            },
        ],
    )

    def fail_meili(*_args, **_kwargs):
        raise AssertionError("Meili fallback must not run when exact serviceInfoAll matched")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fail_meili)

    res = run(svc.test_prepare("ЭКГ подскажите", {"service_name": "ЭКГ"}))

    assert "к экг" in str(res.get("prepare") or "").lower()
    assert res["note"] == "prepare: serviceInfoAll"


def test_test_prepare_rejects_unrelated_hormone_service_info_and_falls_back_to_meili(monkeypatch):
    svc = Services()
    calls = {"meili": 0}

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Анализ крови на гормоны",
                "preparation": (
                    "Подготовка к исследованию. "
                    "Гистологическое исследование предварительной подготовки не требует. "
                    "Доставка материала осуществляется в емкости с 10% формалином."
                ),
            }
        ],
    )

    def fake_search(_index, _query, *args, **kwargs):
        calls["meili"] += 1
        return "Подготовка к анализу крови на гормоны: кровь сдаётся утром натощак."

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(
        svc.test_prepare(
            "Здравствуйте скажите пожалуйста а кровь на гормоны сдают на голодный желудок?",
            {},
        )
    )

    answer = str(res.get("prepare") or "").lower()
    assert calls["meili"] >= 1
    assert "натощак" in answer
    assert "формалин" not in answer
    assert res.get("note") != "prepare: serviceInfoAll"


def test_prepare_roots_match_does_not_match_holesterol_with_sterile_substring():
    assert _svc_prepare_roots_match("холестерин", {"стерильн"}) is False
    assert _svc_prepare_roots_match("холестерин", {"холестерин"}) is True


def test_test_prepare_cholesterol_not_confused_by_urogenital_soskob(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Gardnerella vaginalis [кач.]",
                "preparation": (
                    "Подготовка к исследованию. "
                    "Соскоб урогенитальный берется в стерильный контейнер."
                ),
            },
            {
                "serviceName": "Анализ крови на холестерин",
                "preparation": (
                    "Для анализа на холестерин кровь сдают утром натощак, "
                    "через 8-14 часов после еды."
                ),
            },
        ],
    )

    def fail_meili(*_args, **_kwargs):
        raise AssertionError("Meili fallback must not run when serviceInfoAll matched")

    def fake_runtime_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
        values = {
            "MR_PREPARE_RELEVANCE_LOW_THRESHOLD": 0.30,
            "MR_PREPARE_RELEVANCE_HIGH_THRESHOLD": 0.60,
            "MR_PREPARE_RELEVANCE_MARGIN_THRESHOLD": 0.08,
        }
        return values.get(name, default)

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name in {"MR_PREPARE_LLM_WRAP_ENABLED", "MR_PREPARE_RELEVANCE_LLM_ENABLED"}:
            return False
        return default

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fail_meili)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)
    monkeypatch.setattr(_common_mod, "_runtime_float", fake_runtime_float)
    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)

    res = run(svc.test_prepare("Как подготовиться к анализу на холестерин?", {}))
    answer = str(res.get("prepare") or "").lower()
    assert "холестерин" in answer
    assert "натощак" in answer
    assert "урогениталь" not in answer
    assert "стерильн" not in answer
    assert res.get("note") == "prepare: serviceInfoAll"


def test_prepare_relevance_gate_thresholds(monkeypatch):
    def fake_runtime_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
        values = {
            "MR_PREPARE_RELEVANCE_LOW_THRESHOLD": 0.30,
            "MR_PREPARE_RELEVANCE_HIGH_THRESHOLD": 0.60,
            "MR_PREPARE_RELEVANCE_MARGIN_THRESHOLD": 0.10,
        }
        return values.get(name, default)

    monkeypatch.setattr(_common_mod, "_runtime_float", fake_runtime_float)

    assert _svc_prepare_relevance_gate(0.20, 0.30) == "reject"
    assert _svc_prepare_relevance_gate(0.72, 0.12) == "accept"
    assert _svc_prepare_relevance_gate(0.72, 0.01) == "llm"
    assert _svc_prepare_relevance_gate(0.45, 0.30) == "llm"


def test_test_prepare_mid_score_uses_llm_validator_and_accepts_api(monkeypatch):
    svc = Services()
    llm_calls = {"n": 0}

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Гормональный профиль",
                "preparation": "Кровь рекомендуется сдавать утром натощак, воду пить можно.",
            }
        ],
    )

    def fail_meili(*_args, **_kwargs):
        raise AssertionError("Meili fallback must not run when API candidate approved by LLM")

    async def fake_generate_text(prompt, *, timeout_s, queue_timeout_ms, fmt=None, llm=None, think=None):
        llm_calls["n"] += 1
        assert fmt == "json"
        assert "гормон" in str(prompt).lower()
        return '{"verdict":"RELEVANT","confidence":0.86,"reason":"тема подготовки совпадает"}'

    def fake_runtime_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
        values = {
            "MR_PREPARE_RELEVANCE_LOW_THRESHOLD": 0.25,
            "MR_PREPARE_RELEVANCE_HIGH_THRESHOLD": 0.95,
            "MR_PREPARE_RELEVANCE_MARGIN_THRESHOLD": 0.20,
        }
        return values.get(name, default)

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return False
        if name == "MR_PREPARE_RELEVANCE_LLM_ENABLED":
            return True
        return default

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fail_meili)
    monkeypatch.setattr(llm_runtime_mod, "generate_text", fake_generate_text)
    monkeypatch.setattr(_common_mod, "_runtime_float", fake_runtime_float)
    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)

    res = run(svc.test_prepare("Кровь на гормоны сдают натощак?", {}))

    assert llm_calls["n"] == 1
    assert "натощак" in str(res.get("prepare") or "").lower()
    assert res.get("note") == "prepare: serviceInfoAll"


def test_test_prepare_mid_score_llm_reject_falls_back_to_meili(monkeypatch):
    svc = Services()
    llm_calls = {"n": 0}
    meili_calls = {"n": 0}

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [
            {
                "serviceName": "Анализ крови на гормоны",
                "preparation": "Подготовка к исследованию. Необходимо заполнить анкету пациента.",
            }
        ],
    )

    async def fake_generate_text(prompt, *, timeout_s, queue_timeout_ms, fmt=None, llm=None, think=None):
        llm_calls["n"] += 1
        assert fmt == "json"
        p = str(prompt or "").lower()
        if "необходимо заполнить анкету" in p:
            return '{"verdict":"IRRELEVANT","confidence":0.91,"reason":"нет конкретной подготовки по запросу"}'
        if "кровь сдаётся утром натощак" in p:
            return '{"verdict":"RELEVANT","confidence":0.89,"reason":"релевантная подготовка к анализу"}'
        return '{"verdict":"IRRELEVANT","confidence":0.60,"reason":"неуверенно"}'

    def fake_search(_index, _query, *args, **kwargs):
        meili_calls["n"] += 1
        return "Подготовка к анализу крови на гормоны: кровь сдаётся утром натощак."

    def fake_runtime_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
        values = {
            "MR_PREPARE_RELEVANCE_LOW_THRESHOLD": 0.15,
            "MR_PREPARE_RELEVANCE_HIGH_THRESHOLD": 0.95,
            "MR_PREPARE_RELEVANCE_MARGIN_THRESHOLD": 0.20,
        }
        return values.get(name, default)

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return False
        if name == "MR_PREPARE_RELEVANCE_LLM_ENABLED":
            return True
        return default

    monkeypatch.setattr(llm_runtime_mod, "generate_text", fake_generate_text)
    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)
    monkeypatch.setattr(_common_mod, "_runtime_float", fake_runtime_float)
    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)

    res = run(svc.test_prepare("Кровь на гормоны сдают натощак?", {}))

    assert llm_calls["n"] >= 2
    assert meili_calls["n"] >= 1
    assert "натощак" in str(res.get("prepare") or "").lower()
    assert res.get("note") == "prepare: main_index"


def test_test_prepare_no_matches_returns_clarify_without_handoff(monkeypatch):
    svc = Services()

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    def fake_search(_index, _query, *args, **kwargs):
        return "Совпадений не найдено, cформулируйте запрос иначе"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(svc.test_prepare("подготовка к анализу крови", {}))

    assert "подготов" in str(res.get("prepare") or "").lower()
    assert res.get("handoff_required") is not True


def test_test_prepare_uses_fallback_variant_query(monkeypatch):
    svc = Services()
    calls: list[str] = []

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    def fake_search(_index, _query, *args, **kwargs):
        calls.append(str(_query))
        if str(_query).strip().lower() == "как подготовиться к вульвоскопии":
            return "Совпадений не найдено, cформулируйте запрос иначе"
        if str(_query).strip().lower() == "подготовка к вульвоскопии":
            return "Подготовка к вульвоскопии: за 24 часа исключить половые контакты."
        return "Совпадений не найдено, cформулируйте запрос иначе"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(svc.test_prepare("Как подготовиться к вульвоскопии?", {}))

    assert res.get("handoff_required") is not True
    assert "вульвоскоп" in str(res.get("prepare") or "").lower()
    assert any("как подготовиться к вульвоскопии" in q.lower() for q in calls)
    assert any("подготовка к вульвоскопии" in q.lower() for q in calls)


def test_test_prepare_handles_noisy_prefix_rules_of_prepare(monkeypatch):
    svc = Services()
    calls: list[str] = []

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    def fake_search(_index, _query, *args, **kwargs):
        calls.append(str(_query))
        if str(_query).strip().lower() == "подготовка к фгдс с наркозом":
            return "Подготовка к ФГДС с наркозом: натощак, без курения за 3 часа."
        return "Совпадений не найдено, cформулируйте запрос иначе"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(svc.test_prepare("Здравствуйте! Какие правила подготовки к ФГДС с наркозом?", {}))

    assert "фгдс" in str(res.get("prepare") or "").lower()
    assert any("подготовка к фгдс с наркозом" in q.lower() for q in calls)


def test_test_prepare_compacts_long_meili_answer_with_llm_wrap(monkeypatch):
    svc = Services()
    calls: dict[str, int] = {"llm": 0}

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    source_text = (
        "Подготовка к пайпель-биопсии эндометрия: процедура проводится на 7-11 день цикла. "
        "За 48 часов необходимо исключить половые контакты. За 24 часа не использовать "
        "вагинальные свечи и спринцевания. В день процедуры не применять кремы в интимной зоне. "
        "За 2-3 часа желательно опорожнить мочевой пузырь. При наличии анализов возьмите их с собой."
    )

    def fake_search(_index, _query, *args, **kwargs):
        return source_text

    async def fake_generate_text(prompt, *, timeout_s, queue_timeout_ms, fmt=None, llm=None, think=None):
        calls["llm"] += 1
        assert "пайпель" in str(prompt).lower()
        return (
            "Для подготовки к пайпель-биопсии:\n"
            "- Проводите исследование на 7-11 день цикла.\n"
            "- За 48 часов исключите половые контакты.\n"
            "- За 24 часа не используйте вагинальные свечи и спринцевания."
        )

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)
    monkeypatch.setattr(llm_runtime_mod, "generate_text", fake_generate_text)

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return True
        return default

    def fake_runtime_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
        if name == "MR_PREPARE_LLM_WRAP_MIN_CHARS":
            return 1
        return default

    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)
    monkeypatch.setattr(_common_mod, "_runtime_int", fake_runtime_int)

    res = run(svc.test_prepare("Как подготовиться к пайпель-биопсии?", {"service_name": "Пайпель-биопсия"}))

    assert calls["llm"] >= 2
    assert "за 48 часов" in str(res.get("prepare") or "").lower()
    assert len(str(res.get("prepare") or "")) < len(source_text)
    assert res.get("prepare_wrap_status") == "llm_wrapped"


def test_test_prepare_llm_wrap_uses_deterministic_fallback_on_invalid_compaction(monkeypatch):
    svc = Services()

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    source_text = (
        "Подготовка к анализу крови на холестерин: кровь сдаётся натощак 8-12 часов, "
        "разрешена негазированная вода, за сутки исключить алкоголь и жирную пищу."
    )

    def fake_search(_index, _query, *args, **kwargs):
        return source_text

    async def fake_generate_text(*args, **kwargs):
        return "NO_RELEVANT_CONTENT"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)
    monkeypatch.setattr(llm_runtime_mod, "generate_text", fake_generate_text)

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return True
        return default

    def fake_runtime_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
        if name == "MR_PREPARE_LLM_WRAP_MIN_CHARS":
            return 1
        return default

    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)
    monkeypatch.setattr(_common_mod, "_runtime_int", fake_runtime_int)

    res = run(svc.test_prepare("Как подготовиться к анализу на холестерин?", {"service_name": "Холестерин"}))

    answer = str(res.get("prepare") or "")
    assert "холестерин" in answer.lower()
    assert "натощак" in answer.lower()
    assert "стоим" not in answer.lower()
    assert res.get("prepare_wrap_status") in {"fallback_compact", "fallback_not_shorter"}
    assert res.get("prepare_wrap_reason") == "llm_wrap_invalid_output"


def test_test_prepare_llm_wrap_timeout_uses_deterministic_fallback(monkeypatch):
    svc = Services()

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    source_text = (
        "Подготовка к анализу крови на холестерин: кровь сдают натощак 8-12 часов. "
        "Разрешена только негазированная вода. За сутки исключить алкоголь и жирную пищу. "
        "Стоимость услуги 490 руб. Адреса и запись уточняйте у администратора."
    )

    def fake_search(_index, _query, *args, **kwargs):
        return source_text

    async def fake_generate_text(*args, **kwargs):
        raise TimeoutError("llm timeout")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)
    monkeypatch.setattr(llm_runtime_mod, "generate_text", fake_generate_text)

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return True
        if name == "MR_PREPARE_RELEVANCE_LLM_ENABLED":
            return False
        return default

    def fake_runtime_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
        if name == "MR_PREPARE_LLM_WRAP_MIN_CHARS":
            return 1
        if name == "MR_PREPARE_FALLBACK_MAX_CHARS":
            return 800
        return default

    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)
    monkeypatch.setattr(_common_mod, "_runtime_int", fake_runtime_int)

    res = run(svc.test_prepare("Как подготовиться к анализу на холестерин?", {"service_name": "Холестерин"}))

    answer = str(res.get("prepare") or "")
    assert "натощак" in answer.lower()
    assert "стоим" not in answer.lower()
    assert "руб" not in answer.lower()
    assert len(answer) < len(source_text)
    assert res.get("prepare_wrap_status") == "fallback_compact"
    assert "TimeoutError" in str(res.get("prepare_wrap_reason") or "")


def test_prepare_subject_hint_prefers_full_phrase_from_query_over_truncated_entity():
    hint = _svc_prepare_subject_hint(
        "Как подготовиться к гастроскопии?",
        {"service_name": "гастроскопи"},
    )
    assert hint == "гастроскопии"


def test_test_prepare_main_index_override_after_llm_reject(monkeypatch):
    svc = Services()

    monkeypatch.setattr(svc_mod.api_service_info, "load_service_info", lambda: [])

    meili_text = (
        "ПАМЯТКА ПАЦИЕНТУ ФКС + ФГДС с наркозом. "
        "Подготовка к исследованию: за день исключить тяжелую пищу, "
        "утром в день исследования не есть и не пить, "
        "воду можно за 3 часа до процедуры."
    )

    def fake_search(_index, _query, *args, **kwargs):
        return meili_text

    async def fake_llm_reject(_query, _candidate):
        return False, 0.90, "mixed_document"

    def fake_runtime_bool(name: str, default: bool) -> bool:
        if name == "MR_PREPARE_LLM_WRAP_ENABLED":
            return False
        if name == "MR_PREPARE_RELEVANCE_LLM_ENABLED":
            return True
        return default

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)
    monkeypatch.setattr(svc, "_prepare_llm_validate_candidate", fake_llm_reject)
    monkeypatch.setattr(_common_mod, "_runtime_bool", fake_runtime_bool)

    res = run(svc.test_prepare("Как подготовиться к ФГДС?", {"service_name": "ФГДС"}))

    assert res.get("note") == "prepare: main_index"
    assert "подготов" in str(res.get("prepare") or "").lower()
    assert "фгдс" in str(res.get("prepare") or "").lower()


def test_test_assist_source_unavailable_returns_clarify_without_handoff(monkeypatch):
    svc = Services()

    def fail_price(*_args, **_kwargs):
        raise RuntimeError("price unavailable")

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fail_price)

    res = run(svc.test_assist("Какие анализы сдать на щитовидку?", {}))

    assert res.get("handoff_required") is not True
    assert "подобрать анализы" in str(res.get("message") or "").lower()


def test_test_prepare_falls_back_to_meili_when_service_info_has_no_preparation(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_service_info,
        "load_service_info",
        lambda: [{"serviceName": "Анализ крови на холестерин", "preparation": ""}],
    )

    def fake_search(_index, _query, *args, **kwargs):
        return "Подготовка к анализу крови на холестерин: кровь сдают натощак."

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)

    res = run(svc.test_prepare("Как подготовиться к анализу на холестерин?", {"service_name": "Холестерин"}))

    assert "натощак" in str(res.get("prepare") or "").lower()


def test_test_result_status_stub():
    svc = Services()
    res = run(svc.test_result_status("результаты", {}))
    assert res["ready"] is False


def test_quick_fill_test_goal_checkup():
    out = quick_fill_core_entities("чекап", {}, ["_any_of:test_goal,test_name"])
    assert out.get("test_goal"), "Expected quick-fill to capture test goal for checkup keyword"


def test_quick_fill_core_entities_is_split_into_domain_helpers():
    for helper_name in (
        "_fill_insurance_entities",
        "_fill_datetime_entities",
        "_fill_test_result_entities",
        "_fill_appointment_entities",
        "_fill_patient_entities",
    ):
        assert callable(getattr(policies_mod, helper_name, None))


def test_quick_fill_insurance_and_child_age_are_preserved_together():
    out = quick_fill_core_entities("детям 5 лет по дмс", {}, ["child_age"])

    assert out.get("insurance_type") == "dms"
    assert out.get("accepts_children") is True
    assert out.get("child_age") == 5


def test_quick_fill_city_typo_is_normalized():
    out = quick_fill_core_entities("Самраа", {}, ["_any_of:city,branch_name,branch_id"])
    assert out.get("city") == "Самара"


def test_match_city_is_found_inside_mixed_phrase():
    assert match_city("Анализы Самара") == "Самара"
    assert match_city("Самраа анализы") == "Самара"


def test_quick_fill_service_name_not_taken_from_generic_phrase():
    out = quick_fill_core_entities("можно записаться к врачу", {}, ["_any_of:doctor_id,doctor_name,specialty,service_name"])
    assert "service_name" not in out


def test_extract_service_phrase_handles_freeform_procedure():
    assert extract_service_phrase("на торакоцентез") == "Торакоцентез"
    assert extract_service_phrase("услуга торакоцентез") == "Торакоцентез"


def test_extract_service_phrase_ignores_generic_appointment_phrase():
    assert extract_service_phrase("можно записаться к врачу") is None


def test_quick_fill_doctor_name_not_taken_from_greeting_phrase():
    out = quick_fill_core_entities(
        "Здравствуйте! Я хотела бы записаться к врачу на прием",
        {},
        ["_any_of:doctor_id,doctor_name,specialty,service_name"],
    )
    assert "doctor_name" not in out


def test_match_branch_hint_handles_typo():
    index = build_branch_index(
        [
            {"id": "1", "name": "г. Самара, ул. Победы, 83", "aliases": "победы 83, победы"},
            {"id": "2", "name": "г. Самара, пр. Ленина, 5", "aliases": "ленина 5, ленина"},
        ]
    )
    bid, bname = match_branch_hint("пбеды 83", index)
    assert bid == "1"
    assert "Победы" in str(bname)


def test_price_info_doctor_prices(monkeypatch):
    svc = Services()

    def fake_doctor_prices():
        return [{"doctorId": 1, "serviceName": "Прием", "cost": 1000}]

    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.price_info("", {"doctor_id": 1}))

    assert res["prices"], "Expected doctor prices"


def test_price_info_price_by_region(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [{"serviceName": "ЭКГ", "cost": 650}]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("ЭКГ", {}))

    assert res["prices"], "Expected prices from priceByRegion"


def test_price_info_source_unavailable_uses_domain_handoff_message(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        raise RuntimeError("price api unavailable")

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("ЭКГ", {}))

    assert res.get("handoff_required") is True
    assert res.get("handoff_reason") == "service_error"
    assert res.get("handoff_message") == handoff_message("service_error_prices")


def test_price_info_enriches_care_setting_from_price_units(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [{"serviceName": "Тонзиллотомия", "cost": 20000, "priceUnitId": 195}]

    def fake_price_units():
        return [
            {"id": 146, "parent": 1, "name": "III Амбулаторно-поликлиническая помощь"},
            {"id": 311, "parent": 190, "name": "Дневной стационар"},
            {"id": 195, "parent": 311, "name": "Дневной стационар (Оториноларингология)"},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)
    monkeypatch.setattr(svc_mod.api_price, "load_price_units", fake_price_units)

    res = run(svc.price_info("Стоимость тонзиллотомии", {}))

    assert res["prices"], "Expected prices from priceByRegion"
    top = res["prices"][0]
    assert top["care_setting_label"] == "дневной стационар"
    assert "Ленина" in str(top.get("care_setting_address") or "")


def test_price_info_ranks_analysis_matches(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Выезд на дом (г. Самара)", "serviceHomecode": "905", "cost": 1000},
            {
                "serviceName": "Расширенный комплексный анализ на витамины (A, D, E)",
                "serviceHomecode": "5420",
                "cost": 22500,
            },
            {"serviceName": "Анализ крови на витамин D (25-OH)", "serviceHomecode": "5100", "cost": 1800},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Сколько стоит анализ на витамин д?", {}))

    assert res["prices"], "Expected ranked prices"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "витамин d" in top_name


def test_price_info_does_not_return_random_rows_for_noisy_generic_price_question(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Лечение периодонтита более чем 3 канального зуба", "cost": 2500},
            {"serviceName": "Ген рецептора витамина D (VDR). Выявление мутации G283A", "cost": 980},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(
        svc.price_info(
            "Здравствуйте! Скажите пожалуйста, как рассчитывается стоимость анализы!? По выходным дешевле чем в будни?",
            {},
        )
    )

    assert res.get("prices") == []


def test_price_info_ranks_endoscopy_matches(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "ЭКГ", "cost": 650},
            {"serviceName": "Эндоскопия диагностическая", "cost": 2100},
            {"serviceName": "Консультация гастроэнтеролога", "cost": 1800},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Сколько стоит эндоскопия?", {}))

    assert res["prices"], "Expected ranked prices for endoscopy"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "эндоскоп" in top_name


def test_price_info_matches_service_homecode(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Анализ A", "serviceHomecode": "5001", "cost": 170},
            {"serviceName": "Анализ B", "serviceHomecode": "5420", "cost": 220},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("5001", {}))

    assert res["prices"], "Expected homecode match"
    assert str(res["prices"][0].get("serviceHomecode") or "") == "5001"


def test_price_info_keeps_service_name_on_city_only_reply(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [{"serviceName": "ЭКГ", "cost": 650}]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Самара", {"service_name": "ЭКГ"}))

    assert res["prices"], "Expected prices when city-only follow-up keeps previous service"
    assert str(res["entities_used"].get("service_name_effective") or "").lower() == "экг"


def test_price_info_keeps_entity_service_name_when_query_candidate_is_noisy(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_price,
        "load_price_by_region",
        lambda _region_id: [{"serviceName": "УЗДГ сосудов шеи", "cost": 1800}],
    )

    res = run(
        svc.price_info(
            "День добрый!\nСамара. Победы 83.\nНам нужно пройти обследование  уздг сосудов шеи и сдать кровь на ЛПНП .\nЭто возможно сделать по данному адресу?\nКакова стоимость услуг?",
            {"service_name": "УЗДГ сосудов шеи", "secondary_intents": ["TEST_ASSIST", "ADDRESS"]},
        )
    )

    assert res["prices"], "Expected retail match for grounded diagnostic service"
    assert str(res["entities_used"].get("service_name_effective") or "") == "УЗДГ сосудов шеи"


def test_resolve_price_service_name_from_catalog_matches_biochemistry():
    rows = [
        {"serviceName": "Бодилифт 1 категория", "cost": 400000},
        {"serviceName": "Биохимия крови", "cost": 2690},
    ]

    resolved = resolve_price_service_name_from_catalog("биохимия крови", rows=rows)

    assert resolved == "Биохимия крови"


def test_resolve_price_service_name_from_catalog_matches_alat_alias():
    rows = [
        {"serviceName": "АсАТ", "cost": 190},
        {"serviceName": "АлАТ", "cost": 190},
    ]

    resolved = resolve_price_service_name_from_catalog("Стоимость АЛТ", rows=rows)

    assert resolved == "АлАТ"


def test_resolve_price_service_name_from_catalog_matches_oak_alias():
    rows = [
        {"serviceName": "Общий анализ мочи", "cost": 240},
        {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "cost": 490},
    ]

    resolved = resolve_price_service_name_from_catalog("ОАК", rows=rows)

    assert resolved == "Общий анализ крови (Le, Er, Hb, СОЭ)"


def test_resolve_price_service_name_from_catalog_returns_none_for_noisy_generic_query():
    rows = [
        {"serviceName": "Лечение периодонтита более чем 3 канального зуба", "cost": 2500},
        {"serviceName": "Ген рецептора витамина D (VDR). Выявление мутации G283A", "cost": 980},
    ]

    resolved = resolve_price_service_name_from_catalog(
        "Здравствуйте! Скажите пожалуйста, как рассчитывается стоимость анализы!? По выходным дешевле чем в будни?",
        rows=rows,
    )

    assert resolved is None


def test_resolve_price_service_name_from_catalog_prefers_new_price_query_over_stale_context():
    rows = [
        {
            "serviceName": "Семейная гиперхолестеринемия, ген LDLR (Familial Hypercholesterolemia, Gene LDLR)",
            "cost": 8990,
        },
        {
            "serviceName": "Cito Общий анализ крови (Le, Er,Hb)",
            "cost": 580,
        },
    ]

    resolved = resolve_price_service_name_from_catalog(
        "Какова стоимость общего анализа крови?",
        current_service_name="холестерин",
        rows=rows,
    )

    assert resolved == "Cito Общий анализ крови (Le, Er,Hb)"


def test_resolve_price_service_name_from_catalog_consult_specialty_ignores_stale_context():
    rows = [
        {"serviceName": "Прием (осмотр, консультация) врача-фониатра", "cost": 4000},
        {"serviceName": "Прием (осмотр, консультация) врача-терапевта", "cost": 2500},
        {"serviceName": "Прием (осмотр, консультация) врача-уролога первичный", "cost": 2000},
    ]

    resolved = resolve_price_service_name_from_catalog(
        "Какая стоимость приема уролога?",
        current_service_name="Прием (осмотр, консультация) врача-фониатра",
        rows=rows,
    )

    assert resolved == "Прием (осмотр, консультация) врача-уролога первичный"


def test_resolve_price_service_name_from_catalog_consult_specialty_cardio():
    rows = [
        {"serviceName": "Прием (осмотр, консультация) врача-фониатра", "cost": 4000},
        {"serviceName": "Прием (осмотр, консультация) врача-кардиолога первичный", "cost": 1200},
    ]

    resolved = resolve_price_service_name_from_catalog(
        "Какова стоимость приема у кардиолога?",
        current_service_name="Прием (осмотр, консультация) врача-фониатра",
        rows=rows,
    )

    assert resolved == "Прием (осмотр, консультация) врача-кардиолога первичный"


def test_extract_price_service_from_query_consult_without_specialty_returns_none():
    assert _svc_extract_price_service_from_query("Сколько стоит консультация?") is None


def test_select_effective_price_service_name_keeps_clean_entity_over_noisy_query():
    selected = _svc_select_effective_price_service_name(
        "Rv-вич гепатит",
        "сдачи анализа rv вич гепатит г",
    )

    assert selected == "Rv-вич гепатит"


def test_select_effective_price_service_name_keeps_doctor_entity_when_query_has_address_noise():
    selected = _svc_select_effective_price_service_name(
        "УЗДГ сосудов шеи",
        "победы 83 нам обследование уздг сосудов шеи кровь",
    )

    assert selected == "УЗДГ сосудов шеи"


def test_select_effective_price_service_name_prefers_new_query_over_stale_context():
    selected = _svc_select_effective_price_service_name(
        "ЭКГ",
        "холестерин",
    )

    assert selected == "холестерин"


def test_select_effective_price_service_name_accepts_more_specific_query_variant():
    selected = _svc_select_effective_price_service_name(
        "УЗДГ сосудов",
        "УЗДГ сосудов шеи",
    )

    assert selected == "УЗДГ сосудов шеи"


def test_service_name_matches_specialty_does_not_match_substring_therapist_in_hirudotherapist():
    assert _svc_service_name_matches_specialty(
        "Прием (осмотр, консультация) гирудотерапевта первичный",
        "терапевт",
    ) is False


def test_service_name_matches_specialty_rejects_hybrid_specialty_for_direct_query():
    assert _svc_service_name_matches_specialty(
        "Прием (осмотр, консультация) врача-кардиолога-ревматолога первичный",
        "кардиолог",
    ) is False


def test_price_info_resolves_biochemistry_catalog_query(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Бодилифт 1 категория", "cost": 400000},
            {"serviceName": "Биохимия крови", "cost": 2690},
            {"serviceName": "Биохимический анализ кала", "cost": 1980},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Сколько стоит биохимия крови?", {}))

    assert res["prices"], "Expected prices from catalog-grounded price matching"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "биохимия крови" in top_name


def test_price_info_resolves_alat_alias_from_catalog(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "АсАТ", "cost": 190},
            {"serviceName": "АлАТ", "cost": 190},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Стоимость АЛТ", {}))

    assert res["prices"], "Expected price row for АлАТ/АЛТ alias"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "алат" in top_name


def test_price_info_drops_stale_prepare_service_for_new_price_query(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {
                "serviceName": "Семейная гиперхолестеринемия, ген LDLR (Familial Hypercholesterolemia, Gene LDLR)",
                "cost": 8990,
            },
            {
                "serviceName": "Cito Общий анализ крови (Le, Er,Hb)",
                "cost": 580,
            },
            {
                "serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)",
                "cost": 390,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Какова стоимость общего анализа крови?", {"service_name": "холестерин"}))

    assert res["prices"], "Expected OAK prices instead of stale cholesterol context"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "общий анализ крови" in top_name
    assert "холестерин" not in str(res["entities_used"].get("service_name_effective") or "").lower()


def test_price_info_general_oak_returns_base_variants_without_special_modifiers(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Cito Общий анализ крови (Le, Er,Hb)", "serviceHomecode": "802", "cost": 580},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
                "serviceHomecode": "501",
                "cost": 520,
            },
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты) капиллярная кровь",
                "serviceHomecode": "501к",
                "cost": 470,
            },
            {
                "serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ) капиллярная кровь",
                "serviceHomecode": "502к",
                "cost": 380,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Сколько стоит общий анализ крови?", {}))

    assert res.get("service_kind") == "family_query"
    assert res.get("visible_limit") == 2
    variants = res.get("family_variants") or []
    assert isinstance(variants, list) and len(variants) >= 2
    visible_codes = [str(row.get("serviceHomecode") or "") for row in variants[:2]]
    assert visible_codes == ["502", "501"]
    assert res.get("remaining_count", 0) >= 1


def test_price_info_keeps_cito_variant_when_user_requests_it_explicitly(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Cito Общий анализ крови (Le, Er,Hb)", "serviceHomecode": "802", "cost": 580},
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("Сколько стоит cito общий анализ крови?", {}))

    assert res["prices"]
    assert str(res["prices"][0].get("serviceHomecode") or "") == "802"


def test_price_info_resolves_mixed_ecg_question_to_catalog_service(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Бодилифт 1 категория", "cost": 400000},
            {"serviceName": "ЭКГ", "cost": 650},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("как пройти экг и его стоимость?", {}))

    assert res["prices"], "Expected price row for mixed ECG price/address wording"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert top_name == "экг"


def test_price_info_generic_uzi_returns_clarify():
    svc = Services()

    res = run(svc.price_info("Сколько стоит УЗИ?", {}))

    assert res.get("prices") == []
    assert "узи брюшной полости" in str(res.get("clarify_text") or "").lower()
    assert "молочной железы" in str(res.get("clarify_text") or "").lower()


def test_service_bundle_info_generic_uzi_returns_clarify():
    svc = Services()

    res = run(svc.service_bundle_info("Сколько стоит УЗИ?", {}))

    assert res.get("retail_prices") == []
    assert res.get("doctors") == []
    assert "узи брюшной полости" in str(res.get("clarify_text") or "").lower()
    assert "молочной железы" in str(res.get("clarify_text") or "").lower()


def test_format_price_for_patient_prefers_clarify_text():
    text = format_price_for_patient(
        {
            "prices": [],
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
        },
        {},
    )

    assert "узи брюшной полости" in text.lower()
    assert "молочной железы" in text.lower()


def test_format_service_bundle_for_patient_prefers_clarify_text():
    text = format_service_bundle_for_patient(
        {
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
            "retail_prices": [],
            "doctors": [],
        },
        {},
    )

    assert "узи брюшной полости" in text.lower()
    assert "молочной железы" in text.lower()


def test_format_service_bundle_for_patient_lists_multiple_lab_price_variants():
    text = format_service_bundle_for_patient(
        {
            "service_name": "общий анализ крови",
            "service_kind": "lab",
            "retail_prices": [
                {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
                {
                    "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
                    "serviceHomecode": "501",
                    "cost": 490,
                },
            ],
            "doctors": [],
            "show_prepare": False,
        },
        {},
    )

    low = text.lower()
    assert "розничные варианты" in low
    assert "общий анализ крови (le, er, hb, соэ) — 390 руб." in low
    assert "общий анализ крови (полный)" in low
    assert "490 руб." in low


def test_format_service_bundle_for_patient_limits_lab_variants_to_five():
    payload = {
        "service_name": "анализ крови",
        "service_kind": "lab",
        "retail_prices": [
            {"serviceName": f"Вариант {idx}", "cost": 100 + idx}
            for idx in range(1, 7)
        ],
        "doctors": [],
        "show_prepare": False,
    }

    text = format_service_bundle_for_patient(payload, {})

    assert "Вариант 1" in text
    assert "Вариант 5" in text
    assert "Вариант 6" not in text


def test_price_info_plomba_does_not_match_cmv(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Цитомегаловирус [п\\кол.]", "cost": 560, "serviceHomecode": "cmv"},
            {"serviceName": "Постановка пломбы светоотверждаемой", "cost": 4500, "serviceHomecode": "dent1"},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость постановки пломбы", {}))

    assert res["prices"]
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "пломб" in top_name
    assert "цитомегаловирус" not in top_name


def test_service_bundle_info_alat_classified_as_lab_without_doctors(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {
                "serviceName": "АлАТ (аланинаминотрансфераза)",
                "serviceHomecode": "184",
                "deadline": "1-2",
                "cost": 320,
            }
        ]

    def fake_doctor_prices():
        return [
            {
                "doctorId": 11,
                "serviceName": "Прием терапевта первичный",
                "serviceHomecode": "999.1",
                "cost": 2000,
            }
        ]

    async def fake_doctors():
        return [{"id": 11, "fio": "Иванов Иван Иванович", "ord": 1, "regions": ["г. Самара, пр. Ленина, 5"]}]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)

    res = run(svc.service_bundle_info("стоимость алат", {}))

    assert res["service_kind"] == "lab"
    assert res["doctors"] == []
    assert res["retail_prices"]
    assert "алат" in str(res["retail_prices"][0].get("serviceName") or "").lower()


def test_service_bundle_info_ecg_skips_doctors_even_with_exact_link(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {
                "serviceName": "ЭКГ",
                "serviceHomecode": "39.10.9",
                "cost": 650,
            }
        ]

    def fake_doctor_prices():
        return [
            {
                "doctorId": 21,
                "serviceName": "ЭКГ",
                "serviceHomecode": "39.10.9",
                "cost": 650,
            }
        ]

    async def fake_doctors():
        return [{"id": 21, "fio": "Просвиров Евгений Юрьевич", "ord": 1, "regions": ["г. Самара, пр. Ленина, 5"]}]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_doctors)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)

    res = run(svc.service_bundle_info("стоимость экг", {}))

    assert res["service_kind"] == "diagnostic_no_doctor"
    assert res["doctors"] == []


def test_price_info_hepatitis_returns_family_query_with_hint(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": f"Гепатит вариант {idx}", "serviceHomecode": f"hep-{idx}", "cost": 300 + idx}
            for idx in range(1, 13)
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость гепатита", {}))
    text = format_price_for_patient(res, {})

    assert res["service_kind"] == "family_query"
    assert len(res["family_variants"]) == 12
    assert res["remaining_count"] == 2
    assert "напишите: \"все\"" in text.lower()
    assert "\n\nПо вашему запросу найдено еще 2 вариантов." in text
    assert "Гепатит вариант 10" in text
    assert "Гепатит вариант 11" not in text


def test_price_info_hiv_returns_multiple_relevant_variants(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Экспресс-тест для определения ВИЧ", "serviceHomecode": "hiv-exp", "cost": 850},
            {"serviceName": "Кровь на ВИЧ", "serviceHomecode": "hiv-blood", "cost": 410},
            {"serviceName": "АлАТ", "serviceHomecode": "alt", "cost": 320},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость вич", {}))

    assert len(res.get("prices") or []) >= 2
    names = [str(row.get("serviceName") or "").lower() for row in (res.get("prices") or []) if isinstance(row, dict)]
    assert any("кровь на вич" in name for name in names)
    assert any("экспресс" in name for name in names)


def test_resolve_price_service_name_prefers_adult_uzi_over_child():
    rows = [
        {
            "serviceName": "УЗИ печени и желчного пузыря (детское)",
            "serviceHomecode": "u1",
            "cost": 2100,
        },
        {
            "serviceName": "Ультразвуковое исследование печени и желчного пузыря",
            "serviceHomecode": "u2",
            "cost": 1800,
        },
    ]

    resolved = resolve_price_service_name_from_catalog("стоимость узи печени", rows=rows)

    assert resolved == "Ультразвуковое исследование печени и желчного пузыря"


def test_extract_specialty_from_text_prefers_compound_traumatologist_orthopedist():
    extracted = _svc_extract_specialty_from_text("стоимость приема травматолога ортопеда")

    assert extracted == "травматолог-ортопед"


def test_price_info_vitamin_d_prefers_non_genetic_variant(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {
                "serviceName": "Ген рецептора витамина D (VDR). Выявление мутации G283A",
                "serviceHomecode": "gen-vdr",
                "cost": 3100,
            },
            {
                "serviceName": "Витамин D суммарный (25-OH D2 и D3, общий результат)",
                "serviceHomecode": "vit-d",
                "cost": 1600,
            },
            {
                "serviceName": "Витамин B12",
                "serviceHomecode": "vit-b12",
                "cost": 690,
            },
            {
                "serviceName": "«Витамин D – COMBO»",
                "serviceHomecode": "vit-d-combo",
                "cost": 2280,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость витамина д", {}))

    assert res["prices"]
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "витамин d суммарный" in top_name
    assert "мутац" not in top_name


def test_price_info_generic_vitamins_returns_family_query_variants(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Расширенный комплексный анализ на витамины (A, D, E)", "serviceHomecode": "vit-complex", "cost": 22500},
            {"serviceName": "Витамин D суммарный", "serviceHomecode": "vit-d", "cost": 1400},
            {"serviceName": "Витамин B12", "serviceHomecode": "vit-b12", "cost": 690},
            {"serviceName": "Ген рецептора витамина D (VDR). Выявление мутации G283A", "serviceHomecode": "vdr", "cost": 980},
            {"serviceName": "«Витамин D – COMBO»", "serviceHomecode": "vit-combo", "cost": 2280},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("анализы на витамины", {}))

    assert res.get("service_kind") == "family_query"
    variants = [row for row in (res.get("family_variants") or []) if isinstance(row, dict)]
    assert len(variants) >= 3
    names = [str(row.get("serviceName") or "").lower() for row in variants]
    assert any("витамин d" in name for name in names)
    assert any("витамин b12" in name for name in names)
    assert not any("мутац" in name for name in names)


def test_price_info_generic_family_mode_is_data_driven_without_root_regex(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Аллерген береза, IgE", "serviceHomecode": "alg-1", "cost": 520},
            {"serviceName": "Аллерген ольха, IgE", "serviceHomecode": "alg-2", "cost": 530},
            {"serviceName": "Аллерген полынь, IgE", "serviceHomecode": "alg-3", "cost": 540},
            {"serviceName": "АлАТ", "serviceHomecode": "alt", "cost": 320},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость аллергена", {}))

    assert res.get("service_kind") == "family_query"
    names = [str(row.get("serviceName") or "").lower() for row in (res.get("family_variants") or []) if isinstance(row, dict)]
    assert any("береза" in name for name in names)
    assert any("ольха" in name for name in names)
    assert any("полынь" in name for name in names)


def test_price_info_tooth_removal_returns_family_query_variants(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Простое удаление зуба", "serviceHomecode": "tooth-1", "cost": 3000},
            {"serviceName": "Сложное удаление зуба", "serviceHomecode": "tooth-2", "cost": 6000},
            {"serviceName": "Удаление зуба мудрости", "serviceHomecode": "tooth-3", "cost": 10000},
            {"serviceName": "Удаление серной пробки", "serviceHomecode": "ent-1", "cost": 1200},
            {"serviceName": "Удаление полипа уретры", "serviceHomecode": "uro-1", "cost": 4500},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость удаления зуба", {}))

    assert res.get("service_kind") == "family_query"
    names = [str(row.get("serviceName") or "").lower() for row in (res.get("family_variants") or []) if isinstance(row, dict)]
    assert any("простое удаление зуба" in name for name in names)
    assert any("сложное удаление зуба" in name for name in names)
    assert not any("серной пробки" in name for name in names)
    assert not any("полипа уретры" in name for name in names)


def test_price_info_oak_keeps_canonical_variants_over_noise(monkeypatch):
    """ОАК-запрос: канонические строки должны вытеснять шум вроде «группа крови»."""

    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Группа крови", "serviceHomecode": "g1", "cost": 290},
            {"serviceName": "Медь в крови", "serviceHomecode": "cu", "cost": 850},
            {"serviceName": "Хром в крови", "serviceHomecode": "cr", "cost": 960},
            {"serviceName": "Свинец в крови", "serviceHomecode": "pb", "cost": 910},
            {"serviceName": "Биохимия крови", "serviceHomecode": "bx", "cost": 2690},
            {"serviceName": "Гистамин в крови", "serviceHomecode": "his", "cost": 2400},
            {"serviceName": "Серотонин в крови", "serviceHomecode": "ser", "cost": 2200},
            {"serviceName": "ЗППП: анализ крови", "serviceHomecode": "sti", "cost": 2900},
            {"serviceName": "Коэнзим Q10 в крови", "serviceHomecode": "q10", "cost": 3500},
            {"serviceName": "Взятие крови из вены", "serviceHomecode": "vene", "cost": 190},
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
                "serviceHomecode": "501",
                "cost": 490,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость общего анализа крови", {}))

    assert res.get("service_kind") == "family_query"
    assert res.get("visible_limit") == 2
    variants = res.get("family_variants") or []
    assert len(variants) >= 2
    visible_codes = [str(row.get("serviceHomecode") or "") for row in variants[:2]]
    assert visible_codes == ["502", "501"]
    assert res.get("remaining_count", 0) >= 1
    assert '"все"' in str(res.get("show_all_hint") or "")


def test_price_info_oak_alias_returns_canonical_variants(monkeypatch):
    """Короткий alias «ОАК» должен попадать на тот же canonical-путь."""

    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Группа крови", "serviceHomecode": "g1", "cost": 290},
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
                "serviceHomecode": "501",
                "cost": 490,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость ОАК", {}))

    variants = res.get("family_variants") or []
    visible_codes = [str(row.get("serviceHomecode") or "") for row in variants[:2]]
    assert visible_codes == ["502", "501"]


def test_price_info_oak_show_all_expands_to_full_list(monkeypatch):
    """Follow-up «все» должен раскрывать остальные варианты ОАК (cito/капиллярная)."""

    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Cito Общий анализ крови (Le, Er, Hb)", "serviceHomecode": "802", "cost": 580},
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ) капиллярная кровь", "serviceHomecode": "502к", "cost": 380},
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
                "serviceHomecode": "501",
                "cost": 490,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    first = run(svc.price_info("стоимость общего анализа крови", {}))
    assert first.get("service_kind") == "family_query"
    family_ctx = {
        "_price_family_context": {
            "service_name": first.get("service_name"),
            "family_variants": first.get("family_variants"),
            "visible_limit": first.get("visible_limit"),
        }
    }

    expanded = run(svc.price_info("все", family_ctx))
    assert expanded.get("showing_all") is True
    all_codes = {str(row.get("serviceHomecode") or "") for row in expanded.get("family_variants") or []}
    assert {"502", "501", "802", "502к"}.issubset(all_codes)


def test_price_info_multi_service_returns_grouped_prices(monkeypatch):
    """Мульти-услуговый запрос должен отдать цены каждой услуги в одном ответе."""

    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
            {"serviceName": "Кровь на ВИЧ", "serviceHomecode": "hiv", "cost": 410},
            {"serviceName": "Гепатит B (HBsAg, качественный)", "serviceHomecode": "hep-b", "cost": 520},
            {"serviceName": "Группа крови", "serviceHomecode": "g1", "cost": 290},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость гепатита в, общего анализа крови, вич", {}))

    assert str(res.get("note") or "") == "price_multi_service"
    assert res.get("service_kind") == "family_query"
    names = [str(row.get("serviceName") or "").lower() for row in res.get("family_variants") or []]
    assert any("вич" in name for name in names)
    assert any("гепатит" in name for name in names)
    assert any("общий анализ крови" in name for name in names)


def test_price_info_single_service_query_bypasses_multi_splitter(monkeypatch):
    """Одиночный запрос не должен попадать в multi-путь, даже если содержит «и»."""

    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "УЗИ брюшной полости и почек", "serviceHomecode": "uzi-bp", "cost": 1800},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость узи брюшной полости и почек", {}))

    assert str(res.get("note") or "") != "price_multi_service"


def test_split_price_query_items_splits_on_common_delimiters():
    from messengers_router.services._prices_helpers import _split_price_query_items

    assert _split_price_query_items("стоимость гепатит в, оак, вич") == ["гепатит в", "оак", "вич"]
    assert _split_price_query_items("цена вич и гепатит") == ["вич", "гепатит"]
    assert _split_price_query_items("стоимость общего анализа крови") == ["общего анализа крови"]


def test_price_info_trauma_orthopedist_prefers_base_consultation_over_kmn_uzi(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {
                "serviceName": "Прием травматолога-ортопеда, к.м.н. совместно с УЗИ одной группы суставов",
                "serviceHomecode": "14.1.13",
                "cost": 3500,
            },
            {
                "serviceName": "Приём (осмотр, консультация) травматолога-ортопеда первичный",
                "serviceHomecode": "14.1.1",
                "cost": 2700,
            },
            {
                "serviceName": "Прием (осмотр, консультация) врача-травматолога-ортопеда, к.м.н первичный",
                "serviceHomecode": "14.1.4",
                "cost": 3500,
            },
            {
                "serviceName": "Приём (осмотр, консультация) травматолога-ортопеда повторный (в течение месяца)",
                "serviceHomecode": "14.1.2",
                "cost": 2200,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость приема травматолога ортопеда", {}))

    assert res["prices"]
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "травматолога-ортопеда первичный" in top_name
    assert "совместно с узи" not in top_name


def test_price_info_surgeon_prefers_primary_before_repeat_and_home(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {
                "serviceName": "Прием (осмотр, консультация) врача-хирурга повторный",
                "serviceHomecode": "30.2",
                "cost": 2000,
            },
            {
                "serviceName": "Прием (осмотр, консультация) врача-хирурга первичный",
                "serviceHomecode": "30.1",
                "cost": 2500,
            },
            {
                "serviceName": "Прием (осмотр, консультация) врача-хирурга, к.м.н первичный",
                "serviceHomecode": "30.4",
                "cost": 3500,
            },
            {
                "serviceName": "Прием (осмотр, консультация) врача-хирурга на дому (1 зона)",
                "serviceHomecode": "42.1.6.1",
                "cost": 5000,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость приема хирурга", {}))

    assert res["prices"]
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "первичный" in top_name
    assert "повторный" not in top_name
    assert "на дому" not in top_name


def test_price_info_thigh_lift_family_does_not_leak_other_plastic_or_unrelated_services(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Подтяжка бедер 1 категория", "serviceHomecode": "hip-1", "cost": 150000},
            {"serviceName": "Подтяжка бедер 2 категория", "serviceHomecode": "hip-2", "cost": 250000},
            {"serviceName": "Подтяжка ягодиц 1 категория", "serviceHomecode": "butt-1", "cost": 150000},
            {"serviceName": "Гепатит В - HBsAg", "serviceHomecode": "hep", "cost": 410},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.price_info("стоимость подтяжки бедер", {}))

    rows = res.get("family_variants") or res.get("prices") or []
    names = [str(row.get("serviceName") or "").lower() for row in rows if isinstance(row, dict)]
    assert names
    assert any("подтяжка бедер" in name for name in names)
    assert not any("подтяжка ягодиц" in name for name in names)
    assert not any("гепатит" in name for name in names)


def test_classifier_price_family_show_all_followup_returns_price():
    decision = run(
        classifier_mod.deterministic_rule_decision(
            "все",
            {
                "_last_label": "PRICE",
                "_price_family_context": {
                    "service_name": "гепатит",
                    "family_variants": [{"serviceName": "Гепатит A", "cost": 300}],
                    "visible_limit": 10,
                },
            },
        )
    )

    assert decision is not None
    assert decision.label == "PRICE"
    assert "rule_price_family_show_all" in decision.flags


def test_build_price_response_stores_and_clears_price_family_context():
    state = SessionState(session_id="sid")

    family = build_price_response(
        "PRICE",
        Evidence(
            {
                "price": {
                    "service_kind": "family_query",
                    "service_name": "гепатит",
                    "family_variants": [{"serviceName": "Гепатит A", "cost": 300}],
                    "visible_limit": 10,
                    "showing_all": False,
                }
            }
        ),
        state,
    )
    assert family is not None
    assert "_price_family_context" in state.last_entities

    single = build_price_response(
        "PRICE",
        Evidence({"price": {"prices": [{"serviceName": "АлАТ", "cost": 320}]}}),
        state,
    )
    assert single is not None
    assert "_price_family_context" not in state.last_entities


def test_ambiguous_price_kind_llm_fallback_respects_exact_link(monkeypatch):
    async def fake_generate(*_args, **_kwargs):
        return '{"kind":"procedure_with_doctor","reason":"exact service match"}'

    monkeypatch.setattr(llm_runtime_mod, "generate_text", fake_generate)

    good = run(
        _svc_resolve_ambiguous_price_kind_with_llm(
            "стоимость узи печени",
            [{"serviceName": "УЗИ печени"}],
            has_exact_doctor_link=True,
            runtime_llm_mode="hybrid",
        )
    )
    bad = run(
        _svc_resolve_ambiguous_price_kind_with_llm(
            "стоимость узи печени",
            [{"serviceName": "УЗИ печени"}],
            has_exact_doctor_link=False,
            runtime_llm_mode="hybrid",
        )
    )

    assert good == "procedure_with_doctor"
    assert bad == "ambiguous"


def test_extract_price_service_from_query_strips_politeness_tail():
    assert _svc_extract_price_service_from_query("стоимость экг подскажите") == "экг"


def test_extract_price_service_from_query_strips_gratitude_prefix():
    assert (
        _svc_extract_price_service_from_query("Спасибо\nПодскажи стоимость общего анализа крови")
        == "общего анализа крови"
    )


def test_price_info_resolves_doctor_name_to_doctor_prices(monkeypatch):
    svc = Services()

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 11,
                "fio": "Иванов Иван Иванович",
                "ord": 1,
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_doctor_prices():
        return [
            {
                "doctorId": 11,
                "fio": "Иванов Иван Иванович",
                "serviceName": "УЗИ брюшной полости",
                "cost": 1700,
            },
            {
                "doctorId": 12,
                "fio": "Петров Петр Петрович",
                "serviceName": "УЗИ брюшной полости",
                "cost": 1900,
            },
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(
        svc.price_info(
            "Сколько стоит УЗИ брюшной полости у Иванова",
            {"doctor_name": "Иванова", "service_name": "УЗИ брюшной полости"},
        )
    )

    assert res["prices"], "Expected doctor-specific prices after doctor_name resolution"
    assert all(int(p.get("doctorId") or 0) == 11 for p in res["prices"])


def test_price_info_resolves_doctor_name_from_query_when_entities_empty(monkeypatch):
    svc = Services()

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 11,
                "fio": "Иванов Иван Иванович",
                "ord": 1,
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    async def fake_samara_tokens():
        return {"г. самара, пр. ленина, 5"}

    def fake_doctor_prices():
        return [
            {
                "doctorId": 11,
                "fio": "Иванов Иван Иванович",
                "serviceName": "УЗИ брюшной полости",
                "cost": 1700,
            },
            {
                "doctorId": 12,
                "fio": "Петров Петр Петрович",
                "serviceName": "УЗИ брюшной полости",
                "cost": 1900,
            },
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc, "_samara_region_tokens", fake_samara_tokens)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(
        svc.price_info(
            "Сколько стоит УЗИ брюшной полости у Иванова",
            {"service_name": "УЗИ брюшной полости"},
        )
    )

    assert res["prices"], "Expected doctor-specific prices resolved from query text"
    assert all(int(p.get("doctorId") or 0) == 11 for p in res["prices"])


def test_price_info_doctor_query_overrides_stale_service_name(monkeypatch):
    svc = Services()

    def fake_doctor_prices():
        return [
            {
                "doctorId": 11,
                "fio": "Джовмардов Саид Саидович",
                "serviceName": "Консультация врача-уролога",
                "cost": 2000,
            },
            {
                "doctorId": 11,
                "fio": "Джовмардов Саид Саидович",
                "serviceName": "УЗИ желудка",
                "cost": 1100,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(
        svc.price_info(
            "А сколько стоит прием у Джовмардова Саида?",
            {
                "doctor_id": 11,
                "doctor_name": "Джовмардов",
                "service_name": "УЗИ желудка",
            },
        )
    )

    assert res["prices"], "Expected doctor-specific prices"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "консультац" in top_name or "прием" in top_name


def test_price_info_extracts_service_from_free_form_doctor_price_query(monkeypatch):
    svc = Services()

    def fake_doctor_prices():
        return [
            {
                "doctorId": 11,
                "fio": "Султанова Алина Рустамовна",
                "serviceName": "Удаление серной пробки",
                "cost": 1600,
            },
            {
                "doctorId": 11,
                "fio": "Султанова Алина Рустамовна",
                "serviceName": "УЗИ желудка",
                "cost": 1100,
            },
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(
        svc.price_info(
            "Цена удаления серной пробки у Султановой",
            {
                "doctor_id": 11,
                "doctor_name": "Султанова",
                "service_name": "УЗИ желудка",
            },
        )
    )

    assert res["prices"], "Expected doctor-specific prices"
    top_name = str(res["prices"][0].get("serviceName") or "").lower()
    assert "удаление серной пробки" in top_name


def test_address_info(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)

    res = run(svc.address_info("Ленина", {}))

    assert any("Ленина" in a for a in res["addresses"])


def test_address_info_appointment_mode_filters_to_doctor_branches(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, ул. Гастелло, 46", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара"},
        ]

    async def fake_ensure_doctors_cache():
        return [
            {
                "fio": "Иванов Иван",
                "regions": ["г. Самара, ул. Победы, 83"],
            }
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)

    res = run(svc.address_info("Самара", {"city": "Самара", "__appointment_mode": True}))

    assert any("Победы, 83" in a for a in res["addresses"])
    assert not any("Гастелло, 46" in a for a in res["addresses"])


def test_address_info_does_not_treat_generic_query_as_branch_filter(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)

    res = run(svc.address_info("Анализы Самара", {}))

    assert any("Ленина" in a for a in res["addresses"])
    assert any("Победы" in a for a in res["addresses"])


def test_address_info_filters_regions_for_ekg(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара", "ecg": True},
            {"id": 2, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара", "ecg": False},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)

    res = run(svc.address_info("Где пройти ЭКГ?", {"service_name": "ЭКГ"}))

    assert any("Ленина" in a for a in res["addresses"])
    assert not any("Победы" in a for a in res["addresses"])


def test_address_info_appointment_mode_uses_price_units_care_setting_addresses(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Ново-Садовая, 106, кор. 82", "city": "Самара"},
            {"id": 3, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара"},
        ]

    async def fake_ensure_doctors_cache():
        return []

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Тонзиллотомия", "cost": 20000, "priceUnitId": 195},
            {"serviceName": "Тонзиллотомия 2 категория", "cost": 50000, "priceUnitId": 489},
        ]

    def fake_price_units():
        return [
            {"id": 311, "parent": 190, "name": "Дневной стационар"},
            {"id": 312, "parent": 190, "name": "Круглосуточный стационар"},
            {"id": 195, "parent": 311, "name": "Дневной стационар (Оториноларингология)"},
            {"id": 489, "parent": 312, "name": "Круглосуточный стационар (Оториноларингология)"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)
    monkeypatch.setattr(svc_mod.api_price, "load_price_units", fake_price_units)

    res = run(
        svc.address_info(
            "хочу записаться на тонзиллотомию",
            {"service_name": "Тонзиллотомия", "__appointment_mode": True, "city": "Самара"},
        )
    )

    assert res["addresses"] == [
        "г. Самара, пр. Ленина, 5",
        "г. Самара, ул. Ново-Садовая, 106, кор. 82",
    ]
    assert not any("Победы" in a for a in res["addresses"])


def test_address_info_price_units_keeps_precise_inpatient_address_when_regions_have_generic_city(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "name": "Самара", "city": "Самара"},
        ]

    async def fake_ensure_doctors_cache():
        return []

    def fake_price_by_region(_region_id):
        return [
            {"serviceName": "Септопластика", "cost": 50000, "priceUnitId": 195},
            {"serviceName": "Септопластика 2 категория", "cost": 70000, "priceUnitId": 489},
        ]

    def fake_price_units():
        return [
            {"id": 311, "parent": 190, "name": "Дневной стационар"},
            {"id": 312, "parent": 190, "name": "Круглосуточный стационар"},
            {"id": 195, "parent": 311, "name": "Дневной стационар (Оториноларингология)"},
            {"id": 489, "parent": 312, "name": "Круглосуточный стационар (Оториноларингология)"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)
    monkeypatch.setattr(svc_mod.api_price, "load_price_units", fake_price_units)

    res = run(
        svc.address_info(
            "где можно сделать?",
            {"service_name": "Септопластика"},
        )
    )

    assert res["addresses"] == [
        "г. Самара, пр. Ленина, 5",
        "г. Самара, ул. Ново-Садовая, 106, кор. 82",
    ]
    assert any(
        str(branch.get("address") or "") == "г. Самара, ул. Ново-Садовая, 106, кор. 82"
        for branch in (res.get("branches") or [])
        if isinstance(branch, dict)
    )


def test_address_info_uses_family_expansion_when_top_price_rows_hide_second_care_setting(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Ново-Садовая, 106, кор. 82", "city": "Самара"},
            {"id": 3, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара"},
        ]

    def fake_price_by_region(_region_id):
        day_rows = [
            {
                "serviceName": f"Тонзиллэктомия {idx} категория",
                "serviceHomecode": f"tonsil-day-{idx}",
                "cost": 20000 + idx,
                "priceUnitId": 195,
            }
            for idx in range(1, 11)
        ]
        inpatient_rows = [
            {
                "serviceName": "Тонзиллэктомия двусторонняя",
                "serviceHomecode": "tonsil-inpatient-1",
                "cost": 80000,
                "priceUnitId": 489,
            }
        ]
        return day_rows + inpatient_rows

    def fake_price_units():
        return [
            {"id": 190, "parent": None, "name": "Медицинская помощь"},
            {"id": 311, "parent": 190, "name": "Дневной стационар"},
            {"id": 312, "parent": 190, "name": "Круглосуточный стационар"},
            {"id": 195, "parent": 311, "name": "Дневной стационар (Оториноларингология)"},
            {"id": 489, "parent": 312, "name": "Круглосуточный стационар (Оториноларингология)"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)
    monkeypatch.setattr(svc_mod.api_price, "load_price_units", fake_price_units)

    res = run(svc.address_info("Где можно сделать тонзиллэктомию?", {"service_name": "Тонзиллэктомия"}))

    assert res["addresses"] == [
        "г. Самара, пр. Ленина, 5",
        "г. Самара, ул. Ново-Садовая, 106, кор. 82",
    ]


def test_news_info(monkeypatch):
    svc = Services()

    def fake_news(**_kwargs):
        return [{"title": "Акция", "content": "Описание"}]

    monkeypatch.setattr(svc_mod.meilisearch, "search_news_active", fake_news)

    res = run(svc.news_info("акция", {}))

    assert res["news"], "Expected news hits"


def test_get_branches(monkeypatch):
    svc = Services()

    monkeypatch.setattr(
        svc_mod.api_nayka,
        "site_regions",
        lambda: [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}],
    )

    branches = svc.get_branches()

    assert branches, "Expected branches list"


def test_service_bundle_info_builds_topn_with_availability_and_prepare(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара"},
            {"id": 3, "addressForSite": "г. Самара, ул. Гастелло, 46", "city": "Самара"},
        ]

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 10,
                "fio": "Бета Доктор",
                "ord": 2,
                "specialization": "УЗИ",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "units": ["Диагностика"],
            },
            {
                "id": 20,
                "fio": "Альфа Доктор",
                "ord": 1,
                "specialization": "УЗИ",
                "regions": ["г. Самара, ул. Победы, 83"],
                "units": ["Диагностика"],
            },
            {
                "id": 30,
                "fio": "Гамма Доктор",
                "ord": 3,
                "specialization": "УЗИ",
                "regions": ["г. Самара, ул. Гастелло, 46"],
                "units": ["Диагностика"],
            },
        ]

    def fake_retail(_region_id):
        return [{"serviceName": "УЗИ брюшной полости", "cost": 1800}]

    def fake_doctor_prices():
        return [
            {"doctorId": 10, "fio": "Бета Доктор", "serviceName": "УЗИ брюшной полости", "cost": 1700},
            {"doctorId": 20, "fio": "Альфа Доктор", "serviceName": "УЗИ брюшной полости", "cost": 1600},
            {"doctorId": 30, "fio": "Гамма Доктор", "serviceName": "УЗИ брюшной полости", "cost": 1900},
        ]

    def fake_schedule(last_name, _region_name=None):
        if str(last_name).lower().startswith("альфа"):
            return [{"fio": "Альфа Доктор", "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-20", "slots": ["09:00"]}]}}]
        if str(last_name).lower().startswith("бета"):
            return [{"fio": "Бета Доктор", "schedule": {"г. Самара, пр. Ленина, 5": [{"date": "2026-03-20", "slots": []}]}}]
        return []

    async def fake_prepare(_query, _entities):
        return {"prepare": "Натощак 8 часов, воду можно.", "entities_used": {"service_name": "УЗИ брюшной полости"}}

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)
    monkeypatch.setattr(svc_mod.api_nayka, "find_doctor_schedule", fake_schedule)
    monkeypatch.setattr(svc, "test_prepare", fake_prepare)

    res = run(
        svc.service_bundle_info(
            "Сколько стоит УЗИ брюшной полости и как подготовиться?",
            {"service_name": "УЗИ брюшной полости"},
            top_n=2,
        )
    )

    assert res["retail_prices"], "Expected retail prices block"
    assert len(res["doctors"]) == 2, "Expected top-2 doctors"
    assert [d["fio"] for d in res["doctors"]] == ["Альфа Доктор", "Бета Доктор"]
    assert res["doctors"][0]["available"] is True
    assert res["doctors"][1]["available"] is False
    assert res["show_prepare"] is True
    assert "Натощак" in str(res.get("prepare") or "")


def test_service_bundle_info_keeps_service_name_on_city_only_reply(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, ул. Ново-Садовая, 106, кор. 82", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return []

    async def fake_prepare(_query, _entities):
        return {"prepare": "", "entities_used": {"service_name": "ЭКГ"}}

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", lambda _region_id: [{"serviceName": "ЭКГ", "cost": 650}])
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", lambda: [])
    monkeypatch.setattr(svc, "test_prepare", fake_prepare)

    res = run(svc.service_bundle_info("Самара", {"service_name": "ЭКГ"}))

    assert str(res.get("service_name") or "").lower() == "экг"


def test_service_bundle_info_keeps_entity_service_name_when_query_extraction_is_noisy(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return []

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(
        svc_mod.api_price,
        "load_price_by_region",
        lambda _region_id: [{"serviceName": "УЗДГ сосудов шеи", "cost": 1800}],
    )
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", lambda: [])

    res = run(
        svc.service_bundle_info(
            "День добрый!\nСамара. Победы 83.\nНам нужно пройти обследование  уздг сосудов шеи и сдать кровь на ЛПНП .\nЭто возможно сделать по данному адресу?\nКакова стоимость услуг?",
            {"service_name": "УЗДГ сосудов шеи", "secondary_intents": ["TEST_ASSIST", "ADDRESS"]},
        )
    )

    assert str(res.get("service_name") or "") == "УЗДГ сосудов шеи"
    assert str(res.get("entities_used", {}).get("service_name_effective") or "") == "УЗДГ сосудов шеи"


def test_service_bundle_info_compound_price_query_returns_clarify(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return []

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(
        svc_mod.api_price,
        "load_price_by_region",
        lambda _region_id: [
            {"serviceName": "УЗДГ сосудов шеи", "cost": 1800},
            {"serviceName": "ЛПНП", "cost": 450},
        ],
    )
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", lambda: [])

    res = run(
        svc.service_bundle_info(
            "День добрый!\nСамара. Победы 83.\nНам нужно пройти обследование уздг сосудов шеи и сдать кровь на ЛПНП.\nЭто возможно сделать по данному адресу?\nКакова стоимость услуг?",
            {"service_name": "УЗДГ сосудов шеи", "secondary_intents": ["TEST_ASSIST", "ADDRESS"]},
        )
    )

    assert str(res.get("service_kind") or "") == "compound_clarify"
    assert "Вижу в запросе две услуги" in str(res.get("clarify_text") or "")
    assert list(res.get("compound_price_services") or []) == ["УЗДГ сосудов шеи", "ЛПНП"]


def test_resolve_price_service_keeps_primary_in_compound_query():
    # Regression guard: the catalog alias resolver must NOT grab the secondary lab
    # item («ЛПНП») and override an explicit primary that is itself present in the
    # compound utterance — otherwise service_bundle_info loses compound_clarify.
    from messengers_router.services._prices_helpers import (
        resolve_price_service_name_from_catalog,
    )

    rows = [
        {"serviceName": "УЗДГ сосудов шеи", "cost": 1800},
        {"serviceName": "ЛПНП", "cost": 450},
    ]
    got = resolve_price_service_name_from_catalog(
        "обследование уздг сосудов шеи и сдать кровь на ЛПНП, какова стоимость?",
        current_service_name="УЗДГ сосудов шеи",
        rows=rows,
    )
    assert got == "УЗДГ сосудов шеи"


def test_resolve_price_service_still_switches_on_pure_lab_query():
    # The fix must not over-correct: an honest topic-switch to a new service
    # (primary absent from this turn) must still resolve via the alias.
    from messengers_router.services._prices_helpers import (
        resolve_price_service_name_from_catalog,
    )

    rows = [
        {"serviceName": "УЗДГ сосудов шеи", "cost": 1800},
        {"serviceName": "ЛПНП", "cost": 450},
    ]
    got = resolve_price_service_name_from_catalog(
        "а сколько стоит ЛПНП?",
        current_service_name="УЗДГ сосудов шеи",
        rows=rows,
    )
    assert got == "ЛПНП"


def test_price_modifier_regexes_exclude_false_positives():
    # M3 regression: price-modifier regexes were too broad and flagged
    # non-modifier catalog rows (verified against the live price cache).
    from messengers_router.services import _prices_helpers as ph
    from messengers_router.services._common import _normalise_input

    def gen(s):
        return bool(ph._PRICE_GENETIC_ROW_RE.search(_normalise_input(s)))

    def cito(s):
        return bool(ph._PRICE_CITO_ROW_RE.search(_normalise_input(s)))

    def child(s):
        return bool(ph._PRICE_CHILD_ROW_RE.search(_normalise_input(s)))

    # genetic: гент*/гени* are NOT genetic; real genetic tests still are.
    assert not gen("Гентамицин (с60)")
    assert not gen("Гениопластика")
    assert not gen("Удаление генитальных образований")
    assert gen("Анализ генетических полиморфизмов")
    assert gen("Анализ мутаций в гене MPL")
    # cito: «экспресс-тест» is a product name, not a cito surcharge.
    assert not cito("Экспресс-тест Helicobacter pylori")
    assert cito("Cito Общий анализ крови")
    assert cito("Срочное выявление РНК коронавируса")
    # child: «детекция»/«детартрин» are not pediatric; real pediatric rows are.
    assert not child("Детекция мутации V600E в гене BRAF")
    assert child("Детский массаж (старше 3 лет)")
    assert child("аллергены значимые для детей")
    # query side mirrors row side
    assert ph._PRICE_GENETIC_QUERY_RE.search(_normalise_input("генетический анализ"))
    assert not ph._PRICE_CITO_QUERY_RE.search(_normalise_input("экспресс-тест на вич"))
    assert ph._PRICE_CHILD_QUERY_RE.search(_normalise_input("детский прием"))


def test_service_bundle_info_enriches_tonsillotomy_with_care_setting(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [
            {"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"},
            {"id": 2, "addressForSite": "г. Самара, ул. Ново-Садовая, 106, кор. 82", "city": "Самара"},
        ]

    async def fake_ensure_doctors_cache():
        return []

    def fake_retail(_region_id):
        return [
            {"serviceName": "Тонзиллотомия", "cost": 20000, "priceUnitId": 195},
            {"serviceName": "Тонзиллотомия 2 категория", "cost": 50000, "priceUnitId": 489},
        ]

    def fake_doctor_prices():
        return []

    def fake_price_units():
        return [
            {"id": 311, "parent": 190, "name": "Дневной стационар"},
            {"id": 312, "parent": 190, "name": "Круглосуточный стационар"},
            {"id": 195, "parent": 311, "name": "Дневной стационар (Оториноларингология)"},
            {"id": 489, "parent": 312, "name": "Круглосуточный стационар (Оториноларингология)"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)
    monkeypatch.setattr(svc_mod.api_price, "load_price_units", fake_price_units)

    res = run(svc.service_bundle_info("Стоимость тонзиллотомии", {"service_name": "Тонзиллотомия"}))

    variants = [row for row in (res.get("family_variants") or []) if isinstance(row, dict)]
    retail = [row for row in (res.get("retail_prices") or []) if isinstance(row, dict)]
    rows = variants or retail

    assert rows, "Expected price rows with care-setting context"
    assert any("Ленина" in str(row.get("care_setting_address") or "") for row in rows)
    assert any("Ново-Садовая" in str(row.get("care_setting_address") or "") for row in rows)


def test_service_bundle_info_skips_prepare_for_consult_service(monkeypatch):
    svc = Services()
    prepare_called = {"v": False}

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return []

    def fake_retail(_region_id):
        return [{"serviceName": "Прием (осмотр, консультация) врача-уролога первичный", "cost": 2000}]

    def fake_doctor_prices():
        return []

    async def fake_prepare(_query, _entities):
        prepare_called["v"] = True
        return {"prepare": "Не должно вызываться"}

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)
    monkeypatch.setattr(svc, "test_prepare", fake_prepare)

    res = run(
        svc.service_bundle_info(
            "Самара",
            {"service_name": "Прием (осмотр, консультация) врача-уролога первичный"},
        )
    )

    assert prepare_called["v"] is False
    assert str(res.get("prepare") or "").strip() == ""


def test_service_bundle_info_does_not_request_prepare_for_price_only_query(monkeypatch):
    svc = Services()
    prepare_called = {"v": False}

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return []

    def fake_retail(_region_id):
        return [{"serviceName": "УЗИ брюшной полости", "cost": 1800}]

    def fake_doctor_prices():
        return []

    async def fake_prepare(_query, _entities):
        prepare_called["v"] = True
        return {"prepare": "Не должно вызываться"}

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)
    monkeypatch.setattr(svc, "test_prepare", fake_prepare)

    res = run(svc.service_bundle_info("Сколько стоит УЗИ брюшной полости?", {"service_name": "УЗИ брюшной полости"}))

    assert prepare_called["v"] is False
    assert res["show_prepare"] is False
    assert str(res.get("prepare") or "").strip() == ""


def test_service_bundle_info_marks_lab_and_skips_doctors(monkeypatch):
    svc = Services()
    calls = {"doctor_prices": 0, "doctors_cache": 0}

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        calls["doctors_cache"] += 1
        return [
            {
                "id": 7,
                "fio": "Иванов Иван Иванович",
                "ord": 1,
                "specialization": "Терапевт",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    def fake_retail(_region_id):
        return [{"serviceName": "Общий анализ крови", "serviceHomecode": "501", "cost": 490}]

    def fake_doctor_prices():
        calls["doctor_prices"] += 1
        return [{"doctorId": 7, "serviceName": "Прием терапевта", "cost": 2000}]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(
        svc.service_bundle_info(
            "Сколько стоит общий анализ крови?",
            {"service_name": "Общий анализ крови"},
        )
    )

    assert str(res.get("service_kind") or "") == "lab"
    assert res.get("doctors") == []
    assert calls["doctor_prices"] == 0
    assert calls["doctors_cache"] == 0


def test_service_bundle_info_general_oak_returns_base_variants_without_special_modifiers(monkeypatch):
    svc = Services()
    calls = {"doctor_prices": 0}

    def fake_retail(_region_id):
        return [
            {"serviceName": "Cito Общий анализ крови (Le, Er,Hb)", "serviceHomecode": "802", "cost": 580},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
                "serviceHomecode": "501",
                "cost": 520,
            },
            {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
            {
                "serviceName": "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты) капиллярная кровь",
                "serviceHomecode": "501к",
                "cost": 470,
            },
            {
                "serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ) капиллярная кровь",
                "serviceHomecode": "502к",
                "cost": 380,
            },
        ]

    def fake_doctor_prices():
        calls["doctor_prices"] += 1
        return []

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.service_bundle_info("Сколько стоит общий анализ крови?", {}))

    assert str(res.get("service_kind") or "") == "family_query"
    assert res.get("visible_limit") == 2
    variants = res.get("family_variants") or []
    visible_codes = [str(row.get("serviceHomecode") or "") for row in variants[:2]]
    assert visible_codes == ["502", "501"]
    assert str(res.get("service_name") or "").lower() == "общий анализ крови"
    assert res.get("remaining_count", 0) >= 1
    assert calls["doctor_prices"] == 0


def test_service_bundle_info_filters_weak_partial_doctor_matches_for_surgery(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 10,
                "fio": "Казакова Ирина Михайловна",
                "ord": 1,
                "specialization": "УЗИ",
                "regions": ["г. Самара, пр. Ленина, 5"],
            },
            {
                "id": 11,
                "fio": "Свиридова Елена Александровна",
                "ord": 1,
                "specialization": "УЗИ",
                "regions": ["г. Самара, пр. Ленина, 5"],
            },
        ]

    def fake_retail(_region_id):
        return [{"serviceName": "Шунтирование желудка", "serviceHomecode": "83.9.2.22", "cost": 199000}]

    def fake_doctor_prices():
        return [
            {"doctorId": 10, "serviceName": "УЗИ желудка", "serviceHomecode": "12.1.1", "cost": 1100},
            {"doctorId": 11, "serviceName": "УЗИ органов брюшной полости", "serviceHomecode": "12.1.2", "cost": 1100},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.service_bundle_info("Сколько стоит шунтирование желудка?", {"service_name": "Шунтирование желудка"}))

    assert res["retail_prices"], "Expected surgery retail row"
    assert res["doctors"] == [], "Weak one-token overlaps must not leak unrelated doctors"


def test_service_bundle_info_accepts_doctor_by_exact_homecode_match(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 20,
                "fio": "Хирург Иванов И.И.",
                "ord": 1,
                "specialization": "Хирург",
                "regions": ["г. Самара, пр. Ленина, 5"],
            }
        ]

    def fake_retail(_region_id):
        return [{"serviceName": "Шунтирование желудка", "serviceHomecode": "83.9.2.22", "cost": 199000}]

    def fake_doctor_prices():
        # Имя услуги у врача может отличаться, но код совпадает с retail.
        return [
            {"doctorId": 20, "serviceName": "Бариатрическая операция", "serviceHomecode": "83.9.2.22", "cost": 170000},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.service_bundle_info("Сколько стоит шунтирование желудка?", {"service_name": "Шунтирование желудка"}))

    assert len(res["doctors"]) == 1
    assert res["doctors"][0]["fio"] == "Хирург Иванов И.И."


def test_service_bundle_info_consult_filters_doctors_by_primary_specialty(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 101,
                "fio": "Кардиолог Основной",
                "ord": 1,
                "specialization": "Кардиолог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "unit_links": [{"company_unit_name": "Врач-кардиолог", "main": True, "specialization": "Кардиолог"}],
                "main_units": ["Врач-кардиолог"],
            },
            {
                "id": 102,
                "fio": "Ревматолог Смежный",
                "ord": 1,
                "specialization": "Ревматолог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "unit_links": [{"company_unit_name": "Врач-ревматолог", "main": True, "specialization": "Ревматолог"}],
                "main_units": ["Врач-ревматолог"],
            },
            {
                "id": 103,
                "fio": "Гибрид Кардио-Ревма",
                "ord": 1,
                "specialization": "Кардиолог-ревматолог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "unit_links": [{"company_unit_name": "Врач-кардиолог-ревматолог", "main": True, "specialization": "Кардиолог-ревматолог"}],
                "main_units": ["Врач-кардиолог-ревматолог"],
            },
        ]

    def fake_retail(_region_id):
        return [{"serviceName": "Прием (осмотр, консультация) врача-кардиолога первичный", "serviceHomecode": "30.1", "cost": 3000}]

    def fake_doctor_prices():
        # Источник может отдать "грязные" строки по serviceName, поэтому фильтруем по primary specialty.
        return [
            {"doctorId": 101, "serviceName": "Прием (осмотр, консультация) врача-кардиолога первичный", "serviceHomecode": "30.1", "cost": 3000},
            {"doctorId": 102, "serviceName": "Прием (осмотр, консультация) врача-кардиолога первичный", "serviceHomecode": "30.1", "cost": 2700},
            {"doctorId": 103, "serviceName": "Прием (осмотр, консультация) врача-кардиолога первичный", "serviceHomecode": "30.1", "cost": 2800},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.service_bundle_info("Сколько стоит прием кардиолога?", {"service_name": "Прием (осмотр, консультация) врача-кардиолога первичный"}))
    names = [str(row.get("fio") or "") for row in (res.get("doctors") or [])]

    assert names == ["Кардиолог Основной"]


def test_service_bundle_info_consult_doctor_uses_query_specialty_label(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, ул. Ново-Садовая, 106, кор. 82", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 872,
                "fio": "Дурасов Владимир Владимирович",
                "ord": 1,
                "specialization": "Акушер-гинеколог",
                "regions": ["г. Самара, ул. Ново-Садовая, 106, кор. 82"],
                "unit_links": [
                    {"company_unit_name": "Врач акушер-гинеколог", "main": False, "specialization": "Акушер-гинеколог"},
                    {"company_unit_name": "Врач-хирург", "main": True, "specialization": "Хирург"},
                ],
                "units": ["Врач акушер-гинеколог", "Врач-хирург"],
                "main_units": ["Врач-хирург"],
            }
        ]

    async def fake_availability(_fio: str, samara_tokens=None):
        _ = samara_tokens
        return {"available": False, "nearest_slot": "", "regions_with_slots": [], "note": "availability_unmatched"}

    def fake_retail(_region_id):
        return [{"serviceName": "Прием (осмотр, консультация) врача-хирурга первичный", "serviceHomecode": "15.2.1", "cost": 2500}]

    def fake_doctor_prices():
        return [{"doctorId": 872, "serviceName": "Прием (осмотр, консультация) врача-хирурга первичный", "serviceHomecode": "15.2.1", "cost": 2500}]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc, "_doctor_availability_snapshot", fake_availability)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.service_bundle_info("стоимость первичного приема хирурга", {}))

    assert res["doctors"]
    assert res["doctors"][0]["specialty_label"] == "Хирург"


def test_service_bundle_info_ajovy_is_not_lab(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

    async def fake_ensure_doctors_cache():
        return [
            {
                "id": 1288,
                "fio": "Третьякова Наталья Александровна",
                "ord": 1,
                "specialization": "Невролог",
                "regions": ["г. Самара, пр. Ленина, 5"],
                "unit_links": [{"company_unit_name": "Врач-невролог", "main": True, "specialization": "Невролог"}],
                "main_units": ["Врач-невролог"],
            }
        ]

    async def fake_availability(_fio: str, samara_tokens=None):
        _ = samara_tokens
        return {"available": False, "nearest_slot": "", "regions_with_slots": [], "note": "availability_unmatched"}

    def fake_retail(_region_id):
        return [{"serviceName": "Лечение и профилактика мигрени (препарат Аджови)", "serviceHomecode": "12.1.10.3", "cost": 25000}]

    def fake_doctor_prices():
        return [{"doctorId": 1288, "serviceName": "Лечение и профилактика мигрени (препарат Аджови)", "serviceHomecode": "12.1.10.3", "cost": 25000}]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_ensure_regions)
    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_doctors_cache)
    monkeypatch.setattr(svc, "_doctor_availability_snapshot", fake_availability)
    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)
    monkeypatch.setattr(svc_mod.api_price, "load_doctor_prices", fake_doctor_prices)

    res = run(svc.service_bundle_info("стоимость Аджови", {}))

    assert str(res.get("service_kind") or "") != "lab"


def test_service_bundle_info_thyroid_uzi_family_filters_out_lab_profiles(monkeypatch):
    svc = Services()

    def fake_retail(_region_id):
        return [
            {"serviceName": "Щитовидная железа (сокращенное обследование)", "serviceHomecode": "976", "deadline": "1-2", "cost": 860},
            {"serviceName": "Профиль. \"Здоровая щитовидная железа\"", "serviceHomecode": "1287", "deadline": "1-2", "cost": 1615},
            {"serviceName": "Ультразвуковое исследование щитовидной железы и паращитовидных желез", "serviceHomecode": "2.4.1", "deadline": " ", "cost": 2100},
            {"serviceName": "Узи щитовидной железы (экспертное)", "serviceHomecode": "2.4.1.3", "deadline": " ", "cost": 2300},
            {"serviceName": "Тонкоигольная аспирационная биопсия щитовидной железы", "serviceHomecode": "3.3.9.5", "deadline": " ", "cost": 4300},
            {"serviceName": "Резекция перешейка щитовидной железы с использованием нейромонитора", "serviceHomecode": "3.3.9.14", "deadline": " ", "cost": 155700},
        ]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_retail)

    res = run(svc.service_bundle_info("стоимость узи щитовидной железы", {}))

    assert str(res.get("service_kind") or "") == "family_query"
    names = [str(row.get("serviceName") or "") for row in (res.get("family_variants") or [])]
    assert "Ультразвуковое исследование щитовидной железы и паращитовидных желез" in names
    assert "Узи щитовидной железы (экспертное)" in names
    assert "Тонкоигольная аспирационная биопсия щитовидной железы" in names
    assert "Резекция перешейка щитовидной железы с использованием нейромонитора" in names
    assert "Щитовидная железа (сокращенное обследование)" not in names
    assert 'Профиль. "Здоровая щитовидная железа"' not in names


def test_samara_region_tokens_recognise_lenina5_by_value_slug(monkeypatch):
    """Регрессия: филиал «Ленина 5» (region 8882) попадает в samara-allowlist.

    В CRM у этого филиала пустой `addressForSite` и `name='Ленина 5'` без слова
    «Самара» — самарский признак есть только в slug `value='region_samara_lenina'`.
    Без распознавания по `value` врачи этого филиала (напр. Дразнин) выпадали из
    расписания → «расписание не найдено».
    """
    svc = Services()

    async def fake_regions():
        return [
            {"id": 8882, "name": "Ленина 5", "addressForSite": "", "value": "region_samara_lenina"},
            {"id": 3, "name": "Московское шоссе", "addressForSite": "г. Самара, Московское шоссе, 100", "value": "region_samara_msk"},
            {"id": 99, "name": "Оренбург центр", "addressForSite": "г. Оренбург, ул. Советская, 1", "value": "region_orenburg_1"},
        ]

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)

    tokens = run(svc._samara_region_tokens())

    # «Ленина 5» теперь распознаётся как самарский филиал по value-слагу
    assert "ленина 5" in tokens
    from messengers_router.services._regions import _region_matches_samara_tokens
    assert _region_matches_samara_tokens("Ленина 5", tokens) is True
    # Оренбург не должен попасть в самарский allowlist
    assert not any("оренбург" in t for t in tokens)
