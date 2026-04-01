import asyncio

import pytest

from messengers_router import services as svc_mod
from messengers_router.city import match_city
from messengers_router.services import Services
from messengers_router.policies import (
    build_branch_index,
    extract_specialty,
    extract_service_phrase,
    match_branch_hint,
    quick_fill_core_entities,
)


def run(coro):
    return asyncio.run(coro)


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("расписание нейрохирурга", "нейрохирург"),
        ("расписание нейрохирургов", "нейрохирург"),
        ("расписание флеболога", "флеболог"),
        ("расписание флебологов", "флеболог"),
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
        return "<b>Справка для налоговой</b>"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: "Справка для налоговой")

    res = run(svc.main_index_info("Как получить справку для налоговой?", {}))

    assert res["content"] == "Справка для налоговой"
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
    assert "не удалось найти информацию" in str(res.get("handoff_message") or "").lower()


def test_main_index_info_tax_source_unavailable_returns_guidance_without_handoff(monkeypatch):
    svc = Services()

    def fake_search(_index, _query, *args, **kwargs):
        raise RuntimeError("meili unavailable")

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)

    res = run(svc.main_index_info("справка для ФНС", {}))

    assert res.get("handoff_required") is not True
    assert "справк" in str(res.get("content") or "").lower()
    assert "налог" in str(res.get("content") or "").lower()


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


def test_extract_price_service_from_query_strips_politeness_tail():
    assert svc_mod._extract_price_service_from_query("стоимость экг подскажите") == "экг"


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
            "Сколько стоит УЗИ брюшной полости?",
            {"service_name": "УЗИ брюшной полости"},
            top_n=2,
        )
    )

    assert res["retail_prices"], "Expected retail prices block"
    assert len(res["doctors"]) == 2, "Expected top-2 doctors"
    assert [d["fio"] for d in res["doctors"]] == ["Альфа Доктор", "Бета Доктор"]
    assert res["doctors"][0]["available"] is True
    assert res["doctors"][1]["available"] is False
    assert "Натощак" in str(res.get("prepare") or "")


def test_service_bundle_info_keeps_service_name_on_city_only_reply(monkeypatch):
    svc = Services()

    async def fake_ensure_regions():
        return [{"id": 1, "addressForSite": "г. Самара, пр. Ленина, 5", "city": "Самара"}]

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
