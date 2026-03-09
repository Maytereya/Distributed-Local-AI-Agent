import asyncio

from messengers_router import services as svc_mod
from messengers_router.city import match_city
from messengers_router.services import Services
from messengers_router.policies import (
    build_branch_index,
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
                "units": ["Терапия"],
            },
            {
                "id": 2,
                "fio": "Врач Первый",
                "ord": 5,
                "specialization": "терапевт",
                "regions": ["Ленина 5"],
                "units": ["Терапия"],
            },
        ]

    monkeypatch.setattr(svc, "_ensure_doctors_cache_loaded", fake_ensure_cache)

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


def test_appointment_help_meili(monkeypatch):
    svc = Services()

    def fake_search(_index, _query):
        return "<b>info</b>"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: "info")

    res = run(svc.appointment_help("запись", {}))

    assert res["instructions"] == "info"


def test_test_assist_price_by_region(monkeypatch):
    svc = Services()

    def fake_price_by_region(_region_id):
        return [{"serviceName": "Анализ крови общий", "cost": 500}]

    monkeypatch.setattr(svc_mod.api_price, "load_price_by_region", fake_price_by_region)

    res = run(svc.test_assist("анализ крови", {}))

    assert res["tests"], "Expected matches in priceByRegion"


def test_test_prepare_meili(monkeypatch):
    svc = Services()

    def fake_search(_index, _query):
        return "<i>подготовка</i>"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: "подготовка")

    res = run(svc.test_prepare("анализ крови", {}))

    assert res["prepare"] == "подготовка"


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
