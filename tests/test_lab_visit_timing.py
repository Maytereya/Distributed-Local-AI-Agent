import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services.prepare import _is_lab_visit_timing_query


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    "text",
    [
        "во сколько прийти чтобы сдать кровь",
        "когда сдавать кровь",
        "во сколько можно сдать мочу",
        "до скольки принимают кал",
    ],
)
def test_lab_visit_timing_positive(text):
    assert _is_lab_visit_timing_query(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "подготовка к ОАК",
        "как подготовиться к анализу крови",  # «как», не «во сколько/когда»
        "во сколько работает клиника",  # нет действия сдачи и биоматериала
        "расписание врача Иванова",
    ],
)
def test_lab_visit_timing_negative(text):
    assert _is_lab_visit_timing_query(text) is False


def test_test_prepare_returns_branches_with_hours_for_blood_visit(monkeypatch):
    svc = Services()

    async def fake_regions():
        return [
            {
                "id": 8882,
                "name": "Ленина 5",
                "city": "Самара",
                "addressForSite": "г. Самара, пр. Ленина, 5",
                "weekdaysFrom": "07:00:00",
                "weekdaysTo": "20:00:00",
                "saturdayFrom": "07:00:00",
                "saturdayTo": "16:00:00",
                "sundayFrom": "07:00:00",
                "sundayTo": "16:00:00",
            },
            {
                "id": 99,
                "name": "Оренбург центр",
                "city": "Оренбург",
                "addressForSite": "г. Оренбург, ул. Чкалова, 51/1",
                "weekdaysFrom": "08:00:00",
                "weekdaysTo": "20:00:00",
            },
        ]

    async def boom(*_a, **_k):
        raise AssertionError("meili must not be called for lab-visit timing query")

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_prepare_candidates_from_analysis_api_cache", boom)

    res = run(svc.test_prepare("во сколько прийти чтобы сдать кровь", {}))
    text = str(res.get("prepare") or "")

    # для крови — памятка о сдаче, затем филиалы с графиком
    assert text.startswith("Кровь сдается без записи")
    assert "натощак" in text and "паспорт" in text
    assert "Адреса филиалов в Самаре" in text
    assert "Ленина, 5" in text
    assert "график: будни 07:00–20:00" in text
    assert "Оренбург" not in text  # не-самарский филиал отсечён
    assert "branches" in str(res.get("note") or "")


def test_urine_visit_query_has_branches_without_blood_memo(monkeypatch):
    svc = Services()

    async def fake_regions():
        return [
            {
                "id": 8882,
                "name": "Ленина 5",
                "city": "Самара",
                "addressForSite": "г. Самара, пр. Ленина, 5",
                "weekdaysFrom": "07:00:00",
                "weekdaysTo": "20:00:00",
            },
        ]

    async def boom(*_a, **_k):
        raise AssertionError("meili must not be called")

    monkeypatch.setattr(svc, "_ensure_regions_loaded", fake_regions)
    monkeypatch.setattr(svc, "_prepare_candidates_from_analysis_api_cache", boom)

    res = run(svc.test_prepare("во сколько можно сдать мочу", {}))
    text = str(res.get("prepare") or "")

    # для мочи памятки про кровь быть не должно, но филиалы+график — да
    assert "Кровь сдается" not in text
    assert text.startswith("Адреса филиалов в Самаре")
