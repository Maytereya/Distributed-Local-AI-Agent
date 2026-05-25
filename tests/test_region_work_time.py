from messengers_router.services._regions import _extract_region_work_time


def test_work_time_from_structured_fields_combines_identical_weekend():
    region = {
        "weekdaysFrom": "07:00:00",
        "weekdaysTo": "20:00:00",
        "saturdayFrom": "07:00:00",
        "saturdayTo": "16:00:00",
        "sundayFrom": "07:00:00",
        "sundayTo": "16:00:00",
    }
    assert _extract_region_work_time(region) == "будни 07:00–20:00, выходные 07:00–16:00"


def test_work_time_structured_distinct_saturday_sunday():
    region = {
        "weekdaysFrom": "08:00:00",
        "weekdaysTo": "20:00:00",
        "saturdayFrom": "08:00:00",
        "saturdayTo": "16:00:00",
        "sundayFrom": "08:00:00",
        "sundayTo": "14:00:00",
    }
    assert _extract_region_work_time(region) == "будни 08:00–20:00, сб 08:00–16:00, вс 08:00–14:00"


def test_work_time_weekdays_only():
    assert _extract_region_work_time({"weekdaysFrom": "08:00", "weekdaysTo": "20:00"}) == "будни 08:00–20:00"


def test_work_time_legacy_string_takes_priority():
    # старая строковая форма /regions должна сохраняться без изменений
    assert _extract_region_work_time({"workTime": "Пн-Пт 08:00-20:00"}) == "Пн-Пт 08:00-20:00"


def test_work_time_absent_returns_empty():
    assert _extract_region_work_time({"id": 1, "name": "Ленина 5"}) == ""
