from datetime import date

import pytest

from messengers_router.policies import parse_date_time_ru


# Опорная дата — среда 2026-05-20. Ближайшие выходные: сб 23.05 + вс 24.05.
WED = date(2026, 5, 20)


@pytest.mark.parametrize(
    "text",
    [
        "на этих выходных",
        "можно в выходные",
        "выходные дни",
        "в выходной",
        "Какие есть варианты записи на этих выходных",
    ],
)
def test_parse_weekend_maps_to_saturday_sunday_range(text):
    out = parse_date_time_ru(text, today=WED)
    assert out.get("date_from") == "2026-05-23"  # суббота
    assert out.get("date_to") == "2026-05-24"  # воскресенье


def test_parse_weekend_when_today_is_saturday_uses_current_weekend():
    out = parse_date_time_ru("на выходных", today=date(2026, 5, 23))
    assert out.get("date_from") == "2026-05-23"
    assert out.get("date_to") == "2026-05-24"


def test_parse_weekend_when_today_is_sunday_uses_next_weekend():
    out = parse_date_time_ru("на выходных", today=date(2026, 5, 24))
    assert out.get("date_from") == "2026-05-30"
    assert out.get("date_to") == "2026-05-31"


def test_weekend_does_not_break_existing_weekday_parsing():
    # одиночный день недели — прежнее поведение (одна дата, не диапазон)
    out = parse_date_time_ru("в субботу", today=WED)
    assert out.get("date_from") == out.get("date_to") == "2026-05-23"


def test_weekend_regex_does_not_match_unrelated_words():
    # «выходить» не должно распознаваться как «выходные»
    out = parse_date_time_ru("как выходить из кабинета", today=WED)
    assert "date_from" not in out
    assert "date_to" not in out


def test_explicit_date_takes_precedence_over_weekend_word():
    out = parse_date_time_ru("25.05 в выходные", today=WED)
    assert out.get("date_from") == out.get("date_to") == "2026-05-25"
