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


@pytest.mark.parametrize(
    "text",
    [
        "На 03.06.2026",
        "03.06.2026",
        "03.06",
        "запишите на 15.06.2026",
        "2026-06-03",
    ],
)
def test_dotted_date_does_not_produce_spurious_time(text):
    # BUG-2026-06-02-06: дата DD.MM[.YYYY] не должна ловиться time-регэкспом как
    # «HH:MM» (разделитель «.» совпадает с DD.MM). Класс-инвариант: цифровая дата
    # даёт date_*, но НЕ time_*.
    out = parse_date_time_ru(text, today=WED)
    assert out.get("date_from"), text
    assert "time_from" not in out, (text, out)
    assert "time_to" not in out, (text, out)


@pytest.mark.parametrize(
    "text,expected_from,expected_to",
    [
        ("в 15.30", "15:30", "15:30"),
        ("15.30", "15:30", "15:30"),  # как дата 15.30 невалидна (месяц 30) → время
        ("к 9 утра", "09:00", "09:00"),
        ("с 9 до 18", "09:00", "18:00"),
    ],
)
def test_dotted_time_still_parsed(text, expected_from, expected_to):
    # Обратная сторона инварианта: явное/валидное время «HH.MM» / «в HH.MM» не
    # должно ломаться вырезанием дат (его и не вырезаем — это не валидная дата).
    out = parse_date_time_ru(text, today=WED)
    assert out.get("time_from") == expected_from, (text, out)
    assert out.get("time_to") == expected_to, (text, out)


def test_date_and_explicit_time_both_parsed():
    # Дата вырезается только для поиска времени; явное «в 9» рядом сохраняется.
    out = parse_date_time_ru("03.06.2026 в 9", today=WED)
    assert out.get("date_from") == "2026-06-03"
    assert out.get("time_from") == "09:00"
