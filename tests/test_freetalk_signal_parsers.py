from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.signal_parsers import (
    extract_confirmation_head,
    extract_contextual_entities,
    extract_branch_reference,
    extract_date_filters,
    extract_doctor_reference_candidate,
    extract_expected_slot_entities,
    extract_person_name,
    extract_result_lookup_fields,
    extract_service_reference_candidate,
    extract_service_variant,
    extract_time_filters,
    looks_like_specific_doctor_reference,
    looks_like_no_preference_answer,
    looks_like_slot_correction,
    looks_like_uncertainty_answer,
    looks_like_doctor_followup_message,
    looks_like_service_followup_message,
    parse_yes_no,
    split_mixed_utterance,
)


def test_parse_yes_no_supports_strict_and_guard_profiles():
    assert parse_yes_no("да", profile="strict") == "yes"
    assert parse_yes_no("нет", profile="strict") == "no"
    assert parse_yes_no("угу", profile="strict") == "unknown"
    assert parse_yes_no("угу", profile="guard") == "yes"
    assert parse_yes_no("пока нет", profile="guard") == "no"


def test_extract_confirmation_head_supports_yes_and_no_prefixes():
    assert extract_confirmation_head("Да, а как подготовиться?") == ("yes", "а как подготовиться?")
    assert extract_confirmation_head("Нет, Суворов") == ("no", "Суворов")
    assert extract_confirmation_head("Суворов") == ("unknown", "Суворов")


def test_extract_person_name_accepts_fio_and_rejects_digits():
    assert extract_person_name("Иванов Иван Иванович") == "Иванов Иван Иванович"
    assert extract_person_name("Иванов 1989") == ""


def test_extract_branch_reference_cleans_time_suffix():
    assert extract_branch_reference("на Ленина утром") == "Ленина"
    assert extract_branch_reference("в филиале на Стара-Загора") == "Стара-Загора"


def test_extract_date_and_time_filters():
    assert extract_date_filters("на 2026-05-17") == {
        "date": "2026-05-17",
        "date_from": "2026-05-17",
        "date_to": "2026-05-17",
    }
    assert extract_time_filters("вечером") == {
        "time": "вечером",
        "time_from": "17:00",
        "time_to": "21:00",
    }
    assert extract_time_filters("после 15:30")["time_from"] == "15:30"


def test_extract_service_variant():
    assert extract_service_variant("МРТ с контрастом") == "с контрастом"
    assert extract_service_variant("обычная консультация") == ""


def test_extract_contextual_entities_combines_parsers():
    assert extract_contextual_entities("на Ленина вечером") == {
        "branch_name": "Ленина",
        "time": "вечером",
        "time_from": "17:00",
        "time_to": "21:00",
    }


def test_doctor_and_service_followup_detection():
    assert looks_like_doctor_followup_message(
        user_message="Напиши, чем он занимается",
        remembered_doctor="Дразнин Антон Владимирович",
    ) is True
    assert looks_like_service_followup_message(
        user_message="А с наркозом можно?",
        remembered_service="ФГДС",
    ) is True


def test_extract_doctor_reference_candidate_supports_explicit_and_short_forms():
    assert extract_doctor_reference_candidate("к врачу Суворову") == "Суворову"
    assert extract_doctor_reference_candidate("нет, Суворов") == "Суворов"
    assert extract_doctor_reference_candidate("Дразнин") == "Дразнин"
    assert extract_doctor_reference_candidate("дразнин") == "дразнин"


def test_doctor_reference_candidate_rejects_greetings_and_specialties():
    assert extract_doctor_reference_candidate("Привет") == ""
    assert extract_doctor_reference_candidate("кардиолог") == ""
    assert looks_like_specific_doctor_reference("Привет") is False
    assert looks_like_specific_doctor_reference("кардиолог") is False
    assert looks_like_specific_doctor_reference("Дразнин") is True


def test_extract_service_reference_candidate_and_slot_correction():
    assert extract_service_reference_candidate("нужна услуга ФГДС") == "ФГДС"
    assert looks_like_slot_correction("не этого врача") is True
    assert looks_like_slot_correction("другой филиал") is True
    assert looks_like_slot_correction("Сколько стоит анализ?") is False


def test_uncertainty_and_no_preference_detection():
    assert looks_like_uncertainty_answer("не знаю") is True
    assert looks_like_uncertainty_answer("все равно не помню") is True
    assert looks_like_no_preference_answer("без разницы") is True
    assert looks_like_no_preference_answer("любой") is True


def test_extract_result_lookup_fields_supports_partial_tuple():
    assert extract_result_lookup_fields(
        "Иванов, 1989",
        expected_slots=["result_surname", "result_year_of_birth", "result_analysis_code"],
    ) == {
        "result_surname": "Иванов",
        "result_year_of_birth": "1989",
    }
    assert extract_result_lookup_fields(
        "Бг, 12345",
        expected_slots=["result_analysis_code", "result_analysis_number"],
    ) == {
        "result_analysis_code": "Бг",
        "result_analysis_number": "12345",
    }


def test_split_mixed_utterance_only_on_controlled_markers():
    assert split_mixed_utterance("Иванов, 1990, Бг, 12345") == ("", "")
    assert split_mixed_utterance("16 апреля, 09:00") == ("", "")
    assert split_mixed_utterance("Иванов, 1990, а лучше покажи урологов") == (
        "Иванов, 1990",
        "покажи урологов",
    )
    assert split_mixed_utterance("Суворов, а сколько стоит прием?") == (
        "Суворов",
        "сколько стоит прием?",
    )


def test_extract_expected_slot_entities_for_mixed_flow_part():
    assert extract_expected_slot_entities(
        "Иванов, 1990",
        expected_slots=["result_surname", "result_year_of_birth", "result_analysis_code"],
    ) == {
        "result_surname": "Иванов",
        "result_year_of_birth": "1990",
    }
    assert extract_expected_slot_entities(
        "на 17 мая",
        expected_slots=["date", "time"],
    )["date"].endswith("-05-17")
