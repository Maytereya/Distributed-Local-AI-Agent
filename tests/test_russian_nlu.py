# tests/test_russian_nlu.py
import pytest
from messengers_router.russian_nlu import normalize_ru, ENTITY_WHITELIST


def test_normalize_ru_lowercases():
    assert normalize_ru("ПРИВЕТ") == "привет"


def test_normalize_ru_replaces_yo():
    assert normalize_ru("Ёлка") == "елка"


def test_normalize_ru_strips():
    assert normalize_ru("  привет  ") == "привет"


def test_normalize_ru_combined():
    assert normalize_ru("  ПЕЧЁНЬ  ") == "печень"


def test_normalize_ru_empty():
    assert normalize_ru("") == ""
    assert normalize_ru(None) == ""


def test_entity_whitelist_is_frozenset():
    assert isinstance(ENTITY_WHITELIST, frozenset)


def test_entity_whitelist_has_required_keys():
    required = {"doctor_name", "specialty", "service_name", "city", "date_from", "patient_name"}
    assert required <= ENTITY_WHITELIST
