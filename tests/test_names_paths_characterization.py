"""Пути имён до замены списков «не имя» — как решают сейчас (этап 3, STEP-2026-10-24-NAMES-NO-STOPWORDS).

Характеристика при выключателях `names_*` = off: результаты анализов («фамилия, год, код, номер»), шаг ФИО записи, врач по
фамилии. Тесты не говорят «так правильно» — они держат поведение, от которого меряется замена; известные дефекты отмечены.
ПДн — только выдуманные (репо публичный); фамилии врачей берутся кодом из выдержки справочника в репо.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from messengers_router import classifier, entity_grounder, flow_policy, planner, policies
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.services import Services
from messengers_router.services import _doctors_helpers
from tests.names_modes import set_names

_EXCERPT = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"
_RESULT_FIELDS = ["surname", "year", "filial", "number"]


@pytest.fixture(autouse=True)
def _names_off(monkeypatch):
    """Характеристика пути off — явно, при любом конфиге прогона."""

    set_names(monkeypatch, "off")


def _excerpt_doctors() -> list[dict]:
    with _EXCERPT.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _doctor_surname() -> str:
    """Фамилия врача выдержки, одна такая в справочнике (выбор кодом, не по имени)."""

    surnames = [str(d.get("fio") or "").split()[0] for d in _excerpt_doctors() if str(d.get("fio") or "").strip()]
    return next(s for s in sorted(surnames) if surnames.count(s) == 1 and len(s) >= 6 and s.isalpha())


def _services(doctors: list[dict] | None = None) -> Services:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    if doctors is not None:
        async def loaded():
            return doctors

        services._ensure_doctors_cache_loaded = loaded
    return services


# --- результаты анализов: «фамилия, год, код, номер» -------------------------------------------------------------


def _result_fields(text: str, missing: list[str]) -> dict:
    out: dict = {}
    policies._fill_test_result_entities(text, missing, out)
    return out


@pytest.mark.parametrize(
    "text",
    ["Сидорова, 1985, Бг, 123", "Сидорова 1985 Бг 123", "сидорова, 1985, бг, 123"],
)
def test_result_ordered_four_fields_fill_all(text: str) -> None:
    out = _result_fields(text, _RESULT_FIELDS)
    assert (out["surname"], out["year"], out["number"]) == ("Сидорова", 1985, 123)
    assert out["filial"].lower() == "бг"


@pytest.mark.parametrize("text", ["Сидорова", "сидорова"])
def test_result_single_word_is_surname_when_missing(text: str) -> None:
    assert _result_fields(text, _RESULT_FIELDS)["surname"] == "Сидорова"


@pytest.mark.parametrize("text", ["Результаты", "Анализы", "Хочу", "Подскажите"])
def test_result_listed_word_is_not_surname(text: str) -> None:
    """Список 4 (`_QF_RESULT_SURNAME_STOPWORDS`) держит служебное слово вне фамилии."""

    assert "surname" not in _result_fields(text, _RESULT_FIELDS)


def test_result_surname_after_marker_word() -> None:
    assert _result_fields("Фамилия Сидорова", _RESULT_FIELDS)["surname"] == "Сидорова"


def test_result_single_word_also_lands_in_filial_known_defect() -> None:
    """Известный дефект (не список имён): одно слово при недостающем филиале — ещё и филиал (класс «слот берёт команду
    как данные», FEAT-2026-07-15-SERVICE-SLOT-LLM, отложенные сиблинги)."""

    assert _result_fields("Сидорова", _RESULT_FIELDS)["filial"] == "Сидорова"
    assert _result_fields("Результаты", _RESULT_FIELDS)["filial"] == "Результаты"


def _say(texts: list[str]) -> list[str]:
    state = SessionState(session_id="names-characterization-result")
    memory = MemoryStore()
    services = _services()

    async def go() -> list[str]:
        answers = []
        for text in texts:
            out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
            answers.append("".join(env.text for env in out if env.text))
        return answers

    return asyncio.run(go())


def test_result_flow_asks_four_fields_then_gives_link() -> None:
    ask, link = _say(["Хочу получить результаты анализов", "Сидорова, 1985, Бг, 123"])
    assert "фамилия, год рождения, код анализа, номер анализа" in ask
    assert "getanaliz.php?" in link and "year=1985" in link and "nom2=123" in link


# --- ФИО пациента: форма + списки 1–3 ------------------------------------------------------------------------------

_FIO_SHAPE = [
    ("Сидорова Анна", True),
    ("сидорова анна петровна", True),
    ("Общий анализ крови", False),
    ("Расписание гинеколог", False),
    ("Уролог цена приема", False),
    ("завтра утром", False),
    ("хочу к врачу", False),
    ("Меня зовут Сидорова Анна", False),
]


@pytest.mark.parametrize(("text", "expected"), _FIO_SHAPE)
def test_patient_fio_shape(text: str, expected: bool) -> None:
    """Списки 1 (`flow_policy`) и 2 (`classifier`) решают одинаково на этих входах."""

    assert flow_policy._looks_like_patient_fio(text) is expected
    assert classifier._looks_like_patient_name_only(text) is expected


def _filled_patient_name(text: str) -> str | None:
    out: dict = {}
    policies._fill_patient_entities(text, ["patient_name"], out)
    return out.get("patient_name")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Сидорова Анна", "Сидорова Анна"),
        ("Меня зовут Сидорова Анна", "Сидорова Анна"),
        ("ФИО Сидорова Анна Петровна", "Сидорова Анна Петровна"),
        ("завтра утром", None),
        ("Уролог цена приема", None),
    ],
)
def test_patient_name_quick_fill(text: str, expected: str | None) -> None:
    assert _filled_patient_name(text) == expected


@pytest.mark.parametrize("text", ["Общий анализ крови", "Анализ крови"])
def test_patient_name_quick_fill_takes_service_known_defect(text: str) -> None:
    """Известный дефект списка 3: названия услуги в нём нет — быстрое заполнение кладёт услугу в ФИО. Путь почти мёртвый:
    после варианта A (08.10) ФИО ждёт только отмена, а отмена — сразу оператор."""

    assert _filled_patient_name(text) == text.title()


def test_patient_fio_names_unbookable_target() -> None:
    assert planner._unbookable_target_kind("Сидорова Анна Петровна", set()) == "patient_name"
    assert planner._unbookable_target_kind("Общий анализ крови", set()) != "patient_name"


def test_unverified_doctor_with_fio_shape_becomes_patient_name() -> None:
    async def not_a_doctor(_raw: str):
        return None

    services = _services()
    services.resolve_doctor_name = not_a_doctor
    decision = RouteDecision(label="APPOINTMENT", confidence=0.8, entities={"doctor_name": "Сидорова Анна"}, source="llm_primary")
    out = asyncio.run(entity_grounder.verify_doctor_entities_in_decision(decision, "Сидорова Анна", services))
    assert out.entities.get("patient_name") == "Сидорова Анна" and "doctor_name" not in out.entities
    assert {"doctor_name_unverified", "patient_name_from_unverified_doctor"} <= out.flags


# --- врач по фамилии: списки 5–7 + справочник ------------------------------------------------------------------------


def test_catalog_doctor_query_skips_listed_words() -> None:
    surname = _doctor_surname()
    queries = _doctors_helpers._doctor_catalog_query_candidates(f"Запишите, пожалуйста, к {surname}")
    assert queries[0] == surname.lower().replace("ё", "е")


def test_catalog_doctor_query_list_misses_word_forms_known_defect() -> None:
    """Список 7 знает «врач», но не «врачу»: первым словом сверки становится форма служебного слова."""

    queries = _doctors_helpers._doctor_catalog_query_candidates(f"Запишите к врачу {_doctor_surname()}")
    assert queries[0] == "врачу"


def test_catalog_doctor_match_finds_surname_after_listed_words() -> None:
    surname = _doctor_surname()
    services = _services(_excerpt_doctors())
    match = asyncio.run(services.match_catalog_doctor(f"Хочу записаться на прием к {surname}"))
    assert (match["status"], match["canonical"]) == ("exact", surname)


@pytest.mark.parametrize("word", ["хочу", "Запись", "да", "подскажите"])
def test_noise_word_as_doctor_name_is_dropped(word: str) -> None:
    flags: set[str] = set()
    entities = {"doctor_name": word}
    assert entity_grounder._sanitize_raw_doctor_name(word, flags, entities) == ""
    assert "doctor_name" not in entities and "doctor_name_unverified" in flags


def test_doctor_surname_entity_kept_for_verification() -> None:
    surname = _doctor_surname()
    flags: set[str] = set()
    assert entity_grounder._sanitize_raw_doctor_name(surname, flags, {"doctor_name": surname}) == surname
    assert not flags


_DOCTOR_CONTEXT = {"_last_label": "DOCTOR_SCHEDULE", "specialty": "кардиолог"}


def test_doctor_followup_surname_with_filler() -> None:
    surname = _doctor_surname()
    assert classifier._doctor_followup_name_in_context(f"{surname}, да", dict(_DOCTOR_CONTEXT)) == surname


@pytest.mark.parametrize("text", ["да, хорошо", "ладно", "подходит"])
def test_doctor_followup_fillers_only_is_not_a_name(text: str) -> None:
    assert classifier._doctor_followup_name_in_context(text, dict(_DOCTOR_CONTEXT)) is None
