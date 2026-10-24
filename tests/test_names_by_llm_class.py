"""Имена без списков стоп-слов: сущности LLM хода и справочник МИС (этап 3, STEP-2026-10-24-NAMES-NO-STOPWORDS).

Инварианты класса:
- имя пациента (ФИО, фамилия) — только сущность LLM-классификатора этого хода, и только подстрока реплики (регистр, «ё», знаки
  не важны); LLM не звали, сбой, модель имени не назвала — имени нет (решение владельца 22.10 (а): «уточните»);
- имена прошлого хода не читаются;
- фамилия врача — точное совпадение слова реплики с фамилией справочника МИС в любой падежной форме; нечётко — только имя,
  которое назвала LLM, и исход — вопрос, а не ответ;
- путь `on` каждого выключателя `names_*` не читает свой список.
ПДн — только выдуманные (репо публичный); фамилии врачей выбирает код из выдержки справочника в репо.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from pathlib import Path

import pytest

from messengers_router import classifier, turn_names, turn_understanding
from messengers_router.services import Services


@pytest.fixture(autouse=True)
def _fresh_turn():
    """Память хода — своя на тест и не переживает его (соседи по воркеру `-n` ждут «вне хода»)."""

    token = turn_names._CURRENT.set(turn_names._Turn())
    yield
    turn_names._CURRENT.reset(token)


def _switch(monkeypatch, name: str, value: str = "on") -> None:
    monkeypatch.setattr(turn_names._cfg, "MR_" + name.upper(), value, raising=False)


# --- память имён хода ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "said"),
    [
        ("Сидорова Анна", "Сидорова Анна"),
        ("сидорова анна", "Сидорова Анна"),
        ("СИДОРОВА АННА!", "сидорова анна"),
        ("Сидорова Алёна", "Сидорова Алена"),
    ],
)
def test_llm_patient_name_is_kept_when_substring_of_reply(reply: str, said: str) -> None:
    turn_names.note_llm(reply, {"patient_name": said})
    assert turn_names.llm_patient_fio(reply, switch="names_patient_fio")


@pytest.mark.parametrize("said", ["Сидорова Анна Петровна", "Петрова Анна", "Анна Сидорова"])
def test_llm_patient_name_not_in_reply_is_dropped(said: str) -> None:
    turn_names.note_llm("Сидорова Анна", {"patient_name": said})
    assert not turn_names.llm_patient_fio("Сидорова Анна", switch="names_patient_fio")


def test_no_llm_this_turn_means_no_name() -> None:
    assert not turn_names.llm_patient_fio("Сидорова Анна", switch="names_patient_fio")
    assert not turn_names.llm_person_word("Сидорова", switch="names_result_surname")


def test_new_turn_forgets_previous_llm_names() -> None:
    def two_turns() -> bool:
        turn_names.note_llm("Сидорова Анна", {"patient_name": "Сидорова Анна"})
        turn_understanding.begin_turn()
        return turn_names.llm_patient_fio("Сидорова Анна", switch="names_patient_fio")

    assert not contextvars.copy_context().run(two_turns)  # ход — в своём контексте, как у роутера


def test_llm_names_noted_in_child_task_are_seen_by_parent() -> None:
    async def turn() -> bool:
        turn_names.begin()

        async def classify() -> None:
            turn_names.note_llm("Сидорова", {"surname": "Сидорова"})

        await asyncio.gather(classify())
        return turn_names.llm_person_word("Сидорова", switch="names_result_surname")

    assert asyncio.run(turn())


@pytest.mark.parametrize("word", ["Сидорова", "сидорова", "СИДОРОВА"])
def test_llm_surname_or_patient_name_word(word: str) -> None:
    turn_names.note_llm("Сидорова", {"surname": "Сидорова"})
    assert turn_names.llm_person_word(word, switch="names_result_surname")
    turn_names.note_llm("Сидорова Анна", {"patient_name": "Сидорова Анна"})
    assert turn_names.llm_person_word(word, switch="names_result_surname")


def test_classifier_llm_answer_is_the_producer(monkeypatch) -> None:
    """Классификатор `legacy_v2` кладёт сущности имён ответа LLM в память хода (правило не сработало)."""

    async def llm(_prompt: str, *, queue_timeout_ms: int = 30000) -> dict:
        return {"label": "TEST_RESULT", "confidence": 0.9, "entities": {"surname": "Сидорова"}}

    monkeypatch.setattr(classifier, "ollama_classify_json", llm)

    async def turn() -> bool:
        turn_understanding.begin_turn()
        await classifier.analyze("Сидорова", {}, skip_rule=True)
        return turn_names.llm_person_word("Сидорова", switch="names_result_surname")

    assert asyncio.run(turn())


def test_switch_of_other_module_is_programmer_error() -> None:
    with pytest.raises(KeyError):
        turn_names.on("pending_reply")


SWITCHES = {
    "names_patient_fio": ("flow_policy", "_PATIENT_FIO_STOPWORDS"),
    "names_patient_rule": ("classifier", "_PATIENT_NAME_STOPWORDS"),
    "names_patient_fill": ("policies", "_QF_PATIENT_NAME_STOPWORDS"),
    "names_result_surname": ("policies", "_QF_RESULT_SURNAME_STOPWORDS"),
    "names_doctor_followup": ("classifier", "_DOCTOR_FOLLOWUP_FILLERS"),
    "names_doctor_noise": ("entity_grounder", "_DOCTOR_NOISE_TOKENS"),
    "names_doctor_catalog": ("services/_doctors_helpers", "_CATALOG_DOCTOR_STOPWORDS"),
}


def test_switches_are_off_by_default() -> None:
    assert not any(turn_names.on(name) for name in SWITCHES)


# --- names_doctor_catalog: врач реплики — по справочнику МИС --------------------------------------------------------

_EXCERPT = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"


def _excerpt_doctors() -> list[dict]:
    with _EXCERPT.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _surnames() -> list[str]:
    found = [str(d.get("fio") or "").split()[0] for d in _excerpt_doctors() if str(d.get("fio") or "").strip()]
    return sorted(s for s in set(found) if found.count(s) == 1 and s.isalpha())


def _word_forms(surname: str) -> list[str]:
    """Падежные формы фамилии на -ов/-ев/-ин и -ова/-ева/-ина (как пишут пациенты: «к Иванову», «у Ивановой»)."""

    low = surname.lower()
    if low.endswith(("ова", "ева", "ина")):
        stem = surname[:-1]
        return [surname, stem + "ой", stem + "у"]
    if low.endswith(("ов", "ев", "ин")):
        return [surname, surname + "а", surname + "у", surname + "ым", surname + "е"]
    return [surname]


def _declined_surnames() -> list[str]:
    male = next(s for s in _surnames() if s.lower().endswith(("ов", "ев", "ин")))
    female = next(s for s in _surnames() if s.lower().endswith(("ова", "ева", "ина")))
    return [male, female]


def _catalog_services() -> Services:
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    async def loaded():
        return _excerpt_doctors()

    services._ensure_doctors_cache_loaded = loaded
    return services


def _match(text: str) -> dict:
    return asyncio.run(_catalog_services().match_catalog_doctor(text))


@pytest.mark.parametrize("surname", _declined_surnames())
@pytest.mark.parametrize("case", [str, str.lower, str.upper])
def test_catalog_doctor_by_data_any_word_form_and_case(monkeypatch, surname: str, case) -> None:
    _switch(monkeypatch, "names_doctor_catalog")
    for form in _word_forms(surname):
        match = _match(f"Хочу записаться на приём к врачу {case(form)}")
        assert (match["status"], match["canonical"]) == ("exact", surname), form


@pytest.mark.parametrize(
    "text",
    [
        "По каким адресам можно сделать флюорографию",
        "УЗИ органов малого таза",
        "Какова стоимость услуг",
        "Хочу записаться к врачу",
        "да, хорошо",
    ],
)
def test_catalog_doctor_by_data_no_surname_no_guess(monkeypatch, text: str) -> None:
    """Слово без фамилии справочника — не врач; прежний путь находил «Ким» в «каким» нечётким сравнением (свип 24.10)."""

    _switch(monkeypatch, "names_doctor_catalog")
    assert _match(text)["status"] == "miss"


def _typo(surname: str) -> str:
    middle = len(surname) // 2
    return surname[:middle] + ("о" if surname[middle] != "о" else "а") + surname[middle + 1 :]


def test_catalog_doctor_typo_only_in_name_llm_named(monkeypatch) -> None:
    _switch(monkeypatch, "names_doctor_catalog")
    surname = max(_surnames(), key=len)
    typo = _typo(surname)
    reply = f"Запишите к {typo}"
    assert _match(reply)["status"] == "miss"  # LLM имени не называла — по буквам не угадываем
    turn_names.note_llm(reply, {"doctor_name": typo})
    match = _match(reply)
    assert (match["status"], match["canonical"]) == ("fuzzy", surname)  # исход — вопрос «Похоже, вы имели в виду…?»


# --- names_doctor_noise: служебное слово как doctor_name — решает справочник ------------------------------------------


def _verify(services: Services, doctor_name: str, text: str):
    from messengers_router import entity_grounder
    from messengers_router.mess_types import RouteDecision

    decision = RouteDecision(label="DOCTOR_SCHEDULE", confidence=0.8, entities={"doctor_name": doctor_name}, source="llm_primary")
    return asyncio.run(entity_grounder.verify_doctor_entities_in_decision(decision, text, services))


@pytest.mark.parametrize("word", ["хочу", "Запись", "ЗАПИСАТЬСЯ", "когда", "да", "нет", "подскажите", "приёма", "записи", "хотела"])
def test_service_word_as_doctor_name_is_not_a_doctor_by_data(monkeypatch, word: str) -> None:
    """Любая форма служебного слова (и те, которых в списке не было) — не врач: справочник его не находит."""

    _switch(monkeypatch, "names_doctor_noise")
    out = _verify(_catalog_services(), word, f"{word} к врачу")
    assert "doctor_name" not in out.entities and "doctor_name_unverified" in out.flags


@pytest.mark.parametrize("surname", _declined_surnames())
def test_doctor_surname_entity_any_form_verified_by_data(monkeypatch, surname: str) -> None:
    _switch(monkeypatch, "names_doctor_noise")
    for form in _word_forms(surname):
        out = _verify(_catalog_services(), form.lower(), f"когда принимает {form.lower()}")
        assert out.entities.get("doctor_name") == surname and "doctor_name_verified" in out.flags, form
