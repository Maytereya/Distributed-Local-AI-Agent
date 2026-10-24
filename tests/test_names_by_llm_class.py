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

import pytest

from messengers_router import classifier, turn_names, turn_understanding


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
