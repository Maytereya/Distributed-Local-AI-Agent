"""Имена в реплике без списков стоп-слов (DESIGN-2026-10-23-NO-STOPWORDS §4, этап 3; STEP-2026-10-24-NAMES-NO-STOPWORDS).

Решения владельца 22.10: списки «не имя» удалить; без LLM имя не угадывать — «уточните» (решение (а)). Форма ФИО (2–3 кириллических
слова) сама принимает «Общий анализ крови» и «Расписание гинеколог» — семь списков держали именно это (опись 24.10). Здесь:

- **ФИО и фамилия пациента** — сущности LLM-классификатора этого хода (`patient_name`, `surname`), и только если значение —
  подстрока реплики (модель не придумала имя). LLM не звали (правило решило ход раньше), сбой, модель имени не назвала — имени нет:
  бот переспросит. Записи и результаты по ФИО бот не ищет (R-01): имя нужно только реквизитом.
- **Фамилия врача** — справочник врачей МИС: слово реплики совпадает с фамилией врача точно, с падежными формами
  (`surname_variants` — механика склонения, не список). Нечёткое сравнение (опечатки) — только для имени, которое назвала LLM, и
  исход его — вопрос «Похоже, вы имели в виду…?», а не ответ.

Выключатель на каждый заменённый список (`[MESSENGER_ROUTER] names_*`, off — как было, путь читает список; on — список не читается):
`names_patient_fio` (`flow_policy._PATIENT_FIO_STOPWORDS`), `names_patient_rule` (`classifier._PATIENT_NAME_STOPWORDS`),
`names_patient_fill` (`policies._QF_PATIENT_NAME_STOPWORDS`), `names_result_surname` (`policies._QF_RESULT_SURNAME_STOPWORDS`),
`names_doctor_followup` (`classifier._DOCTOR_FOLLOWUP_FILLERS`), `names_doctor_noise` (`entity_grounder._DOCTOR_NOISE_TOKENS`),
`names_doctor_catalog` (`_doctors_helpers._CATALOG_DOCTOR_STOPWORDS`).
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from . import turn_telemetry
from .doctor_name_port import surname_variants
from .runtime_config import config as _cfg

_PREFIX = "names_"
_PERSON_KEYS = ("patient_name", "surname")
_MIN_SURNAME_LEN = 3
# Опечатка в фамилии, которую назвала LLM: тот же порог, что у прежнего нечёткого сравнения каталога (`match_catalog_doctor`).
_FUZZY_CUTOFF = 0.84
_FUZZY_MIN_LEN = 4


def on(switch: str) -> bool:
    """Выключатель `names_*` включён (`MR_NAMES_…` = on); имя не этого модуля — ошибка программиста."""

    if not switch.startswith(_PREFIX):
        raise KeyError(switch)
    return str(getattr(_cfg, "MR_" + switch.upper(), "off") or "off").strip().lower() == "on"


def _words(text: str) -> tuple[str, ...]:
    """Слова без регистра, «ё» и знаков; дефис внутри слова остаётся («Петрова-Водкина»)."""

    low = str(text or "").casefold().replace("ё", "е")
    spaced = "".join(ch if ch.isalnum() or ch == "-" else " " for ch in low)
    return tuple(w.strip("-") for w in spaced.split() if w.strip("-"))


def _contains(reply_words: tuple[str, ...], part: tuple[str, ...]) -> bool:
    """`part` — слова реплики подряд (подстрока реплики с точностью до регистра, «ё» и знаков)."""

    n = len(part)
    return n > 0 and any(reply_words[i : i + n] == part for i in range(len(reply_words) - n + 1))


@dataclass(frozen=True)
class _LlmNames:
    person: dict[str, tuple[str, ...]]  # patient_name / surname → слова; только подстроки реплики
    doctor_name: str  # как назвала LLM; проверка — справочником


@dataclass
class _Turn:
    """Память хода — изменяемая: классификатор может работать в дочерней задаче (`gather`), читатели — в родительской."""

    names: _LlmNames | None = None


_CURRENT: contextvars.ContextVar[_Turn | None] = contextvars.ContextVar("mr_turn_names", default=None)


def begin() -> None:
    """Новый ход: имена прошлого хода не читаются (зовёт `turn_understanding.begin_turn`)."""

    _CURRENT.set(_Turn())


def _names() -> _LlmNames | None:
    turn = _CURRENT.get()
    return turn.names if turn else None


def note_llm(text: str, entities: dict[str, Any] | None) -> None:
    """Сущности имён из ответа LLM-классификатора этого хода (единственный производитель — `classifier`)."""

    entities = entities if isinstance(entities, dict) else {}
    reply_words = _words(text)
    person: dict[str, tuple[str, ...]] = {}
    for key in _PERSON_KEYS:
        words = _words(entities.get(key) if isinstance(entities.get(key), str) else "")
        if _contains(reply_words, words):
            person[key] = words
    doctor = entities.get("doctor_name")
    turn = _CURRENT.get()
    if turn is None:  # вне хода (прямой вызов классификатора) — память до следующего `begin`
        turn = _Turn()
        _CURRENT.set(turn)
    turn.names = _LlmNames(person=person, doctor_name=doctor.strip() if isinstance(doctor, str) else "")


def _acted(switch: str, outcome: str) -> None:
    turn_telemetry.append("names_acted", f"{switch}:{outcome}")


def llm_patient_fio(text: str, *, switch: str) -> bool:
    """Реплика целиком — ФИО пациента по LLM этого хода (`patient_name` — те же слова, не меньше двух)."""

    current = _names()
    words = current.person.get("patient_name") if current else None
    found = bool(words) and len(words) >= 2 and words == _words(text)
    _acted(switch, "llm" if found else ("no_llm" if current is None else "not_named"))
    return found


def llm_person_word(word: str, *, switch: str) -> bool:
    """Одно слово реплики — фамилия пациента по LLM этого хода (`surname` или слово `patient_name`)."""

    current = _names()
    target = _words(word)
    found = bool(current) and len(target) == 1 and any(
        target == words or target[0] in words for words in current.person.values()
    )
    _acted(switch, "llm" if found else ("no_llm" if current is None else "not_named"))
    return found


def _surname_index(doctors: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for doc in doctors or []:
        fio = str(doc.get("fio") or "").split() if isinstance(doc, dict) else []
        key = _words(fio[0]) if fio else ()
        if len(key) == 1:
            index.setdefault(key[0], fio[0])
    return index


def doctor_surname_in(text: str, doctors: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Первое слово реплики, которое — фамилия врача справочника (точно, с падежной формой): ``(слово, фамилия)``."""

    index = _surname_index(doctors)
    for word in _words(text):
        if len(word) < _MIN_SURNAME_LEN:
            continue
        for variant in surname_variants(word):
            key = _words(variant)
            if len(key) == 1 and key[0] in index:
                return word, index[key[0]]
    return None


def match_catalog_doctor_by_data(text: str, doctors: list[dict[str, Any]]) -> dict[str, Any]:
    """`names_doctor_catalog = on`: врач реплики — по справочнику, без выбора слов списком (`match_catalog_doctor`)."""

    exact = doctor_surname_in(text, doctors)
    if exact:
        _acted("names_doctor_catalog", "exact")
        return {"status": "exact", "query": exact[0], "canonical": exact[1]}
    current = _names()
    named = _words(current.doctor_name) if current else ()
    index = _surname_index(doctors)
    for word in named:
        if len(word) < _FUZZY_MIN_LEN:
            continue
        hit = get_close_matches(word, list(index), n=1, cutoff=_FUZZY_CUTOFF)
        if hit and hit[0] != word:
            _acted("names_doctor_catalog", "fuzzy_llm")
            return {"status": "fuzzy", "query": word, "canonical": index[hit[0]], "matched_key": hit[0]}
    _acted("names_doctor_catalog", "miss")
    return {"status": "miss", "query": "", "canonical": ""}
