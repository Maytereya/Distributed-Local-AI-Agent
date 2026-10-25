"""П5 (прод 11.07 и 18.07): отказ всегда говорил «этого врача нет».

Симптомы:
- пациент представился «<ФИО>» → на «Записаться на приём» получил
  «К сожалению, этого врача нет в системе онлайн-записи»;
- «Могу ли я завтра пройти гинекологическое УЗИ» → тот же ответ, хотя названа
  УСЛУГА, а не врач.

Логика отказа корректна и сделана намеренно (`BUG-2026-06-01-01`, коммит
`03ecb0b`: не выдумывать филиалы под неопознанную цель). Неверна была
ФОРМУЛИРОВКА: единый текст `_doctor_not_bookable_via_bot_offer` всегда про врача.

Инвариант (класс): предмет отказа соответствует типу обронённой цели — врач /
услуга / нераспознанное. Собственное имя пациента целью записи не становится: на
него отвечаем вопросом о цели, а не «такого врача нет». Механика отказа
(сброс runtime-стейта, оффер оператора вопросом) при этом не меняется.
"""

from __future__ import annotations

import pytest

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import SessionState
from messengers_router.planner import _unbookable_target_kind
from messengers_router.response_builder import _doctor_not_bookable_via_bot_offer
from tests.names_modes import after_classifier, llm_answers, set_names


# --- Классификация обронённой цели ----------------------------------------

@pytest.mark.parametrize(
    "text, flags, expected",
    [
        ("Иванов Иван Иванович", set(), "patient_name"),
        ("Петрова Мария Сергеевна", set(), "patient_name"),
        ("Записаться к Дразнину", {"entity_dropped_doctor_like_service_name"}, "doctor"),
        ("Могу ли я завтра пройти гинекологическое УЗИ", set(), "service"),
        ("хочу записаться", set(), "unknown"),
        ("", set(), "unknown"),
    ],
    ids=["fio_full", "fio_female", "doctor_flag", "service_named", "no_target", "empty"],
)
def test_unbookable_target_kind_classification(monkeypatch, text, flags, expected):
    set_names(monkeypatch, "off")  # форма и списки; путь сущностей LLM — `test_patient_fio_is_never_a_missing_doctor_names_on`
    assert _unbookable_target_kind(text, flags) == expected


@pytest.mark.parametrize("text", ["Иванов Иван Иванович", "Петрова Мария Сергеевна"], ids=["fio_full", "fio_female"])
def test_patient_fio_is_never_a_missing_doctor_names_on(monkeypatch, text):
    """`names_* = on`: вид «ФИО пациента» — по LLM хода; без LLM — «нераспознанное». Инвариант тот же: на своё имя пациент
    получает вопрос о цели (тексты двух видов одинаковы), а не «такого врача нет»."""

    set_names(monkeypatch, "on")
    assert _unbookable_target_kind(text, set()) == "unknown"
    llm_answers(monkeypatch, label="APPOINTMENT", entities={"patient_name": text})
    assert after_classifier(text, {}, lambda: _unbookable_target_kind(text, set())) == "patient_name"
    lead = {kind: _doctor_not_bookable_via_bot_offer(SessionState(session_id="u"), MemoryStore(), kind=kind).text for kind in ("patient_name", "unknown")}
    assert lead["patient_name"] == lead["unknown"] and "врача нет" not in lead["unknown"]


def test_doctor_flag_wins_over_fio_shape():
    """Флаг грундера «похоже на врача» сильнее формы ФИО: пациент назвал ВРАЧА."""
    assert (
        _unbookable_target_kind("Дразнин Антон Владимирович", {"entity_dropped_doctor_like_service_name"})
        == "doctor"
    )


# --- Формулировки ---------------------------------------------------------

def _text(kind: str) -> str:
    state = SessionState(session_id=f"p5-{kind}")
    return _doctor_not_bookable_via_bot_offer(state, MemoryStore(), kind=kind).text


def test_doctor_wording_unchanged():
    assert "этого врача нет в системе онлайн-записи" in _text("doctor")


def test_service_wording_talks_about_service_not_doctor():
    text = _text("service")
    assert "услугу" in text
    assert "врача нет" not in text, text


@pytest.mark.parametrize("kind", ["patient_name", "unknown"])
def test_unrecognized_target_asks_for_goal_not_claims_missing_doctor(kind):
    text = _text(kind)
    assert "врача нет" not in text, text
    assert "не понял" in text.lower()


@pytest.mark.parametrize("kind", ["doctor", "service", "patient_name", "unknown"])
def test_all_kinds_keep_operator_offer_and_flow_reset(kind):
    """Механика отказа не меняется: вопрос про оператора + pending подтверждения."""
    state = SessionState(session_id=f"p5-flow-{kind}")
    memory = MemoryStore()

    envelope = _doctor_not_bookable_via_bot_offer(state, memory, kind=kind)

    assert envelope.text.endswith("Перевести на оператора?")
    assert envelope.handoff is False
    assert state.last_entities.get("_operator_offer_pending") is True
    pending = memory.get_pending(state)
    assert isinstance(pending, dict) and "operator_offer_confirm" in (pending.get("missing") or [])


def test_unknown_kind_value_falls_back_to_doctor_text():
    """Неизвестное значение маркера не должно ронять ответ (fail-safe)."""
    assert "этого врача нет" in _text("что-то новое")
