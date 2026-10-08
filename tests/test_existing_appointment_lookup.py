"""Вопрос о СВОЕЙ оформленной записи → честный оффер оператора (BUG-2026-10-08-EXISTING-APPT-LOOKUP-MISSED).

Разбор реальной переписки 08.10: узкий детектор пропускал «Мои записи», «Мои приёмы», «Мне
подтверждена запись?», «Я к лору записывался на сегодня», и пациент получал оформление НОВОЙ
записи, «Пока не понял, к какому врачу» или поиск результата анализов. Решение владельца 08.10:
бот оформленную запись не ищет и не показывает — вопрос о ней получает оффер оператора.

Механика: правило отбирает кандидатов (объект записи + признак уже оформленной записи, без
просьбы записаться и без цены), LLM отвечает «да / нет». Инварианты класса:
- вопрос о своей записи на ПЕРВОМ ходу → «Перевести на оператора?», без новой записи, без
  «Уточните детали» и без поиска результата — какую бы метку ни поставил NLU;
- новая запись, цена, «нужна ли запись» до LLM не доходят (встречный свип);
- LLM молчит или говорит «нет» → поведение как до 08.10.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from messengers_router import endpoint as endpoint_mod
from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.policies import detect_existing_appointment_candidate
from messengers_router.services import Services
from messengers_router.services import _existing_appointment_validator as V

_ROOT = Path(__file__).resolve().parents[1]

# Формулировки из реальной переписки 08.10 — обобщённые: без фамилий, дат и ПДн.
OWN_APPOINTMENT = [
    "Мои записи",
    "Мои приемы",
    "Мои приёмы",
    "Мне подтверждена запись?",
    "Я к лору записывался на сегодня",
    "Запись моя стоит к ревматологу на 22 октября?",
    "Когда у меня запись и по какому адресу",
    "Как проверить записана ли я",
    "забыла на сколько запись к врачу завтра",
    "Узнать о записях",
]

# Встречный свип: новая запись, цена, «нужна ли запись» — LLM не задаём вовсе.
NOT_A_CANDIDATE = [
    "Запись на УЗИ",
    "Запись к неврологу",
    "Есть запись к терапевту на завтра?",
    "Приём уролога",
    "ПЕРВИЧНЫЙ ПРИЕМ УРОЛОГА!",
    "Мне нужна запись к терапевту на воскресенье",
    "хочу записаться к кардиологу",
    "запишите меня на завтра",
    "сколько стоит приём уролога",
    "нужно ли записываться на анализы",
    "До скольки прием крови?",
]

_BANNED_IN_ANSWER = (
    "Уточните, пожалуйста, детали запроса",
    "Пока не понял",
    "год рождения",
    "выберите филиал",
    "На какую дату",
    "этого врача нет в системе онлайн-записи",
)


def _nlu_says(monkeypatch, label: str) -> None:
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(label=label, confidence=0.8, entities={}, flags=set(), source="llm_primary"),
            candidates=[],
            merged_from="llm",
        )

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        router_mod,
        "_env_flag",
        lambda name, default: True if name == "MR_ROUTER_V2_ENABLE" else (False if name == "MR_ROUTER_V2_SHADOW" else default),
    )


def _llm_says(monkeypatch, verdict: bool | None) -> list[str]:
    asked: list[str] = []

    async def validator(text):
        asked.append(text)
        return verdict

    monkeypatch.setattr(router_mod, "is_own_existing_appointment", validator)
    return asked


def _patient_sees(text: str, state: SessionState | None = None) -> tuple[str, bool]:
    async def go():
        st, memory = state or SessionState(session_id="existing-appt"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, st, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return asyncio.run(go())


# --- класс: вопрос о своей записи → оффер оператора, кто бы ни решал метку ------------

@pytest.mark.parametrize("label", ["OTHER", "APPOINTMENT", "DOCTOR_SCHEDULE", "TEST_RESULT"])
@pytest.mark.parametrize("text", OWN_APPOINTMENT)
def test_own_appointment_question_gets_operator_offer(monkeypatch, text, label):
    _nlu_says(monkeypatch, label)
    _llm_says(monkeypatch, True)

    answer, handed_off = _patient_sees(text)

    assert answer.endswith("Перевести на оператора?"), answer
    assert handed_off is False  # проверка записи — ВОПРОС, не принудительный перевод (14.08)
    for banned in _BANNED_IN_ANSWER:
        assert banned not in answer, (banned, answer)


@pytest.mark.parametrize("text", OWN_APPOINTMENT)
def test_rule_proposes_every_own_appointment_phrase(text):
    # Либо старый детектор, либо кандидат для LLM — фраза не теряется до LLM.
    from messengers_router.policies import detect_existing_appointment_request

    assert detect_existing_appointment_request(text) == "check" or detect_existing_appointment_candidate(text)


@pytest.mark.parametrize("label", ["APPOINTMENT", "PRICE", "OTHER"])
@pytest.mark.parametrize("text", NOT_A_CANDIDATE)
def test_new_booking_price_and_walkin_never_reach_llm(monkeypatch, text, label):
    asked = _llm_says(monkeypatch, True)  # даже «да» от LLM не должно понадобиться
    decision = RouteDecision(label=label, confidence=0.8, entities={}, flags=set())

    after = asyncio.run(
        router_mod._confirm_existing_appointment_question(
            decision, user_text=text, state=SessionState(session_id="existing-appt-sweep")
        )
    )

    assert detect_existing_appointment_candidate(text) is False
    assert asked == []
    assert after is decision


@pytest.mark.parametrize("verdict", [None, False], ids=["llm_unavailable", "llm_says_no"])
def test_llm_silence_or_no_keeps_previous_behaviour(monkeypatch, verdict):
    _nlu_says(monkeypatch, "OTHER")
    asked = _llm_says(monkeypatch, verdict)
    state = SessionState(session_id="existing-appt-fallback")

    decision, _plan, evidence = asyncio.run(
        router_mod.route_patient_message("Мои записи", state, Services(), MemoryStore())
    )

    assert asked == ["Мои записи"]
    assert not {f for f in decision.flags if f.startswith("existing_appointment")}
    assert evidence.get("operator_offer_response") is None


def test_not_asked_inside_active_booking(monkeypatch):
    # В активном оформлении «мои записи» — не повод бросать текущий шаг.
    asked = _llm_says(monkeypatch, True)
    state = SessionState(session_id="existing-appt-active")
    state.last_entities["appointment_flow_active"] = True
    decision = RouteDecision(label="APPOINTMENT", confidence=0.8, entities={}, flags=set())

    after = asyncio.run(router_mod._confirm_existing_appointment_question(decision, user_text="Мои записи", state=state))

    assert asked == []
    assert after is decision


def test_endpoint_once_own_appointment(monkeypatch):
    # Сквозь обработчик /api/messenger-generate-once — то, что видит шлюз.
    _nlu_says(monkeypatch, "APPOINTMENT")
    _llm_says(monkeypatch, True)
    memory, services = MemoryStore(), Services()
    services.ensure_background_refresh_started = lambda: None
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services

    resp = TestClient(app).post(
        "/api/messenger-generate-once",
        json={"session_id": "tg_existing_appt", "text": "Мне подтверждена запись?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"].endswith("Перевести на оператора?"), body["text"]
    assert body["handoff"] is False


# --- различитель: строгий разбор ответа, кэш, выключатель, промпт ------------------------

def _generate_answers(monkeypatch, answer) -> list[str]:
    prompts: list[str] = []

    async def fake_generate_text(prompt, **_kwargs):
        prompts.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(V, "_CACHE", {})
    monkeypatch.setattr(V, "generate_text", fake_generate_text)
    return prompts


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [("ДА", True), ("Да.", True), ("«НЕТ»\nэто новая запись", False), ("Возможно", None), ("", None), (TimeoutError(), None)],
    ids=["yes", "yes_punct", "no_with_reason", "unclear", "empty", "llm_failure"],
)
def test_validator_verdict_is_parsed_strictly(monkeypatch, answer, verdict):
    prompts = _generate_answers(monkeypatch, answer)

    assert asyncio.run(V.is_own_existing_appointment("Мои записи")) is verdict
    assert "«Мои записи»" in prompts[0]


def test_validator_caches_per_phrase(monkeypatch):
    prompts = _generate_answers(monkeypatch, "ДА")

    for text in ("Мои  записи", "мои записи"):
        assert asyncio.run(V.is_own_existing_appointment(text)) is True
    assert len(prompts) == 1


def test_validator_kill_switch_skips_llm(monkeypatch):
    prompts = _generate_answers(monkeypatch, "ДА")
    monkeypatch.setattr(V._cfg, "MR_LLM_EXISTING_APPOINTMENT_VALIDATION", False)

    assert asyncio.run(V.is_own_existing_appointment("Мои записи")) is None
    assert prompts == []


def test_prompt_copies_are_identical():
    bundle = (_ROOT / "messengers_router" / "prompts" / "existing_appointment_validator.txt").read_text(encoding="utf-8")
    host = (_ROOT / "app_data" / "prompts" / "mr_existing_appointment_validator.txt").read_text(encoding="utf-8")
    assert bundle == host
    assert "<<TEXT>>" in bundle
