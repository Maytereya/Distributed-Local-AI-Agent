"""П6 плана — политика: факты и эскалация (три пункта одним коммитом).

1. **Онлайн-консультации клиника НЕ проводит** — факт в промпт, обе локации.
   Делается ПОСЛЕ П2 (код уже не подставляет обычный приём вместо онлайн),
   иначе промпт прикрыл бы симптом и оставил корень.
2. **Существующая запись → оператор вопросом.** Решение владельца 14.08:
   проверку существующей записи бот не делает; распознаём намерение и предлагаем
   оператора, НЕ предлагая создать новую. Дополнение владельца: **отмена записи
   обрабатывается так же** — реквизиты отмены бот не собирает.
3. **Два непонятых подряд → предложить оператора ВОПРОСОМ**, не переводить
   принудительно: принудительный перевод сбрасывал людей, которым бот отвечает
   нормально.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router import router as router_mod
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import Evidence, RouteDecision, SessionState
from messengers_router.nlu_pipeline import NLUResult
from messengers_router.policies import detect_existing_appointment_request
from messengers_router.recovery_policy import (
    UNCLEAR_OPERATOR_OFFER_TEXT,
    evaluate_recovery,
)
from messengers_router.services import Services

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "messengers_router" / "prompts"
_RUNTIME_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "app_data" / "prompts"


def run(coro):
    return asyncio.run(coro)


# --- П6-1: факт про онлайн-консультации в обеих локациях -------------------

@pytest.mark.parametrize("key", ["renderer_patient", "renderer_patient_rich"])
def test_online_consultation_policy_in_both_prompt_locations(key):
    """Промпты живут в ДВУХ локациях; менять надо обе, иначе прод возьмёт старую."""
    for path in (_PROMPTS_DIR / f"{key}.txt", _RUNTIME_PROMPTS_DIR / f"mr_{key}.txt"):
        assert path.exists(), f"prompt missing: {path}"
        body = path.read_text(encoding="utf-8")
        assert "Онлайн-консультаций" in body, f"факт про онлайн отсутствует в {path}"
        assert "НЕ проводит" in body, path


def test_online_policy_prompt_copies_are_identical():
    """Расхождение копий = прод и локальный прогон ведут себя по-разному."""
    for key in ("renderer_patient", "renderer_patient_rich"):
        bundle = (_PROMPTS_DIR / f"{key}.txt").read_text(encoding="utf-8")
        host = (_RUNTIME_PROMPTS_DIR / f"mr_{key}.txt").read_text(encoding="utf-8")
        assert bundle == host, f"копии промпта {key} разошлись"


# --- П6-2: существующая запись (проверка И отмена) → оператор вопросом ------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Проверьте пожалуйста мою запись", "check"),
        ("я записан на завтра?", "check"),
        ("подтвердите мою запись", "check"),
        ("моя запись в силе?", "check"),
        ("есть ли у меня запись на приём", "check"),
        ("хочу отменить запись", "cancel"),
        ("отмените мою запись пожалуйста", "cancel"),
        ("отменить приём", "cancel"),
        # НЕ обращение по существующей записи — обычные сценарии:
        ("хочу записаться к кардиологу", None),
        ("записаться на приём", None),
        ("запишите меня на завтра", None),
        ("перенести запись", None),
        ("сколько стоит приём кардиолога", None),
        ("Иванов Иван Иванович", None),
    ],
    ids=[
        "check_verb", "check_participle", "confirm", "in_force", "have_any",
        "cancel_plain", "cancel_my", "cancel_visit",
        "book_new", "book_plain", "book_imperative", "reschedule", "price", "fio",
    ],
)
def test_existing_appointment_detector(text, expected):
    assert detect_existing_appointment_request(text) == expected


def _install_stubs(monkeypatch, label: str = "APPOINTMENT"):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label=label,
                confidence=0.9,
                entities={},
                flags={"rule_appointment"},
                needs_handoff=False,
            ),
            candidates=[],
            merged_from="rule",
        )

    def fake_env_flag(name: str, default: bool) -> bool:
        if name == "MR_ROUTER_V2_ENABLE":
            return True
        if name == "MR_ROUTER_V2_SHADOW":
            return False
        return default

    async def fake_execute_plan(_plan, _state, _services):
        return Evidence()

    monkeypatch.setattr(router_mod, "analyze_with_candidates", fake_analyze_with_candidates)
    monkeypatch.setattr(router_mod, "_env_flag", fake_env_flag)
    monkeypatch.setattr(router_mod, "execute_plan", fake_execute_plan)


@pytest.mark.parametrize(
    "text, marker",
    [
        ("Проверьте пожалуйста мою запись", "existing_appointment_check"),
        ("хочу отменить запись", "existing_appointment_cancel"),
    ],
    ids=["check", "cancel"],
)
def test_existing_appointment_routes_to_operator_offer(monkeypatch, text, marker):
    _install_stubs(monkeypatch)
    state = SessionState(session_id=f"p6-{marker}")
    memory = MemoryStore()

    decision, plan, evidence = run(
        router_mod.route_patient_message(text, state, Services(), memory)
    )

    assert decision.label == "OTHER"
    assert plan.label == "OTHER"
    assert marker in decision.flags
    payload = evidence.get("operator_offer_response")
    assert isinstance(payload, dict)
    assert str(payload.get("text") or "").endswith("Перевести на оператора?")
    # Оффер — ВОПРОС, а не принудительный перевод.
    assert payload.get("handoff") is False
    assert state.last_entities.get("_operator_offer_pending") is True


def test_existing_appointment_does_not_offer_new_booking(monkeypatch):
    """Решение владельца: не предлагать создание НОВОЙ записи."""
    _install_stubs(monkeypatch)
    state = SessionState(session_id="p6-no-new-booking")

    _decision, plan, evidence = run(
        router_mod.route_patient_message("я записан на завтра?", state, Services(), MemoryStore())
    )

    assert plan.steps == []
    text = str((evidence.get("operator_offer_response") or {}).get("text") or "")
    for banned in ("запишу", "записать вас", "выберите филиал", "к какому врачу"):
        assert banned not in text.lower(), text


def test_cancel_inside_active_flow_is_not_hijacked(monkeypatch):
    """В АКТИВНОМ оформлении «отменить» = прервать текущий процесс, а не отменить
    существующую запись — старый путь сохраняется."""
    _install_stubs(monkeypatch)
    state = SessionState(session_id="p6-active-flow")
    state.last_entities["appointment_flow_active"] = True

    decision, _plan, _evidence = run(
        router_mod.route_patient_message("отменить", state, Services(), MemoryStore())
    )

    assert "existing_appointment_operator_offer" not in decision.flags


# --- П6-3: два непонятых подряд → оффер оператора вопросом ------------------

def _unclear_decision() -> RouteDecision:
    return RouteDecision(
        label="OTHER",
        confidence=0.31,
        source="llm_primary",
        flags={"low_confidence"},
        clarify_needed=False,
    )


def _recover(state_entities: dict) -> object:
    return evaluate_recovery(
        user_text="ыыы",
        decision=_unclear_decision(),
        flow_label="OTHER",
        pending_exists=False,
        flow_active=False,
        state_entities=state_entities,
        summary="",
        max_unclear=3,
    )


def test_first_unclear_turn_is_plain_clarify():
    entities: dict = {}
    action = _recover(entities)
    assert action.kind == "clarify"
    assert action.offer_operator is False
    assert UNCLEAR_OPERATOR_OFFER_TEXT not in action.text


def test_second_unclear_turn_offers_operator_as_question():
    entities: dict = {}
    _recover(entities)
    action = _recover(entities)

    assert action.kind == "clarify", action
    assert action.handoff is False, "перевод должен быть предложением, а не принуждением"
    assert action.offer_operator is True
    assert UNCLEAR_OPERATOR_OFFER_TEXT in action.text
    assert action.text.strip().endswith("Перевести на оператора?")


def test_unclear_escalation_never_forces_handoff():
    """Класс: сколько бы непонятых подряд ни было, бот не переводит сам."""
    entities: dict = {}
    for _ in range(5):
        action = _recover(entities)
        assert action.handoff is False, action


def test_understood_turn_resets_unclear_counter():
    entities: dict = {}
    _recover(entities)
    clear = evaluate_recovery(
        user_text="сколько стоит ОАК",
        decision=RouteDecision(label="PRICE", confidence=0.9, source="rule"),
        flow_label="PRICE",
        pending_exists=False,
        flow_active=False,
        state_entities=entities,
        summary="",
        max_unclear=3,
    )
    assert clear.kind == "none"
    assert entities.get("_nlu_unclear_count") == 0
