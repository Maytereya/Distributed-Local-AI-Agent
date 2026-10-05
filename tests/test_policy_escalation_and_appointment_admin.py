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
from messengers_router.policies import detect_existing_appointment_request, handoff_message
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
        # Перенос — путь переноса на боте (04.10), а не «проверить запись не могу».
        ("перенесите мою запись на пятницу", None),
        ("сколько стоит приём кардиолога", None),
        ("Иванов Иван Иванович", None),
        # «Похоже по буквам» (BUG-2026-10-04-APPOINTMENT-VERB-LOOKALIKE): глагол без
        # объекта записи — не обращение по записи, и оператору сразу не уходит.
        ("нужно ли отменять лекарства перед анализом", None),
        ("надо ли отменить витамины перед сдачей крови", None),
        ("я перенесла ковид, какие анализы сдать", None),
    ],
    ids=[
        "check_verb", "check_participle", "confirm", "in_force", "have_any",
        "cancel_plain", "cancel_my", "cancel_visit",
        "book_new", "book_plain", "book_imperative", "reschedule", "reschedule_my", "price", "fio",
        "lookalike_cancel_drugs", "lookalike_cancel_vitamins", "lookalike_carried_illness",
    ],
)
def test_existing_appointment_detector(text, expected):
    assert detect_existing_appointment_request(text) == expected


def _install_stubs(monkeypatch, label: str = "APPOINTMENT", entities: dict | None = None):
    async def fake_analyze_with_candidates(_text, _state, runtime_options=None):
        _ = runtime_options
        return NLUResult(
            decision=RouteDecision(
                label=label,
                confidence=0.9,
                entities=dict(entities or {}),
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


def test_existing_appointment_check_routes_to_operator_offer(monkeypatch):
    _install_stubs(monkeypatch)
    state = SessionState(session_id="p6-check")
    memory = MemoryStore()

    decision, plan, evidence = run(
        router_mod.route_patient_message("Проверьте пожалуйста мою запись", state, Services(), memory)
    )

    assert decision.label == "OTHER"
    assert plan.label == "OTHER"
    assert "existing_appointment_check" in decision.flags
    payload = evidence.get("operator_offer_response")
    assert isinstance(payload, dict)
    assert str(payload.get("text") or "").endswith("Перевести на оператора?")
    # Проверка записи — ВОПРОС, а не принудительный перевод (решение 14.08).
    assert payload.get("handoff") is False
    assert state.last_entities.get("_operator_offer_pending") is True


@pytest.mark.parametrize("text", ["хочу отменить запись", "Отменить запись", "Отмените запись на 24.09 в 18:30"])
def test_existing_appointment_cancel_hands_off_at_once(monkeypatch, text):
    # Решение владельца 25.09 (заменяет 14.08 для отмены): сразу оператор, без
    # вопроса, тем же текстом, что у кнопки «Перенести или отменить запись».
    _install_stubs(monkeypatch)
    state = SessionState(session_id="p6-cancel")
    memory = MemoryStore()

    decision, _plan, evidence = run(router_mod.route_patient_message(text, state, Services(), memory))

    assert "existing_appointment_cancel" in decision.flags
    payload = evidence.get("operator_offer_response")
    assert payload == {"text": handoff_message("existing_appointment_change"), "handoff": True}
    assert not state.last_entities.get("_operator_offer_pending")
    assert memory.get_pending(state) is None


def _llm_says(monkeypatch, verdict: bool | None) -> list[str]:
    asked: list[str] = []

    async def validator(text):
        asked.append(text)
        return verdict

    monkeypatch.setattr(router_mod, "is_own_appointment_change", validator)
    return asked


def _route(text: str, session_id: str):
    state = SessionState(session_id=session_id)
    return run(router_mod.route_patient_message(text, state, Services(), MemoryStore()))


@pytest.mark.parametrize("text", ["нужно ли отменять лекарства перед анализом", "надо ли отменить витамины перед сдачей крови"])
def test_cancel_verb_without_appointment_is_not_sent_to_operator(monkeypatch, text):
    # Прод 04.10: вопрос о подготовке со словом «отменить» получал «Перенести или
    # отменить запись поможет оператор — соединяю» и уходил оператору. Глагол без
    # объекта записи — вопрос к LLM; она отвечает «другое».
    _install_stubs(monkeypatch, label="PREPARE")
    asked = _llm_says(monkeypatch, False)

    decision, _plan, evidence = _route(text, "p6-lookalike")

    assert asked == [text]
    assert decision.label == "PREPARE"
    assert not {f for f in decision.flags if f.startswith("existing_appointment")}
    assert evidence.get("operator_offer_response") is None


# Свип 04.10 (прод-LLM): без слова «запись» классификатор NLU относит эти просьбы к
# «прочему», и пациент получал «Не совсем понял ваш запрос».
_CANCEL_WITHOUT_OBJECT = ["хочу отменить, заболела", "не смогу прийти, отмените пожалуйста"]


@pytest.mark.parametrize("verdict", [True, None], ids=["llm_yes", "llm_unavailable"])
@pytest.mark.parametrize("text", _CANCEL_WITHOUT_OBJECT)
def test_cancel_without_appointment_word_goes_to_operator(monkeypatch, text, verdict):
    # LLM подтвердила или не ответила (тогда — как до 04.10): отмена — сразу оператор.
    _install_stubs(monkeypatch, label="OTHER")
    _llm_says(monkeypatch, verdict)

    decision, _plan, evidence = _route(text, "p6-cancel-no-object")

    assert "existing_appointment_cancel" in decision.flags
    assert evidence.get("operator_offer_response") == {"text": handoff_message("existing_appointment_change"), "handoff": True}


def test_reschedule_without_appointment_word_stays_with_the_bot(monkeypatch):
    # Перенос — на боте (решение владельца 04.10): путь переноса, не оператор.
    _install_stubs(monkeypatch, label="OTHER")
    _llm_says(monkeypatch, True)

    decision, _plan, evidence = _route("хочу перенести на пятницу", "p6-reschedule-no-object")

    assert decision.label == "APPOINTMENT"
    assert decision.entities.get("appointment_action") == "reschedule"
    assert evidence.get("operator_offer_response") is None


@pytest.mark.parametrize("text", ["отменить запись", *_CANCEL_WITHOUT_OBJECT])
def test_cancel_recognized_by_llm_classifier_goes_to_operator(monkeypatch, text):
    # Инвариант: отмена существующей записи — сразу оператор, кто бы её ни распознал —
    # правило или классификатор LLM (APPOINTMENT с действием «отмена»).
    _install_stubs(monkeypatch, label="APPOINTMENT", entities={"appointment_action": "cancel"})
    asked = _llm_says(monkeypatch, True)

    decision, _plan, evidence = _route(text, "p6-llm-cancel")

    assert asked == []  # метка уже APPOINTMENT — лишнего вопроса к LLM нет
    assert "existing_appointment_cancel" in decision.flags
    assert evidence.get("operator_offer_response", {}).get("handoff") is True


def _patient_sees(text: str) -> tuple[str, bool]:
    async def go():
        state, memory = SessionState(session_id="p6-render"), MemoryStore()
        out = [env async for env in router_mod.patient_routing_stream(text, state, Services(), memory)]
        return "".join(env.text for env in out if env.text), any(env.handoff for env in out)

    return run(go())


@pytest.mark.parametrize(
    ("text", "expected", "handoff"),
    [
        ("Проверьте пожалуйста мою запись", "Перевести на оператора?", False),
        ("Отменить запись", "Перенести или отменить запись поможет оператор — соединяю.", True),
    ],
    ids=["check", "cancel"],
)
def test_patient_sees_the_handler_answer_not_a_generic_clarify(monkeypatch, text, expected, handoff):
    """Инвариант `handler_answer_preempted_by_pending` — через ПОЛНЫЙ рендер.

    Тесты выше смотрят тройку решения: в evidence ответ лежал верный, а пациент
    получал «Уточните, пожалуйста, детали запроса» — рендер отдавал общий переспрос
    по pending раньше готового ответа (BUG-2026-09-25-CANCEL-OFFER-PREEMPTED).
    """

    _install_stubs(monkeypatch)
    answer, handed_off = _patient_sees(text)

    assert expected in answer, answer
    assert "Уточните, пожалуйста, детали запроса" not in answer
    assert handed_off is handoff


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
