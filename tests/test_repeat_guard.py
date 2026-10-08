import pytest

from messengers_router.memory import MemoryStore
from messengers_router.mess_types import ResponseEnvelope, SessionState
from messengers_router.router import (
    _REPEAT_GUARD_OFFER,
    _maybe_offer_operator_on_repeat,
)


def _state():
    return SessionState(session_id="t")


def _run(state, memory, text, handoff=False, user_text=""):
    env = ResponseEnvelope(text=text, handoff=handoff)
    return _maybe_offer_operator_on_repeat(env, state, memory, user_text)


def test_offer_appears_on_second_identical_answer():
    # Пункт 5 принципов (08.10): шаг дважды без продвижения → оператор.
    st, mem = _state(), MemoryStore()
    msg = "На какую дату и время вам удобно записаться на приём?"
    assert _REPEAT_GUARD_OFFER not in _run(st, mem, msg, user_text="к терапевту").text
    r2 = _run(st, mem, msg, user_text="в субботу утром")  # 1-й повтор → offer
    assert _REPEAT_GUARD_OFFER in r2.text
    assert msg in r2.text  # исходный текст сохранён: пациент видит, чего не хватает
    assert st.last_entities.get("_operator_offer_pending") is True


def test_patient_repeating_the_same_question_is_not_a_loop():
    # Тот же вопрос — тот же ответ: это не залип, а законный повтор.
    st, mem = _state(), MemoryStore()
    msg = "По услуге «ОАК» нашёл следующее: 390 руб."
    for _ in range(3):
        r = _run(st, mem, msg, user_text="Сколько стоит ОАК?")
    assert _REPEAT_GUARD_OFFER not in r.text
    assert not st.last_entities.get("_operator_offer_pending")


@pytest.mark.parametrize(
    "question",
    [
        "Скажите, пожалуйста, название услуги/анализа — я уточню стоимость.",
        "Для какой цели хотите подобрать анализы? Например: «проверить щитовидку», «витамины».",
        "Уточните, пожалуйста, детали запроса.",
        "Чтобы получить результат, укажите данные в таком порядке: фамилия, год рождения, код.",
    ],
    ids=["name_service", "test_assist_goal", "details", "result_data"],
)
def test_same_question_twice_on_different_replies_offers_operator(question):
    # Петли с трафика 08.10: тот же вопрос бота на две разные реплики пациента.
    st, mem = _state(), MemoryStore()
    _run(st, mem, question, user_text="АМГ")
    r = _run(st, mem, question, user_text="антимюллеров гормон")
    assert _REPEAT_GUARD_OFFER in r.text
    assert question in r.text


def test_same_data_answer_twice_is_allowed_third_offers():
    # Ответ с данными (цена, ссылка) второй раз подряд — ещё не залип; третий — оффер.
    st, mem = _state(), MemoryStore()
    msg = "По услуге «Флюорография» нашёл следующее: 900 руб."
    _run(st, mem, msg, user_text="Стоимость флюорографии")
    assert _REPEAT_GUARD_OFFER not in _run(st, mem, msg, user_text="Стоимость ЭКГ взрослого").text
    assert _REPEAT_GUARD_OFFER in _run(st, mem, msg, user_text="ЭКГ").text


def test_counter_resets_on_different_answer():
    st, mem = _state(), MemoryStore()
    a = "На какую дату и время вам удобно записаться на приём?"
    b = "Уточните, пожалуйста, фамилию врача или город приёма."
    _run(st, mem, a)
    _run(st, mem, b)  # другой ответ — сброс
    r = _run(st, mem, a)  # снова a, но не подряд — не повтор
    assert _REPEAT_GUARD_OFFER not in r.text
    assert not st.last_entities.get("_operator_offer_pending")


def test_short_answers_not_guarded():
    st, mem = _state(), MemoryStore()
    r = None
    for _ in range(4):
        r = _run(st, mem, "Хорошо.")
    assert _REPEAT_GUARD_OFFER not in r.text


def test_handoff_answer_not_guarded():
    st, mem = _state(), MemoryStore()
    r = None
    for _ in range(4):
        r = _run(st, mem, "Передаю заявку в обработку, ожидайте подтверждения записи.", handoff=True)
    assert _REPEAT_GUARD_OFFER not in r.text


def test_answer_mentioning_operator_not_guarded():
    st, mem = _state(), MemoryStore()
    r = None
    for _ in range(4):
        r = _run(st, mem, "Для уточнения могу перевести на оператора. Перевести на оператора?")
    assert _REPEAT_GUARD_OFFER not in r.text


def test_active_appointment_flow_skips_guard():
    """В активном APPOINTMENT-флоу O4 не должен срабатывать — у записи свои
    счётчики попыток и свой порог эскалации (на слоте), которые точнее."""
    st, mem = _state(), MemoryStore()
    st.last_entities["appointment_flow_active"] = True
    msg = "Есть возможность записи на приём к врачу Трубин в городе Самара по адресам: …"
    r = None
    for _ in range(5):
        r = _run(st, mem, msg)
    assert _REPEAT_GUARD_OFFER not in r.text
    assert not st.last_entities.get("_operator_offer_pending")


def test_already_pending_offer_is_skipped():
    st, mem = _state(), MemoryStore()
    st.last_entities["_operator_offer_pending"] = True
    msg = "Уточните, пожалуйста, фамилию врача или город приёма для записи."
    r = None
    for _ in range(4):
        r = _run(st, mem, msg)
    assert _REPEAT_GUARD_OFFER not in r.text


def test_loop_through_patient_stream_offers_operator_on_second_turn(monkeypatch):
    # Сквозь patient_routing_stream: реплика пациента доходит до гарда, оффер — в тексте.
    import asyncio
    from types import SimpleNamespace

    from messengers_router import orchestrator, router as router_mod
    from messengers_router.mess_types import Evidence, Plan, RouteDecision
    from messengers_router.services import Services

    question = "Скажите, пожалуйста, название услуги/анализа — я уточню стоимость."

    async def fake_run_pipeline(_text, _state, **_kwargs):
        return SimpleNamespace(
            response=ResponseEnvelope(text=question),
            decision=RouteDecision(label="PRICE", confidence=0.8),
            plan=Plan(label="PRICE"),
            evidence=Evidence(),
            stage_timings={},
        )

    monkeypatch.setattr(orchestrator, "run_pipeline", fake_run_pipeline)
    state, memory, services = _state(), MemoryStore(), Services()
    services.ensure_background_refresh_started = lambda: None

    def say(text):
        async def go():
            return "".join(e.text for e in [e async for e in router_mod.patient_routing_stream(text, state, services, memory)])

        return asyncio.run(go())

    assert _REPEAT_GUARD_OFFER not in say("сколько стоит Феринжект")
    assert say("Феринжект капельница").endswith(_REPEAT_GUARD_OFFER)
