"""Структурный вызов LLM (ход (в), минимум (е)): роли, схема ответа, явные параметры, место вызова.

Инварианты класса:
- инструкция уходит только в роль system, текст пациента — только в user (SEC-3): реплика не может
  стать частью инструкции;
- форма ответа — JSON Schema (dict) в `format`;
- параметры генерации — только параметры места вызова: температура, `stop` и длина ответа из общих
  настроек (Gradio, операторский эндпоинт) в решение не попадают (LLM-2);
- прежний промпт (`system=None`) уходит тем же запросом, что до переноса: перенос клиента на модуль
  выбора доказывается на записанных ответах прод-LLM;
- сбой LLM — исключение из `LLM_UNAVAILABLE_ERRORS` и строка телеметрии с явным местом вызова;
- бюджет хода (L-11): решений сверх бюджета в LLM не уходит, тайм-аут вызова — не больше остатка
  бюджета; у каждого хода бюджет свой, вне хода бюджета нет;
- предохранитель (L-11): N сбоев генерации подряд — пауза без вызовов, после неё одна проба;
  таймаут очереди (занятость соседями) не размыкает; текст и прежние слои он не трогает.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest
from ollama import GenerateResponse

from agent_logic_2 import ollama_settings
from messengers_router import llm_runtime, turn_telemetry

_SCHEMA = {"type": "object", "properties": {"picks": {"type": "array", "items": {"type": "integer", "enum": [1, 2]}}}, "required": ["picks"]}
_SHARED = {"temperature": 0.2, "top_p": 0.9, "num_predict": 2048, "stop": ["<|eot_id|>", "<|im_end|>"]}


class _Capture:
    """Клиент ollama: запоминает запрос, отвечает заданным текстом или падает."""

    def __init__(self, reply: str = '{"picks": [1]}', exc: BaseException | None = None) -> None:
        self.calls: list[dict] = []
        self.reply = reply
        self.exc = exc

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return GenerateResponse(model=str(kwargs.get("model", "")), response=self.reply, done=True, prompt_eval_count=40, eval_count=5)


@pytest.fixture
def client(monkeypatch) -> _Capture:
    fake = _Capture()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
    monkeypatch.setattr(ollama_settings, "_cached_opts", dict(_SHARED))
    return fake


def _params(**overrides) -> dict:
    params = {
        "site": "choose_demo",
        "system": "ИНСТРУКЦИЯ: выбери номер из списка.",
        "user": "Сообщение пациента: «сколько стоит оак»\n1. ОАК — 500 руб.\n2. ОАМ — 300 руб.",
        "schema": _SCHEMA,
        "options": {"temperature": 0, "num_predict": 64},
        "timeout_s": 5,
        "queue_timeout_ms": 1000,
    }
    params.update(overrides)
    return params


def _structured(**overrides):
    return asyncio.run(llm_runtime.generate_structured(**_params(**overrides)))


def test_instruction_goes_to_system_and_patient_text_to_user(client):
    text, usage = _structured()

    request = client.calls[0]
    assert request["system"] == "ИНСТРУКЦИЯ: выбери номер из списка."
    assert request["prompt"].startswith("Сообщение пациента: «сколько стоит оак»")
    assert "ИНСТРУКЦИЯ" not in request["prompt"]
    assert request["format"] == _SCHEMA
    assert (text, usage["eval_count"]) == ('{"picks": [1]}', 5)


def test_generation_options_are_the_call_site_ones_not_shared_settings(client):
    _structured()

    assert dict(client.calls[0]["options"]) == {"temperature": 0, "num_predict": 64}


def test_legacy_prompt_is_the_same_request_as_before_the_move(client):
    prompt = "Сообщение пациента: «сколько стоит оак»\nОтветь только JSON"
    asyncio.run(llm_runtime.generate_text(prompt, timeout_s=12, queue_timeout_ms=3000, fmt="json", think=False, options={"temperature": 0}))
    _structured(system=None, user=prompt, schema="json", options={"temperature": 0}, inherit_options=True)

    before, after = client.calls
    assert after == before
    assert "system" not in after


def test_failure_is_unavailable_and_noted_with_explicit_site(monkeypatch):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Capture(exc=ConnectionError("ollama down")))
    rows: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", rows.append)
    token = turn_telemetry.begin("once", session_id="s")

    with pytest.raises(llm_runtime.LLM_UNAVAILABLE_ERRORS):
        _structured()
    turn_telemetry.finish(token, handoff=False, chars=0)

    assert [(c["site"], c["outcome"]) for c in rows[0]["llm"]] == [("choose_demo", "error:ConnectionError")]


def test_success_is_noted_with_explicit_site_and_tokens(monkeypatch, client):
    rows: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", rows.append)
    token = turn_telemetry.begin("once", session_id="s")

    _structured()
    turn_telemetry.finish(token, handoff=False, chars=0)

    call = rows[0]["llm"][0]
    assert (call["site"], call["outcome"], call["in_tok"], call["out_tok"]) == ("choose_demo", "ok", 40, 5)


# --------------------------------------------------------------------------- бюджет хода и предохранитель (L-11)


@pytest.fixture
def limits(monkeypatch):
    """Бюджет хода и предохранитель с явными значениями (по умолчанию — как в config.py)."""

    from messengers_router.runtime_config import config as cfg

    def set_limits(calls: int = 2, seconds: int = 10, failures: int = 3, open_s: int = 30) -> None:
        monkeypatch.setattr(cfg, "MR_LLM_DECISION_TURN_CALLS", calls, raising=False)
        monkeypatch.setattr(cfg, "MR_LLM_DECISION_TURN_SECONDS", seconds, raising=False)
        monkeypatch.setattr(cfg, "MR_LLM_DECISION_BREAKER_FAILURES", failures, raising=False)
        monkeypatch.setattr(cfg, "MR_LLM_DECISION_BREAKER_OPEN_S", open_s, raising=False)

    set_limits()
    return set_limits


def _in_turn(make_coro):
    """Корутина внутри хода: бюджет решений открыт, как в эндпоинте."""

    async def run():
        token = llm_runtime.begin_turn_budget()
        try:
            return await make_coro()
        finally:
            llm_runtime.end_turn_budget(token)

    return asyncio.run(run())


async def _decisions(count: int) -> list[str]:
    out = []
    for _ in range(count):
        try:
            await llm_runtime.generate_structured(**_params())
            out.append("ok")
        except llm_runtime.LLMBudgetExhaustedError:
            out.append("budget")
    return out


def test_decision_over_the_turn_budget_is_not_sent_to_llm(client, limits):
    assert _in_turn(lambda: _decisions(3)) == ["ok", "ok", "budget"]
    assert len(client.calls) == 2


def test_budget_is_per_turn_and_absent_outside_a_turn(client, limits):
    assert _in_turn(lambda: _decisions(2)) == ["ok", "ok"]
    assert _in_turn(lambda: _decisions(2)) == ["ok", "ok"]
    assert asyncio.run(_decisions(3)) == ["ok", "ok", "ok"]  # скрипт, тест сервиса — без хода
    assert len(client.calls) == 7


def test_turn_time_budget_caps_the_call_timeout(monkeypatch, limits):
    limits(seconds=1)

    class _Hung:
        calls = 0

        async def generate(self, **_kwargs):
            _Hung.calls += 1
            await asyncio.sleep(30)

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Hung())

    async def two_decisions():
        started = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await llm_runtime.generate_structured(**_params(timeout_s=8))
        waited = time.monotonic() - started
        with pytest.raises(llm_runtime.LLMBudgetExhaustedError):
            await llm_runtime.generate_structured(**_params(timeout_s=8))
        return waited

    waited = _in_turn(two_decisions)
    assert waited < 3  # остаток бюджета хода (1 с), а не тайм-аут места вызова (8 с)
    assert _Hung.calls == 1


def test_breaker_opens_after_consecutive_failures_and_probes_once_after_pause(monkeypatch, limits):
    fake = _Capture(exc=ConnectionError("ollama down"))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)

    for _ in range(3):
        with pytest.raises(ConnectionError):
            _structured()
    with pytest.raises(llm_runtime.LLMCircuitOpenError):
        _structured()
    assert len(fake.calls) == 3

    llm_runtime._BREAKER.open_until = 0.0  # пауза прошла: одна проба, её сбой снова размыкает
    with pytest.raises(ConnectionError):
        _structured()
    with pytest.raises(llm_runtime.LLMCircuitOpenError):
        _structured()
    assert len(fake.calls) == 4

    llm_runtime._BREAKER.open_until = 0.0  # успех пробы замыкает
    fake.exc = None
    _structured()
    _structured()
    assert len(fake.calls) == 6


def test_queue_timeout_does_not_open_the_breaker(monkeypatch, limits):
    # Очередь к LLM заняли соседи (колл-центр, FreeTalk): это сигнал, а не сбой модели — иначе
    # качество бота зависит от чужой нагрузки (L-11).
    @asynccontextmanager
    async def full_queue(_queue_timeout_ms):
        raise llm_runtime.LLMQueueTimeoutError("llm_queue_timeout")
        yield  # pragma: no cover

    monkeypatch.setattr(llm_runtime, "llm_slot", full_queue)
    for _ in range(5):
        with pytest.raises(llm_runtime.LLMQueueTimeoutError):
            _structured()
    assert llm_runtime._BREAKER.allows(time.monotonic())


def test_budget_and_breaker_do_not_touch_text_generation(client, limits):
    llm_runtime._BREAKER.open_until = time.monotonic() + 60

    async def four_texts():
        for _ in range(4):
            await llm_runtime.generate_text("свободный ответ", timeout_s=5, queue_timeout_ms=1000)

    _in_turn(four_texts)
    assert len(client.calls) == 4


def test_budget_and_breaker_outcomes_reach_turn_telemetry(monkeypatch, client, limits):
    rows: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", rows.append)

    async def turn():
        token = turn_telemetry.begin("once", session_id="s")
        await _decisions(3)
        llm_runtime._BREAKER.open_until = time.monotonic() + 60
        with pytest.raises(llm_runtime.LLMCircuitOpenError):
            await llm_runtime.generate_structured(**_params())
        turn_telemetry.finish(token, handoff=False, chars=0)

    _in_turn(turn)
    assert [c["outcome"] for c in rows[0]["llm"]] == ["ok", "ok", "budget", "circuit_open"]


def test_endpoint_opens_a_fresh_decision_budget_per_turn(monkeypatch):
    from messengers_router import endpoint
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import ResponseEnvelope
    from messengers_router.services import Services

    seen: list = []

    async def fake_routing(*_args, **_kwargs):
        budget = llm_runtime._TURN_BUDGET.get()
        seen.append((budget, budget.calls if budget is not None else None))
        if budget is not None:
            budget.calls += 1
        yield ResponseEnvelope(text="ок")

    monkeypatch.setattr(endpoint, "patient_routing_stream", fake_routing)

    async def two_turns():
        memory, services = MemoryStore(), Services()
        for _ in range(2):
            await messenger_generate_once(MessengerGenerateRequest(session_id="budget", text="ферритин"), memory=memory, services=services)

    asyncio.run(two_turns())
    assert [calls for _budget, calls in seen] == [0, 0]  # бюджет хода не наследует вызовы прошлого
    assert seen[0][0] is not None and seen[0][0] is not seen[1][0]
