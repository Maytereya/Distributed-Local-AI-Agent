"""Структурный вызов LLM (ход (в), минимум (е)): роли, схема ответа, явные параметры, место вызова.

Инварианты класса:
- инструкция уходит только в роль system, текст пациента — только в user (SEC-3): реплика не может
  стать частью инструкции;
- форма ответа — JSON Schema (dict) в `format`;
- параметры генерации — только параметры места вызова: температура, `stop` и длина ответа из общих
  настроек (Gradio, операторский эндпоинт) в решение не попадают (LLM-2);
- прежний промпт (`system=None`) уходит тем же запросом, что до переноса: перенос клиента на модуль
  выбора доказывается на записанных ответах прод-LLM;
- сбой LLM — исключение из `LLM_UNAVAILABLE_ERRORS` и строка телеметрии с явным местом вызова.
"""

from __future__ import annotations

import asyncio

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


def _structured(**overrides):
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
    return asyncio.run(llm_runtime.generate_structured(**params))


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
