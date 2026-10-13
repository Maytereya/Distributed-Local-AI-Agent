"""Сбой соединения с LLM объясним по телеметрии; повисшее соединение не дольше 2 с.

DEBT-2026-10-13-LLM-CONNECT-DIAG: в прогонах 11.10 мгновенные `ConnectionError` к LLM (сеть / VPN, не
пул httpx), но причину не видно — ollama 0.5.3 переводит `httpx.ConnectError` в
`ConnectionError(...) from None`, и errno сокета остаётся только в цепочке `__context__` / `__cause__`.
А соединение без таймаута повисает на весь бюджет хода.

Инварианты класса:
- любой вызов LLM (текст, поток, структурное решение) при сбое соединения пишет в запись `llm[]`
  телеметрии хода корневую причину — errno (`EHOSTDOWN`, `ECONNREFUSED`…) или класс корня — отдельным
  полем `err`; строка `outcome` прежняя (`error:ConnectionError`): по ней точным совпадением считают
  карточка стенда и `HarnessClient`;
- лог модуля выбора называет ту же причину;
- клиент LLM ждёт установки соединения не дольше 2 с, длину генерации не ограничивает (её держит
  `asyncio.wait_for` места вызова).
"""

from __future__ import annotations

import asyncio
import errno
import logging

import httpx
import pytest

from messengers_router import choice as ch
from messengers_router import llm_runtime, turn_telemetry
from tests.test_choice import _TEMPLATE, OPTIONS, TASK


def _host_down() -> ConnectionError:
    """Как бросает ollama 0.5.3, когда VPN не поднят: OSError → httpx.ConnectError → ConnectionError."""

    try:
        try:
            try:
                raise OSError(errno.EHOSTDOWN, "Host is down")
            except OSError:
                # Неявная цепочка (`__context__`), как в httpcore.
                raise httpx.ConnectError("All connection attempts failed")
        except httpx.ConnectError:
            raise ConnectionError("Failed to connect to Ollama.") from None
    except ConnectionError as exc:
        return exc


class _HostDown:
    async def generate(self, **_kwargs):
        raise _host_down()


async def _text():
    await llm_runtime.generate_text("промпт", timeout_s=5, queue_timeout_ms=1000)


async def _stream():
    async for _chunk in llm_runtime.generate_stream_text("промпт", timeout_s=5, queue_timeout_ms=1000):
        pass


async def _structured():
    await llm_runtime.generate_structured(
        site="choose_demo", system="инструкция", user="данные", schema={"type": "object"}, options={"temperature": 0}, timeout_s=5, queue_timeout_ms=1000
    )


@pytest.mark.parametrize("call", [_text, _stream, _structured], ids=["text", "stream", "structured"])
def test_connection_failure_names_its_errno_and_keeps_the_outcome(call, monkeypatch):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _HostDown())
    rows: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", rows.append)

    async def turn():
        token = turn_telemetry.begin("once", session_id="s")
        with pytest.raises(ConnectionError):
            await call()
        turn_telemetry.finish(token, handoff=False, chars=0)

    asyncio.run(turn())
    [entry] = rows[0]["llm"]
    assert entry["outcome"] == "error:ConnectionError"
    assert entry["err"] == "EHOSTDOWN"


def test_success_has_no_err_field(monkeypatch):
    from ollama import GenerateResponse

    class _Ok:
        async def generate(self, **_kwargs):
            return GenerateResponse(model="m", response="ок", done=True)

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Ok())
    rows: list[dict] = []
    monkeypatch.setattr(turn_telemetry, "_emit", rows.append)

    async def turn():
        token = turn_telemetry.begin("once", session_id="s")
        await _text()
        turn_telemetry.finish(token, handoff=False, chars=0)

    asyncio.run(turn())
    assert "err" not in rows[0]["llm"][0]


def _chained(outer: Exception, inner: BaseException) -> Exception:
    outer.__context__ = inner
    return outer


@pytest.mark.parametrize(
    ("exc", "detail"),
    [
        (_host_down(), "EHOSTDOWN"),
        # Как на деле: errno лежит под OSError без errno («All connection attempts failed» у anyio).
        (_chained(ConnectionError("x"), _chained(httpx.ConnectError("y"), _chained(OSError("all failed"), ConnectionRefusedError(errno.ECONNREFUSED, "refused")))), "ECONNREFUSED"),
        (_chained(ConnectionError("x"), httpx.ReadError("eof")), "ReadError"),  # errno нет — класс корня
        (ConnectionError("без цепочки"), None),
        (asyncio.TimeoutError(), None),  # исход «timeout» объясняет сам
        (llm_runtime.LLMQueueTimeoutError("q"), None),
    ],
    ids=["errno", "errno-deep", "root-class", "no-chain", "timeout", "queue"],
)
def test_error_detail_is_the_root_cause(exc, detail):
    assert llm_runtime.error_detail(exc) == detail


def test_choose_log_names_the_root_cause(monkeypatch, caplog):
    monkeypatch.setattr(ch, "load_prompt_text", lambda key: _TEMPLATE)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _HostDown())
    with caplog.at_level(logging.WARNING, logger=ch.logger.name):
        choice = asyncio.run(ch.choose(TASK, "сколько стоит оак", OPTIONS))
    assert choice.status is ch.Status.UNAVAILABLE
    assert any("choose_unavailable" in r.getMessage() and "EHOSTDOWN" in r.getMessage() for r in caplog.records)


def test_client_waits_for_connection_two_seconds_and_not_for_generation():
    # Без сети: клиент создаётся, соединения нет. Гейт подменяет `_OLLAMA_CLIENT` (`_LLMDown`), поэтому
    # проверяется фабрика, которой модуль создаёт клиента.
    timeout = llm_runtime._new_client()._client.timeout
    assert timeout.connect == 2.0
    assert timeout.read is None and timeout.write is None and timeout.pool is None
