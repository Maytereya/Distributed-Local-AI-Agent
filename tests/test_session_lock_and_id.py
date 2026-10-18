"""Замок сессии и номер чата — ход (д), DATA-11 (18.10, решение владельца: «делай по рекомендации»; «ни в коем случае
не передавать данные чатов пациентов между ними»).

Было:
- стрим (`/api/messenger-generate`) брал замок сессии ДО `StreamingResponse`, а отпускал в `finally` генератора: клиент,
  оборвавший соединение до первой итерации, оставлял замок занятым навсегда — чат пациента молчал до рестарта, ошибки
  в логе не было; `MemoryStore.aget` ждал замка без таймаута;
- запрос без `session_id` получал общий номер `anon` — диалоги разных пациентов смешивались бы в одной сессии.

Стало: замок берётся внутри генератора (не стартовал — не брался); ожидание замка — не дольше `SESSION_LOCK_TIMEOUT_S`,
затем честное «Предыдущее сообщение ещё обрабатывается…» и `log.error`; запрос без номера чата (нет поля, `null`,
пустая строка) — 422 и `log.error` (баг шлюза виден сразу).

Инварианты класса: незапущенный ответ не держит замок сессии; ход не ждёт замка вечно; сессии без номера нет.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from messengers_router import endpoint as endpoint_mod
from messengers_router.endpoint import MessengerGenerateRequest
from messengers_router.memory import MemoryStore
from messengers_router.services import Services


def _services() -> Services:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    return services


def _client(memory: MemoryStore) -> TestClient:
    services = _services()
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    return TestClient(app)


def test_stream_that_never_started_does_not_hold_the_session_lock():
    memory, services = MemoryStore(), _services()
    req = MessengerGenerateRequest(session_id="tg_disconnect", text="привет")

    async def scenario():
        await endpoint_mod.messenger_generate(req, memory=memory, services=services)  # клиент ушёл: тело не читали
        return memory._get_lock("tg_disconnect").locked()

    assert asyncio.run(scenario()) is False


def test_stream_releases_the_lock_after_a_full_answer():
    memory = MemoryStore()
    client = _client(memory)

    first = client.post("/api/messenger-generate", json={"session_id": "tg_stream", "text": "привет"})
    second = client.post("/api/messenger-generate", json={"session_id": "tg_stream", "text": "где вы находитесь?"})

    assert first.status_code == second.status_code == 200
    assert all(json.loads(line)["text"] for line in second.text.splitlines() if line.strip())
    assert memory._get_lock("tg_stream").locked() is False


@pytest.mark.parametrize("path", ["/api/messenger-generate-once", "/api/messenger-generate"])
def test_busy_session_gets_an_honest_answer_instead_of_waiting_forever(path, monkeypatch, caplog):
    monkeypatch.setattr(endpoint_mod, "SESSION_LOCK_TIMEOUT_S", 0.05)
    memory = MemoryStore()
    asyncio.run(memory.aget("tg_busy"))  # замок занят «предыдущим сообщением», которое не закончилось

    with caplog.at_level(logging.ERROR, logger=endpoint_mod.log.name):
        resp = _client(memory).post(path, json={"session_id": "tg_busy", "text": "алло?"})

    texts = [json.loads(line)["text"] for line in resp.text.splitlines() if line.strip()]
    assert resp.status_code == 200 and texts == [endpoint_mod.SESSION_BUSY_TEXT]
    assert any("session_lock_timeout" in r.getMessage() and "tg_busy" not in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("path", ["/api/messenger-generate-once", "/api/messenger-generate"])
@pytest.mark.parametrize("payload", [{"text": "сколько стоит ферритин"}, {"session_id": None, "text": "привет"}, {"session_id": "  ", "text": "привет"}], ids=["missing", "null", "blank"])
def test_request_without_a_chat_id_is_rejected(path, payload, caplog):
    memory = MemoryStore()

    with caplog.at_level(logging.ERROR, logger=endpoint_mod.log.name):
        resp = _client(memory).post(path, json=payload)

    assert resp.status_code == 422
    assert any("session_id_missing" in r.getMessage() for r in caplog.records)
    assert memory._db == {}  # никакой общей сессии не появилось
