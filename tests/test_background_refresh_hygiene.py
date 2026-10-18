"""Гигиена фонового обновления срезов МИС — ход (д), шаг 9а, DATA-10 (17.10).

Было: (1) заметки колл-центра качались `requests.get` без таймаута — зависшее соединение навсегда останавливало
утренний цикл обновления врачей 07:45; (2) все писатели среза писали во временный файл с одним именем `<срез>.tmp` —
две параллельные сборки (фон и первый запрос пациента) смешивали строки, и битый срез жил весь день; (3) сбой
фонового обновления врачей печатался `print`, а сбой старта планировщиков гасился `pass` — в логе контейнера ничего.

Стало: таймаут `(10, 60)`; временный файл — свой у каждого писателя (`mkstemp`), права итогового файла прежние
(0644); сбои — `logger.warning` с трассировкой. Ответы пациенту не меняются.

Инварианты класса: фоновый сетевой вызов МИС — с таймаутом; запись среза атомарна и не делит временный файл с другим
писателем; сбой фона виден в логе. Тесты — во временной папке (ловушка `CLAUDE.md` про очистку срезов).
"""

from __future__ import annotations

import asyncio
import json
import logging
import stat
from pathlib import Path

import pytest
import requests

from agent_logic_2.nayka_api import snapshots
from messengers_router.services import Services

# Настоящий запуск планировщиков: `conftest` подменяет его пустышкой на каждый тест.
_REAL_START = Services.ensure_background_refresh_started


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "price_region_3_20261017.jsonl"

    snapshots.write_rows(path, [{"id": 1}, {"id": 2}])

    assert _rows(path) == [{"id": 1}, {"id": 2}]
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_failed_write_keeps_the_previous_snapshot(tmp_path):
    path = tmp_path / "price_region_3_20261017.jsonl"
    snapshots.write_rows(path, [{"id": "old"}])

    def broken():
        yield {"id": "new"}
        raise ConnectionError("обрыв на середине")

    with pytest.raises(ConnectionError):
        snapshots.write_rows(path, broken())

    assert _rows(path) == [{"id": "old"}]
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_two_writers_do_not_share_a_temp_file(tmp_path):
    """Второй писатель стартует, пока первый пишет: у каждого свой временный файл — итог цельный, строки не смешаны."""

    path = tmp_path / "doctors_20261017.jsonl"

    def first():
        yield {"writer": 1, "n": 0}
        snapshots.write_rows(path, [{"writer": 2, "n": 0}, {"writer": 2, "n": 1}])  # параллельная сборка
        yield {"writer": 1, "n": 1}

    snapshots.write_rows(path, first())

    rows = _rows(path)
    assert {row["writer"] for row in rows} == {1} and len(rows) == 2  # последний записавший — целиком
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_call_center_notes_are_fetched_with_a_timeout(monkeypatch, tmp_path):
    from agent_logic_2.nayka_api import doctors_cc_info

    monkeypatch.setattr(doctors_cc_info, "CACHE_DIR", str(tmp_path))
    calls: list[dict] = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        raise requests.ConnectionError("МИС недоступна")

    monkeypatch.setattr(doctors_cc_info.requests, "get", fake_get)

    doctors_cc_info.get_doctors_cc_info(force=True)

    assert calls and calls[0].get("timeout") == doctors_cc_info.HTTP_TIMEOUT
    assert list(tmp_path.iterdir()) == []


def test_doctors_refresh_failure_is_logged(monkeypatch, caplog):
    from agent_logic_2.nayka_api import api_nayka

    def no_doctors():
        raise ConnectionError("МИС недоступна")

    monkeypatch.setattr(api_nayka, "get_all_doctors", no_doctors)
    monkeypatch.setattr(api_nayka, "save_doctors_data", lambda docs: pytest.fail("сбой не должен записывать срез"))

    with caplog.at_level(logging.WARNING, logger=api_nayka.logger.name):
        asyncio.run(api_nayka._refresh_once())

    failures = [r for r in caplog.records if "DAILY REFRESH" in r.getMessage()]
    assert failures and failures[0].exc_info is not None


def test_scheduler_start_failure_is_logged(monkeypatch, caplog):
    from agent_logic_2.nayka_api import api_nayka
    from messengers_router.services import core

    def broken():
        raise RuntimeError("планировщик упал")

    monkeypatch.setattr(api_nayka, "ensure_daily_refresh_started", broken)

    with caplog.at_level(logging.WARNING, logger=core.logger.name):
        _REAL_START(Services.__new__(Services))  # не бросает

    assert any("обновления срезов" in r.getMessage() and r.exc_info for r in caplog.records)
