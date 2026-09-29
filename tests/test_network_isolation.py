"""Гейт герметичен: тесты не выходят в сеть (29.09).

Раньше тесты через локальный config.ini ходили в прод-LLM и в МИС: прогон шёл
больше часа при плохой сети и грузил LLM, отвечающую пациентам. Защита — в
tests/conftest.py («Изоляция сети»). Эти тесты держат её от тихой поломки.
"""

import socket
import time

import pytest
import requests


def test_external_connection_is_blocked_fast():
    started = time.monotonic()
    with pytest.raises(ConnectionError):
        socket.create_connection(("172.16.0.28", 11434), timeout=5)
    assert time.monotonic() - started < 1.0  # отказ сразу, без сетевого таймаута


def test_http_client_sees_ordinary_connection_error():
    # Код бота должен видеть обычный отказ сети и идти по штатной деградации.
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("http://172.16.0.28:11434/api/tags", timeout=5)


def test_dns_is_blocked():
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.com", 80)


def test_localhost_is_allowed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        client = socket.create_connection(("127.0.0.1", server.getsockname()[1]), timeout=1)
        client.close()
    finally:
        server.close()


@pytest.mark.live
def test_live_marker_is_excluded_from_gate():
    # Под addopts `-m 'not live'` этот тест не выбирается. Если он выполнился в
    # обычном прогоне — фильтр живых тестов сломан.
    pytest.fail("тест с @pytest.mark.live попал в гейт")
