"""Слой устойчивости: классификация исхода апстрим-вызова + degraded-логи.

Изолированный модуль: без сетевых вызовов и доменных зависимостей, чтобы
тестироваться независимо. См. docs/superpowers/specs/2026-06-06-resilience-degraded-mode-design.md
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# --- Классы исхода апстрим-вызова ---
OK = "ok"
NOT_FOUND = "not_found"          # дозвонились, но по данным пусто (404 / ok+empty)
TECH_UNAVAILABLE = "tech_unavailable"  # не дозвонились/не распарсили (timeout/5xx/conn/exc)

# --- failure_mode для логов ---
FM_TIMEOUT = "timeout"
FM_CONN_ERROR = "conn_error"
FM_HTTP_5XX = "http_5xx"
FM_HTTP_404 = "http_404"
FM_EMPTY_DATA = "empty_data"
FM_EXCEPTION = "exception"


def classify_api_response(resp: Any, *, exc: Exception | None = None) -> str:
    """Классифицирует исход вызова api_nayka.

    :param resp: dict от api_nayka ({ok,status_code,error,data}) или None
    :param exc: пойманное исключение при вызове (если было)
    :return: OK | NOT_FOUND | TECH_UNAVAILABLE
    """
    if exc is not None:
        return TECH_UNAVAILABLE
    if not isinstance(resp, dict):
        return TECH_UNAVAILABLE
    if resp.get("ok"):
        return OK if resp.get("data") else NOT_FOUND
    sc = resp.get("status_code")
    if sc == 404:
        return NOT_FOUND
    if sc is None:
        return TECH_UNAVAILABLE
    if isinstance(sc, int) and 500 <= sc <= 599:
        return TECH_UNAVAILABLE
    return TECH_UNAVAILABLE
