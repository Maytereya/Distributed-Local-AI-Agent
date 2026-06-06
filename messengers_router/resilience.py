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


def failure_mode_from_response(resp: Any, *, exc: Exception | None = None) -> str:
    """Детализирует режим сбоя для логов/метрик."""
    if exc is not None:
        return FM_EXCEPTION
    if not isinstance(resp, dict):
        return FM_EXCEPTION
    if resp.get("ok"):
        return FM_EMPTY_DATA
    sc = resp.get("status_code")
    if sc == 404:
        return FM_HTTP_404
    if sc is None:
        return FM_TIMEOUT
    if isinstance(sc, int) and 500 <= sc <= 599:
        return FM_HTTP_5XX
    return FM_CONN_ERROR


def log_degraded(
    *,
    upstream: str,
    failure_mode: str,
    latency_ms: int | None = None,
    session_id: str | None = None,
    fallback_used: bool = False,
) -> None:
    """Одна структурная строка на degraded-событие (Datadog-ready схема)."""
    log.warning(
        "degraded_upstream upstream=%s failure_mode=%s latency_ms=%s session_id=%s fallback_used=%s",
        upstream, failure_mode, latency_ms, session_id, fallback_used,
    )


def mark_degraded(payload: dict, *, upstream: str, failure_mode: str, fallback_used: bool) -> dict:
    """Помечает payload как degraded (для логов и Фаза-3 сигнала оператору)."""
    payload["degraded"] = True
    payload["degraded_upstream"] = upstream
    payload["degraded_mode"] = failure_mode
    payload["degraded_fallback_used"] = fallback_used
    return payload


def tech_unavailable_text(what: str = "информацию") -> str:
    """Честное сообщение при тех-сбое апстрима."""
    return (
        f"По техническим причинам сейчас не удаётся загрузить {what}. "
        "Пожалуйста, попробуйте позже или напишите «оператор»."
    )
