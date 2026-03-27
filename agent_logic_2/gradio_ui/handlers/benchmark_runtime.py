from __future__ import annotations

import threading

_state_lock = threading.Lock()
_running = False
_stop_requested = False


def benchmark_is_running() -> bool:
    with _state_lock:
        return _running


def benchmark_try_start() -> bool:
    global _running, _stop_requested
    with _state_lock:
        if _running:
            return False
        _running = True
        _stop_requested = False
        return True


def benchmark_request_stop() -> bool:
    global _stop_requested
    with _state_lock:
        if not _running:
            return False
        _stop_requested = True
        return True


def benchmark_stop_requested() -> bool:
    with _state_lock:
        return _stop_requested


def benchmark_finish() -> None:
    global _running, _stop_requested
    with _state_lock:
        _running = False
        _stop_requested = False

