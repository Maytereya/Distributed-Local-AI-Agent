"""Сторож видеокарт в проверке живости (решение владельца 08.10, повестка №10).

Урок 28–31.08: после работ на сервере ollama молча ушла на процессор — ответы по 60 с, а
синтетический зонд оставался «здоров». Инвариант: модель не загружена или не целиком на GPU →
«болен».
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_bot_health", Path(__file__).resolve().parents[1] / "scripts" / "check_bot_health.py"
)
health = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(health)  # type: ignore[union-attr]

_GB = 1024**3


@pytest.mark.parametrize(
    ("payload", "healthy"),
    [
        ({"models": [{"name": "mistral", "size": 56 * _GB, "size_vram": 56 * _GB}]}, True),
        ({"models": [{"name": "mistral", "size": 56 * _GB, "size_vram": 24 * _GB}]}, False),
        ({"models": [{"name": "mistral", "size": 56 * _GB, "size_vram": 0}]}, False),
        ({"models": []}, False),
    ],
    ids=["all_on_gpu", "partly_on_cpu", "all_on_cpu", "nothing_loaded"],
)
def test_gpu_verdict(payload, healthy):
    ok, detail = health.gpu_verdict(payload)

    assert ok is healthy, detail
