"""Правило двух локаций промптов (`CLAUDE.md`, DATA-13): прод читает `app_data/prompts/mr_<ключ>.txt`, пакет — дефолт.

Копия хоста побеждает (`prompt_registry.load_prompt_text`): правка только пакетного промпта молча оставляет прод на
старой версии, а снятый из пакета промпт хост читал бы вечно. До 18.10 равенство проверяли точечные тесты примерно на
9 ключей из 24; новый промпт без такого теста расходился незаметно. Инвариант — на все ключи сразу, в обе стороны.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _ROOT / "messengers_router" / "prompts"
_HOST = _ROOT / "app_data" / "prompts"
_HOST_PREFIX = "mr_"

_BUNDLE_KEYS = sorted(path.stem for path in _BUNDLE.glob("*.txt"))
_HOST_KEYS = sorted(path.stem[len(_HOST_PREFIX) :] for path in _HOST.glob(f"{_HOST_PREFIX}*.txt"))


def test_there_are_prompts_to_check():
    assert len(_BUNDLE_KEYS) >= 20, _BUNDLE_KEYS  # сторож: пустой glob не должен давать зелёный тест


@pytest.mark.parametrize("key", _BUNDLE_KEYS)
def test_every_bundle_prompt_has_an_identical_host_copy(key):
    host = _HOST / f"{_HOST_PREFIX}{key}.txt"
    assert host.exists(), f"нет копии {host.name}: прод возьмёт пакетную по умолчанию, но правка хоста её не увидит"
    assert host.read_bytes() == (_BUNDLE / f"{key}.txt").read_bytes(), f"копии промпта {key} разошлись"


def test_every_host_copy_has_a_bundle_prompt():
    orphans = sorted(set(_HOST_KEYS) - set(_BUNDLE_KEYS))
    assert not orphans, f"копии хоста без пакетного промпта (прод читал бы снятый промпт): {orphans}"
