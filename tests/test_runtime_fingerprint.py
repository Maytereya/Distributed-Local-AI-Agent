"""Отпечаток рантайма в debug-ответе (DATA-13, `messengers_router/runtime_fingerprint.py`).

Расхождение прода и репозитория по HTTP не диагностировалось: модель из `main_model.txt`, промпт, поправленный в одной
локации из двух, битый реестр тем, пустой ночной словарь синонимов разбирали часами. Инварианты:
- ход с `"debug": true` через эндпоинт несёт `debug.runtime`: сборку, хэш кода, мир (модель, срезы, словарь), хэш и
  источник каждого промпта, хэш и разбор реестра тем, пояс;
- расхождение копии хоста с пакетным промптом и битый реестр тем видны в отпечатке;
- в отпечатке нет хостов, адресов, путей и текста промптов (репозиторий публичный, debug видит вызывающий).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from messengers_router import prompt_registry, runtime_fingerprint, topic_registry

_BUNDLE_KEYS = sorted(path.stem for path in prompt_registry._PROMPTS_DIR.glob("*.txt"))
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _turn(text: str, session_id: str, *, debug: bool = True) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=session_id, text=text, debug=debug)
        return json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    monkeypatch.setitem(runtime_fingerprint._CACHE, "value", None)


def _runtime(body: dict[str, Any]) -> dict[str, Any]:
    return ((body.get("state_update") or {}).get("debug") or {}).get("runtime") or {}


def test_debug_turn_carries_the_runtime_fingerprint():
    runtime = _runtime(_turn("сколько стоит оак", "fp-debug"))
    assert set(runtime) == {"build", "code", "world", "prompts", "topic_registry", "tz"}, runtime
    assert re.fullmatch(r"[0-9a-f]{12}", runtime["code"]), runtime["code"]
    assert {"model", "price", "service_info", "doctors", "synonyms"} <= set(runtime["world"]), runtime["world"]
    assert sorted(runtime["prompts"]) == _BUNDLE_KEYS
    for key, item in runtime["prompts"].items():
        assert re.fullmatch(r"[0-9a-f]{12}", item["sha"]), (key, item)
        assert item["source"] in {"host", "bundle"} and item["differs"] is False, (key, item)
    assert runtime["topic_registry"]["parses"] is True and runtime["topic_registry"]["sha"], runtime["topic_registry"]
    assert {"env", "name", "offset"} == set(runtime["tz"]), runtime["tz"]


def test_turn_without_debug_has_no_fingerprint():
    body = _turn("сколько стоит оак", "fp-no-debug", debug=False)
    assert "runtime" not in json.dumps(body, ensure_ascii=False)


def test_fingerprint_has_no_hosts_paths_or_prompt_text():
    runtime = runtime_fingerprint.runtime_fingerprint(fresh=True)
    dumped = json.dumps(runtime, ensure_ascii=False)
    assert "://" not in dumped and not _IP.search(dumped), dumped
    without_world = json.dumps({k: v for k, v in runtime.items() if k != "world"}, ensure_ascii=False)
    assert "/" not in without_world, without_world  # путей нет; в имени модели «/» бывает (hf.co/…)
    assert str(Path.home()) not in dumped
    longest_value = max((len(str(v)) for v in _leaves(runtime)), default=0)
    assert longest_value <= 80, longest_value  # текст промпта или ответа сюда не попадает


def _leaves(node: Any):
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _leaves(value)
    else:
        yield node


def test_diverged_host_copy_is_visible(tmp_path, monkeypatch):
    key = _BUNDLE_KEYS[0]
    host_dir = tmp_path / "prompts"
    host_dir.mkdir()
    for k in _BUNDLE_KEYS:
        (host_dir / f"mr_{k}.txt").write_bytes((prompt_registry._PROMPTS_DIR / f"{k}.txt").read_bytes())
    (host_dir / f"mr_{key}.txt").write_text("правка только на хосте\n", encoding="utf-8")
    (host_dir / f"mr_{_BUNDLE_KEYS[1]}.txt").unlink()
    monkeypatch.setattr(prompt_registry, "_RUNTIME_PROMPTS_DIR", host_dir)

    prompts = runtime_fingerprint.runtime_fingerprint(fresh=True)["prompts"]

    assert prompts[key]["differs"] is True and prompts[key]["source"] == "host", prompts[key]
    missing = prompts[_BUNDLE_KEYS[1]]
    assert missing["source"] == "bundle" and missing["differs"] is False, missing
    assert all(not item["differs"] for k, item in prompts.items() if k != key)


def test_broken_topic_registry_is_visible(tmp_path, monkeypatch):
    path = tmp_path / "topic_registry.yaml"
    path.write_text('topics:\n  - topic_id: "x\n', encoding="utf-8")
    monkeypatch.setattr(topic_registry, "_REGISTRY_PATH", path)
    assert runtime_fingerprint.runtime_fingerprint(fresh=True)["topic_registry"]["parses"] is False


def test_build_sha_from_environment(monkeypatch):
    monkeypatch.setenv("MR_BUILD_SHA", "0123456789abcdef0123")
    assert runtime_fingerprint.build_sha() == "0123456789ab"


def test_code_digest_follows_the_source(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(runtime_fingerprint, "_PACKAGE_DIR", tmp_path)
    runtime_fingerprint.code_digest.cache_clear()
    try:
        before = runtime_fingerprint.code_digest()
        runtime_fingerprint.code_digest.cache_clear()
        (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
        assert runtime_fingerprint.code_digest() != before
    finally:
        runtime_fingerprint.code_digest.cache_clear()
