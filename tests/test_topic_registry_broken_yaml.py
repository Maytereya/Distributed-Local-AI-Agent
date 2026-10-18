"""DATA-12: битый `topic_registry.yaml` не переводит каждый ход на оператора (`messengers_router/topic_registry.py`).

До фикса `yaml.safe_load` без обработки: ошибка разбора долетала до общего `except` роутера — `service_error`, перевод
на оператора и сброс состояния на КАЖДОМ ходу (кэш по mtime битую версию не запоминал). Риск — ручная правка на
сервере или конфликт `git pull`. Решение владельца 18.10: без выключателя — последний исправный реестр + `log.error`.

Инварианты класса (любая порча файла: незакрытая кавычка, маркеры конфликта git, табуляция, не-UTF-8):
- ход пациента идёт так же, как при исправном реестре, если исправный уже был прочитан (темы-override сохраняются);
- холодный старт на битом файле — без тем-override, но без перевода на оператора и без ошибки маршрута;
- `log.error` — один раз на версию файла, не на каждый ход;
- правки редактора (`upsert_topic`, `delete_topic`) на битом файле падают и файл не переписывают — ручную правку
  не затирают устаревшими данными.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalbench import world as ev_world
from messengers_router import topic_registry
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

_HEALTHY = Path(topic_registry.__file__).resolve().parent / "data" / "topic_registry.yaml"

BROKEN: dict[str, bytes] = {
    "unclosed_quote": b'topics:\n  - topic_id: "price\n    label: PRICE\n',
    "git_conflict": b"<<<<<<< HEAD\ntopics: []\n=======\ntopics: [1]\n>>>>>>> origin/release\n",
    "tab_indent": b"topics:\n\t- topic_id: price\n",
    "not_utf8": b"topics:\n  - topic_id: \xff\xfe\n",
}

# Второй ход попадает в тему-override (`service_home_visit`): последний исправный реестр должен её сохранить.
DIALOG = ("сколько стоит оак", "вызвать врача на дом")


@pytest.fixture
def registry_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "topic_registry.yaml"
    path.write_bytes(_HEALTHY.read_bytes())
    monkeypatch.setattr(topic_registry, "_REGISTRY_PATH", path)
    monkeypatch.setattr(topic_registry, "_CACHE_DATA", None)
    monkeypatch.setattr(topic_registry, "_CACHE_MTIME", -1.0)
    return path


def _break(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 5))  # новая версия файла и при грубом разрешении mtime


def _record(body: dict[str, Any]) -> dict[str, Any]:
    dbg = (body.get("state_update") or {}).get("debug") or {}
    decision = dbg.get("decision") or {}
    return {
        "text": body.get("text", ""),
        "handoff": bool(body.get("handoff")),
        "label": decision.get("label"),
        "route_error": dbg.get("route_error"),
        "topic": topic_registry.extract_topic_id_from_flags(decision.get("flags")),
    }


def _dialog(session_id: str, *, break_before: int | None = None, path: Path | None = None, payload: bytes = b"") -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for i, text in enumerate(DIALOG):
            if break_before == i:
                _break(path, payload)
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True)
            resp = await messenger_generate_once(req, memory=memory, services=services)
            out.append(_record(json.loads(resp.body)))
        return out

    return asyncio.run(run())


def _reset_cache(monkeypatch) -> None:
    monkeypatch.setattr(topic_registry, "_CACHE_DATA", None)
    monkeypatch.setattr(topic_registry, "_CACHE_MTIME", -1.0)


@pytest.mark.parametrize("kind", sorted(BROKEN))
def test_broken_registry_keeps_the_last_good_one(kind, registry_file, mini_prod_world, monkeypatch):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        healthy = _dialog(f"data12-healthy-{kind}")
        _reset_cache(monkeypatch)
        broken = _dialog(f"data12-warm-{kind}", break_before=1, path=registry_file, payload=BROKEN[kind])
    assert healthy[1]["topic"] == "service_home_visit", healthy  # ход 2 действительно зависит от реестра
    assert broken == healthy


@pytest.mark.parametrize("kind", sorted(BROKEN))
def test_cold_start_on_broken_registry_is_not_a_handoff(kind, registry_file, mini_prod_world, monkeypatch):  # noqa: F811
    _break(registry_file, BROKEN[kind])
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = _dialog(f"data12-cold-{kind}")
    first = record[0]
    assert first["route_error"] is None, first
    assert not first["handoff"], first
    assert first["label"] == "PRICE", first
    assert all(turn["topic"] is None for turn in record), record  # тем-override нет, а не мусор из битого файла


def test_error_is_logged_once_per_file_version(registry_file, caplog):
    topic_registry.load_registry()
    _break(registry_file, BROKEN["unclosed_quote"])
    with caplog.at_level(logging.ERROR, logger=topic_registry.__name__):
        for _ in range(5):
            assert topic_registry.match_topic("вызвать врача на дом") is not None
        _break(registry_file, BROKEN["tab_indent"])
        for _ in range(3):
            topic_registry.match_topic("вызвать врача на дом")
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 2, [r.getMessage() for r in errors]


def test_repaired_file_is_picked_up(registry_file):
    topic_registry.load_registry()
    _break(registry_file, BROKEN["git_conflict"])
    assert topic_registry.match_topic("вызвать врача на дом") is not None
    repaired = yaml.safe_load(_HEALTHY.read_text(encoding="utf-8"))
    repaired["topics"] = [t for t in repaired["topics"] if t.get("topic_id") != "service_home_visit"]
    _break(registry_file, yaml.safe_dump(repaired, allow_unicode=True).encode("utf-8"))
    match = topic_registry.match_topic("вызвать врача на дом")
    assert match is None or match.topic_id != "service_home_visit"


@pytest.mark.parametrize(
    "edit",
    [
        lambda: topic_registry.upsert_topic({"topic_id": "x", "label": "OTHER"}),
        lambda: topic_registry.create_topic("x"),
        lambda: topic_registry.delete_topic("price"),
        lambda: topic_registry.set_topic_enabled("price", False),
    ],
    ids=["upsert", "create", "delete", "set_enabled"],
)
def test_editor_does_not_overwrite_a_broken_file(edit, registry_file):
    topic_registry.load_registry()
    _break(registry_file, BROKEN["unclosed_quote"])
    with pytest.raises(yaml.YAMLError):
        edit()
    assert registry_file.read_bytes() == BROKEN["unclosed_quote"]
