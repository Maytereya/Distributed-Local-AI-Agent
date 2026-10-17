"""Отпечаток рантайма для debug-ответа (DATA-13): чем отвечает бот — сборка, модель, промпты, реестр тем, TZ, срезы.

Зачем: расхождение прода и репозитория по HTTP не диагностировалось — ловушки `CLAUDE.md` (модель из
`app_data/settings/main_model.txt` побеждает конфиг, промпт правят в одной локации из двух, ночью словарь синонимов
пуст, сборка мимо схемы) разбирали часами по SSH. Теперь запрос с `"debug": true` кладёт отпечаток в
`state_update.debug.runtime`.

Только имена, числа и хэши: без хостов, путей, ключей и текста промптов (репозиторий публичный, debug-ответ видит
любой, кто может позвать эндпоинт с ключом debug — SEC-2).

- `build` — SHA сборки из окружения (`MR_BUILD_SHA` / `GIT_SHA` / `SOURCE_COMMIT`) или из `.git` рядом с кодом (в
  образе `.git` нет — тогда None); `code` — хэш исходников `messengers_router/**/*.py`, сверяется с репо локально:
  `python -c "from messengers_router.runtime_fingerprint import code_digest; print(code_digest())"`;
- `world` — отпечаток мира из сервисного слоя (`services.core._world_fingerprint`: модель, имена срезов МИС, размер
  словаря синонимов) — тот же, что пишет телеметрия хода;
- `prompts` — на ключ: хэш действующего текста, откуда он (`host` — `app_data/prompts/mr_*`, `bundle` — пакет) и
  `differs` — копия хоста разошлась с пакетом (правило двух локаций);
- `topic_registry` — хэш файла и разбирается ли он (DATA-12: битый — бот работает на последнем исправном);
- `tz` — пояс процесса.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import yaml

from . import prompt_registry, topic_registry, turn_telemetry

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PACKAGE_DIR.parent
_BUILD_ENV = ("MR_BUILD_SHA", "GIT_SHA", "SOURCE_COMMIT")
_HASH_CHARS = 12
# debug-ходов в гейте и на стенде тысячи: отпечаток пересчитывается не чаще раза в _TTL_S (правка промпта видна через
# полминуты — для диагностики достаточно).
_TTL_S = 30.0
_CACHE: dict[str, Any] = {"at": 0.0, "value": None}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:_HASH_CHARS]


def _git_dirs(repo: Path) -> tuple[Path, Path] | None:
    """(папка HEAD, общая папка refs): в рабочем дереве `git worktree` `.git` — файл `gitdir: …`."""

    dot_git = repo / ".git"
    if dot_git.is_dir():
        return dot_git, dot_git
    try:
        pointer = dot_git.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not pointer.startswith("gitdir:"):
        return None
    git_dir = (repo / pointer.split(":", 1)[1].strip()).resolve()
    try:
        common = (git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        common = git_dir
    return git_dir, common


def _git_head_sha(repo: Path) -> str | None:
    """SHA из `.git` без вызова git: HEAD → ref → файл ref или packed-refs."""

    dirs = _git_dirs(repo)
    if dirs is None:
        return None
    git_dir, common = dirs
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head[:_HASH_CHARS] or None
    ref = head.split(":", 1)[1].strip()
    for base in (git_dir, common):
        try:
            return (base / ref).read_text(encoding="utf-8").strip()[:_HASH_CHARS] or None
        except OSError:
            continue
    try:
        for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha[:_HASH_CHARS]
    except OSError:
        pass
    return None


def build_sha() -> str | None:
    for name in _BUILD_ENV:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value[:_HASH_CHARS]
    return _git_head_sha(_REPO_DIR)


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    """Хэш исходников пакета: путь относительно пакета + содержимое, в порядке путей. Код меняется только с рестартом
    процесса — считаем один раз."""

    h = hashlib.sha256()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(_PACKAGE_DIR).as_posix().encode("utf-8") + b"\0")
        h.update(path.read_bytes() + b"\0")
    return h.hexdigest()[:_HASH_CHARS]


def _prompts() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bundle in sorted(prompt_registry._PROMPTS_DIR.glob("*.txt")):
        key = bundle.stem
        host = prompt_registry._host_path(key)
        bundle_bytes = bundle.read_bytes()
        try:
            host_bytes: bytes | None = host.read_bytes()
        except OSError:
            host_bytes = None
        effective = host_bytes if host_bytes is not None else bundle_bytes
        out[key] = {
            "sha": _digest(effective),
            "source": "host" if host_bytes is not None else "bundle",
            "differs": host_bytes is not None and host_bytes != bundle_bytes,
        }
    return out


def _topic_registry() -> dict[str, Any]:
    path = topic_registry._REGISTRY_PATH
    try:
        raw = path.read_bytes()
    except OSError:
        return {"sha": None, "parses": False}
    try:
        yaml.safe_load(raw.decode("utf-8"))
        parses = True
    except (yaml.YAMLError, UnicodeDecodeError):
        parses = False
    return {"sha": _digest(raw), "parses": parses}


def _world() -> dict[str, Any]:
    # Источник — сервисный слой (`services.core` регистрирует его при импорте); своего геттера у телеметрии нет, а сам
    # модуль телеметрии сейчас меняет параллельная работа (ход (д)) — читаем атрибут.
    provider = getattr(turn_telemetry, "_world_provider", None)
    if provider is None:
        return {}
    try:
        return dict(provider())
    except Exception as exc:  # отпечаток — справка, не повод ломать ответ
        return {"error": type(exc).__name__}


def _tz() -> dict[str, Any]:
    return {
        "env": os.environ.get("TZ") or None,
        "name": time.strftime("%Z"),
        "offset": time.strftime("%z"),
    }


def runtime_fingerprint(*, fresh: bool = False) -> dict[str, Any]:
    """Отпечаток для `debug.runtime` (копия; кэш на `_TTL_S`). Каждая часть считается отдельно: сбой одной не прячет
    остальные."""

    now = time.monotonic()
    cached = _CACHE["value"]
    if not fresh and cached is not None and now - float(_CACHE["at"]) < _TTL_S:
        return copy.deepcopy(cached)
    out: dict[str, Any] = {}
    parts = (
        ("build", build_sha),
        ("code", code_digest),
        ("world", _world),
        ("prompts", _prompts),
        ("topic_registry", _topic_registry),
        ("tz", _tz),
    )
    for name, part in parts:
        try:
            out[name] = part()
        except Exception as exc:
            out[name] = {"error": type(exc).__name__}
    _CACHE.update(at=now, value=out)
    return copy.deepcopy(out)
