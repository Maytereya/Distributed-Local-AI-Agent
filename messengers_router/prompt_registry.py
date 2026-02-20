"""Реестр prompt-шаблонов с версионированием и fallback.

Ответственность модуля:
1) Определить активную версию prompt-слоя (через env).
2) Загружать prompt из versioned каталога с безопасным fallback на legacy.
3) Централизовать доступ к шаблонам, чтобы убрать прямые file-read в модулях.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def active_prompt_version() -> str:
    raw = str(os.getenv("MR_PROMPT_VERSION", "v2")).strip().lower()
    if raw in {"v1", "v2"}:
        return raw
    return "v2"


def _candidate_paths(key: str, version: str) -> list[Path]:
    # Новые файлы храним в prompts/versions/<version>/...
    # Старые — в корне prompts/.
    v2_name = f"{key}_{version}.txt"
    return [
        _PROMPTS_DIR / "versions" / version / v2_name,
        _PROMPTS_DIR / f"{key}.txt",
    ]


@lru_cache(maxsize=64)
def load_prompt_text(key: str, version: str | None = None) -> str:
    ver = (version or active_prompt_version()).strip().lower()
    for path in _candidate_paths(key, ver):
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found for key={key}, version={ver}")
