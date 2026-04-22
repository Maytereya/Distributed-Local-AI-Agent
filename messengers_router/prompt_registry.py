"""Единая загрузка prompt-шаблонов для `messengers_router`.

Принципы (синхронизированы с общим подходом проекта):
1) Источник runtime-правок — `APP_DATA_DIR/prompts` (через `agent_logic_2.config/persist`).
2) Дефолты поставляются из дистрибутива: `messengers_router/prompts/*.txt`.
3) Версионирование prompt-файлов и переключение через env не используются.
4) Кэш чтения не используется намеренно: изменения файлов подхватываются без рестарта процесса.

Порядок загрузки для ключа `<key>`:
- `/app_data/prompts/mr_<key>.txt` (host override, если есть),
- `messengers_router/prompts/<key>.txt` (bundle default).

Для мягкой миграции поддерживается legacy-override имя
`/app_data/prompts/mr_<key>_v2.txt`: при обнаружении содержимое переносится
в новый файл `mr_<key>.txt`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_logic_2.persist import PROMPTS_DIR as APP_PROMPTS_DIR, ensure_dir

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_RUNTIME_PROMPTS_DIR = ensure_dir(APP_PROMPTS_DIR, "prompts")


def _safe_key(key: str) -> str:
    cleaned = str(key or "").strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError(f"Invalid prompt key: {key!r}")
    return cleaned


def _host_path(key: str) -> Path:
    return _RUNTIME_PROMPTS_DIR / f"mr_{key}.txt"


def _host_legacy_v2_path(key: str) -> Path:
    return _RUNTIME_PROMPTS_DIR / f"mr_{key}_v2.txt"


def _bundle_path(key: str) -> Path:
    return _PROMPTS_DIR / f"{key}.txt"


def _try_migrate_legacy_override(key: str) -> Path | None:
    current = _host_path(key)
    if current.exists():
        return current

    legacy = _host_legacy_v2_path(key)
    if legacy.exists():
        try:
            current.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("migrated legacy prompt override %s -> %s", legacy.name, current.name)
            return current
        except Exception:
            logger.exception("failed to migrate legacy prompt override: %s", legacy)
            return legacy

    return None


def load_prompt_text(key: str) -> str:
    key = _safe_key(key)

    host = _host_path(key)
    if host.exists():
        return host.read_text(encoding="utf-8")

    migrated = _try_migrate_legacy_override(key)
    if migrated and migrated.exists():
        return migrated.read_text(encoding="utf-8")

    bundle = _bundle_path(key)
    if not bundle.exists():
        raise FileNotFoundError(f"Prompt not found for key={key}")

    # Пробуем посеять дефолт на host для прозрачного редактирования/персистентности.
    # Если запись недоступна (права/FS), продолжаем работу на bundle-копии.
    try:
        host.write_text(bundle.read_text(encoding="utf-8"), encoding="utf-8")
        return host.read_text(encoding="utf-8")
    except Exception:
        logger.warning("failed to seed host prompt override from bundle: %s", host)
        return bundle.read_text(encoding="utf-8")
