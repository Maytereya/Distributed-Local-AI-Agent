from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from agent_logic_2 import config as c

logger = logging.getLogger(__name__)

PKG = "agent_logic_2"

DATA_DIR = Path(c.APP_DATA_DIR).expanduser().resolve()
PROMPTS_DIR = (DATA_DIR / "prompts").resolve()
SETTINGS_DIR = (DATA_DIR / "settings").resolve()


def ensure_dir(path: Path, fallback_name: str) -> Path:
    """Создать директорию. Если нельзя — fallback в ./app_data/<fallback_name>."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (OSError, PermissionError) as e:
        fb = (Path.cwd() / "app_data" / fallback_name).resolve()
        fb.mkdir(parents=True, exist_ok=True)
        logger.error("❌ Не удалось создать %s: %s. Fallback -> %s", path, e, fb)
        return fb


def copy_defaults(src_subdir: tuple[str, ...], dest_dir: Path, suffixes: tuple[str, ...], overwrite: bool = False) -> int:
    """
    Копирует файлы из пакета (agent_logic_2/<src_subdir>) в dest_dir.
    """
    dest_dir = ensure_dir(dest_dir, dest_dir.name)
    copied = 0
    src_root = resources.files(PKG).joinpath(*src_subdir)

    for entry in src_root.iterdir():
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if suffixes and entry.suffix.lower() not in suffixes:
            continue

        target = dest_dir / entry.name
        if target.exists() and not overwrite:
            continue

        with entry.open("r", encoding="utf-8") as r:
            target.write_text(r.read(), encoding="utf-8")

        copied += 1

    if copied:
        logger.info("✅ Defaults copied=%d from %s -> %s", copied, "/".join(src_subdir), dest_dir)
    return copied


def parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "on")