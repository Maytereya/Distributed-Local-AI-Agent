from __future__ import annotations

import logging
import shutil
from pathlib import Path

from agent_logic_2 import config as c

logger = logging.getLogger(__name__)

_LEGACY_DATA_DIR = (Path(__file__).resolve().parent / "apidata").resolve()
_DEFAULT_APP_DATA_ROOT = Path("/app_data")


def _app_data_root() -> Path:
    raw = str(getattr(c, "APP_DATA_DIR", "") or "").strip()
    root = Path(raw).expanduser() if raw else _DEFAULT_APP_DATA_ROOT
    return root.resolve()


def _shared_data_dir() -> Path:
    return (_app_data_root() / "nayka_api" / "apidata").resolve()


def _has_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def _copy_tree(src: Path, dst: Path) -> int:
    copied = 0
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    return copied


def resolve_cache_data_dir() -> Path:
    """Returns shared Nayka cache dir in APP_DATA_DIR with fallback to legacy path."""
    shared = _shared_data_dir()
    try:
        shared.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        _LEGACY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.error(
            "Cannot create shared cache dir %s (%s). Fallback to legacy dir %s",
            shared,
            exc,
            _LEGACY_DATA_DIR,
        )
        return _LEGACY_DATA_DIR

    if _has_files(shared) or not _has_files(_LEGACY_DATA_DIR):
        return shared

    try:
        copied = _copy_tree(_LEGACY_DATA_DIR, shared)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to migrate legacy cache from %s to %s: %s", _LEGACY_DATA_DIR, shared, exc)
        return shared

    if copied > 0:
        logger.info("Migrated %d cache file(s) from %s to %s", copied, _LEGACY_DATA_DIR, shared)
    return shared

