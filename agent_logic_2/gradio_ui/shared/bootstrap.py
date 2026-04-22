from __future__ import annotations

from agent_logic_2.persist import PROMPTS_DIR, SETTINGS_DIR, copy_defaults, ensure_dir


def bootstrap_files() -> None:
    # гарантируем папки в APP_DATA_DIR (или fallback)
    pdir = ensure_dir(PROMPTS_DIR, "prompts")
    sdir = ensure_dir(SETTINGS_DIR, "settings")

    # копируем дефолты из пакета, но НЕ перетираем существующие
    copy_defaults(("data", "prompts"), pdir, suffixes=(".txt",), overwrite=False)
    copy_defaults(("data", "settings"), sdir, suffixes=(".txt", ".json"), overwrite=False)

