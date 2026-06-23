"""Port for shared specialty parsing used by FreeTalk."""

from __future__ import annotations

import importlib
from typing import Any

from .legacy import import_legacy_alias

import_legacy_alias("messengers_router")


def extract_specialty_from_text(text: str) -> str:
    import_legacy_alias("messengers_router")
    parser_mod: Any = importlib.import_module("localragagent.messengers_router.specialty_parser")
    parser = getattr(parser_mod, "extract_specialty_from_text", None)
    if not callable(parser):
        return ""
    return str(parser(text) or "").strip()
