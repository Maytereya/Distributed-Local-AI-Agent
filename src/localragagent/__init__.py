"""Top-level package boundary for LocalRAGagent."""

from __future__ import annotations

from .ports.legacy import LEGACY_ALIASES, install_legacy_aliases

__version__ = "0.1.0"

__all__ = ["LEGACY_ALIASES", "install_legacy_aliases", "__version__"]
