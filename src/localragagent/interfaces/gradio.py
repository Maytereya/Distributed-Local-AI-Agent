"""UI interface adapter for Gradio app."""

from __future__ import annotations

from ..ports.legacy import import_legacy_alias

_legacy_gradio = import_legacy_alias("gradio_interface")


def main() -> None:
    """Starts the legacy Gradio interface."""
    _legacy_gradio.main()
