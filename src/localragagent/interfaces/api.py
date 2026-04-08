"""HTTP interface adapter for FastAPI app."""

from __future__ import annotations

from localragagent.ports.legacy import import_legacy_alias

_legacy_api = import_legacy_alias("agent_api")
app = _legacy_api.app


def get_app():
    """Returns the legacy FastAPI application instance."""
    return app

