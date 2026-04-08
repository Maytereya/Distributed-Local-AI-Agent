"""Messenger orchestration wrappers."""

from __future__ import annotations

import importlib
from typing import Any

from localragagent.ports.legacy import import_legacy_alias

import_legacy_alias("messengers_router")


def get_endpoint_router() -> Any:
    """Returns messenger FastAPI router."""
    endpoint_mod = importlib.import_module("localragagent.messengers_router.endpoint")
    return endpoint_mod.router


def get_patient_routing_stream() -> Any:
    """Returns streaming route orchestrator."""
    router_mod = importlib.import_module("localragagent.messengers_router.router")
    return router_mod.patient_routing_stream
