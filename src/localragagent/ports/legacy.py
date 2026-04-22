"""Legacy-module bridge for phased migration into the `localragagent` package."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType

from .observability import log_port_event


@dataclass(frozen=True)
class LegacyImportErrorInfo:
    alias: str
    target: str
    error_type: str
    message: str


LEGACY_ALIASES: dict[str, str] = {
    # Root modules
    "agent_api": "agent_api",
    "gradio_interface": "gradio_interface",
    "messenger_simulator": "messenger_simulator",
    "schedule_ttl_cache": "schedule_ttl_cache",
    # Root packages
    "agent_logic_1": "agent_logic_1",
    "agent_logic_2": "agent_logic_2",
    "messengers_router": "messengers_router",
    "converters": "converters",
    "container_managenment": "container_managenment",
    "openclaw": "openclaw",
    "VOSK": "VOSK",
    "whisper": "whisper",
}


def import_legacy_alias(alias: str, *, package_name: str = "localragagent") -> ModuleType:
    """
    Imports a legacy module and binds it to `f\"{package_name}.{alias}\"`.

    :param alias: alias used inside `localragagent`
    :param package_name: package name to bind into (default: localragagent)
    :return: imported legacy module object
    :raises KeyError: alias is not registered
    :raises ImportError: target import failed
    """

    target = LEGACY_ALIASES[alias]
    alias_fqn = f"{package_name}.{alias}"
    existing = sys.modules.get(alias_fqn)
    if existing is not None:
        return existing

    try:
        module = importlib.import_module(target)
    except Exception as exc:
        log_port_event(
            "legacy_alias_import_failed",
            level=logging.ERROR,
            alias=alias,
            target=target,
            error_type=type(exc).__name__,
        )
        raise
    sys.modules[alias_fqn] = module
    log_port_event("legacy_alias_import_ok", alias=alias, target=target)
    return module


def install_legacy_aliases(
    *,
    package_name: str = "localragagent",
    aliases: Iterable[str] | None = None,
    strict: bool = False,
) -> list[LegacyImportErrorInfo]:
    """
    Installs legacy module aliases under `package_name`.

    :param package_name: package name to bind aliases under
    :param aliases: subset of aliases to install (default: all)
    :param strict: if True, re-raises import errors
    :return: collected non-fatal import errors
    """

    names = list(aliases) if aliases is not None else list(LEGACY_ALIASES)
    errors: list[LegacyImportErrorInfo] = []
    for alias in names:
        target = LEGACY_ALIASES.get(alias)
        if target is None:
            errors.append(
                LegacyImportErrorInfo(
                    alias=alias,
                    target="",
                    error_type="KeyError",
                    message=f"Unknown legacy alias: {alias}",
                )
            )
            if strict:
                raise KeyError(f"Unknown legacy alias: {alias}")
            continue
        try:
            import_legacy_alias(alias, package_name=package_name)
        except Exception as exc:  # pragma: no cover - path depends on runtime deps
            info = LegacyImportErrorInfo(
                alias=alias,
                target=target,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            errors.append(info)
            if strict:
                raise
    return errors
