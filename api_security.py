"""Authentication for the messenger service; credentials never enter responses."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def messenger_api_key() -> str:
    path = os.environ.get("MESSENGER_API_KEY_FILE", "")
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""  # A missing configured credential must fail closed.
    key = os.environ.get("MESSENGER_API_KEY", "").strip()
    if key:
        return key
    import agent_logic_2.config as config

    return os.environ.get("AGENT_API_KEY", "").strip() or config.AGENT_API_KEY.strip()


def verify_secret(provided: str | None, expected: str) -> None:
    if not expected:
        raise HTTPException(status_code=503, detail="Service authentication unavailable")
    if not provided or not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key")


def verify_messenger_api_key(api_key: str | None = Security(_header)) -> None:
    verify_secret(api_key, messenger_api_key())


def messenger_debug_allowed(requested: bool) -> bool:
    return bool(requested and os.environ.get("MESSENGER_ALLOW_DEBUG", "0") == "1")
