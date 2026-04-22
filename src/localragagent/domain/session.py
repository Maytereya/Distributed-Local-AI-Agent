"""Domain entities for session scoping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionRef:
    """Minimal immutable session descriptor."""

    session_id: str
    source: str = "unknown"

