"""Explicit adapter contracts for FreeTalk tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AdapterToolRequest:
    tool_name: str
    user_message: str
    ft_entities: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedToolCall:
    tool_name: str
    backend_query: str
    backend_entities: dict[str, Any] = field(default_factory=dict)
    ft_entities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterToolResult:
    tool_name: str
    ft_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
