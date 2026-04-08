"""Key builders for runtime storage layers."""

from __future__ import annotations


def free_talk_turns_key(prefix: str, session_id: str) -> str:
    return f"{prefix}:session:{session_id}:turns"


def free_talk_summary_key(prefix: str, session_id: str) -> str:
    return f"{prefix}:session:{session_id}:summary"


def free_talk_meta_key(prefix: str, session_id: str) -> str:
    return f"{prefix}:session:{session_id}:meta"

