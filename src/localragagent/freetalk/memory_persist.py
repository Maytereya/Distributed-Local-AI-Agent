"""Persistent FreeTalk memory snapshots."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PersistSettings:
    jsonl_path: Path


class PersistentSummaryStore:
    def __init__(self, settings: PersistSettings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()

    async def append_snapshot(
        self,
        *,
        session_id: str,
        summary: str,
        key_facts: list[str] | None = None,
        open_loops: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "session_id": str(session_id or "").strip(),
            "ts": int(time.time()),
            "summary": str(summary or "").strip(),
            "key_facts": list(key_facts or []),
            "open_loops": list(open_loops or []),
            "extra": dict(extra or {}),
        }
        payload = json.dumps(record, ensure_ascii=False) + "\n"
        path = self._settings.jsonl_path

        async with self._lock:
            await asyncio.to_thread(self._append_line, path, payload)

    @staticmethod
    def _append_line(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload)

