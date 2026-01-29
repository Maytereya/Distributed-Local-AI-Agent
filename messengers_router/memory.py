from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from mess_types import SessionState


@dataclass
class _MemRecord:
    state: SessionState
    updated_at: float


class MemoryStore:
    """
    In-memory session store (MVP).

    Улучшения:
    - TTL GC
    - per-session asyncio.Lock (защита от гонок)
    - pending (многошаговые уточнения) храним в state.last_entities["_pending"]
    - merge_entities() с правилами сброса зависимых слотов
    """

    def __init__(self, ttl_seconds: int = 3600, pending_ttl_seconds: int = 900) -> None:
        self._ttl = ttl_seconds
        self._pending_ttl = pending_ttl_seconds

        self._db: dict[str, _MemRecord] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ---------- locks ----------

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    # ---------- basic CRUD (sync) ----------

    def get(self, session_id: str) -> SessionState:
        self._gc()
        rec = self._db.get(session_id)
        if rec:
            rec.updated_at = time.time()
            return rec.state

        st = SessionState(session_id=session_id)
        self._db[session_id] = _MemRecord(state=st, updated_at=time.time())
        return st

    def set(self, state: SessionState) -> None:
        self._db[state.session_id] = _MemRecord(state=state, updated_at=time.time())

    # ---------- CRUD (async, safe for messengers) ----------

    async def aget(self, session_id: str) -> SessionState:
        lock = self._get_lock(session_id)
        await lock.acquire()
        try:
            return self.get(session_id)
        finally:
            # lock удерживаем до aset/arelease
            pass

    async def aset(self, state: SessionState) -> None:
        try:
            self.set(state)
        finally:
            lock = self._get_lock(state.session_id)
            if lock.locked():
                lock.release()

    async def arelease(self, session_id: str) -> None:
        lock = self._get_lock(session_id)
        if lock.locked():
            lock.release()

    # ---------- history ----------

    def append_turn(self, state: SessionState, role: str, text: str, limit: int = 20) -> None:
        state.history.append({"role": role, "text": text})
        if len(state.history) > limit:
            state.history = state.history[-limit:]

    # ---------- pending ----------

    def set_pending(self, state: SessionState, label: str, missing_slots: list[str]) -> None:
        state.last_entities["_pending"] = {
            "label": label,
            "missing": missing_slots,
            "ts": time.time(),
            "ttl": self._pending_ttl,
        }

    def clear_pending(self, state: SessionState) -> None:
        state.last_entities.pop("_pending", None)

    def get_pending(self, state: SessionState) -> dict[str, Any] | None:
        p = state.last_entities.get("_pending")
        if not isinstance(p, dict):
            return None
        ts = float(p.get("ts", 0.0) or 0.0)
        ttl = float(p.get("ttl", self._pending_ttl) or self._pending_ttl)
        if time.time() - ts > ttl:
            self.clear_pending(state)
            return None
        return p

    # ---------- entities merge with rules ----------

    def merge_entities(self, state: SessionState, new: dict[str, Any], label: str | None = None) -> None:
        old = state.last_entities

        # remove None/empty-string
        cleaned: dict[str, Any] = {}
        for k, v in (new or {}).items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            cleaned[k] = v

        if not cleaned:
            return

        # object change rules
        doctor_changed = False
        if "doctor_id" in cleaned and cleaned.get("doctor_id") != old.get("doctor_id"):
            doctor_changed = True
        if "doctor_name" in cleaned and cleaned.get("doctor_name") != old.get("doctor_name"):
            doctor_changed = True

        if doctor_changed:
            for k in ("date_from", "date_to", "time_from", "time_to", "date_hint"):
                old.pop(k, None)

        test_changed = False
        if "test_name" in cleaned and cleaned.get("test_name") != old.get("test_name"):
            test_changed = True
        if test_changed:
            old.pop("order_id", None)

        # merge query_terms list
        if "query_terms" in cleaned and isinstance(cleaned["query_terms"], list):
            prev = old.get("query_terms")
            merged: list[str] = []
            if isinstance(prev, list):
                merged.extend([x for x in prev if isinstance(x, str)])
            merged.extend([x for x in cleaned["query_terms"] if isinstance(x, str)])
            seen = set()
            uniq: list[str] = []
            for x in merged:
                key = x.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(x.strip())
            cleaned["query_terms"] = uniq[:20]

        old.update(cleaned)
        if label:
            old["_last_label"] = label

    # ---------- housekeeping ----------

    def _gc(self) -> None:
        now = time.time()
        dead = [sid for sid, rec in self._db.items() if now - rec.updated_at > self._ttl]
        for sid in dead:
            del self._db[sid]
            lock = self._locks.get(sid)
            if lock is not None and (not lock.locked()):
                self._locks.pop(sid, None)