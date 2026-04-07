"""In-memory хранилище состояния диалога по session_id.

Отвечает за историю сообщений, pending-слоты, merge entities и TTL-очистку.
Используется роутером как легковесная session memory для многошаговых флоу.
Ответственность модуля: управлять state/pending и concurrency-safe доступом к сессиям.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .mess_types import SessionState


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
        state.last_entities["_pending_label"] = label

    def clear_pending(self, state: SessionState) -> None:
        state.last_entities.pop("_pending", None)
        state.last_entities.pop("_pending_label", None)

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

        def _norm_name(value: Any) -> str:
            return " ".join(str(value or "").replace("ё", "е").lower().split())

        def _surname(value: Any) -> str:
            norm = _norm_name(value)
            if not norm:
                return ""
            return norm.split()[0]

        def _doctor_names_equivalent(prev: Any, curr: Any) -> bool:
            prev_norm = _norm_name(prev)
            curr_norm = _norm_name(curr)
            if not prev_norm or not curr_norm:
                return False
            if prev_norm == curr_norm:
                return True
            prev_surname = _surname(prev_norm)
            curr_surname = _surname(curr_norm)
            if not prev_surname or prev_surname != curr_surname:
                return False
            # "Дразнин" и "Дразнин Антон Владимирович" считаем одним врачом.
            prev_parts = len(prev_norm.split())
            curr_parts = len(curr_norm.split())
            return prev_parts == 1 or curr_parts == 1

        # remove None/empty-string
        cleaned: dict[str, Any] = {}
        for k, v in (new or {}).items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            cleaned[k] = v

        if not cleaned:
            if label:
                old["_last_label"] = label
            return

        action = str(cleaned.get("appointment_action") or "").strip().lower()
        if action in {"cancel", "reschedule"}:
            has_explicit_target = bool(
                cleaned.get("doctor_id")
                or cleaned.get("doctor_name")
            )
            # Если пользователь только запускает отмену/перенос без конкретного врача,
            # чистим прилипший контекст специальности/услуги из прошлой темы.
            if not has_explicit_target:
                for k in (
                    "specialty",
                    "service_name",
                    "test_name",
                    "appointment_selection_mode",
                    "appointment_branch_options",
                ):
                    old.pop(k, None)

        # object change rules
        doctor_changed = False
        if "doctor_id" in cleaned and cleaned.get("doctor_id") != old.get("doctor_id"):
            doctor_changed = True
        if "doctor_name" in cleaned:
            prev_name = old.get("doctor_name")
            curr_name = cleaned.get("doctor_name")
            if _doctor_names_equivalent(prev_name, curr_name):
                # Сохраняем более полную форму ФИО и не считаем это сменой врача.
                prev_norm = _norm_name(prev_name)
                curr_norm = _norm_name(curr_name)
                if len(prev_norm.split()) > len(curr_norm.split()):
                    cleaned["doctor_name"] = prev_name
            elif curr_name != prev_name:
                doctor_changed = True

        if doctor_changed:
            for k in (
                "date_from",
                "date_to",
                "time_from",
                "time_to",
                "date_hint",
                "appointment_windows",
                "appointment_branch_options",
                "appointment_flow_active",
                "appointment_confirm_pending",
                "appointment_confirmed",
                "branch_id",
                "branch_name",
            ):
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
