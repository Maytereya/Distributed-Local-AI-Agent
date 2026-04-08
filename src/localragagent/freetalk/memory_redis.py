"""Redis-backed session memory with in-process fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from localragagent.infrastructure.memory_keys import (
    free_talk_meta_key,
    free_talk_summary_key,
    free_talk_turns_key,
)

from .contracts import SessionContext
from .observability import log_event

try:  # pragma: no cover - depends on optional runtime dependency
    from redis import asyncio as redis_asyncio
except Exception:  # pragma: no cover - fallback path is tested by usage
    redis_asyncio = None


@dataclass(slots=True)
class RedisSettings:
    url: str
    prefix: str
    ttl_sec: int


class RedisMemoryStore:
    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._lock = asyncio.Lock()

        self._fallback_turns: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._fallback_summary: dict[str, str] = {}
        self._fallback_meta: dict[str, dict[str, str]] = defaultdict(dict)
        self._redis_unavailable_logged = False

    async def _redis(self) -> Any | None:
        if redis_asyncio is None:
            if not self._redis_unavailable_logged:
                log_event("redis_module_unavailable", level=logging.WARNING)
                self._redis_unavailable_logged = True
            return None
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            try:
                client = redis_asyncio.from_url(
                    self._settings.url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await client.ping()
                self._client = client
                if self._redis_unavailable_logged:
                    log_event("redis_connected", redis_url=self._settings.url)
                self._redis_unavailable_logged = False
            except Exception as exc:
                if not self._redis_unavailable_logged:
                    log_event(
                        "redis_connect_failed",
                        level=logging.WARNING,
                        redis_url=self._settings.url,
                        error_type=type(exc).__name__,
                    )
                    self._redis_unavailable_logged = True
                self._client = None
            return self._client

    async def load_context(self, session_id: str, *, history_tail_turns: int) -> SessionContext:
        sid = str(session_id or "").strip()
        if not sid:
            return SessionContext(session_id="")
        client = await self._redis()
        if client is None:
            turns = list(self._fallback_turns.get(sid, []))[-history_tail_turns:]
            summary = str(self._fallback_summary.get(sid, "") or "")
            return SessionContext(session_id=sid, summary=summary, turns=turns)

        turns_key = free_talk_turns_key(self._settings.prefix, sid)
        summary_key = free_talk_summary_key(self._settings.prefix, sid)
        try:
            raw_turns, summary = await asyncio.gather(
                client.lrange(turns_key, -history_tail_turns, -1),
                client.get(summary_key),
            )
        except Exception as exc:
            log_event(
                "redis_load_context_failed",
                level=logging.WARNING,
                session_id=sid,
                error_type=type(exc).__name__,
            )
            turns = list(self._fallback_turns.get(sid, []))[-history_tail_turns:]
            summary = str(self._fallback_summary.get(sid, "") or "")
            return SessionContext(session_id=sid, summary=summary, turns=turns)

        turns: list[dict[str, Any]] = []
        for raw in raw_turns or []:
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                turns.append(item)

        return SessionContext(
            session_id=sid,
            summary=str(summary or "").strip(),
            turns=turns,
        )

    async def append_exchange(self, session_id: str, *, user_text: str, assistant_text: str, source: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        now = int(time.time())
        turns = [
            {"role": "user", "content": str(user_text or ""), "ts": now},
            {"role": "assistant", "content": str(assistant_text or ""), "ts": now, "source": str(source or "")},
        ]
        client = await self._redis()
        if client is None:
            self._fallback_turns[sid].extend(turns)
            return

        turns_key = free_talk_turns_key(self._settings.prefix, sid)
        summary_key = free_talk_summary_key(self._settings.prefix, sid)
        meta_key = free_talk_meta_key(self._settings.prefix, sid)
        encoded = [json.dumps(item, ensure_ascii=False) for item in turns]
        try:
            await client.rpush(turns_key, *encoded)
            await client.expire(turns_key, self._settings.ttl_sec)
            await client.expire(summary_key, self._settings.ttl_sec)
            await client.expire(meta_key, self._settings.ttl_sec)
        except Exception as exc:
            log_event(
                "redis_append_exchange_failed",
                level=logging.WARNING,
                session_id=sid,
                error_type=type(exc).__name__,
            )
            self._fallback_turns[sid].extend(turns)

    async def get_turn_count(self, session_id: str) -> int:
        sid = str(session_id or "").strip()
        if not sid:
            return 0
        client = await self._redis()
        if client is None:
            return len(self._fallback_turns.get(sid, []))
        turns_key = free_talk_turns_key(self._settings.prefix, sid)
        try:
            return int(await client.llen(turns_key) or 0)
        except Exception as exc:
            log_event(
                "redis_turn_count_failed",
                level=logging.WARNING,
                session_id=sid,
                error_type=type(exc).__name__,
            )
            return len(self._fallback_turns.get(sid, []))

    async def save_summary(self, session_id: str, summary: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        summary_text = str(summary or "").strip()
        client = await self._redis()
        if client is None:
            self._fallback_summary[sid] = summary_text
            return
        summary_key = free_talk_summary_key(self._settings.prefix, sid)
        try:
            await client.set(summary_key, summary_text, ex=self._settings.ttl_sec)
        except Exception as exc:
            log_event(
                "redis_save_summary_failed",
                level=logging.WARNING,
                session_id=sid,
                error_type=type(exc).__name__,
            )
            self._fallback_summary[sid] = summary_text

    async def get_meta_int(self, session_id: str, key: str, default: int = 0) -> int:
        sid = str(session_id or "").strip()
        if not sid:
            return default
        k = str(key or "").strip()
        if not k:
            return default
        client = await self._redis()
        if client is None:
            try:
                return int(self._fallback_meta.get(sid, {}).get(k, default))
            except Exception:
                return default
        meta_key = free_talk_meta_key(self._settings.prefix, sid)
        try:
            value = await client.hget(meta_key, k)
            return int(value) if value is not None else default
        except Exception as exc:
            log_event(
                "redis_get_meta_int_failed",
                level=logging.WARNING,
                session_id=sid,
                key=k,
                error_type=type(exc).__name__,
            )
            return default

    async def set_meta_int(self, session_id: str, key: str, value: int) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        k = str(key or "").strip()
        if not k:
            return
        v = int(value)
        client = await self._redis()
        if client is None:
            self._fallback_meta[sid][k] = str(v)
            return
        meta_key = free_talk_meta_key(self._settings.prefix, sid)
        try:
            await client.hset(meta_key, mapping={k: str(v)})
            await client.expire(meta_key, self._settings.ttl_sec)
        except Exception as exc:
            log_event(
                "redis_set_meta_int_failed",
                level=logging.WARNING,
                session_id=sid,
                key=k,
                error_type=type(exc).__name__,
            )
            self._fallback_meta[sid][k] = str(v)

    async def get_meta_str(self, session_id: str, key: str, default: str = "") -> str:
        sid = str(session_id or "").strip()
        if not sid:
            return str(default or "")
        k = str(key or "").strip()
        if not k:
            return str(default or "")
        client = await self._redis()
        if client is None:
            return str(self._fallback_meta.get(sid, {}).get(k, default) or "")
        meta_key = free_talk_meta_key(self._settings.prefix, sid)
        try:
            value = await client.hget(meta_key, k)
            return str(value) if value is not None else str(default or "")
        except Exception as exc:
            log_event(
                "redis_get_meta_str_failed",
                level=logging.WARNING,
                session_id=sid,
                key=k,
                error_type=type(exc).__name__,
            )
            return str(default or "")

    async def set_meta_str(self, session_id: str, key: str, value: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        k = str(key or "").strip()
        if not k:
            return
        v = str(value or "")
        client = await self._redis()
        if client is None:
            self._fallback_meta[sid][k] = v
            return
        meta_key = free_talk_meta_key(self._settings.prefix, sid)
        try:
            await client.hset(meta_key, mapping={k: v})
            await client.expire(meta_key, self._settings.ttl_sec)
        except Exception as exc:
            log_event(
                "redis_set_meta_str_failed",
                level=logging.WARNING,
                session_id=sid,
                key=k,
                error_type=type(exc).__name__,
            )
            self._fallback_meta[sid][k] = v

    async def clear_session(self, session_id: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        client = await self._redis()
        if client is None:
            self._fallback_turns.pop(sid, None)
            self._fallback_summary.pop(sid, None)
            self._fallback_meta.pop(sid, None)
            return

        turns_key = free_talk_turns_key(self._settings.prefix, sid)
        summary_key = free_talk_summary_key(self._settings.prefix, sid)
        meta_key = free_talk_meta_key(self._settings.prefix, sid)
        try:
            await client.delete(turns_key, summary_key, meta_key)
        except Exception as exc:
            log_event(
                "redis_clear_session_failed",
                level=logging.WARNING,
                session_id=sid,
                error_type=type(exc).__name__,
            )
            self._fallback_turns.pop(sid, None)
            self._fallback_summary.pop(sid, None)
            self._fallback_meta.pop(sid, None)
