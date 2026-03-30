from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Hashable, Mapping


@dataclass
class _ListCacheEntry:
    payload: list[Any]
    cached_at: float
    fresh_until: float
    stale_until: float
    is_negative: bool


class AsyncListTTLStaleCache:
    """In-memory TTL cache for list payloads with stale-on-error fallback."""

    def __init__(
        self,
        *,
        fresh_ttl_seconds: int,
        stale_ttl_seconds: int,
        negative_ttl_seconds: int,
        max_keys: int,
        logger: logging.Logger | None = None,
        log_events: bool = False,
        name: str = "schedule_cache",
        clone_payload: bool = True,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self._fresh_ttl_seconds = max(1, int(fresh_ttl_seconds))
        self._stale_ttl_seconds = max(1, int(stale_ttl_seconds))
        self._negative_ttl_seconds = max(1, int(negative_ttl_seconds))
        self._max_keys = max(1, int(max_keys))
        self._logger = logger
        self._log_events = bool(log_events)
        self._name = str(name or "schedule_cache")
        self._clone_payload = bool(clone_payload)
        self._time = time_func or time.time

        self._cache: dict[Hashable, _ListCacheEntry] = {}
        self._locks: dict[Hashable, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def clear(self) -> None:
        self._cache.clear()
        self._locks.clear()

    def _clone(self, payload: list[Any]) -> list[Any]:
        if not self._clone_payload:
            return payload
        return copy.deepcopy(payload)

    def _log(self, event: str, key: Hashable, key_details: Mapping[str, Any] | None = None, **details: Any) -> None:
        if not self._log_events or self._logger is None:
            return
        chunks: list[str] = []
        if key_details:
            for name in sorted(key_details):
                value = key_details[name]
                if value is not None:
                    chunks.append(f"{name}={value!r}")
        else:
            chunks.append(f"key={key!r}")
        for name in sorted(details):
            value = details[name]
            if value is not None:
                chunks.append(f"{name}={value!r}")
        suffix = (" " + " ".join(chunks)) if chunks else ""
        self._logger.info("%s event=%s%s", self._name, event, suffix)

    async def _get_key_lock(self, key: Hashable) -> asyncio.Lock:
        async with self._locks_guard:
            existing = self._locks.get(key)
            if existing is not None:
                return existing
            created = asyncio.Lock()
            self._locks[key] = created
            return created

    async def _prune(self, now: float) -> None:
        if len(self._cache) <= self._max_keys:
            return

        evicted: list[Hashable] = []
        for key, entry in list(self._cache.items()):
            if now > entry.stale_until:
                self._cache.pop(key, None)
                evicted.append(key)

        if len(self._cache) > self._max_keys:
            overflow = len(self._cache) - self._max_keys
            oldest = sorted(self._cache.items(), key=lambda kv: kv[1].cached_at)[:overflow]
            for key, _ in oldest:
                self._cache.pop(key, None)
                evicted.append(key)

        if evicted:
            async with self._locks_guard:
                for key in evicted:
                    self._locks.pop(key, None)

    async def _store(self, key: Hashable, payload: list[Any], key_details: Mapping[str, Any] | None) -> None:
        now = self._time()
        is_negative = not payload
        fresh_ttl = self._negative_ttl_seconds if is_negative else self._fresh_ttl_seconds
        stale_ttl = max(self._stale_ttl_seconds, fresh_ttl)
        self._cache[key] = _ListCacheEntry(
            payload=self._clone(payload),
            cached_at=now,
            fresh_until=now + fresh_ttl,
            stale_until=now + stale_ttl,
            is_negative=is_negative,
        )
        self._log(
            "store_negative" if is_negative else "store_positive",
            key,
            key_details=key_details,
            size=len(payload),
            fresh_ttl=fresh_ttl,
            stale_ttl=stale_ttl,
        )
        await self._prune(now)

    async def get_or_fetch(
        self,
        key: Hashable,
        fetcher: Callable[[], Awaitable[Any]],
        *,
        key_details: Mapping[str, Any] | None = None,
    ) -> Any:
        now = self._time()
        entry = self._cache.get(key)
        if entry and now <= entry.fresh_until:
            self._log(
                "hit_fresh",
                key,
                key_details=key_details,
                age_sec=max(0, int(now - entry.cached_at)),
                negative=entry.is_negative,
            )
            return self._clone(entry.payload)

        self._log(
            "miss_cold" if entry is None else "miss_expired",
            key,
            key_details=key_details,
            age_sec=(None if entry is None else max(0, int(now - entry.cached_at))),
        )

        lock = await self._get_key_lock(key)
        async with lock:
            now = self._time()
            entry = self._cache.get(key)
            if entry and now <= entry.fresh_until:
                self._log(
                    "hit_fresh_after_lock",
                    key,
                    key_details=key_details,
                    age_sec=max(0, int(now - entry.cached_at)),
                    negative=entry.is_negative,
                )
                return self._clone(entry.payload)

            try:
                payload = await fetcher()
            except Exception as exc:
                if entry and not entry.is_negative and now <= entry.stale_until:
                    self._log(
                        "hit_stale_on_error",
                        key,
                        key_details=key_details,
                        error=type(exc).__name__,
                        age_sec=max(0, int(now - entry.cached_at)),
                    )
                    return self._clone(entry.payload)
                self._log(
                    "source_error_no_stale",
                    key,
                    key_details=key_details,
                    error=type(exc).__name__,
                )
                raise

            if isinstance(payload, list):
                await self._store(key, payload, key_details)
                return self._clone(payload)

            if entry and not entry.is_negative and now <= entry.stale_until:
                self._log(
                    "hit_stale_on_non_list",
                    key,
                    key_details=key_details,
                    payload_type=type(payload).__name__,
                    age_sec=max(0, int(now - entry.cached_at)),
                )
                return self._clone(entry.payload)

            self._log(
                "source_non_list_no_cache",
                key,
                key_details=key_details,
                payload_type=type(payload).__name__,
            )
            return payload
