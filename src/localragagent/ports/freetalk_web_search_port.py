"""Port for web search via SearXNG."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class WebSearchSettings:
    base_url: str
    timeout_s: int
    max_results: int
    language: str
    healthcheck_timeout_s: int = 3
    healthcheck_ttl_s: int = 30


class WebSearchPort:
    def __init__(self, settings: WebSearchSettings) -> None:
        self._settings = settings
        self._health_is_ok: bool | None = None
        self._health_cache_expires_at: float = 0.0

    def _health_ttl(self) -> int:
        return max(1, int(self._settings.healthcheck_ttl_s))

    def _mark_healthy(self) -> None:
        self._health_is_ok = True
        self._health_cache_expires_at = time.monotonic() + self._health_ttl()

    def _mark_unhealthy(self) -> None:
        self._health_is_ok = False
        self._health_cache_expires_at = time.monotonic() + self._health_ttl()

    async def is_healthy(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and self._health_is_ok is not None and now < self._health_cache_expires_at:
            return self._health_is_ok

        healthy = await self._probe_health()
        if healthy:
            self._mark_healthy()
        else:
            self._mark_unhealthy()
        return healthy

    async def _probe_health(self) -> bool:
        url = str(self._settings.base_url or "").rstrip("/")
        if not url:
            return False
        endpoint = f"{url}/search"
        params = {
            "q": "healthcheck",
            "format": "json",
            "language": self._settings.language,
            "safesearch": "1",
        }
        try:
            async with httpx.AsyncClient(timeout=max(1, int(self._settings.healthcheck_timeout_s))) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False
        return isinstance(payload.get("results"), list)

    async def search(self, query: str, entities: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Searches the web through SearXNG `/search?format=json`.

        :param query: user query
        :param entities: optional extra params (currently not used)
        :return: normalized result payload
        """

        _ = entities
        q = str(query or "").strip()
        if not q:
            return {"query": "", "results": [], "note": "web_search: no query"}

        url = str(self._settings.base_url or "").rstrip("/")
        if not url:
            return {"query": q, "results": [], "note": "web_search: empty base_url"}
        if not await self.is_healthy():
            return {"query": q, "results": [], "note": "web_search source unavailable"}

        endpoint = f"{url}/search"
        params = {
            "q": q,
            "format": "json",
            "language": self._settings.language,
            "safesearch": "1",
        }

        try:
            async with httpx.AsyncClient(timeout=max(2, int(self._settings.timeout_s))) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            self._mark_unhealthy()
            return {
                "query": q,
                "results": [],
                "note": "web_search source unavailable",
            }

        raw_results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(raw_results, list):
            raw_results = []

        out: list[dict[str, Any]] = []
        for row in raw_results:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            url_value = str(row.get("url") or "").strip()
            snippet = str(row.get("content") or row.get("snippet") or "").strip()
            source = str(row.get("engine") or "").strip()
            if not title and not url_value and not snippet:
                continue
            out.append(
                {
                    "title": title,
                    "url": url_value,
                    "snippet": snippet,
                    "source": source,
                }
            )
            if len(out) >= max(1, int(self._settings.max_results)):
                break

        self._mark_healthy()
        return {
            "query": q,
            "results": out,
            "note": "web_search: searxng",
        }
