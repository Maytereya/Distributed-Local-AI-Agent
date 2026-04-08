"""FreeTalk runtime entrypoint for UI integration."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any, AsyncGenerator

from localragagent.ports.freetalk_services_port import LegacyServicesPort
from localragagent.ports.freetalk_web_search_port import WebSearchPort, WebSearchSettings

from .agent import FreeTalkAgent
from .config import load_config
from .memory_persist import PersistSettings, PersistentSummaryStore
from .memory_redis import RedisMemoryStore, RedisSettings
from .observability import log_event


@lru_cache(maxsize=1)
def _agent() -> FreeTalkAgent:
    cfg = load_config()
    memory = RedisMemoryStore(
        RedisSettings(url=cfg.redis_url, prefix=cfg.redis_prefix, ttl_sec=cfg.session_ttl_sec)
    )
    persist = PersistentSummaryStore(PersistSettings(jsonl_path=cfg.persistent_memory_path))
    services = LegacyServicesPort()
    web_search = None
    if cfg.enable_web_search_tool:
        web_search = WebSearchPort(
            WebSearchSettings(
                base_url=cfg.web_search_url,
                timeout_s=cfg.web_search_timeout_s,
                max_results=cfg.web_search_max_results,
                language=cfg.web_search_language,
                healthcheck_timeout_s=cfg.web_search_healthcheck_timeout_s,
                healthcheck_ttl_s=cfg.web_search_healthcheck_ttl_s,
            )
        )
    log_event(
        "freetalk_agent_initialized",
        mode_key=cfg.mode_key,
        redis_url=cfg.redis_url,
        web_search_enabled=cfg.enable_web_search_tool,
        web_search_url=cfg.web_search_url if cfg.enable_web_search_tool else "",
    )
    return FreeTalkAgent.build(
        config=cfg,
        services=services,
        memory=memory,
        persist=persist,
        web_search=web_search,
    )


def ensure_session_id(session_id: str | None) -> str:
    sid = str(session_id or "").strip()
    if sid:
        return sid
    return f"gr_ft_{uuid.uuid4().hex[:12]}"


async def run_free_talk(
    *,
    message: str,
    history: list[dict[str, Any]] | None = None,  # kept for ChatInterface compatibility
    session_id: str | None = None,
) -> str:
    text, _next_session_id = await run_free_talk_with_state(
        message=message,
        history=history,
        session_id=session_id,
    )
    return text


async def run_free_talk_with_state(
    *,
    message: str,
    history: list[dict[str, Any]] | None = None,  # kept for ChatInterface compatibility
    session_id: str | None = None,
) -> tuple[str, str]:
    _ = history  # history is stored in Redis by session_id
    sid = ensure_session_id(session_id)
    reply = await _agent().chat(message, sid)
    text = str(reply.text or "").strip()
    next_session_id = str(reply.next_session_id or "").strip() or sid
    if next_session_id != sid:
        log_event(
            "freetalk_session_rotated",
            level=logging.WARNING,
            old_session_id=sid,
            next_session_id=next_session_id,
        )
    return text, next_session_id


async def stream_free_talk(
    *,
    message: str,
    history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    text = await run_free_talk(message=message, history=history, session_id=session_id)
    yield text


async def stream_free_talk_with_state(
    *,
    message: str,
    history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[tuple[str, str], None]:
    text, next_session_id = await run_free_talk_with_state(
        message=message,
        history=history,
        session_id=session_id,
    )
    yield text, next_session_id
