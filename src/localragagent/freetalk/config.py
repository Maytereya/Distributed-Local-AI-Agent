"""FreeTalk runtime configuration loaded from package-local config.ini."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FreeTalkConfig:
    mode_key: str
    redis_url: str
    redis_prefix: str
    session_ttl_sec: int
    history_tail_turns: int
    compaction_trigger_turns: int
    summary_keep_turns: int
    max_tool_steps: int
    llm_timeout_s: int
    llm_queue_timeout_ms: int
    include_meili_tools: bool
    enable_web_search_tool: bool
    web_search_url: str
    web_search_timeout_s: int
    web_search_max_results: int
    web_search_language: str
    web_search_healthcheck_timeout_s: int
    web_search_healthcheck_ttl_s: int
    context_window_tokens: int
    context_warn_ratio: float
    context_estimate_chars_per_token: int
    context_response_reserve_tokens: int
    persistent_memory_path: Path
    system_prompt_path: Path


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _config_path() -> Path:
    return _package_dir() / "config.ini"


def _as_bool(value: str, default: bool = False) -> bool:
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _as_int(value: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        parsed = default
    if min_value is not None and parsed < min_value:
        parsed = min_value
    if max_value is not None and parsed > max_value:
        parsed = max_value
    return parsed


def _as_float(
    value: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        parsed = default
    if min_value is not None and parsed < min_value:
        parsed = min_value
    if max_value is not None and parsed > max_value:
        parsed = max_value
    return parsed


@lru_cache(maxsize=1)
def load_config() -> FreeTalkConfig:
    config_file = _config_path()
    if not config_file.exists():
        raise FileNotFoundError(f"FreeTalk config not found: {config_file}")

    parser = ConfigParser()
    parser.read(config_file, encoding="utf-8")
    section = parser["freetalk"] if parser.has_section("freetalk") else {}

    package_dir = _package_dir()
    persistent_memory_path = Path(str(section.get("persistent_memory_path", "app_data/free_talk_memory/dialog_summaries.jsonl"))).expanduser()
    if not persistent_memory_path.is_absolute():
        persistent_memory_path = Path.cwd() / persistent_memory_path

    system_prompt_rel = str(section.get("system_prompt_path", "system_prompt.txt")).strip() or "system_prompt.txt"
    system_prompt_path = Path(system_prompt_rel)
    if not system_prompt_path.is_absolute():
        system_prompt_path = package_dir / system_prompt_path

    return FreeTalkConfig(
        mode_key=str(section.get("mode_key", "Free-talk-Ai")).strip() or "Free-talk-Ai",
        redis_url=str(section.get("redis_url", "redis://redis:6379/0")).strip() or "redis://redis:6379/0",
        redis_prefix=str(section.get("redis_prefix", "ft")).strip() or "ft",
        session_ttl_sec=_as_int(str(section.get("session_ttl_sec", "86400")), 86400, min_value=300, max_value=604800),
        history_tail_turns=_as_int(str(section.get("history_tail_turns", "14")), 14, min_value=2, max_value=64),
        compaction_trigger_turns=_as_int(
            str(section.get("compaction_trigger_turns", "22")), 22, min_value=8, max_value=200
        ),
        summary_keep_turns=_as_int(str(section.get("summary_keep_turns", "8")), 8, min_value=2, max_value=50),
        max_tool_steps=_as_int(str(section.get("max_tool_steps", "3")), 3, min_value=1, max_value=8),
        llm_timeout_s=_as_int(str(section.get("llm_timeout_s", "40")), 40, min_value=5, max_value=120),
        llm_queue_timeout_ms=_as_int(
            str(section.get("llm_queue_timeout_ms", "30000")), 30000, min_value=1000, max_value=120000
        ),
        include_meili_tools=_as_bool(str(section.get("include_meili_tools", "false")), False),
        enable_web_search_tool=_as_bool(str(section.get("enable_web_search_tool", "true")), True),
        web_search_url=str(section.get("web_search_url", "http://searxng:8080")).strip() or "http://searxng:8080",
        web_search_timeout_s=_as_int(str(section.get("web_search_timeout_s", "12")), 12, min_value=2, max_value=60),
        web_search_max_results=_as_int(str(section.get("web_search_max_results", "5")), 5, min_value=1, max_value=20),
        web_search_language=str(section.get("web_search_language", "ru-RU")).strip() or "ru-RU",
        web_search_healthcheck_timeout_s=_as_int(
            str(section.get("web_search_healthcheck_timeout_s", "3")), 3, min_value=1, max_value=15
        ),
        web_search_healthcheck_ttl_s=_as_int(
            str(section.get("web_search_healthcheck_ttl_s", "30")), 30, min_value=1, max_value=300
        ),
        context_window_tokens=_as_int(
            str(section.get("context_window_tokens", "24576")), 24576, min_value=2048, max_value=262144
        ),
        context_warn_ratio=_as_float(str(section.get("context_warn_ratio", "0.82")), 0.82, min_value=0.5, max_value=0.98),
        context_estimate_chars_per_token=_as_int(
            str(section.get("context_estimate_chars_per_token", "4")), 4, min_value=2, max_value=8
        ),
        context_response_reserve_tokens=_as_int(
            str(section.get("context_response_reserve_tokens", "2048")), 2048, min_value=128, max_value=8192
        ),
        persistent_memory_path=persistent_memory_path,
        system_prompt_path=system_prompt_path,
    )
