"""Topic registry loader/matcher and CRUD helpers for messenger router.

Назначение:
1) Централизованно хранить и читать темы из YAML-реестра.
2) Давать детерминированный match темы по тексту запроса.
3) Предоставлять безопасные CRUD-операции для UI редактора.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import re
import threading

import yaml

_REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "topic_registry.yaml"
_CACHE_LOCK = threading.RLock()
_CACHE_DATA: dict[str, Any] | None = None
_CACHE_MTIME: float = -1.0
_TOPIC_FLAG_PREFIX = "topic_registry:"
_VALID_LABELS = {
    "APPOINTMENT",
    "TEST_ASSIST",
    "TEST_RESULT",
    "DOCTOR_INFO",
    "DOCTOR_SCHEDULE",
    "PRICE",
    "ADDRESS",
    "PREPARE",
    "NEWS",
    "COMPLAINT",
    "URGENT",
    "MEDICAL_ADVICE",
    "OTHER",
}


@dataclass(frozen=True)
class TopicMatch:
    topic_id: str
    title: str
    label: str
    priority: int
    score: int
    matched_keywords: tuple[str, ...]
    matched_regex: tuple[str, ...]


def _normalize_text(text: str) -> str:
    raw = str(text or "").lower().replace("ё", "е")
    tokens = re.findall(r"[a-zа-я0-9]+", raw)
    return " ".join(tokens)


def _as_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _token_stem(token: str) -> str:
    tok = str(token or "").strip()
    if len(tok) >= 5:
        return tok[:5]
    if len(tok) >= 4:
        return tok[:4]
    return tok


def _keyword_matches(text_norm: str, keyword_norm: str) -> bool:
    if not keyword_norm:
        return False
    if keyword_norm in text_norm:
        return True

    text_tokens = [t for t in text_norm.split() if t]
    key_tokens = [t for t in keyword_norm.split() if t]
    if not text_tokens or not key_tokens:
        return False

    # Мягкое совпадение для русских падежей/форм слов:
    # считаем keyword совпавшим, если каждое слово keyword
    # найдено как exact (для коротких токенов) или как близкий префикс (stem).
    for key_tok in key_tokens:
        if len(key_tok) <= 3:
            if key_tok not in text_tokens:
                return False
            continue
        stem = _token_stem(key_tok)
        if not stem:
            continue
        matched_by_stem = False
        for tt in text_tokens:
            if len(tt) <= 3:
                continue
            if tt.startswith(stem) or stem.startswith(tt):
                matched_by_stem = True
                break
        if not matched_by_stem:
            return False
    return True


def _normalize_topic(topic: dict[str, Any]) -> dict[str, Any]:
    out = dict(topic)
    out["topic_id"] = str(out.get("topic_id") or "").strip()
    out["title"] = str(out.get("title") or "").strip()
    out["enabled"] = bool(out.get("enabled", True))
    out["priority"] = _safe_int(out.get("priority"), 0)
    label = str(out.get("label") or "OTHER").strip().upper()
    out["label"] = label if label in _VALID_LABELS else "OTHER"
    match = out.get("match")
    if not isinstance(match, dict):
        match = {}
    out["match"] = {
        "any_keywords": _as_str_list(match.get("any_keywords")),
        "all_keywords": _as_str_list(match.get("all_keywords")),
        "regex": _as_str_list(match.get("regex")),
        "exclude_keywords": _as_str_list(match.get("exclude_keywords")),
    }
    route = out.get("route")
    if not isinstance(route, dict):
        route = {}
    sources = route.get("sources")
    if not isinstance(sources, list):
        sources = []
    route["strategy"] = str(route.get("strategy") or "").strip()
    route["sources"] = [x for x in sources if isinstance(x, dict)]
    out["route"] = route
    if not isinstance(out.get("marks"), dict):
        out["marks"] = {}
    if not isinstance(out.get("context"), dict):
        out["context"] = {}
    if not isinstance(out.get("fallback"), dict):
        out["fallback"] = {}
    return out


def _normalize_registry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    out = copy.deepcopy(raw)
    out.setdefault("registry_version", 1)
    out.setdefault("description", "")
    if not isinstance(out.get("defaults"), dict):
        out["defaults"] = {}
    topics = out.get("topics")
    if not isinstance(topics, list):
        topics = []
    norm_topics: list[dict[str, Any]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        n = _normalize_topic(topic)
        if n.get("topic_id"):
            norm_topics.append(n)
    out["topics"] = norm_topics
    return out


def _topic_sort_key(topic: dict[str, Any]) -> tuple[int, str]:
    return (-_safe_int(topic.get("priority"), 0), str(topic.get("topic_id") or ""))


def _load_registry_from_disk() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return _normalize_registry({"registry_version": 1, "topics": []})
    text = _REGISTRY_PATH.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if text.strip() else {}
    return _normalize_registry(payload)


def _write_registry_to_disk(registry: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _normalize_registry(registry)
    dumped = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    tmp_path = _REGISTRY_PATH.with_suffix(f"{_REGISTRY_PATH.suffix}.tmp")
    tmp_path.write_text(dumped, encoding="utf-8")
    tmp_path.replace(_REGISTRY_PATH)


def load_registry(*, force_reload: bool = False) -> dict[str, Any]:
    global _CACHE_DATA, _CACHE_MTIME
    with _CACHE_LOCK:
        current_mtime = _REGISTRY_PATH.stat().st_mtime if _REGISTRY_PATH.exists() else -1.0
        if not force_reload and _CACHE_DATA is not None and _CACHE_MTIME == current_mtime:
            return copy.deepcopy(_CACHE_DATA)
        data = _load_registry_from_disk()
        _CACHE_DATA = data
        _CACHE_MTIME = current_mtime
        return copy.deepcopy(data)


def save_registry(registry: dict[str, Any]) -> dict[str, Any]:
    global _CACHE_DATA, _CACHE_MTIME
    with _CACHE_LOCK:
        data = _normalize_registry(registry)
        _write_registry_to_disk(data)
        _CACHE_DATA = data
        _CACHE_MTIME = _REGISTRY_PATH.stat().st_mtime if _REGISTRY_PATH.exists() else -1.0
        return copy.deepcopy(data)


def load_registry_text() -> str:
    if not _REGISTRY_PATH.exists():
        return ""
    return _REGISTRY_PATH.read_text(encoding="utf-8")


def save_registry_text(text: str) -> dict[str, Any]:
    parsed = yaml.safe_load(text) if str(text or "").strip() else {}
    data = _normalize_registry(parsed)
    return save_registry(data)


def list_topics(*, enabled_only: bool = False) -> list[dict[str, Any]]:
    data = load_registry()
    topics = data.get("topics")
    if not isinstance(topics, list):
        return []
    out = [t for t in topics if isinstance(t, dict)]
    if enabled_only:
        out = [t for t in out if bool(t.get("enabled", True))]
    out.sort(key=_topic_sort_key)
    return [copy.deepcopy(t) for t in out]


def get_topic(topic_id: str) -> dict[str, Any] | None:
    tid = str(topic_id or "").strip()
    if not tid:
        return None
    for topic in list_topics(enabled_only=False):
        if str(topic.get("topic_id") or "").strip() == tid:
            return topic
    return None


def _upsert_topic_to_registry(data: dict[str, Any], topic_obj: dict[str, Any]) -> dict[str, Any]:
    topic = _normalize_topic(topic_obj)
    tid = str(topic.get("topic_id") or "").strip()
    if not tid:
        raise ValueError("topic_id is required")
    topics = data.get("topics")
    if not isinstance(topics, list):
        topics = []
    replaced = False
    out_topics: list[dict[str, Any]] = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        if str(item.get("topic_id") or "").strip() == tid:
            out_topics.append(topic)
            replaced = True
        else:
            out_topics.append(_normalize_topic(item))
    if not replaced:
        out_topics.append(topic)
    out_topics.sort(key=_topic_sort_key)
    data["topics"] = out_topics
    return topic


def upsert_topic(topic_obj: dict[str, Any]) -> dict[str, Any]:
    data = load_registry()
    saved = _upsert_topic_to_registry(data, topic_obj)
    save_registry(data)
    return saved


def create_topic(topic_id: str, label: str = "OTHER", priority: int = 100) -> dict[str, Any]:
    tid = str(topic_id or "").strip()
    if not tid:
        raise ValueError("topic_id is required")
    lbl = str(label or "OTHER").strip().upper()
    if lbl not in _VALID_LABELS:
        lbl = "OTHER"
    topic = {
        "topic_id": tid,
        "title": tid,
        "enabled": True,
        "priority": int(priority),
        "label": lbl,
        "marks": {"domain": "custom", "subtype": "custom"},
        "match": {
            "any_keywords": [],
            "all_keywords": [],
            "regex": [],
            "exclude_keywords": [],
        },
        "route": {
            "strategy": "meili_first",
            "sources": [{"kind": "meili", "index": "main_index"}],
        },
        "context": {
            "interrupts_active_appointment": True,
            "require_cancel_confirm_if_appointment_active": True,
        },
        "fallback": {
            "action": "handoff",
            "reason": "knowledge_not_found",
            "message_key": "knowledge_not_found_operator",
        },
    }
    return upsert_topic(topic)


def delete_topic(topic_id: str) -> bool:
    tid = str(topic_id or "").strip()
    if not tid:
        return False
    data = load_registry()
    topics = data.get("topics")
    if not isinstance(topics, list):
        return False
    out = [t for t in topics if not (isinstance(t, dict) and str(t.get("topic_id") or "").strip() == tid)]
    if len(out) == len(topics):
        return False
    data["topics"] = out
    save_registry(data)
    return True


def set_topic_enabled(topic_id: str, enabled: bool) -> dict[str, Any] | None:
    topic = get_topic(topic_id)
    if topic is None:
        return None
    topic["enabled"] = bool(enabled)
    return upsert_topic(topic)


def build_topic_flag(topic_id: str) -> str:
    return f"{_TOPIC_FLAG_PREFIX}{str(topic_id or '').strip()}"


def extract_topic_id_from_flags(flags: set[str] | list[str] | tuple[str, ...] | None) -> str | None:
    if not flags:
        return None
    for raw in flags:
        s = str(raw or "").strip()
        if s.startswith(_TOPIC_FLAG_PREFIX):
            tail = s[len(_TOPIC_FLAG_PREFIX):].strip()
            if tail:
                return tail
    return None


def _match_topic(text_norm: str, topic: dict[str, Any]) -> TopicMatch | None:
    match = topic.get("match")
    if not isinstance(match, dict):
        return None

    exclude_keywords = [_normalize_text(x) for x in _as_str_list(match.get("exclude_keywords"))]
    for kw in exclude_keywords:
        if kw and _keyword_matches(text_norm, kw):
            return None

    any_keywords = [_normalize_text(x) for x in _as_str_list(match.get("any_keywords"))]
    all_keywords = [_normalize_text(x) for x in _as_str_list(match.get("all_keywords"))]
    regex_list = _as_str_list(match.get("regex"))

    matched_any = tuple(sorted({kw for kw in any_keywords if kw and _keyword_matches(text_norm, kw)}))
    if any_keywords and not matched_any:
        return None

    if all_keywords and not all(kw and _keyword_matches(text_norm, kw) for kw in all_keywords):
        return None

    matched_regex: list[str] = []
    for pattern in regex_list:
        try:
            if re.search(pattern, text_norm, flags=re.I):
                matched_regex.append(pattern)
        except re.error:
            continue

    has_signal = bool(matched_any or all_keywords or matched_regex)
    if not has_signal:
        return None

    score = len(matched_any) + len(all_keywords) + len(matched_regex)
    topic_id = str(topic.get("topic_id") or "").strip()
    if not topic_id:
        return None
    title = str(topic.get("title") or topic_id).strip()
    label = str(topic.get("label") or "OTHER").strip().upper()
    if label not in _VALID_LABELS:
        label = "OTHER"
    return TopicMatch(
        topic_id=topic_id,
        title=title,
        label=label,
        priority=_safe_int(topic.get("priority"), 0),
        score=score,
        matched_keywords=matched_any,
        matched_regex=tuple(matched_regex),
    )


def match_topic(text: str) -> TopicMatch | None:
    text_norm = _normalize_text(text)
    if not text_norm:
        return None

    candidates: list[TopicMatch] = []
    for topic in list_topics(enabled_only=True):
        matched = _match_topic(text_norm, topic)
        if matched is not None:
            candidates.append(matched)

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x.priority, -x.score, x.topic_id))
    return candidates[0]
