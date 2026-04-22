"""Общие helper-утилиты сервисного слоя (Stage 20, cluster 1).

Содержит чистые/near-pure функции, не привязанные к конкретному домену:
runtime-конфиг, JSON-парсинг, нормализация текста, дедуп, meili-детекторы,
fallback-генераторы.

Переносится из ``services_legacy.py`` в рамках Stage 20 с сохранением
обратной совместимости через re-export shim в legacy.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..runtime_config import config as c
from ..russian_nlu import normalize_ru


try:
    DOCTORS_TOP_N = max(1, int(c.MR_DOCTORS_TOP_N))
except Exception:
    DOCTORS_TOP_N = 4


_NEAREST_HINT_RE = re.compile(
    r"\b(ближайш\w*|сам\w*\s+ранн\w*|раньше|поскорее|свободн\w*\s+окн\w*)\b",
    re.I,
)


_DOC_RELEVANCE_STOPWORDS = {
    "как",
    "что",
    "где",
    "когда",
    "нужно",
    "нужна",
    "нужен",
    "нужны",
    "получить",
    "получения",
    "подскажите",
    "пожалуйста",
    "добрый",
    "день",
    "здравствуйте",
    "мне",
    "для",
    "по",
    "про",
    "это",
    "этого",
    "требуется",
    "делаете",
    "сколько",
    "стоимость",
    "стоимости",
}


def _runtime_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(getattr(c, name))
    except Exception:
        value = int(default)
    value = max(min_value, value)
    value = min(max_value, value)
    return value


def _runtime_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    try:
        value = float(getattr(c, name))
    except Exception:
        value = float(default)
    value = max(min_value, value)
    value = min(max_value, value)
    return value


def _runtime_bool(name: str, default: bool) -> bool:
    try:
        return bool(getattr(c, name))
    except Exception:
        return bool(default)


def _normalise_input(s: str) -> str:
    return re.sub(r"\s+", " ", normalize_ru(s))


def _normalise_catalog_text(s: str) -> str:
    norm = _normalise_input(s)
    norm = re.sub(r"[^a-zа-я0-9\- ]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _dedupe_str(items: list[str], *, max_items: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw or "").strip()
        if not value:
            continue
        key = _normalise_catalog_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Извлекает первый JSON-объект из произвольного текстового ответа модели.

    :param text: raw-ответ LLM
    :return: dict или None
    """

    s = str(text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start : idx + 1]
                try:
                    obj = json.loads(chunk)
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _has_nearest_hint(text: str) -> bool:
    return bool(_NEAREST_HINT_RE.search(text or ""))


def _get_first_present(d: dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _as_int(val: Any) -> int | None:
    try:
        return int(val)
    except Exception:
        return None


def _coerce_top_n(value: Any, *, default: int = DOCTORS_TOP_N) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(1, min(parsed, 20))


def _service_fallback(
    *,
    note: str,
    handoff_message: str,
    entities: dict[str, Any],
    reason: str = "service_error",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": note,
        "handoff_required": True,
        "handoff_reason": reason,
        "handoff_message": handoff_message,
        "entities_used": entities,
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _tax_doc_guidance_response(entities: dict[str, Any], *, note: str) -> dict[str, Any]:
    """
    Возвращает детерминированную ссылку на оформление справки для налогового вычета.

    :param entities: текущие сущности роутера
    :param note: диагностическая пометка источника
    :return: payload OTHER/doc без handoff_required
    """

    return {
        "content": "Заказ справки на налоговый вычет осуществляется на сайте https://naykalab.ru/spravka-nalogoviy-vichet",
        "note": note,
        "entities_used": entities,
    }


def _is_meili_error_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return "Ошибка поисковой системы" in text or "Meilisearch" in text


def _is_meili_no_matches_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    norm = _normalise_input(text)
    if not norm:
        return True
    return "совпадений не найдено" in norm


def _doc_tokens(text: str) -> set[str]:
    norm = _normalise_input(text)
    out: set[str] = set()
    for token in re.findall(r"[a-zа-я0-9]{3,}", norm):
        if token.isdigit() or token in _DOC_RELEVANCE_STOPWORDS:
            continue
        out.add(token)
    return out


def _is_main_index_relevant(query: str, content: str, *, doc_kind: str) -> bool:
    content_norm = _normalise_input(content)
    query_norm = _normalise_input(query)
    if not content_norm:
        return False

    if doc_kind == "tax":
        tax_anchors = (
            "налог",
            "вычет",
            "фнс",
            "налогов",
            "оплате медицинских услуг",
        )
        if not any(anchor in content_norm for anchor in tax_anchors):
            return False
    else:
        if "договор" in query_norm and "договор" not in content_norm:
            return False
        if "амбулатор" in query_norm and not ("амбулатор" in content_norm or "карт" in content_norm):
            return False
        if ("соревн" in query_norm or "допуск" in query_norm) and not (
            "соревн" in content_norm or "допуск" in content_norm
        ):
            return False
        if "справк" in query_norm and not any(x in query_norm for x in ("налог", "вычет", "фнс")) and not (
            "справк" in content_norm or "допуск" in content_norm
        ):
            return False

    q_tokens = _doc_tokens(query_norm)
    if not q_tokens:
        return True
    content_tokens = _doc_tokens(content_norm)
    return bool(q_tokens & content_tokens)
