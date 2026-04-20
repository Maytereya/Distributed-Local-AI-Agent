"""Address / branch helpers (Stage 20 cluster 4).

Pure helpers: real-address heuristics, branch payload shaping, homecode
extraction, and nonbookable-points data access. Moved out of
``services_legacy.py`` behind a re-export shim.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._common import _as_int, _normalise_input
from ._regions import (
    _ADDRESS_HINT_RE,
    _extract_region_phone,
    _extract_region_work_time,
    _norm_city,
    _region_display_name,
)

_PRICE_HOMECODE_DOTTED_RE = re.compile(r"\b\d+(?:\.\d+){1,6}\b")
_PRICE_HOMECODE_NUM_RE = re.compile(r"\b\d{4,}\b")
_PROCEDURE_BRANCH_LOOKUP_RE = re.compile(
    r"\b(где|сдела\w*|пройти|провест\w*|выполня\w*|дела\w*|можно|пройти\s+диагностик\w*)\b",
    re.I,
)
_NONBOOKABLE_POINTS_PATH = Path(__file__).resolve().parent.parent / "data" / "nonbookable_points.json"
# Fallback-карта для процедур, где API не отдает надежный branch-level match.
_STATIC_PROCEDURE_BRANCH_OVERRIDES: dict[str, tuple[str, ...]] = {
    "флюорограф": ("г. Самара, пр. Ленина, 5",),
}


def _is_procedure_branch_lookup_query(query_text: str, service_q: str) -> bool:
    """
    Определяет, что пользователь ищет филиал под конкретную процедуру.

    :param query_text: исходный текст запроса
    :param service_q: нормализованная процедура/услуга
    :return: True, если это адресный lookup по процедуре
    """
    if not str(service_q or "").strip():
        return False
    q = _normalise_input(query_text or "")
    if not q:
        return False
    return bool(_PROCEDURE_BRANCH_LOOKUP_RE.search(q))


def _looks_like_real_address(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if re.fullmatch(r"ID\s+\d+", s, flags=re.I):
        return False
    if _ADDRESS_HINT_RE.search(s):
        return True
    if re.search(r"\d", s):
        return True
    return False


def _soft_address_match(left: str, right: str) -> bool:
    """
    Мягко сопоставляет два адреса из разных источников (API/кэш).

    :param left: адрес из первого источника
    :param right: адрес из второго источника
    :return: True, если строки похожи и описывают один филиал
    """
    left_norm = _normalise_input(left)
    right_norm = _normalise_input(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
        return True
    left_compact = re.sub(r"[^a-zа-я0-9]+", "", left_norm)
    right_compact = re.sub(r"[^a-zа-я0-9]+", "", right_norm)
    if not left_compact or not right_compact:
        return False
    return (
        left_compact == right_compact
        or left_compact in right_compact
        or right_compact in left_compact
    )


def _static_procedure_addresses(service_q: str) -> list[str]:
    """
    Возвращает статические адреса для процедур с известными API-пробелами.

    :param service_q: нормализованное имя процедуры
    :return: список адресов филиалов
    """
    sq = _normalise_input(service_q or "")
    if not sq:
        return []
    out: list[str] = []
    for needle, addresses in _STATIC_PROCEDURE_BRANCH_OVERRIDES.items():
        if needle in sq:
            for addr in addresses:
                if addr not in out:
                    out.append(addr)
    return out


def _addresses_to_branch_payload(
    addresses: list[str],
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Преобразует список адресов в унифицированный branch payload.

    Если адрес находится в live `/regions`, дополняем id/phone/work_time.
    Иначе возвращаем минимальную карточку адреса.

    :param addresses: адреса филиалов
    :param regions: live-список филиалов
    :return: список branches для `address_info`
    """
    out: list[dict[str, Any]] = []
    for addr in addresses:
        best_region: dict[str, Any] | None = None
        for region in regions:
            if not isinstance(region, dict):
                continue
            disp = _region_display_name(region)
            if not disp:
                continue
            # В branch payload подмешиваем только реальные адреса филиалов.
            # Иначе generic live-region вроде "Самара" может перехватить точный
            # адрес из priceUnits и испортить patient-facing рендер.
            if not _looks_like_real_address(disp):
                continue
            if _soft_address_match(addr, disp):
                best_region = region
                break

        out.append(
            {
                "id": _as_int(best_region.get("id")) if isinstance(best_region, dict) else None,
                "address": _region_display_name(best_region) if isinstance(best_region, dict) else addr,
                "city": str(best_region.get("city") or "").strip() if isinstance(best_region, dict) else "Самара",
                "phone": _extract_region_phone(best_region) if isinstance(best_region, dict) else "",
                "work_time": _extract_region_work_time(best_region) if isinstance(best_region, dict) else "",
            }
        )
    return out


def _extract_homecode_query(text: str) -> str:
    s = _normalise_input(text)
    m = _PRICE_HOMECODE_DOTTED_RE.search(s)
    if m:
        return str(m.group(0)).strip()
    m = _PRICE_HOMECODE_NUM_RE.search(s)
    if m:
        return str(m.group(0)).strip()
    return ""


@lru_cache(maxsize=1)
def _load_nonbookable_points() -> dict[str, list[dict[str, Any]]]:
    if not _NONBOOKABLE_POINTS_PATH.exists():
        return {}
    try:
        raw = json.loads(_NONBOOKABLE_POINTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for city, rows in raw.items():
        if not isinstance(city, str) or not isinstance(rows, list):
            continue
        key = _norm_city(city)
        if not key:
            continue
        norm_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            norm_rows.append(
                {
                    "address": addr,
                    "phone": str(row.get("phone") or "").strip(),
                    "work_time": str(row.get("work_time") or "").strip(),
                    "has_analysis": bool(row.get("has_analysis", True)),
                    "has_ekg": bool(row.get("has_ekg", False)),
                    "city": str(row.get("city") or city).strip(),
                }
            )
        if norm_rows:
            out[key] = norm_rows
    return out


def _nonbookable_needs(service_q: str) -> tuple[bool, bool]:
    s = _normalise_input(service_q or "")
    if not s:
        return False, False
    need_analysis = bool(re.search(r"\b(анализ\w*|лаборатор\w*|биоматериал)\b", s))
    need_ekg = bool(re.search(r"\b(экг|электрокардиограм\w*)\b", s))
    return need_analysis, need_ekg


def _static_nonbookable_branches(city: str, service_q: str) -> list[dict[str, Any]]:
    data = _load_nonbookable_points()
    city_key = _norm_city(city)
    if not city_key:
        return []
    rows = data.get(city_key) or []
    if not rows:
        return []
    need_analysis, need_ekg = _nonbookable_needs(service_q)
    out: list[dict[str, Any]] = []
    for row in rows:
        if need_analysis and not bool(row.get("has_analysis", True)):
            continue
        if need_ekg and not bool(row.get("has_ekg", False)):
            continue
        out.append(dict(row))
    return out
