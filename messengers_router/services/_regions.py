"""Region/city helpers (Stage 20 cluster 3).

Pure helpers for region filtering, city extraction, phone/work-time
parsing. Moved out of ``services_legacy.py`` behind a re-export shim.
"""

from __future__ import annotations

import re
from typing import Any

from ..specialty_parser import UZI_QUERY_RE as _UZI_QUERY_RE
from ._common import _normalise_input


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_ADDRESS_HINT_RE = re.compile(
    r"\b(ул\.?|улица|пр\.?|проспект|пр-?т|тракт|б-р|бульвар|шоссе|пер\.?|переулок|наб\.?|площадь|дом|д\.|корп\.?|к\.|пом\.?)\b",
    re.I,
)
_PHONE_EXTRACT_RE = re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d")
_CITY_PREFIX_RE = re.compile(r"\b(?:г|город)\.?\s*([а-яёa-z\-]+)\b", re.I)


def _is_samara_city_value(value: str | None) -> bool:
    city = _extract_city_token(value)
    return city == "самара"


def _is_non_samara_city_value(value: str | None) -> bool:
    city = _extract_city_token(value)
    if city is not None:
        return bool(city and city != "самара")
    # Multi-word place value («Нижний Новгород», «Ульяновская область»):
    # _extract_city_token returns None for >1 token, so the single-word check
    # above misses it. Treat as non-Samara when the value looks like a place
    # (a few alpha tokens, no digits, not an address) and carries no «самар»
    # stem — keeping «Самарская область»/«город Самара» Samara-supported.
    norm = _normalize_region_text(value or "")
    if not norm or "самар" in norm or any(ch.isdigit() for ch in norm):
        return False
    if _ADDRESS_HINT_RE.search(norm):
        return False
    tokens = [t for t in re.findall(r"[a-zа-яё\-]+", norm) if t not in {"г", "город", "обл", "район"}]
    return 1 <= len(tokens) <= 4


def _extract_city_token(value: str | None) -> str | None:
    if not value:
        return None
    norm = _normalise_input(value)
    if not norm:
        return None
    if _ADDRESS_HINT_RE.search(norm):
        return None
    norm = re.sub(r"[^a-zа-я0-9\-]+", " ", norm).strip()
    if not norm or any(ch.isdigit() for ch in norm):
        return None
    parts = norm.split()
    if not parts:
        return None
    if parts[0] in {"г", "город"}:
        parts = parts[1:]
    if len(parts) != 1:
        return None
    city = parts[0].strip()
    return city or None


def _normalize_region_text(value: str) -> str:
    norm = _normalise_input(value)
    return re.sub(r"\s+", " ", norm).strip()


def _compact_region_text(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", _normalize_region_text(value))


def _is_explicit_non_samara_region(value: str) -> bool:
    norm = _normalize_region_text(value)
    if not norm:
        return False
    if "самара" in norm:
        return False
    # Частый кейс в данных: оренбургские площадки не должны попадать в самарский контур.
    if "оренбург" in norm:
        return True
    for city in _CITY_PREFIX_RE.findall(norm):
        city_norm = _normalize_region_text(city)
        if city_norm and city_norm != "самара":
            return True
    return False


def _region_matches_samara_tokens(region: str, samara_tokens: set[str]) -> bool:
    if not samara_tokens:
        return False
    region_norm = _normalize_region_text(region)
    if not region_norm:
        return False
    if region_norm in samara_tokens:
        return True
    region_compact = _compact_region_text(region_norm)
    for token in samara_tokens:
        if region_norm in token or token in region_norm:
            return True
        token_compact = _compact_region_text(token)
        if region_compact and token_compact and (region_compact in token_compact or token_compact in region_compact):
            return True
    return False


def _has_explicit_non_samara_regions(values: list[str]) -> bool:
    for value in values:
        if _is_explicit_non_samara_region(value):
            return True
    return False


def _region_display_name(region: dict[str, Any]) -> str:
    """
    Берем человекочитаемый адрес из live /regions.
    Приоритет: addressForSite -> name.
    """
    addr = str(region.get("addressForSite") or "").strip()
    name = str(region.get("name") or "").strip()
    return addr or name


def _schedule_regions_with_free_slots(schedule: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not isinstance(schedule, dict):
        return out
    for region_name, days in schedule.items():
        if not isinstance(days, list):
            continue
        has_free = False
        for day in days:
            if not isinstance(day, dict):
                continue
            slots = day.get("slots")
            if not isinstance(slots, list):
                continue
            if any(str(s or "").strip() for s in slots):
                has_free = True
                break
        if has_free:
            region_clean = str(region_name or "").strip()
            if region_clean and region_clean not in out:
                out.append(region_clean)
    return out


def _extract_region_phone(region: dict[str, Any]) -> str:
    phone_keys = ("phone", "phoneForSite", "phones", "phoneNumbers", "tel", "telephone")
    for k in phone_keys:
        v = region.get(k)
        if isinstance(v, str):
            nums = _PHONE_EXTRACT_RE.findall(v)
            if nums:
                return ", ".join(dict.fromkeys(n.strip() for n in nums))
            if v.strip():
                return v.strip()
        if isinstance(v, list):
            parts: list[str] = []
            for item in v:
                if isinstance(item, str):
                    nums = _PHONE_EXTRACT_RE.findall(item)
                    parts.extend(nums or [item.strip()])
                elif isinstance(item, dict):
                    val = str(item.get("phone") or item.get("value") or "").strip()
                    if val:
                        parts.append(val)
            clean = [p for p in parts if p]
            if clean:
                return ", ".join(dict.fromkeys(clean))
    return ""


def _extract_region_work_time(region: dict[str, Any]) -> str:
    work_keys = (
        "workTime",
        "work_time",
        "worktime",
        "workHours",
        "work_hours",
        "schedule",
        "scheduleForSite",
        "openingHours",
        "hours",
        "mode",
    )
    for k in work_keys:
        v = region.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if str(x).strip()]
            if parts:
                return "; ".join(parts)
        if isinstance(v, dict):
            parts = []
            for kk, vv in v.items():
                txt = str(vv).strip()
                if txt:
                    parts.append(f"{kk}: {txt}")
            if parts:
                return "; ".join(parts)

    # Live /regions отдаёт график не строкой, а структурно: weekdaysFrom/To,
    # saturdayFrom/To, sundayFrom/To (например «07:00:00»). Собираем из них
    # человекочитаемое расписание — иначе «График» пропадает из адресных ответов.
    def _hhmm(value: Any) -> str:
        s = str(value or "").strip()
        m = re.match(r"^(\d{1,2}):(\d{2})", s)
        return f"{int(m.group(1)):02d}:{m.group(2)}" if m else s

    def _range(from_key: str, to_key: str) -> str:
        a, b = _hhmm(region.get(from_key)), _hhmm(region.get(to_key))
        if a and b:
            return f"{a}–{b}"
        return a or b or ""

    weekday = _range("weekdaysFrom", "weekdaysTo")
    saturday = _range("saturdayFrom", "saturdayTo")
    sunday = _range("sundayFrom", "sundayTo")
    schedule_parts: list[str] = []
    if weekday:
        schedule_parts.append(f"будни {weekday}")
    if saturday and saturday == sunday:
        schedule_parts.append(f"выходные {saturday}")
    else:
        if saturday:
            schedule_parts.append(f"сб {saturday}")
        if sunday:
            schedule_parts.append(f"вс {sunday}")
    if schedule_parts:
        return ", ".join(schedule_parts)
    return ""


def _norm_city(s: str) -> str:
    t = _normalise_input(s or "")
    t = re.sub(r"^г\.?\s*", "", t)
    return t.strip()


def _service_procedure_flag(service_q: str) -> str | None:
    """Флаг филиала из /site/regions, определяющий МЕСТО оказания процедуры.

    Возвращает 'usi' | 'analysis' | 'ecg' | None. Источник адресов оказания
    процедуры — флаги филиалов в /site/regions (подтверждено разработчиком
    Наяки), а НЕ priceUnit care-setting хардкод и НЕ regionName врача (это филиал
    приёма врача, не место процедуры). Когда услуга попадает под такой флаг,
    адреса берём фильтром регионов по флагу, а не из хардкода.

    :param service_q: нормализованный/сырой текст услуги
    :return: имя флага филиала или None (для не-флаговых услуг — напр. консультаций)
    """
    sq = _normalise_input(service_q or "")
    if not sq:
        return None
    from ._addresses_helpers import _nonbookable_needs  # noqa: PLC0415

    # УЗИ проверяем первым: его маркер не пересекается с analysis/ecg.
    if bool(_UZI_QUERY_RE.search(sq)):
        return "usi"
    need_analysis, need_ekg = _nonbookable_needs(sq)
    if need_ekg:
        return "ecg"
    if need_analysis:
        return "analysis"
    return None


def _filter_regions_by_service_flags(regions: list[dict[str, Any]], service_q: str) -> list[dict[str, Any]]:
    sq = _normalise_input(service_q or "")
    if not sq:
        return list(regions)

    # Lazy imports to avoid circular imports with _addresses_helpers / _doctors_helpers.
    from ._addresses_helpers import _nonbookable_needs  # noqa: PLC0415
    from ._doctors_helpers import _extract_specialty_from_text  # noqa: PLC0415

    need_analysis, need_ekg = _nonbookable_needs(sq)
    need_uzi = bool(_UZI_QUERY_RE.search(sq))
    need_doctor = False
    if not (need_analysis or need_ekg or need_uzi):
        need_doctor = (
            "прием" in sq
            or "приём" in sq
            or "консультац" in sq
            or "осмотр" in sq
            or bool(_extract_specialty_from_text(sq))
        )

    if not (need_analysis or need_ekg or need_uzi or need_doctor):
        return list(regions)

    out: list[dict[str, Any]] = []
    for row in regions:
        if not isinstance(row, dict):
            continue
        if need_analysis and not bool(row.get("analysis")):
            continue
        if need_ekg and not bool(row.get("ecg")):
            continue
        if need_uzi and not bool(row.get("usi")):
            continue
        if need_doctor and not bool(row.get("doctorService")):
            continue
        out.append(row)
    return out
