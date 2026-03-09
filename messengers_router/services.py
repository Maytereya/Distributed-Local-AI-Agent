"""Сервисный слой интеграций для роутера пациентов.

Содержит вызовы внешних источников (Nayka API, price, meili), кэш врачей,
поиск расписания/цен/адресов и fallback-контракты для handoff при сбоях.

Ответственность модуля:
1) Доступ к внешним данным и их нормализация к стабильному внутреннему формату.
2) Локальный кэш/дедупликация/ограничение объема данных для рендера.
3) Прозрачный graceful degradation (note/reason/handoff flags) при сбоях.

Модуль не должен принимать state-machine решения по диалогу.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from urllib.parse import quote_from_bytes
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.doctor_name_matching import (
    extract_doctor_name_candidate,
    resolve_schedule_surname,
    surname_variants,
)
from agent_logic_2.nayka_api import api_nayka, api_price
from converters import html_cleaner

_ADDRESS_HINT_RE = re.compile(
    r"\b(ул\.?|улица|пр\.?|проспект|пр-?т|тракт|б-р|бульвар|шоссе|пер\.?|переулок|наб\.?|площадь|дом|д\.|корп\.?|к\.|пом\.?)\b",
    re.I,
)
_SCHEDULE_QUERY_RE = re.compile(r"\b(расписани\w*|график|когда\b.*\bпринима\w*|принима\w*)\b", re.I)
_FIO_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]{2,}")
_PHONE_EXTRACT_RE = re.compile(r"\+?\d[\d\-\s\(\)]{7,}\d")
_PRICE_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
_PRICE_HOMECODE_DOTTED_RE = re.compile(r"\b\d+(?:\.\d+){1,6}\b")
_PRICE_HOMECODE_NUM_RE = re.compile(r"\b\d{4,}\b")
_DOCTOR_PRICE_HINT_RE = re.compile(r"\b(?:у|врач\w*|доктор\w*)\s+[а-яё\-]{3,}\b", re.I)
_PRICE_REQUEST_RE = re.compile(r"\b(стоим\w*|цен\w*|сколько)\b", re.I)
_PRICE_CONSULT_HINT_RE = re.compile(r"\b(при[её]м\w*|консультаци\w*)\b", re.I)
_PRICE_SERVICE_PREFIX_RE = re.compile(
    r"^\s*(?:а\s+)?(?:сколько\s+стоит|сколько\s+будет\s+стоить|цена|стоимость)\s+",
    re.I,
)
_PRICE_DOCTOR_SUFFIX_RE = re.compile(
    r"\bу\s+[а-яё\-]{3,}(?:\s+[а-яё\-]{2,}){0,2}\b.*$",
    re.I,
)
_NONBOOKABLE_POINTS_PATH = Path(__file__).resolve().parent / "data" / "nonbookable_points.json"
_NEAREST_HINT_RE = re.compile(r"\b(ближайш\w*|сам\w*\s+ранн\w*|раньше|поскорее|свободн\w*\s+окн\w*)\b", re.I)
_UZI_QUERY_RE = re.compile(r"\b(узи|узист|ультразвук\w*|ультразвуков\w*)\b", re.I)
_UZI_LINE_RE = re.compile(r"\b(узи|ультразвук\w*|ультразвуков\w*)\b", re.I)
_CITY_PREFIX_RE = re.compile(r"\b(?:г|город)\.?\s*([а-яёa-z\-]+)\b", re.I)
_UZI_FALSE_POSITIVE_RE = re.compile(
    r"\b(под\s+контролем\s+узи|во\s+время\s+консультативн\w*\s+при(е|ё)м\w*|"
    r"в\s+рамках\s+при(е|ё)м\w*|интерпретац\w*|разъяснен\w*)\b",
    re.I,
)
_SPECIALTY_CANONICAL = (
    "гастроэнтеролог",
    "эндокринолог",
    "офтальмолог",
    "дерматолог",
    "кардиолог",
    "невролог",
    "проктолог",
    "травматолог",
    "аллерголог",
    "ревматолог",
    "пульмонолог",
    "гинеколог",
    "терапевт",
    "педиатр",
    "уролог",
    "онколог",
    "хирург",
    "ортопед",
    "лор",
)
_SPECIALTY_RE = re.compile(
    r"\b(" + "|".join(re.escape(x) for x in _SPECIALTY_CANONICAL) + r")\w*\b",
    re.I,
)
# Важно: для /priceByRegion нужен city-level regionId (Самара = 3),
# а для /doctorServicePricesByRegion используются branch-level regionId из doctorRegions.
SAMARA_PRICE_REGION_ID = 3
_PRICE_QUERY_STOPWORDS = {
    "сколько",
    "стоит",
    "стоимость",
    "цена",
    "цена",
    "на",
    "в",
    "по",
    "у",
    "для",
    "и",
    "или",
    "услуга",
    "услуги",
    "процедура",
    "процедуры",
    "анализ",
    "анализы",
    "врач",
    "врача",
    "доктор",
    "доктора",
    "самара",
}


def _normalise_input(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _is_samara_city_value(value: str | None) -> bool:
    if not value:
        return False
    norm = _normalise_input(value).replace("ё", "е")
    return norm == "самара"


def _is_non_samara_city_value(value: str | None) -> bool:
    if not value:
        return False
    return not _is_samara_city_value(value)


def _normalize_region_text(value: str) -> str:
    norm = _normalise_input(value).replace("ё", "е")
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


def _extract_specialty_from_text(text: str) -> str:
    if _UZI_QUERY_RE.search(text or ""):
        return "узи"
    m = _SPECIALTY_RE.search(text or "")
    if not m:
        return ""
    return str(m.group(1) or "").strip().lower().replace("ё", "е")


def _has_nearest_hint(text: str) -> bool:
    return bool(_NEAREST_HINT_RE.search(text or ""))


def _is_uzi_query_text(text: str) -> bool:
    return bool(_UZI_QUERY_RE.search(text or ""))


def _split_spec_lines(spec_text: str) -> list[str]:
    return [ln.strip(" \t•-") for ln in str(spec_text or "").splitlines() if ln.strip()]


def _matches_uzi_doctor_profile(doc: dict[str, Any]) -> bool:
    spec_text = str(doc.get("specialization") or "")
    if not spec_text:
        return False

    units_text = " ".join(str(x or "") for x in (doc.get("units") or []))
    units_norm = _normalise_input(units_text)
    if "ультразвук" in units_norm or re.search(r"\bузи\b", units_norm):
        return True

    for raw_line in _split_spec_lines(spec_text):
        line = _normalise_input(raw_line)
        if not line or not _UZI_LINE_RE.search(line):
            continue
        if _UZI_FALSE_POSITIVE_RE.search(line):
            continue
        if "врач ультразвуковой диагностики" in line or "ультразвуков" in line:
            return True
        if line.startswith("узи "):
            return True
    return False


def _iter_slot_datetimes(schedule: dict[str, Any]) -> list[datetime]:
    out: list[datetime] = []
    if not isinstance(schedule, dict):
        return out
    for days in schedule.values():
        if not isinstance(days, list):
            continue
        for day in days:
            if not isinstance(day, dict):
                continue
            day_date = str(day.get("date") or day.get("curDate") or "").strip()
            if not day_date:
                continue
            slots = day.get("slots") or []
            if not isinstance(slots, list):
                continue
            for slot in slots:
                t = str(slot or "").strip()
                if len(t) < 5:
                    continue
                try:
                    out.append(datetime.fromisoformat(f"{day_date}T{t[:5]}:00"))
                except Exception:
                    continue
    return out


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


def _fio_tokens(text: str) -> list[str]:
    return [t.lower() for t in _FIO_TOKEN_RE.findall(str(text or ""))]


def _doctor_matches_fio(fio: str, doctor_query: str, resolved_surname: str | None = None) -> bool:
    """
    Проверка фамилии/ФИО ТОЛЬКО по fio врача.
    Не ищем по specialization, чтобы "Ким" не матчился на "хроническим".
    """
    tokens = _fio_tokens(fio)
    if not tokens:
        return False

    candidates: list[str] = []
    if resolved_surname:
        candidates.extend([v for v in surname_variants(resolved_surname) if v])
    else:
        q_tokens = _fio_tokens(doctor_query)
        if q_tokens:
            candidates.extend([v for v in surname_variants(q_tokens[0]) if v])

    normalized = sorted({ _normalise_input(x) for x in candidates if len(_normalise_input(x)) >= 2 }, key=len, reverse=True)
    if not normalized:
        return False

    for token in tokens:
        for c in normalized:
            if token.startswith(c):
                return True
    return False


def _compact_specialization(
    text: str,
    max_lines: int | None = None,
    max_chars: int | None = None,
) -> str:
    """
    Сжимает только дубли строк в специализации.
    Ограничения по строкам/символам отключены по умолчанию (полный текст),
    но могут быть включены параметрами max_lines/max_chars.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if not ln:
            continue
        key = re.sub(r"\s+", " ", ln).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
        if isinstance(max_lines, int) and max_lines > 0 and len(out) >= max_lines:
            break

    compact = "\n".join(out).strip()
    if isinstance(max_chars, int) and max_chars > 0 and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


def _dedupe_doctors_by_fio(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in docs:
        fio = _normalise_input(str(d.get("fio") or ""))
        if not fio or fio in seen:
            continue
        seen.add(fio)
        out.append(d)
    return out


def _extract_result_query_fields(entities: dict[str, Any], query: str) -> dict[str, Any]:
    surname = _get_first_present(entities, ["surname", "result_surname"])
    filial = _get_first_present(entities, ["filial", "result_filial"])
    year_raw = entities.get("year")
    number_raw = entities.get("number")

    if number_raw is None:
        number_raw = entities.get("order_id")

    year = _as_int(year_raw)
    number = _as_int(number_raw)
    return {
        "surname": str(surname or "").strip(),
        "year": year,
        "filial": str(filial or "").strip(),
        "number": number,
        "lang": _get_first_present(entities, ["lang", "result_lang"]) or "ru",
    }


def _cp1251_urlencode(value: str) -> str:
    """
    Кодирование параметров под контракт ссылки naykalab/getanaliz:
    Windows-1251 + URL-encode.
    """
    raw = str(value or "").strip().encode("cp1251", errors="replace")
    return quote_from_bytes(raw, safe="")


def _build_public_result_link(fields: dict[str, Any]) -> str | None:
    surname = str(fields.get("surname") or "").strip()
    filial = str(fields.get("filial") or "").strip()
    year = _as_int(fields.get("year"))
    number = _as_int(fields.get("number"))
    if not surname or not filial or year is None or number is None:
        return None
    return (
        "https://naykalab.ru/getanaliz.php"
        f"?fam={_cp1251_urlencode(surname)}"
        f"&year={year}"
        f"&nom={_cp1251_urlencode(filial)}"
        f"&nom2={number}"
        "&fast=1"
    )


def _region_display_name(region: dict[str, Any]) -> str:
    """
    Берем человекочитаемый адрес из live /regions.
    Приоритет: addressForSite -> name.
    """
    addr = str(region.get("addressForSite") or "").strip()
    name = str(region.get("name") or "").strip()
    return addr or name


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


def _service_query_matches(service_q: str, service_name: str) -> bool:
    sq = _normalise_input(service_q)
    sn = _normalise_input(service_name)
    if not sq or not sn:
        return False
    if sq in sn:
        return True
    if "анализ" in sq or "лаборатор" in sq:
        analysis_tokens = (
            "анализ",
            "лаборатор",
            "биоматериал",
            "взятие",
            "кров",
            "моч",
            "мазок",
            "сыворот",
            "плазм",
        )
        return any(tok in sn for tok in analysis_tokens)
    if sq == "экг":
        return "экг" in sn or "электрокардиограм" in sn
    return False


def _extract_homecode_query(text: str) -> str:
    s = _normalise_input(text)
    m = _PRICE_HOMECODE_DOTTED_RE.search(s)
    if m:
        return str(m.group(0)).strip()
    m = _PRICE_HOMECODE_NUM_RE.search(s)
    if m:
        return str(m.group(0)).strip()
    return ""


def _price_query_tokens(text: str) -> list[str]:
    s = _normalise_input(text).replace("ё", "е")
    out: list[str] = []
    for t in _PRICE_TOKEN_RE.findall(s):
        token = str(t or "").strip().lower().replace("ё", "е")
        if len(token) < 2:
            continue
        if token in _PRICE_QUERY_STOPWORDS:
            continue
        out.append(token)
        # "прием" и "консультация" считаем взаимозаменяемыми для ранжирования цен.
        if token.startswith("прием") and "консультац" not in out:
            out.append("консультац")
        elif token.startswith("консультац") and "прием" not in out:
            out.append("прием")
    return out


def _extract_price_service_from_query(query: str) -> str | None:
    raw = str(query or "").strip()
    if not raw:
        return None
    if _PRICE_CONSULT_HINT_RE.search(raw):
        return "прием"
    q = _normalise_input(raw)
    q = _PRICE_DOCTOR_SUFFIX_RE.sub("", q).strip(" ?!.,;:")
    q = _PRICE_SERVICE_PREFIX_RE.sub("", q).strip(" ?!.,;:")
    if not q:
        return None
    if q in {"цена", "стоимость"}:
        return None
    words = [w for w in q.split() if w]
    if not words:
        return None
    # Ограничиваем длину candidate, чтобы не тянуть в ranking целый диалог.
    return " ".join(words[:8])


def _price_row_score(row: dict[str, Any], *, query: str, tokens: list[str], homecode_query: str) -> tuple[int, int]:
    name = _normalise_input(str(row.get("serviceName") or row.get("name") or "")).replace("ё", "е")
    homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
    if not name:
        return 0, 0

    score = 0
    if homecode_query:
        if homecode == homecode_query:
            score += 260
        elif homecode_query in homecode:
            score += 180

    if query:
        if query == name:
            score += 220
        elif query in name:
            score += 150

    matched = 0
    if tokens:
        for tok in tokens:
            if tok in name:
                matched += 1
        score += matched * 25
        if matched == len(tokens):
            score += 80
        elif matched >= max(2, len(tokens) - 1):
            score += 40

    # Слегка понижаем заведомо нерелевантный общий тариф.
    if "выезд на дом" in name and not any(tok in name for tok in tokens):
        score -= 30

    return score, matched


def _rank_price_rows(rows: list[dict[str, Any]], query_text: str, *, limit: int = 10) -> list[dict[str, Any]]:
    query = _normalise_input(query_text).replace("ё", "е")
    tokens = _price_query_tokens(query_text)
    homecode_query = _extract_homecode_query(query_text)

    scored: list[tuple[int, int, int, int, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        score, matched = _price_row_score(row, query=query, tokens=tokens, homecode_query=homecode_query)
        if score <= 0:
            continue
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        name_gap = abs(len(name) - len(query)) if query else len(name)
        cost = _as_int(row.get("cost")) or 0
        scored.append((score, matched, -name_gap, -cost, row))

    if not scored:
        if query:
            fallback = [r for r in rows if isinstance(r, dict) and query in _normalise_input(str(r.get("serviceName") or ""))]
            if fallback:
                return fallback[:limit]
        if homecode_query:
            fallback = [
                r
                for r in rows
                if isinstance(r, dict)
                and homecode_query in _normalise_input(str(r.get("serviceHomecode") or r.get("homecode") or ""))
            ]
            if fallback:
                return fallback[:limit]
        return []

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, _, _, row in scored:
        name = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
        code = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
        key = (name, code)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _doctor_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    try:
        ord_value = int(doc.get("ord"))
    except Exception:
        ord_value = 10**9
    fio = _normalise_input(str(doc.get("fio") or ""))
    return ord_value, fio


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
    return ""


def _norm_city(s: str) -> str:
    t = _normalise_input(s or "")
    t = t.replace("ё", "е")
    t = re.sub(r"^г\.?\s*", "", t)
    return t.strip()


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


def _filter_regions_by_service_flags(regions: list[dict[str, Any]], service_q: str) -> list[dict[str, Any]]:
    sq = _normalise_input(service_q or "")
    if not sq:
        return list(regions)

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


def _is_meili_error_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return "Ошибка поисковой системы" in text or "Meilisearch" in text


@dataclass
class Services:
    """
    Сервисный слой для patient/messenger router.

    - doctors_info: использует ФАЙЛОВЫЙ кэш врачей (JSONL) + in-memory кэш.
    - doctors_schedule_week: всегда ходит в API (реалтайм расписание).
    """

    # in-memory кэш врачей
    # Приходится использовать идентификатор полей "field" и его свойство default_factory=list из @dataclass, так как list
    # относится к изменяемым типам данных. В противном случае переменная спика будет
    # рандомно перезаписываться в неожиданных местах (база).
    _doctors_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    # Тут неизменяемый тип str, не усложняем:
    _doctors_cache_path: Optional[str] = field(default=None, init=False)
    _doctors_cache_loaded_at: float = field(default=0.0, init=False)
    _regions_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _regions_cache_loaded_at: float = field(default=0.0, init=False)

    # блокировка, чтобы несколько запросов параллельно не перегенерировали кэш
    _doctors_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _regions_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # TTL in-memory кэша (латентность, сек, определят свежесть кэша)
    doctors_mem_ttl_seconds: int = 300
    regions_mem_ttl_seconds: int = 300

    # -----------------------------
    # Low-level helpers (async)
    # -----------------------------

    def ensure_background_refresh_started(self) -> None:
        """Запускает фоновые refresh-задачи кэшей (idempotent)."""
        try:
            api_price.ensure_daily_price_refresh_started()
        except Exception:
            pass

    async def _ensure_doctors_cache_loaded(self) -> list[dict[str, Any]]:
        """
        1) Пытаемся получить свежий doctor cache через api_nayka.get_cached_doctors_data()
        2) При неуспехе откатываемся на последний непустой файл
        3) Держим in-memory-кэш поверх файлового кэша
        """
        now = time.time()

        # быстрый путь: ещё не протухло
        if self._doctors_cache and (now - self._doctors_cache_loaded_at) < self.doctors_mem_ttl_seconds:
            return self._doctors_cache

        async with self._doctors_lock:
            # повторная проверка под локом
            now = time.time()
            if self._doctors_cache and (now - self._doctors_cache_loaded_at) < self.doctors_mem_ttl_seconds:
                return self._doctors_cache

            try:
                doctors_loaded = await asyncio.to_thread(api_nayka.get_cached_doctors_data)
                file_path = await asyncio.to_thread(api_nayka.find_existing_doctors_file)

                # Миграция старого кеша: ранние JSONL могли не содержать ord и
                # могли хранить placeholder-адреса вида "ID 8502".
                schema_outdated = False
                if doctors_loaded:
                    preview = doctors_loaded[:20]
                    has_ord = any(isinstance(row, dict) and "ord" in row for row in preview)
                    has_placeholder_region = any(
                        isinstance(row, dict)
                        and any(
                            str(addr).strip().startswith(("ID ", "[ID "))
                            for addr in (row.get("regions") or [])
                        )
                        for row in preview
                    )
                    schema_outdated = (not has_ord) or has_placeholder_region

                if schema_outdated:
                    doctors = await asyncio.to_thread(api_nayka.get_all_doctors)
                    await asyncio.to_thread(api_nayka.save_doctors_data, doctors)
                    file_path = await asyncio.to_thread(api_nayka.find_existing_doctors_file)
                    doctors_loaded = []
                    if file_path is not None:
                        doctors_loaded = await asyncio.to_thread(api_nayka.load_doctors_data, file_path)
            except Exception:
                return self._doctors_cache or []

            self._doctors_cache = doctors_loaded
            self._doctors_cache_path = str(file_path) if file_path is not None else None
            self._doctors_cache_loaded_at = time.time()
            return self._doctors_cache

    async def _ensure_regions_loaded(self) -> list[dict[str, Any]]:
        """
        Live-список регионов/филиалов из Nayka API (/regions) с коротким in-memory TTL.
        """
        now = time.time()
        if self._regions_cache and (now - self._regions_cache_loaded_at) < self.regions_mem_ttl_seconds:
            return self._regions_cache

        async with self._regions_lock:
            now = time.time()
            if self._regions_cache and (now - self._regions_cache_loaded_at) < self.regions_mem_ttl_seconds:
                return self._regions_cache

            try:
                regions = await asyncio.to_thread(api_nayka.site_regions)
                if not isinstance(regions, list):
                    regions = []
            except Exception:
                regions = []

            self._regions_cache = regions
            self._regions_cache_loaded_at = time.time()
            return self._regions_cache

    async def _samara_region_tokens(self) -> set[str]:
        regions = await self._ensure_regions_loaded()
        tokens: set[str] = set()
        for r in regions:
            if not isinstance(r, dict):
                continue
            city = str(r.get("city") or "").strip()
            name = str(r.get("name") or "").strip()
            addr = str(r.get("addressForSite") or "").strip()
            if not (
                _is_samara_city_value(city)
                or "самара" in _normalise_input(name)
                or "самара" in _normalise_input(addr)
            ):
                continue
            for raw in (name, addr):
                n = _normalise_input(raw)
                if n:
                    tokens.add(n)
        return tokens

    async def _schedule_by_specialty(
        self,
        specialty: str,
        entities: dict[str, Any],
        *,
        nearest_only: bool,
    ) -> list[dict[str, Any]]:
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return []
        spec = _normalise_input(specialty)
        if not spec:
            return []

        is_uzi_query = _is_uzi_query_text(spec)
        samara_tokens = await self._samara_region_tokens()
        candidates = sorted(
            [
            d for d in doctors
            if (
                _matches_uzi_doctor_profile(d)
                if is_uzi_query
                else spec in _normalise_input(str(d.get("specialization") or ""))
            )
            and not _has_explicit_non_samara_regions([str(x) for x in (d.get("regions") or []) if str(x).strip()])
            and (
                not samara_tokens
                or any(
                    _region_matches_samara_tokens(str(x), samara_tokens)
                    for x in (d.get("regions") or [])
                    if str(x).strip()
                )
            )
            ],
            key=_doctor_sort_key,
        )[:8]
        if not candidates:
            return []

        out_rows: list[dict[str, Any]] = []
        for doc in candidates:
            fio = str(doc.get("fio") or "").strip()
            if not fio:
                continue
            surname = fio.split()[0]
            try:
                data = await asyncio.to_thread(api_nayka.find_doctor_schedule, surname)
            except Exception:
                continue
            if not isinstance(data, list) or not data:
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                row_fio = str(row.get("fio") or "").strip()
                if row_fio and _normalise_input(row_fio) != _normalise_input(fio):
                    continue
                item = dict(row)
                item["specialization"] = _compact_specialization(str(item.get("specialization") or ""))
                slots = _iter_slot_datetimes(item.get("schedule") or {})
                if slots:
                    item["_nearest_slot"] = min(slots)
                out_rows.append(item)
                break

        if not out_rows:
            return []
        with_slots = [x for x in out_rows if isinstance(x.get("_nearest_slot"), datetime)]
        if with_slots:
            with_slots.sort(key=lambda x: x["_nearest_slot"])  # type: ignore[index]
            chosen = with_slots[:1] if nearest_only else with_slots[:3]
        else:
            chosen = out_rows[:1] if nearest_only else out_rows[:3]

        for item in chosen:
            item.pop("_nearest_slot", None)
        return chosen

    # -----------------------------
    # NAUKA API used by router
    # -----------------------------

    async def resolve_doctor_name(self, raw_text_or_name: str) -> str | None:
        """
        Валидация кандидата фамилии/ФИО по актуальному кэшу врачей.
        Возвращает каноническую фамилию только если удалось сопоставить с кэшем.
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return None
        value = str(raw_text_or_name or "").strip()
        if not value:
            return None
        resolved = resolve_schedule_surname(value, doctors)
        if resolved:
            return resolved

        candidate = extract_doctor_name_candidate(value, prefer_schedule=True)
        if candidate and _normalise_input(candidate) != _normalise_input(value):
            return resolve_schedule_surname(candidate, doctors)
        return None

    async def _resolve_doctor_id_from_name(self, raw_text_or_name: str) -> tuple[int | None, str | None]:
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return None, None

        raw = str(raw_text_or_name or "").strip()
        if not raw:
            return None, None

        resolved_surname = resolve_schedule_surname(raw, doctors)
        samara_tokens = await self._samara_region_tokens()
        matched: list[dict[str, Any]] = []
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            fio = str(doc.get("fio") or "").strip()
            if not fio:
                continue
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            if _has_explicit_non_samara_regions(raw_regions):
                continue
            if samara_tokens and raw_regions and not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
            if _doctor_matches_fio(fio, raw, resolved_surname):
                matched.append(doc)

        if not matched:
            return None, None
        matched = sorted(matched, key=_doctor_sort_key)
        first = matched[0]
        return _as_int(first.get("id")), str(first.get("fio") or "").strip() or None

    async def doctors_info(self, query: str, entities: dict[str, Any], output_max: int = 5) -> dict[str, Any]:
        """
        Возвращает список врачей из кэша (без real-time API).
        Фильтрация делается программно:
        - по фамилии/ФИО
        - по специализации
        - по региону/филиалу (по строкам regions/units если есть)
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return _service_fallback(
                note="doctors_info source unavailable",
                handoff_message="Сейчас не удалось получить список врачей автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"doctors": []},
            )

        q = _normalise_input(query)
        doctor_raw = _get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
        fio_q = _normalise_input(doctor_raw)
        spec_q = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
        if not spec_q:
            spec_q = _extract_specialty_from_text(query)
        region_q = _normalise_input(_get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")
        resolved_surname = resolve_schedule_surname(doctor_raw, doctors) if doctor_raw else None

        query_candidate = extract_doctor_name_candidate(query, prefer_schedule=True) if query else None
        query_resolved = resolve_schedule_surname(query_candidate, doctors) if query_candidate else None
        if not resolved_surname:
            resolved_surname = query_resolved

        # При поиске по специальности игнорируем "залипший" doctor_name из прошлого контекста.
        if spec_q and not query_resolved:
            fio_q = ""
            resolved_surname = None

        samara_tokens = await self._samara_region_tokens()

        # если из entities пусто — попробуем хотя бы query как ключ
        # (но аккуратно: не хотим показывать всех врачей по любому вопросу)
        keyword = ""
        if fio_q:
            keyword = fio_q
        elif spec_q:
            keyword = spec_q
        elif region_q:
            keyword = region_q
        else:
            keyword = q

        keyword = keyword.strip()
        is_uzi_query = _is_uzi_query_text(spec_q or keyword)

        def match_doc(doc: dict[str, Any], ) -> bool:
            fio = _normalise_input(str(doc.get("fio", "")))
            spec = _normalise_input(str(doc.get("specialization", "")))
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            regions = " ".join([_normalise_input(x) for x in raw_regions])
            units = " ".join([_normalise_input(str(x)) for x in (doc.get("units") or [])])

            hay = " | ".join([fio, spec, regions, units])
            # Даже без live /regions не допускаем в выдачу явно не-самарские площадки.
            if _has_explicit_non_samara_regions(raw_regions):
                return False
            if samara_tokens:
                if not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                    return False
            if fio_q:
                if not _doctor_matches_fio(fio, fio_q, resolved_surname):
                    return False
            if spec_q:
                if is_uzi_query:
                    if not _matches_uzi_doctor_profile(doc):
                        return False
                elif spec_q not in hay:
                    return False
            if region_q and region_q not in hay:
                return False

            # если ничего конкретного не задано — используем keyword, но требуем хотя бы 3 символа
            if not (fio_q or spec_q or region_q):
                if len(keyword) < 3:
                    return False
                return keyword in hay

            return True

        filtered = [d for d in doctors if match_doc(d)]
        filtered = _dedupe_doctors_by_fio(filtered)
        filtered = sorted(filtered, key=_doctor_sort_key)

        if resolved_surname:
            # при явной фамилии врача не раздуваем выдачу.
            output_max = min(output_max, 3)

        # ограничим размер, чтобы не отправлять сотни карточек в LLM
        # (далее LLM/рендерер красиво завернёт)
        # Определить сколько тут карточек нужно в выводе обычно
        filtered = filtered[:output_max]
        compact: list[dict[str, Any]] = []
        for d in filtered:
            row = dict(d)
            row["specialization"] = _compact_specialization(str(row.get("specialization") or ""))
            compact.append(row)

        return {
            "doctors": compact,
            "note": "doctors_info: from cached registry (jsonl)",
            "cache_file": self._doctors_cache_path,
            "entities_used": {
                "doctor_query": fio_q,
                "doctor_resolved": resolved_surname,
                "specialty_query": spec_q,
                "region_query": region_q,
            },
        }

    async def doctors_schedule_week(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Реалтайм расписание на неделю.

        Используем high-level функцию api_nayka.find_doctor_schedule(last_name, region_name=None)
        Она сама ходит в:
        - /doctors, /regions, /doctorRegions, /doctorCompanyUnits
        - /doctorSchedule + /doctorScheduleCells

        Возвращаем как есть (агрегированный список).
        """
        raw_name = _get_first_present(
            entities,
            ["last_name", "doctor_last_name", "doctor", "doctor_name", "fio"],
        )
        specialty = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
        if not specialty:
            specialty = _extract_specialty_from_text(query)
        if not raw_name:
            raw_name = query

        doctors = await self._ensure_doctors_cache_loaded()
        raw_for_match = str(raw_name or "").strip()
        # Если прилетело полное ФИО, для расписания берем фамилию (1-е слово),
        # иначе fuzzy-резолвер может схватить отчество и вернуть ложные матчи.
        if " " in raw_for_match:
            first = raw_for_match.split()[0].strip()
            raw_for_match = first or raw_for_match

        query_doctor_candidate = extract_doctor_name_candidate(str(query or ""), prefer_schedule=True) if query else None
        query_name = resolve_schedule_surname(str(query_doctor_candidate), doctors) if query_doctor_candidate else None
        last_name = resolve_schedule_surname(raw_for_match, doctors)
        # Если в текущей реплике явно фигурирует другая фамилия по расписанию,
        # приоритет отдаем ей (сброс от залипшего doctor_name из state).
        has_schedule_signal = bool(_SCHEDULE_QUERY_RE.search(str(query or "")))
        if query_name and (
            not last_name
            or has_schedule_signal
            or _normalise_input(str(query_name)) != _normalise_input(str(last_name))
        ):
            last_name = query_name
        elif not last_name and query and query != raw_name:
            last_name = query_name or resolve_schedule_surname(query, doctors)

        if not last_name and specialty:
            schedule_by_spec = await self._schedule_by_specialty(
                specialty,
                entities,
                nearest_only=_has_nearest_hint(query),
            )
            return {
                "schedule": schedule_by_spec,
                "note": "doctors_schedule_week: by specialty",
                "entities_used": {"specialty": specialty, "raw_name": raw_name},
            }

        if not last_name:
            return {
                "schedule": [],
                "note": "doctors_schedule_week: missing doctor last name",
                "entities_used": entities,
            }

        # необязательный фильтр региона/филиала/города
        region_name = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city", "branch_name"])
        if region_name and _is_non_samara_city_value(region_name):
            return _service_fallback(
                note=f"doctors_schedule_week unsupported city: {region_name}",
                handoff_message="Сейчас могу помочь только по Самаре. Соединяю с оператором.",
                entities=entities,
                reason="city_not_supported",
                extra={"schedule": []},
            )

        # api_nayka.find_doctor_schedule блокирующая (requests) — уводим в thread.
        # Пробуем несколько вариантов фамилии (родительный падеж -> именительный).
        data = None
        candidates = surname_variants(str(last_name))
        if not candidates:
            candidates = [str(last_name)]
        if query_name:
            for qv in surname_variants(str(query_name)):
                if qv not in candidates:
                    candidates.append(qv)
        try:
            for candidate in candidates:
                try:
                    data = await asyncio.to_thread(api_nayka.find_doctor_schedule, candidate, region_name)
                except TypeError:
                    data = await asyncio.to_thread(api_nayka.find_doctor_schedule, candidate)
                if isinstance(data, list) and data:
                    last_name = candidate
                    break
                # fallback: если регионный фильтр дал пусто, пробуем без региона
                if region_name:
                    try:
                        data = await asyncio.to_thread(api_nayka.find_doctor_schedule, candidate)
                    except Exception:
                        data = []
                    if isinstance(data, list) and data:
                        last_name = candidate
                        break
        except Exception:
            return _service_fallback(
                note="doctors_schedule_week unavailable",
                handoff_message="Сейчас не удалось получить расписание автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"schedule": []},
            )

        # Защита от чрезмерно длинных/дублирующихся specialization блоков в realtime API.
        if isinstance(data, list):
            compact_data: list[dict[str, Any]] = []
            samara_tokens = await self._samara_region_tokens()
            for row in data:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                item["specialization"] = _compact_specialization(str(item.get("specialization") or ""))
                regions_src = [str(x) for x in (item.get("regions") or []) if str(x).strip()]
                if _has_explicit_non_samara_regions(regions_src):
                    continue
                if samara_tokens:
                    if regions_src and not any(_region_matches_samara_tokens(x, samara_tokens) for x in regions_src):
                        continue
                    sched = item.get("schedule")
                    if isinstance(sched, dict) and sched:
                        sched_filtered: dict[str, Any] = {}
                        for k, v in sched.items():
                            if _region_matches_samara_tokens(str(k), samara_tokens):
                                sched_filtered[k] = v
                        if sched_filtered:
                            item["schedule"] = sched_filtered
                        elif regions_src:
                            continue
                compact_data.append(item)
            data = compact_data

        return {
            "schedule": data or [],
            "note": "doctors_schedule_week: realtime from Nayka API",
            "entities_used": {"last_name": last_name, "raw_name": raw_name, "region_name": region_name},
        }

    # Остальные методы пока как заглушки
    async def appointment_help(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        if query:
            try:
                raw = await asyncio.to_thread(meilisearch.search_meili, "main_index", query)
                cleaned = html_cleaner.strip_html(raw)
            except Exception:
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message="Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            if _is_meili_error_text(cleaned):
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message="Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            return {"instructions": cleaned, "entities_used": entities}
        return {
            "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
            "entities_used": entities,
        }

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        test_name = _get_first_present(entities, ["test_name", "service_name"]) or query
        needle = _normalise_input(test_name)
        if not needle:
            return {"tests": [], "promos": [], "note": "no test query", "entities_used": entities}

        try:
            price_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
        except Exception:
            return _service_fallback(
                note="test_assist source unavailable",
                handoff_message="Сейчас не удалось подобрать анализы автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"tests": [], "promos": []},
            )
        matches = _rank_price_rows([p for p in price_rows if isinstance(p, dict)], test_name, limit=10)

        return {
            "tests": matches,
            "promos": [],
            "note": f"test_assist: priceByRegion({SAMARA_PRICE_REGION_ID})",
            "entities_used": entities,
        }

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        q = _get_first_present(entities, ["test_name", "service_name"]) or query
        if not q:
            return {"prepare": "", "note": "no query", "entities_used": entities}
        try:
            raw = await asyncio.to_thread(meilisearch.search_meili, "main_index", q)
            cleaned = html_cleaner.strip_html(raw)
        except Exception:
            return _service_fallback(
                note="prepare source unavailable",
                handoff_message="Сейчас не удалось получить правила подготовки автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prepare": ""},
            )
        if _is_meili_error_text(cleaned):
            return _service_fallback(
                note="prepare source unavailable",
                handoff_message="Сейчас не удалось получить правила подготовки автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prepare": ""},
            )
        return {"prepare": cleaned, "entities_used": entities}

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        def _result_fallback(note: str, message: str = "Сейчас не удалось получить результаты автоматически. Соединяю с оператором.") -> dict[str, Any]:
            return _service_fallback(
                note=note,
                handoff_message=message,
                entities=entities,
                reason="test_result_fallback",
                extra={"ready": False},
            )

        fields = _extract_result_query_fields(entities, query)
        missing = [k for k in ("surname", "year", "filial", "number") if not fields.get(k)]
        if missing:
            return {
                "ready": False,
                "note": "missing_result_fields",
                "missing_fields": missing,
                "entities_used": entities,
            }

        try:
            api_resp = await asyncio.to_thread(
                api_nayka.site_result_for_patient,
                surname=fields["surname"],
                year=int(fields["year"]),
                filial=fields["filial"],
                number=int(fields["number"]),
                lang=fields["lang"],
                with_time=None,
            )
        except Exception as e:
            return _result_fallback(f"resultForPatient failed: {e}")

        if not isinstance(api_resp, dict) or not api_resp.get("ok"):
            return _result_fallback(f"resultForPatient error: {api_resp}")

        payload = api_resp.get("data")
        if not payload:
            return {
                "ready": False,
                "note": "result_not_found_or_not_ready",
                "result_payload": payload,
                "result_preview": "По указанным данным результаты пока не найдены или еще не готовы.",
                "entities_used": entities,
            }

        link = _build_public_result_link(fields)
        if not link:
            return _result_fallback(
                "result_link_build_failed",
                "Сейчас не удалось сформировать ссылку на результат автоматически. Соединяю с оператором.",
            )

        return {
            "ready": True,
            "note": "result_link_constructed",
            "result_payload": payload,
            "result_preview": "Ссылка на результат сформирована.",
            "result_links": [link],
            "entities_used": entities,
        }

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        doctor_id = _as_int(entities.get("doctor_id"))
        doctor_name = _get_first_present(entities, ["doctor_name", "doctor", "fio", "last_name", "doctor_last_name"]) or ""
        resolved_doctor_fio: str | None = None
        if not doctor_id and doctor_name:
            doctor_id, resolved_doctor_fio = await self._resolve_doctor_id_from_name(doctor_name)
        if not doctor_id and query and _DOCTOR_PRICE_HINT_RE.search(str(query or "")):
            # Fallback для фраз вида "сколько стоит ... у Белохвостиковой":
            # извлекаем врача из полного текста запроса, даже если classifier не выделил doctor_name.
            doctor_id, q_resolved_fio = await self._resolve_doctor_id_from_name(str(query))
            if doctor_id and q_resolved_fio:
                resolved_doctor_fio = q_resolved_fio
                doctor_name = q_resolved_fio
        entity_service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        query_service_name = _extract_price_service_from_query(query)
        service_name = query_service_name or entity_service_name or query
        # Если вопрос явно doctor-specific и сформулирован как новый price-запрос,
        # не тянем "залипшую" услугу из прошлого контекста.
        if (
            doctor_id
            and not query_service_name
            and query
            and _DOCTOR_PRICE_HINT_RE.search(str(query))
            and _PRICE_REQUEST_RE.search(str(query))
        ):
            service_name = query
        needle = _normalise_input(service_name)

        if doctor_id:
            # doctor prices: branch-level regionId из /doctorServicePricesByRegion cache
            try:
                prices = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception:
                return _service_fallback(
                    note="price_info source unavailable",
                    handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"prices": []},
                )
            doc_prices = [p for p in prices if _as_int(p.get("doctorId")) == doctor_id]
            if needle:
                doc_prices = _rank_price_rows(doc_prices, service_name, limit=10)
            if not needle:
                doc_prices = sorted(
                    [p for p in doc_prices if isinstance(p, dict)],
                    key=lambda p: (_normalise_input(str(p.get("serviceName") or "")), _as_int(p.get("cost")) or 0),
                )
            return {
                "prices": doc_prices[:10],
                "note": "price_info: doctorServicePricesByRegion (branch-level regionId)",
                "entities_used": {
                    **entities,
                    "doctor_id_resolved": doctor_id,
                    "doctor_name_resolved": resolved_doctor_fio or doctor_name or "",
                    "service_name_effective": service_name,
                },
            }

        # retail prices: city-level regionId в /priceByRegion/{cityRegionId}
        try:
            price_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
        except Exception:
            return _service_fallback(
                note="price_info source unavailable",
                handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prices": []},
            )
        if not needle:
            return {"prices": [], "note": "no service query", "entities_used": entities}
        matches = _rank_price_rows([p for p in price_rows if isinstance(p, dict)], service_name, limit=10)
        return {
            "prices": matches,
            "note": f"price_info: priceByRegion({SAMARA_PRICE_REGION_ID})",
            "entities_used": {
                **entities,
                "service_name_effective": service_name,
            },
        }

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        try:
            regions = await self._ensure_regions_loaded()
        except Exception:
            regions = []
        # Работаем только по Самаре.
        regions = [
            r for r in regions
            if isinstance(r, dict)
            and (
                _is_samara_city_value(str(r.get("city") or ""))
                or "самара" in _normalise_input(str(r.get("name") or ""))
                or "самара" in _normalise_input(str(r.get("addressForSite") or ""))
            )
        ]
        appointment_mode = bool(entities.get("__appointment_mode"))
        branch = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city"]) or ""
        if not branch:
            raw_query = str(query or "").strip()
            if raw_query and (_looks_like_real_address(raw_query) or _ADDRESS_HINT_RE.search(raw_query)):
                branch = raw_query
        branch_q = _normalise_input(branch)
        service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        service_q = _normalise_input(service_name)
        city_for_static = _get_first_present(entities, ["city"])
        if city_for_static and _is_non_samara_city_value(city_for_static):
            return _service_fallback(
                note=f"address_info unsupported city: {city_for_static}",
                handoff_message="Сейчас могу помочь только по Самаре. Соединяю с оператором.",
                entities=entities,
                reason="city_not_supported",
                extra={"addresses": [], "branches": []},
            )

        allowed_doctor_addresses_norm: set[str] = set()
        if appointment_mode:
            try:
                doctors = await self._ensure_doctors_cache_loaded()
            except Exception:
                doctors = []
            for d in doctors:
                if not isinstance(d, dict):
                    continue
                for addr in (d.get("regions") or d.get("addresses") or []):
                    a = str(addr).strip()
                    if not a or not _looks_like_real_address(a):
                        continue
                    allowed_doctor_addresses_norm.add(_normalise_input(a))

        if service_q:
            regions = _filter_regions_by_service_flags(regions, service_q)

        addresses: list[str] = []
        branches: list[dict[str, Any]] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            rid = _as_int(r.get("id"))
            disp = _region_display_name(r)
            if not disp:
                continue
            # в выдачу пациенту пускаем только реальные адреса филиалов
            if not _looks_like_real_address(disp):
                continue
            hay = " | ".join(
                [
                    _normalise_input(disp),
                    _normalise_input(str(r.get("name") or "")),
                    _normalise_input(str(r.get("city") or "")),
                ]
            )
            if branch_q and branch_q not in hay:
                continue
            addresses.append(disp)
            branches.append(
                {
                    "id": rid,
                    "address": disp,
                    "city": str(r.get("city") or "").strip(),
                    "phone": _extract_region_phone(r),
                    "work_time": _extract_region_work_time(r),
                }
            )

        uniq = sorted(set(addresses))
        if uniq:
            by_addr: dict[str, dict[str, Any]] = {}
            for b in branches:
                addr = str(b.get("address") or "").strip()
                if not addr:
                    continue
                prev = by_addr.get(addr)
                if prev is None:
                    by_addr[addr] = b
                    continue
                prev_score = int(bool(prev.get("phone"))) + int(bool(prev.get("work_time")))
                cur_score = int(bool(b.get("phone"))) + int(bool(b.get("work_time")))
                if cur_score > prev_score:
                    by_addr[addr] = b

            if appointment_mode and allowed_doctor_addresses_norm:
                def _is_doctor_capable(addr: str) -> bool:
                    n = _normalise_input(addr)
                    for x in allowed_doctor_addresses_norm:
                        if n == x or n in x or x in n:
                            return True
                    return False

                filtered_addrs = [a for a in uniq if _is_doctor_capable(a)]
                if filtered_addrs:
                    uniq = filtered_addrs
                    by_addr = {k: v for k, v in by_addr.items() if _is_doctor_capable(k)}

            note = "address_info: live regions API"
            if service_q:
                note += " + filtered by service flags"
            if appointment_mode:
                note += " + filtered by doctor-capable branches"
            return {
                "addresses": uniq,
                "branches": list(by_addr.values()),
                "note": note,
                "entities_used": entities,
            }

        # fallback: старый путь через кэш врачей
        try:
            doctors = await self._ensure_doctors_cache_loaded()
        except Exception:
            doctors = []
        fallback: list[str] = []
        for d in doctors:
            for addr in (d.get("regions") or d.get("addresses") or []):
                a = str(addr).strip()
                if not a:
                    continue
                if not _looks_like_real_address(a):
                    continue
                if branch_q and branch_q not in _normalise_input(a):
                    continue
                fallback.append(a)
        return {
            "addresses": sorted(set(fallback)),
            "branches": [{"address": a, "phone": "", "work_time": ""} for a in sorted(set(fallback))],
            "note": "address_info: doctors cache fallback",
            "entities_used": entities,
        }

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        try:
            hits = await asyncio.to_thread(
                meilisearch.search_news_active,
                index_name="news",
                keyword=query or None,
                limit=10,
                sort=["from_ts:desc"],
            )
        except Exception:
            # Для новостей деградация источника не критична: возвращаем пустой ответ
            # без принудительного handoff.
            return {
                "news": [],
                "note": "news source unavailable",
                "entities_used": entities,
            }
        return {"news": hits, "entities_used": entities}

    def get_branches(self) -> list[dict[str, str]]:
        """
        Возвращает справочник филиалов.
        Формат:
          [{"id":"branch_1","name":"Филиал на Проспекте Ленина","aliases":"Ленина, Ленинская,Проспект Ленина 5"}]
        Пока заглушка.
        """
        regions = self._regions_cache or []
        if not regions:
            try:
                data = api_nayka.site_regions()
                if isinstance(data, list):
                    regions = data
                    self._regions_cache = data
                    self._regions_cache_loaded_at = time.time()
            except Exception:
                regions = []
        out: list[dict[str, str]] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            if not (
                _is_samara_city_value(str(r.get("city") or ""))
                or "самара" in _normalise_input(str(r.get("name") or ""))
                or "самара" in _normalise_input(str(r.get("addressForSite") or ""))
            ):
                continue
            rid = r.get("id")
            disp = _region_display_name(r)
            if not disp:
                continue
            if not _looks_like_real_address(disp):
                continue
            bid = f"branch_{rid}" if rid is not None else f"branch_{len(out) + 1}"
            aliases = ", ".join(
                [
                    _normalise_input(disp),
                    _normalise_input(str(r.get("name") or "")),
                    _normalise_input(str(r.get("city") or "")),
                ]
            )
            out.append({"id": bid, "name": disp, "aliases": aliases})

        if out:
            # убираем дубли по имени
            uniq_by_name: dict[str, dict[str, str]] = {}
            for b in out:
                uniq_by_name.setdefault(b["name"], b)
            return list(uniq_by_name.values())

        return []


if __name__ == "__main__":
    async def main():
        s = Services()
        print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
        print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

    asyncio.run(main())
