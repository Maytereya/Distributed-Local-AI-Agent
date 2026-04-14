"""Deterministic follow-up and contextual extraction helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import re
from typing import Any


DOCTOR_ANAPHORA_RE = re.compile(r"\b(его|него|нему|ним|он|у\s+него|у\s+него\s+же|у\s+неё|ее|её|она)\b", re.I)
_DOCTOR_FOLLOWUP_RE = re.compile(
    r"\b(доктор|врач|расписан|график|при(е|ё)м|слот|окн|чем\s+занима|о\s+нем|о\s+враче|инфо)\b",
    re.I,
)
_SERVICE_FOLLOWUP_RE = re.compile(
    r"\b(услуг|анализ|процедур|подготовк|цена|стоим|сколько|где|филиал|адрес|с\s+наркозом|без\s+наркоза)\b",
    re.I,
)
_SERVICE_VARIANT_RE = re.compile(r"\b(с\s+наркозом|без\s+наркоза|с\s+контрастом|без\s+контраста)\b", re.I)
_DATE_FILTER_RE = re.compile(
    r"\b("
    r"сегодня|завтра|послезавтра|"
    r"на\s+следующ(?:ей|ую)\s+неделе|"
    r"на\s+этой\s+неделе|"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\.\d{1,2}(?:\.\d{2,4})?"
    r")\b",
    re.I,
)
_TIME_FILTER_RE = re.compile(
    r"\b("
    r"утром|дн[её]м|вечером|"
    r"после\s+\d{1,2}(?::\d{2})?|"
    r"до\s+\d{1,2}(?::\d{2})?|"
    r"в\s+\d{1,2}:\d{2}|"
    r"\d{1,2}:\d{2}"
    r")\b",
    re.I,
)
_SHORT_BRANCH_RE = re.compile(r"^\s*(?:а\s+)?(?:на|в)\s+([^?.!,]+?)\s*[?!.]?\s*$", re.I)
_BRANCH_CAPTURE_RE = re.compile(
    r"\b(?:филиал(?:е|ом)?|адрес(?:е|ом)?)(?:\s+на)?\s+([^?.!,]+?)(?:\s*[?!.]|$)",
    re.I,
)
_TIME_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_AFTER_RE = re.compile(r"\bпосле\s+(\d{1,2})(?::(\d{2}))?\b", re.I)
_TIME_BEFORE_RE = re.compile(r"\bдо\s+(\d{1,2})(?::(\d{2}))?\b", re.I)
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
_BRANCH_FOLLOWUP_STOPWORDS = {
    "сегодня",
    "завтра",
    "послезавтра",
    "утром",
    "вечером",
    "днем",
    "днём",
    "следующей",
    "следующую",
    "этой",
    "неделе",
    "неделю",
    "понедельник",
    "вторник",
    "среду",
    "четверг",
    "пятницу",
    "субботу",
    "воскресенье",
}


def looks_like_doctor_followup_message(*, user_message: str, remembered_doctor: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    remembered = str(remembered_doctor or "").strip()
    if not remembered:
        return False
    if DOCTOR_ANAPHORA_RE.search(text):
        return True
    if _DOCTOR_FOLLOWUP_RE.search(text):
        return True
    lowered = text.lower().replace("ё", "е")
    remembered_parts = [part.strip().lower().replace("ё", "е") for part in remembered.split() if part.strip()]
    if not remembered_parts:
        return False
    if len(lowered.split()) <= 3:
        return any(part in lowered for part in remembered_parts)
    return False


def looks_like_service_followup_message(*, user_message: str, remembered_service: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    remembered = str(remembered_service or "").strip()
    if not remembered:
        return False
    if _SERVICE_FOLLOWUP_RE.search(text):
        return True
    if _SERVICE_VARIANT_RE.search(text):
        return True
    if extract_contextual_entities(text):
        return True
    lowered = text.lower().replace("ё", "е")
    remembered_parts = [part.strip().lower().replace("ё", "е") for part in remembered.split() if part.strip()]
    if len(lowered.split()) <= 3:
        return any(part in lowered for part in remembered_parts)
    return False


def looks_like_contextual_clinical_followup(
    *,
    user_message: str,
    remembered_doctor: str,
    memory_entities: dict[str, Any],
) -> bool:
    if not memory_entities and not str(remembered_doctor or "").strip():
        return False
    if extract_contextual_entities(user_message):
        return True
    remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
    if remembered_service and looks_like_service_followup_message(
        user_message=user_message,
        remembered_service=remembered_service,
    ):
        return True
    return looks_like_doctor_followup_message(
        user_message=user_message,
        remembered_doctor=remembered_doctor,
    )


def contextual_followup_tool_plan(
    *,
    user_message: str,
    remembered_doctor: str,
    memory_entities: dict[str, Any],
) -> list[str]:
    contextual_entities = extract_contextual_entities(user_message)
    has_schedule_filters = bool(
        {"branch_name", "city", "date", "date_from", "date_to", "time", "time_from", "time_to"}
        & set(contextual_entities.keys())
    )
    remembered_doctor = str(remembered_doctor or memory_entities.get("doctor_name") or "").strip()
    remembered_service = str(memory_entities.get("service_name") or memory_entities.get("test_name") or "").strip()
    if remembered_doctor and has_schedule_filters:
        return ["doctors_schedule_week", "doctors_info"]
    if remembered_service:
        if "branch_name" in contextual_entities:
            return ["address_info", "service_bundle_info"]
        if "service_variant" in contextual_entities:
            return ["service_bundle_info", "price_info", "address_info"]
    return []


def extract_contextual_entities(user_message: str) -> dict[str, Any]:
    text = str(user_message or "").strip()
    if not text:
        return {}
    out: dict[str, Any] = {}
    variant = extract_service_variant(text)
    if variant:
        out["service_variant"] = variant
    branch_name = extract_branch_reference(text)
    if branch_name:
        out["branch_name"] = branch_name
    city = extract_city_reference(text)
    if city:
        out["city"] = city
    out.update(extract_date_filters(text))
    out.update(extract_time_filters(text))
    return out


def extract_service_variant(text: str) -> str:
    match = _SERVICE_VARIANT_RE.search(str(text or ""))
    return str(match.group(1) or "").strip().lower() if match else ""


def extract_city_reference(text: str) -> str:
    if re.search(r"\bсамар\w*\b", str(text or ""), re.I):
        return "Самара"
    return ""


def extract_branch_reference(text: str) -> str:
    def _clean_branch_candidate(value: str) -> str:
        candidate = str(value or "").strip(" ,.")
        if not candidate:
            return ""
        candidate = re.split(
            r"\b(утром|дн[её]м|вечером|сегодня|завтра|послезавтра|после\s+\d{1,2}(?::\d{2})?|до\s+\d{1,2}(?::\d{2})?|\d{1,2}:\d{2})\b",
            candidate,
            maxsplit=1,
            flags=re.I,
        )[0].strip(" ,.")
        return candidate

    source = str(text or "").strip()
    if not source:
        return ""
    lowered = source.lower()
    if "другом филиале" in lowered or "другой филиал" in lowered:
        return "другой филиал"
    explicit = _BRANCH_CAPTURE_RE.search(source)
    if explicit:
        candidate = _clean_branch_candidate(str(explicit.group(1) or ""))
        if candidate:
            return candidate
    short = _SHORT_BRANCH_RE.match(source)
    if not short:
        return ""
    candidate = _clean_branch_candidate(str(short.group(1) or ""))
    if not candidate:
        return ""
    low_candidate = candidate.lower()
    if low_candidate in _BRANCH_FOLLOWUP_STOPWORDS:
        return ""
    if _DATE_FILTER_RE.search(candidate) or _TIME_FILTER_RE.search(candidate):
        return ""
    if len(candidate.split()) > 5:
        return ""
    return candidate


def extract_date_filters(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source = str(text or "").strip()
    if not source:
        return out
    today = datetime.now().date()
    lowered = source.lower()
    if "послезавтра" in lowered:
        target = today + timedelta(days=2)
        iso = target.isoformat()
        return {"date": iso, "date_from": iso, "date_to": iso}
    if "завтра" in lowered:
        target = today + timedelta(days=1)
        iso = target.isoformat()
        return {"date": iso, "date_from": iso, "date_to": iso}
    if "сегодня" in lowered:
        iso = today.isoformat()
        return {"date": iso, "date_from": iso, "date_to": iso}
    if "на следующей неделе" in lowered or "на следующую неделю" in lowered:
        return {"date": "next_week"}
    if "на этой неделе" in lowered:
        return {"date": "this_week"}
    iso_match = _DATE_ISO_RE.search(source)
    if iso_match:
        iso = f"{int(iso_match.group(1)):04d}-{int(iso_match.group(2)):02d}-{int(iso_match.group(3)):02d}"
        return {"date": iso, "date_from": iso, "date_to": iso}
    dot_match = _DATE_DOT_RE.search(source)
    if dot_match:
        day = int(dot_match.group(1))
        month = int(dot_match.group(2))
        year_raw = str(dot_match.group(3) or "").strip()
        year = today.year
        if year_raw:
            year = int(year_raw)
            if year < 100:
                year += 2000
        try:
            target = date(year, month, day)
        except Exception:
            return out
        if not year_raw and target < today:
            try:
                target = date(year + 1, month, day)
            except Exception:
                return out
        iso = target.isoformat()
        return {"date": iso, "date_from": iso, "date_to": iso}
    return out


def extract_time_filters(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source = str(text or "").strip()
    if not source:
        return out
    lowered = source.lower()
    if "утром" in lowered:
        out["time"] = "утром"
        out["time_from"] = "08:00"
        out["time_to"] = "12:00"
    elif "днем" in lowered or "днём" in lowered:
        out["time"] = "днем"
        out["time_from"] = "12:00"
        out["time_to"] = "17:00"
    elif "вечером" in lowered:
        out["time"] = "вечером"
        out["time_from"] = "17:00"
        out["time_to"] = "21:00"
    after_match = _TIME_AFTER_RE.search(source)
    if after_match:
        minutes = int(after_match.group(2) or 0)
        out["time_from"] = f"{int(after_match.group(1)):02d}:{minutes:02d}"
        out["time"] = out["time_from"]
    before_match = _TIME_BEFORE_RE.search(source)
    if before_match:
        minutes = int(before_match.group(2) or 0)
        out["time_to"] = f"{int(before_match.group(1)):02d}:{minutes:02d}"
    exact_match = _TIME_HHMM_RE.search(source)
    if exact_match:
        exact = f"{int(exact_match.group(1)):02d}:{int(exact_match.group(2)):02d}"
        out["time"] = exact
        out["time_from"] = exact
    return out
