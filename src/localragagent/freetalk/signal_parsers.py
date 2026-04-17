"""Shared deterministic signal parsers for FreeTalk."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import re


DOCTOR_ANAPHORA_RE = re.compile(r"\b(его|него|нему|ним|он|у\s+него|у\s+него\s+же|у\s+неё|ее|её|она)\b", re.I)
DOCTOR_FOLLOWUP_RE = re.compile(
    r"\b(доктор|врач|расписан|график|при(е|ё)м|слот|окн|чем\s+занима|о\s+нем|о\s+враче|инфо)\b",
    re.I,
)
SERVICE_FOLLOWUP_RE = re.compile(
    r"\b(услуг|анализ|процедур|подготовк|цена|стоим|сколько|где|филиал|адрес|с\s+наркозом|без\s+наркоза)\b",
    re.I,
)
SERVICE_VARIANT_RE = re.compile(r"\b(с\s+наркозом|без\s+наркоза|с\s+контрастом|без\s+контраста)\b", re.I)
DATE_FILTER_RE = re.compile(
    r"\b("
    r"сегодня|завтра|послезавтра|"
    r"на\s+следующ(?:ей|ую)\s+неделе|"
    r"на\s+этой\s+неделе|"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}\.\d{1,2}(?:\.\d{2,4})?"
    r")\b",
    re.I,
)
TIME_FILTER_RE = re.compile(
    r"\b("
    r"утром|дн[её]м|вечером|"
    r"после\s+\d{1,2}(?::\d{2})?|"
    r"до\s+\d{1,2}(?::\d{2})?|"
    r"в\s+\d{1,2}:\d{2}|"
    r"\d{1,2}:\d{2}"
    r")\b",
    re.I,
)
SHORT_BRANCH_RE = re.compile(r"^\s*(?:а\s+)?(?:на|в)\s+([^?.!,]+?)\s*[?!.]?\s*$", re.I)
BRANCH_CAPTURE_RE = re.compile(
    r"\b(?:филиал(?:е|ом)?|адрес(?:е|ом)?)(?:\s+на)?\s+([^?.!,]+?)(?:\s*[?!.]|$)",
    re.I,
)
TIME_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
TIME_AFTER_RE = re.compile(r"\bпосле\s+(\d{1,2})(?::(\d{2}))?\b", re.I)
TIME_BEFORE_RE = re.compile(r"\bдо\s+(\d{1,2})(?::(\d{2}))?\b", re.I)
DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
DATE_WORD_RE = re.compile(
    r"\b(?:на\s+)?(\d{1,2})\s+"
    r"(января|январь|февраля|февраль|марта|март|апреля|апрель|мая|май|июня|июнь|июля|июль|августа|август|"
    r"сентября|сентябрь|октября|октябрь|ноября|ноябрь|декабря|декабрь)"
    r"(?:\s+(\d{4}))?\b",
    re.I,
)
PERSON_NAME_RE = re.compile(
    r"^\s*([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[А-ЯЁA-Z][а-яёa-z\-]+){1,2})\s*[.!?]?\s*$"
)
CAPITALIZED_NAME_CANDIDATE_RE = re.compile(
    r"^\s*([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[А-ЯЁA-Z][а-яёa-z\-]+){0,2})\s*[.!?]?\s*$"
)
DOCTOR_REFERENCE_RE = re.compile(
    r"\b(?:врач(?:а|у|ом)?|доктор(?:а|у|ом)?|к\s+врачу|к\s+доктору)\s+"
    r"([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[А-ЯЁA-Z][а-яёa-z\-]+){0,2})",
    re.I,
)
NEGATED_REFERENCE_RE = re.compile(
    r"^\s*(?:нет|не)\s*,?\s*([А-ЯЁA-Z][а-яёa-z\-]+(?:\s+[А-ЯЁA-Z][а-яёa-z\-]+){0,2})\s*[.!?]?\s*$"
)
SERVICE_REFERENCE_RE = re.compile(
    r"\b(?:услуг[аиуой]?|анализ(?:а|у|ом)?|процедур(?:а|у|ой)?)(?:\s+«?([^»?.!,]+)»?)",
    re.I,
)
STRICT_YES_RE = re.compile(r"^\s*(да|yes|y)\s*[.!?]?\s*$", re.I)
STRICT_NO_RE = re.compile(r"^\s*(нет|не|no|n)\s*[.!?]?\s*$", re.I)
GUARD_YES_RE = re.compile(r"^\s*(да|угу|ага|yes|yep|ok|ок|конечно)\s*[.!?]?\s*$", re.I)
GUARD_NO_RE = re.compile(r"^\s*(нет|неа|no|nope|not now|пока нет)\s*[.!?]?\s*$", re.I)
LEADING_CONFIRM_HEAD_RE = re.compile(r"^\s*(да|yes|y|нет|не|no|n)\b", re.I)
UNCERTAINTY_RE = re.compile(
    r"\b("
    r"не\s+знаю|"
    r"не\s+помню|"
    r"не\s+уверен\w*|"
    r"затрудняюсь|"
    r"не\s+могу\s+сказать"
    r")\b",
    re.I,
)
NO_PREFERENCE_RE = re.compile(
    r"\b("
    r"любой|любая|любое|"
    r"без\s+разниц\w*|"
    r"неважн\w*|"
    r"как\s+угодно|"
    r"вс[её]\s+равно"
    r")\b",
    re.I,
)
SLOT_CORRECTION_RE = re.compile(
    r"\b("
    r"не\s+эт(?:от|ого|у)\s+(?:врач\w*|доктор\w*|специалист\w*|филиал\w*|адрес\w*|анализ\w*|услуг\w*)|"
    r"не\s+тот\s+филиал|"
    r"не\s+та\s+дат\w*|"
    r"не\s+то\s+время|"
    r"друг(?:ой|ая|ое)\s+(?:врач\w*|доктор\w*|специалист\w*|филиал\w*|адрес\w*|дат\w*|врем\w*|услуг\w*|анализ\w*)"
    r")\b",
    re.I,
)
MIXED_SWITCH_MARKER_RE = re.compile(
    r"(?:,\s*|\s+)"
    r"(а\s+лучше|а\s+ещ[её]|а\s+покажи|а\s+расскажи|а\s+сколько|а\s+кто|а\s+что|а\s+где|а\s+как|а\s+когда|а\s+почему|а\s+зачем|но|ладно|тогда|кстати)\b",
    re.I,
)
LEADING_SWITCH_PREFIX_RE = re.compile(
    r"^\s*(?:а\s+лучше\s+|а\s+ещ[её]\s+|а\s+|но\s+|ладно[, ]*|тогда\s+|кстати[, ]*)",
    re.I,
)

BRANCH_FOLLOWUP_STOPWORDS = {
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
TOPIC_WORD_RE = re.compile(
    r"\b(кто|что|где|когда|сколько|как|почему|зачем|расскажи|покажи|дай|найди|поищи|объясни|подскажи)\b",
    re.I,
)
RESULT_SURNAME_TOKEN_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{2,}$")
RESULT_CODE_TOKEN_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]{1,4}$")
RESULT_YEAR_TOKEN_RE = re.compile(r"^(19|20)\d{2}$")
RESULT_NUMBER_TOKEN_RE = re.compile(r"^\d{2,12}$")
MONTH_NAME_TO_NUMBER = {
    "января": 1,
    "январь": 1,
    "февраля": 2,
    "февраль": 2,
    "марта": 3,
    "март": 3,
    "апреля": 4,
    "апрель": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "сентября": 9,
    "сентябрь": 9,
    "октября": 10,
    "октябрь": 10,
    "ноября": 11,
    "ноябрь": 11,
    "декабря": 12,
    "декабрь": 12,
}


def parse_yes_no(text: str, *, profile: str = "strict") -> str:
    probe = str(text or "").strip()
    if not probe:
        return "unknown"
    if str(profile or "").strip().lower() == "guard":
        if GUARD_YES_RE.match(probe):
            return "yes"
        if GUARD_NO_RE.match(probe):
            return "no"
        return "unknown"
    if STRICT_YES_RE.match(probe):
        return "yes"
    if STRICT_NO_RE.match(probe):
        return "no"
    return "unknown"


def extract_confirmation_head(text: str) -> tuple[str, str]:
    probe = str(text or "").strip()
    if not probe:
        return "unknown", ""
    match = LEADING_CONFIRM_HEAD_RE.match(probe)
    if not match:
        return "unknown", probe
    decision = parse_yes_no(str(match.group(1) or "").strip(), profile="strict")
    remainder = probe[match.end():].lstrip(" ,")
    return decision, remainder


def extract_person_name(text: str) -> str:
    probe = str(text or "").strip()
    if not probe or any(ch.isdigit() for ch in probe):
        return ""
    match = PERSON_NAME_RE.match(probe)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def extract_doctor_reference_candidate(text: str) -> str:
    probe = str(text or "").strip()
    if not probe:
        return ""
    explicit = DOCTOR_REFERENCE_RE.search(probe)
    if explicit:
        return str(explicit.group(1) or "").strip()
    negated = NEGATED_REFERENCE_RE.match(probe)
    if negated:
        return str(negated.group(1) or "").strip()
    short = CAPITALIZED_NAME_CANDIDATE_RE.match(probe)
    if short:
        return str(short.group(1) or "").strip()
    return ""


def extract_service_reference_candidate(text: str) -> str:
    probe = str(text or "").strip()
    if not probe:
        return ""
    explicit = SERVICE_REFERENCE_RE.search(probe)
    if explicit:
        return str(explicit.group(1) or "").strip(" ,.?!")
    return ""


def extract_service_variant(text: str) -> str:
    match = SERVICE_VARIANT_RE.search(str(text or ""))
    return str(match.group(1) or "").strip().lower() if match else ""


def looks_like_slot_correction(text: str) -> bool:
    return bool(SLOT_CORRECTION_RE.search(str(text or "").strip()))


def looks_like_uncertainty_answer(text: str) -> bool:
    return bool(UNCERTAINTY_RE.search(str(text or "").strip()))


def looks_like_no_preference_answer(text: str) -> bool:
    return bool(NO_PREFERENCE_RE.search(str(text or "").strip()))


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
    explicit = BRANCH_CAPTURE_RE.search(source)
    if explicit:
        candidate = _clean_branch_candidate(str(explicit.group(1) or ""))
        if candidate:
            return candidate
    short = SHORT_BRANCH_RE.match(source)
    if not short:
        return ""
    candidate = _clean_branch_candidate(str(short.group(1) or ""))
    if not candidate:
        return ""
    low_candidate = candidate.lower()
    if low_candidate in BRANCH_FOLLOWUP_STOPWORDS:
        return ""
    if DATE_FILTER_RE.search(candidate) or TIME_FILTER_RE.search(candidate):
        return ""
    if len(candidate.split()) > 5:
        return ""
    return candidate


def extract_date_filters(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
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
    iso_match = DATE_ISO_RE.search(source)
    if iso_match:
        iso = f"{int(iso_match.group(1)):04d}-{int(iso_match.group(2)):02d}-{int(iso_match.group(3)):02d}"
        return {"date": iso, "date_from": iso, "date_to": iso}
    dot_match = DATE_DOT_RE.search(source)
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
    word_match = DATE_WORD_RE.search(source)
    if word_match:
        day = int(word_match.group(1))
        month_name = str(word_match.group(2) or "").strip().lower()
        month = MONTH_NAME_TO_NUMBER.get(month_name)
        if not month:
            return out
        year_raw = str(word_match.group(3) or "").strip()
        year = today.year
        if year_raw:
            year = int(year_raw)
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


def extract_time_filters(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
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
    after_match = TIME_AFTER_RE.search(source)
    if after_match:
        minutes = int(after_match.group(2) or 0)
        out["time_from"] = f"{int(after_match.group(1)):02d}:{minutes:02d}"
        out["time"] = out["time_from"]
    before_match = TIME_BEFORE_RE.search(source)
    if before_match:
        minutes = int(before_match.group(2) or 0)
        out["time_to"] = f"{int(before_match.group(1)):02d}:{minutes:02d}"
    exact_match = TIME_HHMM_RE.search(source)
    if exact_match:
        exact = f"{int(exact_match.group(1)):02d}:{int(exact_match.group(2)):02d}"
        out["time"] = exact
        out["time_from"] = exact
    return out


def extract_contextual_entities(user_message: str) -> dict[str, str]:
    text = str(user_message or "").strip()
    if not text:
        return {}
    out: dict[str, str] = {}
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


def split_mixed_utterance(text: str) -> tuple[str, str]:
    probe = str(text or "").strip()
    if not probe:
        return "", ""
    match = MIXED_SWITCH_MARKER_RE.search(probe)
    if not match:
        return "", ""
    left = probe[: match.start()].strip(" ,")
    right_raw = probe[match.start():].strip(" ,")
    right = LEADING_SWITCH_PREFIX_RE.sub("", right_raw, count=1).strip(" ,")
    if not left or not right:
        return "", ""
    return left, right


def extract_result_lookup_fields(text: str, *, expected_slots: list[str] | tuple[str, ...] | set[str] = ()) -> dict[str, str]:
    probe = str(text or "").strip()
    if not probe:
        return {}
    slots = {str(slot or "").strip().lower() for slot in expected_slots if str(slot or "").strip()}
    if not slots:
        slots = {
            "result_surname",
            "result_year_of_birth",
            "result_analysis_code",
            "result_analysis_number",
        }
    tokens = [part.strip(" .!?") for part in re.split(r"[,;\n]+", probe) if part.strip(" .!?")]
    if not tokens:
        tokens = [probe.strip(" .!?")]

    out: dict[str, str] = {}
    for token in tokens:
        value = str(token or "").strip()
        if not value:
            continue
        compact = re.sub(r"\s+", " ", value)
        if "result_year_of_birth" in slots and "result_year_of_birth" not in out and RESULT_YEAR_TOKEN_RE.match(compact):
            out["result_year_of_birth"] = compact
            continue
        if "result_analysis_number" in slots and "result_analysis_number" not in out and RESULT_NUMBER_TOKEN_RE.match(compact):
            out["result_analysis_number"] = compact
            continue
        if "result_analysis_code" in slots and "result_analysis_code" not in out and RESULT_CODE_TOKEN_RE.match(compact):
            out["result_analysis_code"] = compact
            continue
        if "result_surname" in slots and "result_surname" not in out:
            person_name = extract_person_name(compact)
            if person_name:
                out["result_surname"] = person_name.split()[0]
                continue
            doctor_candidate = extract_doctor_reference_candidate(compact)
            if doctor_candidate:
                out["result_surname"] = doctor_candidate.split()[0]
                continue
            if RESULT_SURNAME_TOKEN_RE.match(compact):
                out["result_surname"] = compact
    return out


def extract_expected_slot_entities(text: str, *, expected_slots: list[str] | tuple[str, ...] | set[str] = ()) -> dict[str, str]:
    probe = str(text or "").strip()
    if not probe:
        return {}
    slots = {str(slot or "").strip().lower() for slot in expected_slots if str(slot or "").strip()}
    out: dict[str, str] = {}
    if not slots:
        return out

    if {"result_surname", "result_year_of_birth", "result_analysis_code", "result_analysis_number"} & slots:
        out.update(extract_result_lookup_fields(probe, expected_slots=slots))

    if "patient_name" in slots:
        patient_name = extract_person_name(probe)
        if patient_name:
            out["patient_name"] = patient_name

    if {"date", "date_from", "date_to"} & slots:
        out.update(extract_date_filters(probe))
    if {"time", "time_from", "time_to"} & slots:
        out.update(extract_time_filters(probe))
    if {"branch_or_city", "branch_name"} & slots:
        branch_name = extract_branch_reference(probe)
        if branch_name:
            out["branch_name"] = branch_name
        city = extract_city_reference(probe)
        if city:
            out["city"] = city

    if "doctor_name" in slots:
        doctor_name = extract_doctor_reference_candidate(probe)
        if doctor_name:
            out["doctor_name"] = doctor_name
    if "specialty" in slots and not out.get("doctor_name") and _looks_like_short_slot_phrase(probe):
        out["specialty"] = probe
    if "service_or_analysis_name" in slots:
        service_name = extract_service_reference_candidate(probe)
        if service_name:
            out["service_name"] = service_name
        elif extract_service_variant(probe):
            out["service_variant"] = extract_service_variant(probe)
        elif _looks_like_short_slot_phrase(probe):
            out["service_name"] = probe
    return out


def looks_like_doctor_followup_message(*, user_message: str, remembered_doctor: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    remembered = str(remembered_doctor or "").strip()
    if not remembered:
        return False
    if DOCTOR_ANAPHORA_RE.search(text):
        return True
    if DOCTOR_FOLLOWUP_RE.search(text):
        return True
    lowered = text.lower().replace("ё", "е")
    remembered_parts = [part.strip().lower().replace("ё", "е") for part in remembered.split() if part.strip()]
    if not remembered_parts:
        return False
    if len(lowered.split()) <= 3:
        return any(part in lowered for part in remembered_parts)
    return False


def _looks_like_short_slot_phrase(text: str) -> bool:
    probe = str(text or "").strip()
    if not probe or "?" in probe or len(probe.split()) > 8:
        return False
    if TOPIC_WORD_RE.search(probe):
        return False
    return True


def looks_like_service_followup_message(*, user_message: str, remembered_service: str) -> bool:
    text = str(user_message or "").strip()
    if not text:
        return False
    remembered = str(remembered_service or "").strip()
    if not remembered:
        return False
    if SERVICE_FOLLOWUP_RE.search(text):
        return True
    if extract_service_variant(text):
        return True
    if extract_contextual_entities(text):
        return True
    lowered = text.lower().replace("ё", "е")
    remembered_parts = [part.strip().lower().replace("ё", "е") for part in remembered.split() if part.strip()]
    if len(lowered.split()) <= 3:
        return any(part in lowered for part in remembered_parts)
    return False
