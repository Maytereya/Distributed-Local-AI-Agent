"""Главный оркестратор мессенджерного диалога.

Выполняет pipeline: classify -> slot filling -> plan -> services -> rendering,
ведет APPOINTMENT flow, pending-уточнения, handoff решения и debug meta.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import AsyncGenerator, Any

from .mess_types import Evidence, Plan, PlanStep, ResponseEnvelope, RouteDecision, SessionState
from .classifier import analyze
from .policies import (
    require_auth_for_test_result,
    missing_slots,
    clarification_question,
    evidence_requires_handoff,
    handoff_message,
    appointment_step_policy,
    APPOINTMENT_STEP_BRANCH,
    APPOINTMENT_STEP_DATETIME,
    APPOINTMENT_STEP_CONFIRM,
    APPOINTMENT_CONFIRM_YES,
    APPOINTMENT_CONFIRM_NO,
    appointment_confirmation_transition,
    extract_price_rub,
    appointment_summary,
    appointment_addresses_for_city,
    appointment_text_branch_prompt,
    appointment_text_datetime_prompt,
    appointment_text_confirm_prompt,
    appointment_text_confirmed_handoff,
    appointment_text_reask_datetime,
    appointment_text_reask_confirm,
    doctor_schedule_text_clarify_doctor,
    decision_handoff_text,
    extract_service_phrase,
)
from .services import Services
from .renderer import (
    render_urgent,
    render_complaint,
    render_medical_advice,
    render_stream,
    format_doctor_schedule_for_patient,
)
from .memory import MemoryStore
from .city import match_city, looks_like_address


def _apply_pending_override(decision_label: str, pending: dict | None) -> str:
    if not pending:
        return decision_label
    pending_label = pending.get("label")
    if not isinstance(pending_label, str):
        return decision_label
    if decision_label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return decision_label
    return pending_label


def _normalize_secondary_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        if isinstance(x, str) and x and x not in out:
            out.append(x)
    return out


def _get_secondary_queue(state: SessionState) -> list[str]:
    return _normalize_secondary_labels(state.last_entities.get("_secondary_queue"))


def _set_secondary_queue(state: SessionState, labels: list[str]) -> None:
    clean = _normalize_secondary_labels(labels)
    if clean:
        state.last_entities["_secondary_queue"] = clean
        state.last_entities["secondary_intents"] = clean
    else:
        state.last_entities.pop("_secondary_queue", None)
        state.last_entities.pop("secondary_intents", None)
        state.last_entities.pop("_secondary_offer_pending", None)


def _is_affirmative(text: str) -> bool:
    return bool(_YES_RE.match(text or ""))


def _is_negative(text: str) -> bool:
    return bool(_NO_RE.match(text or ""))


def _secondary_followup_text(labels: list[str]) -> str | None:
    if not labels:
        return None
    first = labels[0]
    mapping = {
        "PRICE": "Также вижу вопрос по стоимости. Могу сразу подсказать цену по услуге.",
        "ADDRESS": "Также могу подсказать адрес и режим работы подходящего филиала.",
        "APPOINTMENT": "Также могу помочь с записью на прием после уточнения текущего вопроса.",
        "TEST_ASSIST": "Также могу помочь с подбором анализов под вашу цель.",
        "DOCTOR_SCHEDULE": "Также могу показать расписание нужного врача.",
        "DOCTOR_INFO": "Также могу подсказать, какие врачи принимают по вашему запросу.",
    }
    return mapping.get(first)


def _safe_get_branches(services: Services) -> list[dict[str, str]]:
    """
    Надежный доступ к справочнику филиалов:
    при ошибке интеграции возвращаем пустой список, а не исключение.
    """
    try:
        branches = services.get_branches()
    except Exception:
        return []
    if not isinstance(branches, list):
        return []
    return [b for b in branches if isinstance(b, dict)]


# ----------------------------
# Quick slot filling (NO LLM!)
# ----------------------------

_DMS_RE = re.compile(r"\bдмс\b", re.I)
_OMS_RE = re.compile(r"\bомс\b", re.I)
_PAID_RE = re.compile(r"\bплатн(о|ый|ая)\b|\bза наличн|\bоплат", re.I)

_CHILD_RE = re.compile(r"\bдет(и|ям|ский|ская|ского|ских)\b", re.I)
_AGE_RE = re.compile(r"\b(\d{1,2})\s*(?:лет|года|год)\b", re.I)

_NEXT_WEEK_RE = re.compile(r"\bна следующ(ей|ую)\s+недел", re.I)
_THIS_WEEK_RE = re.compile(r"\bна эт(ой|у)\s+недел|\bв эт(ой|у)\s+недел", re.I)
_TOMORROW_RE = re.compile(r"\bзавтра\b", re.I)
_TODAY_RE = re.compile(r"\bсегодня\b", re.I)
_YES_RE = re.compile(r"^\s*(да|ага|угу|ок|окей|хорошо|давайте|конечно)\s*[!.,?]*\s*$", re.I)
_NO_RE = re.compile(r"^\s*(нет|не надо|не нужно|неа|отмена|не хочу)\s*[!.,?]*\s*$", re.I)

_BRANCH_EXPLICIT_RE = re.compile(r"\bфилиал\b[:\s]*([^\n,;.]{2,80})", re.I)
_BRANCH_ON_RE = re.compile(r"\bна\s+([А-ЯЁа-яё0-9\-]{3,40})(?:\s+([0-9]{1,4}))?\b")
_SPECIALTY_HINT_RE = re.compile(
    r"\b(уролог|гинеколог|терапевт|эндокринолог|невролог|кардиолог|лор|офтальмолог|дерматолог|педиатр)\b",
    re.I,
)

_DOCTOR_FIO_RE = re.compile(r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b")
_SURNAME_RE = re.compile(r"\b([А-ЯЁ][а-яё]+)\b")

_ORDER_ID_RE = re.compile(r"(?:заказ|order|№)\s*([0-9]{4,})", re.I)

_TEST_WORDS_RE = re.compile(r"\b(анализ|пцр|hba1c|глюкоз|витамин|ферритин|ттг|т4|т3|холестер|оак|оам)\b", re.I)

# --- date/time parsing ---
_WEEKDAYS = {
    "понедельник": 0, "пн": 0,
    "вторник": 1, "вт": 1,
    "среда": 2, "ср": 2,
    "четверг": 3, "чт": 3,
    "пятница": 4, "пт": 4,
    "суббота": 5, "сб": 5,
    "воскресенье": 6, "вс": 6,
}

_MONTHS = {
    "января": 1, "январь": 1,
    "февраля": 2, "февраль": 2,
    "марта": 3, "март": 3,
    "апреля": 4, "апрель": 4,
    "мая": 5, "май": 5,
    "июня": 6, "июнь": 6,
    "июля": 7, "июль": 7,
    "августа": 8, "август": 8,
    "сентября": 9, "сентябрь": 9,
    "октября": 10, "октябрь": 10,
    "ноября": 11, "ноябрь": 11,
    "декабря": 12, "декабрь": 12,
}

_DATE_DOT_RE = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b")
_DATE_WORD_RE = re.compile(r"\b(\d{1,2})\s+([А-ЯЁа-яё]+)(?:\s+(\d{4}))?\b", re.I)

_RANGE_WORD_RE = re.compile(
    r"\bс\s+(\d{1,2})\s+(?:по|-)\s+(\d{1,2})\s+([А-ЯЁа-яё]+)(?:\s+(\d{4}))?\b",
    re.I,
)
_RANGE_DASH_RE = re.compile(
    r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s+([А-ЯЁа-яё]+)(?:\s+(\d{4}))?\b",
    re.I,
)

_WEEKDAY_RE = re.compile(
    r"\b(в|во|на)\s+(понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье|пн|вт|ср|чт|пт|сб|вс)\b",
    re.I,
)

# time patterns
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.](\d{2})\b")
_TIME_HOUR_RE = re.compile(r"\b([01]?\d|2[0-3])\b")
_AFTER_TIME_RE = re.compile(r"\b(после|с)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b", re.I)
_BEFORE_TIME_RE = re.compile(r"\b(до|раньше)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b", re.I)
_EXACT_TIME_RE = re.compile(r"\b(в|к)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b", re.I)
_RANGE_TIME_RE = re.compile(
    r"\b(с)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\s+(до|-)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b",
    re.I,
)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> list[str]:
    s = _norm(s)
    s = re.sub(r"[^a-zа-яё0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if t]


def _build_branch_index(branches: list[dict[str, str]]) -> list[dict[str, Any]]:
    idx: list[dict[str, Any]] = []
    for b in branches:
        bid = (b.get("id") or "").strip()
        name = (b.get("name") or "").strip()
        aliases_raw = (b.get("aliases") or "").strip()
        if not bid or not name:
            continue

        alias_list = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []

        bag = [name, *alias_list]

        tokens: set[str] = set()
        for item in bag:
            for t in _tokenize(item):
                tokens.add(t)

        idx.append({"id": bid, "name": name, "aliases": alias_list, "tokens": tokens})
    return idx


def _branch_options_to_indexable(branch_options: Any) -> list[dict[str, str]]:
    """
    Превращает список ранее показанных адресов в branch-like список для локального матчинга.
    """
    if not isinstance(branch_options, list):
        return []
    out: list[dict[str, str]] = []
    for i, raw in enumerate(branch_options, start=1):
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name:
            continue
        out.append(
            {
                "id": f"shown_{i}",
                "name": name,
                "aliases": _norm(name),
            }
        )
    return out


def _match_branch(text: str, branch_index: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not branch_index:
        return None, None

    txt_tokens = set(_tokenize(text))
    if not txt_tokens:
        return None, None
    input_numbers = {
        t for t in txt_tokens
        if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)
    }

    best_id: str | None = None
    best_name: str | None = None
    best_score = 0

    for b in branch_index:
        if input_numbers:
            branch_numbers = {
                t for t in b["tokens"]
                if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)
            }
            if branch_numbers and not (input_numbers & branch_numbers):
                continue
        common = txt_tokens & b["tokens"]
        score = len(common)
        if any(len(t) >= 5 for t in common):
            score += 1
        if score > best_score:
            best_score = score
            best_id = b["id"]
            best_name = b["name"]

    if best_score >= 2:
        return best_id, best_name

    # fallback #1: один "средний" токен, но только если кандидат уникален
    medium_candidates: list[tuple[str, str]] = []
    for b in branch_index:
        if input_numbers:
            branch_numbers = {
                t for t in b["tokens"]
                if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)
            }
            if branch_numbers and not (input_numbers & branch_numbers):
                continue
        common = txt_tokens & b["tokens"]
        if any(len(t) >= 5 for t in common):
            medium_candidates.append((b["id"], b["name"]))
    if len(medium_candidates) == 1:
        return medium_candidates[0]

    # fallback #2: один длинный токен
    for b in branch_index:
        if input_numbers:
            branch_numbers = {
                t for t in b["tokens"]
                if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)
            }
            if branch_numbers and not (input_numbers & branch_numbers):
                continue
        common = txt_tokens & b["tokens"]
        if any(len(t) >= 7 for t in common):
            return b["id"], b["name"]

    return None, None


def _next_weekday(from_date: date, target_weekday: int) -> date:
    """Ближайший день недели target_weekday (0=Mon), включая сегодня."""
    delta = (target_weekday - from_date.weekday()) % 7
    return from_date + timedelta(days=delta)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except Exception:
        return None


def _parse_date_time_ru(text: str, today: date | None = None) -> dict[str, Any]:
    """
    Парсим дату/диапазон/время.
    Возвращает частичный dict: date_from/date_to/time_from/time_to/date_hint
    """
    if today is None:
        today = datetime.now().date()

    out: dict[str, Any] = {}
    s = text.strip()
    low = s.lower()

    # --- relative day hints (keep as hint unless exact date parsed later)
    if _NEXT_WEEK_RE.search(low):
        out["date_hint"] = "next_week"
    elif _THIS_WEEK_RE.search(low):
        out["date_hint"] = "this_week"
    elif _TOMORROW_RE.search(low):
        out["date_hint"] = "tomorrow"
    elif _TODAY_RE.search(low):
        out["date_hint"] = "today"

    # --- date ranges: "с 12 по 15 февраля" / "12–15 февраля"
    m = _RANGE_WORD_RE.search(s)
    if not m:
        m = _RANGE_DASH_RE.search(s)
    if m:
        d1 = int(m.group(1))
        d2 = int(m.group(2))
        mon_word = m.group(3).lower()
        y = m.group(4)
        month = _MONTHS.get(mon_word)
        if month:
            year = int(y) if y else today.year
            dt1 = _safe_date(year, month, d1)
            dt2 = _safe_date(year, month, d2)
            # если диапазон в прошлом и год не указан — двигаем на следующий год
            if not y and dt2 and dt2 < today:
                dt1 = _safe_date(year + 1, month, d1)
                dt2 = _safe_date(year + 1, month, d2)

            if dt1 and dt2:
                if dt2 < dt1:
                    dt1, dt2 = dt2, dt1
                out["date_from"] = dt1.isoformat()
                out["date_to"] = dt2.isoformat()
                out.pop("date_hint", None)

    # --- single date: dd.mm(.yyyy)
    if "date_from" not in out:
        md = _DATE_DOT_RE.search(s)
        if md:
            d = int(md.group(1))
            mth = int(md.group(2))
            y = md.group(3)
            year = int(y) if y else today.year
            if year < 100:  # 26 -> 2026
                year += 2000
            dt = _safe_date(year, mth, d)
            if dt and not y and dt < today:
                dt = _safe_date(year + 1, mth, d)
            if dt:
                out["date_from"] = dt.isoformat()
                out["date_to"] = dt.isoformat()
                out.pop("date_hint", None)

    # --- single date: "12 февраля"
    if "date_from" not in out:
        mw = _DATE_WORD_RE.search(s)
        if mw:
            d = int(mw.group(1))
            mon_word = mw.group(2).lower()
            y = mw.group(3)
            month = _MONTHS.get(mon_word)
            if month:
                year = int(y) if y else today.year
                dt = _safe_date(year, month, d)
                if dt and not y and dt < today:
                    dt = _safe_date(year + 1, month, d)
                if dt:
                    out["date_from"] = dt.isoformat()
                    out["date_to"] = dt.isoformat()
                    out.pop("date_hint", None)

    # --- weekday: "в понедельник"
    if "date_from" not in out:
        wd = _WEEKDAY_RE.search(s)
        if wd:
            token = wd.group(2).lower()
            # нормализуем "среду/пятницу/субботу/воскресенье"
            token = {
                "среду": "среда",
                "пятницу": "пятница",
                "субботу": "суббота",
                "воскресенье": "воскресенье",
                "понедельник": "понедельник",
                "вторник": "вторник",
                "четверг": "четверг",
            }.get(token, token)
            target = _WEEKDAYS.get(token)
            if target is not None:
                dt = _next_weekday(today, target)
                out["date_from"] = dt.isoformat()
                out["date_to"] = dt.isoformat()
                out.pop("date_hint", None)

    # --- time ranges: "с 10 до 12"
    tr = _RANGE_TIME_RE.search(s)
    if tr:
        h1 = int(tr.group(2))
        m1 = int(tr.group(3)) if tr.group(3) else 0
        h2 = int(tr.group(5))
        m2 = int(tr.group(6)) if tr.group(6) else 0
        out["time_from"] = f"{h1:02d}:{m1:02d}"
        out["time_to"] = f"{h2:02d}:{m2:02d}"

    # --- after/before
    if "time_from" not in out:
        a = _AFTER_TIME_RE.search(s)
        if a:
            h = int(a.group(2))
            m = int(a.group(3)) if a.group(3) else 0
            out["time_from"] = f"{h:02d}:{m:02d}"

    if "time_to" not in out:
        b = _BEFORE_TIME_RE.search(s)
        if b:
            h = int(b.group(2))
            m = int(b.group(3)) if b.group(3) else 0
            out["time_to"] = f"{h:02d}:{m:02d}"

    # --- exact time: "в 18:30" / "к 16"
    if "time_from" not in out and "time_to" not in out:
        ex = _EXACT_TIME_RE.search(s)
        if ex:
            h = int(ex.group(2))
            m = int(ex.group(3)) if ex.group(3) else 0
            tm = f"{h:02d}:{m:02d}"
            # интерпретируем как "точно в это время"
            out["time_from"] = tm
            out["time_to"] = tm

    # --- bare time: "13:00" (без "в/к")
    if "time_from" not in out and "time_to" not in out:
        bare = _TIME_RE.search(s)
        if bare:
            h = int(bare.group(1))
            m = int(bare.group(2))
            tm = f"{h:02d}:{m:02d}"
            out["time_from"] = tm
            out["time_to"] = tm

    return out


def quick_fill_entities_from_text(
    text: str,
    state_entities: dict[str, Any],
    missing_rules: list[str],
    services: Services,
) -> dict[str, Any]:
    """
    Пытаемся заполнить частые слоты из текста без LLM.
    + резолв филиала (branch_id)
    + парсинг даты/времени RU
    """
    t = text.strip()
    low = t.lower()
    out: dict[str, Any] = {}

    # insurance_type
    if _DMS_RE.search(low):
        out["insurance_type"] = "dms"
    elif _OMS_RE.search(low):
        out["insurance_type"] = "oms"
    elif _PAID_RE.search(low):
        out["insurance_type"] = "paid"

    # accepts_children
    if _CHILD_RE.search(low):
        out["accepts_children"] = True

    # date/time parsing (always useful for schedule/appointment)
    dt = _parse_date_time_ru(t)
    out.update({k: v for k, v in dt.items() if v is not None})

    # order_id
    m_oid = _ORDER_ID_RE.search(t)
    if m_oid:
        out["order_id"] = m_oid.group(1)

    # specialty hint
    m_spec = _SPECIALTY_HINT_RE.search(low)
    if m_spec:
        out["specialty"] = m_spec.group(1).lower()

    # doctor_name heuristic (only if we need doctor/specialty and message looks like a short answer)
    needs_doctor_or_spec = any("doctor" in r or "specialty" in r for r in missing_rules)
    if needs_doctor_or_spec:
        m_fio = _DOCTOR_FIO_RE.search(t)
        if m_fio:
            out["doctor_name"] = f"{m_fio.group(1)} {m_fio.group(2)}"
        else:
            words = [w for w in re.split(r"\s+", t) if w]
            if 1 <= len(words) <= 2:
                m_s = _SURNAME_RE.fullmatch(words[0])
                if m_s and not state_entities.get("doctor_name"):
                    out["doctor_name"] = m_s.group(1)

    # test goal/name heuristic
    needs_test = any("test_goal" in r or "test_name" in r for r in missing_rules)
    if needs_test and _TEST_WORDS_RE.search(low):
        if not state_entities.get("test_name"):
            out["test_goal"] = t[:200]

    # service_name heuristic (price/appointment/prepare)
    if any("service_name" in r for r in missing_rules) and len(t) >= 3:
        service_phrase = extract_service_phrase(t)
        out["service_name"] = (service_phrase or t[:200]).strip()

    # child_age heuristic
    if "child_age" in missing_rules:
        m_age = _AGE_RE.search(low)
        if m_age:
            try:
                out["child_age"] = int(m_age.group(1))
            except Exception:
                pass

    # ----------------------------
    # Branch resolution
    # ----------------------------
    needs_city = any("city" in r for r in missing_rules)
    if needs_city and not state_entities.get("city"):
        city = match_city(t)
        if not city:
            words = [w for w in re.split(r"\s+", t) if w]
            if 1 <= len(words) <= 2 and len(t) <= 30 and not looks_like_address(t):
                city = t
        if city:
            out["city"] = city.strip()

    if not state_entities.get("branch_id"):
        branch_hint = None

        m_bx = _BRANCH_EXPLICIT_RE.search(t)
        if m_bx:
            branch_hint = m_bx.group(1).strip()

        if not branch_hint:
            m_on = _BRANCH_ON_RE.search(t)
            if m_on:
                street = (m_on.group(1) or "").strip()
                num = (m_on.group(2) or "").strip()
                candidate = f"{street} {num}".strip()
                if num or looks_like_address(t):
                    branch_hint = candidate

        if not branch_hint:
            words = [w for w in re.split(r"\s+", t) if w]
            if 1 <= len(words) <= 3 and len(t) <= 30:
                # Если город уже известен, допускаем короткий выбор филиала ("Кирова").
                city_guess = match_city(t)
                is_city_only = bool(city_guess and _norm(city_guess) == _norm(t))
                if (looks_like_address(t) or state_entities.get("city")) and not is_city_only:
                    branch_hint = t

        if branch_hint:
            # Если мы уже показывали список адресов пользователю, резолвим только внутри него.
            shown_options = _branch_options_to_indexable(state_entities.get("appointment_branch_options"))
            branches = shown_options
            if not branches:
                branches = _safe_get_branches(services)
                city_hint = str(out.get("city") or state_entities.get("city") or "").strip().lower()
                if city_hint:
                    city_filtered = []
                    for b in branches:
                        if not isinstance(b, dict):
                            continue
                        hay = f"{b.get('name', '')} {b.get('aliases', '')}".lower()
                        if city_hint in hay:
                            city_filtered.append(b)
                    if city_filtered:
                        branches = city_filtered
            idx = _build_branch_index(branches)
            bid, bname = _match_branch(branch_hint, idx)
            if bid and not str(bid).startswith("shown_"):
                out["branch_id"] = bid
            if bname and not state_entities.get("branch_name"):
                out["branch_name"] = bname
            if (
                not shown_options
                and not bid
                and not state_entities.get("branch_name")
                and looks_like_address(branch_hint)
            ):
                out["branch_name"] = branch_hint[:80]

    return out


# ----------------------------
# Planning & execution
# ----------------------------

def build_plan(decision: RouteDecision, state: SessionState, user_text: str, memory: MemoryStore) -> Plan:
    pending = memory.get_pending(state)
    effective_label = _apply_pending_override(decision.label, pending)

    entities = state.last_entities
    missing = missing_slots(effective_label, entities)

    if missing:
        memory.set_pending(state, label=effective_label, missing_slots=missing)
        return Plan(label=effective_label, steps=[])

    memory.clear_pending(state)

    label = effective_label
    steps: list[PlanStep] = []

    if label == "TEST_RESULT":
        steps.append(PlanStep(tool="test_result_status", input={"query": user_text, "entities": dict(entities)}, auth="patient_token"))
        steps.append(PlanStep(tool="test_result_pdf", input={"query": user_text, "entities": dict(entities)}, auth="patient_token", required=False))
        return Plan(label=label, steps=steps)

    if label == "TEST_ASSIST":
        steps.append(PlanStep(tool="test_assist", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_SCHEDULE":
        steps.append(PlanStep(tool="doctors_schedule_week", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_INFO":
        steps.append(PlanStep(tool="doctors_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "APPOINTMENT":
        if entities.get("doctor_id") or entities.get("doctor_name"):
            steps.append(PlanStep(tool="doctors_schedule_week", input={"query": user_text, "entities": dict(entities)}))
        else:
            steps.append(PlanStep(tool="address_info", input={"query": user_text, "entities": dict(entities)}))
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}, required=False))
        return Plan(label=label, steps=steps)

    if label == "PRICE":
        steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "ADDRESS":
        steps.append(PlanStep(tool="address_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "PREPARE":
        steps.append(PlanStep(tool="test_prepare", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "NEWS":
        steps.append(PlanStep(tool="news_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    return Plan(label=label, steps=[])


async def execute_plan(plan: Plan, state: SessionState, services: Services) -> Evidence:
    ev = Evidence()

    for step in plan.steps:
        if step.auth == "patient_token":
            need_auth, msg = require_auth_for_test_result(state.is_authenticated)
            if need_auth:
                ev.put("auth_required", True)
                ev.put("auth_message", msg)
                return ev

        tool = step.tool
        inp = step.input
        q = inp.get("query", "")
        ent = inp.get("entities") or {}

        try:
            if tool == "doctors_info":
                ev.put("doctors_info", await services.doctors_info(q, ent, output_max=5))
            elif tool == "doctors_schedule_week":
                ev.put("doctor_schedule", await services.doctors_schedule_week(q, ent))
            elif tool == "appointment_help":
                ev.put("appointment", await services.appointment_help(q, ent))
            elif tool == "test_assist":
                ev.put("test_assist", await services.test_assist(q, ent))
            elif tool == "test_prepare":
                ev.put("prepare", await services.test_prepare(q, ent))
            elif tool == "test_result_status":
                ev.put("test_result_status", await services.test_result_status(q, ent))
            elif tool == "test_result_pdf":
                ev.put("test_result_pdf", await services.test_result_pdf(q, ent))
            elif tool == "price_info":
                ev.put("price", await services.price_info(q, ent))
            elif tool == "address_info":
                ev.put("address", await services.address_info(q, ent))
            elif tool == "news_info":
                ev.put("news", await services.news_info(q, ent))
            else:
                ev.put("unknown_tool", tool)
        except Exception as e:
            ev.put(
                f"{tool}_error",
                {
                    "tool": tool,
                    "message": str(e),
                    "query": q,
                },
            )
            ev.put("handoff_required", True)
            ev.put("handoff_reason", "service_error")
            ev.put(
                "handoff_message",
                handoff_message("service_error"),
            )
            return ev

    return ev


async def route_patient_message(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> tuple[RouteDecision, Plan, Evidence]:
    # Вежливое переключение на вторичный интент по короткому "да/нет".
    queue = _get_secondary_queue(state)
    if (
        state.last_entities.get("_secondary_offer_pending")
        and queue
        and not state.last_entities.get("appointment_confirm_pending")
        and not state.last_entities.get("appointment_flow_active")
    ):
        if _is_affirmative(user_text):
            next_label = queue.pop(0)
            _set_secondary_queue(state, queue)
            state.last_entities["_secondary_offer_pending"] = False
            decision = RouteDecision(
                label=next_label,  # type: ignore[arg-type]
                confidence=0.9,
                entities={"secondary_intent_from_queue": True},
                flags={"secondary_intent_activated"},
                needs_handoff=False,
            )
            plan = build_plan(decision, state, user_text, memory=memory)
            evidence = await execute_plan(plan, state, services)
            return decision, plan, evidence
        if _is_negative(user_text):
            _set_secondary_queue(state, [])
            state.last_entities["_secondary_offer_pending"] = False
        else:
            # Пользователь продолжил диалог в другом направлении.
            state.last_entities["_secondary_offer_pending"] = False

    decision = await analyze(user_text, state.last_entities)

    # Сохраняем сценарий записи на операторских уточнениях (ветка "да/нет", короткие ответы и т.п.).
    if decision.label == "OTHER" and state.last_entities.get("appointment_flow_active"):
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=max(decision.confidence, 0.51),
            entities=decision.entities,
            flags=set(decision.flags) | {"flow_appointment_override"},
            needs_handoff=False,
        )

    # merge entities from LLM+rules
    memory.merge_entities(state, decision.entities, label=decision.label)
    sec_now = _normalize_secondary_labels(decision.entities.get("secondary_intents"))
    if sec_now:
        existing = _get_secondary_queue(state)
        merged = [x for x in existing if x != decision.label]
        for x in sec_now:
            if x != decision.label and x not in merged:
                merged.append(x)
        _set_secondary_queue(state, merged)

    # quick fill on current turn (before pending exists)
    pending = memory.get_pending(state)
    if not pending:
        missing_now = missing_slots(decision.label, state.last_entities)
        if missing_now:
            quick_now = quick_fill_entities_from_text(user_text, state.last_entities, missing_now, services)
            if quick_now:
                memory.merge_entities(state, quick_now, label=decision.label)
        elif decision.label == "APPOINTMENT" and state.last_entities.get("appointment_flow_active"):
            # В активном сценарии записи продолжаем извлекать филиал/дату/время
            # даже если формально required slots уже заполнены.
            quick_flow = quick_fill_entities_from_text(
                user_text,
                state.last_entities,
                ["_any_of:city,branch_name,branch_id", "date_from", "time_from"],
                services,
            )
            if quick_flow:
                memory.merge_entities(state, quick_flow, label=decision.label)

    # if pending exists, try quick fill missing slots (NO LLM)
    pending = memory.get_pending(state)
    if pending:
        pend_label = pending.get("label")
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        if isinstance(pend_label, str) and isinstance(missing, list) and missing:
            quick = quick_fill_entities_from_text(user_text, state.last_entities, missing, services)
            if quick:
                memory.merge_entities(state, quick, label=pend_label)

    plan = build_plan(decision, state, user_text, memory=memory)
    evidence = await execute_plan(plan, state, services)
    return decision, plan, evidence



def _debug_meta(decision: RouteDecision, plan: Plan, evidence: Evidence, state: SessionState, pending: Any) -> dict[str, Any]:
    return {
        "decision": {
            "label": decision.label,
            "confidence": decision.confidence,
            "flags": sorted(list(decision.flags)),
            "entities": decision.entities,
            "needs_handoff": decision.needs_handoff,
        },
        "plan": {
            "label": plan.label,
            "steps": [
                {"tool": s.tool, "input": s.input, "required": s.required, "auth": s.auth}
                for s in plan.steps
            ],
        },
        "evidence": {
            "items": evidence.items,
            "debug_trace": evidence.debug_trace,
        },
        "pending": pending,
        "history_tail": (state.history or [])[-10:],
        "last_entities": state.last_entities,
    }


async def patient_routing_stream(
    user_text: str,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    debug: bool = False,
) -> AsyncGenerator[ResponseEnvelope, None]:
    try:
        decision, plan, evidence = await route_patient_message(user_text, state, services, memory)
    except Exception as e:
        fallback_text = handoff_message("service_error")
        state_update: dict[str, Any] = {}
        if debug:
            state_update = {"debug": {"route_error": str(e)}}
        yield ResponseEnvelope(
            text=fallback_text,
            attachments=[],
            handoff=True,
            state_update=state_update,
        )
        return
    flow_label = plan.label

    if debug:
        pending = memory.get_pending(state)
        yield ResponseEnvelope(
            text="",
            attachments=[],
            handoff=False,
            state_update={"debug": _debug_meta(decision, plan, evidence, state, pending)},
        )

    if decision.label == "URGENT":
        yield render_urgent()
        return
    if decision.label == "COMPLAINT":
        yield render_complaint()
        return
    if decision.label == "MEDICAL_ADVICE":
        yield render_medical_advice()
        return

    if "doc_request_handoff" in decision.flags:
        yield ResponseEnvelope(
            text=handoff_message("doc_request_handoff"),
            handoff=True,
        )
        return

    if flow_label == "TEST_RESULT":
        yield ResponseEnvelope(
            text=handoff_message("test_result_fallback"),
            handoff=True,
        )
        return

    # APPOINTMENT confirmation loop: после вопроса "Подтверждаете?"
    if state.last_entities.get("appointment_confirm_pending"):
        confirm_transition = appointment_confirmation_transition(user_text)
        if confirm_transition == APPOINTMENT_CONFIRM_YES:
            summary = appointment_summary(state.last_entities)
            state.last_entities["appointment_confirmed"] = True
            state.last_entities.pop("appointment_confirm_pending", None)
            state.last_entities.pop("appointment_flow_active", None)
            yield ResponseEnvelope(
                text=appointment_text_confirmed_handoff(summary),
                handoff=True,
            )
            return
        if confirm_transition == APPOINTMENT_CONFIRM_NO:
            state.last_entities["appointment_confirmed"] = False
            state.last_entities.pop("appointment_confirm_pending", None)
            for k in ("date_from", "date_to", "time_from", "time_to", "date_hint"):
                state.last_entities.pop(k, None)
            state.last_entities["appointment_flow_active"] = True
            yield ResponseEnvelope(
                text=appointment_text_reask_datetime(),
                handoff=False,
            )
            return
        yield ResponseEnvelope(
            text=appointment_text_reask_confirm(),
            handoff=False,
        )
        return

    if flow_label == "DOCTOR_SCHEDULE":
        # есть специальность, но нет врача → уточняем
        if (
                "specialty" in decision.entities
                and "last_name" not in decision.entities
                and "doctor_last_name" not in decision.entities
        ):
            yield ResponseEnvelope(
                text=doctor_schedule_text_clarify_doctor(),
                handoff=False,
            )
            return

    pending = memory.get_pending(state)
    if not plan.steps and pending:
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        yield ResponseEnvelope(
            text=clarification_question(flow_label, missing if isinstance(missing, list) else []),
            handoff=False,
        )
        return

    if evidence.get("auth_required"):
        yield ResponseEnvelope(text=evidence.get("auth_message", "Нужна авторизация."), handoff=False)
        return

    handoff_required, handoff_msg, handoff_reason = evidence_requires_handoff(evidence)
    if handoff_required:
        yield ResponseEnvelope(text=handoff_message(handoff_reason, handoff_msg), handoff=True)
        return

    schedule_payload = evidence.get("doctor_schedule")
    if flow_label in {"DOCTOR_SCHEDULE", "APPOINTMENT"} and isinstance(schedule_payload, dict):
        if schedule_payload.get("schedule"):
            text = format_doctor_schedule_for_patient(schedule_payload, decision.entities)
            yield ResponseEnvelope(text=text, attachments=[], handoff=False)
            return

    if flow_label == "APPOINTMENT":
        entities = state.last_entities
        state.last_entities["appointment_flow_active"] = True

        appointment_step = appointment_step_policy(entities)
        service = str(entities.get("service_name") or entities.get("test_name") or "услугу").strip()
        city = str(entities.get("city") or "").strip()

        if appointment_step == APPOINTMENT_STEP_BRANCH:
            branches = _safe_get_branches(services)
            addresses = appointment_addresses_for_city(
                evidence.get("address"),
                branches,
                city=city or None,
                limit=5,
            )
            state.last_entities["appointment_branch_options"] = addresses
            yield ResponseEnvelope(
                text=appointment_text_branch_prompt(service, city, addresses),
                handoff=False,
            )
            return

        if appointment_step == APPOINTMENT_STEP_DATETIME:
            state.last_entities.pop("appointment_branch_options", None)
            price_rub = extract_price_rub(evidence.get("price"))
            branch = str(entities.get("branch_name") or entities.get("city") or "выбранном филиале").strip()
            yield ResponseEnvelope(
                text=appointment_text_datetime_prompt(service, branch, price_rub),
                handoff=False,
            )
            return

        if appointment_step == APPOINTMENT_STEP_CONFIRM:
            state.last_entities["appointment_confirm_pending"] = True
            summary = appointment_summary(entities)
            yield ResponseEnvelope(text=appointment_text_confirm_prompt(summary), handoff=False)
            return

    attachments: list[dict[str, Any]] = []
    pdf_payload = evidence.get("test_result_pdf")
    if isinstance(pdf_payload, dict) and pdf_payload.get("pdf"):
        attachments.append({"type": "pdf", "name": "Результаты анализов.pdf", "url": pdf_payload["pdf"]})

    try:
        async for chunk in render_stream(user_text, decision, evidence):
            yield ResponseEnvelope(text=chunk, attachments=[], handoff=False)
    except Exception:
        yield ResponseEnvelope(
            text=handoff_message("renderer_error"),
            attachments=[],
            handoff=True,
        )
        return

    if decision.needs_handoff:
        yield ResponseEnvelope(text=decision_handoff_text(decision.flags), attachments=[], handoff=True)

    if attachments:
        yield ResponseEnvelope(text="", attachments=attachments, handoff=False)

    secondary = _get_secondary_queue(state)
    followup = _secondary_followup_text(secondary)
    if (
        followup
        and not state.last_entities.get("_secondary_offer_pending")
        and not decision.needs_handoff
        and flow_label not in {"APPOINTMENT", "URGENT", "COMPLAINT", "MEDICAL_ADVICE", "TEST_RESULT"}
    ):
        state.last_entities["_secondary_offer_pending"] = True
        yield ResponseEnvelope(text=followup, attachments=[], handoff=False)
