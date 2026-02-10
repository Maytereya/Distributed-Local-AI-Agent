"""Централизованные policy-правила роутера.

Здесь находятся regex-детекторы интентов, slot/clarify/handoff политики,
тексты переходов APPOINTMENT flow и вспомогательные нормализаторы.
"""

from __future__ import annotations

import re
from typing import Any

from .city import looks_like_address

# ---------------------------
# Fast detectors (cheap gates)
# ---------------------------

URGENT_PATTERNS = [
    r"\bзадыхаюсь\b",
    r"\bне могу дышать\b",
    r"\bболь в груди\b",
    r"\bсильн(ая|ый)\s+боль\b",
    r"\bкровотечени(е|я)\b",
    r"\bкровь\b.*\bидет\b",
    r"\bпотер(я|ял)\s+сознани",
    r"\bсудорог(и|а)\b",
    r"\bинсульт\b",
    r"\bинфаркт\b",
    r"\bонемел(о|а)\b",
    r"\bперекосило лицо\b",
    r"\bне могу говорить\b",
]

COMPLAINT_PATTERNS = [
    r"\bжалоб",
    r"\bпретензи",
    r"\bхамств",
    r"\bплох(о|ая)\s+обслуж",
    r"\bверните деньги\b",
    r"\bобман\b",
    r"\bв суд\b",
    r"\bв прокуратур",
]

MEDICAL_ADVICE_PATTERNS = [
    r"\bчто со мной\b",
    r"\bпостав(ь|ьте)\s+диагноз\b",
    r"\bназнач(ь|ьте)\s+лечени",
    r"\bкакие таблетки\b",
    r"\bчем лечить\b",
]

TEST_INTERPRET_PATTERNS = [
    r"\bрасшифруй(те)?\b.*\bанализ",
    r"\bчто знач(ит|ат)\b.*\b(показател|результат)\b",
    r"\bнорм(а|ы)\b.*\b(анализ|показател)\b",
]

TEST_RESULT_PATTERNS = [
    r"\bрезультат(ы|ов)\s+анализ",
    r"\bанализ(ы)?\s+готов(ы|о)\b",
    r"\bготовност(ь|и)\s+анализ",
    r"\bпришлите\b.*\b(pdf|пдф|файл)\b",
    r"\bскачать\b.*\bрезультат",
    r"\bбланк\b.*\bанализ",
]

TEST_ASSIST_PATTERNS = [
    r"\bподобрат(ь|ь)\s+анализ",
    r"\bкакие анализ(ы)?\s+сдат(ь|ь)\b",
    r"\bчто сдат(ь|ь)\b.*\bанализ",
    r"\bчекап\b",
    r"\bкомплекс\b.*\bанализ",
    r"\bскрининг\b",
    r"\b(оак|оам|ферритин|ттг|т3|т4|глюкоз|витамин\s*д)\b",
]

SCHEDULE_PATTERNS = [
    r"\bрасписани(е|я)\b",
    r"\bграфик\b",
    r"\bкогда\b.*\bпринима(ет|ют)\b",
    r"\bна следующ(ей|ую)\s+недел",
    r"\bв ближайш(ие|ую)\s+7\s*дн",
]

DOC_REQUEST_PATTERNS = [
    r"\bсправк\w*",
    r"\bналог\w*\s+вычет\w*",
    r"\bвычет\w*",
    r"\bкопи(я|ю)\s+договор\w*",
    r"\bдоговор\w*\s+с\s+печат\w*",
    r"\bамбулаторн\w*\s+карт\w*",
    r"\bпротокол\w*\s+при(е|ё)м\w*",
    r"\bзаявлени\w*\s+на\s+возврат\w*",
]

APPOINTMENT_INTENT_PATTERNS = [
    r"\bзапис\w*",
    r"\bзапиш\w*",
    r"\bзапись\b",
    r"\bпри(е|ё)м\w*",
    r"\bперен\w*",
    r"\bперезапис\w*",
    r"\bотмен\w*",
    r"\bхолтер\w*",
]

PRICE_PATTERNS = [
    r"\bстоим\w*",
    r"\bцен\w*",
    r"\bпрайс\w*",
    r"\bсколько\b.*\bстоит\b",
]

ADDRESS_PATTERNS = [
    r"\bадрес\w*",
    r"\bфилиал\w*",
    r"\bотделени\w*",
    r"\bрежим\w*.*\bработ\w*",
    r"\bкак\b.*\bдобрат\w*",
    r"\bгде\b.*\bнаходит\w*",
]

SERVICE_ANCHORS = (
    "узи",
    "экг",
    "холтер",
    "мрт",
    "кт",
    "фгдс",
    "фкс",
    "рентген",
    "флюорограф",
    "колоноскоп",
)

SERVICE_BOUNDARY_WORDS = {
    "в", "во", "на", "к", "ко", "по", "из", "до", "после",
    "и", "или", "но",
    "сегодня", "завтра", "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
}

SERVICE_UPPERCASE = {"узи", "экг", "мрт", "кт", "фгдс", "фкс", "уздг"}

# ---------------------------
# Router policy maps
# ---------------------------

REQUIRED_SLOTS: dict[str, list[str]] = {
    "APPOINTMENT": [
        "_any_of:doctor_id,doctor_name,specialty,service_name",
        "_any_of:city,branch_name,branch_id",
    ],
    "TEST_ASSIST": [
        "_any_of:city,branch_name,branch_id",
        "_any_of:test_goal,test_name",
    ],
    "TEST_RESULT": ["_any_of:order_id"],
    "DOCTOR_INFO": ["_any_of:specialty,doctor_id,doctor_name"],
    # For schedule we require a concrete doctor reference, not specialty-only.
    "DOCTOR_SCHEDULE": ["_any_of:doctor_id,doctor_name"],
    "PRICE": [
        "_any_of:city,branch_name,branch_id",
        "service_name",
    ],
    "ADDRESS": ["_any_of:city,branch_name,branch_id"],
    "PREPARE": ["_any_of:test_name,service_name"],
    "NEWS": [],
    "COMPLAINT": [],
    "URGENT": [],
    "MEDICAL_ADVICE": [],
    "OTHER": [],
}

# Centralized reason -> patient message map for operator handoff.
HANDOFF_REASON_MATRIX: dict[str, str] = {
    "doc_request_handoff": "Для оформления справок и документов подключаю оператора. Он уточнит детали и поможет с заявкой.",
    "test_result_fallback": "Для проверки и выдачи результатов анализов подключаю оператора. Это нужно для корректной идентификации пациента.",
    "service_error": "Сейчас не удалось получить данные автоматически. Соединяю с оператором.",
    "renderer_error": "Сейчас не удалось сформировать ответ автоматически. Передаю диалог оператору.",
    "low_confidence": "Чтобы не ошибиться в ответе, подключаю оператора для уточнения деталей.",
    "generic": "Передаю диалог оператору.",
}

APPOINTMENT_STEP_BRANCH = "select_branch"
APPOINTMENT_STEP_DATETIME = "select_datetime"
APPOINTMENT_STEP_CONFIRM = "confirm"
APPOINTMENT_STEP_DONE = "done"
APPOINTMENT_CONFIRM_YES = "yes"
APPOINTMENT_CONFIRM_NO = "no"
APPOINTMENT_CONFIRM_OTHER = "other"

# ---------------------------
# Simple PII detector (MVP)
# ---------------------------

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s\(\)]{8,}\d)(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)


def detect_pii(text: str) -> set[str]:
    flags: set[str] = set()
    if _PHONE_RE.search(text):
        flags.add("pii_phone")
    if _EMAIL_RE.search(text):
        flags.add("pii_email")
    return flags


def detect_urgent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in URGENT_PATTERNS)


def detect_complaint(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in COMPLAINT_PATTERNS)


def detect_medical_advice(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in MEDICAL_ADVICE_PATTERNS)


def detect_test_interpretation(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in TEST_INTERPRET_PATTERNS)


def detect_test_result_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in TEST_RESULT_PATTERNS)


def detect_test_assist_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in TEST_ASSIST_PATTERNS)


def detect_schedule_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in SCHEDULE_PATTERNS)


def detect_doc_request_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in DOC_REQUEST_PATTERNS)


def detect_appointment_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in APPOINTMENT_INTENT_PATTERNS)


def detect_appointment_action(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bотмен\w*", t):
        return "cancel"
    if re.search(r"\bперен\w*|\bперезапис\w*", t):
        return "reschedule"
    if re.search(r"\bзапис\w*|\bзапиш\w*", t):
        return "book"
    return None


def detect_price_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in PRICE_PATTERNS)


def detect_address_intent(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in ADDRESS_PATTERNS)


def extract_service_phrase(text: str) -> str | None:
    """
    Выделяет компактную фразу услуги/процедуры из пользовательского текста.
    Пример: "записаться на узи печени в самаре" -> "УЗИ печени".
    """
    if not isinstance(text, str) or not text.strip():
        return None

    low = text.lower()
    m = re.search(r"\b(" + "|".join(SERVICE_ANCHORS) + r")\b", low)
    if not m:
        return None

    tail = low[m.start():]
    tokens = re.findall(r"[a-zа-яё0-9:-]+", tail)
    if not tokens:
        return None

    service_tokens: list[str] = []
    for idx, tok in enumerate(tokens):
        if idx == 0:
            service_tokens.append(tok)
            continue
        if len(service_tokens) >= 5:
            break
        if tok in SERVICE_BOUNDARY_WORDS:
            break
        if ":" in tok or re.search(r"\d", tok):
            break
        service_tokens.append(tok)

    if not service_tokens:
        return None

    first = service_tokens[0]
    if first in SERVICE_UPPERCASE:
        first_out = first.upper()
    else:
        first_out = first.capitalize()

    if len(service_tokens) == 1:
        return first_out
    return " ".join([first_out, *service_tokens[1:]])


def missing_slots(label: str, entities: dict[str, Any]) -> list[str]:
    req = REQUIRED_SLOTS.get(label, [])
    missing: list[str] = []
    for r in req:
        if r.startswith("_any_of:"):
            keys = [k.strip() for k in r.split(":", 1)[1].split(",") if k.strip()]
            if not any(entities.get(k) for k in keys):
                missing.append(r)
        elif not entities.get(r):
            missing.append(r)
    if label == "APPOINTMENT" and entities.get("accepts_children") and not entities.get("child_age"):
        missing.append("child_age")
    return missing


def clarification_question(label: str, missing: list[str]) -> str:
    need_city = any(m.startswith("_any_of:city") for m in missing)
    need_service = any(
        "doctor_id" in m or "doctor_name" in m or "specialty" in m or "service_name" in m
        for m in missing
    )

    if "child_age" in missing:
        return "Сколько полных лет ребенку?"
    if label in {"PRICE", "TEST_ASSIST", "ADDRESS"} and need_city:
        return "Из какого города вы обращаетесь?"
    if label == "DOCTOR_SCHEDULE":
        return (
            "Чтобы показать расписание, нужна фамилия врача (или ID). "
            "Напишите, например: «расписание уролога Дразнина»."
        )
    if label == "DOCTOR_INFO":
        return "Какого врача или специалиста вы ищете? (например: «уролог», или фамилия врача)."
    if label == "PRICE":
        return "Скажите, пожалуйста, название услуги/анализа — я уточню стоимость."
    if label == "PREPARE":
        return "К какому анализу или исследованию нужна подготовка? Напишите название."
    if label == "TEST_ASSIST":
        return (
            "Для какой цели хотите подобрать анализы? "
            "Например: «проверить щитовидку», «витамины», «чекап»."
        )
    if label == "APPOINTMENT":
        if need_service:
            return (
                "Чтобы помочь с записью, уточните: к какому врачу/специалисту "
                "или на какую услугу вы хотите записаться?"
            )
        if need_city:
            return "Из какого города вы обращаетесь?"
        if "date_from" in missing or "time_from" in missing:
            return "Уточните, пожалуйста, дату и время записи."
        return "Уточните, пожалуйста, детали записи."
    if label == "TEST_RESULT":
        return (
            "Чтобы проверить готовность результатов, нужен номер заказа "
            "(обычно вида «№12345»). Если удобнее — подскажу, как пройти авторизацию."
        )
    return "Уточните, пожалуйста, детали запроса."


def evidence_requires_handoff(evidence: Any) -> tuple[bool, str | None, str | None]:
    items: dict[str, Any] = {}
    if isinstance(evidence, dict):
        items = evidence
    else:
        evidence_items = getattr(evidence, "items", None)
        if isinstance(evidence_items, dict):
            items = evidence_items

    for val in items.values():
        if isinstance(val, dict) and val.get("handoff_required"):
            reason = val.get("handoff_reason")
            reason_str = reason.strip() if isinstance(reason, str) and reason.strip() else None
            msg = val.get("handoff_message")
            if isinstance(msg, str) and msg.strip():
                return True, msg.strip(), reason_str
            return True, None, reason_str
    return False, None, None


def handoff_message(reason: str | None = None, override: str | None = None) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()
    if isinstance(reason, str) and reason in HANDOFF_REASON_MATRIX:
        return HANDOFF_REASON_MATRIX[reason]
    return HANDOFF_REASON_MATRIX["generic"]


def appointment_step_policy(entities: dict[str, Any]) -> str:
    branch_selected = bool(entities.get("branch_id") or entities.get("branch_name"))
    has_date_time = bool((entities.get("date_from") or entities.get("date_hint")) and entities.get("time_from"))
    if not branch_selected:
        return APPOINTMENT_STEP_BRANCH
    if not has_date_time:
        return APPOINTMENT_STEP_DATETIME
    if not entities.get("appointment_confirm_pending") and not entities.get("appointment_confirmed"):
        return APPOINTMENT_STEP_CONFIRM
    return APPOINTMENT_STEP_DONE


_YES_RE = re.compile(r"^\s*(да|ага|угу|подтверждаю|подтверждаем|верно|ок|окей)\s*[!.]?\s*$", re.I)
_NO_RE = re.compile(r"^\s*(нет|неа|не подтверждаю|не подтверждаем|неверно)\s*[!.]?\s*$", re.I)


def appointment_confirmation_transition(text: str) -> str:
    if _YES_RE.match(text or ""):
        return APPOINTMENT_CONFIRM_YES
    if _NO_RE.match(text or ""):
        return APPOINTMENT_CONFIRM_NO
    return APPOINTMENT_CONFIRM_OTHER


def extract_price_rub(price_payload: dict[str, Any] | None) -> str | None:
    if not isinstance(price_payload, dict):
        return None
    prices = price_payload.get("prices")
    if not isinstance(prices, list) or not prices:
        return None
    first = prices[0] if isinstance(prices[0], dict) else {}
    for k in ("servicePrice", "price", "service_price", "amount", "cost"):
        v = first.get(k)
        if isinstance(v, (int, float)):
            return f"{int(v)}"
        if isinstance(v, str):
            digits = re.sub(r"[^\d]", "", v)
            if digits:
                return digits
    return None


def appointment_summary(entities: dict[str, Any]) -> str:
    service = str(entities.get("service_name") or entities.get("test_name") or "услуга").strip()
    place = str(entities.get("branch_name") or entities.get("city") or "выбранный филиал").strip()
    date_part = str(entities.get("date_from") or entities.get("date_hint") or "уточним дату").strip()
    time_from = str(entities.get("time_from") or "").strip()
    time_to = str(entities.get("time_to") or "").strip()
    if time_from and time_to and time_from != time_to:
        time_part = f"{time_from}-{time_to}"
    elif time_from:
        time_part = time_from
    else:
        time_part = "уточним время"
    return f"Запись: {service}, {place}, {date_part}, {time_part}."


def appointment_addresses(address_payload: Any, branches: list[dict[str, str]], limit: int = 5) -> list[str]:
    return appointment_addresses_for_city(address_payload, branches, city=None, limit=limit)


def appointment_addresses_for_city(
    address_payload: Any,
    branches: list[dict[str, str]],
    city: str | None,
    limit: int = 5,
) -> list[str]:
    city_norm = re.sub(r"\s+", " ", str(city or "").strip().lower())

    if isinstance(address_payload, dict):
        addresses = address_payload.get("addresses")
        if isinstance(addresses, list):
            clean = [
                str(a).strip()
                for a in addresses
                if isinstance(a, str) and a.strip() and looks_like_address(str(a))
            ]
            if city_norm:
                city_filtered = [a for a in clean if city_norm in a.lower()]
                if city_filtered:
                    clean = city_filtered
            if clean:
                return clean[:limit]

    fallback: list[str] = []
    for b in branches:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name") or "").strip()
        if not name or not looks_like_address(name):
            continue
        aliases = str(b.get("aliases") or "").lower()
        hay = f"{name.lower()} | {aliases}"
        if city_norm and city_norm not in hay:
            continue
        fallback.append(name)

    return [x for x in fallback if x][:limit]


def appointment_text_branch_prompt(service: str, city: str, addresses: list[str]) -> str:
    if addresses:
        lines = "\n".join([f"- {a}" for a in addresses])
        city_part = f" в городе {city}" if city else ""
        return (
            f"Есть возможность записи на {service}{city_part} по адресам:\n"
            f"{lines}\n"
            "Какой филиал вам удобен?"
        )
    return "Уточните, пожалуйста, удобный филиал/адрес для записи."


def appointment_text_datetime_prompt(service: str, branch: str, price_rub: str | None) -> str:
    if price_rub:
        return f"Да, можем записать на {service} ({branch}), стоимость {price_rub} руб. На какую дату и время вам удобно?"
    return f"Да, можем записать на {service} ({branch}). На какую дату и время вам удобно?"


def appointment_text_confirm_prompt(summary: str) -> str:
    return f"{summary} Подтверждаете?"


def appointment_text_confirmed_handoff(summary: str) -> str:
    return f"{summary}\nПередаю заявку оператору для окончательного подтверждения записи."


def appointment_text_reask_datetime() -> str:
    return "Хорошо, тогда уточните новую дату и время для записи."


def appointment_text_reask_confirm() -> str:
    return "Подтвердите запись, пожалуйста: ответьте «да» или «нет»."


def doctor_schedule_text_clarify_doctor() -> str:
    return "Уточните, пожалуйста, фамилию врача."


def decision_handoff_reason(flags: set[str]) -> str:
    if "low_confidence" in flags:
        return "low_confidence"
    return "generic"


def decision_handoff_text(flags: set[str]) -> str:
    return handoff_message(decision_handoff_reason(flags))


_NOTE_RE = re.compile(r"📞\s*Заметка.*?(?=\n{2,}|\Z)", flags=re.S | re.I)

def sanitize_for_patient(text: str) -> str:
    if not text:
        return ""

    # удаляем служебные блоки/теги
    text = _NOTE_RE.sub("", text)
    text = text.replace("[SAFE_LIST]", "")
    text = text.replace("<NO_POSTPROC>", "")
    text = text.replace("RAW_FULL:", "")
    text = text.replace("SEGMENT_SEPARATOR", "")

    return text


def require_auth_for_test_result(state_is_authenticated: bool) -> tuple[bool, str]:
    if state_is_authenticated:
        return False, ""
    return True, (
        "Для получения результатов анализов нужна авторизация (медицинская тайна).\n\n"
        "Пожалуйста, пройдите авторизацию в личном кабинете/по ссылке от клиники или подтвердите личность в мессенджере."
    )


def low_confidence_policy(confidence: float, threshold: float = 0.55) -> bool:
    return confidence < threshold
