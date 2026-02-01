from __future__ import annotations

import re

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
]

SCHEDULE_PATTERNS = [
    r"\bрасписани(е|я)\b",
    r"\bграфик\b",
    r"\bкогда\b.*\bпринима(ет|ют)\b",
    r"\bна следующ(ей|ую)\s+недел",
    r"\bв ближайш(ие|ую)\s+7\s*дн",
]

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