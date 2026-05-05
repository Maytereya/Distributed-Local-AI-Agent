"""Централизованные policy-правила роутера.

Здесь находятся regex-детекторы интентов, slot/clarify/handoff политики,
тексты переходов APPOINTMENT flow и вспомогательные нормализаторы.
Ответственность модуля: единая точка бизнес-правил без доступа к внешним сервисам.
"""

from __future__ import annotations

from difflib import get_close_matches
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from .city import looks_like_address, match_city
from .doctor_name_port import resolve_cached_doctor_name_candidate, surname_variants
from .russian_nlu import normalize_ru
from .service_phrase import extract_service_phrase
from .specialty_parser import extract_specialty_from_text

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
    r"\bрезультат(ы|ов)\s+тест\w*",
    r"\bрезультат\b.*\b(анализ|тест)\w*",
    r"\bнуж\w*\s+результат\w*",
    r"\bполуч(ить|у|ите)\b.*\bрезультат\w*",
    r"\bанализ(ы)?\s+готов(ы|о)\b",
    r"\bготовност(ь|и)\s+анализ",
    r"\bрезультат(ы)?\s+готов",
    r"\bне\s+пришл\w*\s+.*\b(почт\w*|результат\w*)",
    r"\bне\s+выслал\w*\s+.*\b(почт\w*|результат\w*)",
    r"\bна\s+почт\w*\s+нет\b",
    r"\bсмс\b.*\bготов\w*",
    r"\bпришлите\b.*\b(pdf|пдф|файл)\b",
    r"\bскачать\b.*\bрезультат",
    r"\bбланк\b.*\bанализ",
    r"\bличн\w*\s+кабинет\w*",
]

TEST_ASSIST_PATTERNS = [
    r"\bподскажите\s+пожалуйста\b",
    r"\bанализ\w*",
    r"\b[сз]дат\w*\s+(?:кров\w*|моч\w*|анализ\w*)",
    r"\bкров\w*\s+на\s+анализ\w*",
    r"\bподобрат(ь|ь)\s+анализ",
    r"\bкакие анализ(ы)?\s+[сз]дат(ь|ь)\b",
    r"\bчто\s+[сз]дат(ь|ь)\b.*\bанализ",
    r"\bчекап\b",
    r"\bкомплекс\b.*\bанализ",
    r"\bскрининг\b",
    r"\bдо\s+скольки\b.*\b[сз]дат\w*",
    r"\bнатощак\b",
    r"\bреференд\w*",
    r"\bнорм\w*\s+значени\w*",
    r"\bгормон\w*",
    r"\bпотерял\w*\s+распечат\w*",
    r"\bраспечат\w*\s+анализ\w*",
    r"\b(оак|оам|ферритин|ттг|т3|т4|глюкоз|витамин\s*д)\b",
]

PREPARE_PATTERNS = [
    r"\bподготов\w*\b",
    r"\bкак\s+подготов\w*\b",
    r"\bперед\b.*\b(анализ\w*|узи|фгдс|гастроскоп\w*|кольпоскоп\w*|вульвоскоп\w*)\b",
    r"\b(натощак|на\s+голодный\s+желудок)\b",
    r"\bможно\s+ли\s+(есть|пить)\b.*\bперед\b",
    r"\b(утром|вечером|в\s+первую\s+половину\s+дня)\b.*\b[сз]дат\w*\b",
    r"\bможно\s+вечером\s+[сз]дат\w*\b",
    r"\bобязател\w*\s+ли\b.*\b(утром|в\s+первую\s+половину\s+дня)\b",
    r"\b(когда|во\s+сколько)\b.*\b[сз]дат\w*\b.*\b(анализ\w*|кров\w*|моч\w*)?\b",
]

SCHEDULE_PATTERNS = [
    r"\bрасписани(е|я)\b",
    r"\bграфик\b",
    r"\bкогда\b.*\bпринима(ет|ют)\b",
    r"\bсвободн\w*\s+окн\w*\b",
    r"\bкакие\s+есть\s+окн\w*\b",
    r"\bслот\w*\b",
    r"\bокн\w*\b.*\bпри(е|ё)м\w*\b",
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

TAX_DOC_REQUEST_PATTERNS = [
    r"\bфнс\b",
    r"\bналог\w*\s+вычет\w*",
    r"\bсправк\w*.*\bналог\w*",
    r"\bсправк\w*.*\bвычет\w*",
    r"\bсправк\w*.*\bоплат\w*.*\bмедицинск\w*.*\bуслуг\w*",
]

APPOINTMENT_INTENT_PATTERNS = [
    r"\bзапис\w*",
    r"\bзапиш\w*",
    r"\bзапись\b",
    r"\bпри(е|ё)м\w*",
    r"\bперен\w*",
    r"\bсмест\w*",
    r"\bперезапис\w*",
    r"\bотмен\w*",
    r"\bконсультаци\w*",
    r"\bхолтер\w*",
]

PRICE_PATTERNS = [
    r"\bстоим\w*",
    r"\bцен\w*",
    r"\bпрайс\w*",
    r"\bсколько\b.*\bстоит\b",
    # Аббревиатура анализа + модификатор «срочно/cito/экспресс» —
    # это de-facto price-вопрос («ОАК срочно», «cito ОАК»). Без
    # этого правила pipeline уходил в TEST_ASSIST flow и переспрашивал
    # «какие симптомы». Жалоба заказчика 2026-05-05.
    r"\b(?:оак|оам|алт|аст|алат|асат|ттг|сое|соэ|мно|пти|ггт|лдг|кфк)\b.*\b(?:сроч\w*|cito|экспресс\w*)\b",
    r"\b(?:сроч\w*|cito|экспресс\w*)\b.*\b(?:оак|оам|алт|аст|алат|асат|ттг|сое|соэ|мно|пти|ггт|лдг|кфк)\b",
]

ADDRESS_PATTERNS = [
    r"\bадрес\w*",
    r"\bфилиал\w*",
    r"\bотделени\w*",
    r"\bрежим\w*.*\bработ\w*",
    r"\bкак\b.*\bдобрат\w*",
    r"\bгде\b.*\bнаходит\w*",
]

WORK_HOURS_PATTERNS = [
    r"\bработа(ет|ете|ют)\b",
    r"\b(открыт\w*|закрыт\w*)\b",
    r"\b(до|со)\s+скольки\b",
    r"\bграфик\w*.*\bработ\w*",
    r"\bрежим\w*.*\bработ\w*",
]

NEWS_PATTERNS = [
    r"\bакци\w*",
    r"\bскидк\w*",
    r"\bспецпредложени\w*",
    r"\bдефицит\s+желез\w*",
    r"\bновост\w*",
]

DOCTOR_INFO_PATTERNS = [
    r"\bинф\w*\b.*\b(врач\w*|доктор\w*)\b",
    r"\bинформац\w*\b.*\b(врач\w*|доктор\w*)\b",
    r"\bрасскаж\w*\b.*\b(о|про)\b.*\b(врач\w*|доктор\w*)\b",
    r"\b(о|про)\b\s+(врач\w*|доктор\w*)\b",
    r"\bкто\b.*\b(врач\w*|доктор\w*)\b",
    r"\bчем\b.*\bзанима\w*\b",
]
_DOCTOR_INFO_HINT_RE = re.compile(
    r"\b(инф\w*|расскаж\w*|о\s+врач\w*|про\s+врач\w*|кто\s+так\w*|чем\s+занима\w*)\b",
    re.I,
)
_DOCTOR_SCHEDULE_HINT_RE = re.compile(r"\b(расписани\w*|график|окн\w*|слот\w*|когда\b.*\bпринима\w*)\b", re.I)
_SERVICE_LOCATION_QUERY_RE = re.compile(r"\bгде\b.*\b(сделать|пройти|сдать)\b", re.I)


def _compile_patterns(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.I) for p in patterns)


_URGENT_RE = _compile_patterns(URGENT_PATTERNS)
_COMPLAINT_RE = _compile_patterns(COMPLAINT_PATTERNS)
_MEDICAL_ADVICE_RE = _compile_patterns(MEDICAL_ADVICE_PATTERNS)
_TEST_INTERPRET_RE = _compile_patterns(TEST_INTERPRET_PATTERNS)
_TEST_RESULT_RE = _compile_patterns(TEST_RESULT_PATTERNS)
_TEST_ASSIST_RE = _compile_patterns(TEST_ASSIST_PATTERNS)
_PREPARE_RE = _compile_patterns(PREPARE_PATTERNS)
_SCHEDULE_RE = _compile_patterns(SCHEDULE_PATTERNS)
_DOC_REQUEST_RE = _compile_patterns(DOC_REQUEST_PATTERNS)
_TAX_DOC_REQUEST_RE = _compile_patterns(TAX_DOC_REQUEST_PATTERNS)
_APPOINTMENT_INTENT_RE = _compile_patterns(APPOINTMENT_INTENT_PATTERNS)
_PRICE_RE = _compile_patterns(PRICE_PATTERNS)
_ADDRESS_RE = _compile_patterns(ADDRESS_PATTERNS)
_WORK_HOURS_RE = _compile_patterns(WORK_HOURS_PATTERNS)
_NEWS_RE = _compile_patterns(NEWS_PATTERNS)
_DOCTOR_INFO_RE = _compile_patterns(DOCTOR_INFO_PATTERNS)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    t = text or ""
    return any(p.search(t) for p in patterns)

_DIAGNOSTIC_RE = re.compile(r"\b(экг|узи|мрт|кт|фгдс|фкс|рентген|флюорограф|колоноскоп|холтер)\b", re.I)
_DOCTOR_WORDS_RE = re.compile(
    r"\b(врач\w*|специалист\w*|кардиолог\w*|эндокринолог\w*|уролог\w*|гинеколог\w*|терапевт\w*|педиатр\w*|невролог\w*|лор\w*|хирург\w*|стоматолог\w*|гастроэнтеролог\w*|онколог\w*|проктолог\w*|дерматолог\w*|дерматовенеролог\w*|эндоскопист\w*|эндоскопи\w*|офтальмолог\w*)\b",
    re.I,
)
_DOCTOR_NAME_HINT_RE = re.compile(r"\bк\s+[А-ЯЁа-яё\-]{3,}\b")
_BOOK_ACTION_STRICT_RE = re.compile(r"\b(записат\w*|запиш\w*|записыва\w*)\b", re.I)
_DATE_TIME_SIGNAL_RE = re.compile(
    r"\b(сегодня|завтра|послезавтра|понедельник|вторник|среда|четверг|пятница|суббота|воскресенье|"
    r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{1,2}:\d{2}|\b(?:в|к)\s+\d{1,2}\s*(?:утра|дня|вечера|ночи)?)\b",
    re.I,
)
_RESULT_DELIVERY_QUESTION_RE = re.compile(r"\b(можно|куда)\b.*\b(почт\w*|придут)\b", re.I)
_ASSIST_PRICE_CONTEXT_RE = re.compile(
    r"\b(по\s+направлен\w*|данн\w*\s+анализ\w*|сдач\w*\s+анализ\w*)\b",
    re.I,
)
_NONBOOKABLE_ANALYSIS_RE = re.compile(
    r"\b(анализ\w*|лаборатор\w*|биоматериал\w*|[сз]дат\w*\s+(?:кров\w*|моч\w*|анализ\w*))\b",
    re.I,
)
_NONBOOKABLE_ECG_RE = re.compile(r"\b(экг|электрокардиограм\w*)\b", re.I)
_NONBOOKABLE_VISIT_RE = re.compile(r"\b(запис\w*|[сз]дат\w*|пройти|сделат\w*|хочу|нуж\w*|можно)\b", re.I)
_NONBOOKABLE_SUBMIT_RE = re.compile(r"\b[сз]дат\w*\b", re.I)
_NONBOOKABLE_PROFILE_RE = re.compile(r"\b(профил\w*|панел\w*)\b", re.I)
_DIAGNOSTIC_BOOKING_RE = re.compile(
    r"\b(сделат\w*|пройти|провест\w*|хочу|нуж\w*|можно|требует\w*|нужно)\b",
    re.I,
)
_TEST_SELECTION_RE = re.compile(
    r"\b(какие|какой|подобрат\w*|посовет\w*|чекап|чек[-\s]?ап|скрининг|для\s+чего|цель|по\s+направлен\w*)\b",
    re.I,
)
_QF_DMS_RE = re.compile(r"\bдмс\b", re.I)
_QF_OMS_RE = re.compile(r"\bомс\b", re.I)
_QF_PAID_RE = re.compile(r"\bплатн(о|ый|ая)\b|\bза наличн|\bоплат", re.I)
_QF_CHILD_RE = re.compile(r"\bдет(и|ям|ский|ская|ского|ских)\b", re.I)
_QF_AGE_RE = re.compile(r"\b(\d{1,2})\s*(?:лет|года|год)\b", re.I)
_QF_BRANCH_EXPLICIT_RE = re.compile(r"\bфилиал\b[:\s]*([^\n,;.]{2,80})", re.I)
_QF_BRANCH_ON_RE = re.compile(r"\bна\s+([А-ЯЁа-яё0-9\-]{3,40})(?:\s+([0-9]{1,4}))?\b")
_QF_BRANCH_CITY_PREFIX_RE = re.compile(r"^\s*(?:г(?:ород)?\.?\s*)?[А-ЯЁа-яё\-]{3,40}\s*,\s*", re.I)
_QF_BRANCH_IN_CITY_PREFIX_RE = re.compile(r"^\s*в\s+[А-ЯЁа-яё\-]{3,40}\s*,\s*", re.I)
_QF_BRANCH_FREEFORM_RE = re.compile(
    r"^\s*(?:на\s+)?(?:ул\.?|улица|пр\.?|проспект|пр-?т|шоссе|бульвар|пер\.?|переулок)?\s*"
    r"([А-ЯЁа-яё\-]{4,40})(?:\s*,?\s*([0-9]{1,4}[A-Za-zА-Яа-яЁё]?))?\s*$",
    re.I,
)
_NEAREST_SCHEDULE_HINT_RE = re.compile(
    r"\b(ближайш\w*|сам\w*\s+ранн\w*|раньше|поскорее|свободн\w*\s+окн\w*)\b",
    re.I,
)
_QF_TIME_FRAGMENT_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_QF_APPOINTMENT_WORD_RE = re.compile(r"\b(запис\w*|перен\w*|отмен\w*|при(е|ё)м\w*)\b", re.I)
_QF_APPOINTMENT_CHOICE_DOCTOR_RE = re.compile(
    r"\b(врач\w*|доктор\w*|специалист\w*|кто\s+принима\w*|покажи\s+врач\w*)\b",
    re.I,
)
_QF_APPOINTMENT_CHOICE_BRANCH_RE = re.compile(
    r"\b(филиал\w*|адрес\w*|отделени\w*|покажи\s+адрес\w*)\b",
    re.I,
)
_QF_TIME_FLEXIBLE_RE = re.compile(r"\b(люб\w+\s+время|когда\s+угодно|в\s+любое\s+время)\b", re.I)
_QF_ORDER_ID_RE = re.compile(r"(?:заказ|order|№)\s*([0-9]{4,})", re.I)
_QF_RESULT_SURNAME_RE = re.compile(r"\bфамили[яиюе]\s*[:\-]?\s*([А-ЯЁа-яё\-]{2,})", re.I)
_QF_RESULT_YEAR_RE = re.compile(r"\b(?:год\s*рождени[яея]|г\.?\s*р\.?)\s*[:\-]?\s*((?:19|20)\d{2})\b", re.I)
_QF_RESULT_FILIAL_RE = re.compile(r"\b(?:филиал|отделени[ея]|город)\s*[:\-]?\s*([A-Za-zА-Яа-яЁё0-9 .,\-]{2,80})", re.I)
_QF_RESULT_NUMBER_RE = re.compile(r"\b(?:номер|код)\s*(?:анализа)?\s*[:#№\-]?\s*(\d{3,})\b", re.I)
_QF_RESULT_ORDERED_RE = re.compile(
    r"^\s*([A-Za-zА-Яа-яЁё\-]{2,})\s*[,;]\s*((?:19|20)\d{2})\s*[,;]\s*([A-Za-zА-Яа-яЁё0-9\-]{1,20})\s*[,;]\s*(\d{3,})\s*$",
    re.I,
)
_QF_RESULT_ORDERED_SPACE_RE = re.compile(
    r"^\s*([A-Za-zА-Яа-яЁё\-]{2,})\s+((?:19|20)\d{2})\s+([A-Za-zА-Яа-яЁё0-9\-]{1,20})\s+(\d{3,})\s*$",
    re.I,
)
_QF_RESULT_SURNAME_STOPWORDS = {
    "хочу", "хотел", "хотела", "получить", "получу", "получите",
    "узнать", "подскажите", "покажите", "пришлите",
    "результат", "результаты", "тест", "тесты", "тестов", "анализ", "анализы", "анализов",
    "готов", "готово", "готовы", "нужен", "нужны",
}
_QF_TEST_WORDS_RE = re.compile(
    r"\b(анализ\w*|пцр|hba1c|глюкоз|витамин|ферритин|ттг|т4|т3|холестер|оак|оам|чекап|чек[-\s]?ап|щитовид|анеми|скрининг)\b",
    re.I,
)
_QF_PATIENT_NAME_PREFIX_RE = re.compile(
    r"\b(?:фио|ф\.?\s*и\.?\s*о\.?|меня\s+зовут|зовут)\b[:\s\-]*([А-ЯЁа-яё\-]{2,}(?:\s+[А-ЯЁа-яё\-]{2,}){1,2})",
    re.I,
)
_QF_PLAIN_NAME_RE = re.compile(r"^\s*([А-ЯЁа-яё\-]{2,}(?:\s+[А-ЯЁа-яё\-]{2,}){1,2})\s*$")
_QF_NAME_FRAGMENT_RE = re.compile(r"\b([А-ЯЁа-яё\-]{2,})\s+([А-ЯЁа-яё\-]{2,})\s+([А-ЯЁа-яё\-]{2,})\b")
_QF_PATIENT_NAME_STOPWORDS = {
    "на",
    "в",
    "во",
    "к",
    "с",
    "со",
    "до",
    "после",
    "утром",
    "днем",
    "днём",
    "вечером",
    "сегодня",
    "завтра",
    "послезавтра",
    "нет",
    "подождите",
    "подожди",
    "пока",
    "ладно",
    "извините",
    "не",
    "то",
    "это",
    "другое",
    "другой",
    "хотел",
    "хотела",
    "буду",
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
}
_QF_DOCTOR_CONTEXT_RE = re.compile(
    r"\b("
    r"расписани\w*|график|свободн\w*\s+(?:окн\w*|слот\w*)|"
    r"когда\s+принима\w*|принима\w*\s+когда|"
    r"к\s+(?:врач\w*\s+|доктор\w*\s+)?[А-ЯЁа-яё\-]{3,}"
    r")\b",
    re.I,
)
_QF_NEXT_WEEK_RE = re.compile(r"\bна следующ(ей|ую)\s+недел", re.I)
_QF_THIS_WEEK_RE = re.compile(r"\bна эт(ой|у)\s+недел|\bв эт(ой|у)\s+недел", re.I)
_QF_TOMORROW_RE = re.compile(r"\bзавтра\b", re.I)
_QF_TODAY_RE = re.compile(r"\bсегодня\b", re.I)
_QF_DATE_DOT_RE = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})(?:[.\-/](\d{2,4}))?\b")
_QF_DATE_ISO_RE = re.compile(r"\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b")
_QF_DATE_WORD_RE = re.compile(r"\b(\d{1,2})\s+([А-ЯЁа-яё]+)(?:\s+(\d{4}))?\b", re.I)
_QF_RANGE_WORD_RE = re.compile(
    r"\bс\s+(\d{1,2})\s+(?:по|-)\s+(\d{1,2})\s+([А-ЯЁа-яё]+)(?:\s+(\d{4}))?\b",
    re.I,
)
_QF_RANGE_DASH_RE = re.compile(
    r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s+([А-ЯЁа-яё]+)(?:\s+(\d{4}))?\b",
    re.I,
)
_QF_WEEKDAY_RE = re.compile(
    r"\b(в|во|на)\s+(понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье|пн|вт|ср|чт|пт|сб|вс)\b",
    re.I,
)
_QF_TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.](\d{2})\b")
_QF_AFTER_TIME_RE = re.compile(r"\b(после|с)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b", re.I)
_QF_BEFORE_TIME_RE = re.compile(r"\b(до|раньше)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b", re.I)
_QF_EXACT_TIME_RE = re.compile(r"\b(в|к)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b", re.I)
_QF_RANGE_TIME_RE = re.compile(
    r"\b(с)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\s+(до|-)\s+([01]?\d|2[0-3])(?:[:.](\d{2}))?\b",
    re.I,
)
_QF_WEEKDAYS = {
    "понедельник": 0, "пн": 0, "вторник": 1, "вт": 1, "среда": 2, "ср": 2, "четверг": 3, "чт": 3,
    "пятница": 4, "пт": 4, "суббота": 5, "сб": 5, "воскресенье": 6, "вс": 6,
}
_QF_MONTHS = {
    "января": 1, "январь": 1, "февраля": 2, "февраль": 2, "марта": 3, "март": 3, "апреля": 4, "апрель": 4,
    "мая": 5, "май": 5, "июня": 6, "июнь": 6, "июля": 7, "июль": 7, "августа": 8, "август": 8,
    "сентября": 9, "сентябрь": 9, "октября": 10, "октябрь": 10, "ноября": 11, "ноябрь": 11, "декабря": 12,
    "декабрь": 12,
}

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
    "TEST_RESULT": ["surname", "year", "filial", "number"],
    "DOCTOR_INFO": ["_any_of:specialty,doctor_id,doctor_name,service_name"],
    # Для расписания поддерживаем:
    # - конкретного врача (doctor_id/doctor_name)
    # - или специальность (specialty), если пользователь просит ближайшего врача по профилю.
    "DOCTOR_SCHEDULE": ["_any_of:doctor_id,doctor_name,specialty"],
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

CLARIFY_TEXT_MAP: dict[str, str] = {
    "DOCTOR_SCHEDULE": (
        "Чтобы показать расписание, нужна фамилия врача (или ID). "
        "Напишите, например: «расписание уролога Дразнина»."
    ),
    "DOCTOR_INFO": "Какого врача или специалиста вы ищете? (например: «уролог», или фамилия врача).",
    "PRICE": "Скажите, пожалуйста, название услуги/анализа — я уточню стоимость.",
    "PREPARE": "К какому анализу или исследованию нужна подготовка? Напишите название.",
    "TEST_ASSIST": (
        "Для какой цели хотите подобрать анализы? "
        "Например: «проверить щитовидку», «витамины», «чекап»."
    ),
}

TEST_RESULT_CLARIFY_MAP: dict[frozenset[str], str] = {
    frozenset({"surname"}): "Укажите, пожалуйста, фамилию пациента.",
    frozenset({"year"}): "Укажите год рождения пациента (например: 1985).",
    frozenset({"filial"}): "Укажите код анализа (например: Бг).",
    frozenset({"number"}): "Укажите номер анализа.",
}

APPOINTMENT_CLARIFY_MAP: dict[str, str] = {
    "need_service": (
        "Чтобы помочь с записью, уточните: к какому врачу/специалисту "
        "или на какую услугу вы хотите записаться?"
    ),
    "need_city": "Из какого города вы обращаетесь?",
    "need_patient": "Укажите, пожалуйста, ФИО пациента для оформления записи.",
    "need_datetime": "Уточните, пожалуйста, дату и время записи.",
    "default": "Уточните, пожалуйста, детали записи.",
}

# Centralized reason -> patient message map for operator handoff.
HANDOFF_REASON_MATRIX: dict[str, str] = {
    "manual_operator": "Соединяю с оператором по вашему запросу.",
    "doc_request_handoff": "Для оформления справок и документов подключаю оператора. Он уточнит детали и поможет с заявкой.",
    "test_result_fallback": "Для проверки и выдачи результатов анализов подключаю оператора. Это нужно для корректной идентификации пациента.",
    "ambiguous_price_service": "Сейчас по этой услуге безопаснее уточнить у оператора. Соединяю с оператором.",
    "city_not_supported": "Сейчас могу помочь только по Самаре. Соединяю с оператором.",
    "knowledge_not_found": "В моей базе данных информации недостаточно, перевожу на оператора.",
    "service_error": "Сейчас не удалось получить данные автоматически. Соединяю с оператором.",
    "service_error_doctors_list": "Сейчас не удалось получить список врачей автоматически. Соединяю с оператором.",
    "service_error_schedule": "Сейчас не удалось получить расписание автоматически. Соединяю с оператором.",
    # Процедуры с фиксированным оборудованием (флюорограф/маммограф на Ленина 5)
    # — у врача-рентгенолога нет приёмных слотов в Naika, запись ведёт регистратура.
    # Сообщение здесь — generic-фолбэк; при возможности передать реальный телефон
    # филиала вызов идёт через override= с подставленным номером.
    "schedule_via_registry_fixed_equipment": "Запись на флюорографию и маммографию ведётся через регистратуру филиала на пр. Ленина, 5. Соединяю с оператором.",
    "service_error_doctor_info": "Сейчас не удалось найти информацию автоматически. Соединяю с оператором.",
    "service_error_appointments": "Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
    "service_error_results": "Сейчас не удалось получить результаты автоматически. Соединяю с оператором.",
    "service_error_result_link": "Сейчас не удалось сформировать ссылку на результат автоматически. Соединяю с оператором.",
    "service_error_prices": "Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
    "renderer_error": "Сейчас не удалось сформировать ответ автоматически. Передаю диалог оператору.",
    "low_confidence": "Чтобы не ошибиться в ответе, подключаю оператора для уточнения деталей.",
    "generic": "Передаю диалог оператору.",
}

APPOINTMENT_STEP_BRANCH = "select_branch"
APPOINTMENT_STEP_DATETIME = "select_datetime"
APPOINTMENT_STEP_PATIENT = "collect_patient_name"
APPOINTMENT_STEP_CONFIRM = "confirm"
APPOINTMENT_STEP_DONE = "done"
APPOINTMENT_CONFIRM_YES = "yes"
APPOINTMENT_CONFIRM_NO = "no"
APPOINTMENT_CONFIRM_OTHER = "other"

APPOINTMENT_REPLY_MAP: dict[str, str] = {
    "patient_name_prompt": "Укажите, пожалуйста, ФИО пациента для оформления заявки на запись.",
    "confirm_suffix": "Подтверждаете?",
    "confirmed_handoff_suffix": "Передаю заявку оператору для окончательного подтверждения записи.",
    "reask_datetime": "Хорошо, тогда уточните новую дату и время для записи.",
    "reask_confirm": "Подтвердите запись, пожалуйста: ответьте «да» или «нет».",
    "cancel_confirm": "Отменить текущий процесс записи? Ответьте «да» или «нет».",
    "topic_switch_confirm": (
        "Сейчас идет оформление записи. Отменить этот процесс и перейти к новому вопросу? "
        "Ответьте «да» или «нет»."
    ),
    "cancelled": "Процесс записи отменён. Можете задать новый вопрос, и я помогу дальше.",
}


@dataclass(frozen=True)
class DoctorIntentOverridePolicy:
    """
    Политика мягкого повышения интента, когда врач уже подтвержден по кэшу.
    Нужна для фраз типа "к Дразнину какие есть окна?" без точного шаблона.
    """
    promote_from_labels: frozenset[str] = frozenset({"OTHER", "TEST_ASSIST", "ADDRESS", "NEWS"})


DOCTOR_INTENT_OVERRIDE_POLICY = DoctorIntentOverridePolicy()


@dataclass(frozen=True)
class UnsupportedCatalogMatch:
    """
    Результат точечного матчинга недоступных услуг/специалистов.

    :param kind: тип недоступного запроса
    :param canonical_name: каноническое имя услуги/роли/документа
    """

    kind: str
    canonical_name: str


_UNSUPPORTED_SERVICE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("мрт", re.compile(r"\bмрт\b", re.I)),
    ("кт", re.compile(r"\bкт\b", re.I)),
    ("рентген", re.compile(r"\bрентген\w*\b", re.I)),
    ("вакцинация", re.compile(r"\bвакцин\w*\b", re.I)),
    ("вакцинация", re.compile(r"\bпривив\w*\b", re.I)),
    ("вакцинация", re.compile(r"\bакдс(?:[-\s]*м)?\b", re.I)),
    ("вакцинация", re.compile(r"\bадс(?:[-\s]*м)?\b", re.I)),
    ("вакцинация", re.compile(r"\bкпк\b", re.I)),
    (
        "вакцинация",
        re.compile(
            r"\b(?:пентаксим|инфанрикс|полиорикс|имовакс\s+полио|превенар|приорикс|варилрикс|менактра|менвео|хиберикс|ротатек|ротарикс)\b",
            re.I,
        ),
    ),
    (
        "вакцинация",
        re.compile(
            r"\b(?:привив\w*|вакцин\w*|постав\w*|сдела\w*)\b[^.!?\n]{0,40}\b(?:полиомиелит\w*|полио\b|кор(?:ь|и)\b|краснух\w*|паротит\w*|ветрян\w*|пневмокок\w*|грипп\w*|гепатит(?:а|в|с|b|c)?)\b",
            re.I,
        ),
    ),
    (
        "вакцинация",
        re.compile(
            r"\b(?:полиомиелит\w*|полио\b|кор(?:ь|и)\b|краснух\w*|паротит\w*|ветрян\w*|пневмокок\w*|грипп\w*|гепатит(?:а|в|с|b|c)?)\b[^.!?\n]{0,40}\b(?:привив\w*|вакцин\w*|постав\w*|сдела\w*)\b",
            re.I,
        ),
    ),
)
_UNSUPPORTED_SPECIALIST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("офтальмолог", re.compile(r"\bофтальмолог\w*\b", re.I)),
    ("детский хирург", re.compile(r"\bдет\w*\s+хирург\w*\b", re.I)),
    ("детский уролог", re.compile(r"\bдет\w*\s+уролог\w*\b", re.I)),
    ("детский кардиолог", re.compile(r"\bдет\w*\s+кардиолог\w*\b", re.I)),
    ("косметолог", re.compile(r"\bкосметолог\w*\b", re.I)),
    ("челюстно-лицевой хирург", re.compile(r"\bчелюстно[-\s]*лицев\w*\s+хирург\w*\b", re.I)),
    ("сурдолог", re.compile(r"\bсурдолог\w*\b", re.I)),
    ("психиатр", re.compile(r"\bпсихиатр\w*\b", re.I)),
    ("нарколог", re.compile(r"\bнарколог\w*\b", re.I)),
)
_UNSUPPORTED_DOCUMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("справка в гибдд", re.compile(r"\bгибдд\b", re.I)),
    ("справка в гибдд", re.compile(r"\bсправк\w*\b[^.!?\n]{0,40}\bгибдд\b", re.I)),
    ("медкомиссия для спортсменов", re.compile(r"\bмед\s*комисс\w*\b[^.!?\n]{0,40}\b(?:спортсмен\w*|соревнован\w*)\b", re.I)),
    ("медкомиссия для спортсменов", re.compile(r"\bмедкомисс\w*\b[^.!?\n]{0,40}\b(?:спортсмен\w*|соревнован\w*)\b", re.I)),
    ("медкомиссия для спортсменов", re.compile(r"\b(?:спортсмен\w*|соревнован\w*)\b[^.!?\n]{0,60}\b(справк\w*|допуск\w*|медкомисс\w*)\b", re.I)),
    ("медкомиссия для спортсменов", re.compile(r"\bзаказ\w*\b[^.!?\n]{0,40}\bсправк\w*\b[^.!?\n]{0,40}\b(?:спортсмен\w*|соревнован\w*)\b", re.I)),
    ("медкомиссия для спортсменов", re.compile(r"\bсправк\w*\b[^.!?\n]{0,40}\b(?:для\s+)?(?:спортсмен\w*|соревнован\w*)\b", re.I)),
)

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
    return _matches_any(text, _URGENT_RE)


def detect_complaint(text: str) -> bool:
    return _matches_any(text, _COMPLAINT_RE)


def detect_medical_advice(text: str) -> bool:
    return _matches_any(text, _MEDICAL_ADVICE_RE)


def detect_test_interpretation(text: str) -> bool:
    return _matches_any(text, _TEST_INTERPRET_RE)


def detect_test_result_intent(text: str) -> bool:
    if _matches_any(text, _TEST_RESULT_RE):
        return True
    # Поддержка "чистого" ввода реквизитов без слов "результат/анализ":
    # "Иванов, 1989, Бг, 1234" или "Иванов 1989 Бг 1234"
    if _QF_RESULT_ORDERED_RE.match(text or ""):
        return True
    if _QF_RESULT_ORDERED_SPACE_RE.match(text or ""):
        return True
    return False


def detect_test_assist_intent(text: str) -> bool:
    return _matches_any(text, _TEST_ASSIST_RE)


def detect_prepare_intent(text: str) -> bool:
    return _matches_any(text, _PREPARE_RE)


def detect_schedule_intent(text: str) -> bool:
    return _matches_any(text, _SCHEDULE_RE)


def detect_doc_request_intent(text: str) -> bool:
    return _matches_any(text, _DOC_REQUEST_RE)


def detect_tax_doc_request_intent(text: str) -> bool:
    return _matches_any(text, _TAX_DOC_REQUEST_RE)


def detect_appointment_intent(text: str) -> bool:
    if _matches_any(text, _APPOINTMENT_INTENT_RE):
        return True
    t = text or ""
    # Поддержка естественных формулировок без слова "записаться":
    # "Хочу сделать УЗИ брюшной полости", "Можно пройти МРТ?".
    # Для диагностических услуг такие фразы считаем сценарием записи/подбора адреса.
    if _DIAGNOSTIC_RE.search(t) and _DIAGNOSTIC_BOOKING_RE.search(t):
        return True
    return False


def detect_appointment_action(text: str) -> str | None:
    t = text.lower()
    has_cancel = bool(re.search(r"\bотмен\w*", t))
    has_reschedule = bool(re.search(r"\bперен\w*|\bперезапис\w*|\bсмест\w*", t))
    has_book = bool(re.search(r"\bзапис\w*|\bзапиш\w*", t))
    if has_cancel and has_reschedule:
        return "ambiguous"
    if has_cancel:
        return "cancel"
    if has_reschedule:
        return "reschedule"
    if has_book:
        return "book"
    return None


def detect_price_intent(text: str) -> bool:
    return _matches_any(text, _PRICE_RE)


def _looks_like_service_location_query(text: str) -> bool:
    """
    Определяет вопрос о месте оказания услуги по формулировке пользователя.

    :param text: исходный текст пользователя
    :return: True, если запрос похож на «где можно сделать <услугу>»
    """

    raw = str(text or "").strip()
    if not raw or not _SERVICE_LOCATION_QUERY_RE.search(raw):
        return False
    return bool(extract_service_phrase(raw))


def detect_address_intent(text: str) -> bool:
    raw = str(text or "").strip()
    return _matches_any(raw, _ADDRESS_RE) or _looks_like_service_location_query(raw)


def detect_branch_hours_intent(text: str) -> bool:
    """
    Определяет вопросы о графике работы филиалов без явного слова `адрес`.

    :param text: текст пользователя
    :return: True, если вопрос похож на запрос о режиме работы/открытии филиала
    """

    raw = str(text or "").strip()
    if not raw:
        return False
    if detect_price_intent(raw):
        return False
    return _matches_any(raw, _WORK_HOURS_RE)


def detect_news_intent(text: str) -> bool:
    return _matches_any(text, _NEWS_RE)


def detect_doctor_info_intent(text: str) -> bool:
    return _matches_any(text, _DOCTOR_INFO_RE)


def detect_unsupported_catalog(text: str) -> UnsupportedCatalogMatch | None:
    """
    Определяет запросы по услугам/врачам, которых в клинике нет.

    Матчер намеренно узкий и опирается только на заранее утвержденный
    бизнес-каталог, чтобы не перехватывать обычные медицинские запросы.

    :param text: исходный текст пользователя
    :return: описание найденного недоступного кейса или None
    """

    raw = str(text or "").strip()
    if not raw:
        return None

    for canonical_name, pattern in _UNSUPPORTED_DOCUMENT_PATTERNS:
        if pattern.search(raw):
            return UnsupportedCatalogMatch(
                kind="unsupported_document_service",
                canonical_name=canonical_name,
            )

    for canonical_name, pattern in _UNSUPPORTED_SPECIALIST_PATTERNS:
        if pattern.search(raw):
            return UnsupportedCatalogMatch(
                kind="unsupported_specialist",
                canonical_name=canonical_name,
            )

    for canonical_name, pattern in _UNSUPPORTED_SERVICE_PATTERNS:
        if pattern.search(raw):
            return UnsupportedCatalogMatch(
                kind="unsupported_service",
                canonical_name=canonical_name,
            )

    return None


def extract_specialty(text: str) -> str | None:
    value = extract_specialty_from_text(text or "")
    return value or None


def has_nearest_schedule_hint(text: str) -> bool:
    return bool(_NEAREST_SCHEDULE_HINT_RE.search(text or ""))


def apply_verified_doctor_override(label: str, flags: set[str], text: str) -> tuple[str, set[str]]:
    """
    Если doctor_name уже верифицирован, переводим слабый label в профильный doctor intent.
    Это снижает ложные handoff на естественных формулировках пользователя.
    """
    if "doctor_name_verified" not in flags:
        return label, flags
    if label in {"DOCTOR_SCHEDULE", "DOCTOR_INFO", "APPOINTMENT", "PRICE"}:
        return label, flags
    if label not in DOCTOR_INTENT_OVERRIDE_POLICY.promote_from_labels:
        return label, flags

    out_flags = set(flags)
    low = text.lower()
    if detect_schedule_intent(low) or _DOCTOR_SCHEDULE_HINT_RE.search(low):
        out_flags.add("policy_promote_doctor_schedule")
        return "DOCTOR_SCHEDULE", out_flags
    if detect_doctor_info_intent(low) or _DOCTOR_INFO_HINT_RE.search(low):
        out_flags.add("policy_promote_doctor_info")
        return "DOCTOR_INFO", out_flags

    out_flags.add("policy_promote_doctor_info")
    return "DOCTOR_INFO", out_flags


def normalize_appointment_action(action: str | None, text: str) -> str | None:
    if action == "ambiguous":
        return "unknown"
    if action == "book" and not _BOOK_ACTION_STRICT_RE.search(text or ""):
        return None
    return action


def has_datetime_signal(text: str) -> bool:
    return bool(_DATE_TIME_SIGNAL_RE.search(text or ""))


def has_appointment_context(text: str, last_entities: dict[str, Any], appointment_action: str | None) -> bool:
    ctx = last_entities or {}
    return bool(
        _DIAGNOSTIC_RE.search(text or "")
        or _DOCTOR_WORDS_RE.search(text or "")
        or _DOCTOR_NAME_HINT_RE.search(text or "")
        or ctx.get("doctor_id")
        or ctx.get("doctor_name")
        or ctx.get("service_name")
        or appointment_action is not None
    )


def is_address_dominant_intent(
    text: str,
    *,
    address_intent: bool,
    price_intent: bool,
    appointment_action: str | None,
) -> bool:
    return address_intent and not price_intent and appointment_action is None and not has_datetime_signal(text)


def is_price_dominant_intent(
    text: str,
    *,
    price_intent: bool,
    appointment_action: str | None,
) -> bool:
    dominant = price_intent and appointment_action in {None, "book"} and not has_datetime_signal(text)
    if not dominant:
        return False
    if re.search(r"\b(стоим\w*|цен\w*|сколько\b.*\bстоит)\b", text or "", flags=re.I):
        return True
    if detect_test_assist_intent(text) and _ASSIST_PRICE_CONTEXT_RE.search(text or ""):
        return False
    return True


def should_treat_result_delivery_as_test_assist(text: str) -> bool:
    low = (text or "").lower()
    return (
        detect_test_assist_intent(text)
        and bool(_RESULT_DELIVERY_QUESTION_RE.search(text or ""))
        and "не высл" not in low
        and "почему" not in low
        and "личн" not in low
    )


def _has_analysis_context_entities(entities: dict[str, Any] | None = None) -> bool:
    ent = entities or {}
    if ent.get("test_name") or ent.get("test_goal"):
        return True
    service_name = str(ent.get("service_name") or "").strip()
    if not service_name:
        return False
    if _NONBOOKABLE_ANALYSIS_RE.search(service_name):
        return True
    if _NONBOOKABLE_PROFILE_RE.search(service_name):
        return True
    if _QF_TEST_WORDS_RE.search(service_name):
        return True
    return False


def nonbookable_service_hint(text: str, entities: dict[str, Any] | None = None) -> str | None:
    t = text or ""
    has_analysis = bool(_NONBOOKABLE_ANALYSIS_RE.search(t))
    if not has_analysis and _NONBOOKABLE_PROFILE_RE.search(t):
        has_analysis = True
    if not has_analysis and _has_analysis_context_entities(entities):
        has_analysis = True
    has_ecg = bool(_NONBOOKABLE_ECG_RE.search(t))
    if has_analysis and has_ecg:
        return "анализы и ЭКГ"
    if has_analysis:
        return "анализы"
    if has_ecg:
        return "ЭКГ"
    return None


def detect_nonbookable_walkin_intent(text: str, entities: dict[str, Any] | None = None) -> bool:
    """
    ЭКГ и сдача анализов принимаются без записи (живая очередь),
    поэтому запросы "записаться на ЭКГ/анализы" переводим в ADDRESS flow.
    """
    t = text or ""
    if not t.strip():
        return False
    # Вопрос "как подготовиться" должен идти в PREPARE, а не в ADDRESS.
    if detect_prepare_intent(t):
        return False
    if detect_test_result_intent(t):
        return False
    if _TEST_SELECTION_RE.search(t):
        return False
    ent = entities or {}
    has_nonbookable = bool(_NONBOOKABLE_ANALYSIS_RE.search(t) or _NONBOOKABLE_ECG_RE.search(t))
    if not has_nonbookable:
        # Контекстный кейс follow-up:
        # "Диабетический профиль 1 где можно сдать?" после подбора анализов.
        has_profile_hint = bool(_NONBOOKABLE_PROFILE_RE.search(t))
        has_submit_verb = bool(_NONBOOKABLE_SUBMIT_RE.search(t))
        has_analysis_context = _has_analysis_context_entities(ent)
        if has_submit_verb and (has_profile_hint or has_analysis_context):
            has_nonbookable = True
    if not has_nonbookable:
        return False
    compact = [w for w in re.split(r"\s+", t.strip()) if w]
    short_direct = len(compact) <= 2
    if not short_direct and not (_NONBOOKABLE_VISIT_RE.search(t) or _BOOK_ACTION_STRICT_RE.search(t)):
        return False

    has_doctor_context = bool(ent.get("doctor_name") or ent.get("doctor_id") or ent.get("specialty"))
    if _BOOK_ACTION_STRICT_RE.search(t) and _DOCTOR_WORDS_RE.search(t):
        return False
    if has_doctor_context and _BOOK_ACTION_STRICT_RE.search(t):
        return False
    return True


def normalize_loose_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize_loose(s: str) -> list[str]:
    s = normalize_loose_text(s)
    s = re.sub(r"[^a-zа-яё0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if t]


def looks_like_branch_hint(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if _QF_TIME_FRAGMENT_RE.search(s):
        return False
    if _QF_APPOINTMENT_WORD_RE.search(s):
        return False
    if not re.search(r"[А-Яа-яЁё]", s):
        return False
    return looks_like_address(s)


def branch_options_to_indexable(branch_options: Any) -> list[dict[str, str]]:
    if not isinstance(branch_options, list):
        return []
    out: list[dict[str, str]] = []
    for i, raw in enumerate(branch_options, start=1):
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name:
            continue
        out.append({"id": f"shown_{i}", "name": name, "aliases": normalize_loose_text(name)})
    return out


def build_branch_index(branches: list[dict[str, str]]) -> list[dict[str, Any]]:
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
            for t in _tokenize_loose(item):
                tokens.add(t)
        idx.append({"id": bid, "name": name, "aliases": alias_list, "tokens": tokens})
    return idx


def match_branch_hint(text: str, branch_index: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    if not branch_index:
        return None, None
    txt_tokens = set(_tokenize_loose(text))
    if not txt_tokens:
        return None, None
    input_numbers = {t for t in txt_tokens if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)}
    best_id: str | None = None
    best_name: str | None = None
    best_score = 0
    for b in branch_index:
        if input_numbers:
            branch_numbers = {t for t in b["tokens"] if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)}
            if branch_numbers and not (input_numbers & branch_numbers):
                continue
        common = txt_tokens & b["tokens"]
        score = len(common) + (1 if any(len(t) >= 5 for t in common) else 0)
        if score > best_score:
            best_score = score
            best_id = b["id"]
            best_name = b["name"]
    if best_score >= 2:
        return best_id, best_name

    medium_candidates: list[tuple[str, str]] = []
    for b in branch_index:
        if input_numbers:
            branch_numbers = {t for t in b["tokens"] if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)}
            if branch_numbers and not (input_numbers & branch_numbers):
                continue
        common = txt_tokens & b["tokens"]
        if any(len(t) >= 5 for t in common):
            medium_candidates.append((b["id"], b["name"]))
    if len(medium_candidates) == 1:
        return medium_candidates[0]

    for b in branch_index:
        if input_numbers:
            branch_numbers = {t for t in b["tokens"] if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)}
            if branch_numbers and not (input_numbers & branch_numbers):
                continue
        common = txt_tokens & b["tokens"]
        if any(len(t) >= 7 for t in common):
            return b["id"], b["name"]
    # typo-tolerant fallback for short branch answers ("победы 38" -> "победы 83")
    norm_text = normalize_loose_text(text)
    if len(norm_text) >= 4:
        by_phrase: dict[str, tuple[str, str]] = {}
        for b in branch_index:
            options = [b.get("name", ""), *b.get("aliases", [])]
            for opt in options:
                phrase = normalize_loose_text(str(opt or ""))
                if len(phrase) < 4:
                    continue
                if input_numbers:
                    phrase_numbers = {t for t in _tokenize_loose(phrase) if re.fullmatch(r"\d{1,4}[a-zа-яё]?", t)}
                    if phrase_numbers and not (input_numbers & phrase_numbers):
                        continue
                by_phrase.setdefault(phrase, (b["id"], b["name"]))
        if by_phrase:
            best = get_close_matches(norm_text, list(by_phrase.keys()), n=1, cutoff=0.84)
            if best:
                return by_phrase[best[0]]
    return None, None


def _next_weekday(from_date: date, target_weekday: int) -> date:
    delta = (target_weekday - from_date.weekday()) % 7
    return from_date + timedelta(days=delta)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except Exception:
        return None


def parse_date_time_ru(text: str, today: date | None = None) -> dict[str, Any]:
    if today is None:
        today = datetime.now().date()
    out: dict[str, Any] = {}
    s = (text or "").strip()
    low = s.lower()

    if _QF_NEXT_WEEK_RE.search(low):
        out["date_hint"] = "next_week"
    elif _QF_THIS_WEEK_RE.search(low):
        out["date_hint"] = "this_week"
    elif _QF_TOMORROW_RE.search(low):
        out["date_hint"] = "tomorrow"
    elif _QF_TODAY_RE.search(low):
        out["date_hint"] = "today"

    m = _QF_RANGE_WORD_RE.search(s) or _QF_RANGE_DASH_RE.search(s)
    if m:
        d1, d2 = int(m.group(1)), int(m.group(2))
        mon_word = m.group(3).lower()
        y = m.group(4)
        month = _QF_MONTHS.get(mon_word)
        if month:
            year = int(y) if y else today.year
            dt1 = _safe_date(year, month, d1)
            dt2 = _safe_date(year, month, d2)
            if not y and dt2 and dt2 < today:
                dt1 = _safe_date(year + 1, month, d1)
                dt2 = _safe_date(year + 1, month, d2)
            if dt1 and dt2:
                if dt2 < dt1:
                    dt1, dt2 = dt2, dt1
                out["date_from"], out["date_to"] = dt1.isoformat(), dt2.isoformat()
                out.pop("date_hint", None)

    if "date_from" not in out:
        mi = _QF_DATE_ISO_RE.search(s)
        if mi:
            year, mth, d = int(mi.group(1)), int(mi.group(2)), int(mi.group(3))
            dt = _safe_date(year, mth, d)
            if dt:
                out["date_from"] = out["date_to"] = dt.isoformat()
                out.pop("date_hint", None)

    if "date_from" not in out:
        md = _QF_DATE_DOT_RE.search(s)
        if md:
            d, mth = int(md.group(1)), int(md.group(2))
            y = md.group(3)
            year = int(y) if y else today.year
            if year < 100:
                year += 2000
            dt = _safe_date(year, mth, d)
            if dt and not y and dt < today:
                dt = _safe_date(year + 1, mth, d)
            if dt:
                out["date_from"] = out["date_to"] = dt.isoformat()
                out.pop("date_hint", None)

    if "date_from" not in out:
        mw = _QF_DATE_WORD_RE.search(s)
        if mw:
            d = int(mw.group(1))
            month = _QF_MONTHS.get(mw.group(2).lower())
            y = mw.group(3)
            if month:
                year = int(y) if y else today.year
                dt = _safe_date(year, month, d)
                if dt and not y and dt < today:
                    dt = _safe_date(year + 1, month, d)
                if dt:
                    out["date_from"] = out["date_to"] = dt.isoformat()
                    out.pop("date_hint", None)

    if "date_from" not in out:
        wd = _QF_WEEKDAY_RE.search(s)
        if wd:
            token = wd.group(2).lower()
            token = {
                "среду": "среда", "пятницу": "пятница", "субботу": "суббота", "воскресенье": "воскресенье",
                "понедельник": "понедельник", "вторник": "вторник", "четверг": "четверг",
            }.get(token, token)
            target = _QF_WEEKDAYS.get(token)
            if target is not None:
                dt = _next_weekday(today, target)
                out["date_from"] = out["date_to"] = dt.isoformat()
                out.pop("date_hint", None)

    tr = _QF_RANGE_TIME_RE.search(s)
    if tr:
        h1, m1 = int(tr.group(2)), int(tr.group(3)) if tr.group(3) else 0
        h2, m2 = int(tr.group(5)), int(tr.group(6)) if tr.group(6) else 0
        out["time_from"] = f"{h1:02d}:{m1:02d}"
        out["time_to"] = f"{h2:02d}:{m2:02d}"
    if "time_from" not in out:
        a = _QF_AFTER_TIME_RE.search(s)
        if a:
            h, m = int(a.group(2)), int(a.group(3)) if a.group(3) else 0
            out["time_from"] = f"{h:02d}:{m:02d}"
    if "time_to" not in out:
        b = _QF_BEFORE_TIME_RE.search(s)
        if b:
            h, m = int(b.group(2)), int(b.group(3)) if b.group(3) else 0
            out["time_to"] = f"{h:02d}:{m:02d}"
    if "time_from" not in out and "time_to" not in out:
        ex = _QF_EXACT_TIME_RE.search(s)
        if ex:
            h, m = int(ex.group(2)), int(ex.group(3)) if ex.group(3) else 0
            tm = f"{h:02d}:{m:02d}"
            out["time_from"] = out["time_to"] = tm
    if "time_from" not in out and "time_to" not in out:
        bare = _QF_TIME_RE.search(s)
        if bare:
            h, m = int(bare.group(1)), int(bare.group(2))
            tm = f"{h:02d}:{m:02d}"
            out["time_from"] = out["time_to"] = tm
    return out


def _doctor_candidate_is_contextual(text: str, candidate: str) -> bool:
    """
    Кандидат врача валиден только при явном doctor/schedule контексте.
    Это защищает от ложных срабатываний вида "Здравствуйте" -> doctor_name.
    """
    t = str(text or "")
    cand_raw = str(candidate or "").strip()
    cand = normalize_loose_text(cand_raw)
    if not cand_raw or not cand:
        return False
    if not _QF_DOCTOR_CONTEXT_RE.search(t):
        return False
    c = re.escape(cand_raw)
    near_patterns = (
        rf"\b(?:к|про|о)\s+(?:врач\w*\s+|доктор\w*\s+)?{c}\b",
        rf"\b(?:расписани\w*|график|слот\w*|окн\w*|когда\s+принима\w*|принима\w*)\b[^.!?\n]{{0,40}}\b{c}\b",
        rf"\b{c}\b[^.!?\n]{{0,40}}\b(?:расписани\w*|график|слот\w*|окн\w*|когда\s+принима\w*|принима\w*)\b",
    )
    return any(re.search(p, t, re.I) for p in near_patterns)


def _missing_rules_include(missing_rules: list[str], *fragments: str) -> bool:
    """Проверяет, упоминаются ли нужные слоты в списке missing-rules.

    :param missing_rules: список правил незаполненных слотов
    :param fragments: фрагменты имён слотов для поиска
    :return: ``True``, если хотя бы один фрагмент найден
    """

    return any(any(fragment in rule for fragment in fragments) for rule in missing_rules)


def _fill_insurance_entities(text: str, out: dict[str, Any]) -> None:
    """Быстро извлекает сущности по страховке и детскому контексту.

    :param text: исходный текст пользователя
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    low = str(text or "").strip().lower()
    if _QF_DMS_RE.search(low):
        out["insurance_type"] = "dms"
    elif _QF_OMS_RE.search(low):
        out["insurance_type"] = "oms"
    elif _QF_PAID_RE.search(low):
        out["insurance_type"] = "paid"

    if _QF_CHILD_RE.search(low):
        out["accepts_children"] = True


def _fill_datetime_entities(text: str, out: dict[str, Any]) -> None:
    """Извлекает дату, время и флаг гибкого времени.

    :param text: исходный текст пользователя
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    t = str(text or "").strip()
    low = t.lower()
    dt = parse_date_time_ru(t)
    out.update({k: v for k, v in dt.items() if v is not None})
    if _QF_TIME_FLEXIBLE_RE.search(low):
        out["time_flexible"] = True


def _fill_test_result_entities(text: str, missing_rules: list[str], out: dict[str, Any]) -> None:
    """Извлекает номер заказа и данные для сценария получения результатов.

    :param text: исходный текст пользователя
    :param missing_rules: список правил незаполненных слотов
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    t = str(text or "").strip()
    low = t.lower()

    m_oid = _QF_ORDER_ID_RE.search(t)
    if m_oid:
        out["order_id"] = m_oid.group(1)

    if not _missing_rules_include(missing_rules, "surname", "year", "filial", "number"):
        return

    ordered = _QF_RESULT_ORDERED_RE.match(t)
    if ordered:
        out["surname"] = ordered.group(1).strip().capitalize()
        out["year"] = int(ordered.group(2))
        out["filial"] = ordered.group(3).strip()
        out["number"] = int(ordered.group(4))
    else:
        ordered_space = _QF_RESULT_ORDERED_SPACE_RE.match(t)
        if ordered_space:
            out["surname"] = ordered_space.group(1).strip().capitalize()
            out["year"] = int(ordered_space.group(2))
            out["filial"] = ordered_space.group(3).strip()
            out["number"] = int(ordered_space.group(4))

    m_surname = _QF_RESULT_SURNAME_RE.search(t)
    if m_surname:
        out["surname"] = m_surname.group(1).strip().capitalize()

    m_year = _QF_RESULT_YEAR_RE.search(t)
    if m_year:
        out["year"] = int(m_year.group(1))
    elif re.fullmatch(r"\s*(?:19|20)\d{2}\s*", t):
        out["year"] = int(t.strip())
    else:
        y_any = re.search(r"\b((?:19|20)\d{2})\b", t)
        if y_any:
            out["year"] = int(y_any.group(1))

    m_filial = _QF_RESULT_FILIAL_RE.search(t)
    if m_filial:
        out["filial"] = m_filial.group(1).strip(" ,.")
    elif "filial" in missing_rules:
        city_guess = match_city(t)
        if city_guess:
            out["filial"] = city_guess
        words = [w for w in re.split(r"\s+", t) if w]
        if "filial" not in out and 1 <= len(words) <= 3 and not re.search(r"\d", t) and not _QF_APPOINTMENT_WORD_RE.search(low):
            out["filial"] = t.strip(" ,.")

    m_number = _QF_RESULT_NUMBER_RE.search(t)
    if m_number:
        out["number"] = int(m_number.group(1))
    elif m_oid:
        out["number"] = int(m_oid.group(1))
    elif re.fullmatch(r"\s*\d{3,}\s*", t):
        out["number"] = int(t.strip())
    else:
        nums = [int(x) for x in re.findall(r"\b\d{3,}\b", t)]
        if nums:
            year_val = out.get("year")
            filtered = [n for n in nums if year_val is None or n != year_val]
            if filtered:
                out["number"] = filtered[-1]

    if "surname" in missing_rules and "surname" not in out:
        words = [w for w in re.split(r"\s+", t) if w]
        if (
            len(words) == 1
            and re.fullmatch(r"[А-Яа-яЁё\-]{2,}", words[0])
            and words[0].lower() not in _QF_RESULT_SURNAME_STOPWORDS
        ):
            out["surname"] = words[0].capitalize()


def _fill_appointment_entities(
    text: str,
    state_entities: dict[str, Any],
    missing_rules: list[str],
    out: dict[str, Any],
) -> None:
    """Извлекает сущности записи: специальность, врача, тест/услугу.

    :param text: исходный текст пользователя
    :param state_entities: уже накопленные сущности состояния
    :param missing_rules: список правил незаполненных слотов
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    t = str(text or "").strip()
    low = t.lower()

    spec = extract_specialty(low)
    if spec:
        out["specialty"] = spec

    needs_doctor_or_spec = _missing_rules_include(missing_rules, "doctor", "specialty")
    patient_name_like_text = bool(_QF_PATIENT_NAME_PREFIX_RE.search(t) or _QF_PLAIN_NAME_RE.match(t))
    doctor_already_selected = bool(state_entities.get("doctor_name") or state_entities.get("doctor_id"))
    if needs_doctor_or_spec and not patient_name_like_text and not doctor_already_selected:
        extracted_name = resolve_cached_doctor_name_candidate(t)
        if extracted_name and _doctor_candidate_is_contextual(t, extracted_name):
            out["doctor_name"] = extracted_name

    needs_test = _missing_rules_include(missing_rules, "test_goal", "test_name")
    if needs_test and _QF_TEST_WORDS_RE.search(low) and not state_entities.get("test_name"):
        out["test_goal"] = t[:200]

    if _missing_rules_include(missing_rules, "service_name") and len(t) >= 3:
        service_phrase = extract_service_phrase(t)
        if service_phrase:
            out["service_name"] = service_phrase.strip()


def _fill_child_age_entities(text: str, missing_rules: list[str], out: dict[str, Any]) -> None:
    """Извлекает возраст ребёнка, если он отдельно запрашивается.

    :param text: исходный текст пользователя
    :param missing_rules: список правил незаполненных слотов
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    if "child_age" not in missing_rules:
        return

    m_age = _QF_AGE_RE.search(str(text or "").strip().lower())
    if not m_age:
        return
    try:
        out["child_age"] = int(m_age.group(1))
    except Exception:
        pass


def _is_valid_patient_name_candidate(candidate_text: str) -> bool:
    """Проверяет, похож ли кандидат на реальное ФИО пациента.

    :param candidate_text: текст-кандидат
    :return: ``True``, если строка похожа на ФИО пациента
    """

    raw = str(candidate_text or "").strip()
    if not raw:
        return False
    if has_datetime_signal(raw):
        return False
    if match_city(raw):
        return False
    low_raw = raw.lower()
    if _QF_APPOINTMENT_WORD_RE.search(low_raw):
        return False
    if _QF_TIME_FRAGMENT_RE.search(low_raw):
        return False
    tokens = [w for w in re.split(r"\s+", low_raw) if w]
    if len(tokens) < 2:
        return False
    if any(token in _QF_PATIENT_NAME_STOPWORDS for token in tokens):
        return False
    if not any(len(token) >= 3 for token in tokens):
        return False
    return True


def _fill_patient_entities(text: str, missing_rules: list[str], out: dict[str, Any]) -> None:
    """Извлекает ФИО пациента из явного ответа пользователя.

    :param text: исходный текст пользователя
    :param missing_rules: список правил незаполненных слотов
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    if "patient_name" not in missing_rules:
        return

    t = str(text or "").strip()
    candidate: str | None = None
    m_name = _QF_PATIENT_NAME_PREFIX_RE.search(t)
    if m_name:
        candidate = m_name.group(1).strip()
    if not candidate:
        m_plain = _QF_PLAIN_NAME_RE.match(t)
        if m_plain:
            plain = m_plain.group(1).strip()
            low_plain = plain.lower()
            if not (
                _QF_APPOINTMENT_WORD_RE.search(low_plain)
                or _QF_TIME_FRAGMENT_RE.search(low_plain)
                or match_city(low_plain)
            ):
                candidate = plain
    if not candidate:
        fragments = list(_QF_NAME_FRAGMENT_RE.finditer(t))
        if fragments:
            raw = " ".join(fragments[-1].groups()).strip()
            low_raw = raw.lower()
            if not (
                _QF_APPOINTMENT_WORD_RE.search(low_raw)
                or _QF_TIME_FRAGMENT_RE.search(low_raw)
                or match_city(low_raw)
            ):
                candidate = raw
    if candidate and _is_valid_patient_name_candidate(candidate):
        normalized_tokens = [w.capitalize() for w in re.split(r"\s+", candidate) if w]
        if len(normalized_tokens) >= 2:
            out["patient_name"] = " ".join(normalized_tokens)


def _fill_location_entities(text: str, missing_rules: list[str], out: dict[str, Any]) -> None:
    """Извлекает город, если он нужен для текущего уточнения.

    :param text: исходный текст пользователя
    :param missing_rules: список правил незаполненных слотов
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    if not _missing_rules_include(missing_rules, "city"):
        return

    city = match_city(str(text or "").strip())
    if city:
        out["city"] = city.strip()


def _fill_appointment_selection_mode(text: str, missing_rules: list[str], out: dict[str, Any]) -> None:
    """Извлекает режим выбора в appointment-flow: врач или филиал.

    :param text: исходный текст пользователя
    :param missing_rules: список правил незаполненных слотов
    :param out: словарь, который пополняется найденными сущностями
    :return: None
    """

    if not _missing_rules_include(missing_rules, "branch", "city"):
        return
    if out.get("branch_id") or out.get("branch_name"):
        return

    t = str(text or "").strip()
    doctor_choice = bool(_QF_APPOINTMENT_CHOICE_DOCTOR_RE.search(t))
    branch_choice = bool(_QF_APPOINTMENT_CHOICE_BRANCH_RE.search(t))
    if doctor_choice and not branch_choice:
        out["appointment_selection_mode"] = "doctor"
    elif branch_choice and not doctor_choice:
        out["appointment_selection_mode"] = "branch"


def quick_fill_core_entities(text: str, state_entities: dict[str, Any], missing_rules: list[str]) -> dict[str, Any]:
    """Быстро дополняет базовые сущности из короткой реплики пользователя.

    :param text: исходный текст пользователя
    :param state_entities: уже накопленные сущности состояния
    :param missing_rules: список правил незаполненных слотов
    :return: словарь с быстро извлечёнными сущностями
    """

    out: dict[str, Any] = {}
    _fill_insurance_entities(text, out)
    _fill_datetime_entities(text, out)
    _fill_test_result_entities(text, missing_rules, out)
    _fill_appointment_entities(text, state_entities, missing_rules, out)
    _fill_child_age_entities(text, missing_rules, out)
    _fill_patient_entities(text, missing_rules, out)
    _fill_location_entities(text, missing_rules, out)
    _fill_appointment_selection_mode(text, missing_rules, out)
    return out


def extract_branch_hint(text: str, state_entities: dict[str, Any]) -> str | None:
    t = (text or "").strip()
    branch_hint: str | None = None
    m_bx = _QF_BRANCH_EXPLICIT_RE.search(t)
    if m_bx:
        branch_hint = m_bx.group(1).strip()
    if not branch_hint:
        m_on = _QF_BRANCH_ON_RE.search(t)
        if m_on:
            street = (m_on.group(1) or "").strip()
            num = (m_on.group(2) or "").strip()
            candidate = f"{street} {num}".strip()
            if num and street.isdigit():
                candidate = ""
            cand_norm = normalize_loose_text(candidate)
            if candidate and (num or looks_like_address(t) or (len(cand_norm) >= 5 and not has_datetime_signal(cand_norm))):
                branch_hint = candidate
    if not branch_hint:
        probe = _QF_BRANCH_CITY_PREFIX_RE.sub("", t, count=1)
        probe = _QF_BRANCH_IN_CITY_PREFIX_RE.sub("", probe, count=1)
        probe = re.sub(r"^\s*(?:в\s+)", "", probe, count=1, flags=re.I)
        probe = probe.strip(" ,.;")
        m_free = _QF_BRANCH_FREEFORM_RE.match(probe)
        if m_free:
            street = (m_free.group(1) or "").strip()
            num = (m_free.group(2) or "").strip()
            if street and not has_datetime_signal(street):
                branch_hint = f"{street} {num}".strip()
    if not branch_hint:
        words = [w for w in re.split(r"\s+", t) if w]
        if 1 <= len(words) <= 6 and len(t) <= 120:
            city_guess = match_city(t)
            is_city_only = bool(city_guess and normalize_loose_text(city_guess) == normalize_loose_text(t))
            if (state_entities.get("city") or looks_like_branch_hint(t)) and not is_city_only:
                branch_hint = t
    return branch_hint

def _normalize_alpha_token(value: str) -> str:
    return re.sub(r"[^a-zа-яё\-]", "", normalize_ru(value)).strip()


def _first_alpha_token(value: str) -> str:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё\-]{2,}", str(value or ""))
    return _normalize_alpha_token(tokens[0]) if tokens else ""


def service_name_conflicts_with_doctor(service_name: str, doctor_name: str | None) -> bool:
    """
    True, если service_name выглядит как фамилия того же врача
    (включая падежные формы: "Дразнину" <-> "Дразнин").
    """
    service_token = _first_alpha_token(service_name)
    doctor_token = _first_alpha_token(str(doctor_name or ""))
    if len(service_token) < 3 or len(doctor_token) < 3:
        return False

    doctor_vars = {
        _normalize_alpha_token(v)
        for v in ([doctor_token] + surname_variants(doctor_token))
        if _normalize_alpha_token(v)
    }
    if not doctor_vars:
        return False

    service_vars = {
        _normalize_alpha_token(v)
        for v in ([service_token] + surname_variants(service_token))
        if _normalize_alpha_token(v)
    }
    if service_vars & doctor_vars:
        return True

    # Легкая страховка от опечаток/ё/и, но с высоким cutoff.
    return bool(get_close_matches(service_token, list(doctor_vars), n=1, cutoff=0.92))


def _appointment_required_slots(entities: dict[str, Any]) -> list[str]:
    action = str(entities.get("appointment_action") or "").strip().lower()
    if action in {"unknown", "ambiguous"}:
        return ["appointment_action"]
    if action in {"cancel", "reschedule"}:
        return [
            "appointment_action",
            "_any_of:doctor_id,doctor_name",
            "patient_name",
        ]
    return REQUIRED_SLOTS.get("APPOINTMENT", [])


def missing_slots(label: str, entities: dict[str, Any]) -> list[str]:
    req = _appointment_required_slots(entities) if label == "APPOINTMENT" else REQUIRED_SLOTS.get(label, [])
    missing: list[str] = []
    for r in req:
        if r.startswith("_any_of:"):
            keys = [k.strip() for k in r.split(":", 1)[1].split(",") if k.strip()]
            if not any(entities.get(k) for k in keys):
                missing.append(r)
        elif r == "appointment_action":
            action = str(entities.get("appointment_action") or "").strip().lower()
            if action in {"", "unknown", "ambiguous"}:
                missing.append(r)
        elif not entities.get(r):
            missing.append(r)
    # Если уже известен конкретный врач, город не обязателен:
    # расписание/адреса берем из live расписания врача.
    if (
        label == "APPOINTMENT"
        and str(entities.get("appointment_action") or "").strip().lower() not in {"cancel", "reschedule"}
        and (entities.get("doctor_id") or entities.get("doctor_name"))
    ):
        missing = [m for m in missing if m != "_any_of:city,branch_name,branch_id"]
    if label == "PRICE" and (entities.get("doctor_id") or entities.get("doctor_name")):
        missing = [m for m in missing if m not in {"_any_of:city,branch_name,branch_id", "service_name"}]
    if label == "APPOINTMENT" and entities.get("accepts_children") and not entities.get("child_age"):
        missing.append("child_age")
    return missing


def clarification_question(label: str, missing: list[str], entities: dict[str, Any] | None = None) -> str:
    entities = entities or {}
    need_city = any(m.startswith("_any_of:city") for m in missing)
    need_service = any(
        "doctor_id" in m or "doctor_name" in m or "specialty" in m or "service_name" in m
        for m in missing
    )
    need_datetime = any(
        m in {"date_from", "date_to", "time_from", "time_to", "date_hint"}
        or ("date_from" in m)
        or ("time_from" in m)
        or ("date_hint" in m)
        for m in missing
    )

    if "child_age" in missing:
        return "Сколько полных лет ребенку?"
    if label in {"PRICE", "TEST_ASSIST", "ADDRESS"} and need_city:
        return "Из какого города вы обращаетесь?"
    if label in CLARIFY_TEXT_MAP:
        return CLARIFY_TEXT_MAP[label]
    if label == "APPOINTMENT":
        action = str(entities.get("appointment_action") or "").strip().lower()
        if "appointment_action" in missing:
            return "Хотите отменить или перенести запись?"
        if action == "cancel":
            if need_service:
                return "Здравствуйте! Укажите ФИО врача, запись к которому нужно отменить."
            if "patient_name" in missing:
                return "Здравствуйте! Для отмены записи укажите ФИО пациента."
            return "Уточните, пожалуйста, детали записи для отмены."
        if action == "reschedule":
            if need_service:
                return "Здравствуйте! Укажите ФИО врача, запись к которому нужно перенести."
            if need_datetime:
                return APPOINTMENT_CLARIFY_MAP["need_datetime"]
            if "patient_name" in missing:
                return APPOINTMENT_CLARIFY_MAP["need_patient"]
            return "Уточните, пожалуйста, детали записи для переноса."
        if need_city:
            return APPOINTMENT_CLARIFY_MAP["need_city"]
        if need_service:
            return APPOINTMENT_CLARIFY_MAP["need_service"]
        if "patient_name" in missing:
            return APPOINTMENT_CLARIFY_MAP["need_patient"]
        if need_datetime:
            return APPOINTMENT_CLARIFY_MAP["need_datetime"]
        return APPOINTMENT_CLARIFY_MAP["default"]
    if label == "TEST_RESULT":
        need = frozenset(missing)
        if need in TEST_RESULT_CLARIFY_MAP:
            return TEST_RESULT_CLARIFY_MAP[need]
        return (
            "Чтобы получить результат, укажите данные в таком порядке: "
            "фамилия, год рождения, код анализа, номер анализа. "
            "Пример: «Иванов, 1989, Бг, 1234»."
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
    has_date_time = bool(
        (entities.get("date_from") or entities.get("date_hint"))
        and (entities.get("time_from") or entities.get("time_flexible"))
    )
    has_patient_name = bool(str(entities.get("patient_name") or "").strip())
    if not branch_selected:
        return APPOINTMENT_STEP_BRANCH
    if not has_date_time:
        return APPOINTMENT_STEP_DATETIME
    if not has_patient_name:
        return APPOINTMENT_STEP_PATIENT
    if not entities.get("appointment_confirm_pending") and not entities.get("appointment_confirmed"):
        return APPOINTMENT_STEP_CONFIRM
    return APPOINTMENT_STEP_DONE


_YES_RE = re.compile(
    r"^\s*(да|ага|угу|подтверждаю|подтверждаем|верно|ок|окей|конечно|давайте|наверное|хорошо|ладно)\s*[!.,?;:]?\s*$",
    re.I,
)
_NO_RE = re.compile(
    r"^\s*(нет|неа|не\s+подтверждаю|не\s+подтверждаем|неверно|не\s+верно|не\s+надо|не\s+нужно|"
    r"неправильно|не\s+правильно)\s*[!.,?;:]?\s*$",
    re.I,
)
_CONFIRM_STRIP_QUOTES_RE = re.compile(r"[\"'`«»]+")
_CONFIRM_PUNCT_RE = re.compile(r"[!.,?;:]+")
_CONFIRM_SPACE_RE = re.compile(r"\s+")
_CONFIRM_YES_EXACT = {
    "да",
    "ага",
    "угу",
    "ок",
    "окей",
    "хорошо",
    "ладно",
    "верно",
    "конечно",
    "давайте",
    "подтверждаю",
    "подтверждаем",
}
_CONFIRM_NO_EXACT = {
    "нет",
    "неа",
    "неверно",
    "не верно",
    "неправильно",
    "не правильно",
    "не надо",
    "не нужно",
    "не подтверждаю",
    "не подтверждаем",
}


def _normalize_confirmation_text(text: str) -> str:
    norm = normalize_ru(text)
    if not norm:
        return ""
    norm = _CONFIRM_STRIP_QUOTES_RE.sub(" ", norm)
    norm = _CONFIRM_PUNCT_RE.sub(" ", norm)
    norm = _CONFIRM_SPACE_RE.sub(" ", norm).strip()
    return norm


def _is_affirmative_text(text: str) -> bool:
    raw = str(text or "")
    if _YES_RE.match(raw):
        return True
    norm = _normalize_confirmation_text(raw)
    if not norm:
        return False
    if norm in _CONFIRM_YES_EXACT:
        return True
    if norm.startswith("да "):
        return True
    if norm.startswith("хорошо "):
        return True
    if norm.startswith("ладно "):
        return True
    if "подтвержда" in norm and "не подтвержда" not in norm:
        return True
    return False


def _is_negative_text(text: str) -> bool:
    raw = str(text or "")
    if _NO_RE.match(raw):
        return True
    norm = _normalize_confirmation_text(raw)
    if not norm:
        return False
    if norm in _CONFIRM_NO_EXACT:
        return True
    if norm.startswith("не подтвержда"):
        return True
    return False


def appointment_confirmation_transition(text: str) -> str:
    if _is_affirmative_text(text):
        return APPOINTMENT_CONFIRM_YES
    if _is_negative_text(text):
        return APPOINTMENT_CONFIRM_NO
    return APPOINTMENT_CONFIRM_OTHER


def is_context_affirmative(text: str) -> bool:
    return _is_affirmative_text(text)


def is_context_negative(text: str) -> bool:
    return _is_negative_text(text)


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


def _render_appointment_date_part(date_raw: str) -> str:
    raw = str(date_raw or "").strip()
    if not raw:
        return "уточним дату"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        try:
            d = datetime.strptime(raw, "%Y-%m-%d").date()
            month = {
                1: "января",
                2: "февраля",
                3: "марта",
                4: "апреля",
                5: "мая",
                6: "июня",
                7: "июля",
                8: "августа",
                9: "сентября",
                10: "октября",
                11: "ноября",
                12: "декабря",
            }[d.month]
            return f"{d.day} {month}"
        except Exception:
            return raw

    hint_map = {
        "today": "сегодня",
        "tomorrow": "завтра",
        "after tomorrow": "послезавтра",
        "after_tomorrow": "послезавтра",
        "this_week": "на этой неделе",
        "next_week": "на следующей неделе",
    }
    return hint_map.get(raw.lower(), raw)


def appointment_summary(entities: dict[str, Any]) -> str:
    patient_name = str(entities.get("patient_name") or "").strip()
    service = appointment_service_display(entities)
    place = str(entities.get("branch_name") or entities.get("city") or "выбранный филиал").strip()
    date_part_raw = str(entities.get("date_from") or entities.get("date_hint") or "").strip()
    date_part = _render_appointment_date_part(date_part_raw)
    time_from = str(entities.get("time_from") or "").strip()
    time_to = str(entities.get("time_to") or "").strip()
    if time_from and time_to and time_from != time_to:
        time_part = f"{time_from}-{time_to}"
    elif time_from:
        time_part = time_from
    elif entities.get("time_flexible"):
        time_part = "любое время"
    else:
        time_part = "уточним время"
    if patient_name:
        return f"Запись: {patient_name}, {service}, {place}, {date_part}, {time_part}."
    return f"Запись: {service}, {place}, {date_part}, {time_part}."


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


def appointment_text_branch_prompt(
    service: str,
    city: str,
    addresses: list[str],
    *,
    allow_doctor_option: bool = False,
) -> str:
    if addresses:
        lines = "\n".join([f"- {a}" for a in addresses])
        city_part = f" в городе {city}" if city else ""
        tail = "Какой филиал вам удобен?"
        if allow_doctor_option:
            tail = "Какой филиал вам удобен? Или написать список врачей по этой услуге?"
        return (
            f"Есть возможность записи на {service}{city_part} по адресам:\n"
            f"{lines}\n"
            f"{tail}"
        )
    if allow_doctor_option:
        return "Уточните, пожалуйста, удобный филиал/адрес для записи. Если удобнее, могу показать врачей по этой услуге."
    return "Уточните, пожалуйста, удобный филиал/адрес для записи."


def appointment_text_datetime_prompt(service: str, branch: str, price_rub: str | None) -> str:
    if price_rub:
        return f"Да, можем записать на {service} ({branch}), стоимость {price_rub} руб. На какую дату и время вам удобно?"
    return f"Да, можем записать на {service} ({branch}). На какую дату и время вам удобно?"


def appointment_text_patient_name_prompt() -> str:
    return APPOINTMENT_REPLY_MAP["patient_name_prompt"]


def appointment_text_confirm_prompt(summary: str) -> str:
    return f"{summary} {APPOINTMENT_REPLY_MAP['confirm_suffix']}"


def appointment_text_confirmed_handoff(summary: str) -> str:
    return f"{summary}\n{APPOINTMENT_REPLY_MAP['confirmed_handoff_suffix']}"


def appointment_text_reask_datetime() -> str:
    return APPOINTMENT_REPLY_MAP["reask_datetime"]


def appointment_text_reask_confirm() -> str:
    return APPOINTMENT_REPLY_MAP["reask_confirm"]


def appointment_text_cancel_confirm() -> str:
    return APPOINTMENT_REPLY_MAP["cancel_confirm"]


def appointment_text_topic_switch_confirm() -> str:
    return APPOINTMENT_REPLY_MAP["topic_switch_confirm"]


def appointment_text_cancelled() -> str:
    return APPOINTMENT_REPLY_MAP["cancelled"]


_STALE_LAB_SERVICE_RE = re.compile(
    r"(?:"
    r"антитела|анализ\w*\s+(?:крови|мочи|кала)|"
    r"\b(?:оак|оам|алт|аст|алат|асат|ттг|сое|соэ|"
    r"мно|пти|ггт|лдг|кфк|глюкоз\w*|холестерин\w*|"
    r"инсулин\w*|липид\w*|ферритин\w*|витамин\b)|"
    r"\b(?:пцр|ифа|элиза|hba1c|hbsag|hcv|hiv|спид|"
    r"кардиолипин\w*|щитовидк\w*)|"
    r"копрологи\w*|общий\s+анализ|развернут\w+\s+анализ|"
    r"кровь\s+на\b|моча\s+на\b"
    r")",
    re.I,
)


def _service_looks_like_stale_lab_test(service: str) -> bool:
    """Эвристика: service_name похож на лабораторный анализ.

    Используется в `appointment_service_display`: если у записи к
    врачу `service_name` выглядит как анализ (но прямо в карточке к
    приёму — нерелевантен), мы предпочитаем «приём к врачу <ФИО>»
    вместо ложного «Запись: ..., Антитела к рецепторам ТТГ, ...».
    Жалоба заказчика 2026-05-05: stale service_name из предыдущей
    PRICE-турны попадал в подтверждение записи, оператор получал
    карточку «Антитела к рецепторам ТТГ» вместо «приём к Арцыбашевой».

    :param service: значение `service_name`/`test_name`
    :return: True, если строка пахнет лабораторным анализом
    """
    if not service:
        return False
    return bool(_STALE_LAB_SERVICE_RE.search(service))


def appointment_service_display(entities: dict[str, Any]) -> str:
    service_raw = str(entities.get("service_name") or entities.get("test_name") or "").strip()
    doctor_name = str(entities.get("doctor_name") or "").strip()
    if service_raw and doctor_name and service_name_conflicts_with_doctor(service_raw, doctor_name):
        service_raw = ""
    # Защита от утечки stale `service_name` из предыдущей PRICE-турны
    # в карточку записи. Если врач явно указан, а service_name
    # выглядит как лабораторный анализ — пациент почти наверняка
    # записывается на ПРИЁМ К ВРАЧУ, а не на анализ. Берём display
    # «приём к врачу <ФИО>» вместо stale lab-service.
    # Подтверждённый кейс: «Записаться к Арцыбашевой» после диалога
    # про ТТГ → бот показывал «Запись: ..., Антитела к рецепторам ТТГ»
    # → оператор путался. См. policies._STALE_LAB_SERVICE_RE.
    if service_raw and doctor_name and _service_looks_like_stale_lab_test(service_raw):
        service_raw = ""
    if (
        not service_raw
        or _QF_APPOINTMENT_WORD_RE.search(service_raw)
        or _QF_TIME_FRAGMENT_RE.search(service_raw)
    ):
        specialty = str(entities.get("specialty") or "").strip()
        if doctor_name:
            return f"приём к врачу {doctor_name}"
        if specialty:
            return f"приём к {specialty}"
        return "услугу"
    return service_raw


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
