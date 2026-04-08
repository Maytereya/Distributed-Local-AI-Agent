"""Tool registry and heuristic tool-plan selection for FreeTalk."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.I)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    medical: bool = True
    meili_based: bool = False


TOOL_SPECS: dict[str, ToolSpec] = {
    "match_catalog_service": ToolSpec("match_catalog_service", "Catalog grounding for service names."),
    "match_catalog_doctor": ToolSpec("match_catalog_doctor", "Catalog grounding for doctor names."),
    "get_catalog_health": ToolSpec("get_catalog_health", "Catalog health check."),
    "service_bundle_info": ToolSpec("service_bundle_info", "Combined service summary: prices/doctors/preparation."),
    "price_info": ToolSpec("price_info", "Price info for service or doctor."),
    "test_prepare": ToolSpec("test_prepare", "Preparation for analyses/services."),
    "test_assist": ToolSpec("test_assist", "Analysis search and shortlist."),
    "doctors_info": ToolSpec("doctors_info", "Doctor cards and filtering."),
    "doctors_schedule_week": ToolSpec("doctors_schedule_week", "Doctor schedule."),
    "address_info": ToolSpec("address_info", "Branches and addresses."),
    "test_result_status": ToolSpec("test_result_status", "Lab results availability."),
    "main_index_info": ToolSpec("main_index_info", "Main knowledge index lookup.", meili_based=True),
    "news_info": ToolSpec("news_info", "News lookup.", meili_based=True),
    "web_search": ToolSpec("web_search", "Internet search via SearXNG.", medical=False),
}


_MEDICAL_TOPIC_RE = _compile(
    r"\b("
    r"клиник|медцентр|медицинск|врач|доктор|терапевт|теарапевт|терапефт|кардиолог|невролог|гастроэнтеролог|эндокринолог|"
    r"гинеколог|уролог|онколог|педиатр|хирург|дерматолог|аллерголог|иммунолог|"
    r"офтальмолог|лор|отоларинголог|специалист|прием|приём|приним|консультац|"
    r"расписан|график|услуг|процедур|анализ|тест|лаборатор|подготовк|адрес|"
    r"филиал|результат|узи|мрт|кт"
    r")\w*\b"
)
_PRICE_RE = _compile(r"\b(цена|стоим|прайс|сколько\s+стоит)\b")
_PREPARE_RE = _compile(r"\b(подготовк|натощак|можно\s+ли\s+есть|как\s+подготовиться)\b")
_SCHEDULE_RE = _compile(r"\b(расписан|график|когда\s+принимает|свободн\w*\s+окн)\b")
_ADDRESS_RE = _compile(r"\b(адрес|филиал|где\s+сдать|где\s+находит)\b")
_RESULT_RE = _compile(r"\b(результат\w*|result|номер\s+анализа|год\s+рожд)\b")
_DOCTOR_RE = _compile(
    r"\b("
    r"врач|доктор|терапевт|теарапевт|терапефт|кардиолог|невролог|гастроэнтеролог|эндокринолог|"
    r"гинеколог|уролог|онколог|педиатр|хирург|дерматолог|аллерголог|иммунолог|"
    r"офтальмолог|лор|отоларинголог|к\s+[а-яё\-]{3,}"
    r")\w*\b"
)
_ANALYSIS_RE = _compile(r"\b(анализ|тест|лаборатор)\w*\b")
_SERVICE_RE = _compile(r"\b(услуг|процедур|исследован|узи|мрт|кт)\w*\b")
_NEWS_RE = _compile(r"\b(новост|акц|объявлен)\w*\b")
_DOC_RE = _compile(r"\b(документ|справк|налог|вычет|договор|лиценз)\w*\b")
_WEB_SIGNAL_RE = _compile(
    r"\b("
    r"найди\s+в\s+интернет|поищи\s+в\s+интернет|поиск\s+в\s+сети|в\s+сети|"
    r"последн\w+\s+новост|свеж\w+\s+новост|что\s+нового|актуальн\w+\s+данн|"
    r"поищи|найди|погугли|загугли|search|lookup|today|latest|breaking\s+news"
    r")\b"
)
_ABOUT_AGENT_RE = _compile(
    r"\b("
    r"расскажи\s+о\s+себе|кто\s+ты|что\s+ты\s+умеешь|что\s+ты\s+можешь|"
    r"как\s+с\s+тобой\s+работать|какие\s+у\s+тебя\s+возможности|"
    r"what\s+can\s+you\s+do|who\s+are\s+you"
    r")\b"
)
_CLINIC_DOCTOR_LIST_RE = _compile(
    r"\b(кто|какие|список|выведи|покажи)\b.*\b(врач|доктор|терап|специал)\w*\b|"
    r"\b(врач|доктор|терап|специал)\w*\b.*\bклиник\w*\b"
)


def is_medical_query(text: str) -> bool:
    return bool(_MEDICAL_TOPIC_RE.search(str(text or "")))


def should_use_web_search(text: str, *, allow_for_medical: bool = False) -> bool:
    q = str(text or "")
    if not q.strip():
        return False
    if not allow_for_medical and is_medical_query(q):
        return False
    return bool(_WEB_SIGNAL_RE.search(q))


def is_about_agent_query(text: str) -> bool:
    q = str(text or "").strip()
    if not q:
        return False
    if len(q) > 180:
        return False
    return bool(_ABOUT_AGENT_RE.search(q))


def select_tool_plan(text: str, *, include_meili_tools: bool) -> list[str]:
    q = str(text or "")

    if _CLINIC_DOCTOR_LIST_RE.search(q):
        return ["doctors_info", "doctors_schedule_week"]
    if _RESULT_RE.search(q):
        return ["test_result_status", "test_assist"]
    if _SCHEDULE_RE.search(q):
        return ["doctors_schedule_week", "doctors_info"]
    if _PRICE_RE.search(q):
        return ["price_info", "service_bundle_info", "test_assist"]
    if _PREPARE_RE.search(q):
        return ["test_prepare", "test_assist", "service_bundle_info"]
    if _ADDRESS_RE.search(q):
        return ["address_info", "service_bundle_info"]
    if _DOCTOR_RE.search(q):
        return ["doctors_info", "doctors_schedule_week"]
    if _ANALYSIS_RE.search(q):
        return ["test_assist", "test_prepare", "price_info"]
    if _SERVICE_RE.search(q):
        return ["service_bundle_info", "price_info", "address_info"]

    if include_meili_tools and _NEWS_RE.search(q):
        return ["news_info"]
    if include_meili_tools and _DOC_RE.search(q):
        return ["main_index_info"]

    if is_medical_query(q):
        return ["service_bundle_info", "doctors_info", "address_info"]
    return []
