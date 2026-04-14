"""Catalog resolution policy for FreeTalk."""

from __future__ import annotations

import re
from typing import Any

from .contracts import AgentReply


_TOKEN_RE = re.compile(r"[a-zа-яё0-9\\-]+", re.I)
_DOCTOR_NON_PERSON_TOKENS = {
    "врач",
    "доктор",
    "специалист",
    "терапевт",
    "кардиолог",
    "невролог",
    "гастроэнтеролог",
    "эндокринолог",
    "гинеколог",
    "уролог",
    "онколог",
    "педиатр",
    "хирург",
    "дерматолог",
    "аллерголог",
    "иммунолог",
    "офтальмолог",
    "лор",
    "отоларинголог",
}
_SERVICE_NON_SPECIFIC_TOKENS = {
    "услуга",
    "услуги",
    "процедура",
    "процедуры",
    "анализ",
    "анализы",
    "исследование",
    "исследования",
    "цена",
    "стоимость",
    "сколько",
    "стоит",
    "прайс",
    "подготовка",
    "врач",
    "доктор",
    "клиника",
}


def looks_like_specific_doctor_lookup(value: str) -> bool:
    tokens = [t.lower() for t in _TOKEN_RE.findall(str(value or "")) if len(t) >= 3]
    if not tokens:
        return False
    return any(token not in _DOCTOR_NON_PERSON_TOKENS for token in tokens)


def looks_like_specific_service_lookup(value: str) -> bool:
    tokens = [t.lower() for t in _TOKEN_RE.findall(str(value or "")) if len(t) >= 3]
    if not tokens:
        return False
    return any(token not in _SERVICE_NON_SPECIFIC_TOKENS for token in tokens)


def catalog_resolution_reply_if_needed(
    *,
    tool_plan: list[str],
    entities: dict[str, Any],
    fallback_only: bool = False,
) -> AgentReply | None:
    doctor_tools = {"doctors_info", "doctors_schedule_week"}
    if any(tool in doctor_tools for tool in tool_plan):
        doctor_status = str(entities.get("_ft_doctor_match_status") or "").strip().lower()
        doctor_query = str(entities.get("_ft_doctor_match_query") or "").strip()
        if doctor_status == "unavailable" and not fallback_only:
            return AgentReply(
                text="Сейчас каталог врачей клиники недоступен. Повторите запрос немного позже.",
                source="clinic_data",
            )
        if doctor_status == "miss" and fallback_only and looks_like_specific_doctor_lookup(doctor_query):
            label = _clean_user_fragment(doctor_query)
            suffix = f" «{label}»" if label else ""
            return AgentReply(
                text=f"В данных клиники врач{suffix} не найден. Проверьте фамилию или уточните специальность.",
                source="clinic_data",
            )

    service_tools = {"price_info", "service_bundle_info", "test_prepare", "test_assist"}
    if any(tool in service_tools for tool in tool_plan):
        service_status = str(entities.get("_ft_service_match_status") or "").strip().lower()
        service_query = str(entities.get("_ft_service_match_query") or "").strip()
        if service_status == "unavailable" and not fallback_only:
            return AgentReply(
                text="Сейчас каталог услуг клиники недоступен. Повторите запрос немного позже.",
                source="clinic_data",
            )
        if service_status == "miss" and fallback_only and looks_like_specific_service_lookup(service_query):
            label = _clean_user_fragment(service_query)
            suffix = f" «{label}»" if label else ""
            return AgentReply(
                text=f"В данных клиники услуга или анализ{suffix} не найдены. Уточните название.",
                source="clinic_data",
            )

    return None


def _clean_user_fragment(value: str, *, max_len: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text
