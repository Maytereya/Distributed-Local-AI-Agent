from __future__ import annotations

import json
import re
from typing import Any, cast

from mess_types import PATIENT_LABEL_PRIORITY, Label, RouteDecision
from policies import (
    detect_urgent,
    detect_complaint,
    detect_medical_advice,
    detect_test_interpretation,
    detect_test_result_intent,
    detect_test_assist_intent,
    detect_schedule_intent,
    detect_pii,
    low_confidence_policy,
)

# ---------------------------
# Ollama hook (replace later)
# ---------------------------

async def ollama_classify_json(prompt: str) -> dict[str, Any]:
    """
    ЗАГЛУШКА: заменить на реальный вызов Ollama:
      - generate(..., format="json")
      - return parsed dict
    """
    return {"label": "OTHER", "confidence": 0.2, "entities": {}, "flags": []}


# ---------------------------
# Entity extractors (просто MVP)
# ---------------------------

_ORDER_ID_RE = re.compile(r"(?:заказ|order|№)\s*([0-9]{4,})", re.I)

def _extract_order_id(text: str) -> str | None:
    m = _ORDER_ID_RE.search(text)
    return m.group(1) if m else None


def _seed_entities_from_memory(last_entities: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "doctor_id", "doctor_name", "specialty",
        "branch_id", "branch_name", "city",
        "insurance_type", "accepts_children",
        "date_from", "date_to", "time_from", "time_to", "date_hint",
        "test_name", "service_name", "order_id",
    )
    return {k: last_entities[k] for k in keep if k in last_entities}


def _build_classify_prompt(text: str, seeded: dict[str, Any]) -> str:
    allowed = ", ".join(PATIENT_LABEL_PRIORITY)
    return f"""
Ты классификатор запросов пациента клиники.
Верни JSON строго следующего формата (без лишних ключей):
{{
  "label": "<one of: {allowed}>",
  "confidence": 0.0-1.0,
  "entities": {{
    "doctor_name": string|null,
    "doctor_id": string|null,
    "specialty": string|null,
    "branch_name": string|null,
    "branch_id": string|null,
    "city": string|null,

    "service_name": string|null,
    "appointment_action": "book"|"reschedule"|"cancel"|null,

    "test_name": string|null,
    "test_goal": string|null,
    "order_id": string|null,
    "result_action": "status"|"get_pdf"|null,
    "include_promos": boolean|null,

    "insurance_type": "dms"|"oms"|"paid"|null,
    "accepts_children": boolean|null,

    "date_hint": "today"|"tomorrow"|"this_week"|"next_week"|null,
    "date_from": "YYYY-MM-DD"|null,
    "date_to": "YYYY-MM-DD"|null,
    "time_from": "HH:MM"|null,
    "time_to": "HH:MM"|null
  }},
  "flags": ["..."]
}}

Правила:
- Не выдумывай doctor_id/order_id. Если их нет в тексте — ставь null.
- TEST_RESULT: готовность/получение результатов анализов/PDF.
- TEST_ASSIST: подобрать анализ/комплекс/чекап, интерес к скидкам.
- DOCTOR_SCHEDULE: расписание/график/когда принимает/следующая неделя.
- DOCTOR_INFO: найти врача/специальность/подбор врача.
- APPOINTMENT: записаться/перенести/отменить.
- ADDRESS: адрес/как добраться.
- PRICE: стоимость/прайс.
- PREPARE: подготовка к анализам/исследованиям.
- NEWS: акции/скидки/новости.
- MEDICAL_ADVICE: диагноз/лечение/интерпретация результатов.
- иначе OTHER.

Контекст (сущности из прошлых сообщений, если релевантно): {json.dumps(seeded, ensure_ascii=False)}

Текст пациента: {text}
""".strip()


def _normalize_label(x: Any) -> Label:
    if isinstance(x, str) and x in PATIENT_LABEL_PRIORITY:
        return cast(Label, x)
    return cast(Label, "OTHER")


def _normalize_confidence(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.2
    return max(0.0, min(1.0, v))


_ALLOWED_ENTITY_KEYS = {
    "doctor_name", "doctor_id", "specialty",
    "branch_name", "branch_id", "city",
    "service_name", "appointment_action",
    "test_name", "test_goal", "order_id", "result_action", "include_promos",
    "insurance_type", "accepts_children",
    "date_hint", "date_from", "date_to", "time_from", "time_to",
}

def _sanitize_entities(entities: Any) -> dict[str, Any]:
    if not isinstance(entities, dict):
        return {}
    clean: dict[str, Any] = {}
    for k, v in entities.items():
        if k in _ALLOWED_ENTITY_KEYS:
            clean[k] = v
    return clean


def _normalize_flags(flags: Any) -> set[str]:
    if not isinstance(flags, list):
        return set()
    out: set[str] = set()
    for f in flags:
        if isinstance(f, str) and f.strip():
            out.add(f.strip())
    return out


async def analyze(text: str, last_entities: dict[str, Any]) -> RouteDecision:
    flags: set[str] = set()
    flags |= detect_pii(text)

    # hard gates
    if detect_urgent(text):
        return RouteDecision(label="URGENT", confidence=1.0, entities={}, flags=flags | {"urgent"}, needs_handoff=True)

    if detect_complaint(text):
        return RouteDecision(label="COMPLAINT", confidence=1.0, entities={}, flags=flags | {"complaint"}, needs_handoff=True)

    if detect_medical_advice(text) or detect_test_interpretation(text):
        return RouteDecision(label="MEDICAL_ADVICE", confidence=1.0, entities={}, flags=flags | {"medical_advice"}, needs_handoff=True)

    # light hints
    if detect_test_result_intent(text):
        flags.add("hint_test_result")
    if detect_test_assist_intent(text):
        flags.add("hint_test_assist")
    if detect_schedule_intent(text):
        flags.add("hint_schedule")

    seeded = _seed_entities_from_memory(last_entities)

    prompt = _build_classify_prompt(text, seeded)
    data = await ollama_classify_json(prompt)

    label = _normalize_label(data.get("label"))
    conf = _normalize_confidence(data.get("confidence"))
    entities = _sanitize_entities(data.get("entities"))
    flags |= _normalize_flags(data.get("flags"))

    # deterministic enrich
    if not entities.get("order_id"):
        oid = _extract_order_id(text)
        if oid:
            entities["order_id"] = oid

    if label == "TEST_RESULT" and not entities.get("result_action"):
        t = text.lower()
        if "pdf" in t or "пдф" in t or "файл" in t or "скач" in t:
            entities["result_action"] = "get_pdf"
        else:
            entities["result_action"] = "status"

    if low_confidence_policy(conf):
        flags.add("low_confidence")

    needs_handoff = False
    if "low_confidence" in flags and label in {"OTHER", "TEST_ASSIST", "TEST_RESULT"}:
        needs_handoff = True
        flags.add("handoff_recommended")

    return RouteDecision(label=label, confidence=conf, entities=entities, flags=flags, needs_handoff=needs_handoff)