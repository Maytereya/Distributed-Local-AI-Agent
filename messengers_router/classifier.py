from __future__ import annotations

# точка . позволяет следующее:
# код работает одинаково в контейнере и локально
# не зависит от PYTHONPATH
# не конфликтует с чужими пакетами

import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from ollama import AsyncClient

from agent_logic_2 import config as c, ollama_settings
from agent_logic_2.ollama_settings import LLMName

from .mess_types import PATIENT_LABEL_PRIORITY, Label, RouteDecision
from .policies import (
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

ollama_client = AsyncClient(c.ollama_url)
_CLASSIFY_TIMEOUT = 45
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache
def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        try:
                            return json.loads(" ".join(chunk.split()))
                        except Exception:
                            return None
    return None


async def ollama_classify_json(prompt: str) -> dict[str, Any]:
    """
    Реальный вызов Ollama: generate(format="json") + безопасный парсинг.
    """
    llm = LLMName.get()
    think = ollama_settings.resolve_think(None)

    try:
        res = await asyncio.wait_for(
            ollama_client.generate(
                model=llm,
                prompt=prompt,
                options=ollama_settings.options_set(),
                format="json",
                keep_alive=-1,
                think=think,
            ),
            timeout=_CLASSIFY_TIMEOUT,
        )
    except Exception:
        return {"label": "OTHER", "confidence": 0.2, "entities": {}, "flags": ["ollama_timeout"]}

    raw = res.get("response") if isinstance(res, dict) else None
    if isinstance(raw, str):
        obj = _extract_json(raw)
        if isinstance(obj, dict):
            return obj

    return {"label": "OTHER", "confidence": 0.2, "entities": {}, "flags": []}


# ---------------------------
# Entity extractors (просто MVP)
# ---------------------------

_ORDER_ID_RE = re.compile(r"(?:заказ|order|№)\s*([0-9]{4,})", re.I)
_APPOINTMENT_RE = re.compile(r"\bзапис(аться|ать|ываюсь|ываться|ь|ки)?\b|перенос|перенести|перезапис|отменить|отмена", re.I)
_APPOINTMENT_CANCEL_RE = re.compile(r"\bотмен(ить|а|у)\b", re.I)
_APPOINTMENT_RESCHEDULE_RE = re.compile(r"\bперен(ести|ос|есу|есём|есем)|перезапис", re.I)
_DIAGNOSTIC_RE = re.compile(r"\b(экг|узи|мрт|кт|фгдс|фкс|рентген|флюорограф|колоноскоп)\b", re.I)
_DOCTOR_WORDS_RE = re.compile(
    r"\b(врач|специалист|кардиолог|эндокринолог|уролог|гинеколог|терапевт|педиатр|невролог|лор|хирург|стоматолог|гастроэнтеролог|онколог|проктолог|дерматолог|офтальмолог)\b",
    re.I,
)

def _extract_order_id(text: str) -> str | None:
    m = _ORDER_ID_RE.search(text)
    return m.group(1) if m else None


def _extract_service_keyword(text: str) -> str | None:
    m = _DIAGNOSTIC_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def _seed_entities_from_memory(last_entities: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "doctor_id", "doctor_name", "specialty",
        "branch_id", "branch_name", "city",
        "insurance_type", "accepts_children",
        "child_age",
        "date_from", "date_to", "time_from", "time_to", "date_hint",
        "test_name", "service_name", "order_id",
    )
    return {k: last_entities[k] for k in keep if k in last_entities}


def _build_classify_prompt(text: str, seeded: dict[str, Any]) -> str:
    allowed = ", ".join(PATIENT_LABEL_PRIORITY)
    tmpl = _load_prompt("classifier_patient.txt")
    return (
        tmpl.replace("<<ALLOWED_LABELS>>", allowed)
        .replace("<<SEEDED>>", json.dumps(seeded, ensure_ascii=False))
        .replace("<<TEXT>>", text)
    ).strip()


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
    "insurance_type", "accepts_children", "child_age",
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

    # hard rule: appointment intents (before LLM)
    if _APPOINTMENT_RE.search(text) and (_DIAGNOSTIC_RE.search(text) or _DOCTOR_WORDS_RE.search(text)):
        entities: dict[str, Any] = {}
        if _APPOINTMENT_CANCEL_RE.search(text):
            entities["appointment_action"] = "cancel"
        elif _APPOINTMENT_RESCHEDULE_RE.search(text):
            entities["appointment_action"] = "reschedule"
        svc = _extract_service_keyword(text)
        if svc:
            entities["service_name"] = svc
        return RouteDecision(label="APPOINTMENT", confidence=0.75, entities=entities, flags=flags | {"rule_appointment"}, needs_handoff=False)

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
