"""Классификатор пользовательского сообщения в канонический label роутера.

Содержит hard-rules (безопасность и бизнес-триггеры), LLM fallback,
нормализацию entities/flags и поддержку primary+secondary intent hints.
"""

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
from agent_logic_2.doctor_name_matching import extract_doctor_name_candidate
from agent_logic_2.ollama_settings import LLMName

from .mess_types import PATIENT_LABEL_PRIORITY, Label, RouteDecision, ContextAction
from .policies import (
    detect_urgent,
    detect_complaint,
    detect_medical_advice,
    detect_test_interpretation,
    detect_test_result_intent,
    detect_test_assist_intent,
    detect_schedule_intent,
    detect_doc_request_intent,
    detect_appointment_intent,
    detect_appointment_action,
    detect_price_intent,
    detect_address_intent,
    detect_news_intent,
    detect_doctor_info_intent,
    normalize_appointment_action,
    has_appointment_context,
    is_address_dominant_intent,
    is_price_dominant_intent,
    should_treat_result_delivery_as_test_assist,
    detect_pii,
    low_confidence_policy,
    extract_service_phrase,
)

ollama_client = AsyncClient(c.ollama_url)
_CLASSIFY_TIMEOUT = 45
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_TOPIC_SWITCH_RE = re.compile(r"\b(передумал\w*|передумала\w*|друг(ой|ая)\s+врач\w*|нуж\w+)\b", re.I)
_CANCEL_FLOW_RE = re.compile(r"\b(отмен\w*|не\s+надо|не\s+хочу)\b", re.I)
_DOCTOR_SWITCH_SIGNAL_RE = re.compile(r"\b(расписани\w*|график|врач\w*|доктор\w*|когда\b.*\bпринима\w*)\b", re.I)
_INVALID_DOCTOR_TOKEN_RE = re.compile(r"^(отмен|перен|запис|покаж|подскаж|скажи|нуж|хоч|надо)", re.I)


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
_DIAGNOSTIC_RE = re.compile(r"\b(экг|узи|мрт|кт|фгдс|фкс|рентген|флюорограф|колоноскоп|холтер)\b", re.I)
_GREETING_ONLY_RE = re.compile(
    r"^\s*(привет|здравствуйте|здраствуйте|добрый день|доброе утро|добрый вечер|доброго дня|hello|hi)\s*[!.,?]*\s*$",
    re.I,
)
_LABEL_RANK = {lbl: i for i, lbl in enumerate(PATIENT_LABEL_PRIORITY)}

def _extract_order_id(text: str) -> str | None:
    m = _ORDER_ID_RE.search(text)
    return m.group(1) if m else None


def _extract_service_keyword(text: str) -> str | None:
    phrase = extract_service_phrase(text)
    if phrase:
        return phrase
    m = _DIAGNOSTIC_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def _extract_schedule_doctor_name(text: str) -> str | None:
    return extract_doctor_name_candidate(text, prefer_schedule=True)


def _extract_appointment_doctor_name(text: str) -> str | None:
    candidate = extract_doctor_name_candidate(text)
    if candidate and _INVALID_DOCTOR_TOKEN_RE.search(candidate):
        return None
    return candidate


def _seed_entities_from_memory(last_entities: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "doctor_id", "doctor_name", "specialty",
        "branch_id", "branch_name", "city",
        "insurance_type", "accepts_children",
        "child_age", "patient_name",
        "date_from", "date_to", "time_from", "time_to", "date_hint",
        "test_name", "service_name", "order_id",
        "surname", "year", "filial", "number", "lang",
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
    "patient_name",
    "date_hint", "date_from", "date_to", "time_from", "time_to",
    "surname", "year", "filial", "number", "lang",
    "secondary_intents",
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


def _normalize_context_action(x: Any) -> ContextAction:
    if isinstance(x, str) and x in {"continue", "overwrite_doctor", "new_topic", "cancel_flow"}:
        return cast(ContextAction, x)
    return cast(ContextAction, "continue")


def _derive_context_action(
    text: str,
    label: Label,
    entities: dict[str, Any],
    last_entities: dict[str, Any] | None,
) -> ContextAction:
    prev = last_entities or {}
    if label == "APPOINTMENT" and _CANCEL_FLOW_RE.search(text or ""):
        return cast(ContextAction, "cancel_flow")

    prev_doctor = str(prev.get("doctor_name") or "").strip().lower().replace("ё", "е")
    new_doctor = str(entities.get("doctor_name") or "").strip().lower().replace("ё", "е")
    if new_doctor and prev_doctor and new_doctor != prev_doctor:
        return cast(ContextAction, "overwrite_doctor")
    extracted = extract_doctor_name_candidate(text, prefer_schedule=True)
    if extracted and _INVALID_DOCTOR_TOKEN_RE.search(str(extracted)):
        extracted = None
    extracted_norm = str(extracted or "").strip().lower().replace("ё", "е")
    if (
        extracted_norm
        and prev_doctor
        and extracted_norm != prev_doctor
        and (
            label in {"DOCTOR_SCHEDULE", "DOCTOR_INFO"}
            or _DOCTOR_SWITCH_SIGNAL_RE.search(text or "")
            or _TOPIC_SWITCH_RE.search(text or "")
        )
    ):
        return cast(ContextAction, "overwrite_doctor")
    if _TOPIC_SWITCH_RE.search(text or "") and extracted_norm:
        return cast(ContextAction, "overwrite_doctor")
    return cast(ContextAction, "continue")


def _collect_rule_intent_hints(text: str, last_entities: dict[str, Any] | None = None) -> list[Label]:
    ctx = last_entities or {}
    labels: set[Label] = set()
    price_intent = detect_price_intent(text)
    appointment_intent = detect_appointment_intent(text)
    appointment_action = normalize_appointment_action(detect_appointment_action(text), text)
    appt_ctx = has_appointment_context(text, ctx, appointment_action)

    if detect_test_result_intent(text):
        labels.add("TEST_RESULT")
    if appointment_intent and appt_ctx and not (price_intent and appointment_action is None):
        labels.add("APPOINTMENT")
    if price_intent:
        labels.add("PRICE")
    if detect_address_intent(text):
        labels.add("ADDRESS")
    if detect_test_assist_intent(text):
        labels.add("TEST_ASSIST")
    if detect_news_intent(text):
        labels.add("NEWS")
    if detect_doctor_info_intent(text):
        labels.add("DOCTOR_INFO")
    if detect_schedule_intent(text):
        labels.add("DOCTOR_SCHEDULE")

    return sorted(labels, key=lambda x: _LABEL_RANK.get(x, 10**9))


def _attach_secondary_intents(
    text: str,
    decision: RouteDecision,
    last_entities: dict[str, Any] | None = None,
) -> RouteDecision:
    if decision.label in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}:
        return decision
    hints = _collect_rule_intent_hints(text, last_entities)
    secondary = [lbl for lbl in hints if lbl != decision.label]
    if not secondary:
        return decision
    entities = dict(decision.entities)
    entities["secondary_intents"] = secondary
    flags = set(decision.flags)
    flags.add("multi_intent_detected")
    return RouteDecision(
        label=decision.label,
        confidence=decision.confidence,
        entities=entities,
        flags=flags,
        needs_handoff=decision.needs_handoff,
        context_action=decision.context_action,
    )


async def analyze(text: str, last_entities: dict[str, Any]) -> RouteDecision:
    flags: set[str] = set()
    flags |= detect_pii(text)

    if _GREETING_ONLY_RE.match(text or ""):
        return _attach_secondary_intents(text, RouteDecision(
            label="OTHER",
            confidence=0.99,
            entities={},
            flags=flags | {"smalltalk_greeting"},
            needs_handoff=False,
            context_action="continue",
        ), last_entities)

    # hard gates
    if detect_urgent(text):
        return _attach_secondary_intents(
            text,
            RouteDecision(label="URGENT", confidence=1.0, entities={}, flags=flags | {"urgent"}, needs_handoff=True, context_action="new_topic"),
            last_entities,
        )

    if detect_complaint(text):
        return _attach_secondary_intents(
            text,
            RouteDecision(label="COMPLAINT", confidence=1.0, entities={}, flags=flags | {"complaint"}, needs_handoff=True, context_action="new_topic"),
            last_entities,
        )

    if detect_medical_advice(text) or detect_test_interpretation(text):
        return _attach_secondary_intents(
            text,
            RouteDecision(label="MEDICAL_ADVICE", confidence=1.0, entities={}, flags=flags | {"medical_advice"}, needs_handoff=True, context_action="new_topic"),
            last_entities,
        )

    # hard rule: doc/legal/certificate requests -> operator
    if detect_doc_request_intent(text):
        return _attach_secondary_intents(text, RouteDecision(
            label="OTHER",
            confidence=0.99,
            entities={},
            flags=flags | {"doc_request_handoff"},
            needs_handoff=True,
            context_action="new_topic",
        ), last_entities)

    # hard rule: result/doc delivery intents
    if detect_test_result_intent(text):
        # Смешанные вопросы "результаты + можно/куда на почту" без явной жалобы
        # трактуем как информационный сценарий ассиста по анализам.
        if should_treat_result_delivery_as_test_assist(text):
            return _attach_secondary_intents(text, RouteDecision(
                label="TEST_ASSIST",
                confidence=0.7,
                entities={},
                flags=flags | {"rule_test_assist", "result_delivery_info"},
                needs_handoff=False,
                context_action="continue",
            ), last_entities)
        entities: dict[str, Any] = {}
        oid = _extract_order_id(text)
        if oid:
            entities["order_id"] = oid
        return _attach_secondary_intents(text, RouteDecision(
            label="TEST_RESULT",
            confidence=0.85,
            entities=entities,
            flags=flags | {"rule_test_result"},
            needs_handoff=False,
            context_action="continue",
        ), last_entities)

    # hard rule: doctor schedule queries
    if detect_schedule_intent(text):
        entities: dict[str, Any] = {}
        doctor_name = _extract_schedule_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name
        return _attach_secondary_intents(
            text,
            RouteDecision(
                label="DOCTOR_SCHEDULE",
                confidence=0.74,
                entities=entities,
                flags=flags | {"rule_schedule"},
                needs_handoff=False,
                context_action=_derive_context_action(text, "DOCTOR_SCHEDULE", entities, last_entities),
            ),
            last_entities,
        )

    # hard rule: doctor info queries
    if detect_doctor_info_intent(text):
        entities: dict[str, Any] = {}
        doctor_name = _extract_appointment_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name
        return _attach_secondary_intents(
            text,
            RouteDecision(
                label="DOCTOR_INFO",
                confidence=0.72,
                entities=entities,
                flags=flags | {"rule_doctor_info"},
                needs_handoff=False,
                context_action=_derive_context_action(text, "DOCTOR_INFO", entities, last_entities),
            ),
            last_entities,
        )

    # hard rules: appointment / price
    # Guard: explicit price query ("стоимость приема ...") should stay PRICE,
    # unless there is a clear appointment action (book/reschedule/cancel).
    appointment_intent = detect_appointment_intent(text)
    appointment_action = normalize_appointment_action(detect_appointment_action(text), text)
    price_intent = detect_price_intent(text)
    address_intent = detect_address_intent(text)
    appt_ctx = has_appointment_context(text, last_entities, appointment_action)
    address_dominant = is_address_dominant_intent(
        text,
        address_intent=address_intent,
        price_intent=price_intent,
        appointment_action=appointment_action,
    )
    if address_dominant:
        return _attach_secondary_intents(
            text,
            RouteDecision(label="ADDRESS", confidence=0.72, entities={}, flags=flags | {"rule_address"}, needs_handoff=False, context_action="continue"),
            last_entities,
        )

    # Цена должна побеждать "запись" в смешанных фразах без конкретного шага времени/даты.
    price_dominant = is_price_dominant_intent(
        text,
        price_intent=price_intent,
        appointment_action=appointment_action,
    )
    if price_dominant:
        entities: dict[str, Any] = {}
        svc = _extract_service_keyword(text)
        if svc:
            entities["service_name"] = svc
        return _attach_secondary_intents(
            text,
            RouteDecision(label="PRICE", confidence=0.72, entities=entities, flags=flags | {"rule_price"}, needs_handoff=False, context_action="continue"),
            last_entities,
        )

    if appointment_intent and appt_ctx and not price_dominant:
        entities: dict[str, Any] = {}
        if appointment_action:
            entities["appointment_action"] = appointment_action
        doctor_name = _extract_appointment_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name
        svc = _extract_service_keyword(text)
        if svc:
            entities["service_name"] = svc
        return _attach_secondary_intents(text, RouteDecision(
            label="APPOINTMENT",
            confidence=0.75,
            entities=entities,
            flags=flags | {"rule_appointment"},
            needs_handoff=False,
            context_action=_derive_context_action(text, "APPOINTMENT", entities, last_entities),
        ), last_entities)

    if detect_news_intent(text):
        return _attach_secondary_intents(
            text,
            RouteDecision(label="NEWS", confidence=0.72, entities={}, flags=flags | {"rule_news"}, needs_handoff=False, context_action="continue"),
            last_entities,
        )

    if detect_test_assist_intent(text):
        return _attach_secondary_intents(
            text,
            RouteDecision(label="TEST_ASSIST", confidence=0.7, entities={}, flags=flags | {"rule_test_assist"}, needs_handoff=False, context_action="continue"),
            last_entities,
        )

    if price_intent:
        entities: dict[str, Any] = {}
        svc = _extract_service_keyword(text)
        if svc:
            entities["service_name"] = svc
        return _attach_secondary_intents(
            text,
            RouteDecision(label="PRICE", confidence=0.72, entities=entities, flags=flags | {"rule_price"}, needs_handoff=False, context_action="continue"),
            last_entities,
        )

    if detect_address_intent(text):
        return _attach_secondary_intents(
            text,
            RouteDecision(label="ADDRESS", confidence=0.72, entities={}, flags=flags | {"rule_address"}, needs_handoff=False, context_action="continue"),
            last_entities,
        )

    # light hints
    if detect_test_result_intent(text):
        flags.add("hint_test_result")
    if detect_test_assist_intent(text):
        flags.add("hint_test_assist")
    if detect_schedule_intent(text):
        flags.add("hint_schedule")
    if detect_news_intent(text):
        flags.add("hint_news")

    seeded = _seed_entities_from_memory(last_entities)

    prompt = _build_classify_prompt(text, seeded)
    data = await ollama_classify_json(prompt)

    label = _normalize_label(data.get("label"))
    conf = _normalize_confidence(data.get("confidence"))
    entities = _sanitize_entities(data.get("entities"))
    flags |= _normalize_flags(data.get("flags"))
    context_action = _normalize_context_action(data.get("context_action"))

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
    if "low_confidence" in flags and label in {"OTHER", "TEST_ASSIST"}:
        needs_handoff = True
        flags.add("handoff_recommended")

    # LLM может вернуть OTHER по confidence, но deterministic-hints подсказывают явный intent.
    if label == "OTHER" and "low_confidence" in flags:
        hints = _collect_rule_intent_hints(text, last_entities)
        if hints:
            promoted = hints[0]
            if promoted != "OTHER":
                label = promoted
                conf = max(conf, 0.55)
                flags.discard("handoff_recommended")
                needs_handoff = False
                flags.add("promoted_from_rule_hints")

    derived_action = _derive_context_action(text, label, entities, last_entities)
    if context_action == "continue" and derived_action != "continue":
        context_action = derived_action

    return _attach_secondary_intents(
        text,
        RouteDecision(
            label=label,
            confidence=conf,
            entities=entities,
            flags=flags,
            needs_handoff=needs_handoff,
            context_action=context_action,
        ),
        last_entities,
    )
