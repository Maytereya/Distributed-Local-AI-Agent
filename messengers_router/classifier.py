"""Классификатор пользовательского сообщения в канонический label роутера.

Содержит hard-rules (безопасность и бизнес-триггеры), LLM fallback,
нормализацию entities/flags и поддержку primary+secondary intent hints.
Ответственность модуля: выдать единое `RouteDecision` для следующего шага pipeline.
"""

from __future__ import annotations

# точка . позволяет следующее:
# код работает одинаково в контейнере и локально
# не зависит от PYTHONPATH
# не конфликтует с чужими пакетами

import json
import re
from typing import Any, cast

from .llm_mode_policy import RuntimeOptions
from .llm_runtime import generate_text
from .doctor_name_port import extract_doctor_name_candidate
from .mess_types import PATIENT_LABEL_PRIORITY, Label, RouteDecision, ContextAction
from .prompt_contracts import sanitize_classifier_json
from .prompt_registry import load_prompt_text
from .policies import (
    detect_urgent,
    detect_complaint,
    detect_medical_advice,
    detect_test_interpretation,
    detect_test_result_intent,
    detect_test_assist_intent,
    detect_prepare_intent,
    detect_schedule_intent,
    detect_doc_request_intent,
    detect_tax_doc_request_intent,
    detect_appointment_intent,
    detect_appointment_action,
    detect_price_intent,
    detect_address_intent,
    detect_news_intent,
    detect_doctor_info_intent,
    normalize_appointment_action,
    has_appointment_context,
    has_datetime_signal,
    is_address_dominant_intent,
    is_price_dominant_intent,
    should_treat_result_delivery_as_test_assist,
    detect_nonbookable_walkin_intent,
    nonbookable_service_hint,
    detect_pii,
    low_confidence_policy,
    extract_service_phrase,
    extract_specialty,
    has_nearest_schedule_hint,
    missing_slots,
    service_name_conflicts_with_doctor,
)

_CLASSIFY_TIMEOUT = 45
_TOPIC_SWITCH_RE = re.compile(r"\b(передумал\w*|передумала\w*|друг(ой|ая)\s+врач\w*|нуж\w+)\b", re.I)
_CANCEL_FLOW_RE = re.compile(r"\b(отмен\w*|не\s+надо|не\s+хочу)\b", re.I)
_DOCTOR_SWITCH_SIGNAL_RE = re.compile(r"\b(расписани\w*|график|врач\w*|доктор\w*|когда\b.*\bпринима\w*)\b", re.I)
_INVALID_DOCTOR_TOKEN_RE = re.compile(r"^(отмен|перен|запис|покаж|подскаж|скажи|нуж|хоч|надо)", re.I)
_REFINE_INTENTS = {"DOCTOR_INFO", "DOCTOR_SCHEDULE", "APPOINTMENT"}
_REFINE_SIGNAL_RE = re.compile(r"\b(передумал\w*|передумала\w*|лучше|или|а\s+если|а\s+вот|уточн\w*)\b", re.I)
_ALLOWED_CLARIFY_REASONS = {"", "intent_disambiguation", "slot_request", "context_repair", "low_confidence"}
_PATIENT_NAME_ONLY_RE = re.compile(r"^\s*[А-ЯЁа-яё\-]{2,}(?:\s+[А-ЯЁа-яё\-]{2,}){1,2}\s*$")
_PATIENT_NAME_STOPWORDS = {
    "анализ",
    "анализы",
    "результат",
    "результаты",
    "врач",
    "доктор",
    "адрес",
    "филиал",
    "город",
    "стоимость",
    "цена",
    "запись",
    "расписание",
    "да",
    "нет",
    "самара",
}
_SPECIALTY_LIKE_NAME_TOKENS = {
    "кардиолог",
    "эндокринолог",
    "педиатр",
    "хирург",
    "терапевт",
    "травматолог",
    "ортопед",
    "проктолог",
    "колопроктолог",
    "уролог",
    "онколог",
    "гинеколог",
    "невролог",
    "гастроэнтеролог",
    "дерматолог",
    "лор",
    "оториноларинголог",
    "узи",
    "узист",
    "мрт",
    "кт",
    "фгдс",
    "фкс",
    "экг",
}
_PROCEDURE_DOCTOR_INFO_RE = re.compile(
    r"\b((кто|какой|какая|какие)\b.*\b(врач|специалист|доктор)\b|"
    r"(кто|какой|какая|какие)\b.*\b(делает|выполняет|проводит)\b)\b",
    re.I,
)

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


async def ollama_classify_payload(prompt: str, *, queue_timeout_ms: int = 30000) -> tuple[str, dict[str, Any]]:
    """Возвращает исходный текст модели и санитизированный JSON классификатора."""
    try:
        raw = await generate_text(
            prompt,
            timeout_s=_CLASSIFY_TIMEOUT,
            queue_timeout_ms=queue_timeout_ms,
            fmt="json",
        )
    except Exception:
        return "", sanitize_classifier_json(
            {"label": "OTHER", "confidence": 0.2, "entities": {}, "flags": ["ollama_timeout"]}
        )
    if isinstance(raw, str):
        obj = _extract_json(raw)
        if isinstance(obj, dict):
            return raw, sanitize_classifier_json(obj)

    raw_text = raw if isinstance(raw, str) else ""
    return raw_text, sanitize_classifier_json(
        {"label": "OTHER", "confidence": 0.2, "entities": {}, "flags": ["ollama_non_json"]}
    )


async def ollama_classify_json(prompt: str, *, queue_timeout_ms: int = 30000) -> dict[str, Any]:
    """
    Реальный вызов Ollama: generate(format="json") + безопасный парсинг.
    """
    _, payload = await ollama_classify_payload(prompt, queue_timeout_ms=queue_timeout_ms)
    return payload


# ---------------------------
# Entity extractors (просто MVP)
# ---------------------------

_ORDER_ID_RE = re.compile(r"(?:заказ|order|№)\s*([0-9]{4,})", re.I)
_GREETING_ONLY_RE = re.compile(
    r"^\s*(привет\w*|здравствуйте|здраствуйте|добрый день|доброе утро|добрый вечер|доброго дня|hello|hi)\s*[!.,?]*\s*$",
    re.I,
)
_GREETING_PREFIX_RE = re.compile(
    r"^\s*(привет\w*|здравствуйте|здраствуйте|добрый день|доброе утро|добрый вечер|доброго дня|hello|hi)\b",
    re.I,
)
_SMALLTALK_RE = re.compile(
    r"^\s*(как\s+дела|как\s+жизнь|как\s+ты|ч[её]\s+как|что\s+нового)\s*[!.,?]*\s*$",
    re.I,
)
_LABEL_RANK = {lbl: i for i, lbl in enumerate(PATIENT_LABEL_PRIORITY)}


def _is_smalltalk_greeting(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if _GREETING_ONLY_RE.match(s):
        return True
    m = _GREETING_PREFIX_RE.match(s)
    if not m:
        return False
    tail_raw = s[m.end():]
    # Убираем пунктуацию/скобки/эмодзи-символы в хвосте, чтобы
    # "привет)", "привет )))", "привет 🙂" считались приветствием.
    tail = re.sub(r"[^\wА-Яа-яЁё]+", " ", tail_raw, flags=re.U).strip()
    if not tail:
        return True
    return bool(_SMALLTALK_RE.match(tail))

def _extract_order_id(text: str) -> str | None:
    m = _ORDER_ID_RE.search(text)
    return m.group(1) if m else None


def _extract_service_keyword(text: str) -> str | None:
    return extract_service_phrase(text)


def _extract_schedule_doctor_name(text: str) -> str | None:
    candidate = extract_doctor_name_candidate(text, prefer_schedule=True)
    if candidate and _looks_like_specialty_or_service_token(candidate, text):
        return None
    return candidate


def _extract_appointment_doctor_name(text: str) -> str | None:
    candidate = extract_doctor_name_candidate(text)
    if candidate and _INVALID_DOCTOR_TOKEN_RE.search(candidate):
        return None
    return candidate


def _extract_price_doctor_name(text: str) -> str | None:
    candidate = extract_doctor_name_candidate(text, prefer_schedule=True)
    if candidate and _INVALID_DOCTOR_TOKEN_RE.search(candidate):
        return None
    return candidate


def _looks_like_specialty_or_service_token(token: str, text: str = "") -> bool:
    """
    Отсекает псевдо-ФИО, когда в doctor_name попала специальность/услуга.

    :param token: кандидат на фамилию врача
    :param text: исходный текст пользователя
    :return: True, если токен похож на специальность/услугу, а не на фамилию
    """
    norm = str(token or "").strip().lower().replace("ё", "е")
    if not norm:
        return False
    if norm in _SPECIALTY_LIKE_NAME_TOKENS:
        return True
    spec = str(extract_specialty(text or "") or "").strip().lower().replace("ё", "е")
    if spec and norm == spec:
        return True
    return False


def _extract_schedule_specialty(text: str) -> str | None:
    """
    Извлекает специальность для запросов расписания.

    :param text: текст запроса
    :return: специальность или None
    """
    spec = extract_specialty(text or "")
    if spec:
        return spec
    if re.search(r"\bузи\b", text or "", re.I):
        return "узи"
    return None


def _is_procedure_doctor_info_query(text: str, service_name: str | None) -> bool:
    """
    Определяет запрос "какой врач делает конкретную процедуру".

    :param text: текст пользователя
    :param service_name: извлеченная услуга
    :return: True, если это поиск врача по процедуре
    """
    if not service_name:
        return False
    return bool(_PROCEDURE_DOCTOR_INFO_RE.search(text or ""))


def _pending_waits_appointment_patient_name(last_entities: dict[str, Any]) -> bool:
    pending = last_entities.get("_pending")
    if not isinstance(pending, dict):
        return False
    if pending.get("label") != "APPOINTMENT":
        return False
    missing = pending.get("missing")
    if not isinstance(missing, list):
        return False
    return any(str(slot).strip() == "patient_name" for slot in missing)


def _looks_like_patient_name_only(text: str) -> bool:
    s = str(text or "").strip()
    if not _PATIENT_NAME_ONLY_RE.fullmatch(s):
        return False
    if has_datetime_signal(s):
        return False
    tokens = [t.lower().replace("ё", "е") for t in re.findall(r"[А-Яа-яЁёA-Za-z\-]+", s) if t]
    if len(tokens) < 2:
        return False
    if any(t in _PATIENT_NAME_STOPWORDS for t in tokens):
        return False
    return True


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
    tmpl = load_prompt_text("classifier_patient")
    hints = _collect_rule_intent_hints(text, seeded)
    return (
        tmpl.replace("<<ALLOWED_LABELS>>", allowed)
        .replace("<<SEEDED>>", json.dumps(seeded, ensure_ascii=False))
        .replace("<<RULE_HINTS>>", json.dumps(hints, ensure_ascii=False))
        .replace("<<TEXT>>", text)
    ).strip()


def _refine_allowed(base_label: Label) -> list[str]:
    if base_label == "DOCTOR_SCHEDULE":
        return ["DOCTOR_SCHEDULE", "DOCTOR_INFO", "APPOINTMENT", "OTHER"]
    if base_label == "DOCTOR_INFO":
        return ["DOCTOR_INFO", "DOCTOR_SCHEDULE", "APPOINTMENT", "OTHER"]
    if base_label == "APPOINTMENT":
        return ["APPOINTMENT", "DOCTOR_SCHEDULE", "DOCTOR_INFO", "PRICE", "OTHER"]
    return [base_label]


def _build_refine_prompt(text: str, seeded: dict[str, Any], base: RouteDecision) -> str:
    tmpl = load_prompt_text("classifier_refine_patient")
    allowed = ", ".join(_refine_allowed(base.label))
    base_dump = {
        "label": base.label,
        "confidence": base.confidence,
        "context_action": base.context_action,
        "entities": base.entities,
        "flags": sorted(list(base.flags)),
    }
    return (
        tmpl.replace("<<ALLOWED_LABELS>>", allowed)
        .replace("<<BASE_DECISION>>", json.dumps(base_dump, ensure_ascii=False))
        .replace("<<SEEDED>>", json.dumps(seeded, ensure_ascii=False))
        .replace("<<TEXT>>", text)
    ).strip()


async def _maybe_refine_live_intent(
    text: str,
    last_entities: dict[str, Any],
    base: RouteDecision,
    *,
    runtime_options: RuntimeOptions | None = None,
) -> RouteDecision:
    if base.label not in _REFINE_INTENTS:
        return base
    # Не гоняем лишний LLM-call на понятных фразах, чтобы не ухудшать стабильность.
    # Рефайн включаем только для "серых" кейсов: нет ключевых сущностей или явное переключение/уточнение.
    if base.label == "DOCTOR_SCHEDULE" and base.entities.get("doctor_name") and not _REFINE_SIGNAL_RE.search(text or ""):
        return base
    if (
        base.label == "APPOINTMENT"
        and (base.entities.get("doctor_name") or base.entities.get("service_name") or base.entities.get("appointment_action"))
        and not _REFINE_SIGNAL_RE.search(text or "")
    ):
        return base

    seeded = _seed_entities_from_memory(last_entities)
    prompt = _build_refine_prompt(text, seeded, base)
    queue_timeout_ms = int(runtime_options.queue_timeout_ms) if runtime_options else 30000
    data = await ollama_classify_json(prompt, queue_timeout_ms=queue_timeout_ms)
    data_flags = _normalize_flags(data.get("flags"))
    if "ollama_timeout" in data_flags or "ollama_non_json" in data_flags:
        return RouteDecision(
            label=base.label,
            confidence=base.confidence,
            entities=base.entities,
            flags=set(base.flags) | data_flags | {"llm_refine_unavailable"},
            needs_handoff=base.needs_handoff,
            context_action=base.context_action,
        )

    cand_label = _normalize_label(data.get("label"))
    allowed = set(_refine_allowed(base.label))
    if cand_label not in allowed:
        cand_label = base.label

    cand_conf = _normalize_confidence(data.get("confidence"))
    if cand_conf < 0.45:
        return base

    cand_entities = _sanitize_entities(data.get("entities"))
    entities = dict(base.entities)
    entities.update({k: v for k, v in cand_entities.items() if v not in (None, "", [])})

    # Бережно сохраняем уже найденный appointment_action.
    if base.entities.get("appointment_action") and not entities.get("appointment_action"):
        entities["appointment_action"] = base.entities.get("appointment_action")

    context_action = _normalize_context_action(data.get("context_action"))
    if context_action == "continue":
        derived = _derive_context_action(text, cand_label, entities, last_entities)
        if derived != "continue":
            context_action = derived

    return RouteDecision(
        label=cand_label,
        confidence=max(base.confidence, cand_conf),
        entities=entities,
        flags=set(base.flags) | data_flags | {"llm_refine_used"},
        needs_handoff=False,
        context_action=context_action,
    )


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


def _normalize_clarify_reason(x: Any) -> str:
    if isinstance(x, str) and x in _ALLOWED_CLARIFY_REASONS:
        return x
    return ""


def _normalize_intent_candidates(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        label = item.strip()
        if label in PATIENT_LABEL_PRIORITY and label not in out:
            out.append(label)
    return out[:4]


def _normalize_clarify_slots(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        slot = str(item or "").strip()
        if slot and slot not in out:
            out.append(slot)
    return out[:8]


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
    prepare_intent = detect_prepare_intent(text)
    nonbookable_walkin = detect_nonbookable_walkin_intent(text, ctx)

    if detect_test_result_intent(text):
        labels.add("TEST_RESULT")
    if nonbookable_walkin and not prepare_intent:
        labels.add("ADDRESS")
    if appointment_intent and appt_ctx and not (price_intent and appointment_action is None):
        labels.add("APPOINTMENT")
    if price_intent:
        labels.add("PRICE")
    if detect_address_intent(text):
        labels.add("ADDRESS")
    if prepare_intent:
        labels.add("PREPARE")
    if detect_test_assist_intent(text):
        labels.add("TEST_ASSIST")
    if detect_news_intent(text):
        labels.add("NEWS")
    if detect_doctor_info_intent(text):
        labels.add("DOCTOR_INFO")
    if detect_schedule_intent(text):
        labels.add("DOCTOR_SCHEDULE")

    return sorted(labels, key=lambda x: _LABEL_RANK.get(x, 10**9))


async def guardrail_precheck(
    text: str,
    last_entities: dict[str, Any],
    *,
    flags: set[str] | None = None,
) -> RouteDecision | None:
    """Hard-gated guardrails до запуска primary LLM NLU."""
    local_flags: set[str] = set(flags or set())
    local_flags |= detect_pii(text)

    if _is_smalltalk_greeting(text):
        return RouteDecision(
            label="OTHER",
            confidence=0.99,
            entities={},
            flags=local_flags | {"smalltalk_greeting"},
            needs_handoff=False,
            context_action="continue",
            source="guardrail",
        )
    if detect_urgent(text):
        return RouteDecision(
            label="URGENT",
            confidence=1.0,
            entities={},
            flags=local_flags | {"urgent"},
            needs_handoff=True,
            context_action="new_topic",
            source="guardrail",
        )
    if detect_complaint(text):
        return RouteDecision(
            label="COMPLAINT",
            confidence=1.0,
            entities={},
            flags=local_flags | {"complaint"},
            needs_handoff=True,
            context_action="new_topic",
            source="guardrail",
        )
    if detect_medical_advice(text) or detect_test_interpretation(text):
        return RouteDecision(
            label="MEDICAL_ADVICE",
            confidence=1.0,
            entities={},
            flags=local_flags | {"medical_advice"},
            needs_handoff=True,
            context_action="new_topic",
            source="guardrail",
        )
    if detect_doc_request_intent(text):
        doc_kind = "tax" if detect_tax_doc_request_intent(text) else "generic"
        kind_flag = "doc_request_tax" if doc_kind == "tax" else "doc_request_generic"
        return RouteDecision(
            label="OTHER",
            confidence=0.85,
            entities={"doc_request_kind": doc_kind},
            flags=local_flags | {"doc_request_main_index", kind_flag},
            needs_handoff=False,
            context_action="new_topic",
            source="guardrail",
        )
    _ = last_entities
    return None


def _postprocess_primary_decision(
    text: str,
    last_entities: dict[str, Any],
    data: dict[str, Any],
) -> RouteDecision:
    label = _normalize_label(data.get("label"))
    conf = _normalize_confidence(data.get("confidence"))
    entities = _sanitize_entities(data.get("entities"))
    flags = _normalize_flags(data.get("flags")) | detect_pii(text)
    context_action = _normalize_context_action(data.get("context_action"))
    clarify_needed = bool(data.get("clarify_needed"))
    clarify_reason = _normalize_clarify_reason(data.get("clarify_reason"))
    clarify_slots = _normalize_clarify_slots(data.get("clarify_slots"))
    intent_candidates = _normalize_intent_candidates(data.get("intent_candidates"))

    if not entities.get("order_id"):
        oid = _extract_order_id(text)
        if oid:
            entities["order_id"] = oid

    if label == "TEST_RESULT" and not entities.get("result_action"):
        t = text.lower()
        entities["result_action"] = "get_pdf" if ("pdf" in t or "пдф" in t or "файл" in t or "скач" in t) else "status"

    if label == "PRICE" and not entities.get("doctor_name"):
        doctor_name = _extract_price_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name

    if low_confidence_policy(conf):
        flags.add("low_confidence")

    hints = _collect_rule_intent_hints(text, last_entities)
    if not intent_candidates:
        intent_candidates = [lbl for lbl in hints if lbl != label][:3]

    if label == "OTHER" and "low_confidence" in flags and intent_candidates:
        clarify_needed = True
        clarify_reason = clarify_reason or "intent_disambiguation"

    missing = missing_slots(label, {**last_entities, **entities})
    if label not in {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"} and missing:
        clarify_needed = True
        clarify_reason = clarify_reason or "slot_request"
        if not clarify_slots:
            clarify_slots = missing

    if context_action == "continue":
        derived_action = _derive_context_action(text, label, entities, last_entities)
        if derived_action != "continue":
            context_action = derived_action

    return _attach_secondary_intents(
        text,
        RouteDecision(
            label=label,
            confidence=conf,
            entities=entities,
            flags=flags,
            needs_handoff=False,
            context_action=context_action,
            source="llm_primary",
            clarify_needed=clarify_needed,
            clarify_reason=clarify_reason,
            clarify_slots=clarify_slots,
            intent_candidates=intent_candidates,
        ),
        last_entities,
    )


async def analyze_llm_primary(
    text: str,
    last_entities: dict[str, Any],
    runtime_options: RuntimeOptions | None = None,
) -> tuple[RouteDecision, dict[str, Any]]:
    flags: set[str] = set()
    guardrail = await guardrail_precheck(text, last_entities, flags=flags)
    if guardrail is not None:
        return guardrail, {
            "guardrail_pre": {
                "hit": True,
                "label": guardrail.label,
                "flags": sorted(list(guardrail.flags)),
            },
            "llm_primary_raw": "",
            "llm_primary_sanitized": {},
            "guardrail_post": {},
            "final_decision": {
                "label": guardrail.label,
                "source": guardrail.source,
            },
        }

    seeded = _seed_entities_from_memory(last_entities)
    prompt = _build_classify_prompt(text, seeded)
    queue_timeout_ms = int(runtime_options.queue_timeout_ms) if runtime_options else 30000
    raw_text, data = await ollama_classify_payload(prompt, queue_timeout_ms=queue_timeout_ms)
    decision = _postprocess_primary_decision(text, last_entities, data)

    if runtime_options and runtime_options.allows_refine_pass and decision.clarify_needed and decision.label in _REFINE_INTENTS:
        refined = await _maybe_refine_live_intent(text, last_entities, decision, runtime_options=runtime_options)
        if refined.label != decision.label or refined.entities != decision.entities:
            decision = RouteDecision(
                label=refined.label,
                confidence=max(decision.confidence, refined.confidence),
                entities=dict(refined.entities),
                flags=set(decision.flags) | set(refined.flags),
                needs_handoff=False,
                context_action=refined.context_action,
                source="llm_primary",
                clarify_needed=decision.clarify_needed,
                clarify_reason=decision.clarify_reason,
                clarify_slots=list(decision.clarify_slots),
                intent_candidates=list(decision.intent_candidates),
            )

    trace = {
        "guardrail_pre": {"hit": False},
        "llm_primary_raw": raw_text,
        "llm_primary_sanitized": data,
        "guardrail_post": {
            "clarify_needed": decision.clarify_needed,
            "clarify_reason": decision.clarify_reason,
            "clarify_slots": list(decision.clarify_slots),
            "intent_candidates": list(decision.intent_candidates),
            "flags": sorted(list(decision.flags)),
        },
        "final_decision": {
            "label": decision.label,
            "confidence": decision.confidence,
            "source": decision.source,
        },
    }
    return decision, trace


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
        source=decision.source,
        clarify_needed=decision.clarify_needed,
        clarify_reason=decision.clarify_reason,
        clarify_slots=list(decision.clarify_slots),
        intent_candidates=list(decision.intent_candidates),
    )


async def deterministic_rule_decision(
    text: str,
    last_entities: dict[str, Any],
    *,
    flags: set[str] | None = None,
    runtime_options: RuntimeOptions | None = None,
    allow_refine: bool = True,
    attach_secondary: bool = True,
) -> RouteDecision | None:
    """Детерминированный pass интентов без LLM-fallback классификации."""
    local_flags: set[str] = set(flags or set())
    local_flags |= detect_pii(text)

    decision: RouteDecision | None = None

    if _is_smalltalk_greeting(text):
        decision = RouteDecision(
            label="OTHER",
            confidence=0.99,
            entities={},
            flags=local_flags | {"smalltalk_greeting"},
            needs_handoff=False,
            context_action="continue",
        )
    elif detect_urgent(text):
        decision = RouteDecision(
            label="URGENT",
            confidence=1.0,
            entities={},
            flags=local_flags | {"urgent"},
            needs_handoff=True,
            context_action="new_topic",
        )
    elif detect_complaint(text):
        decision = RouteDecision(
            label="COMPLAINT",
            confidence=1.0,
            entities={},
            flags=local_flags | {"complaint"},
            needs_handoff=True,
            context_action="new_topic",
        )
    elif detect_medical_advice(text) or detect_test_interpretation(text):
        decision = RouteDecision(
            label="MEDICAL_ADVICE",
            confidence=1.0,
            entities={},
            flags=local_flags | {"medical_advice"},
            needs_handoff=True,
            context_action="new_topic",
        )
    elif detect_doc_request_intent(text):
        doc_kind = "tax" if detect_tax_doc_request_intent(text) else "generic"
        kind_flag = "doc_request_tax" if doc_kind == "tax" else "doc_request_generic"
        decision = RouteDecision(
            label="OTHER",
            confidence=0.85,
            entities={"doc_request_kind": doc_kind},
            flags=local_flags | {"doc_request_main_index", kind_flag},
            needs_handoff=False,
            context_action="new_topic",
        )
    elif detect_test_result_intent(text):
        if should_treat_result_delivery_as_test_assist(text):
            decision = RouteDecision(
                label="TEST_ASSIST",
                confidence=0.7,
                entities={},
                flags=local_flags | {"rule_test_assist", "result_delivery_info"},
                needs_handoff=False,
                context_action="continue",
            )
        else:
            entities: dict[str, Any] = {}
            oid = _extract_order_id(text)
            if oid:
                entities["order_id"] = oid
            decision = RouteDecision(
                label="TEST_RESULT",
                confidence=0.85,
                entities=entities,
                flags=local_flags | {"rule_test_result"},
                needs_handoff=False,
                context_action="continue",
            )
    elif _pending_waits_appointment_patient_name(last_entities) and _looks_like_patient_name_only(text):
        decision = RouteDecision(
            label="APPOINTMENT",
            confidence=0.9,
            entities={"patient_name": str(text or "").strip()},
            flags=local_flags | {"rule_appointment_patient_name"},
            needs_handoff=False,
            context_action="continue",
        )
    elif detect_schedule_intent(text):
        entities: dict[str, Any] = {}
        specialty = _extract_schedule_specialty(text)
        doctor_name = _extract_schedule_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name
        if specialty:
            entities["specialty"] = specialty
        base = RouteDecision(
            label="DOCTOR_SCHEDULE",
            confidence=0.74,
            entities=entities,
            flags=local_flags | {"rule_schedule"},
            needs_handoff=False,
            context_action=_derive_context_action(text, "DOCTOR_SCHEDULE", entities, last_entities),
        )
        decision = (
            await _maybe_refine_live_intent(text, last_entities, base, runtime_options=runtime_options)
            if allow_refine
            else base
        )
    elif detect_doctor_info_intent(text):
        entities: dict[str, Any] = {}
        doctor_name = _extract_appointment_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name
        base = RouteDecision(
            label="DOCTOR_INFO",
            confidence=0.72,
            entities=entities,
            flags=local_flags | {"rule_doctor_info"},
            needs_handoff=False,
            context_action=_derive_context_action(text, "DOCTOR_INFO", entities, last_entities),
        )
        decision = (
            await _maybe_refine_live_intent(text, last_entities, base, runtime_options=runtime_options)
            if allow_refine
            else base
        )
    elif _is_procedure_doctor_info_query(text, _extract_service_keyword(text)):
        svc = _extract_service_keyword(text)
        entities = {}
        if svc:
            entities["service_name"] = svc
        decision = RouteDecision(
            label="DOCTOR_INFO",
            confidence=0.71,
            entities=entities,
            flags=local_flags | {"rule_doctor_info_service"},
            needs_handoff=False,
            context_action="continue",
        )
    else:
        appointment_intent = detect_appointment_intent(text)
        appointment_action = normalize_appointment_action(detect_appointment_action(text), text)
        price_intent = detect_price_intent(text)
        address_intent = detect_address_intent(text)
        specialty = extract_specialty(text or "")
        appt_ctx = has_appointment_context(text, last_entities, appointment_action)
        prepare_intent = detect_prepare_intent(text)
        nonbookable_walkin = detect_nonbookable_walkin_intent(text, last_entities)
        address_dominant = is_address_dominant_intent(
            text,
            address_intent=address_intent,
            price_intent=price_intent,
            appointment_action=appointment_action,
        )
        if specialty and has_nearest_schedule_hint(text):
            decision = RouteDecision(
                label="DOCTOR_SCHEDULE",
                confidence=0.72,
                entities={"specialty": specialty},
                flags=local_flags | {"rule_schedule_nearest", "rule_schedule_with_specialty"},
                needs_handoff=False,
                context_action="continue",
            )
        elif specialty and not appointment_intent:
            decision = RouteDecision(
                label="DOCTOR_INFO",
                confidence=0.70,
                entities={"specialty": specialty},
                flags=local_flags | {"rule_doctor_info_specialty"},
                needs_handoff=False,
                context_action="continue",
            )
        elif address_dominant:
            decision = RouteDecision(
                label="ADDRESS",
                confidence=0.72,
                entities={},
                flags=local_flags | {"rule_address"},
                needs_handoff=False,
                context_action="continue",
            )
        else:
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
                doctor_name = _extract_price_doctor_name(text)
                if doctor_name:
                    entities["doctor_name"] = doctor_name
                decision = RouteDecision(
                    label="PRICE",
                    confidence=0.72,
                    entities=entities,
                    flags=local_flags | {"rule_price"},
                    needs_handoff=False,
                    context_action="continue",
                )
            elif nonbookable_walkin and not prepare_intent:
                entities = {}
                svc = nonbookable_service_hint(text)
                if svc:
                    entities["service_name"] = svc
                decision = RouteDecision(
                    label="ADDRESS",
                    confidence=0.78,
                    entities=entities,
                    flags=local_flags | {"rule_nonbookable_walkin"},
                    needs_handoff=False,
                    context_action="continue",
                )
            elif appointment_intent and appt_ctx and not price_dominant:
                entities = {}
                if appointment_action:
                    entities["appointment_action"] = appointment_action
                doctor_name = _extract_appointment_doctor_name(text)
                if doctor_name:
                    entities["doctor_name"] = doctor_name
                svc = _extract_service_keyword(text)
                if svc and not service_name_conflicts_with_doctor(svc, doctor_name):
                    entities["service_name"] = svc
                base = RouteDecision(
                    label="APPOINTMENT",
                    confidence=0.75,
                    entities=entities,
                    flags=local_flags | {"rule_appointment"},
                    needs_handoff=False,
                    context_action=_derive_context_action(text, "APPOINTMENT", entities, last_entities),
                )
                decision = (
                    await _maybe_refine_live_intent(text, last_entities, base, runtime_options=runtime_options)
                    if allow_refine
                    else base
                )
            elif detect_news_intent(text):
                decision = RouteDecision(
                    label="NEWS",
                    confidence=0.72,
                    entities={},
                    flags=local_flags | {"rule_news"},
                    needs_handoff=False,
                    context_action="continue",
                )
            elif detect_prepare_intent(text):
                decision = RouteDecision(
                    label="PREPARE",
                    confidence=0.72,
                    entities={},
                    flags=local_flags | {"rule_prepare"},
                    needs_handoff=False,
                    context_action="continue",
                )
            elif detect_test_assist_intent(text):
                decision = RouteDecision(
                    label="TEST_ASSIST",
                    confidence=0.7,
                    entities={},
                    flags=local_flags | {"rule_test_assist"},
                    needs_handoff=False,
                    context_action="continue",
                )
            elif price_intent:
                entities = {}
                svc = _extract_service_keyword(text)
                if svc:
                    entities["service_name"] = svc
                doctor_name = _extract_price_doctor_name(text)
                if doctor_name:
                    entities["doctor_name"] = doctor_name
                decision = RouteDecision(
                    label="PRICE",
                    confidence=0.72,
                    entities=entities,
                    flags=local_flags | {"rule_price"},
                    needs_handoff=False,
                    context_action="continue",
                )
            elif detect_address_intent(text):
                decision = RouteDecision(
                    label="ADDRESS",
                    confidence=0.72,
                    entities={},
                    flags=local_flags | {"rule_address"},
                    needs_handoff=False,
                    context_action="continue",
                )

    if decision is None:
        return None
    if attach_secondary:
        return _attach_secondary_intents(text, decision, last_entities)
    return decision


async def analyze(
    text: str,
    last_entities: dict[str, Any],
    runtime_options: RuntimeOptions | None = None,
    prefetched_rule: RouteDecision | None = None,
) -> RouteDecision:
    flags: set[str] = set()
    flags |= detect_pii(text)

    rule = prefetched_rule
    if rule is None:
        rule = await deterministic_rule_decision(
            text,
            last_entities,
            flags=flags,
            runtime_options=runtime_options,
            allow_refine=True,
            attach_secondary=True,
        )
    elif flags and not flags.issubset(set(rule.flags)):
        # Бережно добавляем PII-флаги, если caller передал precomputed-rule без них.
        rule = RouteDecision(
            label=rule.label,
            confidence=rule.confidence,
            entities=dict(rule.entities),
            flags=set(rule.flags) | flags,
            needs_handoff=rule.needs_handoff,
            context_action=rule.context_action,
            source=rule.source,
            clarify_needed=rule.clarify_needed,
            clarify_reason=rule.clarify_reason,
            clarify_slots=list(rule.clarify_slots),
            intent_candidates=list(rule.intent_candidates),
        )
    if rule is not None:
        return rule

    # light hints
    if detect_test_result_intent(text):
        flags.add("hint_test_result")
    if detect_test_assist_intent(text):
        flags.add("hint_test_assist")
    if detect_prepare_intent(text):
        flags.add("hint_prepare")
    if detect_schedule_intent(text):
        flags.add("hint_schedule")
    if detect_news_intent(text):
        flags.add("hint_news")

    seeded = _seed_entities_from_memory(last_entities)

    prompt = _build_classify_prompt(text, seeded)
    queue_timeout_ms = int(runtime_options.queue_timeout_ms) if runtime_options else 30000
    data = await ollama_classify_json(prompt, queue_timeout_ms=queue_timeout_ms)

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
    if label == "PRICE" and not entities.get("doctor_name"):
        doctor_name = _extract_price_doctor_name(text)
        if doctor_name:
            entities["doctor_name"] = doctor_name

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
            source="fallback",
        ),
        last_entities,
    )
