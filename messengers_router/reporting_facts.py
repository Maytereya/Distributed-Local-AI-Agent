"""Semantic evidence only. No answer-text classification and no patient data."""
import math
import re
from reporting_contract import TOPICS, VERSION, unknown_contract, validate_contract

REQUEST_MARKERS = {
    "PRICE":r"стоим|цен[ауы]|сколько\s+стоит",
    "DOCTOR_SCHEDULE":r"расписан|когда\s+(?:принима|работа)|свободн\w*\s+(?:время|окн|слот)",
    "DOCTOR_INFO":r"какие\s+врачи|кто\s+из\s+врач|информаци\w*\s+о\s+врач",
    "TEST_ASSIST":r"какие\s+анализ|какой\s+анализ|помог\w*\s+выбрать\s+анализ",
    "PREPARE":r"подготов|натощак|можно\s+ли\s+(?:есть|пить)",
    "ADDRESS":r"адрес|где\s+(?:находит|сдать|сделать|проходит)",
    "TEST_RESULT":r"результат\w*\s+анализ|готов\w*\s+(?:результат|анализ)",
    "APPOINTMENT":r"запис\w*\s+(?:на|к)|хочу\s+запис",
    "NEWS":r"акци|новост",
}


def _has_price(row):
    if not isinstance(row,dict): return False
    for key in ("servicePrice","price","service_price","amount","cost"):
        value = row.get(key)
        if type(value) in (int,float) and math.isfinite(value) and value >= 0: return True
        if isinstance(value,str) and re.search(r"\d+(?:[.,]\d+)?",value): return True
    return False


def scenario_facts(ctx, memory):
    decision, response = ctx.decision, ctx.response
    if decision is None or response is None:
        return unknown_contract()
    label = ctx.plan.label if ctx.plan else decision.label
    flags = set(decision.flags)
    values = ctx.evidence.items if ctx.evidence else {}
    entities = ctx.state.last_entities
    context = "new_topic" if decision.context_action == "new_topic" else "continue"
    meta = dict(version=VERSION,context=context,phase="unknown",tasks=[],reason="missing_telemetry",operator_event="none")
    neutral = label == "OTHER" and re.fullmatch(r"(?:спасибо(?:\s+большое)?|благодарю|понятно|ясно|ок|окей)[.!\s]*",getattr(ctx,"text","").strip().lower())
    if "operator_offer_declined" in flags:
        meta.update(phase="declined",reason="operator_declined",operator_event="declined")
        return validate_contract(meta)
    if response.handoff:
        meta.update(reason="patient_requested" if "operator_offer_confirmed" in flags else "unsupported_scenario",
                    operator_event="accepted" if "operator_offer_confirmed" in flags else "none")
        return validate_contract(meta)
    if (label == "OTHER" and "smalltalk_greeting" in flags) or neutral:
        meta.update(phase="neutral",reason="none")
        return validate_contract(meta)
    structured = bool(getattr(ctx,"_reporting_structured",False))
    if structured and values.get("unsupported_catalog"):
        meta.update(phase="unanswered",reason="unsupported_scenario",tasks=[{"key":label if label in TOPICS else "OTHER","state":"pending","kind":"none"}])
        return validate_contract(meta)
    primary = label if label in TOPICS else "OTHER"
    solved, kind = False, "structured_answer"
    # Only a deterministic response actually selected by the renderer is proof.
    payload = {}
    if structured:
        if primary == "PRICE":
            payload = values.get("price") or values.get("service_bundle") or {}
            prices = payload.get("prices") or payload.get("retail_prices") or payload.get("doctors") or []
            if payload.get("service_kind")=="family_query" and isinstance(payload.get("family_variants"),list):
                variants=payload["family_variants"]
                limit=payload.get("visible_limit",10)
                limit=max(1,limit) if type(limit) is int else 10
                prices=variants if payload.get("showing_all") else variants[:limit]
            solved = not payload.get("clarify_text") and any(_has_price(row) for row in prices)
        elif primary == "TEST_RESULT":
            payload = values.get("test_result_status") or {}
            solved = bool(payload.get("result_links") or payload.get("result_attachments")) and not payload.get("missing_fields")
            kind = "results_link"
        elif primary == "DOCTOR_INFO":
            payload = values.get("doctors_info") or {}
            solved = bool(payload.get("doctors"))
        elif primary == "DOCTOR_SCHEDULE":
            payload = values.get("doctor_schedule") or {}
            solved = bool(payload.get("schedule")) and not payload.get("schedule_unavailable_reason")
        elif primary == "ADDRESS":
            payload = values.get("address") or {}
            solved = bool(payload.get("addresses") or payload.get("branches"))
        else:
            payload = values.get("main_index_info") or {}
            if payload.get("note") in {"main_index_info: tax direct link","main_index_info: tax fallback unavailable",
                    "main_index_info: tax fallback error","main_index_info: tax fallback no matches","main_index_info: tax fallback weak relevance"}:
                primary, solved, kind = "TAX_DOCUMENT", True, "tax_link"
    pending = bool(memory.get_pending(ctx.state)) or ctx.should_clarify or decision.clarify_needed or bool(payload.get("missing_fields"))
    offered = bool(entities.get("_operator_offer_pending"))
    meta["tasks"] = [{"key":primary,"state":"solved" if solved else "pending" if pending or offered else "unknown",
                      "kind":kind if solved else "none"}]
    secondary = decision.entities.get("secondary_intents") or entities.get("_secondary_queue") or []
    if isinstance(secondary,list):
        for topic in secondary:
            if not isinstance(topic,str): continue
            # NLU also suggests optional follow-ups. Only an explicitly asked
            # second topic belongs to this appeal ("price of a test" is PRICE).
            explicit = re.search(REQUEST_MARKERS.get(topic,r"(?!)"),getattr(ctx,"text","").lower())
            if topic in TOPICS and explicit and topic != primary and len(meta["tasks"]) < 8:
                if not any(t["key"] == topic for t in meta["tasks"]):
                    meta["tasks"].append({"key":topic,"state":"pending","kind":"none"})
    if offered:
        meta.update(phase="operator_offer",reason="no_data",operator_event="offered")
    elif solved:
        meta.update(phase="answer",reason="partial_answer" if len(meta["tasks"]) > 1 else "none")
    elif pending:
        meta.update(phase="clarification",reason="clarification")
    elif structured and payload:
        meta["tasks"][0]["state"]="pending"
        meta.update(phase="unanswered",reason="no_data")
    # Unsupported and unverified render paths remain X, including generic links.
    return validate_contract(meta)
