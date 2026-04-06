"""Построение структурированных ответов patient-router.

Модуль формирует deterministic-ответы по `flow_label` и `evidence`,
не занимаясь NLU, планированием и выполнением сервисных шагов.
"""

from __future__ import annotations

from typing import Any

from .flow_policy import hydrate_appointment_context_from_schedule, safe_get_branches
from .memory import MemoryStore
from .mess_types import Evidence, ResponseEnvelope, RouteDecision, SessionState
from .policies import (
    APPOINTMENT_STEP_BRANCH,
    APPOINTMENT_STEP_CONFIRM,
    APPOINTMENT_STEP_DATETIME,
    APPOINTMENT_STEP_PATIENT,
    appointment_addresses_for_city,
    appointment_service_display,
    appointment_step_policy,
    appointment_summary,
    appointment_text_branch_prompt,
    appointment_text_confirm_prompt,
    appointment_text_datetime_prompt,
    appointment_text_patient_name_prompt,
    clarification_question,
    extract_price_rub,
    nonbookable_service_hint,
)
from .renderer import (
    format_address_for_patient,
    format_doctor_info_for_patient,
    format_doctor_schedule_for_patient,
    format_news_for_patient,
    format_price_for_patient,
    format_service_bundle_for_patient,
)
from .services import Services

_DEFAULT_CITY = "Самара"

_UNSUPPORTED_CATALOG_TEXT: dict[str, str] = {
    "unsupported_service": "К сожалению, в данный момент клиника не оказывает данную услугу. Приносим извинения за неудобства.",
    "unsupported_specialist": "Данные врачи не ведут прием.",
    "unsupported_document_service": "Наша клиника не оказывает данные услуги.",
}


def build_unsupported_catalog_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get("unsupported_catalog")
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind") or "").strip()
    text = _UNSUPPORTED_CATALOG_TEXT.get(kind)
    if not text:
        return None
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_operator_offer_response(evidence: Evidence) -> ResponseEnvelope | None:
    """
    Возвращает ответ для подтверждения/обработки перевода на оператора.

    :param evidence: собранные evidence текущего шага
    :return: готовый envelope или None
    """

    payload = evidence.get("operator_offer_response")
    if not isinstance(payload, dict):
        return None
    text = str(payload.get("text") or "").strip()
    if not text:
        return None
    return ResponseEnvelope(
        text=text,
        attachments=[],
        handoff=bool(payload.get("handoff")),
    )


def build_price_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "PRICE":
        return None
    price_payload = evidence.get("price")
    if not isinstance(price_payload, dict):
        return None
    render_entities = dict(state.last_entities or {})
    used = price_payload.get("entities_used")
    if isinstance(used, dict):
        doctor_id_resolved = used.get("doctor_id_resolved")
        doctor_name_resolved = used.get("doctor_name_resolved")
        service_name_effective = str(used.get("service_name_effective") or "").strip()
        if doctor_id_resolved:
            render_entities["doctor_id"] = doctor_id_resolved
        if isinstance(doctor_name_resolved, str) and doctor_name_resolved.strip():
            render_entities["doctor_name"] = doctor_name_resolved.strip()
        if service_name_effective:
            render_entities["service_name"] = service_name_effective
            render_entities.pop("test_name", None)
    text = format_price_for_patient(price_payload, render_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_service_bundle_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "PRICE":
        return None
    payload = evidence.get("service_bundle")
    if not isinstance(payload, dict):
        return None
    text = format_service_bundle_for_patient(payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_doctor_info_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "DOCTOR_INFO":
        return None
    doctors_info_payload = evidence.get("doctors_info")
    if not isinstance(doctors_info_payload, dict):
        return None
    doctors_raw = doctors_info_payload.get("doctors")
    doctors = doctors_raw if isinstance(doctors_raw, list) else []
    used = doctors_info_payload.get("entities_used")
    explicit_doctor_query = False
    if isinstance(used, dict):
        explicit_doctor_query = bool(str(used.get("doctor_query") or "").strip() or str(used.get("doctor_resolved") or "").strip())

    # Синхронизируем doctor-context после выдачи списка:
    # - 1 врач -> закрепляем его как текущего для follow-up "расписание";
    # - >1 врача и без явной фамилии в запросе -> убираем stale doctor_name,
    #   чтобы не показывать расписание "чужого" врача из старого контекста.
    if len(doctors) == 1 and isinstance(doctors[0], dict):
        only = doctors[0]
        fio = str(only.get("fio") or "").strip()
        if fio:
            state.last_entities["doctor_name"] = fio
        did = only.get("id")
        if did is not None:
            state.last_entities["doctor_id"] = did
    elif len(doctors) > 1 and not explicit_doctor_query:
        state.last_entities.pop("doctor_name", None)
        state.last_entities.pop("doctor_id", None)

    text = format_doctor_info_for_patient(doctors_info_payload, state.last_entities)
    price_payload = evidence.get("price")
    if isinstance(price_payload, dict):
        raw_prices = price_payload.get("prices")
        if isinstance(raw_prices, list) and raw_prices:
            text = (
                f"{text}\n\n"
                "Примеры стоимости услуг этого врача:\n"
                f"{format_price_for_patient(price_payload, state.last_entities)}"
            ).strip()
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_test_result_response(flow_label: str, evidence: Evidence) -> ResponseEnvelope | None:
    if flow_label != "TEST_RESULT":
        return None
    result_status = evidence.get("test_result_status")
    if not isinstance(result_status, dict):
        return None

    if result_status.get("ready") is True:
        note = str(result_status.get("note") or "")
        preview = str(result_status.get("result_preview") or "").strip()
        links_raw = result_status.get("result_links")
        links = [str(x).strip() for x in links_raw] if isinstance(links_raw, list) else []
        links = [x for x in links if x]
        if note == "result_link_constructed":
            text = "Сформировал ссылку для просмотра результата по указанным данным."
        else:
            text = "Результаты по вашим данным найдены."
        if links:
            if len(links) == 1:
                text = f"{text}\n\nСсылка на результат: {links[0]}"
            else:
                lines = "\n".join(f"- {u}" for u in links[:5])
                text = f"{text}\n\nСсылки на результаты:\n{lines}"
        if preview and not links:
            text = f"{text}\n\n{preview}"
        return ResponseEnvelope(text=text, attachments=[], handoff=False)

    missing = result_status.get("missing_fields")
    if isinstance(missing, list) and missing:
        return ResponseEnvelope(
            text=clarification_question("TEST_RESULT", [str(m) for m in missing]),
            attachments=[],
            handoff=False,
        )

    preview = str(result_status.get("result_preview") or "").strip()
    text = preview or "По указанным данным результаты пока не найдены или ещё не готовы."
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_prepare_response(flow_label: str, evidence: Evidence) -> ResponseEnvelope | None:
    if flow_label != "PREPARE":
        return None
    payload = evidence.get("prepare")
    if not isinstance(payload, dict):
        return None
    text = str(payload.get("prepare") or "").strip()
    if not text:
        return None
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_doctor_schedule_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    if flow_label != "DOCTOR_SCHEDULE":
        return None
    schedule_payload = evidence.get("doctor_schedule")
    if not isinstance(schedule_payload, dict):
        return None
    if str(schedule_payload.get("schedule_unavailable_reason") or "").strip() == "no_free_slots_2_weeks":
        state.last_entities["_operator_offer_pending"] = True
        memory.set_pending(state, label="OTHER", missing_slots=["operator_offer_confirm"])
        return ResponseEnvelope(
            text=(
                "Врач найден, но свободных слотов нет в ближайшие 2 недели. "
                "Для уточнения могу перевести на оператора. Перевести на оператора?"
            ),
            attachments=[],
            handoff=False,
        )
    hydrate_appointment_context_from_schedule(state, schedule_payload)
    text = format_doctor_schedule_for_patient(schedule_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_address_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    memory: MemoryStore,
    decision: RouteDecision,
    user_text: str,
) -> ResponseEnvelope | None:
    if flow_label != "ADDRESS":
        return None
    address_payload = evidence.get("address")
    if not isinstance(address_payload, dict):
        return None

    branches_raw = address_payload.get("branches")
    addresses_raw = address_payload.get("addresses")
    has_branches = isinstance(branches_raw, list) and any(
        str((x or {}).get("address") if isinstance(x, dict) else x).strip() for x in branches_raw
    )
    has_addresses = isinstance(addresses_raw, list) and any(str(x).strip() for x in addresses_raw)
    if not has_branches and not has_addresses:
        state.last_entities.pop("city", None)
        memory.set_pending(state, label="ADDRESS", missing_slots=["_any_of:city,branch_name,branch_id"])

    walkin_hint: str | None = None
    if any("nonbookable" in str(f) for f in decision.flags):
        walkin_hint = nonbookable_service_hint(user_text, state.last_entities) or str(state.last_entities.get("service_name") or "").strip()
        if walkin_hint == "":
            walkin_hint = None
    text = format_address_for_patient(
        address_payload,
        state.last_entities,
        nonbookable_service=walkin_hint,
    )
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_news_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "NEWS":
        return None
    news_payload = evidence.get("news")
    if not isinstance(news_payload, dict):
        return None
    text = format_news_for_patient(news_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_main_index_info_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get("main_index_info")
    if not isinstance(payload, dict):
        return None
    content = str(payload.get("content") or "").strip()
    if content:
        return ResponseEnvelope(text=content, attachments=[], handoff=False)
    return None


def build_appointment_schedule_preview_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
) -> ResponseEnvelope | None:
    if flow_label != "APPOINTMENT":
        return None
    schedule_payload = evidence.get("doctor_schedule")
    if not isinstance(schedule_payload, dict):
        return None
    if not (state.last_entities.get("doctor_name") or state.last_entities.get("doctor_id")):
        return None
    if state.last_entities.get("date_from") or state.last_entities.get("date_hint"):
        return None
    if state.last_entities.get("time_from"):
        return None

    hydrate_appointment_context_from_schedule(state, schedule_payload)
    state.last_entities["appointment_flow_active"] = True
    text = format_doctor_schedule_for_patient(schedule_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_appointment_step_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    if flow_label != "APPOINTMENT":
        return None

    entities = state.last_entities
    state.last_entities["appointment_flow_active"] = True

    appointment_step = appointment_step_policy(entities)
    service = appointment_service_display(entities)
    city = str(entities.get("city") or "").strip()

    if appointment_step == APPOINTMENT_STEP_BRANCH:
        if not city:
            city = _DEFAULT_CITY
            state.last_entities["city"] = city
        stored_options = state.last_entities.get("appointment_branch_options")
        addresses = []
        if isinstance(stored_options, list):
            addresses = [str(x).strip() for x in stored_options if str(x).strip()]
        if not addresses:
            branches = safe_get_branches(services)
            addresses = appointment_addresses_for_city(
                evidence.get("address"),
                branches,
                city=city or None,
                limit=5,
            )
        state.last_entities["appointment_branch_options"] = addresses
        return ResponseEnvelope(
            text=appointment_text_branch_prompt(service, city, addresses),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_DATETIME:
        state.last_entities.pop("appointment_branch_options", None)
        price_rub = extract_price_rub(evidence.get("price"))
        branch = str(entities.get("branch_name") or entities.get("city") or "выбранном филиале").strip()
        return ResponseEnvelope(
            text=appointment_text_datetime_prompt(service, branch, price_rub),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_PATIENT:
        memory.set_pending(state, label="APPOINTMENT", missing_slots=["patient_name"])
        return ResponseEnvelope(
            text=appointment_text_patient_name_prompt(),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_CONFIRM:
        state.last_entities["appointment_confirm_pending"] = True
        summary = appointment_summary(entities)
        return ResponseEnvelope(text=appointment_text_confirm_prompt(summary), handoff=False)

    return None


def build_first_structured_response(
    *,
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    services: Services,
    memory: MemoryStore,
    decision: RouteDecision,
    user_text: str,
) -> ResponseEnvelope | None:
    builders = (
        lambda: build_operator_offer_response(evidence),
        lambda: build_unsupported_catalog_response(evidence),
        lambda: build_main_index_info_response(evidence),
        lambda: build_service_bundle_response(flow_label, evidence, state),
        lambda: build_price_response(flow_label, evidence, state),
        lambda: build_test_result_response(flow_label, evidence),
        lambda: build_prepare_response(flow_label, evidence),
        lambda: build_doctor_schedule_response(flow_label, evidence, state, memory),
        lambda: build_doctor_info_response(flow_label, evidence, state),
        lambda: build_address_response(flow_label, evidence, state, memory, decision, user_text),
        lambda: build_news_response(flow_label, evidence, state),
        lambda: build_appointment_schedule_preview_response(flow_label, evidence, state),
        lambda: build_appointment_step_response(flow_label, evidence, state, services, memory),
    )
    for build in builders:
        env = build()
        if env is not None:
            return env
    return None
