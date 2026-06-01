"""Построение структурированных ответов patient-router.

Модуль формирует deterministic-ответы по `flow_label` и `evidence`,
не занимаясь NLU, планированием и выполнением сервисных шагов.
"""

from __future__ import annotations

from typing import Any

from . import evidence_keys as ek
from .flow_policy import (
    hydrate_appointment_context_from_schedule,
    reset_appointment_runtime_state,
    safe_get_branches,
)
from .memory import MemoryStore
from .mess_types import Evidence, ResponseEnvelope, RouteDecision, SessionState
from .state_mutations import (
    activate_appointment_flow,
    clear_appointment_branch_options,
    clear_appointment_selection_mode,
    mark_appointment_confirm_pending,
    reset_appointment_confirmation_flags,
    set_appointment_branch_options,
    set_appointment_selection_mode,
)
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
    _render_appointment_date_part,
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
_COMPOUND_PRICE_PENDING_KEY = "_compound_price_pending"

_UNSUPPORTED_CATALOG_TEXT: dict[str, str] = {
    "unsupported_service": "К сожалению, в данный момент клиника не оказывает данную услугу. Приносим извинения за неудобства.",
    "unsupported_specialist": "Данные врачи не ведут прием.",
    "unsupported_document_service": "Наша клиника не оказывает данные услуги.",
}


def build_unsupported_catalog_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get(ek.UNSUPPORTED_CATALOG)
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

    payload = evidence.get(ek.OPERATOR_OFFER_RESPONSE)
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


def build_catalog_confirm_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get(ek.CATALOG_CONFIRM_RESPONSE)
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


def build_catalog_health_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get(ek.CATALOG_HEALTH_RESPONSE)
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


def _sync_price_family_context(state: SessionState, payload: dict[str, Any]) -> None:
    """
    Сохраняет или очищает family-query контекст для follow-up `все`.

    :param state: состояние сессии
    :param payload: payload `price` или `service_bundle`
    :return: None
    """

    kind = str(payload.get("service_kind") or "").strip().lower()
    variants = payload.get("family_variants")
    if kind == "family_query" and isinstance(variants, list) and variants and not payload.get("showing_all"):
        state.last_entities["_price_family_context"] = {
            "service_name": str(payload.get("service_name") or "").strip(),
            "family_variants": variants,
            "visible_limit": int(payload.get("visible_limit") or 10),
        }
        return
    state.last_entities.pop("_price_family_context", None)


def _sync_compound_price_pending(state: SessionState, payload: dict[str, Any]) -> None:
    """
    Сохраняет короткий pending-контекст для compound PRICE-уточнения.

    :param state: состояние сессии
    :param payload: payload `service_bundle` / `price`
    :return: None
    """

    services_raw = payload.get("compound_price_services")
    services = [str(item).strip() for item in services_raw] if isinstance(services_raw, list) else []
    services = [item for item in services if item]
    default_service = str(payload.get("compound_price_default_service") or "").strip()
    clarify_text = str(payload.get("clarify_text") or "").strip()
    if len(services) >= 2 and clarify_text:
        if default_service not in services:
            default_service = services[0]
        state.last_entities[_COMPOUND_PRICE_PENDING_KEY] = {
            "services": services[:4],
            "default_service": default_service,
        }
        return
    state.last_entities.pop(_COMPOUND_PRICE_PENDING_KEY, None)


def build_price_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "PRICE":
        return None
    price_payload = evidence.get(ek.PRICE)
    if not isinstance(price_payload, dict):
        return None
    _sync_price_family_context(state, price_payload)
    _sync_compound_price_pending(state, price_payload)
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
    payload = evidence.get(ek.SERVICE_BUNDLE)
    if not isinstance(payload, dict):
        return None
    _sync_price_family_context(state, payload)
    _sync_compound_price_pending(state, payload)
    text = format_service_bundle_for_patient(payload, state.last_entities)
    # Defensive fallback: если рендер вернул пустой текст, показываем
    # пациенту инструкцию о том, как сформулировать запрос правильно.
    # Бывает, когда в state остался stale service_name (например,
    # после галлюцинации LLM или ADDRESS-флоу), и pipeline не нашёл
    # под него ни розничной цены, ни doctor-link. Раньше пациент
    # получал пустое сообщение и уходил в очередь оператора.
    # Жалоба заказчика 2026-05-05: «Сдать витамин Д» → «Стоимость» →
    # пустой ответ → «Переключаю на оператора».
    if not text or not text.strip():
        text = (
            "Чтобы я мог уточнить стоимость, напишите запрос полностью. "
            "Например:\n"
            "• «Сколько стоит общий анализ крови»\n"
            "• «Цена ТТГ»\n"
            "• «Стоимость приёма кардиолога»\n"
            "• «Стоимость УЗИ молочных желез»\n\n"
            "Так я смогу точно сказать цену и формат услуги."
        )
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_doctor_info_response(flow_label: str, evidence: Evidence, state: SessionState) -> ResponseEnvelope | None:
    if flow_label != "DOCTOR_INFO":
        return None
    doctors_info_payload = evidence.get(ek.DOCTORS_INFO)
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
    price_payload = evidence.get(ek.PRICE)
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
    result_status = evidence.get(ek.TEST_RESULT_STATUS)
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
    payload = evidence.get(ek.PREPARE)
    if not isinstance(payload, dict):
        return None
    text = str(payload.get("prepare") or "").strip()
    if not text:
        return None
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def _no_free_slots_operator_offer(state: SessionState, memory: MemoryStore) -> ResponseEnvelope:
    """Единый ответ на «врач найден, но слотов нет 2 недели» — и в DOCTOR_SCHEDULE,
    и в APPOINTMENT. Сбрасывает runtime-стейт записи и переводит диалог в ожидание
    подтверждения перевода на оператора."""
    reset_appointment_runtime_state(state)
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


def _doctor_not_bookable_via_bot_offer(state: SessionState, memory: MemoryStore) -> ResponseEnvelope:
    """Ответ на «выбран конкретный врач, которого нет в системе онлайн-записи».

    Онлайн-запись через бот доступна только по филиалам в Самаре. Если врача
    нет в самарском каталоге (например, принимает только в Оренбурге — регион
    исключён из кэша через EXCLUDED_REGION_ROOTS), нельзя предлагать самарские
    филиалы вслепую: честно сообщаем об ограничении и предлагаем оператора.
    """
    reset_appointment_runtime_state(state)
    state.last_entities["_operator_offer_pending"] = True
    memory.set_pending(state, label="OTHER", missing_slots=["operator_offer_confirm"])
    return ResponseEnvelope(
        text=(
            "К сожалению, этого врача нет в системе онлайн-записи — через бот "
            "запись возможна только по филиалам в Самаре. Могу перевести на "
            "оператора, чтобы уточнить запись. Перевести на оператора?"
        ),
        attachments=[],
        handoff=False,
    )


def build_doctor_schedule_response(
    flow_label: str,
    evidence: Evidence,
    state: SessionState,
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    if flow_label != "DOCTOR_SCHEDULE":
        return None
    schedule_payload = evidence.get(ek.DOCTOR_SCHEDULE)
    if not isinstance(schedule_payload, dict):
        return None
    if str(schedule_payload.get("schedule_unavailable_reason") or "").strip() == "no_free_slots_2_weeks":
        return _no_free_slots_operator_offer(state, memory)
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
    address_payload = evidence.get(ek.ADDRESS)
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
    news_payload = evidence.get(ek.NEWS)
    if not isinstance(news_payload, dict):
        return None
    text = format_news_for_patient(news_payload, state.last_entities)
    return ResponseEnvelope(text=text, attachments=[], handoff=False)


def build_main_index_info_response(evidence: Evidence) -> ResponseEnvelope | None:
    payload = evidence.get(ek.MAIN_INDEX_INFO)
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
    memory: MemoryStore,
) -> ResponseEnvelope | None:
    if flow_label != "APPOINTMENT":
        return None
    schedule_payload = evidence.get(ek.DOCTOR_SCHEDULE)
    if not isinstance(schedule_payload, dict):
        return None
    if not (state.last_entities.get("doctor_name") or state.last_entities.get("doctor_id")):
        return None
    action = str(state.last_entities.get("appointment_action") or "").strip().lower()
    if action in {"cancel", "reschedule"}:
        return None
    if state.last_entities.get("date_from") or state.last_entities.get("date_hint"):
        return None
    if state.last_entities.get("time_from"):
        return None

    # Пациент хочет записаться, но свободных слотов нет на 2 недели — честный
    # ответ как в DOCTOR_SCHEDULE, без падения в format_*→«расписание не найдено».
    if str(schedule_payload.get("schedule_unavailable_reason") or "").strip() == "no_free_slots_2_weeks":
        return _no_free_slots_operator_offer(state, memory)

    # Назван конкретный врач, которого нет в самарском каталоге онлайн-записи
    # (doctor_lookup=unresolved из doctors_schedule_week). Не показываем
    # «расписание не найдено» и не предлагаем самарские филиалы вслепую —
    # честно сообщаем об ограничении «только по Самаре» и предлагаем оператора.
    if str(schedule_payload.get("doctor_lookup") or "").strip() == "unresolved":
        return _doctor_not_bookable_via_bot_offer(state, memory)

    hydrate_appointment_context_from_schedule(state, schedule_payload)
    activate_appointment_flow(state)
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
    activate_appointment_flow(state)
    action = str(entities.get("appointment_action") or "").strip().lower()
    if action == "cancel":
        patient_name = str(entities.get("patient_name") or "").strip()
        doctor_name = str(entities.get("doctor_name") or "").strip()
        service_name = str(entities.get("service_name") or entities.get("test_name") or "").strip()
        date_raw = str(entities.get("date_from") or entities.get("date_hint") or "").strip()
        date_text = _render_appointment_date_part(date_raw) if date_raw else ""
        time_text = str(entities.get("time_from") or "").strip()

        subject = doctor_name or service_name or "выбранной записи"
        header = "Перенос записи" if action == "reschedule" else "Отмена записи"
        details: list[str] = []
        if patient_name:
            details.append(f"пациент: {patient_name}")
        if subject:
            details.append(f"запись: {subject}")
        if date_text:
            details.append(f"дата: {date_text}")
        if time_text:
            details.append(f"время: {time_text}")

        reset_appointment_confirmation_flags(state)
        text_lines = [f"{header}: " + ", ".join(details) + "."] if details else [f"{header}: данные получены."]
        text_lines.append("Передаю заявку оператору для подтверждения и дальнейшего оформления.")
        return ResponseEnvelope(text="\n".join(text_lines), handoff=True)

    appointment_step = appointment_step_policy(entities)
    service = appointment_service_display(entities)
    city = str(entities.get("city") or "").strip()
    selection_mode = str(entities.get("appointment_selection_mode") or "").strip().lower()
    doctor_selected = bool(entities.get("doctor_name") or entities.get("doctor_id"))
    if doctor_selected:
        clear_appointment_selection_mode(state)
        selection_mode = ""

    if appointment_step == APPOINTMENT_STEP_BRANCH:
        # Назван конкретный врач, которого нет в самарском каталоге онлайн-записи
        # (например, «Записаться к <ФИО> на завтра»: дата задана → preview-ветка
        # пропущена → доходим сюда). Не предлагаем самарские филиалы вслепую —
        # честно сообщаем «только по Самаре» и предлагаем оператора.
        if doctor_selected:
            schedule_payload = evidence.get(ek.DOCTOR_SCHEDULE)
            if (
                isinstance(schedule_payload, dict)
                and str(schedule_payload.get("doctor_lookup") or "").strip() == "unresolved"
            ):
                return _doctor_not_bookable_via_bot_offer(state, memory)

        if selection_mode == "doctor" and not doctor_selected:
            doctors_info_payload = evidence.get(ek.DOCTORS_INFO)
            doctors_raw = doctors_info_payload.get("doctors") if isinstance(doctors_info_payload, dict) else None
            doctors = doctors_raw if isinstance(doctors_raw, list) else []
            if doctors:
                clear_appointment_branch_options(state)
                return ResponseEnvelope(
                    text=format_doctor_info_for_patient(doctors_info_payload, entities),  # type: ignore[arg-type]
                    handoff=False,
                )
            # Если список врачей не найден, мягко возвращаемся к выбору филиала.
            set_appointment_selection_mode(state, "branch")
            selection_mode = "branch"

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
                evidence.get(ek.ADDRESS),
                branches,
                city=city or None,
                limit=5,
            )
        set_appointment_branch_options(state, addresses)
        allow_doctor_option = not doctor_selected and bool(
            str(entities.get("service_name") or entities.get("test_name") or entities.get("specialty") or "").strip()
        )
        return ResponseEnvelope(
            text=appointment_text_branch_prompt(
                service,
                city,
                addresses,
                allow_doctor_option=allow_doctor_option,
            ),
            handoff=False,
        )

    if appointment_step == APPOINTMENT_STEP_DATETIME:
        clear_appointment_branch_options(state)
        if action in {"cancel", "reschedule"}:
            memory.set_pending(state, label="APPOINTMENT", missing_slots=["_any_of:date_from,time_from,date_hint"])
        price_rub = extract_price_rub(evidence.get(ek.PRICE))
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
        mark_appointment_confirm_pending(state)
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
        lambda: build_catalog_confirm_response(evidence),
        lambda: build_catalog_health_response(evidence),
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
        lambda: build_appointment_schedule_preview_response(flow_label, evidence, state, memory),
        lambda: build_appointment_step_response(flow_label, evidence, state, services, memory),
    )
    for build in builders:
        env = build()
        if env is not None:
            return env
    return None
