"""Планировщик инструментальных шагов для patient-router.

Отвечает только за построение `Plan` по текущему `RouteDecision` и state,
без выполнения сервисных вызовов.
"""

from __future__ import annotations

from typing import Any

from .flow_policy import apply_pending_override
from .llm_mode_policy import RuntimeOptions
from .memory import MemoryStore
from .mess_types import Plan, PlanStep, RouteDecision, SessionState
from .policies import has_datetime_signal, is_compound_uzi_request, is_fixed_equipment_service, missing_slots
from .topic_registry import extract_topic_id_from_flags, get_topic as topic_registry_get_topic


def _unbookable_target_kind(user_text: str, decision_flags: set[str]) -> str:
    """Классифицирует ОБРОНЁННУЮ цель записи, чтобы отказ говорил по делу.

    Единый текст отказа всегда утверждал «этого врача нет в системе онлайн-записи»
    — и на представившегося по ФИО пациента (прод 11.07), и на названную УСЛУГУ
    («Могу ли я завтра пройти гинекологическое УЗИ», прод 18.07). Логика отказа
    верна (не выдумываем филиалы под неопознанную цель, `BUG-2026-06-01-01`),
    неверна была формулировка.

    :param user_text: реплика пациента этого хода
    :param decision_flags: флаги грундированного решения
    :return: ``doctor`` | ``service`` | ``patient_name`` | ``unknown``
    """

    from .flow_policy import looks_like_patient_fio
    from .service_phrase import extract_service_phrase

    text = str(user_text or "").strip()
    if "entity_dropped_doctor_like_service_name" in decision_flags:
        return "doctor"
    # Собственное имя пациента НЕ является целью записи: на «<ФИО>» отвечаем
    # вопросом о цели, а не «такого врача нет».
    if text and looks_like_patient_fio(text):
        return "patient_name"
    if text and extract_service_phrase(text):
        return "service"
    return "unknown"


def _build_other_plan_from_topic_registry(
    decision: RouteDecision,
    user_text: str,
    entities: dict[str, Any],
) -> list[PlanStep]:
    topic_id = extract_topic_id_from_flags(decision.flags)
    if not topic_id:
        return []
    topic = topic_registry_get_topic(topic_id)
    if not isinstance(topic, dict):
        return []
    route = topic.get("route")
    if not isinstance(route, dict):
        return []
    sources = route.get("sources")
    if not isinstance(sources, list):
        return []

    base_input = {"query": user_text, "entities": dict(entities)}
    for source in sources:
        if not isinstance(source, dict):
            continue
        kind = str(source.get("kind") or "").strip().lower()
        if kind == "meili":
            index = str(source.get("index") or "main_index").strip().lower()
            if index == "news":
                return [PlanStep(tool="news_info", input=base_input)]
            return [PlanStep(tool="main_index_info", input=base_input)]

    return []


def build_plan(
    decision: RouteDecision,
    state: SessionState,
    user_text: str,
    memory: MemoryStore,
    runtime_options: RuntimeOptions | None = None,
) -> Plan:
    pending = memory.get_pending(state)
    effective_label = apply_pending_override(decision, pending, user_text=user_text)

    entities = state.last_entities
    if runtime_options is not None:
        entities = {**entities, "__runtime_llm_mode": runtime_options.llm_mode}
    missing = missing_slots(effective_label, entities)

    # «во сколько/когда прийти сдать кровь» — это PREPARE без конкретного анализа.
    # Не уходим в clarify-петлю «к какому анализу нужна подготовка?», а планируем
    # test_prepare: он короткозамкнётся на адреса филиалов с графиком. Иначе
    # clarify-gate отвечает раньше, чем тул успевает отработать.
    if effective_label == "PREPARE" and missing:
        from .services.prepare import _is_lab_visit_timing_query

        if _is_lab_visit_timing_query(user_text):
            missing = []

    # Compound-УЗИ (BUG-2026-06-02-08): набор из >1 УЗИ-исследования («жтк + почки»,
    # follow-up «плюс поджелудочная» к активной УЗИ-записи). По решению владельца
    # такой набор честно уводим к оператору, а не подбираем одну услугу. Проверяем
    # ДО B′/A′-2 и missing-slots, чтобы выдать корректное сообщение про оператора,
    # а не «врача нет по Самаре» / generic clarify. cancel/reschedule не трогаем.
    if (
        effective_label == "APPOINTMENT"
        and str(entities.get("appointment_action") or "").strip().lower() not in {"cancel", "reschedule"}
        and is_compound_uzi_request(user_text, str(state.last_entities.get("service_name") or ""))
    ):
        state.last_entities["_appointment_compound_uzi"] = True
        memory.clear_pending(state)
        return Plan(label=effective_label, steps=[])

    # B′ + A′-2 (BUG-2026-06-01-01): если ГРУНДИРОВАННОЕ решение этого хода
    # обронило цель записи как неверифицированную (напр. ФИО, ошибочно принятое
    # за услугу) и валидного врача/специальности нет — пользователь назвал цель,
    # которую забронировать нельзя. Честно отказываем (запись только по Самаре →
    # оператор) ДО missing-slots short-circuit, иначе вместо честного отказа
    # уходим в generic clarify-петлю «к кому/на что записать?». Источник цели на
    # свежем ходу — ГРУНДИРОВАННЫЙ decision.entities, НЕ stale state: оставшаяся
    # от прошлой темы specialty не должна молча подменять цель (A′-2). В активном
    # flow читаем state — известный врач/специальность должны пережить ход.
    if effective_label == "APPOINTMENT" and "entity_dropped_unverified_service_name" in set(decision.flags):
        appt_action = str(entities.get("appointment_action") or "").strip().lower()
        flow_active = bool(state.last_entities.get("appointment_flow_active"))
        target_src = entities if flow_active else decision.entities
        has_grounded_target = bool(
            target_src.get("doctor_id")
            or target_src.get("doctor_name")
            or str(target_src.get("specialty") or "").strip()
        )
        # Продолжение ТОЙ ЖЕ записи (eval CRIT_APPT_KIM_LOOP_001): врач назван на
        # прошлом ходу, но flow не стал active (ход 1 «Ким» упёрся в «нет слотов»
        # → operator-offer, COLLECTING не наступил). На ходу «на завтра на 9:00»
        # target_src=decision.entities без врача → ложный unbookable «врача нет в
        # системе». Конкретный РЕЗОЛВНУТЫЙ врач из непосредственно предыдущего
        # APPOINTMENT-хода — валидная цель (A′-2 запрещает подмену только stale
        # SPECIALTY/service, не конкретным doctor_name/doctor_id).
        # Гейт по сигналу даты/времени: рескью — ТОЛЬКО для продолжения записи
        # слотом/датой («на завтра на 9:00»), где цель = уже известный врач. Если
        # пациент назвал новую услугу/цель, а не время (has_datetime_signal=False),
        # рескью НЕ срабатывает — честный unbookable вместо молчаливого удержания
        # старого врача (самопроверка аудита: не удерживаем контекст, когда ход —
        # не продолжение). Называние НОВОГО врача сюда не попадает: оно идёт через
        # doctor-резолв (флаг doctor_like), а не unverified_service.
        prev_label = str(state.last_entities.get("_last_label") or "").strip().upper()
        appointment_thread_doctor = (
            prev_label == "APPOINTMENT"
            and bool(state.last_entities.get("doctor_id") or state.last_entities.get("doctor_name"))
            and has_datetime_signal(user_text)
        )
        if (
            appt_action not in {"cancel", "reschedule"}
            and not has_grounded_target
            and not appointment_thread_doctor
        ):
            # П5: тип обронённой цели решает ТЕКСТ отказа. Раньше он всегда был
            # про врача («этого врача нет в системе онлайн-записи») — и пациент,
            # который просто представился по ФИО или назвал УСЛУГУ («гинекологическое
            # УЗИ»), получал ответ не по делу (прод 11.07 и 18.07).
            state.last_entities["_appointment_unbookable_target"] = _unbookable_target_kind(
                user_text, set(decision.flags)
            )
            memory.clear_pending(state)
            return Plan(label=effective_label, steps=[])

    # Fixed-equipment (флюорограф/маммограф на Ленина 5): у клиники нет
    # приёмного врача-радиолога, запись ведёт ТОЛЬКО регистратура. Перехват
    # уже стоял в doctors_schedule_week (расписание) и addresses (адрес), но НЕ
    # в APPOINTMENT-пути → прод #668: «Записаться на флюорографию» проходил
    # полный booking. Перехватываем ДО сбора слотов; cancel/reschedule не
    # трогаем (отмена такой записи — обычный путь). Ответ — response_builder.
    if effective_label == "APPOINTMENT":
        # action/service — из свежего decision.entities с фолбэком на контекст
        # (entities == state.last_entities): в активном флоу цель уже в state,
        # на первом ходу — в decision.
        fe_action = str(
            decision.entities.get("appointment_action") or entities.get("appointment_action") or ""
        ).strip().lower()
        if fe_action not in {"cancel", "reschedule"}:
            fe_probe = str(
                decision.entities.get("service_name")
                or decision.entities.get("test_name")
                or entities.get("service_name")
                or entities.get("test_name")
                or ""
            ).strip() or user_text
            if is_fixed_equipment_service(fe_probe):
                state.last_entities["_appointment_fixed_equipment"] = True
                memory.clear_pending(state)
                return Plan(label=effective_label, steps=[])

    if missing:
        memory.set_pending(state, label=effective_label, missing_slots=missing)
        return Plan(label=effective_label, steps=[])

    memory.clear_pending(state)

    label = effective_label
    steps: list[PlanStep] = []

    if label == "OTHER":
        topic_steps = _build_other_plan_from_topic_registry(decision, user_text, entities)
        if topic_steps:
            return Plan(label=label, steps=topic_steps)

    if "doc_request_main_index" in decision.flags or "doc_request_handoff" in decision.flags:
        steps.append(PlanStep(tool="main_index_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "TEST_RESULT":
        steps.append(PlanStep(tool="test_result_status", input={"query": user_text, "entities": dict(entities)}, auth="none"))
        return Plan(label=label, steps=steps)

    if label == "TEST_ASSIST":
        steps.append(PlanStep(tool="test_assist", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_SCHEDULE":
        # Демоция: «Расписание уролог» (без конкретной фамилии) — это
        # запрос списка специалистов, не загрузки 8 расписаний скопом.
        # Прежде такие запросы триггерили `_schedule_by_specialty`,
        # который под нагрузкой Nayka API отвечал 70+ секунд (а часто
        # вообще не успевал). При запросе только специальности
        # переключаем на DOCTOR_INFO — мгновенная выдача списка из
        # JSONL-кэша. После выбора конкретного врача расписание
        # запрашивается прицельно за разумное время.
        has_specific_doctor = any(
            str(entities.get(key) or "").strip()
            for key in ("doctor_id", "doctor_name", "doctor", "fio", "last_name", "doctor_last_name")
        )
        if not has_specific_doctor:
            steps.append(PlanStep(tool="doctors_info", input={"query": user_text, "entities": dict(entities)}))
            return Plan(label="DOCTOR_INFO", steps=steps)
        steps.append(PlanStep(tool="doctors_schedule_week", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "DOCTOR_INFO":
        steps.append(PlanStep(tool="doctors_info", input={"query": user_text, "entities": dict(entities)}))
        if entities.get("doctor_id") or entities.get("doctor_name"):
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}, required=False))
        return Plan(label=label, steps=steps)

    if label == "APPOINTMENT":
        action = str(entities.get("appointment_action") or "").strip().lower()
        if action == "cancel":
            return Plan(label=label, steps=steps)
        # Дропнутая-цель → честный отказ обрабатывается выше, ДО missing-slots
        # (см. ранний guard «B′ + A′-2»), поэтому сюда доходит только запись с
        # валидной грундированной целью или активный flow.
        flow_active = bool(state.last_entities.get("appointment_flow_active"))
        selection_mode = str(state.last_entities.get("appointment_selection_mode") or "").strip().lower()
        if entities.get("doctor_id") or entities.get("doctor_name"):
            has_cached_windows = bool(entities.get("appointment_windows"))
            has_selected_datetime = bool(
                (entities.get("date_from") or entities.get("date_hint"))
                and (entities.get("time_from") or entities.get("time_flexible"))
            )
            if not has_cached_windows and not has_selected_datetime:
                steps.append(
                    PlanStep(
                        tool="doctors_schedule_week",
                        input={"query": user_text, "entities": dict(entities)},
                        required=not flow_active,
                    )
                )
        else:
            if selection_mode == "doctor":
                steps.append(
                    PlanStep(
                        tool="doctors_info",
                        input={"query": user_text, "entities": dict(entities)},
                        required=not flow_active,
                    )
                )
                address_entities = dict(entities)
                address_entities["__appointment_mode"] = True
                steps.append(
                    PlanStep(
                        tool="address_info",
                        input={"query": user_text, "entities": address_entities},
                        required=False,
                    )
                )
            else:
                address_entities = dict(entities)
                address_entities["__appointment_mode"] = True
                steps.append(
                    PlanStep(
                        tool="address_info",
                        input={"query": user_text, "entities": address_entities},
                        required=not flow_active,
                    )
                )
                steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}, required=False))
        return Plan(label=label, steps=steps)

    if label == "PRICE":
        service_known = bool(str(entities.get("service_name") or entities.get("test_name") or "").strip())
        doctor_known = bool(entities.get("doctor_id") or entities.get("doctor_name"))
        if service_known and not doctor_known:
            steps.append(PlanStep(tool="service_bundle_info", input={"query": user_text, "entities": dict(entities)}))
        else:
            steps.append(PlanStep(tool="price_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "ADDRESS":
        steps.append(PlanStep(tool="address_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "PREPARE":
        steps.append(PlanStep(tool="test_prepare", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    if label == "NEWS":
        steps.append(PlanStep(tool="news_info", input={"query": user_text, "entities": dict(entities)}))
        return Plan(label=label, steps=steps)

    return Plan(label=label, steps=[])
