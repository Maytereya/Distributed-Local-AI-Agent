"""Таблица политик темы: все переклейки метки хода после решения NLU — в одном месте (ход (б), этап 3, шаг 1).

Решение владельца 20.10 (б) 7: собрать все переклейки метки в одну таблицу — рефакторинг без изменения ответов; таблица
даёт картину, где ещё решают правила и метки, а не LLM. Шаг 1 (STEP-2026-10-21-B-POLICY-TABLE): каждая переклейка роутера
идёт через `router._relabel(decision, <строка>)`, строка обязана быть здесь, и каждая сработавшая строка пишет себя в
телеметрию хода (`tu_relabel_src`: `<строка>:<было>><стало>`, только если метка сменилась). Условия пока остаются в коде
роутера, у своих функций (`site`); перенос условий в таблицу — следующие шаги, строка за строкой, по карточке.

Поля строки: `layer` — какой слой переклеивает (поток записи, guardrails после NLU, реестр тем, существующая запись, каталог,
понимание цены, свойство вопроса); `to` — итоговая метка (None — берётся из данных: реестр тем, висящий вопрос, справочник
врачей); `site` — функция роутера с условием; `why` — условие словами; `ref` — запись журнала (пусто — записи нет: правило
без следа бага, первое на проверку LLM в тени).

Архитектурный тест `tests/test_topic_policy_table_class.py`: `_copy_decision(..., label=...)` в роутере — красный (только
через `_relabel`); строка, которой нет в таблице, или строка таблицы без места в коде — красный.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import turn_telemetry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Relabel:
    id: str
    layer: str
    to: str | None
    site: str
    why: str
    ref: str = ""


_ROWS: tuple[Relabel, ...] = (
    # --- понимание цены и каталог (переклейки В PRICE — внутри `is_price_turn`, консилиум 20.10 §3) ---
    Relabel(
        "test_assist_catalog_hit_is_price", "price_understanding", "PRICE", "_relabel_test_assist_catalog_hit_as_price",
        "TEST_ASSIST, а реплика называет услугу каталога как вопрос о цене", "PLAN-2026-10-20-B §2 (43–58% ходов rule_test_assist)",
    ),
    Relabel(
        "understanding_list_is_price", "price_understanding", "PRICE", "_relabel_test_assist_catalog_hit_as_price",
        "TEST_ASSIST, а понимание хода нашло список услуг — это корзина цены", "PLAN-2026-10-20-B §2",
    ),
    Relabel(
        "flow_price_city_reply_override", "guardrail_post", "PRICE", "_apply_post_nlu_guardrails",
        "ответ городом на висящий вопрос цены — продолжение цены",
    ),
    # --- реестр тем ---
    Relabel(
        "topic_registry_override_from_other", "topic_registry", None, "_apply_topic_registry_override",
        "OTHER, слова реестра тем подтверждены по смыслу — метка темы из реестра", "BUG-2026-10-18-TOPIC-BY-LETTERS",
    ),
    Relabel(
        "topic_handoff", "topic_registry", "OTHER", "_maybe_topic_handoff",
        "тема реестра «только оператор» (выезд на дом и др.)", "BUG-2026-09-25-HOME-VISIT-AS-URGENT",
    ),
    # --- поток записи ---
    Relabel(
        "flow_pending_answer", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "идёт запись, реплика — ответ на её висящий вопрос",
    ),
    Relabel(
        "flow_datetime_appointment_override", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "идёт запись, реплика — дата или время",
    ),
    Relabel(
        "flow_appointment_override", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "идёт запись, NLU увёл в другую тему без явной смены темы",
    ),
    Relabel(
        "flow_appointment_guard_override", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "гард потока записи удержал запись",
    ),
    Relabel(
        "appointment_change_verb", "appointment_flow", "APPOINTMENT", "_confirm_appointment_change_verb",
        "глагол переноса/отмены записи при низкой уверенности NLU", "BUG-2026-10-07-CANCEL-LOW-CONF-CLARIFY",
    ),
    Relabel(
        "flow_datetime_appointment_prelock", "guardrail_post", "APPOINTMENT", "_apply_post_nlu_guardrails",
        "дата/время в реплике при начатой записи (prelock)",
    ),
    Relabel(
        "flow_schedule_to_appointment", "guardrail_post", "APPOINTMENT", "_apply_post_nlu_guardrails",
        "вопрос о расписании внутри записи — продолжение записи",
    ),
    Relabel(
        "walkin_named_bookable_service", "guardrail_post", "APPOINTMENT", "to_booking",
        "«без записи?» про услугу, на которую записывают, — запись",
    ),
    Relabel(
        "pending_slot_value", "pending", None, "_slot_value_keeps_topic",
        "значение слота висящего вопроса — тема висящего вопроса", "BUG-2026-10-19-PENDING-BUTTONS-CLASS5",
    ),
    # --- врачи ---
    Relabel(
        "bare_doctor_name", "guardrail_post", "DOCTOR_SCHEDULE", "_promote_bare_doctor_name",
        "голая фамилия врача из справочника", "BUG-2026-10-08-BARE-DOCTOR-SURNAME",
    ),
    Relabel(
        "verified_doctor_override", "guardrail_post", None, "_apply_post_nlu_guardrails",
        "врач подтверждён справочником — метка врача (`apply_verified_doctor_override`)",
    ),
    Relabel(
        "flow_doctor_info_to_schedule", "guardrail_post", "DOCTOR_SCHEDULE", "_apply_post_nlu_guardrails",
        "вопрос о враче с признаком расписания — расписание",
    ),
    # --- свойство вопроса: подготовка / адрес ---
    Relabel(
        "flow_price_followup_prepare", "guardrail_post", "PREPARE", "_apply_post_nlu_guardrails",
        "после цены — вопрос о подготовке к той же услуге",
    ),
    Relabel(
        "flow_prepare_followup_override", "guardrail_post", "PREPARE", "_apply_post_nlu_guardrails",
        "продолжение подготовки", "BUG-2026-10-08-PREPARE-ANESTHESIA-IN-QUERY",
    ),
    Relabel(
        "prepare_over_booking_llm", "question_facet", "PREPARE", "_resolve_nlu_decision_before_doctor_guard",
        "LLM при конфликте детекторов: подготовка, а не запись (`prepare_vs_booking_llm`)", "BUG-2026-10-18-PREPARE-UZI-AS-BOOKING",
    ),
    Relabel(
        "flow_price_followup_address", "guardrail_post", "ADDRESS", "_apply_post_nlu_guardrails",
        "после цены — вопрос «где сдать»",
    ),
    Relabel(
        "policy_nonbookable_walkin", "guardrail_post", "ADDRESS", "_apply_post_nlu_guardrails",
        "услуга без записи (в порядке очереди) — адреса",
    ),
    # --- существующая запись, каталог, справки ---
    Relabel(
        "existing_appointment_operator_handoff", "existing_appointment", "OTHER", "_existing_appointment_cancel_handoff",
        "отмена существующей записи — оператор", "BUG-2026-10-14-CANCEL-EXISTING-LOST",
    ),
    Relabel(
        "existing_appointment_operator_offer", "existing_appointment", "OTHER", "_maybe_offer_operator_for_existing_appointment",
        "вопрос о существующей записи — оффер оператора", "BUG-2026-09-25-CANCEL-OFFER-PREEMPTED",
    ),
    Relabel(
        "catalog_confirm_requested", "catalog", "OTHER", "_maybe_start_catalog_confirm",
        "услуга найдена не уверенно — спросить подтверждение каталога",
    ),
    Relabel(
        "readiness_reply", "catalog", "OTHER", "_maybe_readiness_reply",
        "вопрос о сроке готовности — ответ сроком", "STEP-2026-10-14-L14-3-CONTEXT-SHOWN",
    ),
    Relabel(
        "catalog_unavailable", "catalog", "OTHER", "_complete_route_after_doctor_guard",
        "справочник МИС недоступен — честный ответ и оператор", "STEP-2026-10-18-CATALOG-UNAVAILABLE-HANDOFF",
    ),
)
TABLE: dict[str, Relabel] = {row.id: row for row in _ROWS}


def note(policy: str, before: str, after: str) -> None:
    """Сработавшая строка — в телеметрию хода (`tu_relabel_src`), только если метка сменилась. На решение не влияет."""

    if policy not in TABLE:  # архитектурный тест не пускает такое в код; на проде ход не ломаем
        logger.warning("topic_policy_unknown_row", extra={"policy": policy})
    if before == after:
        return
    entry = f"{policy}:{before}>{after}"
    turn_telemetry.append("tu_relabel_src", entry)
    from . import turn_understanding  # понимание хода импортирует сервисы — не на уровне модуля

    turn_understanding.note_override(entry)
