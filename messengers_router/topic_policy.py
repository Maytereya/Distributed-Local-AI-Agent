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
через `_relabel`); строка, которой нет в таблице, или строка таблицы без места в коде — красный. Переклейки вне роутера
(`flow_policy`, `planner`) — тем же путём: `apply` (решение) и `label_for` (метка плана), со следом и строкой таблицы.

Рамка кнопки (Н-2, STEP-2026-10-30-FRAME-POLICY, консилиум 30.10): у строки обязательное поле `in_frame` — без умолчания,
каждая строка решена явно. `KEEPS` — несрочная переклейка (буквы, контекст прошлого хода): под `frame_policy = on` она не
меняет решение LLM о рамке кнопки этого хода (`button_flow.turn_frame`) — не уводит из темы, которую держит кнопка (ответ на
её вопрос, рамка листа), и не возвращает в тему кнопки реплику, которую LLM признала другой темой (Д3: «Как подготовиться» →
«а сколько стоит ТТГ?» → «другая тема» → PRICE → по буквам снова PREPARE). Такая переклейка не применяется, след —
`tu_relabel_frame_kept`. `OVERRIDES` — рамка её не держит: перевод на оператора, существующая и идущая запись (путь к
необратимому действию), уточнение вместо чужой цены, решение LLM или каталога МИС по словам того же хода, выбор
инструмента. Срочное,
самоповреждение и медвопрос отвечены раньше таблицы (`orchestrator.early_guards`, правила безопасности NLU). Проверка —
одна, `frame_kept`; места, где до переклейки идут вызов LLM или правка состояния, спрашивают её заранее тем же вызовом.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import turn_telemetry
from .runtime_config import config as _cfg

if TYPE_CHECKING:
    from .mess_types import RouteDecision

logger = logging.getLogger(__name__)


KEEPS = "keeps"  # рамка кнопки держит тему: переклейка по буквам решение LLM о рамке не меняет
OVERRIDES = "overrides"  # рамка не держит: перевод, запись, уточнение вместо чужой цены, решение LLM, выбор инструмента


@dataclass(frozen=True)
class Relabel:
    id: str
    layer: str
    to: str | None
    site: str
    why: str
    ref: str = ""
    in_frame: str = field(kw_only=True)


_ROWS: tuple[Relabel, ...] = (
    # --- понимание цены и каталог (переклейки В PRICE — внутри `is_price_turn`, консилиум 20.10 §3) ---
    Relabel(
        "test_assist_catalog_hit_is_price", "price_understanding", "PRICE", "_relabel_test_assist_catalog_hit_as_price",
        "TEST_ASSIST, а реплика называет услугу каталога как вопрос о цене", "PLAN-2026-10-20-B §2 (43–58% ходов rule_test_assist)",
        in_frame=KEEPS,
    ),
    Relabel(
        "understanding_list_is_price", "price_understanding", "PRICE", "_relabel_test_assist_catalog_hit_as_price",
        "TEST_ASSIST, а понимание хода нашло список услуг — это корзина цены", "PLAN-2026-10-20-B §2",
        in_frame=OVERRIDES,  # список из разметки хода — решение LLM, не буквы: цена и в теме кнопки (STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM)
    ),
    Relabel(
        "p1_where_when", "question_facet", "ADDRESS", "_relabel_test_assist_catalog_hit_as_price",
        "ход `topic_by_llm`: тему TEST_ASSIST решила LLM, услуга найдена, свойство вопроса по LLM — «где / когда сдать»",
        "STEP-2026-10-21-B-P1-FACET",
        in_frame=KEEPS,
    ),
    Relabel(
        "p1_facet_none_offer", "question_facet", "OTHER", "_relabel_test_assist_catalog_hit_as_price",
        "ход `topic_by_llm`: тему TEST_ASSIST решила LLM, услуга найдена, свойство вопроса по LLM — «ни одно» → честный оффер"
        " (`p1_facet_none_offer = on`)",
        "BUG-2026-10-22-P1-FACET-NONE-AS-PRICE",
        in_frame=KEEPS,
    ),
    Relabel(
        "p1_facet_fail_offer", "question_facet", "OTHER", "_relabel_test_assist_catalog_hit_as_price",
        "ход `topic_by_llm`: тему TEST_ASSIST решила LLM, услуга найдена, свойство вопроса не решено (сбой LLM) → честный оффер,"
        " а не цена «как было» (`llm_fail_offer` с узлом `facet`)",
        "BUG-2026-10-27-LLM-QUEUE-GIVES-UP",
        in_frame=KEEPS,
    ),
    Relabel(
        "turn_facet_prepare", "question_facet", "PREPARE", "_route_by_turn_facet",
        "ход цены, тип вопроса по LLM — подготовка (`facet_by_llm = on`): памятка, а не шаблон цены",
        "STEP-2026-10-30-FACET-READERS",
        in_frame=KEEPS,
    ),
    Relabel(
        "turn_facet_where", "question_facet", "ADDRESS", "_route_by_turn_facet",
        "ход цены, тип вопроса по LLM — «где / когда» или «нужна ли запись» (`facet_by_llm = on`): адреса, как вариант П1 `where_when`",
        "STEP-2026-10-30-FACET-READERS",
        in_frame=KEEPS,
    ),
    Relabel(
        "turn_facet_none_offer", "question_facet", "OTHER", "_route_by_turn_facet",
        "ход цены, тип вопроса по LLM — «ни одно» (`facet_by_llm = on`): данные клиники или честный оффер, как у П1",
        "STEP-2026-10-30-FACET-READERS",
        in_frame=KEEPS,
    ),
    Relabel(
        "flow_price_city_reply_override", "guardrail_post", "PRICE", "_apply_post_nlu_guardrails",
        "ответ городом на висящий вопрос цены — продолжение цены",
        in_frame=KEEPS,
    ),
    # --- реестр тем ---
    Relabel(
        "topic_registry_override_from_other", "topic_registry", None, "_apply_topic_registry_override",
        "OTHER, слова реестра тем подтверждены по смыслу — метка темы из реестра", "BUG-2026-10-18-TOPIC-BY-LETTERS",
        in_frame=KEEPS,
    ),
    Relabel(
        "topic_handoff", "topic_registry", "OTHER", "_maybe_topic_handoff",
        "тема реестра «только оператор» (выезд на дом и др.)", "BUG-2026-09-25-HOME-VISIT-AS-URGENT",
        in_frame=OVERRIDES,  # перевод на оператора
    ),
    # --- тема по смыслу LLM ---
    Relabel(
        "insurance_topic", "topic_llm", "OTHER", "_maybe_insurance_topic",
        "реплика об обслуживании по ДМС / ОМС по выбору LLM (`insurance_topic = on`): памятка ДМС или «ответит оператор» — с оффером",
        "BUG-2026-10-28-K26-DMS-OMS-TOPIC",
        in_frame=OVERRIDES,  # ДМС / ОМС по выбору LLM — не платная цена в теме кнопки (BUG-2026-10-27-OMS-FOLLOWUP-PRICED)
    ),
    # --- поток записи ---
    Relabel(
        "flow_pending_answer", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "идёт запись, реплика — ответ на её висящий вопрос",
        in_frame=OVERRIDES,  # идёт запись — путь к необратимому действию
    ),
    Relabel(
        "flow_datetime_appointment_override", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "идёт запись, реплика — дата или время",
        in_frame=OVERRIDES,  # идёт запись
    ),
    Relabel(
        "flow_appointment_override", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "идёт запись, NLU увёл в другую тему без явной смены темы",
        in_frame=OVERRIDES,  # идёт запись
    ),
    Relabel(
        "flow_appointment_guard_override", "appointment_flow", "APPOINTMENT", "_apply_appointment_continuity_overrides",
        "гард потока записи удержал запись",
        in_frame=OVERRIDES,  # идёт запись
    ),
    Relabel(
        "appointment_change_verb", "appointment_flow", "APPOINTMENT", "_confirm_appointment_change_verb",
        "глагол переноса/отмены записи при низкой уверенности NLU", "BUG-2026-10-07-CANCEL-LOW-CONF-CLARIFY",
        in_frame=OVERRIDES,  # перенос / отмена существующей записи
    ),
    Relabel(
        "flow_datetime_appointment_prelock", "guardrail_post", "APPOINTMENT", "_apply_post_nlu_guardrails",
        "дата/время в реплике при начатой записи (prelock)",
        in_frame=OVERRIDES,  # идёт запись
    ),
    Relabel(
        "flow_schedule_to_appointment", "guardrail_post", "APPOINTMENT", "_apply_post_nlu_guardrails",
        "вопрос о расписании внутри записи — продолжение записи",
        in_frame=KEEPS,
    ),
    Relabel(
        "walkin_named_bookable_service", "guardrail_post", "APPOINTMENT", "to_booking",
        "«без записи?» про услугу, на которую записывают, — запись",
        in_frame=OVERRIDES,  # нужна ли запись, решает каталог МИС — так рамка листа «Нужно ли записываться» и задумана
    ),
    Relabel(
        "pending_slot_value", "pending", None, "_slot_value_keeps_topic",
        "значение слота висящего вопроса — тема висящего вопроса", "BUG-2026-10-19-PENDING-BUTTONS-CLASS5",
        in_frame=KEEPS,
    ),
    # --- врачи ---
    Relabel(
        "bare_doctor_name", "guardrail_post", "DOCTOR_SCHEDULE", "_promote_bare_doctor_name",
        "голая фамилия врача из справочника", "BUG-2026-10-08-BARE-DOCTOR-SURNAME",
        in_frame=KEEPS,
    ),
    Relabel(
        "verified_doctor_override", "guardrail_post", None, "_apply_post_nlu_guardrails",
        "врач подтверждён справочником — метка врача (`apply_verified_doctor_override`)",
        in_frame=KEEPS,
    ),
    Relabel(
        "flow_doctor_info_to_schedule", "guardrail_post", "DOCTOR_SCHEDULE", "_apply_post_nlu_guardrails",
        "вопрос о враче с признаком расписания — расписание",
        in_frame=KEEPS,
    ),
    # --- свойство вопроса: подготовка / адрес ---
    Relabel(
        "flow_price_followup_prepare", "guardrail_post", "PREPARE", "_apply_post_nlu_guardrails",
        "после цены — вопрос о подготовке к той же услуге",
        in_frame=KEEPS,
    ),
    Relabel(
        "flow_prepare_followup_override", "guardrail_post", "PREPARE", "_apply_post_nlu_guardrails",
        "продолжение подготовки", "BUG-2026-10-08-PREPARE-ANESTHESIA-IN-QUERY",
        in_frame=KEEPS,
    ),
    Relabel(
        "prepare_over_booking_llm", "question_facet", "PREPARE", "_resolve_nlu_decision_before_doctor_guard",
        "LLM при конфликте детекторов: подготовка, а не запись (`prepare_or_booking`)", "BUG-2026-10-18-PREPARE-UZI-AS-BOOKING",
        in_frame=OVERRIDES,  # решение LLM по словам этого хода, не буквы
    ),
    Relabel(
        "flow_price_followup_address", "guardrail_post", "ADDRESS", "_apply_post_nlu_guardrails",
        "после цены — вопрос «где сдать»",
        in_frame=KEEPS,
    ),
    Relabel(
        "policy_nonbookable_walkin", "guardrail_post", "ADDRESS", "_apply_post_nlu_guardrails",
        "услуга без записи (в порядке очереди) — адреса",
        in_frame=KEEPS,
    ),
    # --- существующая запись, каталог, справки ---
    Relabel(
        "existing_appointment_operator_handoff", "existing_appointment", "OTHER", "_existing_appointment_cancel_handoff",
        "отмена существующей записи — оператор", "BUG-2026-10-14-CANCEL-EXISTING-LOST",
        in_frame=OVERRIDES,  # существующая запись — оператор
    ),
    Relabel(
        "existing_appointment_operator_offer", "existing_appointment", "OTHER", "_maybe_offer_operator_for_existing_appointment",
        "вопрос о существующей записи — оффер оператора", "BUG-2026-09-25-CANCEL-OFFER-PREEMPTED",
        in_frame=OVERRIDES,  # существующая запись — оффер оператора
    ),
    Relabel(
        "catalog_confirm_requested", "catalog", "OTHER", "_maybe_start_catalog_confirm",
        "услуга найдена не уверенно — спросить подтверждение каталога",
        in_frame=OVERRIDES,  # уточнение вместо чужой цены (принцип 1): шаг ответа в той же теме, не смена темы
    ),
    Relabel(
        "readiness_reply", "catalog", "OTHER", "_maybe_readiness_reply",
        "вопрос о сроке готовности — ответ сроком", "STEP-2026-10-14-L14-3-CONTEXT-SHOWN",
        in_frame=OVERRIDES,  # тип вопроса того же ценового хода — срок из МИС, не цена
    ),
    Relabel(
        "catalog_unavailable", "catalog", "OTHER", "_complete_route_after_doctor_guard",
        "справочник МИС недоступен — честный ответ и оператор", "STEP-2026-10-18-CATALOG-UNAVAILABLE-HANDOFF",
        in_frame=OVERRIDES,  # справочник недоступен — перевод
    ),
    # --- вне роутера: `flow_policy`, планировщик (`apply` / `label_for`; до 30.10 — мимо таблицы, приложение A п. 20.5, 24a, 24e) ---
    Relabel(
        "context_action_overwrite_doctor", "context_action", "DOCTOR_SCHEDULE", "_apply_context_action",
        "смена врача (`overwrite_doctor`) при OTHER, а прошлый ход — расписание или в реплике признак расписания",
        in_frame=KEEPS,
    ),
    Relabel(
        "plan_pending_override", "pending", None, "build_plan",
        "планировщик держит тему висящего вопроса по форме реплики (город, название услуги, ответ записи, ФИО)"
        " (`flow_policy.apply_pending_override`)",
        in_frame=KEEPS,
    ),
    Relabel(
        "plan_doctor_list", "planner", "DOCTOR_INFO", "build_plan",
        "расписание без конкретного врача — список врачей (8 расписаний скопом шли 70+ с)",
        in_frame=OVERRIDES,  # выбор инструмента по данным хода, не смена темы
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


def frame_policy_on() -> bool:
    """`[MESSENGER_ROUTER] frame_policy = on`: рамку кнопки проверяет таблица (`in_frame`). off — как было."""

    return str(getattr(_cfg, "MR_FRAME_POLICY", "off") or "").strip().lower() == "on"


def frame_kept(policy: str, before: str, after: str) -> bool:
    """Строка `KEEPS` меняет решение LLM о рамке кнопки этого хода — не применять (след `tu_relabel_frame_kept`).

    Рамка держит тему (`held`): переклейка уводит из её темы. LLM вывела реплику из рамки (`held = False`, «другая тема»):
    переклейка возвращает в тему рамки. Строка `OVERRIDES`, рамки нет, метка не меняется, off — применять, как было.
    """

    if before == after or not frame_policy_on():
        return False
    row = TABLE.get(policy)
    if row is None or row.in_frame != KEEPS:
        return False
    from . import button_flow  # рамка хода — у кнопки; не на уровне модуля: таблицу импортируют `flow_policy` и планировщик

    frame = button_flow.turn_frame()
    if frame is None:
        return False
    if frame.held and not (before == frame.topic and after != frame.topic):
        return False
    if not frame.held and not (before != frame.topic and after == frame.topic):
        return False
    turn_telemetry.append("tu_relabel_frame_kept", f"{policy}:{before}>{after}")
    return True


def apply(policy: str, before: RouteDecision, after: RouteDecision) -> RouteDecision:
    """Переклейка строкой таблицы: `after` — со следом (`note`), или `before`, если рамка кнопки её держит (`frame_kept`)."""

    if frame_kept(policy, before.label, after.label):
        return before
    note(policy, before.label, after.label)
    return after


def label_for(policy: str, before: str, after: str) -> str:
    """То же для метки плана (планировщик переклеивает метку, а не решение)."""

    if frame_kept(policy, before, after):
        return before
    note(policy, before, after)
    return after
