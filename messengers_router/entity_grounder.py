"""Entity grounding для NLU-решений роутера.

Задача модуля:
1) Ограничить, какие entities вообще можно принять на текущем шаге.
2) Подтверждать критичные сущности по данным/эвристикам (doctor/city/service).
3) Возвращать только "безопасные" entities для merge в session state.
Ответственность модуля: не допускать попадание в state неподтвержденных сущностей.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from . import address_service, booking_slots, llm_runtime, turn_understanding, walkin
from .city import looks_like_address, match_city
from .flow_policy import looks_like_patient_fio
from .mess_types import RouteDecision, SessionState
from .policies import (
    extract_service_phrase,
    extract_specialty,
    has_datetime_signal,
    is_fixed_equipment_service,
    service_name_conflicts_with_doctor,
)
from .russian_nlu import normalize_ru
from .services import Services, resolve_price_service_name_from_catalog
from .services._booking_service_select import judge_booking_service
from .services._price_select_llm import names_service
from .services._common import _normalise_catalog_text

_CONTROL_KEYS = {
    "secondary_intents",
    "secondary_intent_from_queue",
    "appointment_action",
    "result_action",
    "query_terms",
    "_catalog_doctor_candidate",
    "_catalog_doctor_query",
    "_catalog_service_candidate",
    "_catalog_service_query",
    # NEWS promo follow-up (rule_news_promo_followup): выбор акции из списка
    # номером/названием + прокинутый контекст показанных акций для news_info.
    # promo_query_for — маркер «выдано для этого хода» (анти-протухание).
    "promo_query",
    "promo_query_for",
    "promo_pick_index",
    "_promo_context",
}

_GENERIC_SERVICE_FALLBACK_RE = re.compile(
    r"\b(запис\w*|врач\w*|доктор\w*|специалист\w*|услуг\w*|хочу|нужно|надо|можно)\b",
    re.I,
)
_SERVICE_ANCHOR_HINT_RE = re.compile(
    r"\b(узи|экг|холтер|мрт|кт|фгдс|фкс|эндоскоп|гастроскоп|колоноскоп|анализ|биопс|рентген|флюорограф)\b",
    re.I,
)
_DOCTOR_LIKE_REQUEST_RE = re.compile(r"\bк\s+[А-Яа-яЁёA-Za-z\-]{3,}\b")
_DOCTOR_NOISE_TOKENS = {
    "хочу",
    "нужно",
    "надо",
    "можно",
    "запись",
    "записаться",
    "прием",
    "приём",
    "подскажите",
    "скажите",
    "когда",
    "где",
    "да",
    "нет",
}

_LABEL_ENTITY_WHITELIST: dict[str, set[str]] = {
    "APPOINTMENT": {
        "doctor_id",
        "doctor_name",
        "specialty",
        "service_name",
        "city",
        "branch_id",
        "branch_name",
        "patient_name",
        "date_from",
        "date_to",
        "time_from",
        "time_to",
        "date_hint",
    },
    "DOCTOR_SCHEDULE": {"doctor_id", "doctor_name", "specialty", "branch_name", "city"},
    # service_name: процедурная конкретика («УЗИ органов мошонки») — без неё
    # doctors_info не может отфильтровать врачей по процедуре (кейс 08.07:
    # правило передавало услугу, граундер молча дропал → пациент получал ВСЕХ
    # УЗИ-врачей, включая не делающих процедуру).
    "DOCTOR_INFO": {"doctor_id", "doctor_name", "specialty", "service_name"},
    "PRICE": {"service_name", "city", "branch_id", "branch_name", "doctor_name", "doctor_id"},
    "ADDRESS": {"city", "branch_id", "branch_name", "service_name"},
    "TEST_RESULT": {"surname", "year", "filial", "number", "lang", "result_action", "order_id"},
    "TEST_ASSIST": {"test_name", "test_goal", "service_name", "city", "branch_name", "branch_id"},
    "PREPARE": {"test_name", "service_name"},
    "NEWS": set(),
    "OTHER": set(),
}


def _keys_from_missing(missing: list[Any]) -> set[str]:
    out: set[str] = set()
    for item in missing:
        if not isinstance(item, str):
            continue
        if item.startswith("_any_of:"):
            tail = item.split(":", 1)[1]
            for key in tail.split(","):
                k = key.strip()
                if k:
                    out.add(k)
            continue
        out.add(item)
    return out


def _pending_missing(pending: dict[str, Any] | None) -> list[str]:
    if not isinstance(pending, dict):
        return []
    raw = pending.get("missing")
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, str)]


def _is_allowed_key(key: str, label: str, pending: dict[str, Any] | None) -> bool:
    if key == "patient_name":
        missing = _pending_missing(pending)
        need = _keys_from_missing(missing)
        if (
            isinstance(pending, dict)
            and pending.get("label") == "APPOINTMENT"
            and "patient_name" in need
        ):
            return True

    if key in _CONTROL_KEYS:
        return True
    base = _LABEL_ENTITY_WHITELIST.get(label, set())
    if key not in base:
        return False
    missing = _pending_missing(pending)
    if not missing:
        return True
    need = _keys_from_missing(missing)
    if not need:
        return True
    # В pending-режиме принимаем в первую очередь слоты, которые реально ждем.
    if key in need:
        return True
    # Разрешаем city даже если формально не ждём, чтобы пользователь мог исправить город.
    if key == "city":
        return True
    if label == "PRICE" and key in {"doctor_name", "doctor_id"}:
        return True
    return False


@dataclass
class GroundingResult:
    entities: dict[str, Any] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)


def _sanitize_raw_doctor_name(raw: str, flags: set[str], entities: dict[str, Any]) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    norm = " ".join(normalize_ru(value).split())
    tokens = [t for t in norm.split(" ") if t]
    if len(tokens) == 1 and tokens[0] in _DOCTOR_NOISE_TOKENS:
        entities.pop("doctor_name", None)
        flags.add("doctor_name_unverified")
        return ""
    return value


def _is_doctor_like_service_collision(
    *,
    label: str,
    user_text: str,
    phrase: str,
    decision_flags: set[str],
) -> bool:
    """
    Отсекает ложный service_name в APPOINTMENT, когда пользователь просит
    запись "к <фамилии>", а врача не удалось подтвердить.
    """
    if label != "APPOINTMENT":
        return False
    has_doctor_like_request = bool(_DOCTOR_LIKE_REQUEST_RE.search(user_text or ""))
    if "doctor_name_unverified" not in decision_flags and not has_doctor_like_request:
        return False
    if _SERVICE_ANCHOR_HINT_RE.search(user_text or ""):
        return False
    if _SERVICE_ANCHOR_HINT_RE.search(phrase or ""):
        return False
    # Для многословных и явно процедурных формулировок риск ниже.
    if len((phrase or "").split()) > 1:
        return False
    return True


# Названия услуг на выбор пациенту, когда цель записи не решена (`judge_booking_service` → `choose`): служебный ключ
# решения, ответ задаёт роутер (`router._maybe_ask_booking_service_choice`), в состояние он не попадает.
BOOKING_SERVICE_OPTIONS_KEY = "_booking_service_options"


def _booking_service_picked() -> str | None:
    """Название услуги записи, выбранное одной функцией ответа на висящий вопрос этого хода (класс 5), или None."""

    from . import expects, pending_answer

    return pending_answer.picked(expects.Question.BOOKING_SERVICE)
# Метки, где услугу из реплики судит выбор LLM (`judge_booking_service`), а не догадка правил по буквам: запись и «где
# сделать». Цена судит сама (`_price_select_llm`); TEST_ASSIST догадку по буквам не брал (свип 18.10).
_LLM_JUDGED_SERVICE_LABELS = frozenset({"APPOINTMENT", "ADDRESS"})
# Названо слишком общее: названия — примеры, а не полный список (`router._maybe_ask_booking_service_choice`).
BOOKING_SERVICE_CLARIFY_KEY = "_booking_service_clarify"


def service_dropped_by_llm(flags: Any) -> bool:
    """Услугу этого хода сняла LLM выбора строки прайса (`_apply_service_verdict`), а не правила: флаг снятия
    `entity_dropped_unverified_service_name` и внятный статус выбора `booking_service_<статус>`. «Недоступна» — не
    решение; «сильное» совпадение — данные МИС, не LLM. Одно определение для читателей (инструмент адресов, подавитель
    A′-1; решение владельца 19.10, п. 10)."""

    flags = set(flags or ())
    if "entity_dropped_unverified_service_name" not in flags:
        return False
    return any(f.startswith("booking_service_") and f not in ("booking_service_unavailable", "booking_service_strong") for f in flags)


def _apply_service_verdict(
    *, label: str, verdict: Any, canonical: str, phrase: str, out: dict[str, Any], flags: set[str]
) -> str | None:
    """Исход выбора LLM (`judge_booking_service`) → значение услуги; None — решено иначе (снята, на выбор, на подтверждение).

    Запись: несколько — названия на выбор; одно название без LLM — «Похоже, вы имели в виду …?»; «слишком общее» —
    уточнить; «не нашла» — честный отказ гарда B′. «Где сделать …» (ADDRESS): всё, кроме одной услуги, снимает услугу —
    ответ филиалами без фильтра по чужой строке (адреса «Mg+ (магний)» на «магнитотерапию» — дезинформация местом); LLM
    недоступна — как было, по правилам (решение владельца 18.10 «названия на выбор» — о записи).
    """

    if verdict.action == "keep":
        return canonical
    if verdict.action == "replace":
        return verdict.name
    if label == "ADDRESS":
        if verdict.status == "unavailable":
            return canonical
        if verdict.action == "choose" and verdict.status in ("picked", "strong") and address_service.enabled():
            # Несколько названий выбрала LLM (или данные МИС) — для «где сделать» это ответ, а не выбор пациента: места
            # оказания этих строк, а не общий список филиалов (BUG-2026-10-20-ADDRESS-MULTI-PICK-AS-DROPPED).
            out[address_service.ADDRESS_SERVICE_PICKS_KEY] = list(verdict.names)
        flags.add("entity_dropped_unverified_service_name")
        return None
    if verdict.action == "refuse":
        flags.add("entity_dropped_unverified_service_name")
        return None
    if verdict.action == "clarify":
        out[BOOKING_SERVICE_OPTIONS_KEY] = list(verdict.names)
        out[BOOKING_SERVICE_CLARIFY_KEY] = True
        return None
    if len(verdict.names) == 1:
        # Одно название — прежнее подтверждение «Похоже, вы имели в виду …?».
        out["_catalog_service_candidate"] = verdict.names[0]
        out["_catalog_service_query"] = phrase
        flags.add("entity_catalog_service_fuzzy_candidate")
    else:
        out[BOOKING_SERVICE_OPTIONS_KEY] = list(verdict.names)
    return None


# Метки, где нечёткую догадку каталога перед «Похоже, вы имели в виду …?» судит LLM (`catalog_confirm_llm = on`); у цены
# переспроса нет вовсе — строку выбирает LLM (`router._maybe_start_catalog_confirm`).
_LLM_JUDGED_FUZZY_LABELS = frozenset({"APPOINTMENT", "ADDRESS", "TEST_ASSIST"})


def catalog_confirm_llm_enabled() -> bool:
    from agent_logic_2 import config as _cfg  # noqa: PLC0415 — настройка читается на ход (стенд меняет её на прогон)

    return str(getattr(_cfg, "MR_CATALOG_CONFIRM_LLM", "off") or "off").strip().lower() == "on"


async def judge_fuzzy_service_candidate(
    *,
    label: str,
    candidate: str,
    query: str,
    user_text: str,
    services: Any,
    runtime_llm_mode: str,
    out: dict[str, Any],
    flags: set[str],
    candidate_flag: str,
    prior_flags: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Нечёткая догадка каталога → переспрос «Похоже, вы имели в виду …?» (BUG-2026-10-22-CATALOG-CONFIRM-BY-LETTERS).

    Под off — как было: догадка правил по буквам идёт на «да/нет» («анализ» → «Анализ крови на аминокислоты»,
    «фадиатоп» → «Кровь на ВИЧ»). Под on её судит LLM тем же выбором, что точное совпадение записи
    (`judge_booking_service`): подтвердила — переспрос как был; назвала другую — переспрос с её названием; «не услуга»,
    «слишком общее», «ничего не подходит» — переспроса нет, ход идёт без услуги; LLM недоступна — как было.
    """

    def _confirm(name: str) -> None:
        out["_catalog_service_candidate"] = name
        out["_catalog_service_query"] = query
        flags.add(candidate_flag)

    if not catalog_confirm_llm_enabled() or label not in _LLM_JUDGED_FUZZY_LABELS:
        _confirm(candidate)
        return
    if not names_service(query):
        # Фраза догадки услугу не называет («анализы» правила walk-in): выбор строки LLM не зовёт, и «оставить» суда — не
        # решение, а переспрос по буквам (прод 22.10: «кровь ребенку 9 месяцев», «Анализ T-SPOT» → «Анализ крови на
        # аминокислоты?» — нормализатор услуги переписал «анализы» в эту строку, ответ закэширован в процессе).
        flags.add("catalog_confirm_judged_not_a_service")
        return
    judged_before = any(
        f.startswith("catalog_confirm_judged_") and not f.endswith("_unavailable") for f in set(flags) | set(prior_flags)
    )
    verdict = await judge_booking_service(
        query,
        candidate,
        catalog=services._ensure_service_catalog_rows_loaded,
        reply=user_text,
        runtime_llm_mode=runtime_llm_mode,
    )
    flags.add(f"catalog_confirm_judged_{verdict.status or verdict.action}")
    if verdict.status == "unavailable":
        # LLM этого хода уже судила другую фразу (была доступна) или кончился лимит решений хода — «недоступна» здесь не
        # сбой модели: догадку по буквам не выносим (g0196; g0287 «принести кал на анализ» — лимит потратили другие суды
        # хода, и переспрос шёл «Анализ крови на аминокислоты?»).
        if not judged_before and not llm_runtime.turn_budget_exhausted():
            _confirm(candidate)
        return
    judged = _apply_service_verdict(label=label, verdict=verdict, canonical=candidate, phrase=query, out=out, flags=flags)
    if judged:
        _confirm(judged)


def _branch_answer_keeps_target(*, label: str, user_text: str, state: SessionState) -> bool:
    """Ответ на вопрос о филиале в записи, у которой уже есть цель (врач, специальность или услуга)."""

    if label != "APPOINTMENT":
        return False
    from . import pending_answer

    answer = pending_answer.branch_answer(user_text)
    if answer is None or answer[0] == "new_topic":
        return False
    entities = state.last_entities
    return bool(entities.get("service_name") or entities.get("specialty") or entities.get("doctor_name") or entities.get("doctor_id"))


def _is_verbatim_catalog_name(value: str, user_text: str) -> bool:
    """Значение реплика содержит целиком (вызывающий проверяет, что его подставила каталожная инъекция —
    `catalog_service_exact`: это название строки каталога, а не слова LLM)."""

    name = _normalise_catalog_text(value)
    return len(name) >= 5 and name in _normalise_catalog_text(user_text)


def _is_active_booking_target(*, label: str, value: str, state: SessionState) -> bool:
    """Значение `service_name` — цель активной записи из состояния (её подтвердил прошлый ход).

    Как у гарда B′ планировщика для врача и специальности: в активной записи известная цель переживает ход-ответ на
    шаг записи. Новую услугу реплика назвала бы сама — тогда значение берётся из реплики и сюда не доходит; другое
    значение, чем в состоянии, не совпадёт и проверяется по каталогу, как раньше (A′-2: цель из прошлой темы не
    подменяет новую — без активной записи правило не действует).
    """

    if label != "APPOINTMENT" or not value:
        return False
    entities = state.last_entities
    if not entities.get("appointment_flow_active"):
        return False
    if str(entities.get("appointment_action") or "book").strip().lower() in {"cancel", "reschedule"}:
        return False
    current = str(entities.get("service_name") or "").strip()
    return bool(current) and current.casefold() == value.casefold()


def _should_drop_service_name_in_active_reschedule(
    *,
    label: str,
    user_text: str,
    raw_entities: dict[str, Any],
    state: SessionState,
) -> bool:
    """
    В активном cancel/reschedule сценарии не даем случайному service grounding
    перетирать doctor-context на репликах-слотах (адрес/дата/ФИО пациента).
    """

    if label != "APPOINTMENT":
        return False

    action = str(
        raw_entities.get("appointment_action")
        or state.last_entities.get("appointment_action")
        or ""
    ).strip().lower()
    if action not in {"cancel", "reschedule"}:
        return False

    has_doctor_context = bool(
        raw_entities.get("doctor_name")
        or raw_entities.get("doctor_id")
        or state.last_entities.get("doctor_name")
        or state.last_entities.get("doctor_id")
    )
    if not has_doctor_context:
        return False

    text = str(user_text or "").strip()
    if not text:
        return False
    # Если пользователь явно говорит про услугу/процедуру — разрешаем update.
    if _SERVICE_ANCHOR_HINT_RE.search(text):
        return False

    return (
        looks_like_patient_fio(text)
        or has_datetime_signal(text)
        or bool(match_city(text))
        or looks_like_address(text)
    )


def _should_keep_nonbookable_service_name_as_is(
    *,
    raw_value: str,
    decision_flags: set[str],
    user_text: str = "",
) -> bool:
    """
    Сохраняет generic service_name как есть в ADDRESS-flow для услуг без записи.

    Это нужно для сценариев вида "где можно сдать диабетический профиль",
    где роутеру важен сам класс услуги (`анализы`/`ЭКГ`), а не жесткая
    каталожная канонизация до конкретной строки прайса.

    Защита от галлюцинаций LLM: если в user_text нет ни одного
    префикс-совпадения с raw_value, это, скорее всего, выдумка модели,
    и сохранять её в state опасно — она утечёт в следующий PRICE-турн.
    Подтверждённый кейс 2026-05-05: на запрос «Сдать витамин Д»
    классификатор выдавал service_name=«Маммопластика — увеличение
    груди со стоимостью имплантов», бот отвечал ADDRESS (адреса
    филиалов), а stale «Маммопластика» оседала в state. На следующий
    короткий запрос «Стоимость» бот строил PRICE-ответ для
    маммопластики и рендерил пустой текст.

    :param raw_value: исходное service_name из решения/guardrail
    :param decision_flags: флаги текущего решения
    :param user_text: текущий запрос пользователя (для sanity-check)
    :return: True, если service_name нужно оставить без catalog-grounding
    """

    value = str(raw_value or "").strip()
    if not value or match_city(value):
        return False
    if "policy_nonbookable_walkin" not in decision_flags:
        return False
    # Классы walk-in (`walkin.is_class`: выбор LLM или детектор nonbookable_service_hint) намеренно описывают класс услуги,
    # а не строку прайса, поэтому у них нет общего префикса с тем, что назвал
    # пациент («диабетический профиль»). Их ставит бот, а не модель свободным текстом, —
    # сохраняем без prefix-проверки, которая ниже ловит только free-form галлюцинации.
    if walkin.is_class(value):
        return True
    if user_text:
        text_lower = user_text.lower()
        value_tokens = {
            t for t in re.findall(r"[a-zа-яё0-9]+", value.lower()) if len(t) >= 4
        }
        text_tokens = {
            t for t in re.findall(r"[a-zа-яё0-9]+", text_lower) if len(t) >= 4
        }
        if value_tokens and text_tokens:
            has_common_prefix = any(
                vt[:4] == tt[:4]
                for vt in value_tokens
                for tt in text_tokens
            )
            if not has_common_prefix:
                return False
    return True


def _should_keep_raw_service_anchor_for_unverified_doctor(
    *,
    label: str,
    raw_value: str,
    user_text: str,
    decision_flags: set[str],
) -> bool:
    """
    Сохраняет явный service anchor при неподтвержденной фамилии врача.

    Сценарий: "запишите к Евграфову на холтер". Здесь doctor_name может быть
    неподтвержден, но сама услуга названа явно, поэтому её не нужно
    пере-ground'ивать в длинную каноническую строку.

    :param label: текущий label
    :param raw_value: исходный service_name из решения
    :param user_text: текст пользователя
    :param decision_flags: флаги решения
    :return: True, если service_name следует сохранить как есть
    """

    value = str(raw_value or "").strip()
    if label != "APPOINTMENT" or not value:
        return False
    if "doctor_name_unverified" not in decision_flags:
        return False
    if not _SERVICE_ANCHOR_HINT_RE.search(value):
        return False
    if service_name_conflicts_with_doctor(value, user_text):
        return False
    return True


async def verify_doctor_entities_in_decision(
    decision: RouteDecision,
    user_text: str,
    services: Services,
) -> RouteDecision:
    """
    Единая точка doctor-name валидации для decision.

    Поведение синхронизировано с историческим router pre-ground шагом:
    - doctor_name подтверждается только через `resolve_doctor_name`;
    - неподтвержденный doctor_name удаляется;
    - для APPOINTMENT возможна конверсия неподтвержденного doctor_name -> patient_name;
    - при `overwrite_doctor` и отсутствии подтверждения снимается переключение контекста.
    """
    entities = dict(decision.entities)
    flags = set(decision.flags)
    needs_doctor_verification = bool(entities.get("doctor_name")) or decision.label in {
        "DOCTOR_SCHEDULE",
        "DOCTOR_INFO",
        "APPOINTMENT",
    } or (decision.context_action == "overwrite_doctor")
    if not needs_doctor_verification:
        return decision

    raw = _sanitize_raw_doctor_name(str(entities.get("doctor_name") or ""), flags, entities)
    resolved: str | None = None
    if raw:
        resolved = await services.resolve_doctor_name(raw)
    if not resolved and decision.context_action == "overwrite_doctor":
        resolved = await services.resolve_doctor_name(user_text)

    context_action = decision.context_action
    if resolved:
        entities["doctor_name"] = resolved
        flags.add("doctor_name_verified")
        service_name = str(entities.get("service_name") or "").strip()
        if service_name and service_name_conflicts_with_doctor(service_name, resolved):
            entities.pop("service_name", None)
            flags.add("entity_dropped_doctor_like_service_name")
    elif raw:
        entities.pop("doctor_name", None)
        flags.add("doctor_name_unverified")
        if decision.label == "APPOINTMENT" and looks_like_patient_fio(raw):
            entities["patient_name"] = raw
            flags.add("patient_name_from_unverified_doctor")
            context_action = "continue"
    elif decision.context_action == "overwrite_doctor" and decision.label == "OTHER":
        # Не подтвердили нового врача по кэшу — считаем, что это не переключение врача.
        context_action = "continue"

    return RouteDecision(
        label=decision.label,
        confidence=decision.confidence,
        entities=entities,
        flags=flags,
        needs_handoff=decision.needs_handoff,
        context_action=context_action,
        source=decision.source,
        clarify_needed=decision.clarify_needed,
        clarify_reason=decision.clarify_reason,
        clarify_slots=list(decision.clarify_slots),
        intent_candidates=list(decision.intent_candidates),
    )


async def sanitize_doctor_entities(
    entities: dict[str, Any],
    services: Services,
    *,
    label: str,
) -> dict[str, Any]:
    """
    Санитизация doctor_name в произвольном entities-словаре перед merge в state.

    Используется для quick-fill/pending merge шагов.
    """
    out = dict(entities or {})
    raw = str(out.get("doctor_name") or "").strip()
    if not raw:
        return out

    resolved = await services.resolve_doctor_name(raw)
    if resolved:
        out["doctor_name"] = resolved
        service_name = str(out.get("service_name") or "").strip()
        if service_name and service_name_conflicts_with_doctor(service_name, resolved):
            out.pop("service_name", None)
        return out

    out.pop("doctor_name", None)
    if label == "APPOINTMENT" and looks_like_patient_fio(raw) and not out.get("patient_name"):
        out["patient_name"] = raw
    return out


async def ground_decision_entities(
    decision: RouteDecision,
    user_text: str,
    state: SessionState,
    services: Services,
    pending: dict[str, Any] | None = None,
    runtime_llm_mode: str = "",
) -> GroundingResult:
    raw = dict(decision.entities or {})
    if not raw:
        return GroundingResult()

    label = str(decision.label or "OTHER")
    out: dict[str, Any] = {}
    flags: set[str] = set()

    for key, value in raw.items():
        if not _is_allowed_key(str(key), label, pending):
            flags.add(f"entity_dropped_not_allowed:{key}")
            continue

        if key == "doctor_name":
            name = str(value or "").strip()
            if not name:
                continue
            resolved = await services.resolve_doctor_name(name)
            if resolved:
                out[key] = resolved
                if resolved != name:
                    flags.add("entity_grounded_doctor_name")
            else:
                flags.add("entity_dropped_unverified_doctor_name")
            continue

        if key == "city":
            city = match_city(str(value or "")) or match_city(user_text)
            if city:
                out[key] = city
                if str(value or "").strip() and city != str(value).strip():
                    flags.add("entity_grounded_city")
            else:
                flags.add("entity_dropped_unknown_city")
            continue

        if key == "service_name" and _branch_answer_keeps_target(label=label, user_text=user_text, state=state):
            # Ход — ответ на вопрос «Какой филиал вам удобен?» (решение LLM или запасной путь предпроверки записи): услугу
            # реплика не называет, цель записи остаётся прежней. Иначе слова об адресе становились «услугой»: «до скольки
            # работает филиал на Победы?» → «Похоже, вы имели в виду «Послеоперационная консультация уролога»?» (свип 18.10).
            flags.add("entity_ignored_service_in_branch_answer")
            continue

        if key == "service_name":
            # Понимание хода (ход (б), шаг 1, `on`): реплика — список услуг; вывод одной услуги
            # из всей реплики свёл бы его к одной строке. Список разберёт корзина цены.
            if label in {"PRICE", "TEST_ASSIST"} and turn_understanding.acting_list():
                flags.add("entity_dropped_service_name_list")
                turn_understanding.note_acted("grounder_single_dropped")
                continue
            # Ход (б), этап 2, U2 (`turn_understanding_one = on`): одна сильная по данным МИС услуга с одним названием прайса —
            # это название, заземлять по буквам заново не нужно (дизайн 19.10 §3, исход one_strong). Только цена: подбор анализов
            # с услугой из понимания — довод П1 (`router._inject_catalog_candidates`), здесь его не расширяем.
            chosen = turn_understanding.strong_choice(user_text) if label == "PRICE" else None
            if chosen:
                out[key] = chosen
                flags.add("entity_service_name_from_understanding")
                turn_understanding.note_acted("grounder_from_understanding")
                continue
            raw_value = str(value or "").strip()
            if _should_keep_nonbookable_service_name_as_is(
                raw_value=raw_value,
                decision_flags=set(decision.flags),
                user_text=user_text,
            ):
                out[key] = raw_value
                flags.add("entity_kept_nonbookable_service_name")
                continue

            if _should_keep_raw_service_anchor_for_unverified_doctor(
                label=label,
                raw_value=raw_value,
                user_text=user_text,
                decision_flags=set(decision.flags),
            ):
                out[key] = raw_value
                flags.add("entity_kept_explicit_service_anchor")
                continue

            if _should_drop_service_name_in_active_reschedule(
                label=label,
                user_text=user_text,
                raw_entities=raw,
                state=state,
            ):
                flags.add("entity_dropped_stale_service_name_in_reschedule")
                continue

            # Стараемся брать услугу из текущей реплики, а не "как есть" из LLM,
            # чтобы не залипали ложные service_name.
            phrase: str | None = None
            if label == "PRICE":
                phrase = await asyncio.to_thread(
                    resolve_price_service_name_from_catalog,
                    user_text,
                    current_service_name=str(value or ""),
                )
            phrase_in_reply = False
            if not phrase:
                reply_phrase = extract_service_phrase(user_text)
                phrase_in_reply = bool(reply_phrase)
                phrase = reply_phrase or extract_service_phrase(str(value or ""))
            strict_catalog_labels = {"APPOINTMENT", "PRICE", "ADDRESS", "TEST_ASSIST"}
            if phrase:
                if _is_doctor_like_service_collision(
                    label=label,
                    user_text=user_text,
                    phrase=phrase,
                    decision_flags=set(decision.flags),
                ):
                    flags.add("entity_dropped_doctor_like_service_name")
                    continue
                if label in strict_catalog_labels:
                    match = await services.match_catalog_service(
                        phrase,
                        current_service_name=str(state.last_entities.get("service_name") or ""),
                        context_text=user_text,
                    )
                    status = str(match.get("status") or "")
                    nonbookable_keep = "policy_nonbookable_walkin" in set(decision.flags)
                    if status == "exact":
                        canonical = str(match.get("canonical") or "").strip()
                        if (
                            canonical
                            and label in _LLM_JUDGED_SERVICE_LABELS
                            and phrase_in_reply
                            and not nonbookable_keep
                            # Флюорограф/маммограф — запись только через регистратуру (планировщик, прод #668): какая
                            # строка прайса, не важно, выбор не нужен.
                            and not is_fixed_equipment_service(phrase)
                        ):
                            # Услугу, которую назвал пациент, судит LLM, как у цены, — не буквы
                            # (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS: «гастроскопию» → «Гастрин»).
                            verdict = await judge_booking_service(
                                phrase,
                                canonical,
                                catalog=services._ensure_service_catalog_rows_loaded,
                                reply=user_text,
                                runtime_llm_mode=runtime_llm_mode,
                            )
                            if verdict.status:
                                flags.add(f"booking_service_{verdict.status}")
                            canonical = _apply_service_verdict(
                                label=label, verdict=verdict, canonical=canonical, phrase=phrase, out=out, flags=flags
                            )
                            if canonical is None:
                                continue
                        if canonical:
                            out[key] = canonical
                            if canonical != phrase:
                                flags.add("entity_grounded_service_name")
                            continue
                    if status == "fuzzy":
                        candidate = str(match.get("canonical") or "").strip()
                        query = str(match.get("query") or phrase).strip()
                        if candidate:
                            await judge_fuzzy_service_candidate(
                                label=label,
                                candidate=candidate,
                                query=query,
                                user_text=user_text,
                                services=services,
                                runtime_llm_mode=runtime_llm_mode,
                                out=out,
                                flags=flags,
                                candidate_flag="entity_catalog_service_fuzzy_candidate",
                                prior_flags=set(decision.flags or ()),
                            )
                        else:
                            flags.add("entity_dropped_unverified_service_name")
                        continue
                    if status == "unavailable":
                        if nonbookable_keep and phrase and not match_city(phrase):
                            out[key] = phrase
                            flags.add("entity_kept_nonbookable_service_name")
                            continue
                        if (
                            phrase
                            and not match_city(phrase)
                            and (
                                _SERVICE_ANCHOR_HINT_RE.search(phrase)
                                or _SERVICE_ANCHOR_HINT_RE.search(user_text or "")
                            )
                        ):
                            out[key] = phrase
                            flags.add("entity_kept_service_without_catalog")
                            continue
                        flags.add("entity_dropped_unverified_service_name")
                        continue
                    if nonbookable_keep and phrase and not match_city(phrase):
                        out[key] = phrase
                        flags.add("entity_kept_nonbookable_service_name")
                        continue
                    flags.add("entity_dropped_unverified_service_name")
                    continue

                out[key] = phrase
                if str(value or "").strip() and phrase != str(value).strip():
                    flags.add("entity_grounded_service_name")
            elif isinstance(value, str):
                fallback = value.strip()
                if (
                    label == "APPOINTMENT"
                    and "catalog_service_exact" in set(decision.flags)
                    and _is_verbatim_catalog_name(fallback, user_text)
                ):
                    # Пациент написал название строки каталога дословно (ответ на список названий
                    # `router._maybe_ask_booking_service_choice`) — сильное совпадение по данным; `extract_service_phrase`
                    # каноническое название не узнаёт (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS).
                    out[key] = fallback
                    flags.add("entity_kept_verbatim_catalog_service_name")
                elif label == "APPOINTMENT" and fallback and fallback == _booking_service_picked():
                    # Пункт списка названий услуги записи, выбранный одной функцией ответа на висящий вопрос («1»,
                    # «второй вариант», «массаж спины» → название из показанного списка): сильное совпадение по данным.
                    out[key] = fallback
                    flags.add("entity_kept_pending_answer_service_name")
                elif _is_active_booking_target(label=label, value=fallback, state=state):
                    # BUG-2026-10-18-BRANCH-REPLY-TOPIC-SWITCH (подбаг B): реплика услугу не называет (ответ на шаг
                    # записи — «пр.Ленина, 5»), а значение — цель этой же записи из состояния, уже заземлённая и
                    # показанная. Каноническое название каталога `extract_service_phrase` не узнаёт, и без этой ветки
                    # цель ронялась как «неподтверждённая» → гард B′ → «Пока не понял, к какому врачу».
                    out[key] = fallback
                    flags.add("entity_kept_booking_target_service_name")
                elif label in strict_catalog_labels:
                    if "policy_nonbookable_walkin" in set(decision.flags) and fallback and not match_city(fallback):
                        out[key] = fallback
                        flags.add("entity_kept_nonbookable_service_name")
                    else:
                        flags.add("entity_dropped_unverified_service_name")
                elif fallback and not match_city(fallback) and not _GENERIC_SERVICE_FALLBACK_RE.search(fallback):
                    out[key] = fallback
                    flags.add("entity_kept_llm_service_name")
                else:
                    flags.add("entity_dropped_unverified_service_name")
            else:
                flags.add("entity_dropped_unverified_service_name")
            continue

        if key == "branch_name":
            branch = str(value or "").strip()
            if branch and looks_like_address(branch):
                out[key] = branch
            else:
                flags.add("entity_dropped_non_address_branch")
            continue

        out[key] = value

    # Не позволяем перетирать уже известного врача неподтвержденным значением.
    if (
        not out.get("doctor_name")
        and state.last_entities.get("doctor_name")
        and "doctor_name" in raw
    ):
        flags.add("entity_doctor_kept_from_state")

    if (
        label == "APPOINTMENT"
        and booking_slots.specialty_enabled()
        and "entity_dropped_unverified_service_name" in flags
        and not any(out.get(k) for k in ("specialty", "doctor_name", "doctor_id"))
    ):
        # Услуга реплики не подтвердилась (LLM или данные: «время» в «На какое время запись к неврологу»), цели нет:
        # специальность из слов реплики — кандидат правил, запись к ней подтверждает LLM. Без подтверждения — честный
        # отказ гарда B′, как было.
        # Уточнение пациента («к детскому аллергологу», «по ДМС») или процедура («на узи кисти рук») — исход LLM,
        # не запись к кандидату (DEBT-2026-10-20-BOOKING-SPECIALTY-QUALIFIER); флаг исхода — для стенда.
        specialty = extract_specialty(user_text or "")
        if specialty:
            verdict = await booking_slots.booking_specialty_choice(user_text, specialty)
            if booking_slots.confirms(verdict, specialty):
                out["specialty"] = specialty
                flags.add("booking_specialty_llm")
            elif verdict.outcome:
                flags.add(f"booking_specialty_llm_{verdict.outcome}")

    return GroundingResult(entities=out, flags=flags)
