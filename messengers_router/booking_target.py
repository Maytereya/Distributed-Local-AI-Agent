"""Цель записи по смыслу: LLM размечает слова реплики по ролям — цель, филиал, дата (DESIGN-2026-10-28-booking-target-by-meaning).

Поток записи брал цель из первой содержательной реплики разрезом регулярками: кто услуга, кто филиал, кто дата. Роль слова решали
буквы — «Записаться на ФГДС на Ленина 5 — 8 октября» давал филиал «ФГДС», «Терапевт мне нужен на осетинской 4» — услугу
«Осетинской» и филиал «осетинской 4», которого нет (К13, К8; аудит С6 28.10). Когда тему записи решал LLM-классификатор, он писал
цель свободным текстом («к врачу» → услуга «врач»), заземление её роняло, и поток записи не стартовал.

Здесь, как у разметки цены (`services/_turn_spans`): реплика показывается LLM с номерами слов, ответ — группы номеров по ролям и метка
действия (принцип 2: выбор из перечисленного; фраза — подстрока реплики по построению). Значения — данные: филиал — справочник
филиалов, дата — разбор даты, цель — прежние судьи по смыслу (`role_pick`, `judge_booking_service`) над словами группы
(`flow_policy.booking_target_slots`). Действие монотонно: отмену или перенос, найденные правилом, разметка в «записаться» не
превращает.

Ходы: тема — запись, цель ещё не известна (первый ход записи или ответ на вопрос «к какому врачу или на какую услугу»). Ответы на
шаги филиала, даты, ФИО — структурные, их разбирают правила, как раньше.

Сбой LLM: узел `booking_target` в `llm_fail_offer` — честный оффер; без узла — прежний разрез по буквам.

Выключатель `[MESSENGER_ROUTER] booking_target_llm = off | on` (off — как было).
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

from . import evidence_keys as ek
from . import llm_runtime, turn_telemetry
from .llm_mode_policy import fail_offer
from .policies import REQUIRED_SLOTS, build_branch_index, match_branch_hint
from .prompt_registry import load_prompt_text
from .runtime_config import config as _cfg
from .services._turn_spans import timeout_for, words

if TYPE_CHECKING:
    from .mess_types import RouteDecision
    from .services import Services

logger = logging.getLogger(__name__)

TASK_NAME = "booking_target"
SITE = TASK_NAME  # не задача `choose()`: свой контракт ответа (группы номеров по ролям), промпт `booking_target`
FAIL_NODE = TASK_NAME  # узел `[MESSENGER_ROUTER] llm_fail_offer`
ROLES = ("target", "branch", "date")
ACTIONS = ("book", "cancel", "reschedule", "none")
_MAX_WORDS = 40  # длиннее — письмо, а не реплика записи (как у разметки цены)
_MAX_GROUPS = 16
_QUEUE_TIMEOUT_MS = 1500
_TOKENS_FIXED = 40
_TOKENS_PER_WORD = 8
_TOKENS_MAX = 400



def _any_of(slot: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in slot.split(":", 1)[-1].split(",") if key.strip())


# Слоты, которые на ходу разметки решает она: прежний разрез реплики их не пишет, прежние значения правил и свободный текст
# LLM-классификатора из решения хода снимаются. Цель и филиал — слоты обязательных шагов записи (`REQUIRED_SLOTS`), дата —
# `date_*` и границы времени (`time_from`, `time_to`; «в любое время» — не дата, его разбирает прежнее дозаполнение).
TARGET_KEYS = (*_any_of(REQUIRED_SLOTS["APPOINTMENT"][0]), "test_name")
BRANCH_KEYS = tuple(key for key in _any_of(REQUIRED_SLOTS["APPOINTMENT"][1]) if key != "city")
_DATE_PREFIXES = ("date_", "time_from", "time_to")
_DOCTOR_KEYS = ("doctor_name", "doctor_id")


def owned(key: str) -> bool:
    """Слот цели, филиала или даты — его на ходу разметки решает она."""

    return key in TARGET_KEYS or key in BRANCH_KEYS or str(key).startswith(_DATE_PREFIXES)
_BOOK = "book"
_KEEPS_ACTION = frozenset({"cancel", "reschedule"})

_GROUPS = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": list(ACTIONS)}, **{role: _GROUPS for role in ROLES}},
    "required": ["action", *ROLES],
}


def enabled() -> bool:
    """`booking_target_llm = on`: цель, филиал и дату записи размечает LLM, а не регулярки."""

    return str(getattr(_cfg, "MR_BOOKING_TARGET_LLM", "off") or "off").strip().lower() == "on"


@dataclass(frozen=True)
class BookingParse:
    """Разметка реплики: статус и слова каждой роли (подстроки реплики), действие — метка LLM."""

    reply: str
    status: str  # picked | none_fit | empty | unavailable | invalid | too_many
    action: str = ""
    target: tuple[str, ...] = ()
    branch: tuple[str, ...] = ()
    date: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in ("picked", "none_fit", "empty")

    @property
    def roles(self) -> str:
        return "+".join(role for role in ROLES if getattr(self, role)) or "none"


_PARSE: contextvars.ContextVar[BookingParse | None] = contextvars.ContextVar("mr_booking_target", default=None)


def begin() -> None:
    """Новый ход — разметки прошлого хода нет (зовёт `turn_understanding.begin_turn`)."""

    _PARSE.set(None)


def parsed(reply: str) -> BookingParse | None:
    """Разметка этой реплики в этом ходу, если её спрашивали."""

    parse = _PARSE.get()
    return parse if parse is not None and parse.reply == str(reply or "") else None


def acting(reply: str) -> BookingParse | None:
    """Разметка действует на этом ходу: цель, филиал и дату реплики прежний разрез по буквам не пишет."""

    parse = parsed(reply)
    return parse if parse is not None and parse.ok else None


def prompt(reply: str, toks: Sequence[Any]) -> tuple[str, str]:
    """Инструкция (system) и данные хода (user): реплика пациента — данные, а не инструкция (SEC-3)."""

    template = load_prompt_text(SITE)
    system, _, data = template.partition("<<DATA>>")
    numbered = " ".join(f"{k}:{w.text}" for k, w in enumerate(toks, 1))
    return system.strip(), data.replace("<<QUESTION>>", reply).replace("<<WORDS>>", numbered).strip()


def judge(raw: Any, n_words: int) -> tuple[str, dict[str, list[tuple[int, int]]]] | None:
    """Ответ → (действие, отрезки ролей с нуля); None — ответ нарушил контракт.

    Группа — `[первое, последнее]`, `[k]` или возрастающий перечень номеров; перечень с пропуском («8 октября в 10» без «в» —
    `[7, 8, 10]`) — несколько сплошных групп. Номера 1..n; группы не пересекаются ни внутри роли, ни между ролями.
    """

    if not isinstance(raw, dict):
        return None
    action = raw.get("action")
    if action not in ACTIONS:
        return None
    spans: dict[str, list[tuple[int, int]]] = {}
    taken: list[tuple[int, int]] = []
    for role in ROLES:
        groups = raw.get(role, [])
        if not isinstance(groups, list) or len(groups) > _MAX_GROUPS:
            return None
        spans[role] = []
        for group in groups:
            if not isinstance(group, list) or not group or not all(isinstance(k, int) and not isinstance(k, bool) for k in group):
                return None
            if len(group) <= 2:
                runs = [(group[0], group[-1])]
            elif all(nxt > cur for cur, nxt in zip(group, group[1:])):
                runs = _runs(group)
            else:
                return None
            for first, last in runs:
                if not 1 <= first <= last <= n_words:
                    return None
                spans[role].append((first - 1, last - 1))
                taken.append((first - 1, last - 1))
    taken.sort()
    if any(prev[1] >= cur[0] for prev, cur in zip(taken, taken[1:])):
        return None
    return str(action), {role: sorted(found) for role, found in spans.items()}


def _runs(numbers: Sequence[int]) -> list[tuple[int, int]]:
    """Возрастающий перечень номеров → сплошные отрезки (первый, последний)."""

    runs: list[tuple[int, int]] = []
    for number in numbers:
        if runs and number == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], number)
        else:
            runs.append((number, number))
    return runs


async def understand(reply: str) -> BookingParse:
    """Разметка реплики — один раз за ход (повторный вопрос того же хода читает память)."""

    reply = str(reply or "")
    have = parsed(reply)
    if have is not None:
        return have
    result = await _parse(reply)
    _PARSE.set(result)
    return result


async def _parse(reply: str) -> BookingParse:
    started = time.monotonic()
    toks = words(reply)
    if not toks:
        return BookingParse(reply, "empty")
    if len(toks) > _MAX_WORDS:
        turn_telemetry.note_choice(TASK_NAME, "too_many", options=len(toks), ms=_ms(started))
        return BookingParse(reply, "too_many")
    system, user = prompt(reply, toks)
    try:
        raw, _usage = await llm_runtime.generate_structured(
            site=SITE,
            system=system,
            user=user,
            schema=_SCHEMA,
            options={"temperature": 0, "num_predict": min(_TOKENS_MAX, _TOKENS_FIXED + _TOKENS_PER_WORD * len(toks))},
            timeout_s=timeout_for(len(toks)),
            queue_timeout_ms=_QUEUE_TIMEOUT_MS,
        )
    except Exception as exc:  # сбой LLM — без разметки; что делать, решает узел `llm_fail_offer` (`apply`)
        logger.warning("booking_target_unavailable: %s %s", type(exc).__name__, llm_runtime.error_detail(exc) or "-")
        reason = "budget" if isinstance(exc, llm_runtime.LLMBudgetExhaustedError) else ""
        turn_telemetry.note_choice(TASK_NAME, "unavailable", options=len(toks), ms=_ms(started), reason=reason)
        return BookingParse(reply, "unavailable")
    try:
        data = json.loads(str(raw or "").strip())
    except (TypeError, ValueError):
        data = None
    judged = judge(data, len(toks))
    if judged is None:
        logger.warning("booking_target_invalid: %r", str(raw or "")[:200])
        turn_telemetry.note_choice(TASK_NAME, "invalid", options=len(toks), ms=_ms(started))
        return BookingParse(reply, "invalid")
    action, spans = judged
    texts = {role: tuple(reply[toks[i].start : toks[j].end] for i, j in spans[role]) for role in ROLES}
    status = "picked" if any(texts.values()) else "none_fit"
    turn_telemetry.note_choice(TASK_NAME, status, options=len(toks), ms=_ms(started), outcome=action)
    return BookingParse(reply, status, action=action, **texts)


def target_turn(state_entities: dict[str, Any], pending: dict[str, Any] | None) -> bool:
    """Ход, на котором поток записи берёт цель: первый ход записи или ответ на вопрос «к какому врачу или на какую услугу».

    Не размечаются: выбор из показанного списка (пункт уже выбран), шаги отмены и переноса, ответы на шаги филиала, даты, ФИО.
    """

    from . import expects, pending_answer

    if pending_answer.picked(expects.Question.BOOKING_SERVICE) or pending_answer.picked(expects.Question.APPOINTMENT_BRANCH):
        return False
    flow_active = bool(state_entities.get("appointment_flow_active"))
    if flow_active and str(state_entities.get("appointment_action") or "").strip().lower() in _KEEPS_ACTION:
        return False
    if isinstance(pending, dict) and pending.get("label") == "APPOINTMENT":
        missing = pending.get("missing") if isinstance(pending.get("missing"), list) else []
        as_text = " ".join(str(item or "").lower() for item in missing)
        return any(key in as_text for key in ("doctor_id", "doctor_name", "specialty", "service_name"))
    return not flow_active


def merged_action(rules_action: str, llm_action: str) -> str:
    """Действие хода монотонно: отмену или перенос, найденные правилом, LLM в «записаться» не превращает (принцип 5: отказ,
    понятый как согласие, запрещён); LLM может поднять «записаться» до отмены или переноса. `none` — действие не меняется."""

    rules_action = str(rules_action or "").strip().lower()
    llm_action = str(llm_action or "").strip().lower()
    if rules_action in _KEEPS_ACTION:
        return rules_action
    if llm_action in (_BOOK, *_KEEPS_ACTION):
        return llm_action
    return rules_action


def note(parse: BookingParse, slots: dict[str, Any], acted: str) -> None:
    """Строка хода: статус разметки, роли с словами, исход филиала, что сделано — без текста реплики."""

    branch = "none"
    if parse.branch:
        branch = "matched" if any(slots.get(key) for key in BRANCH_KEYS) else "unmatched"
    turn_telemetry.note(
        bt_status=parse.status,
        bt_action=parse.action or "-",
        bt_roles=parse.roles,
        bt_branch=branch,
        bt_date="parsed" if any(value for key, value in slots.items() if str(key).startswith(_DATE_PREFIXES)) else ("unparsed" if parse.date else "none"),
        bt_acted=acted,
    )


async def names_reception_role(parse: BookingParse, services: "Services | None") -> bool:
    """Слова цели называют только роль приёма врача из каталога ролей МИС («к терапевту», «гинекологу») — цель хода приём, его
    решает роль (`role_pick`), а не суд услуги по строкам прайса: одно слово специальности сильно совпадало с процедурой по
    синониму клиники («гинекологу» → «УЗИ гинекологическое», свип 28.10). Процедура в словах («гинекологическое УЗИ») — услуга."""

    target = " ".join(part for part in parse.target if part.strip())
    if not target or services is None:
        return False
    from . import role_pick

    try:
        found = (await role_pick.catalog(services)).find(target)
    except Exception:  # каталог ролей недоступен — как без этого слоя: слова цели остаются кандидатом услуги
        logger.warning("booking_target_role_catalog_failed", exc_info=True)
        return False
    roles = [*found.exact, *found.partial]
    return bool(roles) and all(role.reception for role in roles)


def branch_outside_target(entities: dict[str, Any], evidence: Any) -> bool:
    """Филиал записи не среди адресов цели по данным хода (адреса услуги или специальности, `ek.ADDRESS`) — в заявку не идёт (К8:
    «Да, можем записать на УЗИ (Финютина)» там, где этого УЗИ нет; разметка читает филиал чаще, чем разрез по буквам). Адресов в
    данных хода нет или врач уже выбран (его адреса — расписание) — проверить нечем, как было."""

    branch = str(entities.get("branch_name") or "").strip()
    if not branch or entities.get("doctor_name") or entities.get("doctor_id"):
        return False
    payload = evidence.get(ek.ADDRESS) if evidence is not None else None
    raw = payload.get("addresses") if isinstance(payload, dict) else None
    addresses = [str(item).strip() for item in raw if str(item or "").strip()] if isinstance(raw, list) else []
    if not addresses:
        return False
    index = build_branch_index([{"id": f"target_{k}", "name": address} for k, address in enumerate(addresses)])
    found, _name = match_branch_hint(branch, index)
    return found is None


def fail_offer_decision(decision: "RouteDecision") -> "RouteDecision":
    """Сбой разметки под узлом `booking_target` — честный оффер оператора (как узлы `topic`, `facet`, `role_pick`)."""

    return replace(decision, label="OTHER", entities={}, flags=set(decision.flags) | {"llm_fail_offer", "booking_target_fail_offer"})


async def apply(
    decision: "RouteDecision",
    *,
    user_text: str,
    state_entities: dict[str, Any],
    pending: dict[str, Any] | None,
    services: "Services | None",
) -> "RouteDecision":
    """Цель, филиал и дата хода записи — из разметки LLM (до роли и суда услуги, которые решают значение по смыслу).

    Решение предпроверки записи (`flow_prelock`) разметку уже применило (`flow_policy.prelock_active_appointment_turn`) — здесь
    только оффер при сбое. Врач, подтверждённый справочником до этого слоя, остаётся.
    """

    if not enabled() or decision.label != "APPOINTMENT" or not str(user_text or "").strip():
        return decision
    if decision.source == "button":
        return decision  # нажатие кнопки: текст хода — подпись кнопки, а не реплика пациента
    if decision.source == "flow_prelock":
        parse = parsed(user_text)
        if parse is not None and not parse.ok and fail_offer(FAIL_NODE):
            note(parse, {}, "fail_offer")
            return fail_offer_decision(decision)
        return decision
    if not target_turn(state_entities, pending):
        return decision
    parse = await understand(user_text)
    if not parse.ok:
        if fail_offer(FAIL_NODE):
            note(parse, {}, "fail_offer")
            return fail_offer_decision(decision)
        note(parse, {}, "rules")
        return decision

    from .flow_policy import booking_target_slots

    verified_doctor = "doctor_name_verified" in (decision.flags or ())
    entities = {
        key: value
        for key, value in (decision.entities or {}).items()
        if not owned(key) or (verified_doctor and key in _DOCTOR_KEYS)
    }
    slots = booking_target_slots(parse, state_entities, services)
    if "service_name" in slots and await names_reception_role(parse, services):
        slots.pop("service_name")
    if verified_doctor and any(entities.get(key) for key in _DOCTOR_KEYS):
        for key in _DOCTOR_KEYS:
            slots.pop(key, None)
    entities.update(slots)
    action = merged_action(str((decision.entities or {}).get("appointment_action") or ""), parse.action)
    if action:
        entities["appointment_action"] = action
    note(parse, slots, "slots")
    return replace(decision, entities=entities, flags=set(decision.flags) | {"booking_target_llm"})


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
