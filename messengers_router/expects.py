"""Поле ожидания `_expects`: какой ответ бот ждёт на свой вопрос — D2.1 (наблюдатель) и D2.4 (предел переспросов).

Пункт 5 принципов (`docs/ARCHITECTURE_PRINCIPLES.md`): бот всегда знает, чего ждёт, и не ходит по кругу. Висящий
вопрос бота жил флагами в разных местах состояния (`_operator_offer_pending`, `_catalog_confirm_pending`,
`appointment_cancel_pending`, фаза CONFIRM, `_pending` с `appointment_action`, `_secondary_offer_pending`,
`_prepare_variant_pending`, `_compound_price_pending`), ответ на него разбирали одиннадцать правил «да/нет», а текста
вопроса, который увидел пациент, в состоянии не было: ни ответу силами LLM (клиент `pending_reply`, D2.3), ни счётчику
переспросов опереться не на что (дизайн `~/Dev/handoffs/2026-10-14_pending_reply_design.md`).

Контракт (общий с клиентом `pending_reply`) — `state.last_entities["_expects"]`, dict:
- `kind` — `yes_no` | `pick` (выбор из вариантов, которые бот назвал);
- `type` — тип вопроса (`Question`): по нему читатель находит свой обработчик, выключатель и строку телеметрии;
- `question` — текст ответа бота, каким его увидел пациент, без персональных данных: ссылки (в ссылке на результат —
  фамилия, год рождения, номер анализа) — `[ссылка]`, ФИО пациента из состояния — `[ФИО]`; не длиннее 500 знаков
  (длиннее — хвост: вопрос в конце);
- `options` — тексты вариантов, если бот их назвал, иначе `[]`;
- `asks` — сколько раз подряд бот задал этот вопрос без продвижения (≥ 1).

Как пишется. Место, где бот задаёт вопрос и ставит висящее ожидание, зовёт `ask()`; обработчик, который не разобрал
ответ и спрашивает то же снова, — `ask(..., reask=True)`. Конец хода (`finish_turn` в `router.patient_routing_stream`)
пишет поле с текстом ответа — или снимает: вопроса в ходе не было, ответ — перевод на оператора, флаг ожидания снят до
конца хода. Счёт `asks`: переспрос — прежний + 1 (прежнего нет — 2: переспрос первым не бывает); новый вопрос того же
типа с тем же текстом, что в прошлом ходе, — прежний + 1; иначе 1. Во время хода в состоянии — вопрос, на который
сейчас отвечают (прошлого хода; его читает тень `pending_reply` до разбора висящего вопроса). Между ходами поле
устаревает только со сроком флага в `_pending` (выбор действия с записью) — такое снимает начало хода; кнопку и смену
темы снимает `flow_policy.clear_on_topic_switch`, конец записи — `clear_on_appointment_end`, перевод — `clear_on_handoff`.

Только наблюдатель: поле не читает ни один обработчик — ответы и решения прежние. В инструменты (а через их
`entities_used` — в промпт свободного ответа LLM) поле не идёт (`without`, планировщик), в NLU и в сводку диалога — тоже
(белые списки `classifier._seed_entities_from_memory`, `context_summary._CRITICAL_KEYS`). Телеметрия хода —
`tu_expects_kind`, `tu_expects_type`, `tu_expects_asks`: ключи и число, без текста пациента и бота.

Предел переспросов (D2.4) — `[MESSENGER_ROUTER] reask_limit = off | on`, по умолчанию off: как было. Петли без
счётчика (аудит 14.10, пп. 2–3): «Подтвердите запись» → «нет, другое время» → тот же вопрос, «Отменить текущий
процесс записи?» → «да, отмените» → тот же вопрос — сколько угодно раз. Инвариант `3bacf58` (шаг дважды без
продвижения → назвать недостающее или оператор): под `on` переспрос звучит, но один вопрос — не больше двух раз
подряд (`MAX_ASKS`); третьего нет — бот называет, какой ответ был нужен, и предлагает оператора одним вопросом
(`router._maybe_offer_operator_on_reask_limit`). Не «сразу оператор» на первом переспросе: лавина переводов.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from . import turn_telemetry
from .mess_types import AppointmentPhase, ResponseEnvelope, SessionState
from .recovery_policy import UNCLEAR_OPERATOR_OFFER_TEXT
from .runtime_config import config as _cfg

EXPECTS_KEY = "_expects"
YES_NO = "yes_no"
PICK = "pick"
QUESTION_MAX_CHARS = 500
# Под `reask_limit = on`: сколько раз подряд звучит один вопрос; переспрос сверх — оффер оператора (D2.4).
MAX_ASKS = 2
REASK_LIMIT_MODES = ("off", "on")
LINK_MASK = "[ссылка]"
NAME_MASK = "[ФИО]"  # та же маска, что у клиента `pending_reply`
# Персональные поля состояния, которые бот может напечатать в ответе (сводка записи, данные результата).
_PERSONAL_KEYS = ("patient_name", "surname")

# Варианты вопросов-выборов — слова самого вопроса, в его порядке (тест сверяет их с текстом, который увидел пациент).
# «Хотите отменить или перенести запись?» — `policies.clarification_question`, переспрос — `router`.
APPOINTMENT_ACTION_OPTIONS = ("отменить", "перенести")
# «…с наркозом (во сне) или без наркоза? Напишите «с наркозом» или «без наркоза».» — `services.prepare`; порядок —
# `with`, `without` (ключи `prepare_variants`).
ANESTHESIA_OPTIONS = ("с наркозом", "без наркоза")


class Question(str, Enum):
    """Тип висящего вопроса; значение — строка `type` контракта. Где бот его задаёт — справа."""

    OPERATOR_OFFER = "operator_offer"  # «Перевести на оператора?» — `response_builder.mark_operator_offer_pending`
    CATALOG_CONFIRM = "catalog_confirm"  # «Похоже, вы имели в виду …? Это верно?» — `router._maybe_start_catalog_confirm`
    APPOINTMENT_CONFIRM = "appointment_confirm"  # «Подтвердите запись» — фаза CONFIRM, `appointment_flow_guard`
    APPOINTMENT_ABORT = "appointment_abort"  # «Отменить текущий процесс записи?» / «…перейти к новому вопросу?»
    APPOINTMENT_ACTION = "appointment_action"  # «Хотите отменить или перенести запись?» — `_pending` с `appointment_action`
    SECONDARY_OFFER = "secondary_offer"  # вторая тема реплики — `orchestrator._mark_secondary_offer_pending`
    ANESTHESIA = "anesthesia"  # «с наркозом или без?» — `response_builder.build_prepare_response`
    COMPOUND_PRICE = "compound_price"  # «Вижу в запросе две услуги: 1. … 2. …» — `response_builder`


# Порядок, в котором следующий ход разбирает висящие вопросы: предпроверка записи (`run_appointment_precheck`), затем
# обработчики до NLU (`orchestrator.pending_dispatch`). Задал ход два вопроса (оффер оператора поверх скрытого
# вторичного оффера, анти-залип поверх переспроса) — в поле тот, на который придётся ответ.
_READING_ORDER = (
    Question.APPOINTMENT_ABORT,
    Question.APPOINTMENT_CONFIRM,
    Question.OPERATOR_OFFER,
    Question.CATALOG_CONFIRM,
    Question.APPOINTMENT_ACTION,
    Question.ANESTHESIA,
    Question.COMPOUND_PRICE,
    Question.SECONDARY_OFFER,
)


# Вопросы, чей флаг живёт в `_pending` со сроком (`memory.get_pending`): между ходами он истекает сам, поэтому начало
# хода сверяет с ним поле. У прочих флагов срока нет — их с полем сверяет конец каждого хода.
_PENDING_SLOT_QUESTIONS = (Question.APPOINTMENT_ACTION,)


@dataclass(frozen=True)
class Asked:
    """Вопрос, заданный в этом ходе. `asks` у переспроса посчитан сразу (известен до конца хода), у нового вопроса —
    0: счёт зависит от текста ответа и считается в конце хода."""

    question: Question
    kind: str
    options: tuple[str, ...]
    reask: bool
    asks: int


# Поле ожидания на начало хода (вопрос, на который отвечает реплика; после сброса кнопкой) и вопросы, заданные ходом.
_PREVIOUS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("mr_expects_previous", default=None)
_ASKED: contextvars.ContextVar[tuple[Asked, ...]] = contextvars.ContextVar("mr_expects_asked", default=())


def read(state: SessionState) -> dict[str, Any] | None:
    """Поле ожидания из состояния (копия), если оно по контракту; иначе None."""

    raw = state.last_entities.get(EXPECTS_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        Question(raw.get("type"))
    except ValueError:
        return None
    asks = raw.get("asks")
    if raw.get("kind") not in (YES_NO, PICK) or not isinstance(asks, int) or isinstance(asks, bool) or asks < 1:
        return None
    return dict(raw)


def begin_turn(state: SessionState) -> None:
    """Новый ход: запомнить вопрос, на который отвечает реплика; вопросов этого хода ещё нет.

    Поле, чей флаг в `_pending` истёк между ходами, снимается сразу: иначе его прочла бы тень `pending_reply` и спросила
    LLM об ответе на вопрос, который никто не разберёт.
    """

    current = read(state)
    question = Question(current["type"]) if current is not None else None
    if question in _PENDING_SLOT_QUESTIONS and not _armed(state, question):
        state.last_entities.pop(EXPECTS_KEY, None)
        current = None
    _PREVIOUS.set(current)
    _ASKED.set(())


def previous() -> dict[str, Any] | None:
    """Поле ожидания на начало хода (после сброса кнопкой) — вопрос, на который отвечает реплика."""

    return _PREVIOUS.get()


def asked() -> tuple[Asked, ...]:
    """Вопросы, которые бот задал в этом ходе, в порядке постановки."""

    return _ASKED.get()


def ask(question: Question, *, kind: str = YES_NO, options: Iterable[str] = (), reask: bool = False) -> None:
    """Бот задаёт вопрос и ставит висящее ожидание. Звать там, где ставится флаг, который читает обработчик ответа.

    :param question: тип вопроса
    :param kind: `yes_no` | `pick`
    :param options: варианты, которые бот назвал (для `pick`), — в порядке вопроса
    :param reask: обработчик не разобрал ответ на свой же вопрос и задаёт его снова
    """

    prev = _PREVIOUS.get()
    asks = 0
    if reask:
        asks = int(prev["asks"]) + 1 if prev is not None and prev.get("type") == question.value else 2
    clean = tuple(text for text in (str(option or "").strip() for option in options) if text)
    _ASKED.set(_ASKED.get() + (Asked(question=question, kind=kind, options=clean, reask=reask, asks=asks),))


def _without_links(line: str) -> str:
    """Ссылки строки — `[ссылка]`: в ссылке на результат фамилия, год рождения и номер анализа пациента."""

    words = line.split(" ")
    for index, word in enumerate(words):
        start = min((pos for pos in (word.find("http://"), word.find("https://")) if pos >= 0), default=-1)
        if start >= 0:
            words[index] = word[:start] + LINK_MASK
    return " ".join(words)


def visible_question(text: str, entities: dict[str, Any] | None = None) -> str:
    """Текст ответа, каким его увидел пациент, без персональных данных, не длиннее `QUESTION_MAX_CHARS`.

    Поле читают тень `pending_reply` (промпт LLM, кэш решений) и отладка — ПДн им не нужны: ссылки — `[ссылка]`, ФИО
    пациента из состояния (`patient_name`, `surname`) — `[ФИО]`. Длиннее предела — хвост: вопрос в конце ответа.
    """

    shown = "\n".join(_without_links(line) for line in str(text or "").strip().split("\n"))
    for key in _PERSONAL_KEYS:
        value = " ".join(str((entities or {}).get(key) or "").split())
        if len(value) >= 2:
            shown = shown.replace(value, NAME_MASK)
    if len(shown) <= QUESTION_MAX_CHARS:
        return shown
    return "…" + shown[-(QUESTION_MAX_CHARS - 1):].lstrip()


def _armed(state: SessionState, question: Question) -> bool:
    """Флаг, по которому обработчик ответа на этот вопрос сработает следующим ходом, ещё стоит."""

    entities = state.last_entities
    if question is Question.OPERATOR_OFFER:
        return bool(entities.get("_operator_offer_pending"))
    if question is Question.CATALOG_CONFIRM:
        return isinstance(entities.get("_catalog_confirm_pending"), dict)
    if question is Question.APPOINTMENT_CONFIRM:
        return state.dialog.phase == AppointmentPhase.CONFIRM
    if question is Question.APPOINTMENT_ABORT:
        return bool(entities.get("appointment_cancel_pending") or entities.get("appointment_topic_switch_pending"))
    if question is Question.APPOINTMENT_ACTION:
        pending = entities.get("_pending")
        if not isinstance(pending, dict) or pending.get("label") != "APPOINTMENT":
            return False
        ttl = float(pending.get("ttl") or 0.0)
        if ttl and time.time() - float(pending.get("ts") or 0.0) > ttl:
            return False  # `memory.get_pending` такой не отдаст — обработчик выбора действия не сработает
        missing = pending.get("missing")
        return isinstance(missing, list) and "appointment_action" in missing
    if question is Question.SECONDARY_OFFER:
        return bool(entities.get("_secondary_offer_pending"))
    if question is Question.ANESTHESIA:
        return isinstance(entities.get("_prepare_variant_pending"), dict)
    return isinstance(entities.get("_compound_price_pending"), dict)


def finish_turn(state: SessionState, response: ResponseEnvelope) -> dict[str, Any] | None:
    """Конец хода: записать поле ожидания (текст — ответ, каким его увидел пациент) или снять его.

    Снимается, когда вопроса в ходе не было (ответ принят, тема сменилась), ответ — перевод на оператора или флаг
    ожидания до конца хода снят: поле не обещает ожидания, которого не прочтёт ни один обработчик. Вопросов несколько —
    в поле тот, который следующий ход разберёт первым (`_READING_ORDER`; одного типа — последний).

    :param state: состояние сессии
    :param response: ответ пациенту этого хода (после анти-залипа)
    :return: записанное поле или None
    """

    armed = [item for item in reversed(_ASKED.get()) if _armed(state, item.question)]
    if not armed or response.handoff:
        state.last_entities.pop(EXPECTS_KEY, None)
        return None
    current = min(armed, key=lambda item: _READING_ORDER.index(item.question))
    question = visible_question(response.text, state.last_entities)
    prev = _PREVIOUS.get()
    # Переспрос обработчика — факт хода, даже если тот же вопрос затем ещё раз поставил общий путь (переспрос выбора
    # действия с записью отвечает через `orchestrator._extract_pending_response`, тем же текстом).
    reasks = [item.asks for item in _ASKED.get() if item.question is current.question and item.reask]
    if reasks:
        asks = max(reasks)
    elif prev is not None and prev.get("type") == current.question.value and prev.get("question") == question:
        asks = int(prev["asks"]) + 1
    else:
        asks = 1
    expects = {
        "kind": current.kind,
        "type": current.question.value,
        "question": question,
        "options": list(current.options),
        "asks": asks,
    }
    state.last_entities[EXPECTS_KEY] = expects
    turn_telemetry.note(tu_expects_kind=current.kind, tu_expects_type=current.question.value, tu_expects_asks=asks)
    return expects


def without(entities: dict[str, Any]) -> dict[str, Any]:
    """Сущности сессии для инструментов — без поля ожидания (тот же объект, если поля нет).

    Планировщик отдаёт инструментам всё состояние, а их `entities_used` уходит в промпт свободного ответа LLM
    (`renderer._final_prompt`): текст вопроса бота поменял бы промпт и ответ — наблюдатель обязан быть невидим.
    """

    if EXPECTS_KEY not in entities:
        return entities
    return {key: value for key, value in entities.items() if key != EXPECTS_KEY}


def reask_limit_mode() -> str:
    """`[MESSENGER_ROUTER] reask_limit`: `off` (по умолчанию, как было) | `on`; нет ключа или мусор — `off`."""

    raw = str(getattr(_cfg, "MR_REASK_LIMIT", "off") or "").strip().lower()
    return raw if raw in REASK_LIMIT_MODES else "off"


def over_limit() -> Asked | None:
    """Переспрос этого хода, который под `reask_limit = on` прозвучал бы третий раз подряд (D2.4), или None."""

    if reask_limit_mode() != "on":
        return None
    return next((item for item in _ASKED.get() if item.reask and item.asks > MAX_ASKS), None)


def reask_limit_text(current: Asked) -> str:
    """Третьего раза нет: какой ответ был нужен — и оффер оператора одним вопросом.

    Вопрос «да/нет» с припиской «Перевести на оператора?» был бы ловушкой: «да» неясно к какому из двух. Поэтому
    прежний вопрос снимается, а текст называет, чего бот ждал.
    """

    if current.kind == PICK and current.options:
        need = "нужно было выбрать: " + " или ".join(f"«{option}»" for option in current.options)
    else:
        need = "нужен был ответ «да» или «нет»"
    return f"Не понял ответ — {need}. {UNCLEAR_OPERATOR_OFFER_TEXT}"
