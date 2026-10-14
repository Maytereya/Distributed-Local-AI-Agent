"""Ответ на висящий вопрос бота силами LLM — клиент `pending_reply` (D2.3; дизайн `docs/DESIGN_2026-10-14_pending_reply.md`).

Было: «да / нет» на вопрос бота решают только правила — `recovery_policy.contextual_reply_kind`
(`policies.is_context_affirmative/negative`: якорные регулярки и целые фразы), подтверждение записи —
`policies.appointment_confirmation_transition`. «нет, спасибо», «да, отмените», «конечно, переведите», «наркоз не
нужен» правила не понимают (аудит 14.10, `docs/HANDOFF_2026-10-14_cloud_L14.md`, пункты 1–6): оффер оператора
снимается без перевода, отмена записи переспрашивается по кругу, «наркоз не нужен» получает памятку «с наркозом».

Станет (принципы 2 и 5): бот знает, какой вопрос задал (`_expects` в `state.last_entities` пишут места, где бот
спрашивает), а ответ на него понимает LLM закрытым выбором `choose()`: да / нет / пауза / другая тема / пункт
показанного списка; «ничего из этого» — обычный путь.

Этот шаг — только тень. Выключатель `[MESSENGER_ROUTER] pending_reply = off | shadow | on` (по умолчанию off):
- `off` — как было: модуль не делает ничего;
- `shadow` — один вызов `choose(..., shadow=True)` за ход: вне бюджета хода и предохранителя, очередь к LLM не
  дольше секунды (решение владельца 13.10, Q1 — как у тени разбора реплики); ответ пациенту прежний, состояние
  сессии тень не трогает, пишет только телеметрию хода `tu_pending_*`;
- `on` — пока то же, что `shadow`: исполнение решения LLM (асимметрия по типам вопросов, выключатель на каждый тип,
  предел переспросов, остаток составного ответа — в NLU) — следующий шаг, после живого замера тени и встречного
  свипа. Тень не меняет ни одного ответа, поэтому `on` до реализации безопасен и не теряет замер.

Место — одно: `router.patient_routing_stream` после срочного, явного запроса оператора и охраны города, до разбора
висящего вопроса (предпроверка записи, обработчики висящих вопросов в оркестраторе). Срочное — раньше тени: «нет,
мне плохо, задыхаюсь» получает шаблон скорой, тень не зовётся. Нажатие кнопки снимает висящий вопрос — тень не
зовётся. Конец хода ждёт тень не дольше секунды (`drain` в обоих эндпоинтах), опоздавшую отменяет — исход `late`.

Быстрый путь без LLM: вся реплика — одна целая формула согласия или отказа из правил («да», «нет», «ок», «не
надо»: множества `policies._CONFIRM_YES_EXACT` / `_CONFIRM_NO_EXACT`, новых слов нет) — решают правила, исход
`fast_rules`. Самый частый ответ не занимает слот LLM.

В промпт (`choose_pending_reply`, две локации) — вопрос бота и реплика (данные хода в роли user, та же чистка, что у
реплики: в вопросе бывают слова пациента) и варианты, если бот их показал (`_expects.options`); без истории
диалога. ФИО пациента из состояния в вопрос не уходит (его печатает сводка записи).

Телеметрия хода — только ключи и числа:
- `tu_pending_type` — тип вопроса (`_expects.type`; не из контракта — `unknown`), `tu_pending_asks` — который раз
  бот его задаёт;
- `tu_pending_rules` — что решают правила сейчас (`yes` / `no` / `other`): подтверждение записи —
  `appointment_confirmation_transition`, остальные — `contextual_reply_kind`. Мягкое «да» вторичного оффера и
  составной цены (`_SECONDARY_SOFT_YES_RE`, `_COMPOUND_PRICE_SOFT_YES_RE`) здесь — `other`;
- `tu_pending_outcome` — исход тени: `yes` / `no` / `pause` / `new_topic` / `pick` / `none`; сбой — `unavailable`,
  `invalid`, `late`, `error`; без LLM — `fast_rules`; `_expects` не по контракту — `malformed`;
- `tu_pending_agree` — согласие с правилами, только когда LLM решила: `same` — тот же yes/no; `other` — правила
  решили yes/no, LLM — иначе; `rules_none` — правила не решили (`other`), LLM решила; `both_none` — не решили ни те,
  ни другая.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from . import turn_telemetry
from .choice import Choice, Option, Status, Task, choose
from .policies import _CONFIRM_NO_EXACT, _CONFIRM_YES_EXACT, _normalize_confirmation_text, appointment_confirmation_transition
from .recovery_policy import contextual_reply_kind
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")
EXPECTS_KEY = "_expects"

# Ответ на вопрос бота: исход или пункт показанного списка (не больше одного).
PENDING_TASK = Task(
    name="pending_reply",
    max_picks=1,
    outcomes=("yes", "no", "pause", "new_topic"),
    timeout_s=8,
    queue_timeout_ms=3000,
)
# Тень ждёт очередь к LLM не дольше секунды (`llm_slot` округляет 300 мс до своего минимума — 1 с; Q1 13.10).
_SHADOW_QUEUE_MS = 300
_SHADOW_TASK = replace(PENDING_TASK, queue_timeout_ms=_SHADOW_QUEUE_MS)
_MAX_QUESTION_LEN = 500  # контракт `_expects.question`
_MAX_OPTIONS = 20
_MAX_OPTION_LEN = 200
_NAME_MASK = "[ФИО]"


class Kind(StrEnum):
    """Чего ждёт бот (`_expects.kind`)."""

    YES_NO = "yes_no"
    PICK = "pick"


class QuestionType(StrEnum):
    """Тип вопроса бота (`_expects.type`) — для телеметрии и будущего выключателя на каждый тип."""

    OPERATOR_OFFER = "operator_offer"
    CATALOG_CONFIRM = "catalog_confirm"
    APPOINTMENT_CONFIRM = "appointment_confirm"
    APPOINTMENT_ABORT = "appointment_abort"
    APPOINTMENT_ACTION = "appointment_action"
    SECONDARY_OFFER = "secondary_offer"
    ANESTHESIA = "anesthesia"
    COMPOUND_PRICE = "compound_price"


class Outcome(StrEnum):
    """Исход тени в телеметрии хода (`tu_pending_outcome`)."""

    YES = "yes"
    NO = "no"
    PAUSE = "pause"
    NEW_TOPIC = "new_topic"
    PICK = "pick"
    NONE = "none"  # LLM: ничего из этого — обычный путь
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    LATE = "late"  # конец хода не дождался тени
    ERROR = "error"  # сбой кода тени
    FAST_RULES = "fast_rules"  # голый якорь — решают правила, LLM не звали
    MALFORMED = "malformed"  # `_expects` не по контракту


# Исходы, когда LLM решила (согласие с правилами считается только для них).
_DECIDED = frozenset({Outcome.YES, Outcome.NO, Outcome.PAUSE, Outcome.NEW_TOPIC, Outcome.PICK, Outcome.NONE})

# Задача тени этого хода: ставит `observe`, ждёт и отменяет `drain` (конец хода в эндпоинте).
_SHADOW: contextvars.ContextVar[asyncio.Future[None] | None] = contextvars.ContextVar("mr_pending_reply_shadow", default=None)


def mode() -> str:
    """`pending_reply`: off | shadow | on; иное — off."""

    raw = str(getattr(_cfg, "MR_PENDING_REPLY", "off") or "").strip().lower()
    return raw if raw in MODES else "off"


@functools.cache
def _warn_on_is_shadow() -> None:
    logger.warning("pending_reply = on не реализован: работает как shadow (ответ прежний, только телеметрия)")


# --------------------------------------------------------------------------- ожидание (контракт `_expects`)


@dataclass(frozen=True)
class Expectation:
    """Вопрос бота, который висит: тип, текст, показанные варианты, который раз задан."""

    kind: str
    type: str
    question: str
    options: tuple[str, ...] = ()
    asks: int = 1


def _flat(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def expectation(entities: Mapping[str, Any] | None) -> Expectation | None:
    """Ожидание из состояния сессии; нет или не по контракту (вид, пустой вопрос) — None.

    Тип не из контракта — `unknown` (тень всё равно спрашивает: вопрос и реплика у неё есть); варианты — только
    непустые строки; `asks` не число ≥ 1 — 1.
    """

    raw = entities.get(EXPECTS_KEY) if isinstance(entities, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    kind = str(raw.get("kind") or "").strip()
    question = _flat(raw.get("question"), _MAX_QUESTION_LEN)
    if kind not in {member.value for member in Kind} or not question:
        return None
    qtype = str(raw.get("type") or "").strip()
    shown = raw.get("options") if isinstance(raw.get("options"), (list, tuple)) else ()
    options = tuple(flat for item in shown if isinstance(item, str) and (flat := _flat(item, _MAX_OPTION_LEN)))
    asks = raw.get("asks")
    return Expectation(
        kind=kind,
        type=qtype if qtype in {member.value for member in QuestionType} else "unknown",
        question=question,
        options=options[:_MAX_OPTIONS],
        asks=asks if isinstance(asks, int) and not isinstance(asks, bool) and asks >= 1 else 1,
    )


def _without_patient_name(question: str, entities: Mapping[str, Any]) -> str:
    """ФИО пациента из состояния — не в промпт: сводка записи («Запись: <ФИО>, …») печатает его в вопросе."""

    name = _flat(entities.get("patient_name"), _MAX_QUESTION_LEN)
    return question.replace(name, _NAME_MASK) if len(name) >= 2 else question


# --------------------------------------------------------------------------- правила и согласие


def is_bare_reply(text: str) -> bool:
    """Вся реплика — одна целая формула согласия или отказа из правил («да», «нет», «ок», «не надо»): её решают
    правила, LLM не нужна. Множества — прежние (`policies`), новых слов нет."""

    norm = _normalize_confirmation_text(text)
    return norm in _CONFIRM_YES_EXACT or norm in _CONFIRM_NO_EXACT


def rules_kind(question_type: str, text: str) -> str:
    """Что решают правила сейчас: yes / no / other — подтверждение записи своей проверкой (только однозначное
    согласие), остальные вопросы — `contextual_reply_kind`."""

    if question_type == QuestionType.APPOINTMENT_CONFIRM:
        return appointment_confirmation_transition(text)
    return contextual_reply_kind(text)


def agreement(outcome: str, rules: str) -> str | None:
    """Решение LLM против правил (`tu_pending_agree`); LLM не решила — None."""

    if outcome not in _DECIDED:
        return None
    if rules in ("yes", "no"):
        return "same" if outcome == rules else "other"
    return "both_none" if outcome == Outcome.NONE else "rules_none"


def outcome_of(choice: Choice) -> Outcome:
    """Выбор модуля → исход тени."""

    if choice.status is Status.OUTCOME:
        return Outcome(choice.outcome)
    if choice.status is Status.PICKED:
        return Outcome.PICK
    if choice.status is Status.NONE_FIT:
        return Outcome.NONE
    return Outcome.INVALID if choice.status is Status.INVALID else Outcome.UNAVAILABLE


# --------------------------------------------------------------------------- тень хода


def observe(user_text: str, entities: Mapping[str, Any] | None, *, button_id: str = "") -> None:
    """Тень ответа на висящий вопрос — в одной точке хода (`patient_routing_stream`, после срочного и до разбора
    висящего вопроса). Ответ хода и состояние сессии не меняет; под `off` и на кнопке — ничего.

    :param user_text: реплика пациента
    :param entities: `state.last_entities` до разбора висящего вопроса (читается, не меняется)
    :param button_id: нажатая кнопка — тень не зовётся
    """

    current = mode()
    if current == "off" or str(button_id or "").strip():
        return
    if current == "on":
        _warn_on_is_shadow()
    stale = _SHADOW.get()
    if stale is not None and not stale.done():
        stale.cancel()  # тень прошлого хода, которую никто не дождался (ход шёл мимо эндпоинта)
    _SHADOW.set(None)
    text = str(user_text or "").strip()
    if not text or not isinstance(entities, Mapping) or EXPECTS_KEY not in entities:
        return
    expected = expectation(entities)
    if expected is None:
        turn_telemetry.note(tu_pending_outcome=Outcome.MALFORMED.value)
        return
    rules = rules_kind(expected.type, text)
    fields: dict[str, Any] = {"tu_pending_type": expected.type, "tu_pending_asks": expected.asks, "tu_pending_rules": rules}
    if is_bare_reply(text):
        turn_telemetry.note(**fields, tu_pending_outcome=Outcome.FAST_RULES.value)
        return
    turn_telemetry.note(**fields)
    asked = _without_patient_name(expected.question, entities)
    _SHADOW.set(asyncio.ensure_future(_decide(text, asked, expected.options, rules)))


async def _decide(text: str, asked: str, options: tuple[str, ...], rules: str) -> None:
    """Решение тени и его след в телеметрии хода (`tu_pending_outcome`, `tu_pending_agree`)."""

    try:
        choice = await choose(_SHADOW_TASK, text, [Option(id=str(i), text=o) for i, o in enumerate(options, 1)], shadow=True, asked=asked)
        outcome = outcome_of(choice)
    except Exception:  # тень — новый слой: её сбой не ломает ход
        logger.warning("pending_reply_shadow_failed", exc_info=True)
        outcome = Outcome.ERROR
    fields: dict[str, Any] = {"tu_pending_outcome": outcome.value}
    agree = agreement(outcome, rules)
    if agree:
        fields["tu_pending_agree"] = agree
    turn_telemetry.note(**fields)


async def drain(grace_s: float) -> None:
    """Конец хода: ждём тень не дольше `grace_s`, затем отменяем — исход `late`. Тени нет или она готова —
    возврат без ожидания (под `off` ход не меняется ни на такт)."""

    task = _SHADOW.get()
    if task is None:
        return
    _SHADOW.set(None)
    if task.done():
        return
    try:
        await asyncio.wait({task}, timeout=grace_s)
    finally:
        if not task.done():
            task.cancel()
    if not task.done():
        await asyncio.wait({task}, timeout=grace_s)  # отмена доходит до задачи — её записи раньше строки хода
    if task.cancelled():
        turn_telemetry.note(tu_pending_outcome=Outcome.LATE.value)
