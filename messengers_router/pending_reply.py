"""Ответ на висящий вопрос бота силами LLM — клиент `pending_reply` (D2.3; дизайн `docs/DESIGN_2026-10-14_pending_reply.md`).

Было: «да / нет» на вопрос бота решают только правила — `recovery_policy.contextual_reply_kind`
(`policies.is_context_affirmative/negative`: якорные регулярки и целые фразы), подтверждение записи —
`policies.appointment_confirmation_transition`. «нет, спасибо», «да, отмените», «конечно, переведите», «наркоз не
нужен» правила не понимают (аудит 14.10, `docs/HANDOFF_2026-10-14_cloud_L14.md`, пункты 1–6): оффер оператора
снимается без перевода, отмена записи переспрашивается по кругу, «наркоз не нужен» получает памятку «с наркозом».

Станет (принципы 2 и 5): бот знает, какой вопрос задал (`_expects` в `state.last_entities` пишут места, где бот
спрашивает), а ответ на него понимает LLM закрытым выбором `choose()`: да / нет / пауза / другая тема / пункт
показанного списка; «ничего из этого» — обычный путь.

Выключатель `[MESSENGER_ROUTER] pending_reply = off | shadow | on` (по умолчанию off):
- `off` — как было: модуль не делает ничего;
- `shadow` — один вызов `choose(..., shadow=True)` за ход: вне бюджета хода и предохранителя, очередь к LLM не
  дольше секунды (решение владельца 13.10, Q1 — как у тени разбора реплики); ответ пациенту прежний, состояние
  сессии тень не трогает, пишет только телеметрию хода `tu_pending_*`;
- `on` — с 18.10 ответ на висящий вопрос решает одна функция `pending_answer` (класс 5, все типы разом, тень не
  зовётся, `pending_reply_types` не читается); этот модуль даёт ей задачу `PENDING_TASK`, разбор `_expects` и
  запасные правила. Ниже — прежнее поведение `on` (до одной функции), его читают только `reply_kind_for` /
  `branch_reply` вне `on`: тип вопроса из `[MESSENGER_ROUTER] pending_reply_types` (через запятую; по умолчанию пусто) решает LLM в
  ходе, из его бюджета, и решение исполняется — с асимметрией безопасности по типу (дизайн, таблица «Асимметрия»;
  решение владельца 15.10). Прочие типы — тень, как под `shadow`. Исполняются два типа:
  `operator_offer` — LLM может только ПОДНЯТЬ ответ до «да» (перевод на оператора: безопасное направление); её
  «нет», «пауза», «другая тема» и сбой ничего не меняют — решают правила, как было (ложное «нет» опасно);
  `show_all` (подсказка «…напишите: "все"») — выбор пункта «все» или «да» от LLM поднимает ответ до «показать все»
  (полный список того же семейства: та же цена, только больше строк); прочие исходы и сбой — ответ правил.
  Решение хода читает обработчик вопроса (`reply_kind_for`).

Место — одно: `router.patient_routing_stream` после срочного, явного запроса оператора и охраны города, до разбора
висящего вопроса (предпроверка записи, обработчики висящих вопросов в оркестраторе). Срочное — раньше тени: «нет,
мне плохо, задыхаюсь» получает шаблон скорой, тень не зовётся. Нажатие кнопки снимает висящий вопрос — тень не
зовётся.

Ход тень не ждёт (15.10): конец хода отцепляет её (`detach` в обоих эндпоинтах), решение дописывается после ответа
отдельной строкой `pending_shadow` (`turn_telemetry.side_finish`, журнал `pending_shadow_<ГГГГММДД>.jsonl`, привязка к
ходу — `sess` + `tid`). Раньше конец хода ждал тень секунду и отменял опоздавшую: 26 из 58 решений замера 14.10
терялись как `late` — генерация дольше секунды. Теперь `late` — только сверх потолка `_SHADOW_CAP_S`. Стенд и тесты
дожидаются тени после замера хода (`settle`) и сливают строки (`turn_telemetry.merge_side`).

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
  `invalid`, `late` (сверх потолка), `error`; без LLM — `fast_rules`; `_expects` не по контракту — `malformed`.
  Исход решения LLM (и `tu_pending_agree`) — в строке `pending_shadow`, `fast_rules` и `malformed` — в строке хода;
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
from .expects import SHOW_ALL_OPTIONS
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
# Потолок жизни тени после ответа: очередь (≤1 с) + генерация (`timeout_s` = 8 с) с запасом; сверх — `late`.
_SHADOW_CAP_S = 15.0
_SIDE_KIND = "pending_shadow"
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
    SHOW_ALL = "show_all"
    BOOKING_SERVICE = "booking_service"
    APPOINTMENT_BRANCH = "appointment_branch"


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
    LATE = "late"  # тень не уложилась в потолок `_SHADOW_CAP_S`
    ERROR = "error"  # сбой кода тени
    FAST_RULES = "fast_rules"  # голый якорь — решают правила, LLM не звали
    MALFORMED = "malformed"  # `_expects` не по контракту


# Исходы, когда LLM решила (согласие с правилами считается только для них).
_DECIDED = frozenset({Outcome.YES, Outcome.NO, Outcome.PAUSE, Outcome.NEW_TOPIC, Outcome.PICK, Outcome.NONE})

# Задача тени этого хода: ставит `observe`, отцепляет `detach` (конец хода в эндпоинте).
_SHADOW: contextvars.ContextVar[asyncio.Future[None] | None] = contextvars.ContextVar("mr_pending_reply_shadow", default=None)
# Решение хода, которое исполняет обработчик вопроса: {"text": реплика, тип вопроса: исход} (под `on`, типы
# `acting_types`); реплика — привязка к ходу: обработчик, вызванный мимо точки решения, его не увидит.
_ACTED: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("mr_pending_reply_acted", default=None)
# Отцеплённые тени: сильные ссылки, пока задача жива (иначе цикл событий может её собрать), — их ждёт `settle`.
_DETACHED: set[asyncio.Future[None]] = set()


def mode() -> str:
    """`pending_reply`: off | shadow | on; иное — off."""

    raw = str(getattr(_cfg, "MR_PENDING_REPLY", "off") or "").strip().lower()
    return raw if raw in MODES else "off"


# Типы, решение LLM на которые умеет исполнять обработчик (асимметрия задана в нём, `reply_kind_for`).
_ACTABLE = frozenset({"operator_offer", "show_all", "appointment_branch"})


def acting_types() -> frozenset[str]:
    """Типы вопросов, на которые под `on` решение LLM исполняется в ходе (`pending_reply_types` ∩ исполнимые)."""

    if mode() != "on":
        return frozenset()
    raw = str(getattr(_cfg, "MR_PENDING_REPLY_TYPES", "") or "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip()) & _ACTABLE


@functools.cache
def _warn_unknown_types(raw: str) -> None:
    logger.warning("pending_reply_types: исполняется только %s, прочее — тень: %s", sorted(_ACTABLE), raw)


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
    согласие), подсказка «напишите: "все"» — определением пути цены (`yes` — правила покажут все; «да» здесь не
    «все»: его читает обработчик вторичного оффера), остальные вопросы — `contextual_reply_kind`."""

    if question_type == QuestionType.APPOINTMENT_CONFIRM:
        return appointment_confirmation_transition(text)
    if question_type == QuestionType.SHOW_ALL:
        # Импорт в функции: модуль хода не тянет сервисный слой при своём импорте (как `classifier`).
        from .services._prices_helpers import _is_price_show_all_request

        if _is_price_show_all_request(text):
            return "yes"
        return "no" if contextual_reply_kind(text) == "no" else "other"
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
    _keep(_SHADOW.get())  # тень прошлого хода, которую не отцепил эндпоинт (ход шёл мимо него), — дописывается сама
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
    _SHADOW.set(asyncio.ensure_future(_decide(text, asked, expected.options, rules, turn_telemetry.current_ref())))


async def _decide(text: str, asked: str, options: tuple[str, ...], rules: str, ref: tuple[str, str] | None) -> None:
    """Решение тени — строкой `pending_shadow` после ответа (`tu_pending_outcome`, `tu_pending_agree`, вызов LLM и
    решение выбора); хода нет (телеметрия выключена, ход мимо эндпоинта) — в поля текущего хода, как раньше."""

    token = turn_telemetry.side_begin(ref) if ref is not None else None
    cancelled = False
    try:
        choice = await asyncio.wait_for(
            choose(_SHADOW_TASK, text, [Option(id=str(i), text=o) for i, o in enumerate(options, 1)], shadow=True, asked=asked),
            _SHADOW_CAP_S,
        )
        outcome = outcome_of(choice)
    except asyncio.TimeoutError:
        outcome = Outcome.LATE
    except asyncio.CancelledError:  # остановка сервиса: решение не получено, строку всё равно пишем
        outcome, cancelled = Outcome.LATE, True
    except Exception:  # тень — новый слой: её сбой не ломает ход
        logger.warning("pending_reply_shadow_failed", exc_info=True)
        outcome = Outcome.ERROR
    fields: dict[str, Any] = {"tu_pending_outcome": outcome.value}
    agree = agreement(outcome, rules)
    if agree:
        fields["tu_pending_agree"] = agree
    if token is not None:
        turn_telemetry.side_finish(token, _SIDE_KIND, **fields)
    else:
        turn_telemetry.note(**fields)
    if cancelled:
        raise asyncio.CancelledError


async def observe_or_act(user_text: str, entities: Mapping[str, Any] | None, *, button_id: str = "") -> None:
    """Точка хода (`patient_routing_stream`): тип висящего вопроса из `acting_types()` — решение LLM в ходе и его
    исход для обработчика (`reply_kind_for`); иначе — тень (`observe`), как было. Состояние сессии не меняет."""

    _ACTED.set(None)
    text = str(user_text or "").strip()
    acting = acting_types()
    if raw := str(getattr(_cfg, "MR_PENDING_REPLY_TYPES", "") or "").strip():
        if mode() == "on" and {part.strip() for part in raw.split(",") if part.strip()} - _ACTABLE:
            _warn_unknown_types(raw)
    expected = expectation(entities) if acting and text and not str(button_id or "").strip() else None
    if expected is None or expected.type not in acting:
        observe(user_text, entities, button_id=button_id)
        return
    _keep(_SHADOW.get())
    _SHADOW.set(None)
    rules = rules_kind(expected.type, text)
    fields: dict[str, Any] = {"tu_pending_type": expected.type, "tu_pending_asks": expected.asks, "tu_pending_rules": rules}
    if is_bare_reply(text) or rules == "yes":
        # Голый якорь или правила уже говорят «да» (перевод) — LLM не нужна: поднимать нечего.
        turn_telemetry.note(**fields, tu_pending_outcome=Outcome.FAST_RULES.value)
        return
    asked = _without_patient_name(expected.question, entities or {})
    try:
        choice = await choose(PENDING_TASK, text, [Option(id=str(i), text=o) for i, o in enumerate(expected.options, 1)], asked=asked)
        outcome = outcome_of(choice)
    except Exception:  # новый слой: его сбой — правила, как было
        logger.warning("pending_reply_act_failed", exc_info=True)
        outcome = Outcome.ERROR
    fields["tu_pending_outcome"] = outcome.value
    if agree := agreement(outcome, rules):
        fields["tu_pending_agree"] = agree
    if expected.type == QuestionType.APPOINTMENT_BRANCH and outcome in _DECIDED:
        # Вопрос о филиале (класс 5, решение владельца 18.10): решение LLM исполняет предпроверка записи — любой исход,
        # не только «да»: «другая тема» — вопрос об отмене, прочее — запись продолжается (`branch_reply`).
        picked = _picked_option(expected, choice) if outcome is Outcome.PICK else ""
        _ACTED.set({"text": text, expected.type: outcome.value, _BRANCH_PICK: picked})
        fields["tu_pending_acted"] = outcome.value
    elif _lifts(expected, outcome, choice if outcome is Outcome.PICK else None):
        _ACTED.set({"text": text, expected.type: Outcome.YES.value})
        fields["tu_pending_acted"] = outcome.value
    turn_telemetry.note(**fields)


_BRANCH_PICK = "appointment_branch_pick"


def _picked_option(expected: Expectation, picked: Choice) -> str:
    shown = dict(enumerate(expected.options, 1))
    return next((shown[int(i)] for i in picked.ids if str(i).isdigit() and int(i) in shown), "")


def note_rules_branch_answer(user_text: str) -> None:
    """Запасной путь предпроверки записи (LLM не решала): правила узнали ответ на вопрос о филиале — тот же признак
    хода, что у решения LLM (исход `rules`), чтобы prelock и заземление прочли его одинаково."""

    text = str(user_text or "").strip()
    if branch_reply(text) is None:
        _ACTED.set({"text": text, QuestionType.APPOINTMENT_BRANCH.value: "rules", _BRANCH_PICK: ""})


def branch_reply(user_text: str) -> tuple[str, str] | None:
    """Ответ этого хода на вопрос «Какой филиал вам удобен?»: (исход, выбранный вариант или ""). Исход — решение LLM
    или `rules` (запасной путь предпроверки, `note_rules_branch_answer`); не ответ — None."""

    acted = _ACTED.get() or {}
    if acted.get("text") != str(user_text or "").strip():
        return None
    outcome = acted.get(QuestionType.APPOINTMENT_BRANCH.value)
    if not outcome:
        return None
    return outcome, str(acted.get(_BRANCH_PICK) or "")


def _lifts(expected: Expectation, outcome: Outcome, picked: Choice | None) -> bool:
    """Поднимает ли решение LLM ответ правил (асимметрия по типу): оффер оператора — только «да»; подсказка «все» —
    «да» или выбранный пункт «все» (единственный пункт поля; иной пункт — нет)."""

    if outcome is Outcome.YES:
        return True
    if expected.type != QuestionType.SHOW_ALL or picked is None:
        return False
    shown = dict(enumerate(expected.options, 1))
    return any(shown.get(int(i)) in SHOW_ALL_OPTIONS for i in picked.ids if str(i).isdigit())


def reply_kind_for(question_type: str, rules: str, user_text: str) -> str:
    """Что исполняет обработчик вопроса `question_type`: решение правил, поднятое решением LLM этого хода.

    Асимметрия (дизайн, «Асимметрия безопасности по типу вопроса»): оффер оператора — LLM поднимает до «да»
    (перевод), подсказка «все» — до «показать все»; «нет» и прочее от LLM ответ правил не меняют."""

    acted = _ACTED.get() or {}
    if acted.get("text") != str(user_text or "").strip():
        return rules
    if question_type in _ACTABLE and acted.get(question_type) == Outcome.YES.value:
        return "yes"
    return rules


def _keep(task: asyncio.Future[None] | None) -> None:
    if task is not None and not task.done():
        _DETACHED.add(task)
        task.add_done_callback(_DETACHED.discard)


def detach() -> None:
    """Конец хода: тень этого хода больше не его забота — ход её не ждёт, решение она допишет сама. Тени нет —
    ничего (под `off` ход не меняется ни на такт)."""

    task = _SHADOW.get()
    _SHADOW.set(None)
    _keep(task)


async def settle(timeout_s: float) -> None:
    """Дождаться отцеплённых теней не дольше `timeout_s` — для стенда и тестов: после замера хода, до чтения его
    телеметрии. На проде не зовётся."""

    pending = {task for task in _DETACHED if not task.done()}
    if pending:
        await asyncio.wait(pending, timeout=timeout_s)
