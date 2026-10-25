"""Ответ на висящий вопрос бота — общие части одной функции `pending_answer` (класс 5; дизайны
`docs/DESIGN_2026-10-14_pending_reply.md`, `docs/DESIGN_2026-10-18_one_pending_answer.md`).

Было: «да / нет» на вопрос бота решали только правила — `recovery_policy.contextual_reply_kind`
(`policies.is_context_affirmative/negative`: якорные регулярки и целые фразы). «нет, спасибо», «да, отмените», «конечно,
переведите», «наркоз не нужен» правила не понимают (аудит 14.10, `docs/HANDOFF_2026-10-14_cloud_L14.md`, пункты 1–6).

Стало (принципы 2 и 5): бот знает, какой вопрос задал (`_expects` в `state.last_entities` пишут места, где бот
спрашивает), а ответ на него понимает LLM закрытым выбором `choose()` в одной функции `pending_answer`: да / нет / пауза /
другая тема / пункт показанного списка; «ничего из этого» — обычный путь. Сбой LLM — правила обработчика вопроса
(решение владельца Q1 18.10).

Выключатель `pending_reply` (off | shadow | on) снят 24.10 (неделя `on` на проде с деплоя 10, решение владельца 21.10,
STEP-2026-10-22-SWITCH-REMOVAL): тень, список исполняемых типов `pending_reply_types` и «подъём» ответа правил решением LLM
(`reply_kind_for`) ушли вместе с путём off. Здесь остались задача `PENDING_TASK`, контракт `_expects` (`expectation`) и
признак запасного пути вопроса о филиале (`note_rules_branch_answer` / `branch_reply`).

В промпт (`choose_pending_reply`, две локации) — вопрос бота и реплика (данные хода в роли user, та же чистка, что у
реплики: в вопросе бывают слова пациента) и варианты, если бот их показал (`_expects.options`); без истории
диалога. ФИО пациента из состояния в вопрос не уходит (его печатает сводка записи).
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .choice import Task

EXPECTS_KEY = "_expects"

# Ответ на вопрос бота: исход или пункт показанного списка (не больше одного).
PENDING_TASK = Task(
    name="pending_reply",
    max_picks=1,
    outcomes=("yes", "no", "pause", "new_topic"),
    timeout_s=8,
    queue_timeout_ms=3000,
)
_MAX_QUESTION_LEN = 500  # контракт `_expects.question`
_MAX_OPTIONS = 20
_MAX_OPTION_LEN = 200
_NAME_MASK = "[ФИО]"


class Kind(StrEnum):
    """Чего ждёт бот (`_expects.kind`)."""

    YES_NO = "yes_no"
    PICK = "pick"
    FREE = "free"  # свободный ответ — значение слота (под `pending_buttons = on`, решает только `pending_answer`)


class QuestionType(StrEnum):
    """Тип вопроса бота (`_expects.type`)."""

    OPERATOR_OFFER = "operator_offer"
    CATALOG_CONFIRM = "catalog_confirm"
    APPOINTMENT_ABORT = "appointment_abort"
    APPOINTMENT_ACTION = "appointment_action"
    SECONDARY_OFFER = "secondary_offer"
    ANESTHESIA = "anesthesia"
    COMPOUND_PRICE = "compound_price"
    SHOW_ALL = "show_all"
    BOOKING_SERVICE = "booking_service"
    APPOINTMENT_BRANCH = "appointment_branch"
    SLOT_VALUE = "slot_value"


# Ответ этого хода на вопрос о филиале, который узнали правила предпроверки записи (функция не решила — сбой LLM):
# {"text": реплика, тип вопроса: "rules", пункт: ""}; реплика — привязка к ходу.
_ACTED: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("mr_pending_reply_acted", default=None)
_BRANCH_PICK = "appointment_branch_pick"


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

    Тип не из контракта — `unknown`; варианты — только непустые строки; `asks` не число ≥ 1 — 1.
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


# --------------------------------------------------------------------------- запасной путь вопроса о филиале


def note_rules_branch_answer(user_text: str) -> None:
    """Запасной путь предпроверки записи (функция не решила): правила узнали ответ на вопрос о филиале — признак хода
    с исходом `rules`, чтобы prelock и заземление прочли его так же, как решение функции (`pending_answer.branch_answer`)."""

    text = str(user_text or "").strip()
    if branch_reply(text) is None:
        _ACTED.set({"text": text, QuestionType.APPOINTMENT_BRANCH.value: "rules", _BRANCH_PICK: ""})


def branch_reply(user_text: str) -> tuple[str, str] | None:
    """Ответ этого хода на вопрос «Какой филиал вам удобен?», узнанный правилами: (`rules`, ""); не ответ — None."""

    acted = _ACTED.get() or {}
    if acted.get("text") != str(user_text or "").strip():
        return None
    outcome = acted.get(QuestionType.APPOINTMENT_BRANCH.value)
    if not outcome:
        return None
    return outcome, str(acted.get(_BRANCH_PICK) or "")
