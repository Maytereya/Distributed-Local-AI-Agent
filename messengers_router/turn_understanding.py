"""Понимание хода — ход (б), шаг 1 (L-14): какие услуги назвал пациент и о каком их свойстве спросил.

Было: решение «о чём реплика» после NLU принимали по буквам в нескольких местах заново. Список
«ферритин, глюкоза, холестерин» сводился к одной услуге раньше инструмента — подстановка каталога,
грундер и дозаполнение ответа на кнопку цены приземляли реплику целиком на одну строку
(«холестерин» → «Липидограмма»), а планировщик с известной услугой звал `service_bundle_info`, где
корзины нет (BUG-2026-10-05-PRICE-LIST-COLLAPSES-TO-ONE). Вопрос о сроке готовности переклеивался
в цену, потому что переклейка смотрела флаг доставки результата, а не свойство
(BUG-2026-10-05-READINESS-AS-PRICE).

Станет: одна функция строит объект один раз за ход, места, решавшие заново, его читают. Шаг 1 —
только метка «цена»: ход с меткой цены или подбора анализов и ответ на висящий вопрос цены.
Объект живёт один ход (contextvar), в состояние сессии не пишется. Одиночную услугу шаг 1 не
разбирает — её по-прежнему находит старый путь; `services` непусто только у списка.

Выключатель `[MESSENGER_ROUTER] turn_understanding_price = off | shadow | on`:
- `off` — объект не строится;
- `shadow` (по умолчанию) — строится и пишется в debug (`understanding`) и в телеметрию хода
  (`tu_list`, `tu_facet`, `tu_rule`); поведение не меняется;
- `on` — его читают (`acting()`): список не сводится к одной услуге (подстановка каталога, грундер,
  дозаполнение ответа на кнопку), планировщик отдаёт список в `price_info` с корзиной, подбор
  анализов из списка — цена (П1), вопрос о сроке — честный перевод на оператора (срока бот не
  знает: ни цены, ни сроков свободным текстом LLM). Какие читатели сработали — `tu_acted` в
  телеметрии и в debug.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass
from typing import Any

from . import turn_parse, turn_telemetry
from .policies import detect_address_intent, detect_nonbookable_walkin_intent, detect_prepare_intent, detect_price_intent
from .runtime_config import config as _cfg
from .services._prices_helpers import is_readiness_question, price_list_services, strip_readiness_phrasing

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")

# Метки, на которых шаг 1 строит понимание: цена и подбор анализов (П1 переклеивает его в цену).
_PRICE_TURN_LABELS = frozenset({"PRICE", "TEST_ASSIST"})


@dataclass(frozen=True)
class ServiceMention:
    """Упоминание услуги. Шаг 1: кандидат один — строка прайса, которую покажет корзина."""

    phrase: str
    candidates: tuple[str, ...]
    strength: str  # list_split | list_segment (как найдено); позже — exact | synonym_fork | fuzzy
    chosen: str | None


@dataclass(frozen=True)
class TurnUnderstanding:
    services: tuple[ServiceMention, ...] = ()
    services_rule: str = ""
    facet: str | None = None  # readiness | prepare | booking | where | price | None
    facet_rule: str = ""

    @property
    def is_list(self) -> bool:
        return len(self.services) >= 2

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(m.chosen for m in self.services if m.chosen)

    def as_debug(self) -> dict[str, Any]:
        return {
            "services": list(self.service_names),
            "services_rule": self.services_rule,
            "facet": self.facet,
            "facet_rule": self.facet_rule,
            "acted": list(_ACTED.get()),
        }


_CURRENT: contextvars.ContextVar[TurnUnderstanding | None] = contextvars.ContextVar("mr_turn_understanding", default=None)
_ACTED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("mr_turn_understanding_acted", default=())


def mode() -> str:
    raw = str(getattr(_cfg, "MR_TURN_UNDERSTANDING_PRICE", "shadow") or "").strip().lower()
    return raw if raw in MODES else "shadow"


def begin_turn() -> None:
    """Новый ход: понимание прошлого хода не читается (тесты и стенд гоняют ходы в одном цикле);
    память разбора реплики — новая, на этот ход (`turn_parse`, во всех режимах выключателя)."""

    _CURRENT.set(None)
    _ACTED.set(())
    turn_parse.begin()


def current() -> TurnUnderstanding | None:
    return _CURRENT.get()


def acting() -> TurnUnderstanding | None:
    """Понимание хода, если по нему разрешено действовать (`on`); в тени — None."""

    return _CURRENT.get() if mode() == "on" else None


def acting_list() -> TurnUnderstanding | None:
    tu = acting()
    return tu if tu is not None and tu.is_list else None


def note_acted(reader: str) -> None:
    """Читатель изменил поведение по пониманию хода — в debug и в телеметрию."""

    acted = _ACTED.get()
    if reader not in acted:
        acted = (*acted, reader)
        _ACTED.set(acted)
    turn_telemetry.note(tu_acted=list(acted))


def price_turn_facet(text: str) -> tuple[str | None, str]:
    """О каком свойстве услуги вопрос — существующими детекторами, без LLM."""

    if is_readiness_question(text) and not detect_price_intent(text):
        return "readiness", "readiness_phrasing"
    if detect_prepare_intent(text):
        return "prepare", "prepare_intent"
    if detect_nonbookable_walkin_intent(text):
        return "booking", "walkin_intent"
    if detect_address_intent(text):
        return "where", "address_intent"
    if detect_price_intent(text):
        return "price", "price_intent"
    return None, ""


def understand_price_turn(text: str) -> TurnUnderstanding:
    """Услуги и свойство ценовой реплики. Чистая функция: без состояния и без LLM.

    Список ищется в реплике без фразы о сроке: «когда будет готов анализ на ферритин» — одна
    услуга и свойство «срок», а не корзина из лишних слов.
    """

    facet, facet_rule = price_turn_facet(text)
    probe = strip_readiness_phrasing(text)[0] if facet == "readiness" else text
    names, rule = price_list_services(probe)
    mentions = tuple(ServiceMention(phrase=name, candidates=(name,), strength=rule, chosen=name) for name in names)
    return TurnUnderstanding(services=mentions, services_rule=rule, facet=facet, facet_rule=facet_rule)


def is_price_turn(label: str, pending: dict[str, Any] | None) -> bool:
    pending_label = (pending or {}).get("label") if isinstance(pending, dict) else None
    return label in _PRICE_TURN_LABELS or pending_label == "PRICE"


async def observe_price_turn(label: str, *, user_text: str, pending: dict[str, Any] | None) -> TurnUnderstanding | None:
    """Построить понимание ценового хода (кроме `off`) и записать его в ход."""

    if mode() == "off" or not is_price_turn(label, pending) or not str(user_text or "").strip():
        return None
    try:
        tu = await asyncio.to_thread(understand_price_turn, user_text)
    except Exception:  # понимание — новый слой; его сбой не ломает ход, старый путь остаётся
        logger.warning("turn_understanding_failed", exc_info=True)
        turn_telemetry.note(tu_rule="error")
        return None
    _CURRENT.set(tu)
    turn_telemetry.note(tu_list=len(tu.services), tu_facet=tu.facet, tu_rule=tu.services_rule)
    return tu
