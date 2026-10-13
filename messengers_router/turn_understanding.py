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

Разбор реплики на фразы (L-15b, шаг 4, `[MESSENGER_ROUTER] turn_parse = off | shadow | on`, по
умолчанию off) запускается здесь же, после понимания, и от его режима не зависит: под `shadow` —
фоновой задачей во время хода, только телеметрия (`tu_parse_status`, `tu_parse_src`, `tu_parse_n`,
`tu_parse_cands`, `tu_parse_ms`, `tu_parse_agree` — согласие с разрезом правил) и `debug.turn_parse`;
под `on` — ждём здесь, и понимание читает его (шаг 5): услуги хода — фразы разбора (`_from_phrases`), его
читатели (`acting_list()`) идут за разбором, а не за разрезом правил; телеметрия `tu_rule = parse_<статус>`.
Разбор не решён — понимание правил, как раньше.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass
from typing import Any

from . import turn_parse, turn_telemetry
from .policies import detect_address_intent, detect_nonbookable_walkin_intent, detect_prepare_intent, detect_price_intent
from .runtime_config import config as _cfg
from .services import _turn_phrases
from .services._prices_helpers import (
    _catalog_rows,
    _multi_price_items,
    is_readiness_question,
    price_list_services,
    strip_readiness_phrasing,
)

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")

# Метки, на которых шаг 1 строит понимание: цена и подбор анализов (П1 переклеивает его в цену).
_PRICE_TURN_LABELS = frozenset({"PRICE", "TEST_ASSIST"})


@dataclass(frozen=True)
class ServiceMention:
    """Упоминание услуги. Шаг 1: кандидат один — строка прайса, которую покажет корзина. Под `turn_parse =
    on` (L-15b, шаг 5) — фраза разбора: кандидаты — строки сильного совпадения по данным МИС (у слабой —
    нет), строку слабой выбирает корзина."""

    phrase: str
    candidates: tuple[str, ...]
    strength: str  # list_split | list_segment (как найдено); фраза разбора — strong | weak
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
    """Построить понимание ценового хода (кроме `off`) и записать его в ход; затем — разбор реплики на
    фразы (`turn_parse`, L-15b шаг 4), независимо от режима понимания."""

    if not is_price_turn(label, pending) or not str(user_text or "").strip():
        return None
    tu: TurnUnderstanding | None = None
    if mode() != "off":
        try:
            tu = await asyncio.to_thread(understand_price_turn, user_text)
        except Exception:  # понимание — новый слой; его сбой не ломает ход, старый путь остаётся
            logger.warning("turn_understanding_failed", exc_info=True)
            turn_telemetry.note(tu_rule="error")
        else:
            _CURRENT.set(tu)
            turn_telemetry.note(tu_list=len(tu.services), tu_facet=tu.facet, tu_rule=tu.services_rule)
    await _observe_phrases(user_text, tu)
    read = _from_phrases(tu, user_text) if tu is not None else None
    if read is not None:
        tu = read
        _CURRENT.set(tu)
        turn_telemetry.note(tu_list=len(tu.services), tu_rule=tu.services_rule)
    return tu


def _from_phrases(tu: TurnUnderstanding, text: str) -> TurnUnderstanding | None:
    """Под `turn_parse = on` (L-15b, шаг 5) услуги хода — фразы разбора, а не разрез правил: одна услуга
    или список решили данные МИС или выбор LLM. Читатели понимания (`acting_list()` — планировщик,
    подстановка каталога, грундер, дозаполнение) берут его без правок. Разбора нет или он не решён —
    None, понимание правил остаётся."""

    parse = turn_parse.acting_for(text)
    if parse is None:
        return None
    services = tuple(
        ServiceMention(
            phrase=item.text,
            candidates=tuple(dict.fromkeys(str(row.get("serviceName") or "").strip() for row in item.rows)),
            strength=str(item.strength.value),
            chosen=item.text,
        )
        for item in parse.items
    ) if parse.is_list else ()
    return TurnUnderstanding(services=services, services_rule=f"parse_{parse.status}", facet=tu.facet, facet_rule=tu.facet_rule)


async def _observe_phrases(user_text: str, tu: TurnUnderstanding | None) -> None:
    """Разбор реплики на фразы — после понимания (сегментацию сплошного списка генератор берёт из памяти
    хода, а не считает второй раз). Тень — фоновой задачей во время хода (её дождётся `drain` в конце
    хода); `on` — ждём здесь же: решение тратит бюджет хода."""

    parse_mode = turn_parse.mode()
    if parse_mode == "off":
        return
    shadow = parse_mode == "shadow"
    facet = tu.facet if tu is not None else price_turn_facet(user_text)[0]
    started = turn_parse.start_phrases(lambda: _parse_phrases(user_text, facet=facet, tu=tu, shadow=shadow), reply=user_text)
    if started is not None and not shadow:
        await turn_parse.phrases()


async def _parse_phrases(text: str, *, facet: str | None, tu: TurnUnderstanding | None, shadow: bool) -> turn_parse.PhraseParse:
    """Разбор и его след в телеметрии хода (`tu_parse_*` — без текста пациента). Тень память разбора
    хода только читает."""

    if shadow:
        turn_parse.read_only_memory()
    started = time.monotonic()
    agree: str | None = None
    try:
        catalog = await asyncio.to_thread(_catalog_rows, None)
        parse = await _turn_phrases.parse_turn_phrases(text, facet=facet, catalog=catalog, shadow=shadow)
        agree = await asyncio.to_thread(_agreement, parse, text, facet, tu, catalog)
    except Exception:  # разбор — новый слой: сбой не ломает ход
        logger.warning("turn_parse_failed", exc_info=True)
        parse = turn_parse.PhraseParse(reply=text, status=turn_parse.ParseStatus.ERROR.value, source="none", ms=int((time.monotonic() - started) * 1000))
    fields: dict[str, Any] = {
        "tu_parse_status": parse.status,
        "tu_parse_src": parse.source,
        "tu_parse_n": len(parse.items),
        "tu_parse_cands": parse.candidates,
        "tu_parse_ms": parse.ms,
    }
    if agree:
        fields["tu_parse_agree"] = agree
    turn_telemetry.note(**fields)
    return parse


def _agreement(parse: turn_parse.PhraseParse, text: str, facet: str | None, tu: TurnUnderstanding | None, catalog: list[dict[str, Any]]) -> str | None:
    """Разрез LLM против разреза правил: `same`; `split` — LLM видит несколько услуг, правила — одну;
    `merge` — наоборот; `other` — оба список, состав разный. Разбор не решён — None.

    Разрез правил — понимание хода (`services_rule`), его фрагменты — из памяти хода; понимание
    выключено — `price_list_services` той же реплики."""

    if not parse.decided:
        return None
    probe = strip_readiness_phrasing(text)[0] if facet == "readiness" else text
    rule = tu.services_rule if tu is not None else price_list_services(probe, rows=catalog)[1]
    if not rule.startswith("list_"):
        return "split" if parse.is_list else "same"
    if not parse.is_list:
        return "merge"
    items, _unrecognized = _multi_price_items(probe, catalog)
    rules = {_turn_phrases.bare(str(item.get("fragment") or item.get("service_name") or "")) for item in items}
    return "same" if rules == {_turn_phrases.bare(item.text) for item in parse.items} else "other"
