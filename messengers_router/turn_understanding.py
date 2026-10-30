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
разбирал — её находит старый путь. С L-14 шаг 0 (14.10) у одиночной услуги тоже есть упоминание
(`services` из одного `ServiceMention`, `source`/`strength` по данным МИС) — только наблюдатель: читатели
действуют по `acting_list()`/`is_list` (список — ≥2), ответы не меняются; телеметрия `tu_list` — по-прежнему
размер списка (0 у одиночной), упоминание — `tu_mention_src`, `tu_mention_strength`.

Выключатель `[MESSENGER_ROUTER] turn_understanding_price = off | shadow | on`:
- `off` — объект не строится;
- `shadow` (запасное при неизвестном значении) — строится и пишется в debug (`understanding`) и в
  телеметрию хода (`tu_list`, `tu_facet`, `tu_rule`); поведение не меняется;
- `on` (умолчание `config.py` — как на проде, STEP-2026-10-15-DEFAULTS-B) — его читают (`acting()`):
  список не сводится к одной услуге (подстановка каталога, грундер, дозаполнение ответа на кнопку),
  планировщик отдаёт список в `price_info` с корзиной, подбор анализов из списка — цена (П1), вопрос
  о сроке — честный перевод на оператора (срока бот не знает: ни цены, ни сроков свободным текстом
  LLM). Какие читатели сработали — `tu_acted` в телеметрии и в debug.

Разбор реплики на фразы (L-15b, `turn_parse`) — единственный режим с T7 шага 3б (16.10; режимы off/shadow сняты):
запускается здесь же и ход его ждёт; понимание читает его (шаг 5): услуги хода — фразы разбора (`_from_phrases`), его
читатели (`acting_list()`) идут за разбором, а не за разрезом правил; телеметрия `tu_parse_status`, `tu_parse_src`,
`tu_parse_n`, `tu_parse_cands`, `tu_parse_ms`, `tu_rule = parse_<статус>`; фразы — только в `debug.turn_parse`.
Разбор не годится для действия (`empty`, `too_many`, `error`) — понимание правил, как раньше.

При нетерпеливых правилах (`parse_agree_telemetry = on`) правила понимания считаются на каждом ценовом ходе
параллельно выбору фраз (O3b) — их разрез нужен согласию `tu_parse_agree`. Выключатель
`[MESSENGER_ROUTER] parse_agree_telemetry = on | off` (T7, шаг 4; по умолчанию off с 16.10, решение владельца): `off` —
согласия нет, правила считаются лениво (`_observe_parse_on_lazy`) —
только когда разбор не годится для действия или решён одной слабой фразой; ответы те же, экономия — CPU хода.

Тип вопроса хода (`facet`) — деплой 22, шаг 2 (STEP-2026-10-30-FACET-BY-LLM). Выключатель
`[MESSENGER_ROUTER] facet_by_llm = off | on` (умолчание off — как было: `facet` ставят детекторы `price_turn_facet`). Под `on`
его решает LLM — одно решение на ход (`turn_facet`, задача `turn_facet`, промпт в двух локациях): цена / есть ли / где-когда /
нужна ли запись / срок / подготовка / просто назвал / ни одно (`FACETS`; правила дают подмножество тех же имён). На ходе цены
вопрос задаётся вместе с разбором реплики и пишется в `TurnUnderstanding.facet` (`facet_src = llm`); на других ходах — первым
читателем, которому он нужен (переклейки после цены и подготовки, обзор названия): объекта понимания там пока нет, решение
хода хранит `_FACET`. Сбой LLM, лимит хода, нет слов пациента — как было: `facet` правил (`facet_src = fail | rules`), читатели
идут своим прежним путём. Телеметрия — `tu_facet`, `tu_facet_src`, `tu_facet_fail` (причина сбоя: `budget` — лимит хода).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field, replace
from collections.abc import Awaitable
from typing import Any, Sequence

from . import shown_context, turn_names, turn_parse, turn_telemetry
from .choice import NO_WORDS, Option, Status, Task, choose
from .policies import detect_address_intent, detect_nonbookable_walkin_intent, detect_prepare_intent, detect_price_intent
from .runtime_config import config as _cfg
from .services import _turn_phrases, _turn_spans
from .services._biomaterial import _SPEECH_WRAPPER
from .services._strong_match import strong_rows
from .services._prices_helpers import (
    _catalog_rows,
    _extract_price_service_from_query,
    _multi_price_items,
    is_readiness_question,
    price_list_services,
    strip_readiness_phrasing,
)

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")

# Метки, на которых шаг 1 строит понимание: цена и подбор анализов (П1 переклеивает его в цену).
_PRICE_TURN_LABELS = frozenset({"PRICE", "TEST_ASSIST"})

# Речевая обвязка с краёв фразы одиночной услуги (L-14, шаг 2): слова вопроса о цене и вежливость
# (`_turn_phrases._WRAPPER`) и речевая обвязка слоя синонимов МИС (`_SPEECH_WRAPPER`: «а», «сдать», «тест»).
# Новых слов нет — оба списка лингвистические и уже служат тем же: отделить название от оформления просьбы.
_MENTION_EDGE = _turn_phrases._WRAPPER | _SPEECH_WRAPPER


@dataclass(frozen=True)
class ServiceMention:
    """Упоминание услуги. Шаг 1: кандидат один — строка прайса, которую покажет корзина. Под `turn_parse =
    on` (L-15b, шаг 5) — фраза разбора: кандидаты — строки сильного совпадения по данным МИС (у слабой —
    нет), строку слабой выбирает корзина. Одиночная услуга (L-14, шаг 0) — фраза пациента, кандидаты —
    строки сильного совпадения по данным МИС, `chosen` — название, если оно одно (иначе выбора нет)."""

    phrase: str
    candidates: tuple[str, ...]
    strength: str  # list_split | list_segment (как найдено); фраза разбора и одиночная — strong | weak
    chosen: str | None
    source: str = "rules"  # rules — правила понимания; parse — разбор реплики (turn_parse)


@dataclass(frozen=True)
class Frame:
    """Рамка кнопки хода (шаг 0, контракт полного понимания): `kind` — `press` (лист нажат на этом ходе), `question` (реплика —
    ответ на вопрос кнопки), `leaf` (реплика после листа без своего вопроса), `address` (реплика после адресного листа);
    `held` — тема хода — тема кнопки (рамку не перебили)."""

    button: str
    kind: str
    topic: str = ""
    held: bool = False


@dataclass(frozen=True)
class Expects:
    """Висящий вопрос бота и ответ на него (шаг 0): тип вопроса (`_expects.type`), исход одной функции ответа
    (`pending_answer.Outcome`) и кто решил. Текста вопроса здесь нет — в нём бывает ФИО пациента."""

    type: str
    outcome: str = ""
    source: str = ""


@dataclass(frozen=True)
class TurnEntities:
    """Сущности хода (шаг 0 — копия решения хода, не новое решение): позже их строит разбор хода (шаг 5)."""

    service: str = ""
    doctor: str = ""
    specialty: str = ""
    branch: str = ""
    city: str = ""
    date: str = ""

    def present(self) -> list[str]:
        return [name for name in ("service", "doctor", "specialty", "branch", "city", "date") if getattr(self, name)]


@dataclass(frozen=True)
class TurnUnderstanding:
    services: tuple[ServiceMention, ...] = ()
    services_rule: str = ""
    facet: str | None = None  # тип вопроса: одно из `FACETS` или None (правила не нашли)
    facet_rule: str = ""
    facet_src: str = ""  # llm | rules | fail (под `facet_by_llm = on`); пусто — решали только правила (off)
    reply: str = ""  # реплика хода, по которой построено понимание
    # Названа ли в реплике услуга — одно решение хода (U6, ход (б), этап 2): ставит производитель под `turn_understanding_one`
    # (`_decide_named`), планировщик цены читает. None — не решено (выключатель off): читатель идёт прежним путём.
    names_service: bool | None = None
    # Контракт полного понимания (ход (б), шаг 0, консилиум 30.10): тема хода, рамка кнопки, висящий вопрос и ответ, сущности.
    # Заполняет только `observe_turn` (полное понимание, `full()`); у понимания цены (`current()`) — None, его читатели не меняются.
    intent: TurnIntent | None = None
    frame: Frame | None = None
    expects: Expects | None = None
    entities: TurnEntities | None = None

    def for_text(self, text: str) -> bool:
        """Читатель берёт понимание, только если его вход — реплика хода (как `PhraseParse.for_text`):
        дозаполнение и контекст прошлого хода — не она."""

        return bool(self.reply) and str(text or "") == self.reply

    @property
    def is_list(self) -> bool:
        return len(self.services) >= 2

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(m.chosen for m in self.services if m.chosen)

    @property
    def single(self) -> ServiceMention | None:
        """Упоминание одиночной услуги (L-14, шаг 0) — наблюдатель; у списка — None."""

        return self.services[0] if len(self.services) == 1 else None

    @property
    def list_size(self) -> int:
        """Размер списка для телеметрии `tu_list`: 0 у одиночной услуги, как до шага 0."""

        return len(self.services) if self.is_list else 0

    def as_debug(self) -> dict[str, Any]:
        out = {
            "services": list(self.service_names),
            "services_rule": self.services_rule,
            "facet": self.facet,
            "facet_rule": self.facet_rule,
            "acted": list(_ACTED.get()),
        }
        if self.facet_src:
            out["facet_src"] = self.facet_src
        return out


@dataclass
class TurnIntent:
    """Тема хода — ход (б), этап 1 (STEP-2026-10-21-B-INTENT): наблюдатель, читатели не меняются.

    Производитель — одна точка сразу после NLU (`observe_intent`): метка, кто её решил (`source` — `rule` | `llm`,
    `rule` — флаги ветки правила), уверенность. Каждая переклейка после NLU (строка таблицы политик `topic_policy`)
    дописывает себя в `overrides` (`<строка>:<было>><стало>`); `final` — метка, с которой ход идёт дальше. Инвариант:
    метка, по которой планировщик строит план, равна `final` (`check_planned`; расхождение — `tu_intent_mismatch`).
    """

    label: str
    source: str
    rule: str
    confidence: float
    overrides: list[str] = field(default_factory=list)

    @property
    def final(self) -> str:
        return self.overrides[-1].rsplit(">", 1)[1] if self.overrides else self.label

    def as_debug(self) -> dict[str, Any]:
        return {"label": self.label, "source": self.source, "rule": self.rule, "final": self.final, "overrides": list(self.overrides)}


_CURRENT: contextvars.ContextVar[TurnUnderstanding | None] = contextvars.ContextVar("mr_turn_understanding", default=None)
_INTENT: contextvars.ContextVar[TurnIntent | None] = contextvars.ContextVar("mr_turn_intent", default=None)
_ACTED: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("mr_turn_understanding_acted", default=())
_FULL: contextvars.ContextVar[TurnUnderstanding | None] = contextvars.ContextVar("mr_turn_understanding_full", default=None)
# Вопрос кнопки, висевший на входе хода: (лист, тема вопроса); нет — None.
_ASKED: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("mr_turn_understanding_asked", default=None)
FULL_MODES = ("off", "shadow")
# Тип вопроса хода под `facet_by_llm = on`: (реплика, задача решения) — одно решение на ход для всех читателей.
_FACET: contextvars.ContextVar[tuple[str, asyncio.Future[FacetDecision]] | None] = contextvars.ContextVar("mr_turn_facet", default=None)


def mode() -> str:
    raw = str(getattr(_cfg, "MR_TURN_UNDERSTANDING_PRICE", "shadow") or "").strip().lower()
    return raw if raw in MODES else "shadow"


def agree_telemetry() -> bool:
    """`parse_agree_telemetry` (T7, шаг 4): согласие разрезов под `turn_parse = on` — on | off (по умолчанию с 16.10)."""

    return str(getattr(_cfg, "MR_PARSE_AGREE_TELEMETRY", "off") or "").strip().lower() == "on"


def begin_turn() -> None:
    """Новый ход: понимание прошлого хода не читается (тесты и стенд гоняют ходы в одном цикле);
    память разбора реплики — новая, на этот ход (`turn_parse`, во всех режимах выключателя); имена LLM прошлого
    хода — тоже (`turn_names`)."""

    _CURRENT.set(None)
    _ACTED.set(())
    _INTENT.set(None)
    _FULL.set(None)
    _ASKED.set(None)
    _FACET.set(None)
    turn_parse.begin()
    turn_names.begin()


def observe_intent(label: str, confidence: float, topic_source: dict[str, Any] | None) -> None:
    """Тема хода сразу после NLU (этап 1 хода (б)) — одна точка производителя; только наблюдение."""

    source = dict(topic_source or {})
    _INTENT.set(TurnIntent(label=str(label), source=str(source.get("src") or "?"), rule=str(source.get("rule") or ""), confidence=float(confidence)))


def intent() -> TurnIntent | None:
    return _INTENT.get()


def note_override(entry: str) -> None:
    """Переклейка метки после NLU (строка таблицы политик) — в тему хода."""

    current_intent = _INTENT.get()
    if current_intent is not None:
        current_intent.overrides.append(entry)


def check_planned(label: str) -> None:
    """План строится по метке `label`: она должна совпасть с итоговой меткой темы хода (наблюдатель этапа 1)."""

    current_intent = _INTENT.get()
    if current_intent is not None and current_intent.final != str(label):
        turn_telemetry.note(tu_intent_mismatch=f"{current_intent.final}>{label}")


def full_mode() -> str:
    """`[MESSENGER_ROUTER] turn_understanding_full = off | shadow` (ход (б), шаг 0); неизвестное значение — shadow."""

    raw = str(getattr(_cfg, "MR_TURN_UNDERSTANDING_FULL", "shadow") or "").strip().lower()
    return raw if raw in FULL_MODES else "shadow"


def note_asked(pending: Any) -> None:
    """Начало хода: какой лист кнопки задал висящий вопрос (рамка полного понимания); off — ничего."""

    if full_mode() != "shadow" or not isinstance(pending, dict):
        return
    from . import button_flow

    button = button_flow.question_button(pending)
    if button:
        _ASKED.set((button, str(pending.get("label") or "")))


def full() -> TurnUnderstanding | None:
    """Полное понимание этого хода (шаг 0, тень): его пока никто не читает для решений — только debug и телеметрия."""

    return _FULL.get()


def _entities_of(entities: dict[str, Any] | None) -> TurnEntities:
    """Сущности решения хода → `TurnEntities` (первый непустой ключ). ФИО пациента сюда не входит."""

    source = entities if isinstance(entities, dict) else {}

    def first(*keys: str) -> str:
        return next((str(source[key]) for key in keys if source.get(key)), "")

    return TurnEntities(
        service=first("service_name", "test_name"),
        doctor=first("doctor_name"),
        specialty=first("specialty"),
        branch=first("branch_name", "filial"),
        city=first("city"),
        date=first("date_from", "date_hint"),
    )


def _frame_of(flags: Sequence[Any], label: str, pending: dict[str, Any] | None) -> Frame | None:
    """Рамка кнопки — из того, что уже решили кнопочные слои хода (копия, без нового решения): держится ли рамка — решение
    `button_flow.note_frame` (его же читает таблица политик), без него — флаг темы ответа на вопрос кнопки."""

    from . import address_flow, button_flow

    decided = button_flow.turn_frame()
    held = decided.held if decided is not None else "button_answer_topic" in flags
    pressed = button_flow.pressed(flags)
    if pressed:
        return Frame(button=pressed, kind="press", topic=label, held=True)
    leaf = button_flow.leaf_frame()
    if leaf is not None:
        return Frame(button=button_flow.leaf_frame_id(), kind="leaf", topic=str(leaf.label or ""), held=label == leaf.label)
    asked = _ASKED.get() or ((button_flow.question_button(pending), str((pending or {}).get("label") or "")) if pending else None)
    if asked and asked[0]:
        return Frame(button=asked[0], kind="question", topic=asked[1], held=held)
    address = address_flow.address_button()
    if address:
        return Frame(button=address, kind="address", topic="ADDRESS", held=label == "ADDRESS")
    return None


def _expects_of() -> Expects | None:
    from . import pending_answer

    found = pending_answer.current()
    if found is None:
        return None
    return Expects(type=str(found.type), outcome=str(getattr(found.outcome, "value", found.outcome)), source=str(getattr(found.source, "value", found.source)))


def _intent_of(label: str, confidence: float, decision_source: str) -> TurnIntent:
    """Тема хода: наблюдатель после NLU (`observe_intent`), а на путях мимо NLU (кнопка, prelock записи, обработчик висящего
    вопроса) — копия решения хода с его источником."""

    observed = _INTENT.get()
    if observed is not None:
        return observed
    return TurnIntent(label=str(label), source=str(decision_source or "?"), rule="", confidence=float(confidence or 0.0))


def observe_turn(*, label: str, confidence: float, decision_source: str, flags: Sequence[Any], entities: dict[str, Any] | None, plan_label: str, user_text: str, pending: dict[str, Any] | None) -> None:
    """Полное понимание хода (ход (б), шаг 0, тень) — одна точка, когда план построен (`router.build_plan`): тема, рамка кнопки,
    висящий вопрос и ответ, сущности — копиями уже принятых решений; новых вызовов LLM нет, ответы не меняются. Понимание цены
    этого хода (`current()`) — основа объекта. Сбой — только след `tu_full_error`, ход идёт как без тени."""

    if full_mode() != "shadow":
        return
    try:
        intent = _intent_of(label, confidence, decision_source)
        base = _CURRENT.get() or TurnUnderstanding(reply=str(user_text or ""))
        flag_list = list(flags or ())
        frame = _frame_of(flag_list, str(label), pending if isinstance(pending, dict) else None)
        expects = _expects_of()
        entities_seen = _entities_of(entities)
        full_tu = replace(base, intent=intent, frame=frame, expects=expects, entities=entities_seen)
        decided = _facet_decided(user_text)
        if decided is not None and decided.src == "llm":
            full_tu = replace(full_tu, facet=decided.value, facet_rule="llm", facet_src="llm")
        _FULL.set(full_tu)
        seen: dict[str, Any] = {"tu_full_intent_src": intent.source}
        if frame is not None:
            seen["tu_frame"] = f"{frame.kind}:{frame.button}:{'held' if frame.held else 'free'}"
        if expects is not None:
            seen["tu_expects"] = f"{expects.type}:{expects.outcome}"
        if entities_seen.present():
            seen["tu_entities"] = entities_seen.present()
        turn_telemetry.note(**seen)
        if intent.final != str(plan_label):
            turn_telemetry.note(tu_plan_mismatch=f"{intent.final}>{plan_label}")
    except Exception as exc:  # тень не ломает ход
        logger.warning("turn_understanding_full_failed", exc_info=True)
        turn_telemetry.note(tu_full_error=type(exc).__name__)


def full_debug() -> dict[str, Any] | None:
    """Полное понимание хода для debug (`understanding_full`); off или ход без плана — None."""

    tu = _FULL.get()
    if tu is None:
        return None
    return {
        **tu.as_debug(),
        "mentions": [{"phrase": m.phrase, "strength": m.strength, "chosen": m.chosen, "source": m.source} for m in tu.services],
        "intent": tu.intent.as_debug() if tu.intent else None,
        "frame": vars(tu.frame) if tu.frame else None,
        "expects": vars(tu.expects) if tu.expects else None,
        "entities": vars(tu.entities) if tu.entities else None,
    }


def current() -> TurnUnderstanding | None:
    return _CURRENT.get()


def one_enabled() -> bool:
    """`[MESSENGER_ROUTER] turn_understanding_one = off | on` (ход (б), этап 2; один выключатель на всех читателей — решение
    владельца 20.10 (б) 4): под `on` «список ли» на пути правил решается один раз — пониманием хода (`rules_list`)."""

    return str(getattr(_cfg, "MR_TURN_UNDERSTANDING_ONE", "off") or "").strip().lower() == "on"


def of_reply(text: str) -> TurnUnderstanding | None:
    """Понимание, по которому читатель с входом `text` действует под `turn_understanding_one = on` (ход (б), этап 2):
    только действующее (`turn_understanding_price = on`, не тень) и только этой реплики хода. None — выключатель off,
    понимания нет или вход — не реплика хода (дозаполнение, контекст прошлого хода): читатель идёт прежним путём."""

    tu = acting()
    return tu if tu is not None and one_enabled() and tu.for_text(text) else None


def strong_choice(text: str) -> str | None:
    """Исход `one_strong` (дизайн 19.10 §3, U1/U2): реплика хода — одна услуга, сильная по данным МИС (дословное название,
    синоним клиники, код) и с одним названием прайса — это название. Иначе (слабая фраза, несколько названий, список, понимания
    этой реплики нет, выключатель off) — None: подстановка каталога и грундер идут прежним путём."""

    tu = of_reply(text)
    single = tu.single if tu is not None else None
    if single is None or single.strength != "strong" or not single.chosen:
        return None
    return single.chosen


def parse_for(text: str) -> turn_parse.PhraseParse | None:
    """Разбор реплики, по которому действует читатель с входом `text` (L-15b). Ход (б), этап 2 (консилиум 20.10 §2): один
    публичный объект — понимание хода, `PhraseParse` — деталь его производителя; читатели берут разбор только здесь
    (правило CI — `tests/test_turn_understanding_one_class.py`). Поведение — как у `turn_parse.acting_for`."""

    return turn_parse.acting_for(text)


def service_phrase(text: str) -> str:
    """Позиции-услуги реплики для слота-услуги (`pending_in_frame`): фразы решённого разбора этой реплики — слова пациента об
    услугах, список — через запятую (следующий ход разберёт его снова списком: корзина, а не одна из услуг). Пустой разбор, сбой
    LLM (даже с разрезом по разделителям) или разбора этой реплики нет — "": реплика целиком значением услуги не становится (Н-1)."""

    parse = parse_for(text)
    if parse is None or not parse.decided:
        return ""
    return ", ".join(item.text for item in parse.items)


# Разбор решил: услуги в реплике нет (не сбой).
_NO_SERVICE_NAMED = frozenset({turn_parse.ParseStatus.NONE_FIT, turn_parse.ParseStatus.EMPTY})
# Темы, где названную услугу проверяет каталог (и судья LLM): слот-услугу им даёт каталог, не разбор.
CATALOG_CHECKED_LABELS = frozenset({"APPOINTMENT", "PRICE", "ADDRESS", "TEST_ASSIST"})


def service_slot_on() -> bool:
    """Слот-услуга — из разбора хода (`service_slot`): под `pending_in_frame` (Н-1) и под `facet_by_llm` (шаг 2). Тип вопроса
    по LLM снимает переклейку по буквам обратно в тему прошлого хода (`flow_prepare_followup_override`), которая прятала слова
    вопроса в слоте: «а запишите меня» после памятки шло в памятку, а под `facet_by_llm` — в запись, и та читает слот
    («запись на Нему подготовиться», финальная карточка набора 22). Защита слота включается вместе с ним. Оба off — как было."""

    from .pending_answer import in_frame_on  # не на уровне модуля: функция ответа импортирует понимание хода

    return in_frame_on() or facet_by_llm()


async def service_slot(label: str, user_text: str) -> str | None:
    """Слот-услуга темы без проверки каталогом — из разбора хода этой реплики (`service_slot_on`; класс Н-1 «реплика —
    значением слота», Д3-слот): позиции разбора через запятую; разбор решил «услуги нет» — показанная прошлым ответом услуга,
    если она одна, иначе ""; сбой — "" (слот не пишется: неизвестно, не назвал ли пациент новую услугу). Одна точка для всех
    писателей слота из реплики — грундера сущностей решения, дозаполнения хода и ответа на вопрос-слот кнопки
    (`router._pending_slot_value`). Позиции разбора — слова пациента: вход инструмента этого хода, а не контекст следующего
    (И-5.5 консилиума шага 5) — после ответа их заменяет показанное или снимает конец хода (`shown_context.parsed_input`).
    Разбор подготовки строится здесь же — тот, что прочтёт инструмент (`observe_prepare_turn`: раз за ход, второго решения LLM
    нет). None — как было: выключатель off, тема с проверкой каталогом или разбора нет."""

    if label in CATALOG_CHECKED_LABELS or not service_slot_on():
        return None
    await observe_prepare_turn(label, user_text=user_text)
    status = parse_status(user_text)
    if status is None:
        return None
    if status in _NO_SERVICE_NAMED:
        # Услуги в реплике нет («а как к нему подготовиться?»): речь о показанном прошлым ответом, если он показал одну
        # услугу (правило владельца 14.10 — в контексте услуга, названная пациенту одной строкой; `shown_context`).
        shown = shown_context.previous()
        return shown[0] if len(shown) == 1 else ""
    phrases = service_phrase(user_text)
    if phrases:
        shown_context.parsed_input(phrases)
    return phrases


def parse_status(text: str) -> str | None:
    """Чем кончился разбор этой реплики (`turn_parse.ParseStatus`), без права действовать по нему: читателю с политикой
    сбоя (подготовка, `prepare_by_parse`) нужно отличить «услуги нет» (`empty`) и сбой (`unavailable`, `invalid`, `too_many`,
    `error`) от «разбора этой реплики в ходе нет» — `parse_for` во всех трёх случаях отдаёт None. None — разбора нет."""

    parse = turn_parse.phrases_now()
    return parse.status if parse is not None and parse.for_text(text) else None


def prepare_by_parse() -> bool:
    """`[MESSENGER_ROUTER] prepare_by_parse = off | on` (этап 2 «без стоп-слов», STEP-2026-10-24-PREPARE-BY-PARSE): разбор
    хода строится и на ходу подготовки (`observe_prepare_turn`), предмет памятки — его фразы (`services/prepare.py`)."""

    return str(getattr(_cfg, "MR_PREPARE_BY_PARSE", "off") or "").strip().lower() == "on"


async def observe_prepare_turn(label: str, *, user_text: str) -> None:
    """Разбор реплики на фразы на ходу подготовки (`label` — тема, по которой строится план; `prepare_by_parse = on`).

    Понимание хода (услуги и свойство для ценовых читателей) здесь не строится — разбор нужен только подготовке. Ход уже
    разобран (ценовой, переклеенный в подготовку) — тот же разбор, второго решения LLM нет."""

    if label != "PREPARE" or not prepare_by_parse() or not str(user_text or "").strip():
        return
    await _observe_phrases(user_text, agree=False)


def rules_list(text: str, rows: list[dict[str, Any]] | None = None) -> bool:
    """Путь правил (разбора нет: `empty` / `too_many` / `error` / вход — не реплика хода): назвал ли пациент список — одно
    решение на ход (ход (б), этап 2, консилиум 20.10 §1). Понимание хода этой реплики уже решило — его вердикт; иначе
    (вход — не реплика хода) — то же правило понимания `price_list_services` с защитами K1. Читатели корзины (R1) и подбора
    анализов (R7) под `turn_understanding_one = on` берут его вместо своего «список ли» без K1."""

    tu = _CURRENT.get()
    if tu is not None and tu.for_text(text) and tu.services_rule and not tu.services_rule.startswith("parse_"):
        verdict, source = tu.services_rule.startswith("list_"), "understanding"
    else:
        verdict, source = bool(price_list_services(text, rows=rows)[0]), "rules"
    turn_telemetry.note(tu_rules_list=f"{source}:{'list' if verdict else 'one'}")
    return verdict


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


# Число решений хода не тратит (`verdict`: секунды хода, дедлайн и предохранитель — как у решения): тип вопроса — понимание
# хода, как вопрос рамки кнопки (STEP-2026-10-29-BUTTON-FRAME) и ответ на висящий вопрос (Q5 18.10). На ходе цены он иначе
# отнимал место у проверки строки С3 (стенд 30.10, лимит 4: отказов `budget` 9 → 41); возврат решения после вызова, как у
# рамки, оставлял бы гонку — вопрос идёт параллельно разбору реплики.
FACET_TASK = Task(name="turn_facet", max_picks=1, num_predict=32, timeout_s=5.0, queue_timeout_ms=1500, verdict=True)
# Варианты — `choose_test_assist_facet_full` (П1, BUG-2026-10-22-P1-FACET-NONE-AS-PRICE) плюс «нужна ли запись», отделённая от
# «где и когда»; `id` — имена `facet` понимания хода (правила дают их подмножество: readiness, prepare, booking, where, price).
FACET_OPTIONS = (
    Option(id="price", text="Узнать стоимость анализа или услуги"),
    Option(id="availability", text="Узнать, есть ли такой анализ или услуга в клинике, делают ли его, что в него входит"),
    Option(id="where", text="Узнать, где или в какое время можно сдать анализ или пройти услугу"),
    Option(id="booking", text="Узнать, нужна ли запись или можно прийти без записи"),
    Option(id="readiness", text="Узнать срок готовности результата"),
    Option(id="prepare", text="Узнать, как подготовиться к анализу или услуге"),
    Option(id="bare", text="Просто назвал анализ или услугу, без вопроса"),
)
FACET_NONE = "none"  # LLM: спрашивают о другом (возраст, значение показателя, свой материал) или смысл неясен
FACETS = (*(o.id for o in FACET_OPTIONS), FACET_NONE)


@dataclass(frozen=True)
class FacetDecision:
    """Тип вопроса хода: `value` — одно из `FACETS` или None; `src` — `llm` (решила LLM), `fail` (сбой или лимит хода —
    значение правил), `rules` (выключатель off или слов пациента нет — значение правил); `rule` — детектор правил."""

    value: str | None
    src: str
    rule: str = ""


def facet_by_llm() -> bool:
    """`[MESSENGER_ROUTER] facet_by_llm = on`; мусор и отсутствие — off (как было)."""

    return str(getattr(_cfg, "MR_FACET_BY_LLM", "off") or "").strip().lower() == "on"


def start_facet(user_text: str) -> asyncio.Future[FacetDecision]:
    """Запустить решение о типе вопроса этой реплики (или вернуть уже запущенное в этом ходе) — без ожидания: на ходе цены
    вопрос идёт параллельно разбору реплики."""

    text = str(user_text or "")
    held = _FACET.get()
    if held is not None and held[0] == text:
        return held[1]
    task = asyncio.ensure_future(_decide_facet(text))
    _FACET.set((text, task))
    return task


def _facet_decided(user_text: str) -> FacetDecision | None:
    """Решение о типе вопроса этой реплики, если оно уже принято в ходе (полное понимание его только читает: не ждёт и не
    запускает); нет или ещё не готово — None."""

    held = _FACET.get()
    if held is None or held[0] != str(user_text or "") or not held[1].done() or held[1].cancelled():
        return None
    return held[1].result() if held[1].exception() is None else None


async def turn_facet(user_text: str) -> FacetDecision:
    """Тип вопроса хода — одно решение на ход (первый читатель задаёт вопрос, остальные берут его ответ)."""

    return await start_facet(user_text)


# Читатели шага 5 (сущности из понятого и показанного) — значения ключа `entities_by_understanding` (решение владельца 30.10, Ш5-3).
ENTITY_READERS = frozenset({"context", "price", "price_select"})


def entity_readers() -> frozenset[str]:
    """Читатели шага 5, которые действуют: `[MESSENGER_ROUTER] entities_by_understanding` — список через запятую; `off`, пусто и
    неизвестные значения — не действуют (неизвестные — громко при старте, `entities_config_errors`)."""

    raw = str(getattr(_cfg, "MR_ENTITIES_BY_UNDERSTANDING", "off") or "")
    return frozenset(item.strip().lower() for item in raw.split(",")) & ENTITY_READERS


def entities_config_errors() -> list[str]:
    """Неизвестные читатели в `entities_by_understanding` — ошибка конфига (Н-7: молчаливо неверный ключ не действует)."""

    raw = str(getattr(_cfg, "MR_ENTITIES_BY_UNDERSTANDING", "off") or "")
    unknown = sorted({item.strip().lower() for item in raw.split(",") if item.strip()} - ENTITY_READERS - {"off"})
    return [f"entities_by_understanding: неизвестные читатели {', '.join(unknown)} (допустимы: {', '.join(sorted(ENTITY_READERS))})"] if unknown else []


def absorbed() -> bool:
    """Ход поглощён ответом на свой вопрос бота (`own_question_reply = on`, BUG-2026-10-30-OWN-LIST-REPLY): одна функция ответа
    на висящий вопрос решила, что реплика — ответ (пункт показанного списка, да / нет, значение слота). Тип вопроса такого хода
    задал вопрос бота — LLM о голой реплике («1», «первый», «почек») не спрашивается, читатели типа вопроса идут прежним путём.
    Одно определение для всех читателей типа вопроса (не белый список флагов правил)."""

    from . import pending_answer  # функция ответа импортирует понимание хода — импорт на вызове

    return pending_answer.absorbs_turn()


async def llm_facet(user_text: str) -> str | None:
    """Тип вопроса, по которому читатель действует под `facet_by_llm = on`: решение LLM этой реплики. None — выключатель off,
    сбой LLM, лимит хода, слов нет, ход поглощён ответом на свой вопрос (`absorbed`): читатель идёт прежним путём (своими
    детекторами)."""

    if not facet_by_llm():
        return None
    if absorbed():
        turn_telemetry.note(tu_facet_src="absorbed")
        return None
    decided = await turn_facet(user_text)
    return decided.value if decided.src == "llm" else None


async def _decide_facet(text: str) -> FacetDecision:
    rule_value, rule = price_turn_facet(text)
    if not facet_by_llm():
        return FacetDecision(rule_value, "rules", rule)
    try:
        picked = await choose(FACET_TASK, text, FACET_OPTIONS)
    except Exception:  # новый слой: его сбой не ломает ход — тип вопроса правил, как было
        logger.warning("turn_facet_failed", exc_info=True)
        turn_telemetry.note(tu_facet=rule_value, tu_facet_src="fail", tu_facet_fail="error")
        return FacetDecision(rule_value, "fail", rule)
    if picked.status is Status.PICKED and picked.ids:
        decided = FacetDecision(picked.ids[0], "llm", "llm")
    elif picked.status is Status.NONE_FIT:
        decided = FacetDecision(FACET_NONE, "llm", "llm")
    elif picked.reason == NO_WORDS:
        decided = FacetDecision(rule_value, "rules", rule)  # ход нажатия кнопки: смысл задала кнопка, судить нечего
    else:
        decided = FacetDecision(rule_value, "fail", rule)
        turn_telemetry.note(tu_facet_fail=picked.reason or picked.status.value)
    turn_telemetry.note(tu_facet=decided.value, tu_facet_src=decided.src)
    return decided


async def _with_facet(tu: TurnUnderstanding, facet: asyncio.Future[FacetDecision]) -> TurnUnderstanding:
    """Понимание хода цены с типом вопроса от LLM; сбой — тип правил остаётся, отмечен источник."""

    decided = await facet
    if decided.src == "llm":
        tu = replace(tu, facet=decided.value, facet_rule="llm", facet_src="llm")
    else:
        tu = replace(tu, facet_src=decided.src)
    _CURRENT.set(tu)
    turn_telemetry.note(tu_facet=tu.facet, tu_facet_src=tu.facet_src)
    return tu


def understand_price_turn(text: str) -> TurnUnderstanding:
    """Услуги и свойство ценовой реплики. Чистая функция: без состояния и без LLM.

    Список ищется в реплике без фразы о сроке: «когда будет готов анализ на ферритин» — одна
    услуга и свойство «срок», а не корзина из лишних слов.
    """

    facet, facet_rule = price_turn_facet(text)
    probe = strip_readiness_phrasing(text)[0] if facet == "readiness" else text
    names, rule = price_list_services(probe)
    mentions = tuple(ServiceMention(phrase=name, candidates=(name,), strength=rule, chosen=name) for name in names)
    if not mentions:
        mentions = _single_mention(probe)
    return TurnUnderstanding(services=mentions, services_rule=rule, facet=facet, facet_rule=facet_rule, reply=text)


def _bare_phrase(raw: str) -> str:
    """Фраза одиночной услуги без речевой обвязки с краёв (`_MENTION_EDGE`), срезом исходной строки:
    «а ггтп» → «ггтп», «а подскажите пожалуйста сколько стоит ггтп» → «ггтп». Внутри фразы ничего не
    трогаем; от фразы осталась одна обвязка — пусто."""

    raw = str(raw or "")
    toks = _turn_phrases._tokens(raw)
    i, j = 0, len(toks) - 1
    while i <= j and toks[i].key in _MENTION_EDGE:
        i += 1
    while j >= i and toks[j].key in _MENTION_EDGE:
        j -= 1
    return raw[toks[i].start : toks[j].end] if i <= j else ""


def _mention(phrase: str, rows: Sequence[dict[str, Any]], source: str) -> ServiceMention:
    names = tuple(dict.fromkeys(str(row.get("serviceName") or row.get("name") or "").strip() for row in rows))
    return ServiceMention(
        phrase=phrase,
        candidates=names,
        strength="strong" if names else "weak",
        chosen=names[0] if len(names) == 1 else None,
        source=source,
    )


def _single_mention(probe: str) -> tuple[ServiceMention, ...]:
    """Упоминание одиночной услуги по правилам (L-14, шаг 0): фраза — то, что вырезатель правил оставил от
    реплики, без речевой обвязки с краёв (шаг 2: «а ггтп» → «ггтп»); сила и кандидаты — сильное совпадение по
    данным МИС (`strong_rows`: название, синоним, код). Слабая — без кандидатов и без выбора (строку выбирает
    путь цены, как раньше). Фразы нет — ничего."""

    phrase = _bare_phrase(_extract_price_service_from_query(probe) or "")
    if all(t.key in _turn_phrases._FILLER for t in _turn_phrases._tokens(phrase)):
        return ()  # от реплики осталась только обвязка («сколько стоит?», «а цена?») — услуги нет
    return (_mention(phrase, strong_rows(probe, _catalog_rows(None)), "rules"),)


def _mention_fields(tu: TurnUnderstanding) -> dict[str, Any]:
    """Телеметрия упоминания одиночной услуги (L-14, шаг 1): откуда и какой силы — без текста пациента."""

    single = tu.single
    if single is None:
        return {"tu_mention_src": None, "tu_mention_strength": None}
    return {"tu_mention_src": single.source, "tu_mention_strength": single.strength}


def is_price_turn(label: str, pending: dict[str, Any] | None) -> bool:
    pending_label = (pending or {}).get("label") if isinstance(pending, dict) else None
    return label in _PRICE_TURN_LABELS or pending_label == "PRICE"


async def observe_price_turn(label: str, *, user_text: str, pending: dict[str, Any] | None) -> TurnUnderstanding | None:
    """Построить понимание ценового хода (кроме `off`) и записать его в ход вместе с разбором реплики на фразы
    (`turn_parse`, L-15b — единственный режим с T7 шага 3б). Понимание выключено — разбор всё равно идёт: его читают
    корзина, путь одной услуги и срок."""

    if not is_price_turn(label, pending) or not str(user_text or "").strip():
        return None
    if mode() != "off":
        # Тип вопроса от LLM (`facet_by_llm = on`) — параллельно разбору реплики; ход ждёт обоих.
        facet = start_facet(user_text) if facet_by_llm() and not absorbed() else None
        tu = await (_observe_parse_on(user_text) if agree_telemetry() else _observe_parse_on_lazy(user_text))
        tu = _decide_named(tu, user_text) if tu is not None and one_enabled() else tu
        if facet is None:
            return tu
        if tu is None:
            await facet  # понимание упало — решение о типе вопроса остаётся читателям хода (`llm_facet`)
            return None
        return await _with_facet(tu, facet)
    await _observe_phrases(user_text)
    return None


def _decide_named(tu: TurnUnderstanding, user_text: str) -> TurnUnderstanding:
    """Названа ли услуга — один раз за ход, в производителе (U6, ход (б), этап 2): есть упоминание — названа; нет — то же
    определение, что у планировщика и выбора строки до этапа 2 (`_price_select_llm.names_service`). Только «нет упоминания»
    не годится: вырезатель правил не видит услугу в «Сколько стоит прием к оталорингологу?», «Посев на Candida показывает
    …?», а разбор «ни одна фраза не услуга» (`none_fit`) бывает при верной строке цены («посев на хламидии») — replay
    test/dev 21.10: 5 потерь верной цены из 1068 ходов."""

    from .services._price_select_llm import names_service  # выбор строки импортирует понимание хода — импорт на вызове

    tu = replace(tu, names_service=bool(tu.services) or names_service(user_text))
    _CURRENT.set(tu)
    return tu


async def _observe_parse_on(user_text: str) -> TurnUnderstanding | None:
    """Под `turn_parse = on` (шаг 6, O3b): правила понимания считаются, пока LLM выбирает фразы.

    Порядок: генератор фраз (CPU, в потоке) → одновременно выбор фраз LLM и правила понимания (CPU, в
    потоке) → читатели. Правила стартуют после генератора: сегментацию сплошного списка (в) генератор
    кладёт в память хода, правила берут её оттуда — один ключ не считается дважды. Бюджет хода прежний:
    правила — не вызов LLM. Ответ хода тот же, что при правилах до разбора: свойство (`facet`) правила
    считают той же `price_turn_facet`, а читатели стартуют после обоих.
    """

    facet = price_turn_facet(user_text)[0]
    prepared = asyncio.Event()
    rules: asyncio.Future[TurnUnderstanding | None] = asyncio.get_running_loop().create_future()
    created = False

    def factory() -> Awaitable[turn_parse.PhraseParse]:
        nonlocal created
        created = True
        return _parse_phrases(user_text, facet=facet, tu=None, prepared=prepared, rules=rules)

    # Разбор уже запущен в этом ходе (повторный вызов) — его правила ждёт его же вызов; здесь не ждём.
    started = turn_parse.start_phrases(factory, reply=user_text)
    if started is not None and created:
        waiter = asyncio.ensure_future(prepared.wait())
        try:
            await asyncio.wait({started, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
    tu: TurnUnderstanding | None = None
    try:
        tu = await asyncio.to_thread(understand_price_turn, user_text)
    except Exception:  # понимание — новый слой; его сбой не ломает ход, старый путь остаётся
        logger.warning("turn_understanding_failed", exc_info=True)
        turn_telemetry.note(tu_rule="error")
    else:
        _CURRENT.set(tu)
        turn_telemetry.note(tu_list=tu.list_size, tu_facet=tu.facet, tu_rule=tu.services_rule, **_mention_fields(tu))
    finally:
        if not rules.done():
            rules.set_result(tu)
    if started is not None:
        await turn_parse.phrases()
    read = _from_phrases(tu, user_text) if tu is not None else None
    if read is not None:
        tu = read
        _CURRENT.set(tu)
        turn_telemetry.note(tu_list=tu.list_size, tu_rule=tu.services_rule, **_mention_fields(tu))
    if tu is not None:
        turn_telemetry.note(tu_understanding_src="object" if read is not None else "rules")
    return tu


async def _observe_parse_on_lazy(user_text: str) -> TurnUnderstanding | None:
    """Под `turn_parse = on` без телеметрии согласия (`parse_agree_telemetry = off`, T7, шаг 4): правила
    понимания — только когда без них не обойтись.

    Разбор решён (или сбой LLM с разрезом по разделителям) — услуги хода берёт `_from_phrases` из фраз, от
    правил ему нужны только свойство и реплика, а свойство — та же `price_turn_facet`: правила не считаются.
    Нужны они в двух случаях, и тогда — после разбора, как раньше: (1) разбор не годится для действия
    (`acting_for` — None: empty, too_many, сбой LLM без разделителей) — понимание правил и есть ответ (запасной
    путь); (2) `single` (кандидат один, слабый) — упоминание одиночной услуги правил, а список ли — решает разрез
    правил (`_from_phrases`). Понимание и телеметрия хода (`tu_list`, `tu_facet`, `tu_rule`, `tu_mention_*`) —
    те же, что при правилах параллельно разбору; `tu_parse_agree` не пишется.
    """

    facet, facet_rule = price_turn_facet(user_text)
    if turn_parse.start_phrases(lambda: _parse_phrases(user_text, facet=facet, tu=None, agree=False), reply=user_text) is not None:
        await turn_parse.phrases()
    parse = turn_parse.acting_for(user_text)
    if parse is not None and parse.status != turn_parse.ParseStatus.SINGLE:
        by_parse = _from_phrases(TurnUnderstanding(facet=facet, facet_rule=facet_rule, reply=user_text), user_text)
        if by_parse is not None:
            _CURRENT.set(by_parse)
            turn_telemetry.note(
                tu_list=by_parse.list_size,
                tu_facet=by_parse.facet,
                tu_rule=by_parse.services_rule,
                tu_understanding_src="object",
                **_mention_fields(by_parse),
            )
            return by_parse
    try:
        tu = await asyncio.to_thread(understand_price_turn, user_text)
    except Exception:  # понимание — новый слой; его сбой не ломает ход, старый путь остаётся
        logger.warning("turn_understanding_failed", exc_info=True)
        turn_telemetry.note(tu_rule="error")
        return None
    _CURRENT.set(tu)
    turn_telemetry.note(tu_list=tu.list_size, tu_facet=tu.facet, tu_rule=tu.services_rule, **_mention_fields(tu))
    read = _from_phrases(tu, user_text)
    if read is not None:
        tu = read
        _CURRENT.set(tu)
        turn_telemetry.note(tu_list=tu.list_size, tu_rule=tu.services_rule, **_mention_fields(tu))
    turn_telemetry.note(tu_understanding_src="object" if read is not None else "rules")
    return tu


def _from_phrases(tu: TurnUnderstanding, text: str) -> TurnUnderstanding | None:
    """Под `turn_parse = on` (L-15b, шаг 5) услуги хода — фразы разбора, а не разрез правил: одна услуга
    или список решили данные МИС или выбор LLM. Читатели понимания (`acting_list()` — планировщик,
    подстановка каталога, грундер, дозаполнение) берут его без правок. Разбора нет или он не решён —
    None, понимание правил остаётся."""

    parse = turn_parse.acting_for(text)
    if parse is None:
        return None
    if parse.status == turn_parse.ParseStatus.SINGLE and not tu.is_list:
        # Кандидат один — реплика целиком, LLM не звали, по данным фраза слабая: разбор знает об услуге не
        # больше правил, упоминание — правил (L-14, шаг 2: «Цена анализа» — услуги не названо, «а … ггтп сдать» —
        # «ггтп», а не W с обвязкой внутри). Список или одна услуга — как у разбора: не список.
        return replace(tu, services_rule=f"parse_{parse.status}")
    services = tuple(
        ServiceMention(
            phrase=item.text,
            candidates=tuple(dict.fromkeys(str(row.get("serviceName") or "").strip() for row in item.rows)),
            strength=str(item.strength.value),
            chosen=item.text,
            source="parse",
        )
        for item in parse.items
    ) if parse.is_list else tuple(_mention(_bare_phrase(item.text) or item.text, item.rows, "parse") for item in parse.items[:1])
    return TurnUnderstanding(
        services=services, services_rule=f"parse_{parse.status}", facet=tu.facet, facet_rule=tu.facet_rule, reply=tu.reply
    )


async def _observe_phrases(user_text: str, *, agree: bool = True) -> None:
    """Разбор реплики на фразы без понимания хода (`turn_understanding_price = off`; ход подготовки — `agree=False`, без
    согласия с разрезом цены): ход ждёт его здесь же — решение тратит бюджет хода; согласие разрезов — с разрезом правил
    той же реплики."""

    facet = price_turn_facet(user_text)[0]
    if turn_parse.start_phrases(lambda: _parse_phrases(user_text, facet=facet, tu=None, agree=agree), reply=user_text) is not None:
        await turn_parse.phrases()


async def _parse_phrases(
    text: str,
    *,
    facet: str | None,
    tu: TurnUnderstanding | None,
    prepared: asyncio.Event | None = None,
    rules: asyncio.Future[TurnUnderstanding | None] | None = None,
    agree: bool = True,
) -> turn_parse.PhraseParse:
    """Разбор и его след в телеметрии хода (`tu_parse_*` — без текста пациента). При нетерпеливых правилах
    (O3b) понимание правил считается параллельно: `prepared` — генератор
    готов, правилам можно стартовать; `rules` — их результат для согласия разрезов. `agree=False` — без
    согласия разрезов (`parse_agree_telemetry = off`: правила не считаются ради телеметрии)."""

    started = time.monotonic()
    agreed: str | None = None
    try:
        catalog = await asyncio.to_thread(_catalog_rows, None)
        if _turn_spans.enabled():
            # Без списков слов: LLM размечает номера слов реплики (DESIGN-2026-10-23-NO-STOPWORDS, этап 1); генератора окон нет —
            # правилам понимания ждать нечего.
            if prepared is not None:
                prepared.set()
            parse = await _turn_spans.parse_turn_spans(text, facet=facet, catalog=catalog)
        else:
            parse = await _turn_phrases.parse_turn_phrases(text, facet=facet, catalog=catalog, prepared=prepared)
        if not agree:
            return _noted(parse, None)
        if rules is not None:
            tu = await rules
            if tu is None:
                return _noted(parse, None)  # правила упали — согласия не считаем (без LLM-разреза правил)
        agreed = await asyncio.to_thread(_agreement, parse, text, facet, tu, catalog)
    except Exception:  # разбор — новый слой: сбой не ломает ход
        logger.warning("turn_parse_failed", exc_info=True)
        parse = turn_parse.PhraseParse(reply=text, status=turn_parse.ParseStatus.ERROR.value, source="none", ms=int((time.monotonic() - started) * 1000))
    finally:
        if prepared is not None:
            prepared.set()  # сбой до генератора — правилам не ждать
    return _noted(parse, agreed)


def _noted(parse: turn_parse.PhraseParse, agree: str | None) -> turn_parse.PhraseParse:
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
