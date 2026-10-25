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
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field, replace
from collections.abc import Awaitable
from typing import Any, Sequence

from . import turn_names, turn_parse, turn_telemetry
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
class TurnUnderstanding:
    services: tuple[ServiceMention, ...] = ()
    services_rule: str = ""
    facet: str | None = None  # readiness | prepare | booking | where | price | None
    facet_rule: str = ""
    reply: str = ""  # реплика хода, по которой построено понимание
    # Названа ли в реплике услуга — одно решение хода (U6, ход (б), этап 2): ставит производитель под `turn_understanding_one`
    # (`_decide_named`), планировщик цены читает. None — не решено (выключатель off): читатель идёт прежним путём.
    names_service: bool | None = None

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
        return {
            "services": list(self.service_names),
            "services_rule": self.services_rule,
            "facet": self.facet,
            "facet_rule": self.facet_rule,
            "acted": list(_ACTED.get()),
        }


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
        tu = await (_observe_parse_on(user_text) if agree_telemetry() else _observe_parse_on_lazy(user_text))
        return _decide_named(tu, user_text) if tu is not None and one_enabled() else tu
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
