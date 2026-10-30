"""Одна функция ответа на висящий вопрос бота — класс 5 до конца (дизайн `docs/DESIGN_2026-10-18_one_pending_answer.md`).

Было: ответ на вопрос бота («Перевести на оператора?», «Какой филиал вам удобен?», «с наркозом или без?» …) разбирали
11 обработчиков и 7 решателей вне их — каждый своими правилами по буквам. Отказ «не надо оператора» становился
переводом, «1» на список филиалов — инструкцией о результатах анализов, «да, давайте отменим» — тем же вопросом
(стенд 18.10: 16 неверных из 33). LLM-клиент `pending_reply` был тенью и надстройкой над правилами (решение владельца
16.10: не включать, пока все типы не переведены на одну функцию разом).

Стало (принципы 2 и 5; решение владельца 18.10 «решает LLM, правила только сужают»): одна функция `answer` в одной
точке хода решает, что ответил пациент, — исход, выбранный пункт, хвост реплики для обычного пути, — а обработчики
вопросов читают решение (`current`) и только исполняют его по таблице политик (`POLICY`).

Порядок внутри функции:
1. сужение без LLM (`fast`), новых слов нет — множества согласия и отказа из правил (`policies._CONFIRM_YES/NO_EXACT`):
   вся реплика — голый якорь, или «якорь + разделитель + хвост» (исход от якоря, хвост — обычным путём); только у
   вопросов «да/нет» — у выборов «не нужно» значит своё («без наркоза»). Реплика — дословно показанный пункт или его
   номер — `pick` у любого вопроса с вариантами;
2. LLM — закрытым выбором `choose()`: исходы да / нет / пауза / другая тема + показанные пункты (задача
   `pending_reply.PENDING_TASK`, промпт `choose_pending_reply`, вопрос бота — в данных хода, без ПДн);
3. вето правил после LLM — здесь, одним местом: отказ или оговорка (`policies._has_refusal_or_hedge`) гасят «да»
   модели; на вопросе об отмене оформления записи вето — это «нет» (собранное цело);
4. сбой LLM — `fallback`: обработчик разбирает ответ своими прежними правилами (решение владельца Q1 18.10).

Решение, не закрывшее ход (нет / другая тема / ничего из этого без якоря), лимит решений хода не тратит (решение
владельца Q5 18.10): обычному пути цены остаются оба решения; телеметрия `tu_pending_refund`.

Выключатель `pending_reply` снят 24.10 (неделя `on` на проде с деплоя 10, STEP-2026-10-22-SWITCH-REMOVAL): функция решает
всегда, все типы разом.

Вопросы-слоты кнопок и остатки класса 5 (решение владельца 19.10, п. 5) — под `[MESSENGER_ROUTER] pending_buttons = on`
(`buttons_on`), по умолчанию off — как было:
- вопрос-слот со свободным ответом (`Question.SLOT_VALUE`, вид `free`: «К какой процедуре нужна подготовка?», «Для
  какой цели подобрать анализы?», реквизиты результата, «Уточните название услуги» после «нет» на «Похоже, вы имели в
  виду …?»): исход «это ответ на вопрос» (`ANSWER`) / «не могу ответить» (`NO`) / другая тема / пауза / ничего. Сужение без
  LLM — правила уже разобрали значение слота (`filled`: «колоноскопия», «Иванов, 1989, Бг, 123») или вся реплика —
  голый отказ; иначе LLM — своя задача `FREE_TASK` (промпт `choose_pending_free_reply`: у вопроса нет вариантов, а
  исход «ответ» — не «да»). Прежний отдельный путь `booking_slots.slot_reply_llm` (задача `slot_reply`) снят — один путь;
- «обе», «все» на вопрос с вариантами, где можно взять несколько (`TypePolicy.many`: составная цена «Вижу в запросе две
  услуги: 1. … 2. …»), — исход `PICK` с несколькими пунктами (`Answer.picks`): задача `MANY_TASK` (те же исходы, выбор до
  N пунктов, промпт `choose_pending_reply_many`); прочие типы — прежняя задача с одним пунктом.

Ответ на свой вопрос бота (`[MESSENGER_ROUTER] own_question_reply = on`, BUG-2026-10-30-OWN-LIST-REPLY), по умолчанию off — как было:
- свой нумерованный список цен — вопрос `show_all` с пунктами-строками (`renderer`): «1», «первый», «давайте второй», название
  строки — пункт-строка, её цену отвечает путь цены (`shown_row_number`); «все» — раскрыть список;
- просьба инструмента цены назвать услугу («Уточните название…», «Введите конкретное название процедуры…») — вопрос-слот темы
  (`ask_name_after_clarify`; идея BUG-2026-10-26-CLARIFY-REPLY-NOT-ACCEPTED): ответ разбирает эта функция, «ответ» держит тему;
- ход, который функция признала ответом на вопрос (`absorbs_turn`), тип вопроса не судит (`turn_understanding.absorbed`): тип
  задал вопрос бота, а не голая реплика («1» → «ни одно» → оффер — регресс деплоя 22 под `facet_by_llm`).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Iterator, Mapping

from . import turn_telemetry
from .choice import Choice, Option, Status, Task, choose
from .expects import (
    ANESTHESIA_OPTIONS,
    APPOINTMENT_ACTION_OPTIONS,
    SHOW_ALL_OPTIONS,
    Question,
)
from .policies import (
    _AFFIRMATIVE_IDIOMS_RE,
    _CONFIRM_NO_EXACT,
    _CONFIRM_YES_EXACT,
    _REFUSAL_OR_HEDGE_RE,
    _has_refusal_or_hedge,
    _normalize_confirmation_text,
)

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    """Что ответил пациент на висящий вопрос."""

    YES = "yes"
    NO = "no"
    PAUSE = "pause"
    NEW_TOPIC = "new_topic"
    PICK = "pick"  # пункт показанного списка — `Answer.pick`
    NONE = "none"  # ничего из этого — обычный путь
    ANSWER = "answer"  # свободный ответ на вопрос-слот назвал то, о чём спросил бот (`pending_buttons`)
    # Свой вопрос пациента в теме нажатой кнопки (исход судьи «domain», `pending_in_frame = on`): тему держит, как ответ, но не
    # называет то, о чём спросил бот, — реплика не значение слота (Н-1, STEP-2026-10-30-PENDING-IN-FRAME)
    IN_FRAME = "in_frame"


class Source(StrEnum):
    """Кто решил."""

    FAST = "fast"  # сужение без LLM: якорь или дословный пункт / номер
    LLM = "llm"
    FALLBACK = "fallback"  # LLM недоступна или ответила не по контракту — обработчик решает прежними правилами


class Polarity(StrEnum):
    """Вид вопроса для сужения: «да/нет» — якоря согласия и отказа значат «да» / «нет»; выбор — не значат."""

    YES_NO = "yes_no"
    PICK = "pick"
    FREE = "free"  # свободный ответ — значение слота: якорь согласия не ответ, голый отказ — «не могу ответить»


@dataclass(frozen=True)
class TypePolicy:
    """Как исполнять решение на вопрос этого типа (асимметрия безопасности — дизайн §2, решение Q3 18.10).

    :param polarity: вид вопроса для якорей
    :param tail_yes: «да» с хвостом («да, но для ребёнка»): `rest` — не «да», хвост — обычным путём (ложное «да» = чужая
        цена); `execute` — «да» (оффер оператора: перевод — безопасное направление); `llm` — решает модель по всей
        реплике (вопрос об отмене оформления: «да, давайте отменим» — да, «да, но сначала адрес» — нет)
    :param veto_is_no: вопрос об отмене оформления: «да» модели гасит только отрицание («не отменяйте», «нет-нет») — до
        «нет» (собранное цело); слова «отмен…» здесь — согласие. Иначе «да» гасит отказ или оговорка
        (`policies._has_refusal_or_hedge`: «не», «передумал», «позже» …) — до «ничего из этого»
    :param many: можно выбрать несколько показанных пунктов («обе» на составную цену) — под `pending_buttons = on`
    """

    polarity: Polarity
    tail_yes: str = "rest"
    veto_is_no: bool = False
    many: bool = False


POLICY: dict[str, TypePolicy] = {
    Question.OPERATOR_OFFER.value: TypePolicy(Polarity.YES_NO, tail_yes="execute"),
    Question.CATALOG_CONFIRM.value: TypePolicy(Polarity.YES_NO, tail_yes="llm"),
    Question.APPOINTMENT_ABORT.value: TypePolicy(Polarity.YES_NO, tail_yes="llm", veto_is_no=True),
    Question.SECONDARY_OFFER.value: TypePolicy(Polarity.YES_NO),
    Question.SHOW_ALL.value: TypePolicy(Polarity.YES_NO, tail_yes="llm"),
    Question.APPOINTMENT_ACTION.value: TypePolicy(Polarity.PICK),
    Question.COMPOUND_PRICE.value: TypePolicy(Polarity.PICK, many=True),
    Question.ANESTHESIA.value: TypePolicy(Polarity.PICK),
    Question.BOOKING_SERVICE.value: TypePolicy(Polarity.PICK),
    Question.APPOINTMENT_BRANCH.value: TypePolicy(Polarity.PICK),
    Question.SLOT_VALUE.value: TypePolicy(Polarity.FREE),
}
# Типы, которые функция решает только под `pending_buttons = on`.
_BUTTON_TYPES = frozenset({Question.SLOT_VALUE.value})

# Темы вопросов-слотов со свободным ответом (`pending_buttons`): тема `_pending` → слот, куда идёт реплика целиком, если
# LLM признала её ответом, а правила значение не разобрали (название процедуры, цель подбора). Реквизиты результата
# анализов разбирают правила формата — у `TEST_RESULT` такого слота нет, ответ только держит тему вопроса.
SLOT_TOPICS: dict[str, str | None] = {"PREPARE": "service_name", "TEST_ASSIST": "test_goal", "TEST_RESULT": None}
# Слоты, чьё значение — позиция-услуга (её читает путь цены и подготовки следующего хода как название): под `pending_in_frame = on`
# — фраза разбора хода, а не реплика целиком. Цель подбора (`test_goal`) — свободные слова пациента, как у правил дозаполнения
# (`policies.quick_fill_entities_from_text`: реплика целиком); названием услуги её не читает никто.
SERVICE_SLOTS = frozenset({"service_name"})
_SLOT_VALUE_MAX = 80

# Свободный ответ на вопрос-слот: вариантов нет, исход «ответ» — свой (у вопроса «да/нет» «да» — согласие, здесь — нет).
FREE_TASK = Task(name="pending_free_reply", outcomes=("answer", "no", "pause", "new_topic"), timeout_s=8, queue_timeout_ms=3000)
# Тот же ответ на вопрос кнопки меню (`button_flow.judge_context`, `button_reply_judge`): судья видит нажатую кнопку и её тему —
# полный вопрос в теме кнопки — ответ, а не «другая тема».
FREE_BUTTON_TASK = Task(name="pending_free_reply_button", outcomes=FREE_TASK.outcomes, timeout_s=8, queue_timeout_ms=3000)
# Вариант Б рамки кнопки (`button_frame_check = judge`, STEP-2026-10-29-BUTTON-FRAME): «другая тема» судьи разделена — «domain»
# (свой вопрос в теме нажатой кнопки: наличие услуги на кнопке цены) — ответ в рамке кнопки, без отдельного вопроса рамки.
FRAME_BUTTON_TASK = Task(
    name="pending_free_reply_button_frame", outcomes=(*FREE_TASK.outcomes, "domain"), timeout_s=8, queue_timeout_ms=3000
)
_DOMAIN = "domain"


def frame_judge_on() -> bool:
    """Рамку кнопки проверяет сам судья ответа (вариант Б): `button_answer_topic = frame` и `button_frame_check = judge`
    (или `judge_leaf` — тот же судья и у листа без своего вопроса, `button_flow.leaf_judge_on`)."""

    from .runtime_config import config as cfg

    def _value(key: str) -> str:
        return str(getattr(cfg, key, "") or "").strip().lower()

    return _value("MR_BUTTON_ANSWER_TOPIC") == "frame" and _value("MR_BUTTON_FRAME_CHECK") in ("judge", "judge_leaf")


def in_frame_on() -> bool:
    """«domain» судьи — свой исход `in_frame`, а слот-услуга — из разбора хода: `[MESSENGER_ROUTER] pending_in_frame = on`."""

    from .runtime_config import config as cfg

    return str(getattr(cfg, "MR_PENDING_IN_FRAME", "off") or "").strip().lower() == "on"


def own_question_on() -> bool:
    """Ответ на свой вопрос бота — висящий вопрос этой функции (`[MESSENGER_ROUTER] own_question_reply = on`,
    BUG-2026-10-30-OWN-LIST-REPLY): показанный список цен — выбор строки, просьба назвать услугу — вопрос-слот темы."""

    from .runtime_config import config as cfg

    return str(getattr(cfg, "MR_OWN_QUESTION_REPLY", "off") or "").strip().lower() == "on"


# Выбор, где можно взять несколько пунктов («обе»): те же исходы, что у `pending_reply.PENDING_TASK`, пунктов — до N.
MANY_TASK = Task(name="pending_reply_many", outcomes=("yes", "no", "pause", "new_topic"), timeout_s=8, queue_timeout_ms=3000)

# Варианты, которые бот называет словами самого вопроса (не данными состояния) — для сверки в тестах.
FIXED_OPTIONS: dict[str, tuple[str, ...]] = {
    Question.APPOINTMENT_ACTION.value: APPOINTMENT_ACTION_OPTIONS,
    Question.ANESTHESIA.value: ANESTHESIA_OPTIONS,
    Question.SHOW_ALL.value: SHOW_ALL_OPTIONS,
}


@dataclass(frozen=True)
class Answer:
    """Решение о реплике-ответе на висящий вопрос.

    :param type: тип вопроса (`_expects.type`)
    :param outcome: исход
    :param pick: текст выбранного пункта (из `_expects.options`) — у `pick`, иначе None; выбрано несколько — первый
    :param picks: все выбранные пункты, если их можно взять несколько (`TypePolicy.many`); иначе пусто
    :param rest: хвост реплики для обычного пути — режет код, не LLM (контракт C) — или None
    :param source: кто решил
    :param text: реплика, к которой привязано решение (обработчик, вызванный мимо точки решения, его не увидит)
    """

    type: str
    outcome: Outcome
    pick: str | None = None
    rest: str | None = None
    source: Source = Source.LLM
    text: str = ""
    picks: tuple[str, ...] = ()

    @property
    def decided(self) -> bool:
        """Решение есть (не запасной путь): обработчик исполняет его, а не свои правила."""

        return self.source is not Source.FALLBACK

    @property
    def closes(self) -> bool:
        """Ход решён: да / выбор / пауза / ответ, а свой вопрос в теме кнопки — как ответ (решение судьи — шаг понимания хода);
        прочее (нет, другая тема, ничего, хвост) — дальше обычный путь."""

        return self.outcome in (Outcome.YES, Outcome.PICK, Outcome.PAUSE, Outcome.ANSWER, Outcome.IN_FRAME) and not self.rest

    @property
    def in_topic(self) -> bool:
        """Реплика в теме висящего вопроса: ответ на вопрос-слот или свой вопрос в теме нажатой кнопки (`in_frame`) — одно
        определение для темы кнопки, лимита решений хода, записи и слота."""

        return self.outcome in (Outcome.ANSWER, Outcome.IN_FRAME)


_CURRENT: contextvars.ContextVar[Answer | None] = contextvars.ContextVar("mr_pending_answer", default=None)

# «якорь + разделитель + хвост»: знак препинания на конце слова или отдельное слово-связка / тире — первое в реплике.
# Механика разбиения на куски, не словарь смысла (храповик правил: без новых регулярок).
_TAIL_PUNCT = ",.!?;:"
_TAIL_WORDS = ("а", "но", "—", "–")
# Основы, которые на вопрос «Отменить текущий процесс записи?» значат согласие отменить, а не оговорку.
_CANCEL_STEMS = ("отмен", "передума")


def buttons_on() -> bool:
    """Вопросы-слоты кнопок и остатки класса 5 — через функцию: `pending_buttons = on`."""

    from .runtime_config import config as cfg

    return str(getattr(cfg, "MR_PENDING_BUTTONS", "off") or "").strip().lower() == "on"


def booking_target_on() -> bool:
    """Вопрос цели записи — вопрос-слот функции: `booking_target_slot_llm = on` (BUG-2026-10-25-BOOK-BUTTON-REPLY-AS-TOPIC-SWITCH)."""

    from .runtime_config import config as cfg

    return str(getattr(cfg, "MR_BOOKING_TARGET_SLOT_LLM", "off") or "").strip().lower() == "on"


def slot_topics() -> dict[str, str | None]:
    """Темы вопросов-слотов этого хода. Под `booking_target_slot_llm = on` — и цель записи («К какому врачу или на какую
    услугу записать?»): слота для реплики целиком нет — врача, специальность или услугу разбирают правила и NLU, решение
    «это ответ» держит тему записи, и предпроверка записи не принимает ответ за смену темы."""

    return {**SLOT_TOPICS, "APPOINTMENT": None} if booking_target_on() else SLOT_TOPICS


def current(question_type: str | Question | None = None, text: str | None = None) -> Answer | None:
    """Решение этого хода; с типом — только если оно о вопросе этого типа; с репликой — только если о ней."""

    found = _CURRENT.get()
    if found is None:
        return None
    wanted = question_type.value if isinstance(question_type, Question) else question_type
    if wanted is not None and found.type != wanted:
        return None
    if text is not None and found.text != str(text or "").strip():
        return None
    return found


def reset() -> None:
    """Начало хода: решения прошлого хода нет."""

    _CURRENT.set(None)


@contextlib.contextmanager
def aside() -> Iterator[None]:
    """Побочный ответ внутри хода (уточняющий вопрос о вопросе бота): обычный путь идёт так, будто вопроса нет; после —
    решение хода снова видно обработчикам."""

    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _anchor(norm: str) -> Outcome | None:
    if norm in _CONFIRM_YES_EXACT:
        return Outcome.YES
    if norm in _CONFIRM_NO_EXACT:
        return Outcome.NO
    return None


def _head_and_tail(raw: str) -> tuple[str, str, bool] | None:
    """Первый разделитель реплики: (голова, хвост без связки, связка — противительная «а» / «но»); нет — None."""

    words = raw.split()
    for index, word in enumerate(words):
        if index and word.lower() in _TAIL_WORDS:
            head, tail, contrast = words[:index], words[index + 1 :], word.lower() in _TAIL_WORDS[:2]
        elif word[-1] in _TAIL_PUNCT and index < len(words) - 1:
            head, tail, contrast = words[: index + 1], words[index + 1 :], False
        else:
            continue
        if tail and tail[0].lower() in _TAIL_WORDS:
            contrast = contrast or tail[0].lower() in _TAIL_WORDS[:2]
            tail = tail[1:]  # «да, но для ребёнка» — связка не хвост
        return " ".join(head), " ".join(tail), contrast
    return None


def split_anchor(text: str) -> tuple[Outcome, str | None, bool] | None:
    """Реплика — голый якорь согласия/отказа или «якорь + разделитель + хвост»: (исход, хвост или None, хвост через
    противительную связку «а» / «но»); иначе None.

    Отказ или оговорка в хвосте за «да» («да нет», «ладно, не надо») — не якорь: решает LLM.
    """

    raw = str(text or "").strip()
    whole = _anchor(_normalize_confirmation_text(raw))
    if whole is not None:
        return whole, None, False
    split = _head_and_tail(raw)
    if split is None:
        return None
    head, tail, contrast = split
    outcome = _anchor(_normalize_confirmation_text(head))
    if outcome is None or not tail:
        return None
    if outcome is Outcome.YES and _has_refusal_or_hedge(_normalize_confirmation_text(raw)):
        return None
    return outcome, tail, contrast


def number(text: str) -> int | None:
    """Реплика — только номер пункта («1», «2.», «№3», «4)»); иначе None. Одно определение «реплика — номер»: им же путь цены
    читает номер строки показанного списка (`_prices_helpers._price_family_row_from_context`)."""

    bare = str(text or "").strip().lstrip("№").strip().rstrip(".)").strip()
    return int(bare) if bare.isdigit() and len(bare) <= 2 else None


def exact_pick(text: str, options: tuple[str, ...]) -> str | None:
    """Реплика — дословно один из показанных пунктов (без регистра и знаков) или его номер (1…N); иначе None."""

    if not options:
        return None
    index = number(text)
    if index is not None:
        return options[index - 1] if 1 <= index <= len(options) else None
    norm = _normalize_confirmation_text(text)
    if not norm:
        return None
    hits = [option for option in options if _normalize_confirmation_text(option) == norm]
    return hits[0] if len(hits) == 1 else None


def shown_rows(options: tuple[str, ...]) -> tuple[str, ...]:
    """Строки показанного списка цен среди пунктов вопроса `show_all` (`own_question_reply = on`): все пункты, кроме «все»;
    под off пункт один — «все», строк нет."""

    return tuple(option for option in options if option not in SHOW_ALL_OPTIONS)


def shown_row_number(text: str) -> int | None:
    """Номер строки показанного списка цен, которую выбрал пациент. Под `own_question_reply = on` — решение этой функции на
    вопрос о списке (номер, «первый», «давайте второй», название строки своими словами — пункт-строка); без такого решения —
    реплика-номер («1», «2.»), как было. Одно место, где путь цены и классификатор читают выбор строки. Пункты — из поля
    ожидания на начало хода (`expects.previous`): инструменты поля в сущностях не видят (`expects.without`).

    :param text: реплика пациента
    :return: номер строки с 1; не выбор строки — None
    """

    found = current(Question.SHOW_ALL)
    if own_question_on() and found is not None and found.decided and found.outcome is Outcome.PICK and not found.rest:
        from . import expects

        asked = expects.previous() or {}
        rows = shown_rows(tuple(asked.get("options") or ())) if asked.get("type") == Question.SHOW_ALL.value else ()
        return rows.index(found.pick) + 1 if found.pick in rows else None
    return number(text)


def _is_show_all_reply(text: str) -> bool:
    """Реплика — пункт «все» подсказки «…напишите: "все"» своими словами из одного определения пути цены
    (`_prices_helpers._is_price_show_all_request`: «все», «покажи все», «все варианты»)."""

    # Импорт в функции: модуль хода не тянет сервисный слой при своём импорте.
    from .services._prices_helpers import _is_price_show_all_request

    return _is_price_show_all_request(text)


def _from_choice(choice: Choice, options: tuple[str, ...], *, many: bool = False) -> tuple[Outcome, str | None] | None:
    """Выбор модуля → (исход, пункт); сбой или не по контракту — None (запасной путь). `many` — пунктов может быть
    несколько: пункт — первый по списку, все — `_picked_many`."""

    def _pick(ids: tuple[str, ...]) -> str | None:
        if many and ids and all(str(i).isdigit() for i in ids):
            ids = (str(min(int(i) for i in ids)),)
        if len(ids) != 1 or not str(ids[0]).isdigit():
            return None
        index = int(ids[0])
        return options[index - 1] if 1 <= index <= len(options) else None

    if choice.status is Status.PICKED:
        picked = _pick(choice.ids)
        return (Outcome.PICK, picked) if picked is not None else None
    if choice.status is Status.OUTCOME:
        picked = _pick(choice.ids) if choice.ids else None
        if picked is not None and choice.outcome == Outcome.YES:
            return Outcome.PICK, picked  # «да, второй» — пункт, исход его не теряет
        try:
            return Outcome(choice.outcome), None
        except ValueError:
            return None
    if choice.status is Status.NONE_FIT:
        return Outcome.NONE, None
    return None


def _picked_many(choice: Choice, options: tuple[str, ...]) -> tuple[str, ...]:
    """Все выбранные пункты в порядке списка."""

    indices = sorted({int(i) for i in choice.ids if str(i).isdigit() and 1 <= int(i) <= len(options)})
    return tuple(options[i - 1] for i in indices)


def _refusal_kinds(norm: str) -> tuple[bool, bool]:
    """(отрицание, оговорка) по `policies._REFUSAL_OR_HEDGE_RE`: первая ветка — «не / нет / неа / ни» (совпадения не
    длиннее трёх букв), вторая — основы «отмен…», «передума…», «подума…», «позже», «потом», «перезвон…». Для вопроса
    об отмене оформления «отмен…» и «передума…» — согласие отменить, не оговорка."""

    negation = hedge = False
    for match in _REFUSAL_OR_HEDGE_RE.finditer(_AFFIRMATIVE_IDIOMS_RE.sub(" ", norm)):
        word = match.group(0)
        if len(word) <= 3:
            negation = True
        elif not word.startswith(_CANCEL_STEMS):
            hedge = True
    return negation, hedge


def _negated(norm: str) -> bool:
    """В реплике есть отрицание «не / нет / неа / ни»."""

    return _refusal_kinds(norm)[0]


def _veto(question_type: str, outcome: Outcome, text: str) -> Outcome:
    """Вето правил после решения: отказ или оговорка гасят «да» (дизайн §2, существующие проверки — одним местом)."""

    if outcome is not Outcome.YES:
        return outcome
    norm = _normalize_confirmation_text(text)
    policy = POLICY.get(question_type)
    if policy is not None and policy.veto_is_no:
        negation, hedge = _refusal_kinds(norm)
        if negation:
            return Outcome.NO  # «не отменяйте», «нет-нет» — собранное цело
        return Outcome.NONE if hedge else outcome  # «хорошо, подумаю» — не стирать
    return Outcome.NONE if _has_refusal_or_hedge(norm) else outcome


def _remember(found: Answer) -> Answer:
    _CURRENT.set(found)
    fields: dict[str, Any] = {"tu_pending_type": found.type, "tu_pending_outcome": found.outcome.value, "tu_pending_src": found.source.value}
    if found.rest:
        fields["tu_pending_rest"] = 1
    turn_telemetry.note(**fields)
    return found


async def answer(
    question_type: str,
    question: str,
    options: tuple[str, ...],
    text: str,
    *,
    filled: bool = False,
    confirm_topic: bool = False,
    button: str = "",
) -> Answer | None:
    """Ответ пациента на висящий вопрос этого типа — одно решение на ход (см. модуль).

    :param question_type: `_expects.type`
    :param question: текст вопроса, каким его видел пациент, без ПДн (`_expects.question`)
    :param options: показанные пункты (`_expects.options`)
    :param text: реплика пациента
    :param filled: вопрос-слот — правила уже разобрали в реплике значение слота (сужение без LLM)
    :param confirm_topic: вопрос кнопки меню (`button_flow.judged_by_llm`): значение, найденное правилами, LLM проверяет только
        на «другую тему» — правила находят значение и в реплике о другом; прочие её исходы — сужение правил, как было
    :param button: нажатая кнопка, её тема и вопрос бота (`button_flow.judge_context`) — вопрос-слот судит `FREE_BUTTON_TASK`;
        пусто — прежний судья по одному вопросу бота
    :return: решение (оно же — `current()` до конца хода); тип не из таблицы или пустая реплика — None
    """

    reply = str(text or "").strip()
    policy = POLICY.get(question_type)
    if policy is None or not reply or (question_type in _BUTTON_TYPES and not buttons_on()):
        return None
    if policy.polarity is Polarity.FREE:
        return await _free_answer(question_type, question, reply, filled=filled, confirm_topic=confirm_topic, button=button)
    many = policy.many and buttons_on()

    if question_type == Question.SHOW_ALL.value and number(reply) is not None:
        # Цифра на подсказку «напишите: "все"» — номер строки показанного списка, а не пункт подсказки (пункт один — «все»):
        # нумерации совпадали, и «1» раскрывало весь список (BUG-2026-10-24-SHOW-ALL-HINT-ON-PROD). Строку с этим номером
        # отвечает путь цены (`_prices_helpers._price_family_row_from_context`), без LLM. Строки списка — пункты вопроса
        # (`own_question_reply = on`): номер — пункт-строка (ход поглощён ответом), номера нет в списке — ничего из этого.
        row = exact_pick(reply, shown_rows(options))
        if row is not None:
            return _remember(Answer(question_type, Outcome.PICK, pick=row, source=Source.FAST, text=reply))
        return _remember(Answer(question_type, Outcome.NONE, source=Source.FAST, text=reply))
    picked = exact_pick(reply, options)
    if picked is None and question_type == Question.SHOW_ALL.value and _is_show_all_reply(reply):
        picked = SHOW_ALL_OPTIONS[0]  # «покажи все», «все варианты» — тот же единственный пункт, одно определение
    if picked is not None:
        return _remember(Answer(question_type, Outcome.PICK, pick=picked, source=Source.FAST, text=reply))
    if policy.polarity is Polarity.YES_NO:
        anchored = split_anchor(reply)
        if anchored is not None:
            outcome, rest, contrast = anchored
            if outcome is Outcome.YES and rest and policy.tail_yes == "execute":
                rest = None  # «да, пожалуйста, соедините» — согласие целиком, исполняет оффер
            if rest is None or outcome is Outcome.NO:
                return _remember(Answer(question_type, _veto(question_type, outcome, reply), rest=rest, source=Source.FAST, text=reply))
            if policy.tail_yes == "rest" or contrast:
                # «да, но для ребёнка», «да, как подготовиться?» — не «да»: хвост идёт обычным путём
                return _remember(Answer(question_type, Outcome.NONE, rest=rest, source=Source.FAST, text=reply))
            # `llm`: «да» с хвостом через запятую («да, верно, его», «да, давайте отменим») решает модель по всей реплике

    from .pending_reply import PENDING_TASK

    try:
        choice = await choose(MANY_TASK if many else PENDING_TASK, reply, [Option(id=str(i), text=o) for i, o in enumerate(options, 1)], asked=question)
    except Exception:  # новый слой: его сбой не ломает ход — решают прежние правила
        logger.warning("pending_answer_failed", exc_info=True)
        choice = None
    decided = _from_choice(choice, options, many=many) if choice is not None else None
    if decided is None:
        return _remember(Answer(question_type, Outcome.NONE, source=Source.FALLBACK, text=reply))
    outcome, pick = decided
    picks = _picked_many(choice, options) if many and outcome is Outcome.PICK and choice is not None else ()
    found = _remember(Answer(question_type, _veto(question_type, outcome, reply), pick=pick, source=Source.LLM, text=reply, picks=picks))
    return _refunded(found)


def _refunded(found: Answer) -> Answer:
    """Решение, не закрывшее ход, возвращает ходу одно решение LLM (Q5 18.10)."""

    from . import llm_runtime

    if not found.closes and llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_pending_refund=1)
    return found


async def _free_answer(question_type: str, question: str, reply: str, *, filled: bool, confirm_topic: bool = False, button: str = "") -> Answer:
    """Свободный ответ на вопрос-слот: голый отказ — «не могу ответить»; правила разобрали значение — ответ (у вопроса кнопки
    — если LLM не сказала «другая тема», `confirm_topic`); иначе LLM (`FREE_TASK`, у вопроса кнопки под `button_reply_judge` —
    `FREE_BUTTON_TASK` с кнопкой и её темой); сбой — запасной путь (как было)."""

    task, asked = ((FRAME_BUTTON_TASK if frame_judge_on() else FREE_BUTTON_TASK), button) if button else (FREE_TASK, question)
    if _anchor(_normalize_confirmation_text(reply)) is Outcome.NO:
        return _remember(Answer(question_type, Outcome.NO, source=Source.FAST, text=reply))  # «нет» — не фамилия
    if filled and confirm_topic:
        try:
            choice = await choose(task, reply, [], asked=asked)
        except Exception:  # новый слой: его сбой не ломает ход — сужение правил, как было
            logger.warning("pending_answer_failed", exc_info=True)
            choice = None
        if choice is not None and choice.status is Status.OUTCOME and choice.outcome == Outcome.NEW_TOPIC.value:
            return _refunded(_remember(Answer(question_type, Outcome.NEW_TOPIC, source=Source.LLM, text=reply)))
        return _remember(Answer(question_type, Outcome.ANSWER, source=Source.FAST, text=reply))
    if filled:
        return _remember(Answer(question_type, Outcome.ANSWER, source=Source.FAST, text=reply))
    try:
        choice = await choose(task, reply, [], asked=asked)
    except Exception:  # новый слой: его сбой не ломает ход — решают прежние правила
        logger.warning("pending_answer_failed", exc_info=True)
        choice = None
    if choice is None or choice.status not in (Status.OUTCOME, Status.NONE_FIT):
        return _remember(Answer(question_type, Outcome.NONE, source=Source.FALLBACK, text=reply))
    if choice.status is Status.OUTCOME and choice.outcome == _DOMAIN:
        # Вариант Б: свой вопрос в теме кнопки — в рамке кнопки; под `pending_in_frame` — свой исход, не ответ на слот (Н-1).
        outcome = Outcome.IN_FRAME if in_frame_on() else Outcome.ANSWER
    else:
        outcome = Outcome(choice.outcome) if choice.status is Status.OUTCOME else Outcome.NONE
    return _refunded(_remember(Answer(question_type, outcome, source=Source.LLM, text=reply)))


def expected_from(entities: Mapping[str, Any] | None) -> tuple[str, str, tuple[str, ...]] | None:
    """(тип, вопрос без ПДн, пункты) из `_expects`; нет или не по контракту — None."""

    from .pending_reply import _without_patient_name, expectation

    expected = expectation(entities)
    if expected is None or expected.type not in POLICY:
        return None
    return expected.type, _without_patient_name(expected.question, entities or {}), expected.options


# --------------------------------------------------------------------------- исполнение решения обработчиками

# Исходы, которые отвечают на вопрос: явный запрос оператора в такой реплике — часть ответа, не просьба о переводе
# («нет, оператор не нужен», «не надо оператора» — отказ на оффер; «да, соедините» — согласие, его исполняет оффер).
_ANSWERING = frozenset({Outcome.YES, Outcome.NO, Outcome.PAUSE, Outcome.PICK})


def kind(question_type: str | Question, rules: Any) -> str:
    """Что ответил пациент — для обработчика вопроса: решение функции этого хода, а без него (выключатель, сбой LLM) —
    прежние правила обработчика (`rules()` → yes / no / other; решение владельца Q1 18.10).

    :return: yes | no | pause | pick | new_topic | other
    """

    found = current(question_type)
    if found is None or not found.decided:
        return rules()
    if found.outcome is Outcome.NONE:
        return "other"
    return found.outcome.value


def passes(question_type: str | Question) -> bool:
    """Ответ уводит ход дальше: другая тема или хвост реплики («нет, мне ферритин») — обработчик снимает свой вопрос и
    отдаёт ход обычному пути (хвост роутер уже подставил вместо реплики)."""

    found = current(question_type)
    return found is not None and found.decided and (found.outcome is Outcome.NEW_TOPIC or bool(found.rest))


def picked(question_type: str | Question) -> str | None:
    """Выбранный пункт показанного списка — решение функции этого хода; иначе None."""

    found = current(question_type)
    return found.pick if found is not None and found.decided and found.outcome is Outcome.PICK else None


def answers_question(found: Answer | None) -> bool:
    """Реплика — ответ на вопрос (да / нет / пауза / пункт), решённый функцией: явный запрос оператора в ней — часть
    ответа, не просьба о переводе."""

    if found is None or not found.decided or found.outcome not in _ANSWERING:
        return False
    if found.type == Question.SLOT_VALUE.value:
        return False  # «не помню номер, дайте оператора» — у вопроса-слота оффера нет: просьба о переводе исполняется
    # «нет, оператор не нужен» — хвост за отказом тоже про отказ: слово «оператор» в нём не просьба о переводе.
    return not found.rest or found.outcome is Outcome.NO


def operator_requested(question_type: str | Question, text: str) -> bool:
    """Явный запрос оператора внутри обработчика висящего вопроса — правило сужает, решение функции первее (вето в
    сторону «нет перевода»: «не надо оператора» на вопрос — не перевод)."""

    from .recovery_policy import explicit_operator_requested

    if answers_question(current(question_type)):
        return False
    return explicit_operator_requested(text)


def shows_all() -> bool:
    """Ответ на подсказку «…напишите: "все"» — «показать все»: «да» или пункт «все», решённые одной функцией."""

    found = current(Question.SHOW_ALL)
    if found is None or not found.decided or found.rest:
        return False
    return found.outcome is Outcome.YES or (found.outcome is Outcome.PICK and found.pick in SHOW_ALL_OPTIONS)


# Исходы, которыми пациент ответил на вопрос бота, не задав своего: да / нет / пауза / пункт / значение слота.
_ABSORBING = frozenset({Outcome.YES, Outcome.NO, Outcome.PAUSE, Outcome.PICK, Outcome.ANSWER})


def absorbs_turn() -> bool:
    """Ход — ответ на свой вопрос бота (`own_question_reply = on`, BUG-2026-10-30-OWN-LIST-REPLY): одна функция решила, что
    реплика ответила на висящий вопрос (да / нет / пауза / пункт списка / значение слота), без хвоста. Тип вопроса такого хода
    задал вопрос бота: «1» на список цен — строка списка, а не «ни одно» по голой реплике. Свой вопрос в рамке кнопки
    (`in_frame`), «ничего из этого», другая тема, хвост и сбой LLM — ход не поглощён."""

    if not own_question_on():
        return False
    found = current()
    return found is not None and found.decided and not found.rest and found.outcome in _ABSORBING


_FLOW_TYPES = (Question.APPOINTMENT_ACTION, Question.APPOINTMENT_BRANCH, Question.BOOKING_SERVICE)


def keeps_flow() -> bool:
    """Реплика ответила на вопрос шага записи (пункт или «да», решено одной функцией): ход остаётся в записи, как бы
    ни назвал реплику NLU («1» — не номер анализа)."""

    for question in _FLOW_TYPES:
        found = current(question)
        if found is not None and found.decided and found.outcome in (Outcome.PICK, Outcome.YES) and not found.rest:
            return True
    return False


def picked_all(question_type: str | Question) -> tuple[str, ...]:
    """Все выбранные пункты списка, где можно взять несколько («обе»), — решение функции этого хода; иначе пусто."""

    found = current(question_type)
    if found is None or not found.decided or found.outcome is not Outcome.PICK or found.rest:
        return ()
    return found.picks


def by_understanding_on() -> bool:
    """«Продолжает ли реплика висящий вопрос» решает одно решение хода — эта функция: `[MESSENGER_ROUTER]
    expects_by_understanding = on` (шаг 3 деплоя 22, STEP-2026-10-30-EXPECTS-BY-UNDERSTANDING). off — запасники решают по форме
    реплики, как было."""

    from .runtime_config import config as cfg

    return str(getattr(cfg, "MR_EXPECTS_BY_UNDERSTANDING", "off") or "").strip().lower() == "on"


# Вопросы, чей ответ заполняет недостающий слот `_pending` (тема висящего вопроса): вопрос-слот, услуга, действие и филиал записи.
PENDING_QUESTIONS = (Question.SLOT_VALUE, Question.BOOKING_SERVICE, Question.APPOINTMENT_ACTION, Question.APPOINTMENT_BRANCH)


def continues(*questions: Question) -> bool | None:
    """Продолжает ли реплика висящий вопрос — одно решение хода для запасников (предпроверка записи, планировщик) под
    `expects_by_understanding = on`: False — другая тема, True — нет (ответ, пункт, «да» / «нет», пауза, свой
    вопрос в рамке, «ничего из этого» модели); None — решения нет: выключатель off, вопрос другого типа, сбой LLM (запасной путь
    правил), хвост реплики идёт обычным путём. Тогда запасник решает по своим правилам, как было.

    :param questions: типы вопроса, о которых спрашивает читатель; пусто — любой
    """

    if not by_understanding_on():
        return None
    found = current()
    if found is None or not found.decided or found.rest:
        return None
    if questions and found.type not in {q.value for q in questions}:
        return None
    return found.outcome is not Outcome.NEW_TOPIC


def acted(site: str) -> None:
    """След хода: запасник `site` поступил по одному решению, а не по своим правилам (`tu_expects_acted`)."""

    turn_telemetry.append("tu_expects_acted", site, once=True)  # планировщик читает тему дважды за ход (разбор подготовки, план)


def slot_value(text: str) -> str:
    """Реплика как значение слота: одна строка, без концевой пунктуации, разумной длины; длиннее — не значение слота."""

    value = " ".join(str(text or "").split()).strip(" .,!?;:")
    return value if 0 < len(value) <= _SLOT_VALUE_MAX else ""


# Просьба инструмента темы назвать услугу («Уточните название…», «Введите конкретное название процедуры…») — висящий вопрос со
# свободным ответом (`own_question_reply`; идея BUG-2026-10-26-CLARIFY-REPLY-NOT-ACCEPTED): пометка ответа инструмента, слот
# названия темы (как у вопроса планировщика, `policies.REQUIRED_SLOTS`) и пометка `_pending`, что вопрос поставлен этой просьбой.
ASKS_NAME_KEY = "asks_service_name"
CLARIFY_KEY = "clarify"
_NAME_SLOTS = {"PRICE": "service_name"}


def ask_name_after_clarify(label: str, payloads: Iterable[Any], handoff: bool, state: Any, memory: Any) -> bool:
    """Инструмент темы попросил назвать услугу — висящий вопрос со свободным ответом (`slot_value`, под `pending_buttons`):
    `_pending` темы с пометкой `CLARIFY_KEY`. Ответ разбирает эта функция (LLM, `FREE_TASK`), а не правило «похоже на название»
    (`flow_policy._looks_like_price_service_reply` ловит и «спасибо», и «хочу записаться к кардиологу»). Другой висящий вопрос
    хода (оффер оператора, анти-залип) не перебивается.

    :return: вопрос поставлен
    """

    slot = _NAME_SLOTS.get(label)
    if not own_question_on() or not buttons_on() or slot is None or handoff or memory.get_pending(state):
        return False
    if not any(isinstance(payload, dict) and payload.get(ASKS_NAME_KEY) for payload in payloads):
        return False
    memory.set_pending(state, label=label, missing_slots=[slot])
    memory.get_pending(state)[CLARIFY_KEY] = True
    if state.last_entities.get("_secondary_offer_pending"):
        # Оффер второй темы реплики текста в ответе не имеет (`orchestrator._mark_secondary_offer_pending`): пациент видит просьбу —
        # ответ на неё, а не «да» на скрытый оффер; вторая тема остаётся в очереди и будет предложена после ответа по услуге.
        state.last_entities["_secondary_offer_pending"] = False
    from . import expects

    expects.ask(Question.SLOT_VALUE, kind=expects.FREE)
    turn_telemetry.note(tu_clarify_topic=label)
    return True


def clarify_question(pending: Any) -> bool:
    """Висящий вопрос поставлен просьбой инструмента назвать услугу (`ask_name_after_clarify`)."""

    return isinstance(pending, dict) and bool(pending.get(CLARIFY_KEY))


def settle_clarify(answered: Answer | None, state: Any, memory: Any, spent_before: int) -> None:
    """После решения функции об ответе на просьбу назвать услугу: «ответ» (решила LLM) — вопрос остаётся до NLU (тема —
    `clarify_topic`), решение заменяет классификатор темы и лимит решений хода не тратит (как судья кнопок цены,
    `button_flow.refund_topic_judge`); иначе (другая тема, «нет», пауза, ничего, сбой LLM) — вопрос снят, реплика идёт обычным
    путём, как без выключателя."""

    if not clarify_question(memory.get_pending(state)):
        return
    if answered is None or answered.source is not Source.LLM or answered.outcome is not Outcome.ANSWER:
        memory.clear_pending(state)
        return
    from . import llm_runtime

    if llm_runtime.turn_decisions_spent() > spent_before and llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_clarify_refund=1)


def clarify_topic(pending: Any) -> str:
    """Тема ответа на просьбу назвать услугу, который эта функция признала ответом: тема вопроса (NLU её не угадывает по словам
    реплики — «узи щитовидной железы» в ответ на вопрос о цене не уходит к врачам); иначе пусто."""

    found = slot_answer()
    if not clarify_question(pending) or found is None or found.outcome is not Outcome.ANSWER:
        return ""
    return str(pending.get("label") or "")


def slot_answer() -> Answer | None:
    """Решение одной функции об ответе на вопрос-слот этого хода (`pending_buttons`); нет или сбой LLM — None."""

    found = current(Question.SLOT_VALUE)
    return found if found is not None and found.decided else None


def branch_side_question(found: Answer | None) -> bool:
    """На «Какой филиал вам удобен?» пациент не ответил, а спросил о самом вопросе («а какой ближе к …?», «а сколько стоит
    приём?»): решение модели «ничего из этого» (правило 4 промпта), без хвоста — под `pending_buttons`."""

    return (
        buttons_on()
        and found is not None
        and found.type == Question.APPOINTMENT_BRANCH.value
        and found.source is Source.LLM
        and found.outcome is Outcome.NONE
        and not found.rest
    )


def branch_answer(text: str) -> tuple[str, str] | None:
    """Ответ на вопрос «Какой филиал вам удобен?» — (исход, выбранный пункт или ""): решение одной функции этого хода,
    без него (сбой LLM) — ответ, который узнали правила предпроверки записи (`pending_reply.branch_reply`, исход
    `rules`); решения нет — None."""

    found = current(Question.APPOINTMENT_BRANCH)
    if found is not None and found.decided:
        return found.outcome.value, found.pick or ""
    from .pending_reply import branch_reply

    return branch_reply(text)
