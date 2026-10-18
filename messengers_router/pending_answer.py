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

Выключатель — прежний `[MESSENGER_ROUTER] pending_reply`: функция решает только под `on`, все типы разом
(`pending_reply_types` для неё не читается). `off` / `shadow` — как было.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from . import turn_telemetry
from .choice import Choice, Option, Status, choose
from .expects import (
    ANESTHESIA_OPTIONS,
    ANY_BRANCH_OPTION,
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


class Source(StrEnum):
    """Кто решил."""

    FAST = "fast"  # сужение без LLM: якорь или дословный пункт / номер
    LLM = "llm"
    FALLBACK = "fallback"  # LLM недоступна или ответила не по контракту — обработчик решает прежними правилами


class Polarity(StrEnum):
    """Вид вопроса для сужения: «да/нет» — якоря согласия и отказа значат «да» / «нет»; выбор — не значат."""

    YES_NO = "yes_no"
    PICK = "pick"


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
    """

    polarity: Polarity
    tail_yes: str = "rest"
    veto_is_no: bool = False


POLICY: dict[str, TypePolicy] = {
    Question.OPERATOR_OFFER.value: TypePolicy(Polarity.YES_NO, tail_yes="execute"),
    Question.CATALOG_CONFIRM.value: TypePolicy(Polarity.YES_NO, tail_yes="llm"),
    Question.APPOINTMENT_ABORT.value: TypePolicy(Polarity.YES_NO, tail_yes="llm", veto_is_no=True),
    Question.SECONDARY_OFFER.value: TypePolicy(Polarity.YES_NO),
    Question.SHOW_ALL.value: TypePolicy(Polarity.YES_NO, tail_yes="llm"),
    Question.APPOINTMENT_ACTION.value: TypePolicy(Polarity.PICK),
    Question.COMPOUND_PRICE.value: TypePolicy(Polarity.PICK),
    Question.ANESTHESIA.value: TypePolicy(Polarity.PICK),
    Question.BOOKING_SERVICE.value: TypePolicy(Polarity.PICK),
    Question.APPOINTMENT_BRANCH.value: TypePolicy(Polarity.PICK),
}

# Варианты, которые бот называет словами самого вопроса (не данными состояния) — для сверки в тестах.
FIXED_OPTIONS: dict[str, tuple[str, ...]] = {
    Question.APPOINTMENT_ACTION.value: APPOINTMENT_ACTION_OPTIONS,
    Question.ANESTHESIA.value: ANESTHESIA_OPTIONS,
    Question.SHOW_ALL.value: SHOW_ALL_OPTIONS,
}
ANY_BRANCH = ANY_BRANCH_OPTION


@dataclass(frozen=True)
class Answer:
    """Решение о реплике-ответе на висящий вопрос.

    :param type: тип вопроса (`_expects.type`)
    :param outcome: исход
    :param pick: текст выбранного пункта (из `_expects.options`) — у `pick`, иначе None
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

    @property
    def decided(self) -> bool:
        """Решение есть (не запасной путь): обработчик исполняет его, а не свои правила."""

        return self.source is not Source.FALLBACK

    @property
    def closes(self) -> bool:
        """Ответ на вопрос дан: да / выбор / пауза; прочее (нет, другая тема, ничего, хвост) — дальше обычный путь."""

        return self.outcome in (Outcome.YES, Outcome.PICK, Outcome.PAUSE) and not self.rest


_CURRENT: contextvars.ContextVar[Answer | None] = contextvars.ContextVar("mr_pending_answer", default=None)

# «якорь + разделитель + хвост»: знак препинания на конце слова или отдельное слово-связка / тире — первое в реплике.
# Механика разбиения на куски, не словарь смысла (храповик правил: без новых регулярок).
_TAIL_PUNCT = ",.!?;:"
_TAIL_WORDS = ("а", "но", "—", "–")
# Основы, которые на вопрос «Отменить текущий процесс записи?» значат согласие отменить, а не оговорку.
_CANCEL_STEMS = ("отмен", "передума")


def on() -> bool:
    """Функция решает: `[MESSENGER_ROUTER] pending_reply = on`."""

    from .pending_reply import mode

    return mode() == "on"


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


def _number(text: str) -> int | None:
    """Реплика — только номер пункта («1», «2.», «№3», «4)»); иначе None."""

    bare = str(text or "").strip().lstrip("№").strip().rstrip(".)").strip()
    return int(bare) if bare.isdigit() and len(bare) <= 2 else None


def exact_pick(text: str, options: tuple[str, ...]) -> str | None:
    """Реплика — дословно один из показанных пунктов (без регистра и знаков) или его номер (1…N); иначе None."""

    if not options:
        return None
    index = _number(text)
    if index is not None:
        return options[index - 1] if 1 <= index <= len(options) else None
    norm = _normalize_confirmation_text(text)
    if not norm:
        return None
    hits = [option for option in options if _normalize_confirmation_text(option) == norm]
    return hits[0] if len(hits) == 1 else None


def _is_show_all_reply(text: str) -> bool:
    """Реплика — пункт «все» подсказки «…напишите: "все"» своими словами из одного определения пути цены
    (`_prices_helpers._is_price_show_all_request`: «все», «покажи все», «все варианты»)."""

    # Импорт в функции: модуль хода не тянет сервисный слой при своём импорте (как `pending_reply.rules_kind`).
    from .services._prices_helpers import _is_price_show_all_request

    return _is_price_show_all_request(text)


def _from_choice(choice: Choice, options: tuple[str, ...]) -> tuple[Outcome, str | None] | None:
    """Выбор модуля → (исход, пункт); сбой или не по контракту — None (запасной путь)."""

    def _pick(ids: tuple[str, ...]) -> str | None:
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


async def answer(question_type: str, question: str, options: tuple[str, ...], text: str) -> Answer | None:
    """Ответ пациента на висящий вопрос этого типа — одно решение на ход (см. модуль).

    :param question_type: `_expects.type`
    :param question: текст вопроса, каким его видел пациент, без ПДн (`_expects.question`)
    :param options: показанные пункты (`_expects.options`)
    :param text: реплика пациента
    :return: решение (оно же — `current()` до конца хода); тип не из таблицы или пустая реплика — None
    """

    reply = str(text or "").strip()
    policy = POLICY.get(question_type)
    if policy is None or not reply:
        return None

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
    from . import llm_runtime

    try:
        choice = await choose(PENDING_TASK, reply, [Option(id=str(i), text=o) for i, o in enumerate(options, 1)], asked=question)
    except Exception:  # новый слой: его сбой не ломает ход — решают прежние правила
        logger.warning("pending_answer_failed", exc_info=True)
        choice = None
    decided = _from_choice(choice, options) if choice is not None else None
    if decided is None:
        return _remember(Answer(question_type, Outcome.NONE, source=Source.FALLBACK, text=reply))
    outcome, pick = decided
    found = _remember(Answer(question_type, _veto(question_type, outcome, reply), pick=pick, source=Source.LLM, text=reply))
    if not found.closes and llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_pending_refund=1)
    return found


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


_FLOW_TYPES = (Question.APPOINTMENT_ACTION, Question.APPOINTMENT_BRANCH, Question.BOOKING_SERVICE)


def keeps_flow() -> bool:
    """Реплика ответила на вопрос шага записи (пункт или «да», решено одной функцией): ход остаётся в записи, как бы
    ни назвал реплику NLU («1» — не номер анализа)."""

    for question in _FLOW_TYPES:
        found = current(question)
        if found is not None and found.decided and found.outcome in (Outcome.PICK, Outcome.YES) and not found.rest:
            return True
    return False


def branch_answer(text: str) -> tuple[str, str] | None:
    """Ответ на вопрос «Какой филиал вам удобен?» — (исход, выбранный пункт или ""): решение одной функции этого хода,
    без неё — прежний клиент `pending_reply` (выключатель ниже `on`); решения нет — None."""

    found = current(Question.APPOINTMENT_BRANCH)
    if found is not None and found.decided:
        return found.outcome.value, found.pick or ""
    from .pending_reply import branch_reply

    return branch_reply(text)
