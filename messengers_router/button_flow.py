"""Кнопка меню после нажатия: тема ответа на её вопрос известна (решение владельца 25.10 — ~90 % пациентов придут через меню).

Нажатие листа меню — явный выбор темы (`button_menu`). Кнопка режима `slot` спрашивает своё («Какой анализ интересует?»,
«К какой процедуре нужна подготовка?»), и висящий вопрос (`_pending`) помнит, какая кнопка его задала, —
`_pending[QUESTION_KEY]`. Пока вопрос висит, тема ответа на него известна из нажатия; кто этим пользуется — каждый под своим
выключателем, по умолчанию off (как было):

* `button_answer_topic` (BUG-2026-10-25-BUTTON-ANSWER-CLASSIFIER) — LLM-классификатор темы ответ на вопрос кнопки не
  переспрашивает. Журнал хода прода 14–24.10: после нажатия кнопки-темы следующая реплика ждала классификатор в 35 % ходов,
  с 21.10 — в 59 % (p50 3,7 с, до 12 с), и в половине из них он возвращал ту же тему. Одна функция ответа на висящий вопрос
  (`pending_answer`, вид `free`) решила «это ответ» — тему решает кнопка там, где решала бы LLM-классификатор (правило
  не сработало или его ветка в `topic_by_llm`); надёжные правила и срочное — как всегда. У кнопок цены своего вопроса-слота
  не было — его ответ судила LLM-классификатор; под on его судит та же функция (≈ 1 с вместо 3,7 с), и это решение лимита
  решений хода не тратит: оно — шаг NLU вместо классификатора, а пути цены остаются её решения, как без кнопки. Режим `slot`
  — только вопросы-слоты (подготовка, подбор, результаты): на кнопках цены карточка 25.10 дала +3 чужие цены — путь цены
  выбирает «соседние» строки для услуги, которой нет в прайсе («антитела к аннексину суммарные» → «… V, IgG / IgM»).
* `button_topic_hint` (BUG-2026-10-25-BUTTON-FOREIGN-TOPIC) — на чужую тему в ответ на вопрос кнопки бот подсказывает
  нужную кнопку, а не угадывает тему общим путём (предложение владельца 25.10). Одна функция решила «другая тема», а тему
  угадывала бы LLM-классификатор (надёжное правило другой темы отвечает само) — LLM
  выбирает из закрытого списка листьев меню (принцип 2), где пациент найдёт ответ (`HINT_TASK`, промпт
  `choose_button_hint`): «Это вопрос про … — основная информация в кнопке «…» раздела «…».». В сообщении и ответ на вопрос
  кнопки, и другой вопрос (ТЗ: 1 % сообщений) — ответ по теме кнопки и подсказка в конце. Лист той же темы, что у кнопки
  («Сколько стоит анализ» на «Цена анализа»), — не подсказка: ход идёт как без выключателя. Ничего не подошло, сбой LLM —
  общий путь, как было; это решение лимит решений хода не тратит (как «другая тема» одной функции, Q5 18.10). Ложная
  подсказка безопаснее неверного ответа, но её долю меряет карточка (`button_hint` в журнале хода).
* `button_reply_judge` (BUG-2026-10-27-BUTTON-REPLY-JUDGE-FOREIGN) — судья ответа на вопрос кнопки видит нажатую кнопку и её
  тему (`judge_context`, задача `pending_answer.FREE_BUTTON_TASK`, промпт `choose_pending_free_reply_button`): полный вопрос в
  теме кнопки («Сколько стоит общий анализ крови?» на «Цене анализа») — ответ на её вопрос. Прежний судья видел только вопрос
  бота («Какой анализ интересует?») и почти любой полный вопрос звал «другой темой» (карточка 25.10: на «Цене анализа» 214 из
  343 вопросов по теме), а тему снова решали правила общего пути.
* `button_side_answer` (BUG-2026-10-27-RESULT-BUTTON-SIDE-QUESTION) — на уточняющий вопрос о вопросе кнопки результатов («Где
  на бланке код?», «Как узнать сам результат?» после «Ввести данные») бот отвечает, где код и номер и как посмотреть
  результат, и ждёт реквизиты дальше (`result_side_question`), а не повторяет инструкцию с оффером оператора. Что реплика —
  вопрос о вопросе бота, решает судья одной функции (исход «ничего из этого», правило 3 промпта).
* `button_price_scope` (BUG-2026-10-25-BUTTON-PRICE-SCOPE; ТЗ 17.08, «сужение поиска по теме») — ответ на вопрос кнопки цены
  ищется в её области прайса по данным МИС (`price_scope`): «Цена приёма врача» — услуги без срока готовности (не анализы),
  «Цена анализа» и «Сколько стоит анализ» — анализы (со сроком). Режим `doctor` сужает только «Цену приёма врача»: на «Цену
  анализа» пациенты ведут и УЗИ, и процедуры (набор buttons_v1, wb-metric 25.10: у 84 из 264 размеченных правильный ответ —
  не только анализ), `all` — обе. Как сужать строки — путь цены (`services.prices`).
* `button_topic_firm` (STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM) — тему ответа, которую задала кнопка (`button_answer_topic`),
  переклейки после NLU по буквам не перебивают, как тему от LLM (`topic_by_llm`): «без записи» (`policy_nonbookable_walkin`) и
  П1 по совпадению с каталогом (`test_assist_catalog_hit_is_price`, `understanding_list_is_price`). На «Подобрать анализы»
  подбор решает LLM по данным хода (`test_assist_answer`), а не буквы: «подберите анализы, где сдать кровь» уходил в список
  филиалов, «комплексный анализ на ИППП» — в цену чужой строки.
* `button_answer_topic = frame` (STEP-2026-10-29-BUTTON-FRAME) — кнопка задаёт рамку хода: как `on`, и тему, которую судья
  ответа признал ответом на вопрос кнопки, не перебивают и несрочные правила NLU. Стенд деплоя 20 (`s20con_btn`): у 86 из 236
  провалов кнопок тема ответа ≠ теме нажатого листа, 50 из них — «Цена анализа», где рамки не было (`slot`); правила
  (`rule_appointment`, `doc_request_*`, `unsupported_*`) уводили и ответ, признанный ответом. Срочное и явный перевод на
  оператора (правило с `needs_handoff`) — как всегда. «Другую тему» судьи ответа под `frame` проверяет отдельный короткий
  вопрос LLM — «сообщение в теме нажатой кнопки?» (`FRAME_TASK`, промпт `choose_button_frame`): судья с пятью исходами звал
  «Есть ли у вас 3D УЗИ?», «Можно ли сделать кардиограмму?» на «Цене анализа» другой темой (стенд: 46 ответов, 37 ушли из
  цены). «В теме» — рамка держится; «другое», сбой — общий путь, как было. Листы без своего вопроса («Нужно ли записываться»,
  «Акции и скидки», «Справка для вычета»: бот отвечает на нажатие сразу) держат рамку на одну следующую реплику — метка нажатия
  на ход (как `address_flow`, адресные листы — у `address_followup`), тот же вопрос рамки; «в теме» — тема, флаги и сущности
  листа, сущности реплики — поверх («Записаться на ХГЧ» на «Нужно ли записываться» — без записи ли ХГЧ, решает каталог МИС в
  `router._guard_walkin_named_service`, а не запись по `rule_appointment`).
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import Any, Iterable

from . import button_menu, pending_answer, turn_telemetry
from .choice import Option, Status, Task, choose
from .expects import Question
from .mess_types import ResponseEnvelope
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

QUESTION_KEY = "button"  # `_pending[QUESTION_KEY]` — лист меню, который задал висящий вопрос
# Темы вопросов кнопок, ответ на которые судит одна функция только под `button_answer_topic` (остальные — `SLOT_TOPICS`).
_JUDGED_UNDER_SWITCH = frozenset({"PRICE"})


def _on(key: str) -> bool:
    return str(getattr(_cfg, key, "off") or "").strip().lower() == "on"


def answer_topic_mode() -> str:
    """`[MESSENGER_ROUTER] button_answer_topic = off | slot | on | frame`: тему ответа на вопрос кнопки решает кнопка, не
    LLM-классификатор, — у вопросов-слотов (`slot`: подготовка, подбор анализов, результаты) или и у кнопок цены (`on`);
    `frame` — как `on`, и эту тему не перебивают несрочные правила (`frame_firm`)."""

    mode = str(getattr(_cfg, "MR_BUTTON_ANSWER_TOPIC", "off") or "").strip().lower()
    return mode if mode in ("slot", "on", "frame") else "off"


def _judges_price() -> bool:
    return answer_topic_mode() in ("on", "frame")


def frame_firm() -> bool:
    """`button_answer_topic = frame`: тему ответа на вопрос кнопки не перебивают несрочные правила NLU — кнопка задаёт рамку
    хода (STEP-2026-10-29-BUTTON-FRAME). Срочное и правило с переводом на оператора — как всегда (`nlu_pipeline`)."""

    return answer_topic_mode() == "frame"


def answer_topic_on() -> bool:
    return answer_topic_mode() != "off"


def hint_on() -> bool:
    """`[MESSENGER_ROUTER] button_topic_hint = on`: на чужую тему в ответ на вопрос кнопки — подсказка кнопки меню."""

    return _on("MR_BUTTON_TOPIC_HINT")


def reply_judge_on() -> bool:
    """`[MESSENGER_ROUTER] button_reply_judge = on`: судья ответа на вопрос кнопки видит нажатую кнопку и её тему."""

    return _on("MR_BUTTON_REPLY_JUDGE")


def side_answer_on() -> bool:
    """`[MESSENGER_ROUTER] button_side_answer = on`: уточняющий вопрос о вопросе кнопки результатов — ответ, не повтор."""

    return _on("MR_BUTTON_SIDE_ANSWER")


def config_conflicts() -> list[str]:
    """Несовместимые сочетания ключей кнопки — громко при старте и в отпечатке рантайма (`debug.runtime.config_conflicts`).

    `button_answer_topic = frame` + `button_frame_check = judge` отдают проверку «другой темы» судье ответа (вариант Б:
    шаг переклейки пропускается, `pending_answer.frame_judge_on`); но задачу с исходом `domain` судья берёт только с
    нажатой кнопкой в данных (`judge_context`), а её даёт лишь `button_reply_judge = on`. Без него «другую тему» в
    ответе на вопрос кнопки не проверяет никто (Н-7, STEP-2026-10-30-BUTTON-KEYS-CONFLICT).
    """

    out: list[str] = []
    if pending_answer.frame_judge_on() and not reply_judge_on():
        out.append("button_frame_check=judge требует button_reply_judge=on: «другую тему» в ответе на вопрос кнопки не проверяет никто")
    return out


def warn_config_conflicts() -> list[str]:
    conflicts = config_conflicts()
    for text in conflicts:
        logger.warning("CONFIG CONFLICT: %s", text)
    return conflicts


def price_scope_mode() -> str:
    """`[MESSENGER_ROUTER] button_price_scope = off | doctor | all`: какие кнопки цены сужают поиск своей областью прайса."""

    mode = str(getattr(_cfg, "MR_BUTTON_PRICE_SCOPE", "off") or "").strip().lower()
    return mode if mode in ("doctor", "all") else "off"


def topic_firm(flags: Iterable[Any] | None) -> bool:
    """`[MESSENGER_ROUTER] button_topic_firm = on` и тему хода задала кнопка (флаг `button_answer_topic`): переклейки после NLU
    по буквам её не перебивают. off — как было."""

    return _on("MR_BUTTON_TOPIC_FIRM") and "button_answer_topic" in set(flags or ())


def marks_questions() -> bool:
    """Висящий вопрос кнопки помечается её листом — только когда пометку кто-то читает (off — состояние как было)."""

    return answer_topic_on() or hint_on() or reply_judge_on() or side_answer_on() or price_scope_mode() != "off"


def price_scope(pending: Any) -> str:
    """Область прайса ответа на вопрос кнопки цены (`tests` | `doctor`) — если её сужает режим `button_price_scope`; иначе
    пусто (поиск по всему прайсу, как было)."""

    mode = price_scope_mode()
    action = button_menu.action_for(question_button(pending)) if mode != "off" else None
    if action is None or action.label != "PRICE" or not action.scope:
        return ""
    return action.scope if mode == "all" or action.scope == "doctor" else ""


def pressed(flags: Iterable[Any] | None) -> str:
    """Лист меню, нажатый на этом ходе, — из флага решения кнопки `button:<id>` (`orchestrator._button_decision`)."""

    for flag in flags or ():
        if str(flag).startswith("button:"):
            return str(flag).split(":", 1)[1]
    return ""


def question_button(pending: Any) -> str:
    """Лист меню, который задал висящий вопрос; вопрос не кнопки — пусто."""

    return str(pending.get(QUESTION_KEY) or "") if isinstance(pending, dict) else ""


def mark_question(pending: Any, button_id: str) -> None:
    """Пометить висящий вопрос листом кнопки — если вопрос её темы (кнопка `slot` спрашивает недостающее своей метки)."""

    action = button_menu.action_for(button_id) if button_id else None
    if isinstance(pending, dict) and action is not None and action.mode == "slot" and pending.get("label") == action.label:
        pending[QUESTION_KEY] = button_id


def judges_answer(label: str, pending: Any) -> bool:
    """Вопрос кнопки этой темы получает свободный ответ одной функции под `button_answer_topic = on` (кнопки цены): вопрос
    `SLOT_TOPICS` получает его и без выключателя (`pending_buttons`). Подсказка (`button_topic_hint`) судью кнопкам цены не
    добавляет: под `slot` классификатор там остаётся, и судья был бы лишней секундой (карточка 25.10: p50 2,5 → 3,3 с)."""

    return _judges_price() and label in _JUDGED_UNDER_SWITCH and bool(question_button(pending))


def judged_by_llm(pending: Any) -> bool:
    """Ответ на вопрос кнопки судит LLM и тогда, когда значение слота нашли правила (`filled`), — но только на «другую тему»:
    правила дозаполнения находят значение и в реплике о другом (31 из 32 чужих вопросов, которым карточка 25.10 навязала тему
    кнопки), а прочие исходы LLM там сужение правил не перебивают («Результаты» на «Ввести данные» LLM звала «не могу
    назвать» — 32 ответа из 89 стали оффером оператора вместо инструкции)."""

    return (answer_topic_on() or hint_on()) and bool(question_button(pending))


def judge_context(pending: Any, question: str) -> str:
    """Нажатая кнопка, её тема и вопрос бота — данные судьи ответа (`button_reply_judge`): полный вопрос в теме кнопки — ответ.
    Выключатель off или вопрос задала не кнопка — пусто: судит прежний судья по одному вопросу бота."""

    action = button_menu.action_for(question_button(pending)) if reply_judge_on() else None
    if action is None or not action.title:
        return ""
    return f"кнопка «{action.title}»{_where(action)} — вопросы про {action.about}; вопрос бота: «{question}»"


def result_side_question(answered: pending_answer.Answer | None, pending: Any) -> bool:
    """Пациент спросил о самом вопросе кнопки результатов («где на бланке код?», «как узнать сам результат?»): судья одной
    функции решил «ничего из этого» (правило 3 промпта — вопрос о вопросе бота), без хвоста; под `button_side_answer`."""

    action = button_menu.action_for(question_button(pending)) if side_answer_on() else None
    return (
        action is not None
        and action.label == "TEST_RESULT"
        and answered is not None
        and answered.type == Question.SLOT_VALUE.value
        and answered.source is pending_answer.Source.LLM
        and answered.outcome is pending_answer.Outcome.NONE
        and not answered.rest
    )


def answer_topic(pending: Any) -> str:
    """Тема ответа на вопрос кнопки, если её знает кнопка: выключатель, вопрос задала кнопка, одна функция решила «это
    ответ» (правилами или LLM). Иначе пусто — тему решает NLU, как было."""

    mixed = _MIXED.get()
    if mixed is not None:
        return mixed[0]  # в сообщении и ответ, и другой вопрос (`button_topic_hint`): ответ — по теме кнопки
    if _REFRAMED.get():
        return _REFRAMED.get()  # «другую тему» судьи вопрос рамки вернул в тему кнопки (`frame`)
    mode = answer_topic_mode()
    label = str(pending.get("label") or "") if isinstance(pending, dict) else ""
    if mode == "off" or not question_button(pending) or (label in _JUDGED_UNDER_SWITCH and not _judges_price()):
        return ""
    found = pending_answer.slot_answer()
    if found is None or found.outcome is not pending_answer.Outcome.ANSWER:
        return ""
    return label


def refund_topic_judge(answered: pending_answer.Answer | None, pending: Any, spent_before: int) -> None:
    """Решение одной функции об ответе на вопрос кнопки — шаг NLU вместо классификатора: лимит решений хода не тратит, пути
    темы остаются её решения, как без кнопки (`button_answer_topic`, `button_topic_hint`). `spent_before` — решений хода до
    функции: ответ из кэша решения не тратил, и возвращать нечего."""

    from . import llm_runtime

    if (
        llm_runtime.turn_decisions_spent() <= spent_before
        or answered is None
        or answered.outcome is not pending_answer.Outcome.ANSWER  # прочие исходы ход не закрывают — их уже вернула функция (Q5)
        or answered.type != Question.SLOT_VALUE.value
        or not judged_by_llm(pending)
    ):
        return
    if llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_button_refund=1)


# --------------------------------------------------------------------------- подсказка кнопки (`button_topic_hint`)

# Где пациент найдёт ответ: листья меню, до двух (в сообщении два вопроса).
HINT_TASK = Task(name="button_hint", max_picks=2, num_predict=48, timeout_s=8, queue_timeout_ms=3000)
# Не подсказываем: перевод на оператора — явный запрос, его исполняет ход раньше.
_NOT_HINTED = frozenset({"menu.operator"})
# Ответ по теме кнопки и подсказка в конце — (тема кнопки, текст подсказки) этого хода.
_MIXED: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("mr_button_mixed", default=None)
# «Другая тема» судьи ответа, которую отдельный вопрос рамки вернул в тему кнопки (`button_answer_topic = frame`): её тема.
_REFRAMED: contextvars.ContextVar[str] = contextvars.ContextVar("mr_button_reframed", default="")
# Лист без своего вопроса, нажатый на прошлом ходе, — рамка этой реплики (`frame`): метка хода и ключ состояния.
LEAF_KEY = "_button_leaf"
_LEAF: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("mr_button_leaf", default=None)
# Лист, в рамку которого вопрос рамки вернул эту реплику: его флаги и сущности — решению хода (`nlu_pipeline`).
_REFRAMED_LEAF: contextvars.ContextVar[str] = contextvars.ContextVar("mr_button_reframed_leaf", default="")
# Сообщение в теме нажатой кнопки? — отдельный короткий вопрос после «другой темы» судьи ответа (`frame`).
FRAME_TASK = Task(name="button_frame", outcomes=("same", "other"), num_predict=24, timeout_s=8, queue_timeout_ms=3000)


@dataclass(frozen=True)
class Hint:
    """Подсказка: листья другой темы и было ли в сообщении ещё и ответ на вопрос кнопки."""

    leaves: tuple[str, ...]
    mixed: bool = False


def reset() -> None:
    """Начало хода: ответа с подсказкой и рамки прошлого хода нет."""

    _MIXED.set(None)
    _REFRAMED.set("")
    _REFRAMED_LEAF.set("")


def _where(action: button_menu.ButtonAction) -> str:
    return f" раздела «{action.section}»" if action.section else " главного меню"


def menu_options() -> list[Option]:
    """Листья меню для выбора LLM: подпись, раздел, о чём лист (`button_menu`)."""

    return [
        Option(id=button_id, text=f"«{action.title}»{_where(action)} — вопросы про {action.about}")
        for button_id, action in button_menu.BUTTONS.items()
        if button_id not in _NOT_HINTED and action.title
    ]


def _same_topic(button_id: str, current: str) -> bool:
    """Лист той же темы, что кнопка вопроса («Сколько стоит анализ» и «Цена анализа»), — не другая тема."""

    a, b = button_menu.action_for(button_id), button_menu.action_for(current)
    return a is not None and b is not None and a.label == b.label


def hint_text(leaves: tuple[str, ...], *, tail: bool = False) -> str:
    """«Это вопрос про … — основная информация в кнопке «…» раздела «…».»; `tail` — вторая часть ответа («А про …»)."""

    actions = [a for a in (button_menu.action_for(b) for b in leaves) if a is not None]
    if not actions:
        return ""
    about = " и ".join(a.about for a in actions)
    buttons = " и ".join(f"«{a.title}»{_where(a)}" for a in actions)
    where = "в кнопке" if len(actions) == 1 else "в кнопках"
    if tail:
        return f"А про {about} — основная информация {where} {buttons}."
    head = "Это вопрос" if len(actions) == 1 else "Это вопросы"
    return f"{head} про {about} — основная информация {where} {buttons}."


def asked_with_button(current: str, question: str) -> str:
    """Нажатая кнопка и вопрос бота — данные выбора подсказки: о теме нажатой кнопки — её пункт, не подсказка."""

    action = button_menu.action_for(current)
    if action is None or not action.title:
        return question
    return f"{action.title}{_where(action)} — {question}" if question else f"{action.title}{_where(action)}"


async def hint(text: str, current: str, asked: str) -> Hint | None:
    """Где пациент найдёт ответ на сообщение о другой теме — листья меню по выбору LLM (`HINT_TASK`).

    :param text: реплика пациента
    :param current: лист кнопки, на чей вопрос ждали ответ
    :param asked: вопрос бота, без ПДн
    :return: подсказка; None — подсказки нет (ничего не подошло, только тема кнопки, сбой LLM): ход идёт общим путём, а
        решение лимит решений хода не тратит
    """

    from . import llm_runtime

    spent = llm_runtime.turn_decisions_spent()
    try:
        choice = await choose(HINT_TASK, text, menu_options(), asked=asked)
    except Exception:  # новый слой: его сбой не ломает ход — общий путь
        logger.warning("button_hint_failed", exc_info=True)
        choice = None
    picked = tuple(choice.ids) if choice is not None and choice.status is Status.PICKED else ()
    others = tuple(dict.fromkeys(b for b in picked if button_menu.action_for(b) is not None and not _same_topic(b, current)))
    if not others:
        if llm_runtime.turn_decisions_spent() > spent and llm_runtime.refund_turn_decision():
            turn_telemetry.note(tu_button_hint_refund=1)
        turn_telemetry.note(button_hint="none")
        return None
    found = Hint(leaves=others, mixed=len(others) < len(picked))
    turn_telemetry.note(button_hint=("mixed:" if found.mixed else "") + "+".join(others))
    return found


def reframed() -> bool:
    """Ход в рамке кнопки по решению вопроса рамки (`frame`): подсказка другой кнопки не нужна."""

    return bool(_REFRAMED.get())


async def reframe(text: str, pending: Any) -> bool:
    """«Другую тему» судьи ответа на вопрос кнопки проверяет отдельный вопрос LLM — сообщение в теме нажатой кнопки?
    (`FRAME_TASK`, под `button_answer_topic = frame`). «В теме» — тема хода — кнопки (`answer_topic`), её не перебивают
    несрочные правила; «другое» и сбой LLM — общий путь, как было. Решение лимит решений хода не тратит: оно — шаг NLU.

    :param text: реплика пациента
    :param pending: висящий вопрос кнопки
    :return: ход остаётся в теме кнопки
    """

    action = button_menu.action_for(question_button(pending))
    label = str(pending.get("label") or "") if isinstance(pending, dict) else ""
    if not frame_firm() or action is None or not action.title or not label:
        return False
    if not await _in_frame(text, action):
        return False
    _REFRAMED.set(label)
    return True


async def _in_frame(text: str, action: button_menu.ButtonAction) -> bool:
    """Вопрос рамки (`FRAME_TASK`): сообщение в теме нажатой кнопки? Сбой — нет; лимит решений хода не тратит."""

    from . import llm_runtime

    spent = llm_runtime.turn_decisions_spent()
    try:
        choice = await choose(FRAME_TASK, text, [], asked=f"«{action.title}»{_where(action)} — вопросы про {action.about}")
    except Exception:  # новый слой: его сбой не ломает ход — общий путь
        logger.warning("button_frame_failed", exc_info=True)
        choice = None
    outcome = choice.outcome if choice is not None and choice.status is Status.OUTCOME else ""
    turn_telemetry.note(button_frame=outcome or "none")
    if llm_runtime.turn_decisions_spent() > spent and llm_runtime.refund_turn_decision():
        turn_telemetry.note(tu_button_frame_refund=1)
    return outcome == "same"


def leaf_frame_leaves() -> frozenset[str]:
    """`[MESSENGER_ROUTER] button_leaf_frame` — листы без своего вопроса, которые держат рамку следующей реплики: рамка без
    ответа по данным темы — шаблон, включать по листу, по замеру; пусто (по умолчанию) — ни один."""

    raw = str(getattr(_cfg, "MR_BUTTON_LEAF_FRAME", "") or "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _frames_next_reply(button_id: str) -> bool:
    """Лист без своего вопроса (отвечает на нажатие сразу: `pass` или `slot` со своими сущностями) из `button_leaf_frame`,
    кроме адресных — их следующую реплику ведёт `address_followup`."""

    action = button_menu.action_for(button_id) if button_id else None
    if action is None or not action.title or button_id not in leaf_frame_leaves():
        return False
    answered_at_press = action.mode == "pass" or (action.mode == "slot" and bool(action.entities))
    return answered_at_press and not (action.mode == "pass" and action.label == "ADDRESS")


def track_leaf(state: Any, button_id: str) -> None:
    """Начало хода (`frame`): нажатие листа без своего вопроса — метка этого хода (возраст 0), следующая реплика — возраст 1;
    другое нажатие и вторая реплика — метки нет. off — состояние как было."""

    _LEAF.set(None)
    if not frame_firm():
        return
    previous = state.last_entities.pop(LEAF_KEY, None)
    if button_id:
        if _frames_next_reply(button_id):
            _LEAF.set({"id": button_id, "age": 0})
        return
    if isinstance(previous, dict) and int(previous.get("age") or 0) == 0 and previous.get("id"):
        _LEAF.set({"id": str(previous["id"]), "age": 1})


def finish_leaf(state: Any) -> None:
    """Конец хода: метку нажатия — в состояние, чтобы её увидела следующая реплика; прочее — снять."""

    mark = _LEAF.get()
    if mark is not None and int(mark.get("age") or 0) == 0:
        state.last_entities[LEAF_KEY] = dict(mark)
    else:
        state.last_entities.pop(LEAF_KEY, None)


async def reframe_leaf(text: str) -> bool:
    """Реплика сразу после листа без своего вопроса: в его теме? (`frame`, вопрос рамки). «В теме» — тема хода — листа, его
    флаги и сущности — решению (`leaf_frame`), несрочные правила её не перебивают; «другое» и сбой — общий путь, как было."""

    mark = _LEAF.get()
    button_id = str(mark.get("id") or "") if mark is not None and int(mark.get("age") or 0) == 1 else ""
    action = button_menu.action_for(button_id) if button_id else None
    if not frame_firm() or action is None or not action.label:
        return False
    if not await _in_frame(text, action):
        return False
    _REFRAMED.set(action.label)
    _REFRAMED_LEAF.set(button_id)
    return True


def leaf_frame() -> button_menu.ButtonAction | None:
    """Лист без своего вопроса, в рамку которого вопрос рамки вернул эту реплику; иначе None."""

    button_id = _REFRAMED_LEAF.get()
    return button_menu.action_for(button_id) if button_id else None


def leaf_frame_id() -> str:
    """Id листа без своего вопроса, в рамку которого вопрос рамки вернул эту реплику; иначе пусто (полное понимание хода)."""

    return str(_REFRAMED_LEAF.get() or "")


def mixed_topic() -> bool:
    """Ход отвечает на смешанное сообщение: тема — кнопки, и её не перебивает надёжное правило другой темы."""

    return _MIXED.get() is not None


def answer_with_hint(topic: str, found: Hint) -> None:
    """Сообщение — и ответ на вопрос кнопки, и другой вопрос: ход отвечает по теме кнопки, подсказка — в конце ответа."""

    _MIXED.set((topic, hint_text(found.leaves, tail=True)))


def with_mixed_hint(response: ResponseEnvelope) -> ResponseEnvelope:
    """Ответ хода с подсказкой в конце, если сообщение было смешанным; перевод на оператора — без подсказки."""

    mixed = _MIXED.get()
    if mixed is None or response.handoff or not str(response.text or "").strip():
        return response
    response.text = f"{response.text.rstrip()}\n\n{mixed[1]}"
    return response
