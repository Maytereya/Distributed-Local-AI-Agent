"""Ответ подбора анализов — только по данным хода (BUG-2026-10-21-WALKIN-LLM-FREE-TEXT; принципы 1–3).

Ход TEST_ASSIST, которому не нашлось готового ответа (цены, памятки, срока по каталогу), отвечал свободным текстом LLM
(`renderer_patient`, `answer_source = llm_free`) поверх строк прайса, найденных по буквам реплики. Текст утверждал о
клинике то, чего в данных нет: «подготовка обычно включает специальную диету и отказ от некоторых лекарств» к строке
«Прото-скрин» (g0783), «Анализ на магний доступен в нашей клинике» при пустом поиске (g0190), два анализа плода как
«комплекс к материнству» (g0266). Под `topic_by_llm` с ветками `rule_test_result` и `rule_prepare` таких ходов больше.

Под `test_assist_from_data = on` ответ решает LLM по данным хода (`choose`, задача `test_assist_answer`; принцип 2):
кандидаты — строки, которые нашёл инструмент хода, и пул кандидатов выбора строки цены по реплике (поиск подбора анализов
ищет по всей реплике и «анализ на магний у вас есть?» не находит «Mg+ (магний)»). LLM выбирает строки, которые прямо
отвечают, или исход: `partly` (это те анализы, но спрошенного условия в строке нет), `need_name` (вопрос об анализах
вообще — нужно название), `no_data` (ни одна строка не отвечает). Исхода «похожее» нет: близкое, но другое, — не ответ
(BUG-2026-10-21-SVC-SIMILAR-NOISE). Текст печатает код (принцип 1): название и срок готовности — из строки прайса, без
цены (свободный текст цен не печатал — карточка цены не меняется). Нет данных и сбой LLM — честный оффер оператора
(принцип 3), не свободный текст.

Под `test_assist_price = on` (решение владельца 26.10, топ-5 разбора 95 цен, п. 1) у выбранных строк — цена из той же
строки прайса: «Антитела на корь» → «Вирус кори Ig G — 800 руб.» вместо «цену подскажу по названию» рядом с уже
напечатанным названием. Частичный ответ — цены найденных строк и «остального в прайсе нет — уточнит оператор». Давность
прайса — по правилам пути цены (`services/_freshness`): срез старше недели — строки без цен, как под off.

Под `qualifier_marks = on` (проверка выбора, STEP-2026-10-28-POSITION-QUALIFIERS-8) «в таком виде нет» пишет не вызов, который
выбирал строки, а отдельный вопрос LLM с данными справочника МИС — тот же, что у цены (`_qualifier_pool.check_rows`): ярлык
`partly` выбора был ложным в 4 из 5 случайных ответов (метод — в описании МИС, «остальное» — станция метро, тот самый чекап).
Уточнения единственной позиции разбора хода проверка находит не все — `partly` («Точно в таком виде в прайсе нет»); находит все —
строки как есть; самой услуги нет — «нет данных». Уточнений у позиции нет (или позиций несколько), а выбор сказал `partly` —
спрошено не об анализе (состав, время, место) или нашлась часть перечня: строки под прежним заголовком «В прайсе клиники есть:»
и «остального нет — уточнит оператор» (`partly_rest`).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from . import evidence_keys as ek
from . import turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

TASK = Task(
    name="test_assist_answer",
    max_picks=6,
    outcomes=("partly", "need_name", "no_data"),
    num_predict=64,
    timeout_s=6.0,
    queue_timeout_ms=1500,
)

PICKED = "picked"
PARTLY = "partly"
PARTLY_REST = "partly_rest"
NEED_NAME = "need_name"
NO_DATA = "no_data"
FAILED = "failed"

# Сколько строк показывать LLM: строк инструмента хода и всего (с пулом по реплике).
_TOOL_ROWS = 8
_ALL_ROWS = 12
_PREVIOUS_CHARS = 300

# Сущность хода с догадкой каталога: суд «да/нет» по ней пропущен, строку выбирает ответ хода. В состояние диалога не
# попадает — роутер переносит её в данные хода (`evidence_keys.TEST_ASSIST_CATALOG_GUESS`).
CATALOG_GUESS_KEY = "_test_assist_catalog_guess"

PICKED_HEADING = "В прайсе клиники есть:"
PICKED_TAIL = "Цену и подготовку подскажу по названию."
PARTLY_TAIL = "Остального из вашего вопроса в данных клиники нет — точно ответит оператор."
NEED_NAME_TEXT = "Ответ зависит от конкретного анализа. Напишите, пожалуйста, названия анализов — подскажу по данным клиники."
# Под `test_assist_price = on`: цена уже напечатана — хвост о ней не обещает; частичный ответ — формулировка владельца 26.10.
PRICED_TAIL = "Подготовку подскажу по названию."
PRICED_PARTLY_TAIL = "Остального из вашего вопроса в прайсе нет — уточнит оператор."
# Частичный ответ с ценой (решение владельца 26.10, вариант 3): «нет» — до строк и цен, иначе «В прайсе клиники есть: … —
# 2 300 руб.» читалось как «да, стоит 2 300» на «отдельно, не в комплексе» (g0811) и «суммарные» при строке IgG (g0388).
# «Ближайшее» не пишем: `partly` — и «из перечня нашлась только часть», там строки — те самые анализы.
PRICED_PARTLY_HEADING = "Точно в таком виде в прайсе нет. Есть:"


def enabled() -> bool:
    """`[MESSENGER_ROUTER] test_assist_from_data = on`; мусор и отсутствие — off (как было)."""

    return str(getattr(_cfg, "MR_TEST_ASSIST_FROM_DATA", "off") or "").strip().lower() == "on"


def price_enabled() -> bool:
    """`[MESSENGER_ROUTER] test_assist_price = on`; мусор и отсутствие — off (строки без цены, как было)."""

    return str(getattr(_cfg, "MR_TEST_ASSIST_PRICE", "off") or "").strip().lower() == "on"


def applies(decision: Any, evidence: Any) -> bool:
    """Ход подбора анализов дошёл до свободного текста LLM — его и заменяет ответ по данным."""

    return enabled() and decision is not None and evidence is not None and getattr(decision, "label", "") == "TEST_ASSIST"


@dataclass(frozen=True)
class Answer:
    """Решение: `kind` — PICKED | PARTLY | PARTLY_REST | NEED_NAME | NO_DATA | FAILED, `rows` — выбранные строки прайса."""

    kind: str
    rows: tuple[dict[str, Any], ...] = ()


# --------------------------------------------------------------------------- кандидаты


def _row_name(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("serviceName") or row.get("name") or "").split())


def _deadline(row: dict[str, Any]) -> str:
    from .services._prices_helpers import format_deadline

    return format_deadline(str(row.get("deadline") or "")) or ""


def _option_text(row: dict[str, Any]) -> str:
    deadline = _deadline(row)
    return f"{_row_name(row)} — срок готовности {deadline}" if deadline else _row_name(row)


def _search_texts(question: str, evidence: Any) -> list[str]:
    """Реплика и название услуги, по которому искал инструмент хода (оно бывает из контекста диалога)."""

    items = getattr(evidence, "items", None) or {}
    payload = items.get("test_assist")
    entities = payload.get("entities_used") if isinstance(payload, dict) else None
    texts = [str(question or "").strip()]
    if isinstance(entities, dict):
        for key in ("test_name", "service_name"):
            name = str(entities.get(key) or "").strip()
            if name and name.lower() not in {t.lower() for t in texts}:
                texts.append(name)
    return [t for t in texts if t]


def _catalog_guess(evidence: Any) -> str:
    items = getattr(evidence, "items", None) or {}
    return " ".join(str(items.get(ek.TEST_ASSIST_CATALOG_GUESS) or "").split())


def _candidates(question: str, evidence: Any) -> tuple[list[dict[str, Any]], int]:
    """Строки догадки каталога и инструмента хода, затем пул выбора строки цены по реплике (поиск правил, синонимы МИС,
    редкие слова). Догадку каталога суд «да/нет» больше не судит — без неё в кандидатах верной строки нет вовсе
    («уровень железа в крови»: поиск по реплике нашёл «CA 15-3 (молочная железа)», каталог — «Железо в сыворотке»).

    :return: (кандидаты, сколько из них — строки инструмента хода)
    """

    from .service_answer import _price_rows, evidence_rows
    from .services._prices_helpers import _catalog_rows

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any], limit: int) -> None:
        name = _row_name(row)
        if name and name.lower() not in seen and len(out) < limit:
            seen.add(name.lower())
            out.append(row)

    try:
        rows = _catalog_rows(None)
        pool = _price_rows(_search_texts(question, evidence), question, rows, ())
    except Exception as exc:  # каталог недоступен — выбираем из строк инструмента
        logger.warning("test_assist_answer_pool_unavailable: %s", type(exc).__name__)
        rows, pool = [], []
    guess = _catalog_guess(evidence).lower()
    if guess:
        for row in rows:
            if isinstance(row, dict) and _row_name(row).lower() == guess:
                add(row, _TOOL_ROWS)
    for row in evidence_rows(evidence):
        add(row, _TOOL_ROWS)
    tool_count = len(out)
    for row in pool:
        add(row, _ALL_ROWS)
    return out, tool_count


def _previous_bot_message(history: Sequence[Any] | None) -> str:
    """Последний ответ бота — контекст для «а сколько делается второй?» (текущую реплику оркестратор уже срезал)."""

    for item in reversed(list(history or ())):
        if isinstance(item, dict) and item.get("role") == "assistant" and str(item.get("text") or "").strip():
            return " ".join(str(item["text"]).split())[:_PREVIOUS_CHARS]
    return ""


# --------------------------------------------------------------------------- решение


async def decide(question: str, evidence: Any, *, history: Sequence[Any] | None = None) -> Answer:
    """Что ответить на ход подбора анализов по данным хода. Сбой LLM — FAILED (вызывающий отвечает оффером)."""

    rows, tool_count = await asyncio.to_thread(_candidates, question, evidence)
    options = [Option(id=str(i), text=_option_text(row)) for i, row in enumerate(rows)]
    picked = await choose(TASK, question, options, asked=_previous_bot_message(history) or "(диалог только начался)")
    chosen = tuple(rows[int(i)] for i in picked.ids)
    if picked.failed:
        answer = Answer(FAILED)
    elif picked.status is Status.PICKED:
        answer = Answer(PICKED, chosen)
    elif picked.status is Status.OUTCOME and picked.outcome == PARTLY:
        answer = Answer(PARTLY, chosen) if chosen else Answer(NO_DATA)
    elif picked.status is Status.OUTCOME and picked.outcome == NEED_NAME:
        answer = Answer(NEED_NAME)
    else:  # no_data или пустой выбор: ни одна строка не отвечает
        answer = Answer(NO_DATA)
    answer = await _checked(question, answer)
    turn_telemetry.note(
        ta_answer=answer.kind,
        ta_rows=len(rows),
        ta_tool_rows=tool_count,
        ta_picked=len(answer.rows),
        **({"ta_fail": picked.reason or picked.status.value} if picked.failed else {}),
    )
    return answer


async def _checked(question: str, answer: Answer) -> Answer:
    """Проверка выбора (`qualifier_marks = on`): вердикт «в таком виде нет» — отдельный вопрос LLM с данными справочника МИС
    (`_qualifier_pool.check_rows`), а не ярлык `partly` вызова выбора. Off, строк нет — решение выбора как есть."""

    from .services import _qualifier_pool

    if answer.kind not in (PICKED, PARTLY) or not answer.rows or not _qualifier_pool.marks_enabled():
        return answer
    if not _qualifier_pool.to_mark(question):  # уточнений позиции нет: `partly` — о другом (состав, время, место, часть перечня)
        return Answer(PARTLY_REST, answer.rows) if answer.kind == PARTLY else answer
    check = await _qualifier_pool.check_rows(question, answer.rows)
    if check is None:  # проверка не ответила — решение выбора как есть
        return answer
    if not check.service:  # выбор и проверка разошлись: самой услуги в строках нет — честный оффер, не чужие строки (принцип 1)
        return Answer(NO_DATA)
    return Answer(PARTLY if _qualifier_pool.note_missing(check.missing) else PICKED, answer.rows)


# --------------------------------------------------------------------------- текст ответа (печатает код)


def _rows_block(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join([PICKED_HEADING, *(f"— {_option_text(row)}" for row in rows)])


def _priced_line(row: dict[str, Any]) -> str:
    """«Название — 800 руб., срок готовности 2–3 дня»: сумма — из этой же строки прайса (принцип 1)."""

    from .renderer import _extract_price_amount, _format_rub

    amount = _extract_price_amount(row)
    if amount is None:
        return _option_text(row)
    deadline = _deadline(row)
    price = f"{_row_name(row)} — {_format_rub(amount)}"
    return f"{price}, срок готовности {deadline}" if deadline else price


def _price_as_of(rows: Sequence[dict[str, Any]]) -> str | None:
    """Можно ли печатать цены и с какой датой: правила давности пути цены (`services/_freshness`, ход (д)).

    :return: None — цены не печатать (срез старше недели или его нет); "" — молча; «ДД.ММ» — приписка давности
    """

    from .services.core import prices_checked

    checked = prices_checked({"prices": list(rows)})
    if not isinstance(checked, dict) or checked.get("handoff_required") or not checked.get("prices"):
        return None
    return str(checked.get("data_as_of") or "")


def _priced_text(rows: Sequence[dict[str, Any]], tail: str, heading: str = PICKED_HEADING) -> str | None:
    """Заголовок, строки с ценами и хвост; приписка о ценах региона — до оффера. None — цен печатать нельзя (ответ как под off)."""

    from .renderer import _bundle_text, _extract_price_amount

    as_of = _price_as_of(rows)
    priced = sum(1 for row in rows if _extract_price_amount(row) is not None)
    if as_of is None or not priced:
        return None
    turn_telemetry.note(ta_priced=priced)
    if tail == PRICED_TAIL and priced < len(rows):
        tail = PICKED_TAIL  # строка без суммы (в прайсе бывает пустая) — о её цене ответ ничего не сказал
    return _bundle_text([heading, *(f"— {_priced_line(row)}" for row in rows), "", tail], as_of, priced=True)


def envelope(answer: Answer, state: Any, memory: Any | None, *, attachments: Sequence[Any] = ()) -> tuple[Any, bool]:
    """Ответ пациенту по решению. Строки — название и срок из прайса; «нет данных» и сбой — честный оффер оператора.

    :return: (ответ, задан ли вопрос-оффер оператора) — у оффера нет вторичного предложения темы
    """

    from .mess_types import ResponseEnvelope
    from .response_builder import UNSUPPORTED_CATALOG_OPERATOR_OFFER, mark_operator_offer_pending
    from .service_answer import NO_DATA_TEXT

    priced = price_enabled() and answer.kind in (PICKED, PARTLY, PARTLY_REST) and bool(answer.rows)
    if answer.kind == PICKED and answer.rows:
        text, offer = f"{_rows_block(answer.rows)}\n\n{PICKED_TAIL}", False
        if priced:
            text = _priced_text(answer.rows, PRICED_TAIL) or text
    elif answer.kind == NEED_NAME:
        text, offer = NEED_NAME_TEXT, False
    elif answer.kind in (PARTLY, PARTLY_REST) and answer.rows:
        text, offer = f"{_rows_block(answer.rows)}\n\n{PARTLY_TAIL}", True
        if priced:  # «в таком виде нет» — только когда проверка не нашла уточнения (под off — по ярлыку выбора, как было)
            heading = PRICED_PARTLY_HEADING if answer.kind == PARTLY else PICKED_HEADING
            text = _priced_text(answer.rows, PRICED_PARTLY_TAIL, heading) or text
    else:
        text, offer = NO_DATA_TEXT, True
    if offer and memory is not None:
        mark_operator_offer_pending(state, memory)
        text = f"{text}\n\n{UNSUPPORTED_CATALOG_OPERATOR_OFFER}"
    return ResponseEnvelope(text=text, attachments=list(attachments or ()), handoff=False), offer
