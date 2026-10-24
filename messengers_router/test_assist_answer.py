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
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from . import turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

TASK = Task(
    name="test_assist_answer",
    max_picks=4,
    outcomes=("partly", "need_name", "no_data"),
    num_predict=64,
    timeout_s=6.0,
    queue_timeout_ms=1500,
)

PICKED = "picked"
PARTLY = "partly"
NEED_NAME = "need_name"
NO_DATA = "no_data"
FAILED = "failed"

# Сколько строк показывать LLM: строк инструмента хода и всего (с пулом по реплике).
_TOOL_ROWS = 8
_ALL_ROWS = 12
_PREVIOUS_CHARS = 300

PICKED_HEADING = "В прайсе клиники есть:"
PICKED_TAIL = "Цену и подготовку подскажу по названию."
PARTLY_TAIL = "Остального из вашего вопроса в данных клиники нет — точно ответит оператор."
NEED_NAME_TEXT = "Сроки готовности и подготовка зависят от анализа. Напишите, пожалуйста, названия анализов — подскажу по каждому."


def enabled() -> bool:
    """`[MESSENGER_ROUTER] test_assist_from_data = on`; мусор и отсутствие — off (как было)."""

    return str(getattr(_cfg, "MR_TEST_ASSIST_FROM_DATA", "off") or "").strip().lower() == "on"


def applies(decision: Any, evidence: Any) -> bool:
    """Ход подбора анализов дошёл до свободного текста LLM — его и заменяет ответ по данным."""

    return enabled() and decision is not None and evidence is not None and getattr(decision, "label", "") == "TEST_ASSIST"


@dataclass(frozen=True)
class Answer:
    """Решение: `kind` — PICKED | PARTLY | NEED_NAME | NO_DATA | FAILED, `rows` — выбранные строки прайса."""

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


def _candidates(question: str, evidence: Any) -> tuple[list[dict[str, Any]], int]:
    """Строки инструмента хода, затем пул выбора строки цены по реплике (поиск правил, синонимы МИС, редкие слова).

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

    for row in evidence_rows(evidence):
        add(row, _TOOL_ROWS)
    tool_count = len(out)
    try:
        rows = _catalog_rows(None)
        texts = _search_texts(question, evidence)
        pool = _price_rows(texts, question, rows, ())
    except Exception as exc:  # каталог недоступен — выбираем из строк инструмента
        logger.warning("test_assist_answer_pool_unavailable: %s", type(exc).__name__)
        pool = []
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
    turn_telemetry.note(
        ta_answer=answer.kind,
        ta_rows=len(rows),
        ta_tool_rows=tool_count,
        ta_picked=len(answer.rows),
        **({"ta_fail": picked.reason or picked.status.value} if picked.failed else {}),
    )
    return answer


# --------------------------------------------------------------------------- текст ответа (печатает код)


def _rows_block(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join([PICKED_HEADING, *(f"— {_option_text(row)}" for row in rows)])


def envelope(answer: Answer, state: Any, memory: Any | None, *, attachments: Sequence[Any] = ()) -> tuple[Any, bool]:
    """Ответ пациенту по решению. Строки — название и срок из прайса; «нет данных» и сбой — честный оффер оператора.

    :return: (ответ, задан ли вопрос-оффер оператора) — у оффера нет вторичного предложения темы
    """

    from .mess_types import ResponseEnvelope
    from .response_builder import UNSUPPORTED_CATALOG_OPERATOR_OFFER, mark_operator_offer_pending
    from .service_answer import NO_DATA_TEXT

    if answer.kind == PICKED and answer.rows:
        text, offer = f"{_rows_block(answer.rows)}\n\n{PICKED_TAIL}", False
    elif answer.kind == NEED_NAME:
        text, offer = NEED_NAME_TEXT, False
    elif answer.kind == PARTLY and answer.rows:
        text, offer = f"{_rows_block(answer.rows)}\n\n{PARTLY_TAIL}", True
    else:
        text, offer = NO_DATA_TEXT, True
    if offer and memory is not None:
        mark_operator_offer_pending(state, memory)
        text = f"{text}\n\n{UNSUPPORTED_CATALOG_OPERATOR_OFFER}"
    return ResponseEnvelope(text=text, attachments=list(attachments or ()), handoff=False), offer
