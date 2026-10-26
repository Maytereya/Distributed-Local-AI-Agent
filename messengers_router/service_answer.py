"""Ответ о клинике — только по её данным (решения владельца 21.10: «ни в коем случае не выдумывать»).

«Есть ли у вас / делаете ли / берёте ли X» — утверждение о клинике. До выключателя его писал свободный текст LLM
(`renderer_patient`, ветка TEST_ASSIST) поверх похожих строк прайса: «Вы берёте анализы крови у детей 2 лет?» → «Да, мы
проводим анализы крови для детей» (g0509, так отвечал прод) — о возрасте в данных ни слова
(BUG-2026-10-21-FACT-WITHOUT-DATA). А где данные были — «Делают ли ФКС с биопсией?»: строка прайса «Биопсия при ФКС» и
строка во всех трёх памятках ФКС, — бот отвечал «врачей не нашёл» или переводил на оператора
(BUG-2026-10-21-SERVICE-AVAILABILITY-BY-DATA).

Под `service_answer_from_data = on` ответ решает LLM по данным клиники (`choose`, задача `service_answer`; принцип 2):
кандидаты — строки прайса и строки памяток пациенту из базы знаний; LLM выбирает пункты, на которых держится ответ, или
исход: `similar` (точно такого нет, есть близкое), `no_data` (вопрос о клинике, ответа в данных нет), `not_about_clinic`
(подбор по жалобам, значение показателя). Нет ответа — LLM может назвать до трёх других названий услуги, по ним второй
круг поиска. Текст печатает код (принцип 1): название и цена — из прайса, строка памятки — как есть, без цены. Нет
данных — честный оффер оператора (принцип 3).

Под `service_answer_no_similar = on` (BUG-2026-10-21-SVC-SIMILAR-NOISE; топ-5 разбора 95 цен, п. 5) «похожее» пациенту не
показывается — как у подбора анализов по данным (`test_assist_answer`): близкое, но другое — честный оффер, второй круг по
терминам модели. Исход у модели остаётся: карточка 26.10 без него показала, что модель называет близкое ответом «да» («3D/4D УЗИ» →
обычное УЗИ беременных). Задача `service_answer_exact`, промпт — прежний с синтетическими примерами вместо двух вопросов эталона.

Где спрашивается (там, где бот иначе написал бы свободный текст или сдался бы, не заглянув в данные):
- `free_text` — перед свободным текстом LLM (оркестратор): ответ, близкое или оффер; «не о клинике» и сбой — как было;
- `other_offer` — OTHER от LLM под `topic_by_llm`; `empty_evidence` — вопрос без данных (L-05); `doctors_by_service` —
  «врачей по услуге не нашёл» (L-06(б)): ответ или близкое — только из данных, иначе прежний оффер.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

from . import llm_runtime, turn_telemetry
from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

logger = logging.getLogger(__name__)

TASK = Task(
    name="service_answer",
    max_picks=4,
    outcomes=("similar", "no_data", "not_about_clinic"),
    max_terms=3,
    num_predict=96,
    timeout_s=8.0,
    queue_timeout_ms=2000,
)

# Под `service_answer_no_similar`: своё имя — свой промпт (синтетические примеры), своё место в телеметрии и на стенде. «Похожее»
# у модели есть, пациенту — нет (`_ask`).
EXACT_TASK = Task(
    name="service_answer_exact",
    max_picks=4,
    outcomes=("similar", "no_data", "not_about_clinic"),
    max_terms=3,
    num_predict=96,
    timeout_s=8.0,
    queue_timeout_ms=2000,
)

ANSWER = "answer"
SIMILAR = "similar"
NO_DATA = "no_data"
NOT_ABOUT_CLINIC = "not_about_clinic"
FAILED = "failed"

# Сколько пунктов показывать LLM: строк прайса, строк памяток (и с одной памятки), всего за круг.
_PRICE_FACTS = 8
_MEMO_FACTS = 6
_MEMO_LINES_PER_TITLE = 2
_ROUND_FACTS = 12
_PER_TEXT_ROWS = 6
_MEMO_LINE_CHARS = (8, 300)
# Корень короче этого — предлог или служебное слово («при», «под», «лет»): кандидатов не называет.
_MIN_ROOT = 4

ANSWER_HEADING = "Да, по данным клиники это есть:"
SIMILAR_HEADING = "Точно такой услуги в данных клиники не нашёл. Есть похожие:"
NO_DATA_TEXT = "По этому вопросу у меня нет данных — точно ответит оператор."


def enabled() -> bool:
    """`[MESSENGER_ROUTER] service_answer_from_data = on`; мусор и отсутствие — off (как было)."""

    return str(getattr(_cfg, "MR_SERVICE_ANSWER_FROM_DATA", "off") or "").strip().lower() == "on"


def no_similar() -> bool:
    """`[MESSENGER_ROUTER] service_answer_no_similar = on`; мусор и отсутствие — off (исход «похожее» есть, как было)."""

    return str(getattr(_cfg, "MR_SERVICE_ANSWER_NO_SIMILAR", "off") or "").strip().lower() == "on"


@dataclass(frozen=True)
class Fact:
    """Пункт данных клиники: строка прайса (`row`) или строка памятки пациенту (`title` — название памятки, `text` — её
    строка без цены)."""

    kind: str  # "price" | "memo"
    title: str
    text: str = ""
    row: dict[str, Any] | None = None

    def option_text(self) -> str:
        if self.kind == "price":
            return f"Прайс клиники: {self.title}"
        return f"Памятка «{_unquoted(self.title)}»: {self.text}"

    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.title, self.text)

    def payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "title": self.title, "text": self.text, "row": dict(self.row) if self.row else None}

    @staticmethod
    def from_payload(data: dict[str, Any]) -> "Fact":
        row = data.get("row")
        return Fact(
            kind=str(data.get("kind") or ""),
            title=str(data.get("title") or ""),
            text=str(data.get("text") or ""),
            row=dict(row) if isinstance(row, dict) else None,
        )


@dataclass(frozen=True)
class Verdict:
    """Решение по вопросу: `kind` — ANSWER | SIMILAR | NO_DATA | NOT_ABOUT_CLINIC | FAILED, `facts` — выбранные пункты."""

    kind: str
    facts: tuple[Fact, ...] = ()
    calls: int = 0  # сколько решений LLM потрачено (круги поиска) — вернуть ходу, если ответ хода не изменился

    @property
    def from_data(self) -> bool:
        """Ответ держится на данных клиники (ответ или близкое)."""

        return self.kind in (ANSWER, SIMILAR) and bool(self.facts)

    def payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "facts": [fact.payload() for fact in self.facts]}

    @staticmethod
    def from_payload(data: Any) -> "Verdict | None":
        if not isinstance(data, dict):
            return None
        facts = tuple(Fact.from_payload(f) for f in data.get("facts") or () if isinstance(f, dict))
        return Verdict(kind=str(data.get("kind") or ""), facts=facts)


# --------------------------------------------------------------------------- кандидаты из данных


def _unquoted(title: str) -> str:
    return " ".join(str(title or "").replace("«", "").replace("»", "").split())


def _row_name(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("serviceName") or row.get("name") or "").split())


def _price_rows(texts: Sequence[str], question: str, rows: list[dict[str, Any]], first: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Строки прайса — кандидаты (не ответ): синонимы МИС, строки инструмента хода, поиск правил и редкие слова."""

    from .services import _price_select_llm as sel

    texts = [str(t or "") for t in texts if str(t or "").strip()]
    if not texts or not rows:
        return [row for row in first if isinstance(row, dict)][:_PRICE_FACTS]
    head = [*sel._synonym_rows(texts, rows), *[row for row in first if isinstance(row, dict)]]
    pool = sel._build_pool(texts, rows, per_text=_PER_TEXT_ROWS, first=head)
    pool = sel._with_rare_tail(pool, sel._rare_rows(texts, rows, short_from=question))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in pool:
        name = _row_name(row)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(row)
    return out[:_PRICE_FACTS]


def _memo_line(line: str) -> str:
    """Строка памятки для пациента и LLM: без номера пункта и без цены (памятки базы знаний — без цен, решение
    владельца 07.10; цену печатает прайс)."""

    from .services.prepare import _PRICE_RE

    text = " ".join(_PRICE_RE.sub("", str(line or "")).split()).strip(" .;,-–—")
    while text[:1].isdigit():
        text = text.lstrip("0123456789").lstrip(".) ").strip()
    cut = text.lower().rfind(", стоимост")
    if cut > 0:
        text = text[:cut]
    return text.strip(" .;,-–—")


def _memo_facts(texts: Sequence[str]) -> list[Fact]:
    """Строки памяток пациенту, где есть редкие слова вопроса (вес слова — его редкость среди строк памяток)."""

    from .services._prepare import _prepare_roots_match, _prepare_term_roots
    from .services._prepare_select_llm import kb_patient_memos

    try:
        memos = kb_patient_memos()
    except Exception as exc:  # база знаний недоступна — отвечаем по прайсу
        logger.warning("service_answer_memos_unavailable: %s", type(exc).__name__)
        return []
    lines: list[tuple[str, str, set[str]]] = []
    for memo in memos:
        for raw in str(memo.text or "").splitlines():
            text = _memo_line(raw)
            if _MEMO_LINE_CHARS[0] <= len(text) <= _MEMO_LINE_CHARS[1]:
                lines.append((memo.title, text, _prepare_term_roots(text)))
    if not lines:
        return []
    wanted = {root for text in texts for root in _prepare_term_roots(str(text or "")) if len(root) >= _MIN_ROOT}
    if not wanted:
        return []
    df: dict[str, int] = {}
    for _title, _text, roots in lines:
        for root in roots:
            df[root] = df.get(root, 0) + 1
    scored: list[tuple[float, int, str, str]] = []
    for index, (title, text, roots) in enumerate(lines):
        matched = [root for root in wanted if _prepare_roots_match(root, roots)]
        if matched:
            weight = sum(math.log((len(lines) + 1) / (1 + df.get(root, 0))) for root in matched)
            scored.append((weight, -index, title, text))
    out: list[Fact] = []
    per_title: dict[str, int] = {}
    seen: set[str] = set()
    for _weight, _order, title, text in sorted(scored, reverse=True):
        if text.lower() in seen or per_title.get(title, 0) >= _MEMO_LINES_PER_TITLE:
            continue
        seen.add(text.lower())
        per_title[title] = per_title.get(title, 0) + 1
        out.append(Fact(kind="memo", title=" ".join(title.split()), text=text))
        if len(out) >= _MEMO_FACTS:
            break
    return out


async def gather_facts(question: str, texts: Sequence[str] = (), first_rows: Sequence[dict[str, Any]] = ()) -> list[Fact]:
    """Пункты данных клиники по вопросу (или по названиям второго круга `texts`)."""

    from .services._prices_helpers import _catalog_rows

    search = [str(t) for t in texts if str(t or "").strip()] or [question]
    rows = await asyncio.to_thread(_catalog_rows, None)
    price = [Fact(kind="price", title=_row_name(row), row=dict(row)) for row in _price_rows(search, question, rows, first_rows)]
    memo = await asyncio.to_thread(_memo_facts, search)
    return [*price, *memo]


def evidence_rows(evidence: Any) -> list[dict[str, Any]]:
    """Строки прайса, которые инструмент хода уже нашёл (подбор анализов, цена) — первые кандидаты."""

    items = getattr(evidence, "items", None) or {}
    out: list[dict[str, Any]] = []
    for key, field in (("test_assist", "tests"), ("price", "prices")):
        payload = items.get(key)
        rows = payload.get(field) if isinstance(payload, dict) else None
        out.extend(row for row in rows or () if isinstance(row, dict) and _row_name(row))
    return out


# --------------------------------------------------------------------------- решение


def _merge(old: Sequence[Fact], new: Sequence[Fact]) -> list[Fact]:
    keys = {fact.key() for fact in old}
    return [*old, *[fact for fact in new if fact.key() not in keys]][:_ROUND_FACTS]


async def _ask(question: str, facts: Sequence[Fact]) -> tuple[Verdict, tuple[str, ...]]:
    options = [Option(id=str(i), text=fact.option_text()) for i, fact in enumerate(facts)]
    picked = await choose(EXACT_TASK if no_similar() else TASK, question, options)
    if picked.failed:
        return Verdict(FAILED), ()
    chosen = tuple(facts[int(i)] for i in picked.ids)
    if picked.status is Status.OUTCOME:
        if picked.outcome == SIMILAR and no_similar():
            # Близкое, но другое — не ответ пациенту: честный оффер (и второй круг по терминам модели, как у «нет данных»).
            turn_telemetry.note(svc_answer_similar_hidden=len(chosen))
            return Verdict(NO_DATA), picked.terms
        if picked.outcome == SIMILAR:
            return (Verdict(SIMILAR, chosen) if chosen else Verdict(NO_DATA)), picked.terms
        return Verdict(picked.outcome), picked.terms
    if picked.status is Status.PICKED:
        return Verdict(ANSWER, chosen), ()
    return Verdict(NO_DATA), picked.terms  # вопрос о клинике, ни один пункт ответа не держит


async def decide(question: str, *, site: str, first_rows: Sequence[dict[str, Any]] = ()) -> Verdict:
    """Что ответить на вопрос о клинике по её данным. Сбой LLM — FAILED: что делать, решает вызывающий."""

    spent_before = llm_runtime.turn_decisions_spent()
    facts = await gather_facts(question, first_rows=first_rows)
    verdict, terms = await _ask(question, facts)
    rounds = 1
    if verdict.kind == NO_DATA and terms:
        extra = await gather_facts(question, texts=terms)
        merged = _merge(facts, extra)
        if len(merged) > len(facts):
            verdict, _terms = await _ask(question, merged)
            rounds = 2
    turn_telemetry.note(svc_answer=verdict.kind, svc_answer_site=site, svc_answer_facts=len(verdict.facts), svc_answer_rounds=rounds)
    return Verdict(verdict.kind, verdict.facts, calls=max(0, llm_runtime.turn_decisions_spent() - spent_before))


def refund(verdict: Verdict) -> None:
    """Решение не изменило ответ хода (место ответило как было) — лимит решений хода оно не тратит: иначе выбору строки
    прайса и другим решениям хода его не хватает (как у висящего вопроса и роли, решение владельца Q5 18.10)."""

    returned = sum(1 for _ in range(verdict.calls) if llm_runtime.refund_turn_decision())
    if returned:
        turn_telemetry.note(svc_answer_refund=returned)


async def from_data_first(question: str, *, site: str, first_rows: Sequence[dict[str, Any]] = ()) -> Verdict | None:
    """Место, где бот сдался бы без данных (свой честный оффер уже есть): сначала данные клиники. Ответ или близкое —
    решение; «нет данных», «не о клинике», сбой — None (прежний оффер). Выключатель снят — None без вызова LLM."""

    if not enabled():
        return None
    verdict = await decide(question, site=site, first_rows=first_rows)
    if verdict.from_data:
        return verdict
    refund(verdict)
    return None


# --------------------------------------------------------------------------- текст ответа (печатает код)


def _fact_line(fact: Fact) -> str:
    from .renderer import _extract_price_amount, _format_rub

    if fact.kind == "price" and fact.row:
        amount = _extract_price_amount(fact.row)
        return f"— {fact.title} — {_format_rub(amount)}" if amount else f"— {fact.title}"
    return f"— в памятке клиники «{_unquoted(fact.title)}»: «{fact.text}»"


def _facts_text(heading: str, facts: Sequence[Fact]) -> str:
    from .renderer import _bundle_text

    # Есть цены — с припиской о ценах региона, как у остальных ответов с ценами.
    priced = any(fact.kind == "price" and fact.row for fact in facts)
    return _bundle_text([heading, *(_fact_line(fact) for fact in facts)], "", priced=priced)


def envelope(verdict: Verdict | None, state: Any, memory: Any | None, *, offer_no_data: bool = True) -> Any | None:
    """Ответ пациенту по решению; None — решения нет (не о клинике, сбой): вызывающий делает как было.

    `offer_no_data = False` — у места уже свой честный оффер на «нет данных» (OTHER, L-05, L-06(б)): здесь только ответ
    по данным."""

    from .mess_types import ResponseEnvelope
    from .response_builder import UNSUPPORTED_CATALOG_OPERATOR_OFFER, mark_operator_offer_pending

    if verdict is None:
        return None
    if verdict.kind == ANSWER and verdict.facts:
        return ResponseEnvelope(text=_facts_text(ANSWER_HEADING, verdict.facts), handoff=False)
    if verdict.kind == SIMILAR and verdict.facts:
        text = _facts_text(SIMILAR_HEADING, verdict.facts)
    elif verdict.kind == NO_DATA and offer_no_data:
        text = NO_DATA_TEXT
    else:
        return None
    if memory is not None:
        mark_operator_offer_pending(state, memory)
        text = f"{text}\n\n{UNSUPPORTED_CATALOG_OPERATOR_OFFER}"
    return ResponseEnvelope(text=text, handoff=False)
