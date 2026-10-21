"""Голое название услуги → максимум по данным клиники (решение владельца 21.10, принцип разметки 7).

Пациент присылает только название: «Анализ T-SPOT», «Фгдс», «ПСА общий и свободный». Без выключателя ответ зависел от того,
какое правило поймало буквы: T-SPOT — адреса «анализы без записи» (детектор walk-in), «Анализ на Впч» — голые цены, «Фгдс» —
список врачей. Решение владельца: голое название — максимум пациенту (что это, цена, подготовка, где) или уточнить, что
интересует; «всё» — максимум.

Под `service_answer_from_data = on` (`choose`, задача `service_overview`, промпт в двух локациях): правила дают кандидатов —
строки прайса по реплике (как у ответа о клинике, `service_answer`); LLM решает, голое ли это название, и выбирает строки,
которые пациент назвал. Одна строка — максимум из данных (принцип 1, печатает код): цена и срок готовности — из прайса, что это
и подготовка — из справочника услуг МИС (`description`, `preparation`), где — правило клиники (анализы — в лабораторных
филиалах без записи, остальное — по записи). Несколько строк — варианты с ценами и вопрос, какой интересует. Вопрос или
просьба в реплике, «не услуга», «нет среди строк», сбой — как было.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from . import llm_runtime, turn_telemetry
from .choice import Option, Status, Task, choose
from .service_answer import enabled  # noqa: F401 — тот же выключатель, что у ответа по данным

TASK = Task(
    name="service_overview",
    max_picks=3,
    outcomes=("asks", "not_service"),
    num_predict=48,
    timeout_s=6.0,
    queue_timeout_ms=1500,
)

# Голое название — короткая реплика: длиннее — почти всегда вопрос (кандидат, решает LLM).
MAX_WORDS = 6
_DESCRIPTION_CHARS = 280
_PREPARATION_CHARS = 360


@dataclass(frozen=True)
class Overview:
    """Решение: строки прайса, которые пациент назвал (одна — максимум, несколько — уточнить вариант)."""

    rows: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {"rows": [dict(row) for row in self.rows]}

    @staticmethod
    def from_payload(data: Any) -> "Overview | None":
        if not isinstance(data, dict):
            return None
        rows = tuple(dict(row) for row in data.get("rows") or () if isinstance(row, dict))
        return Overview(rows) if rows else None


def candidate(user_text: str) -> bool:
    """Похоже на голое название (кандидат, не решение): коротко и непусто."""

    words = str(user_text or "").split()
    return 0 < len(words) <= MAX_WORDS


async def decide(user_text: str) -> Overview | None:
    """Голое ли название и какие строки прайса пациент назвал — решение LLM; иначе или при сбое — None (как было)."""

    from .service_answer import _price_rows, _row_name
    from .services._prices_helpers import _catalog_rows

    rows = await asyncio.to_thread(_catalog_rows, None)
    pool = _price_rows([user_text], user_text, rows, ())
    if not pool:
        turn_telemetry.note(svc_overview="no_candidates")
        return None
    options = [Option(id=str(i), text=_row_name(row)) for i, row in enumerate(pool)]
    spent_before = llm_runtime.turn_decisions_spent()
    picked = await choose(TASK, user_text, options)
    if picked.status is not Status.PICKED:
        # Не голое название или сбой — ход идёт как было: лимит решений хода это решение не тратит.
        spent = llm_runtime.turn_decisions_spent() > spent_before
        turn_telemetry.note(svc_overview=picked.outcome or picked.status.value, svc_overview_refund=int(spent and llm_runtime.refund_turn_decision()))
        return None
    chosen = tuple(pool[int(i)] for i in picked.ids)
    # Проверка исхода (BUG-2026-10-21-SVC-SIMILAR-NOISE): строка — та самая услуга, со всем уточнением («Справка 026» не
    # «070/у», «УЗДГ БЦС ребёнку» не справка о здоровье ребёнка); не подтверждена — как было.
    from .service_answer import Fact, check

    checked = await check(user_text, [Fact(kind="price", title=_row_name(row), row=dict(row)) for row in chosen])
    confirmed = tuple(fact.row for fact in checked[0] if fact.row) if checked else ()
    if not confirmed:
        returned = sum(1 for _ in range(max(0, llm_runtime.turn_decisions_spent() - spent_before)) if llm_runtime.refund_turn_decision())
        turn_telemetry.note(svc_overview="not_confirmed", svc_overview_refund=returned)
        return None
    turn_telemetry.note(svc_overview="one" if len(confirmed) == 1 else "several")
    return Overview(confirmed)


# --------------------------------------------------------------------------- текст (печатает код)


def _plain(html: str) -> str:
    from .services._prepare import _memo_plain_text

    return " ".join(_memo_plain_text(str(html or "")).split())


def _cut(text: str, limit: int) -> tuple[str, bool]:
    """Начало текста до `limit` знаков по границе предложения; (текст, обрезан ли)."""

    if len(text) <= limit:
        return text, False
    end = text.rfind(". ", 0, limit)
    return (text[: end + 1] if end > 0 else text[:limit].rstrip() + "…"), True


def _service_info(row: dict[str, Any]) -> dict[str, Any]:
    """Карточка услуги из справочника МИС (по id, иначе по коду); нет среза или карточки — пусто."""

    from .services import prepare as _prepare_mod

    try:
        infos = _prepare_mod.api_service_info.load_service_info()
    except Exception:
        return {}
    sid, code = row.get("serviceId"), str(row.get("serviceHomecode") or "").strip()
    for info in infos or ():
        if not isinstance(info, dict):
            continue
        if (sid is not None and info.get("serviceId") == sid) or (code and str(info.get("serviceHomecode") or "").strip() == code):
            return info
    return {}


def _deadline(row: dict[str, Any]) -> str:
    return str(row.get("deadline") or "").strip()


def _one(row: dict[str, Any]) -> str:
    from .renderer import _bundle_text, _extract_price_amount, _format_rub

    from .service_answer import _row_name

    name = _row_name(row)
    info = _service_info(row)
    lines = [f"«{name}»:"]
    amount = _extract_price_amount(row)
    if amount:
        lines.append(f"— цена: {_format_rub(amount)}")
    if _deadline(row):
        lines.append(f"— срок готовности: {_deadline(row)}")
    description, _cut_off = _cut(_plain(info.get("description") or ""), _DESCRIPTION_CHARS)
    if description:
        lines.append(f"— что это: {description}")
    preparation, more = _cut(_plain(info.get("preparation") or ""), _PREPARATION_CHARS)
    if preparation:
        lines.append(f"— подготовка: {preparation}")
        if more:
            lines.append("  Полная памятка — по запросу «как подготовиться».")
    if _deadline(row):
        lines.append("— где: в лабораторных филиалах клиники, без записи, в порядке живой очереди; адреса — по запросу «где сдать».")
    else:
        # Где и по записи ли — у неанализа в данных нет (ЭКГ, например, без записи): не утверждаем, только подсказка.
        lines.append("— записаться или узнать адреса — напишите «записаться» или «где сделать».")
    return _bundle_text(lines, "", priced=bool(amount))


def _several(rows: tuple[dict[str, Any], ...]) -> str:
    from .renderer import _bundle_text, _extract_price_amount, _format_rub

    from .service_answer import _row_name

    lines = ["В прайсе клиники несколько вариантов:"]
    for row in rows:
        amount = _extract_price_amount(row)
        lines.append(f"— {_row_name(row)} — {_format_rub(amount)}" if amount else f"— {_row_name(row)}")
    lines.append("")
    lines.append("Какой вариант интересует? Расскажу, что это, как подготовиться и где сдать.")
    return _bundle_text(lines, "", priced=any(_extract_price_amount(row) for row in rows))


def text(overview: Overview) -> str:
    return _one(overview.rows[0]) if len(overview.rows) == 1 else _several(overview.rows)
