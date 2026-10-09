"""Исход хода по ответу эндпоинта, debug и состоянию сессии — без LLM-судьи.

Порядок надёжности источников (разведка 09.10): флаг `handoff` конверта → метка и
флаги `debug.decision` → признак оффера в состоянии сессии (`_operator_offer_pending`)
→ точное совпадение с каноническими текстами бота → разбор цен по прайсу. Регулярки по
свободному тексту — только запасной путь.

Канонические тексты берутся из того дерева кода, которое гоняет стенд (`Templates.from_bot`):
у бейслайна и у новой версии они могут различаться.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .prices import PriceCatalog, PricePair

# Внутренности, которые не должны доходить до пациента (HYP-5, утечка промпта).
_LEAK_RE = re.compile(r"АЛГОРИТМ|<<[A-Z_]+>>|\bcontext_action\b|\"label\"\s*:|\bLabel:\s*[A-Z_]{3,}")
_NOT_FOUND_RE = re.compile(
    r"не наш[её]л|не найден|нет информации|информации недостаточно|не удалось найти|не могу найти|нет данных",
    re.I,
)
_MEMO_RE = re.compile(r"Памятка клиники «|Подходят несколько памяток")
_QUESTION_END_RE = re.compile(r"\?\s*$")


@dataclass(frozen=True)
class Templates:
    handoff: dict[str, str]
    urgent: str = "Соединяю с оператором."
    not_understood_prefix: str = "Не совсем понял"
    offers: tuple[str, ...] = ()
    hours_note: str = ""

    @classmethod
    def from_bot(cls) -> "Templates":
        from messengers_router import policies, recovery_policy, response_builder
        from messengers_router.text_templates import LOW_CONF_CLARIFY_TEXT

        offers = [recovery_policy.UNCLEAR_OPERATOR_OFFER_TEXT, response_builder.UNSUPPORTED_CATALOG_OPERATOR_OFFER]
        try:
            from messengers_router import orchestrator

            offers.append(orchestrator.EMPTY_EVIDENCE_NEUTRAL)
        except AttributeError:  # в старых версиях константы нет
            pass
        return cls(
            handoff=dict(policies.HANDOFF_REASON_MATRIX),
            not_understood_prefix=LOW_CONF_CLARIFY_TEXT[:16],
            offers=tuple(offers),
            hours_note=policies.OPERATOR_HOURS_NOTE,
        )


@dataclass
class TurnOutcome:
    outcome: str
    handoff: bool
    handoff_reason: str | None
    handoff_technical: bool
    urgent: bool
    operator_offer: bool
    clarify: bool
    label: str | None
    decision_source: str | None
    flags: list[str]
    price_pairs: list[PricePair] = field(default_factory=list)
    memo: bool = False
    unsourced_prices: int = 0
    leak: bool = False
    chars: int = 0
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        row = {k: v for k, v in self.__dict__.items() if k != "price_pairs"}
        row["price_pairs"] = [p.__dict__ for p in self.price_pairs]
        return row


def _debug(body: dict[str, Any]) -> dict[str, Any]:
    su = body.get("state_update") or {}
    dbg = su.get("debug") if isinstance(su, dict) else None
    return dbg if isinstance(dbg, dict) else {}


def evidence_items(dbg: dict[str, Any]) -> dict[str, Any]:
    """`debug.evidence.items` — словарь; в части версий сериализован строкой."""

    ev = dbg.get("evidence") or {}
    items = ev.get("items") if isinstance(ev, dict) else None
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            return {}
    return items if isinstance(items, dict) else {}


def _strip_note(text: str, note: str) -> str:
    return text.replace(note, "").strip() if note else text.strip()


def classify_turn(body: dict[str, Any], session_entities: dict[str, Any], tpl: Templates, catalog: PriceCatalog | None) -> TurnOutcome:
    """Главный исход хода и признаки — одна строка результата прогона."""

    text = str(body.get("text") or "")
    core = _strip_note(text, tpl.hours_note)
    dbg = _debug(body)
    decision = dbg.get("decision") if isinstance(dbg.get("decision"), dict) else {}
    label = decision.get("label")
    flags = [str(f) for f in (decision.get("flags") or [])]
    error = dbg.get("route_error") or dbg.get("endpoint_error")
    handoff = bool(body.get("handoff"))

    reason = next((key for key, msg in tpl.handoff.items() if msg and msg in core), None)
    technical = bool(error) or (reason or "").startswith("service_error") or reason in ("renderer_error", "tech_unavailable")
    urgent = handoff and core.split("\n\n")[0].strip() == tpl.urgent and (label == "URGENT" or "urgent" in flags)
    offer = bool(session_entities.get("_operator_offer_pending")) or any(o in core for o in tpl.offers)
    clarify = bool(decision.get("clarify_needed") or decision.get("clarify_reason"))
    pairs = catalog.pairs(core) if catalog is not None else []
    memo = bool(_MEMO_RE.search(core))
    matched = [p for p in pairs if p.status == "matched"]

    if error:
        outcome = "tech_error"
    elif urgent:
        outcome = "urgent"
    elif handoff and technical:
        outcome = "handoff_tech"
    elif handoff:
        outcome = "handoff"
    elif offer:
        outcome = "operator_offer"
    elif core.startswith(tpl.not_understood_prefix):
        outcome = "not_understood"
    elif matched:
        outcome = "price"
    elif memo:
        outcome = "memo"
    elif _NOT_FOUND_RE.search(core) or not core:
        outcome = "no_info"  # класс 4: данных нет, перевода нет
    elif clarify or _QUESTION_END_RE.search(core):
        outcome = "clarify"
    else:
        outcome = "answer"

    return TurnOutcome(
        outcome=outcome,
        handoff=handoff,
        handoff_reason=reason,
        handoff_technical=technical,
        urgent=urgent,
        operator_offer=offer,
        clarify=clarify,
        label=label,
        decision_source=decision.get("source"),
        flags=flags,
        price_pairs=pairs,
        memo=memo,
        unsourced_prices=sum(1 for p in pairs if p.status == "unsourced"),
        leak=bool(_LEAK_RE.search(core)),
        chars=len(text),
        error=(str(error)[:200] if error else None),
    )
