"""Контекст сессии = показанное — ход (б) L-14, шаг 3 (`[MESSENGER_ROUTER] context_shown = off | on`, по умолчанию off).

Было: услугу в состояние (`service_name`) писало только заземление правил по буквам (`merge_entities` в
роутере). Услугу, которую ответ показал пациенту, но выбрали LLM (строка цены по термину: «ггтп» → «ГГТ
гамма-глютамилтрансфераза») или код из МИС (срок готовности — ранний возврат мимо слияния сущностей), состояние
не получало: следующее «сколько стоит?» не знало, о чём шла речь. Сводка диалога (`update_summary`, раз в 4 хода)
копировала `service_name` и называла его, когда состояние уже сменилось или было снято
(STEP-2026-10-14-L14-PREP-CONSUMERS, находка 6).

Стало (под `on`): ответ с ценами пишет в состояние `_shown_services` — названия строк, которые увидел пациент
(как их печатает рендер), а `service_name` — показанную услугу, если она одна; срок из МИС — только
`_shown_services` (ответ без цен: по решению владельца 13.10 услуги в контексте нет — вопрос владельцу). У
корзины `_shown_services` — все её строки, `service_name` не пишется (услуг несколько). Реплика, в понимании
которой услуга не названа («а цена?»), не перебивает контекст фразой вырезателя (`prices._names_no_service`).
Сводка обновляется, как только называет не ту услугу. Под `off` — как было.
"""

from __future__ import annotations

from typing import Any, Iterable

from .mess_types import SessionState
from .runtime_config import config as _cfg

MODES = ("off", "on")
SHOWN_KEY = "_shown_services"
_RENDERED_PRICE_ROWS = 5  # рендер цены печатает не больше пяти строк (`renderer.format_price_for_patient`)


def mode() -> str:
    raw = str(getattr(_cfg, "MR_CONTEXT_SHOWN", "off") or "").strip().lower()
    return raw if raw in MODES else "off"


def _row_name(row: Any) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip() if isinstance(row, dict) else ""


def shown_price_names(payload: dict[str, Any]) -> list[str]:
    """Названия строк с ценой, которые рендер цены покажет пациенту, — в его порядке и без повторов.

    Уточнение (`clarify_text`: ответ политики, вопрос) цен не показывает — пусто; семейство — видимые
    варианты (`visible_limit`, все — после «все»); строки цены — первые пять разных, как печатает рендер.
    """

    if str(payload.get("clarify_text") or "").strip():
        return []
    if str(payload.get("service_kind") or "").strip().lower() == "family_query":
        variants = payload.get("family_variants")
        if isinstance(variants, list) and variants:
            limit = max(1, int(payload.get("visible_limit") or 10))
            rows: Iterable[Any] = variants if payload.get("showing_all") else variants[:limit]
            return list(dict.fromkeys(name for name in map(_row_name, rows) if name))
    prices = payload.get("prices")
    names = [name for name in map(_row_name, prices if isinstance(prices, list) else []) if name]
    return list(dict.fromkeys(names))[:_RENDERED_PRICE_ROWS]


def shown_bundle_names(payload: dict[str, Any]) -> list[str]:
    """То же для ответа `service_bundle` (`renderer.format_service_bundle_for_patient`): уточнение — пусто;
    семейство и корзина — видимые варианты; несколько строк анализа или выбора LLM — первые пять разных; иначе —
    первая строка (рендер печатает одну цену)."""

    if str(payload.get("clarify_text") or "").strip() or str(payload.get("service_kind") or "").strip().lower() == "family_query":
        return shown_price_names(payload)
    rows = payload.get("retail_prices")
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not rows:
        return []
    if len(rows) > 1 and (str(payload.get("service_kind") or "").strip().lower() == "lab" or payload.get("llm_selected")):
        names = [name for name in map(_row_name, rows[:_RENDERED_PRICE_ROWS]) if name]
        return list(dict.fromkeys(names))
    name = _row_name(rows[0]) or str(payload.get("service_name") or "").strip()
    return [name] if name else []


def remember(state: SessionState, names: list[str], *, priced: bool = True) -> None:
    """Показанное — в состояние: `_shown_services` всегда, `service_name` — если показана одна услуга с ценой.

    :param priced: ответ показал цены. Ответ без цен (срок) — только `_shown_services`: по решению владельца
        13.10 (BUG-2026-10-13-L13-CONTEXT-LEAK) после ответа без цен услуги в контексте нет — пациент не выбирал
    """

    state.last_entities[SHOWN_KEY] = list(names)
    if priced and len(names) == 1:
        state.last_entities["service_name"] = names[0]
        state.last_entities.pop("test_name", None)


def sync_price(state: SessionState, payload: dict[str, Any], render_entities: dict[str, Any]) -> None:
    """Ответ цены (`build_price_response`). Цена у врача — услуга врача, а не контекст анализа: не трогаем."""

    if mode() != "on" or render_entities.get("doctor_id") or render_entities.get("doctor_name"):
        return
    remember(state, shown_price_names(payload))


def sync_bundle(state: SessionState, payload: dict[str, Any]) -> None:
    """Ответ `build_service_bundle_response` (одна услуга — «По услуге «X» нашёл следующее», корзина)."""

    if mode() == "on":
        remember(state, shown_bundle_names(payload))
