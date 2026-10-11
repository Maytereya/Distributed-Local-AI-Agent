"""Срок готовности при слабом совпадении: услугу выбирает LLM из кандидатов, срок печатает код (ход (в)).

Вопрос о сроке («сколько готовится хгч», «срок готовности липидограммы») при сильном совпадении по
данным получает срок из поля `deadline` справочника МИС кодом (`readiness_from_catalog`, 4df0745).
Слабое совпадение (буквы, падеж, часть названия) раньше шло в честный перевод — 14 переводов о сроке
в карточке «понимание хода on» (11.10). Здесь его решает выбор модели, как строку прайса:

- кандидаты — строки, которые находит поиск правил по словам вопроса (без фразы о сроке);
- модель видит только названия (не сроки и не цены) и отмечает номера той услуги;
- выбранные названия — сроки из справочника кодом (принцип 1), не больше трёх; срока нет или у
  строк одного названия сроки разные — это название не показывается;
- ничего не выбрала, сбой, список услуг — None: честный перевод, как раньше.

Выключатель `[MESSENGER_ROUTER] choose_readiness = off | on` (по умолчанию off; действует под
пониманием хода `on`, где вопрос о сроке распознаётся).
"""

from __future__ import annotations

from typing import Any

from ..choice import Option, Status, Task, choose
from ..runtime_config import config as _cfg
from ._prices_helpers import (
    _catalog_rows,
    _normalise_input,
    _rank_price_rows,
    format_deadline,
    price_list_services,
    resolve_price_service_name_from_catalog,
    strip_readiness_phrasing,
)

READINESS_TASK = Task(name="readiness_service", max_picks=3, num_predict=64, timeout_s=12, queue_timeout_ms=3000)
_CANDIDATES = 10


def readiness_mode() -> str:
    raw = str(getattr(_cfg, "MR_CHOOSE_READINESS", "off") or "").strip().lower()
    return raw if raw in ("off", "on") else "off"


def _deadline(name: str, catalog: list[dict[str, Any]]) -> str | None:
    key = _normalise_input(name)
    raw = {str(row.get("deadline") or "").strip() for row in catalog if _normalise_input(str(row.get("serviceName") or "")) == key}
    formatted = {format_deadline(value) for value in raw}
    return formatted.pop() if len(formatted) == 1 and None not in formatted else None


async def readiness_by_choice(question: str, *, rows: list[dict[str, Any]] | None = None) -> list[dict[str, str]] | None:
    """Сроки готовности услуг, которые выбрала модель; None — честный перевод.

    :param question: реплика пациента с вопросом о сроке
    :param rows: строки прайса (по умолчанию — срез региона)
    :return: [{"name", "deadline"}] или None
    """

    catalog = _catalog_rows(rows)
    core, _ = strip_readiness_phrasing(question)
    if not core.strip() or price_list_services(core, rows=catalog)[0]:
        return None
    names: list[str] = []
    guess = resolve_price_service_name_from_catalog(core, rows=catalog)
    for name in [guess or "", *(str(row.get("serviceName") or "") for row in _rank_price_rows(catalog, core, limit=_CANDIDATES))]:
        name = name.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return None
    choice = await choose(READINESS_TASK, question, [Option(id=name, text=name) for name in names])
    if choice.status is not Status.PICKED:
        return None
    found = [{"name": name, "deadline": deadline} for name in choice.ids if (deadline := _deadline(name, catalog))]
    return found or None
