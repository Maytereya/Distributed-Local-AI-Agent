"""Услугу «где сделать …» выбирает LLM, правила только собирают кандидатов (принцип 2; решение владельца 18.10).

BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS, остаток: при пустой сущности услуги инструмент адресов (`addresses.address_info`)
сам выводил её из реплики — `extract_service_phrase` и лексический отбор строк прайса (`_select_address_price_rows`), — и
место оказания решали буквы: «где сделать магнитотерапию» → адрес «Mg+ (магний)». Разбор хода к этому моменту услугу уже
снял (LLM её не подтвердила) или вовсе не находил.

Под `[MESSENGER_ROUTER] address_service_llm = on`:
- услугу, которую LLM разбора хода уже не подтвердила (планировщик кладёт `ADDRESS_SERVICE_DROPPED_KEY`), инструмент
  заново не ищет — общий список филиалов, как у снятой услуги (`entity_grounder._apply_service_verdict`);
- иначе строки прайса, найденные правилами по фразе реплики, — кандидаты; какие из них спросил пациент, решает модуль
  выбора (`choose`, задача `address_service`, промпт в двух локациях): выбраны — места оказания по ним; «ни одна» —
  услуга снимается, общий список филиалов; сбой или мусор LLM — как было (решение 18.10 для ADDRESS: «LLM недоступна —
  как было, по правилам»).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

# Сколько различных названий строк показать модели: шире — отбор правил уже шумный (семейство до 50 строк).
OPTIONS_LIMIT = 8
ADDRESS_SERVICE_TASK = Task(name="address_service", max_picks=3, num_predict=64, timeout_s=6.0, queue_timeout_ms=1500)
# Входной ключ шага `address_info` (как `__appointment_mode`, в состояние не попадает): LLM разбора хода уже не
# подтвердила услугу из реплики — инструмент её заново по буквам не ищет.
ADDRESS_SERVICE_DROPPED_KEY = "__service_llm_dropped"


def enabled() -> bool:
    """`address_service_llm = on`."""

    return str(getattr(_cfg, "MR_ADDRESS_SERVICE_LLM", "off") or "").strip().lower() == "on"


@dataclass(frozen=True)
class AddressServiceChoice:
    status: str  # picked | none_fit | unavailable | no_candidates
    rows: list[dict[str, Any]] = field(default_factory=list)
    names: tuple[str, ...] = ()


def _name(row: dict[str, Any]) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip()


async def choose_address_service(query: str, rows: list[dict[str, Any]]) -> AddressServiceChoice:
    """Какие из строк прайса, найденных правилами, — услуга, о месте которой спрашивает реплика `query`.

    :param query: реплика пациента как есть
    :param rows: строки прайса из отбора правил (с полями места оказания)
    :return: выбранные строки (все строки выбранных названий) или статус без выбора
    """

    names: list[str] = []
    for row in rows:
        name = _name(row) if isinstance(row, dict) else ""
        if name and name not in names:
            names.append(name)
    names = names[:OPTIONS_LIMIT]
    if not names:
        return AddressServiceChoice("no_candidates")
    options = [Option(id=str(i), text=name) for i, name in enumerate(names)]
    choice = await choose(ADDRESS_SERVICE_TASK, query, options)
    if choice.status is Status.PICKED:
        picked = tuple(names[int(i)] for i in choice.ids)
        return AddressServiceChoice(
            "picked", rows=[row for row in rows if isinstance(row, dict) and _name(row) in picked], names=picked
        )
    if choice.status is Status.NONE_FIT:
        return AddressServiceChoice("none_fit")
    return AddressServiceChoice("unavailable")
