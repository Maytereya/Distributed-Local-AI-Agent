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
  второй круг по терминам LLM (как у выбора строки цены): строки прайса по названиям, под которыми услуга бывает в прайсе
  («магнитотерапию» → «Магнитотерапия»), и второй выбор; «ни одна» и во втором круге — услуга снимается, общий список
  филиалов (решение владельца 19.10, п. 2); сбой или мусор LLM — как было (решение 18.10 для ADDRESS: «LLM недоступна —
  как было, по правилам»);
- LLM разбора хода выбрала НЕСКОЛЬКО названий («Массаж спины 30 / 45 / 60 мин», варианты магнитотерапии): для записи это
  выбор пациента, а для «где сделать» — ответ: места оказания этих строк. Грундер кладёт названия в `ADDRESS_SERVICE_PICKS_KEY`,
  инструмент ищет места по ним, а не общий список (BUG-2026-10-20-ADDRESS-MULTI-PICK-AS-DROPPED).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from .choice import Option, Status, Task, choose
from .runtime_config import config as _cfg

# Сколько различных названий строк показать модели: шире — отбор правил уже шумный (семейство до 50 строк).
OPTIONS_LIMIT = 8
# Термины второго круга — как у выбора строки цены (`_price_select_llm.PRICE_TASK`): до 3 названий, под которыми услуга
# бывает в прайсе; ответ длиннее номеров — `num_predict` с запасом на них.
ADDRESS_SERVICE_TASK = Task(
    name="address_service", max_picks=3, max_terms=3, num_predict=128, timeout_s=6.0, queue_timeout_ms=1500
)
# Входной ключ шага `address_info` (как `__appointment_mode`, в состояние не попадает): LLM разбора хода уже не
# подтвердила услугу из реплики — инструмент её заново по буквам не ищет.
ADDRESS_SERVICE_DROPPED_KEY = "__service_llm_dropped"
# Названия, которые выбрала LLM разбора хода, когда их несколько (`entity_grounder._apply_service_verdict`, вердикт `choose`):
# служебный ключ решения → состояние → планировщик снимает его в начале хода и отдаёт во вход шага `address_info`.
ADDRESS_SERVICE_PICKS_KEY = "_address_service_picks"


def enabled() -> bool:
    """`address_service_llm = on`."""

    return str(getattr(_cfg, "MR_ADDRESS_SERVICE_LLM", "off") or "").strip().lower() == "on"


@dataclass(frozen=True)
class AddressServiceChoice:
    status: str  # picked | none_fit | unavailable | no_candidates
    rows: list[dict[str, Any]] = field(default_factory=list)
    names: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()  # термины второго круга, если он был


def _name(row: dict[str, Any]) -> str:
    return str(row.get("serviceName") or row.get("name") or "").strip()


def _distinct_names(rows: Sequence[dict[str, Any]], *, skip: Sequence[str] = ()) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = _name(row) if isinstance(row, dict) else ""
        if name and name not in names and name not in skip:
            names.append(name)
    return names[:OPTIONS_LIMIT]


async def _pick(query: str, rows: list[dict[str, Any]], names: list[str]) -> tuple[AddressServiceChoice, tuple[str, ...]]:
    """Один выбор: (решение, термины LLM при «ни одна»)."""

    options = [Option(id=str(i), text=name) for i, name in enumerate(names)]
    choice = await choose(ADDRESS_SERVICE_TASK, query, options)
    if choice.status is Status.PICKED:
        picked = tuple(names[int(i)] for i in choice.ids)
        rows = [row for row in rows if isinstance(row, dict) and _name(row) in picked]
        return AddressServiceChoice("picked", rows=rows, names=picked), ()
    if choice.status is Status.NONE_FIT:
        return AddressServiceChoice("none_fit"), tuple(choice.terms)
    return AddressServiceChoice("unavailable"), ()


async def choose_address_service(
    query: str,
    rows: list[dict[str, Any]],
    *,
    search: Callable[[Sequence[str]], Awaitable[list[dict[str, Any]]]] | None = None,
) -> AddressServiceChoice:
    """Какие из строк прайса, найденных правилами, — услуга, о месте которой спрашивает реплика `query`.

    :param query: реплика пациента как есть
    :param rows: строки прайса из отбора правил (с полями места оказания)
    :param search: термины LLM → строки прайса (тот же отбор правил); None — без второго круга
    :return: выбранные строки (все строки выбранных названий) или статус без выбора
    """

    names = _distinct_names(rows)
    if not names:
        return AddressServiceChoice("no_candidates")
    first, terms = await _pick(query, rows, names)
    if first.status != "none_fit" or not terms or search is None:
        return first
    # «Ни одна» среди строк по буквам реплики («магнитотерапию» → «Mg+ (магний)»): правильную строку правила могли не
    # найти — второй круг по названиям, под которыми услугу пишут в прайсе (как у выбора строки цены).
    found = [row for row in await search(terms) if isinstance(row, dict)]
    second_names = _distinct_names(found, skip=names)
    if not second_names:
        return AddressServiceChoice("none_fit", terms=terms)
    second, _ = await _pick(query, found, second_names)
    if second.status == "picked":
        return AddressServiceChoice("picked", rows=second.rows, names=second.names, terms=terms)
    # Второй круг не решил (пусто, сбой) — в силе вердикт первого: «ни одна».
    return AddressServiceChoice("none_fit", terms=terms)


def turn_picks(value: Any) -> tuple[str, ...]:
    """Названия из `ADDRESS_SERVICE_PICKS_KEY` (значение состояния — недоверенный ввод): только непустые строки."""

    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(v).strip() for v in value if isinstance(v, str) and str(v).strip()))
