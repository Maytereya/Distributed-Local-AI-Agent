"""Ответ адресами по данным филиалов, а не одним шаблоном списка (класс B метрики «рабочий бот», 25.10).

Замер `buttons_v1` (wb-metric, деплой 18): 87 провалов (9,8 п. п.) — на вопрос внутри «Адреса и часы работы» / «Где и до скольки
принимают» бот повторял общий список всех филиалов, а в 26 ответах «где сдать» в списке был филиал без забора анализов
(Ново-Садовая, 106 к.82: `analysis=false`). BUG-2026-10-25-PASS-BUTTON-FOLLOWUP-BRANCH-TEMPLATE.

* `address_branch_flags` — список филиалов в ответе о сдаче анализов (ЭКГ, УЗИ) без записи и на кнопке «Где и до скольки
  принимают» (лист раздела «Сдать анализы») и следующем ходе — только филиалы с этим флагом данных филиалов. Что именно без
  записи, называет ответ (`walkin`: LLM по правилу клиники), флаги — данные МИС (`/site/regions`).

Метка адресной кнопки: нажатый лист адресов режима `pass` и возраст в ходах — на нажатии 0, на следующей реплике 1; другое
нажатие или второй ход после — метки нет. Метка хода — переменная хода; между ходами — `_address_button` в состоянии (пишет конец
хода нажатия), в инструменты не идёт (планировщик её снимает).
"""

from __future__ import annotations

import contextvars
from typing import Any, Iterable

from . import button_menu
from .mess_types import SessionState
from .runtime_config import config as _cfg

ADDRESS_BUTTON_KEY = "_address_button"
_FLAGS = ("analysis", "ecg", "usi")


def branch_flags_on() -> bool:
    """`[MESSENGER_ROUTER] address_branch_flags = on`."""

    return str(getattr(_cfg, "MR_ADDRESS_BRANCH_FLAGS", "off") or "").strip().lower() == "on"


def tracks_address_button() -> bool:
    """Метку адресной кнопки ставим, только когда её кто-то читает (off — состояние как было)."""

    return branch_flags_on()


def _address_leaf(button_id: str) -> bool:
    action = button_menu.action_for(button_id) if button_id else None
    return action is not None and action.mode == "pass" and action.label == "ADDRESS"


_MARK: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("mr_address_button", default=None)


def track_address_button(state: SessionState, button_id: str) -> None:
    """Начало хода: нажатие адресного листа — метка этого хода (возраст 0), реплика после него — возраст 1; другое нажатие и
    вторая реплика — метки нет. Метка хода держится в переменной хода: состояние ход может сбросить (смена темы)."""

    _MARK.set(None)
    if not tracks_address_button():
        return
    entities = state.last_entities
    previous = entities.pop(ADDRESS_BUTTON_KEY, None)
    if button_id:
        if _address_leaf(button_id):
            _MARK.set({"id": button_id, "age": 0})
        return
    if isinstance(previous, dict) and int(previous.get("age") or 0) == 0 and previous.get("id"):
        _MARK.set({"id": str(previous["id"]), "age": 1})


def finish_turn(state: SessionState) -> None:
    """Конец хода: метку нажатия — в состояние, чтобы её увидела следующая реплика; прочее — снять."""

    mark = _MARK.get()
    if mark is not None and int(mark.get("age") or 0) == 0:
        state.last_entities[ADDRESS_BUTTON_KEY] = dict(mark)
    else:
        state.last_entities.pop(ADDRESS_BUTTON_KEY, None)


def address_button() -> str:
    """Адресный лист, нажатый на этом ходе или на прошлом (эта реплика — следующая после нажатия); иначе пусто."""

    mark = _MARK.get()
    return str(mark.get("id") or "") if mark is not None else ""


def needed_flags(walkin_hint: str | None) -> frozenset[str]:
    """Флаги филиала, без которых филиал в этом ответе — неправда: что без записи назвал ответ («анализы», «ЭКГ», «УЗИ») и
    область адресного листа («Где и до скольки принимают» — `analysis`)."""

    from .services._regions import branch_flags_for

    wanted = set(branch_flags_for(walkin_hint or ""))
    action = button_menu.action_for(address_button())
    if action is not None and action.scope in _FLAGS:
        wanted.add(action.scope)
    return frozenset(wanted)


def filter_branches(branches: Iterable[Any], wanted: frozenset[str]) -> list[Any] | None:
    """Филиалы, у которых по данным есть все нужные флаги; флагов в данных нет или фильтр оставил пусто — None (как было)."""

    rows = list(branches)
    if not wanted or not rows:
        return None
    if not all(isinstance(r, dict) and all(f in r for f in wanted) for r in rows):
        return None
    kept = [r for r in rows if all(bool(r.get(f)) for f in wanted)]
    return kept if kept and len(kept) < len(rows) else None


def region_flags(region: dict[str, Any] | None) -> dict[str, bool]:
    """Флаги места оказания из строки `/site/regions` — в данные ответа адресами (только под выключателем)."""

    if not branch_flags_on() or not isinstance(region, dict):
        return {}
    return {flag: bool(region.get(flag)) for flag in _FLAGS if flag in region}
