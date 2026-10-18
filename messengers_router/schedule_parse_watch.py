"""Даты слотов расписания МИС: один формат на входе и счётчик неразобранного — ход (д), шаги 8а/8, DATA-15.

Шаг 8 (решение владельца 18.10): дата дня из ответа МИС приводится к ISO (`ГГГГ-ММ-ДД`) сразу на входе
(`normalize_schedule`): ISO с временем или без, `ДД.ММ.ГГГГ`, `ДД.ММ.ГГ`. Дальше выбор окна и рендер видят один формат.
Не разобралась ни одна дата расписания — под `mis_honest_degradation` перевод на оператора (`dates_all_unparsed`).

Слот, дату или время которого код не разобрал, рендер и выбор «ближайшего окна» молча пропускают (а неразобранная
дата дня выключает фильтр по дате). Ответ на этом шаге не меняется; видно, сколько и каких: поле телеметрии хода
`slot_unparsed` (число) и `logger.warning` с образцом — один раз на место и форму значения (цифры → 9: «99.99.9999»).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from . import turn_telemetry

logger = logging.getLogger(__name__)
_warned: set[str] = set()
_DIGITS_TO_NINE = str.maketrans("0123456789", "9999999999")


def _shape(raw: Any) -> str:
    return str(raw).translate(_DIGITS_TO_NINE)[:32]


def parse_day(raw: Any) -> date | None:
    """Дата дня расписания из ответа МИС; формат не известен — None."""

    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_schedule(payload: Any) -> Any:
    """Даты дней в ответе расписания МИС — к ISO (на месте, в копиях строк). Не разобранное остаётся как есть."""

    if not isinstance(payload, list):
        return payload
    out = []
    for row in payload:
        schedule = row.get("schedule") if isinstance(row, dict) else None
        if not isinstance(schedule, dict):
            out.append(row)
            continue
        fixed: dict[str, Any] = {}
        for branch, days in schedule.items():
            if not isinstance(days, list):
                fixed[branch] = days
                continue
            new_days = []
            for day in days:
                if isinstance(day, dict) and day.get("date"):
                    parsed = parse_day(day["date"])
                    if parsed is not None and parsed.isoformat() != str(day["date"]):
                        day = {**day, "date": parsed.isoformat()}
                new_days.append(day)
            fixed[branch] = new_days
        out.append({**row, "schedule": fixed})
    return out


def dates_all_unparsed(payload: Any) -> bool:
    """В расписании есть дни, и ни одна их дата не разобрана."""

    days = [
        day.get("date")
        for row in (payload if isinstance(payload, list) else [])
        if isinstance(row, dict) and isinstance(row.get("schedule"), dict)
        for branch_days in row["schedule"].values()
        if isinstance(branch_days, list)
        for day in branch_days
        if isinstance(day, dict) and day.get("date")
    ]
    return bool(days) and all(parse_day(d) is None for d in days)


def unparsed(where: str, raw: Any) -> None:
    """Слот не разобран: `where` — что именно (`slot` — дата и время окна, `date` — дата дня, `time` — время)."""

    turn_telemetry.bump("slot_unparsed")
    key = f"{where}:{_shape(raw)}"
    if key not in _warned:
        _warned.add(key)
        logger.warning("расписание МИС: не разобрано (%s) — образец %r, форма %s", where, str(raw)[:32], _shape(raw))
