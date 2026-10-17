"""Неразобранные даты и время слотов расписания МИС: счётчик хода и предупреждение — ход (д), шаг 8а, DATA-15.

Слот, дату или время которого код не разобрал, рендер и выбор «ближайшего окна» молча пропускают (а неразобранная
дата дня выключает фильтр по дате). Ответ на этом шаге не меняется; видно, сколько и каких: поле телеметрии хода
`slot_unparsed` (число) и `logger.warning` с образцом — один раз на место и форму значения (цифры → 9: «99.99.9999»).
"""

from __future__ import annotations

import logging
from typing import Any

from . import turn_telemetry

logger = logging.getLogger(__name__)
_warned: set[str] = set()
_DIGITS_TO_NINE = str.maketrans("0123456789", "9999999999")


def _shape(raw: Any) -> str:
    return str(raw).translate(_DIGITS_TO_NINE)[:32]


def unparsed(where: str, raw: Any) -> None:
    """Слот не разобран: `where` — что именно (`slot` — дата и время окна, `date` — дата дня, `time` — время)."""

    turn_telemetry.bump("slot_unparsed")
    key = f"{where}:{_shape(raw)}"
    if key not in _warned:
        _warned.add(key)
        logger.warning("расписание МИС: не разобрано (%s) — образец %r, форма %s", where, str(raw)[:32], _shape(raw))
