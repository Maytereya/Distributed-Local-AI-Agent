"""Давность прайса в ответе пациенту — ход (д), шаг 3 (решения владельца 08.10 и 18.10; `mis_honest_degradation`).

Правило: срез прайса не старше 2 дней — молча; 3–7 дней — приписка «по данным на ДД.ММ» в конце ответа (вариант «Б»,
18.10: «Цены актуальны для г. …, по данным на 14.10.»); старше 7 дней или срезов нет — перевод `service_error_prices`.
Ночью к переводу дописываются часы операторов (как у любого перевода). С ежедневным обновлением в 07:30 и повтором
каждые 5 минут (`daily_refresh`) срез старше суток означает многодневный сбой МИС.

Давность — из последнего прочитанного среза (`snapshots.last`): розничный прайс (`price_region_*`) или цены врачей
(`doctor_prices`) — по источнику ответа.
"""

from __future__ import annotations

from typing import Any, Callable

from ..policies import handoff_message
from ._common import _service_fallback

SILENT_MAX_DAYS = 2
ANNOTATE_MAX_DAYS = 7


def _has_prices(payload: dict[str, Any]) -> bool:
    return bool(payload.get("prices") or payload.get("retail_prices") or payload.get("family_variants"))


def _source(payload: dict[str, Any]) -> str:
    return "doctor_prices" if "doctorservicepricesbyregion" in str(payload.get("note") or "").lower() else "price_region"


def apply(payload: Any, last_snapshot: Callable[[str], Any]) -> Any:
    """Ответ с ценами (ход (д)): приписка давности (`data_as_of`) или перевод; иначе — как есть.

    :param last_snapshot: `snapshots.last` — передаёт `core` (сервисный слой не множит прямых импортов клиента МИС)
    """

    if not isinstance(payload, dict) or payload.get("handoff_required") or not _has_prices(payload):
        return payload
    snap = last_snapshot(_source(payload))
    if snap is None or snap.status == "fresh":
        return payload
    if snap.status == "unavailable" or snap.age_days is None or snap.age_days > ANNOTATE_MAX_DAYS:
        return _service_fallback(
            note=f"price snapshot {snap.status}, age {snap.age_days}",
            handoff_message=handoff_message("service_error_prices"),
            entities=payload.get("entities_used") or {},
            extra={"prices": []},
        )
    if snap.age_days > SILENT_MAX_DAYS and snap.date is not None:
        return {**payload, "data_as_of": snap.date.strftime("%d.%m")}
    return payload
