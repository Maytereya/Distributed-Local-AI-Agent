"""Кто делает услугу — по связи «врач ↔ услуга» в МИС (`doctor_prices`), а не по буквам процедуры (ход (г), L-06).

Срез `doctor_prices` — строки «врач × филиал × роль приёма (`companyUnitId`) × услуга (`serviceHomecode`)». Код услуги
— тот же, что у строки прайса региона (`price_region_*`, поле `serviceHomecode`; `serviceId` в `doctor_prices` пуст), так
что «кто делает строку прайса» — соединение по коду, без правил «процедура → специальность» в коде
(`PROCEDURE_TO_SPECIALTY`, `ENDOSCOPY_SERVICE_RE`, `UZI_QUERY_RE`). Замер 21.10 на мире 20261020: из 161 строки прайса,
где нынешний резолвер выводит роль из процедуры, у 84 врачей в данных нет вовсе, у 10 врачи по коду — другой роли
(«Эндоскопия полости носа» делает ЛОР, не эндоскопист), у 60 резолвер показывает всех врачей роли вместо тех, кто
делает услугу (дизайн `docs/DESIGN_2026-10-21_step_g_l06_doctors_by_service.md`).

Механика без регулярок и списков слов: модуль не знает ни процедур, ни специальностей, ни города — регион задаёт
предикат вызывающего (храповики правил и переносимости). Пустой результат — не «врачей нет»: что отвечать, решает
вызывающий (пустой или недоступный срез → ход (д), оператор).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ServiceDoctors:
    """Врачи региона, которых МИС связывает с одной услугой (одним кодом)."""

    code: str  # serviceHomecode — ключ строки прайса
    name: str  # serviceName как в МИС
    doctor_ids: tuple[int, ...]
    unit_ids: tuple[int, ...]  # роли приёма (`companyUnitId`), под которыми врачи делают услугу
    branch_ids: tuple[int, ...]  # филиалы (`regionId`), где врачи делают услугу


@dataclass(frozen=True)
class ServiceDoctorIndex:
    services: Mapping[str, ServiceDoctors] = field(default_factory=dict)

    def by_code(self, code: str) -> ServiceDoctors | None:
        return self.services.get(str(code or "").strip())

    def codes_where(self, name_matches: Callable[[str], bool]) -> tuple[str, ...]:
        """Коды услуг, чьё название МИС принимает предикат вызывающего (сопоставление слов — не дело индекса)."""

        return tuple(code for code, item in self.services.items() if name_matches(item.name))

    def doctors_for(self, codes: Iterable[str]) -> tuple[int, ...]:
        """Врачи хотя бы одной из услуг — без повторов, в порядке кодов."""

        out: dict[int, None] = {}
        for code in codes:
            item = self.by_code(code)
            if item is not None:
                out.update(dict.fromkeys(item.doctor_ids))
        return tuple(out)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_service_doctor_index(rows: Sequence[dict[str, Any]], *, in_region: Callable[[dict[str, Any]], bool]) -> ServiceDoctorIndex:
    """Индекс «код услуги → врачи региона» — чистая функция среза `doctor_prices`.

    :param rows: строки среза `doctor_prices_*.jsonl`
    :param in_region: строка относится к врачу и филиалу региона бота (у вызывающего — врачи среза и признаки филиалов)
    :return: индекс; строки без кода или без врача пропускаются; пустой срез — пустой индекс («нет» по нему решать нельзя)
    """

    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("serviceHomecode") or "").strip()
        doctor_id = _as_int(row.get("doctorId"))
        if not code or doctor_id is None or not in_region(row):
            continue
        slot = acc.setdefault(code, {"name": " ".join(str(row.get("serviceName") or "").split()), "doctors": {}, "units": {}, "branches": {}})
        slot["doctors"].setdefault(doctor_id, None)
        for key, raw in (("units", row.get("companyUnitId")), ("branches", row.get("regionId"))):
            value = _as_int(raw)
            if value is not None:
                slot[key].setdefault(value, None)
    return ServiceDoctorIndex(
        services={
            code: ServiceDoctors(
                code=code,
                name=slot["name"],
                doctor_ids=tuple(sorted(slot["doctors"])),
                unit_ids=tuple(sorted(slot["units"])),
                branch_ids=tuple(sorted(slot["branches"])),
            )
            for code, slot in acc.items()
        }
    )


def agreement(data_ids: Iterable[int], shown_ids: Iterable[int]) -> str:
    """Согласие нынешнего ответа («кого показали») с данными МИС («кто делает услугу») — метка телеметрии тени.

    - `no_data` — МИС не связывает услугу ни с одним врачом региона (сравнивать не с чем);
    - `same` — показаны ровно врачи услуги;
    - `shown_wider` — показаны все врачи услуги и ещё другие (врачи роли, а не услуги);
    - `shown_partial` — часть врачей услуги потеряна;
    - `shown_other` — показаны только врачи, которых МИС с услугой не связывает (чужой врач);
    - `shown_none` — не показан никто, хотя врачи услуги есть («не найден» при связи в МИС).
    """

    data, shown = set(data_ids), set(shown_ids)
    if not data:
        return "no_data"
    if not shown:
        return "shown_none"
    if shown == data:
        return "same"
    if not shown & data:
        return "shown_other"
    return "shown_wider" if data <= shown else "shown_partial"


__all__ = ["ServiceDoctorIndex", "ServiceDoctors", "agreement", "build_service_doctor_index"]
