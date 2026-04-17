"""Модуль домена адресов и филиалов.

Содержит pilot-миграцию address-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..services_legacy import Services


def _legacy_module():
    """Лениво импортирует legacy-модуль, чтобы не создать цикл импортов.

    :return: модуль ``messengers_router.services_legacy``
    """

    from .. import services_legacy as legacy

    return legacy


async def address_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает адреса филиалов и контактную информацию.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: словарь с адресами и данными о филиалах
    """

    legacy = _legacy_module()

    try:
        regions = await self._ensure_regions_loaded()
    except Exception:
        regions = []
    # Работаем только по Самаре.
    regions = [
        r for r in regions
        if isinstance(r, dict)
        and (
            legacy._is_samara_city_value(str(r.get("city") or ""))
            or "самара" in legacy._normalise_input(str(r.get("name") or ""))
            or "самара" in legacy._normalise_input(str(r.get("addressForSite") or ""))
        )
    ]
    appointment_mode = bool(entities.get("__appointment_mode"))
    branch = legacy._get_first_present(entities, ["region", "branch", "company_unit", "unit", "city"]) or ""
    if not branch:
        raw_query = str(query or "").strip()
        if raw_query and (legacy._looks_like_real_address(raw_query) or legacy._ADDRESS_HINT_RE.search(raw_query)):
            branch = raw_query
    branch_q = legacy._normalise_input(branch)
    service_name = legacy._get_first_present(entities, ["service_name", "test_name"]) or ""
    if not service_name:
        extracted = legacy.extract_service_phrase(query or "")
        if extracted:
            service_name = extracted
    service_q = legacy._normalise_input(service_name)
    city_for_static = legacy._get_first_present(entities, ["city"])
    if city_for_static and legacy._is_non_samara_city_value(city_for_static):
        return legacy._service_fallback(
            note=f"address_info unsupported city: {city_for_static}",
            handoff_message=legacy.handoff_message("city_not_supported"),
            entities=entities,
            reason="city_not_supported",
            extra={"addresses": [], "branches": []},
        )

    allowed_doctor_addresses_norm: set[str] = set()
    if appointment_mode:
        try:
            doctors = await self._ensure_doctors_cache_loaded()
        except Exception:
            doctors = []
        for d in doctors:
            if not isinstance(d, dict):
                continue
            for addr in (d.get("regions") or d.get("addresses") or []):
                a = str(addr).strip()
                if not a or not legacy._looks_like_real_address(a):
                    continue
                allowed_doctor_addresses_norm.add(legacy._normalise_input(a))

    if service_q and (appointment_mode or legacy._is_procedure_branch_lookup_query(query, service_q)):
        try:
            retail_rows = await asyncio.to_thread(legacy.api_price.load_price_by_region, legacy.SAMARA_PRICE_REGION_ID)
        except Exception:
            retail_rows = []
        if isinstance(retail_rows, list) and retail_rows:
            care_query = str(service_name or query or "").strip()
            retail_matches = legacy._select_address_price_rows(
                [row for row in retail_rows if isinstance(row, dict)],
                care_query,
                limit=10,
                family_limit=50,
            )
            care_addresses = legacy._care_setting_addresses_from_price_rows(retail_matches)
            if branch_q:
                care_addresses = [
                    addr for addr in care_addresses
                    if branch_q in legacy._normalise_input(addr)
                ]
            if care_addresses:
                return {
                    "addresses": care_addresses,
                    "branches": legacy._addresses_to_branch_payload(care_addresses, regions),
                    "note": "address_info: priceUnits care-setting",
                    "entities_used": entities,
                }

    if service_q and legacy._is_procedure_branch_lookup_query(query, service_q):
        procedure_branches = await self._procedure_branches_from_index(service_q, regions)
        if procedure_branches:
            if branch_q:
                procedure_branches = [
                    b
                    for b in procedure_branches
                    if branch_q in legacy._normalise_input(str(b.get("address") or ""))
                ]
            if procedure_branches:
                return {
                    "addresses": [str(b.get("address") or "").strip() for b in procedure_branches if str(b.get("address") or "").strip()],
                    "branches": procedure_branches,
                    "note": "address_info: procedure->branches (doctor_prices index)",
                    "entities_used": entities,
                }

    if service_q:
        regions = legacy._filter_regions_by_service_flags(regions, service_q)

    addresses: list[str] = []
    branches: list[dict[str, Any]] = []
    for r in regions:
        if not isinstance(r, dict):
            continue
        rid = legacy._as_int(r.get("id"))
        disp = legacy._region_display_name(r)
        if not disp:
            continue
        # в выдачу пациенту пускаем только реальные адреса филиалов
        if not legacy._looks_like_real_address(disp):
            continue
        hay = " | ".join(
            [
                legacy._normalise_input(disp),
                legacy._normalise_input(str(r.get("name") or "")),
                legacy._normalise_input(str(r.get("city") or "")),
            ]
        )
        if branch_q and branch_q not in hay:
            continue
        addresses.append(disp)
        branches.append(
            {
                "id": rid,
                "address": disp,
                "city": str(r.get("city") or "").strip(),
                "phone": legacy._extract_region_phone(r),
                "work_time": legacy._extract_region_work_time(r),
            }
        )

    uniq = sorted(set(addresses))
    if uniq:
        by_addr: dict[str, dict[str, Any]] = {}
        for b in branches:
            addr = str(b.get("address") or "").strip()
            if not addr:
                continue
            prev = by_addr.get(addr)
            if prev is None:
                by_addr[addr] = b
                continue
            prev_score = int(bool(prev.get("phone"))) + int(bool(prev.get("work_time")))
            cur_score = int(bool(b.get("phone"))) + int(bool(b.get("work_time")))
            if cur_score > prev_score:
                by_addr[addr] = b

        if appointment_mode and allowed_doctor_addresses_norm:
            def _is_doctor_capable(addr: str) -> bool:
                n = legacy._normalise_input(addr)
                for x in allowed_doctor_addresses_norm:
                    if n == x or n in x or x in n:
                        return True
                return False

            filtered_addrs = [a for a in uniq if _is_doctor_capable(a)]
            if filtered_addrs:
                uniq = filtered_addrs
                by_addr = {k: v for k, v in by_addr.items() if _is_doctor_capable(k)}

        note = "address_info: live regions API"
        if service_q:
            note += " + filtered by service flags"
        if appointment_mode:
            note += " + filtered by doctor-capable branches"
        return {
            "addresses": uniq,
            "branches": list(by_addr.values()),
            "note": note,
            "entities_used": entities,
        }

    # fallback: старый путь через кэш врачей
    try:
        doctors = await self._ensure_doctors_cache_loaded()
    except Exception:
        doctors = []
    fallback: list[str] = []
    for d in doctors:
        for addr in (d.get("regions") or d.get("addresses") or []):
            a = str(addr).strip()
            if not a:
                continue
            if not legacy._looks_like_real_address(a):
                continue
            if branch_q and branch_q not in legacy._normalise_input(a):
                continue
            fallback.append(a)
    return {
        "addresses": sorted(set(fallback)),
        "branches": [{"address": a, "phone": "", "work_time": ""} for a in sorted(set(fallback))],
        "note": "address_info: doctors cache fallback",
        "entities_used": entities,
    }
