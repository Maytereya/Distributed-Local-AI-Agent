"""Модуль домена адресов и филиалов.

Содержит pilot-миграцию address-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent_logic_2.nayka_api import api_price

from ..policies import handoff_message
from ..service_phrase import extract_service_phrase
from ._addresses_helpers import (
    _addresses_to_branch_payload,
    _is_procedure_branch_lookup_query,
    _looks_like_real_address,
    _static_procedure_addresses,
)
from ._common import (
    _as_int,
    _get_first_present,
    _normalise_input,
    _service_fallback,
)
from ._doctors_helpers import (
    _doctor_role_specialty_match_level,
    _procedure_query_role_specialty,
)
from ._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _care_setting_addresses_from_price_rows,
    _select_address_price_rows,
)
from ._regions import (
    _ADDRESS_HINT_RE,
    _extract_region_phone,
    _extract_region_work_time,
    _filter_regions_by_service_flags,
    _has_explicit_non_samara_regions,
    _is_non_samara_city_value,
    _is_samara_city_value,
    _region_display_name,
    _region_matches_samara_tokens,
)

if TYPE_CHECKING:
    from .core import Services


async def address_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает адреса филиалов и контактную информацию.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: словарь с адресами и данными о филиалах
    """

    try:
        regions = await self._ensure_regions_loaded()
    except Exception:
        regions = []
    # Работаем только по Самаре.
    regions = [
        r for r in regions
        if isinstance(r, dict)
        and (
            _is_samara_city_value(str(r.get("city") or ""))
            or "самара" in _normalise_input(str(r.get("name") or ""))
            or "самара" in _normalise_input(str(r.get("addressForSite") or ""))
        )
    ]
    appointment_mode = bool(entities.get("__appointment_mode"))
    branch = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city"]) or ""
    if not branch:
        raw_query = str(query or "").strip()
        if raw_query and (_looks_like_real_address(raw_query) or _ADDRESS_HINT_RE.search(raw_query)):
            branch = raw_query
    branch_q = _normalise_input(branch)
    service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
    if not service_name:
        extracted = extract_service_phrase(query or "")
        if extracted:
            service_name = extracted
    service_q = _normalise_input(service_name)
    city_for_static = _get_first_present(entities, ["city"])
    if city_for_static and _is_non_samara_city_value(city_for_static):
        return _service_fallback(
            note=f"address_info unsupported city: {city_for_static}",
            handoff_message=handoff_message("city_not_supported"),
            entities=entities,
            reason="city_not_supported",
            extra={"addresses": [], "branches": []},
        )

    # Override для процедур с физически фиксированной точкой оказания
    # (маммограф / флюорограф установлены только на Ленина 5). Этот шорт-каррент
    # нужно проверять ДО Path 1/2/3, потому что priceByRegion раскладывает
    # такие процедуры по нескольким priceUnit/care-setting'ам и фолбэк на
    # CARE_SETTING_ADDRESS_BY_ROOT_ID ошибочно тянет Ново-Садовую.
    fixed_addresses = api_price.resolve_diagnostic_fixed_addresses(
        service_name or query or ""
    )
    if fixed_addresses:
        if branch_q:
            fixed_addresses = [
                addr for addr in fixed_addresses
                if branch_q in _normalise_input(addr)
            ]
        if fixed_addresses:
            return {
                "addresses": list(fixed_addresses),
                "branches": _addresses_to_branch_payload(fixed_addresses, regions),
                "note": "address_info: diagnostic equipment fixed address",
                "entities_used": entities,
            }

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
                if not a or not _looks_like_real_address(a):
                    continue
                allowed_doctor_addresses_norm.add(_normalise_input(a))

    if service_q and (appointment_mode or _is_procedure_branch_lookup_query(query, service_q)):
        try:
            retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
        except Exception:
            retail_rows = []
        if isinstance(retail_rows, list) and retail_rows:
            care_query = str(service_name or query or "").strip()
            retail_matches = _select_address_price_rows(
                [row for row in retail_rows if isinstance(row, dict)],
                care_query,
                limit=10,
                family_limit=50,
            )
            care_addresses = _care_setting_addresses_from_price_rows(retail_matches)
            if branch_q:
                care_addresses = [
                    addr for addr in care_addresses
                    if branch_q in _normalise_input(addr)
                ]
            if care_addresses:
                return {
                    "addresses": care_addresses,
                    "branches": _addresses_to_branch_payload(care_addresses, regions),
                    "note": "address_info: priceUnits care-setting",
                    "entities_used": entities,
                }

    if service_q and _is_procedure_branch_lookup_query(query, service_q):
        procedure_branches = await self._procedure_branches_from_index(service_q, regions)
        if procedure_branches:
            if branch_q:
                procedure_branches = [
                    b
                    for b in procedure_branches
                    if branch_q in _normalise_input(str(b.get("address") or ""))
                ]
            if procedure_branches:
                return {
                    "addresses": [str(b.get("address") or "").strip() for b in procedure_branches if str(b.get("address") or "").strip()],
                    "branches": procedure_branches,
                    "note": "address_info: procedure->branches (doctor_prices index)",
                    "entities_used": entities,
                }

    if service_q:
        regions = _filter_regions_by_service_flags(regions, service_q)

    addresses: list[str] = []
    branches: list[dict[str, Any]] = []
    for r in regions:
        if not isinstance(r, dict):
            continue
        rid = _as_int(r.get("id"))
        disp = _region_display_name(r)
        if not disp:
            continue
        # в выдачу пациенту пускаем только реальные адреса филиалов
        if not _looks_like_real_address(disp):
            continue
        hay = " | ".join(
            [
                _normalise_input(disp),
                _normalise_input(str(r.get("name") or "")),
                _normalise_input(str(r.get("city") or "")),
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
                "phone": _extract_region_phone(r),
                "work_time": _extract_region_work_time(r),
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
                n = _normalise_input(addr)
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
            if not _looks_like_real_address(a):
                continue
            if branch_q and branch_q not in _normalise_input(a):
                continue
            fallback.append(a)
    return {
        "addresses": sorted(set(fallback)),
        "branches": [{"address": a, "phone": "", "work_time": ""} for a in sorted(set(fallback))],
        "note": "address_info: doctors cache fallback",
        "entities_used": entities,
    }


async def _procedure_branches_from_index(
    self: "Services",
    service_q: str,
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Находит филиалы для процедуры по индексу `doctor_prices`.

    :param service_q: нормализованная процедура
    :param regions: live-список филиалов /regions (для phone/work_time)
    :return: список branches в формате address_info
    """

    role_specialty = _procedure_query_role_specialty(service_q)
    if role_specialty:
        doctors = await self._ensure_doctors_cache_loaded()
        samara_tokens = await self._samara_region_tokens()
        role_addresses: list[str] = []
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            if _doctor_role_specialty_match_level(doc, role_specialty) <= 0:
                continue
            regions_src = [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()]
            if _has_explicit_non_samara_regions(regions_src):
                continue
            for addr in regions_src:
                if not _looks_like_real_address(addr):
                    continue
                if samara_tokens and not _region_matches_samara_tokens(addr, samara_tokens):
                    continue
                if addr not in role_addresses:
                    role_addresses.append(addr)
        if role_addresses:
            return _addresses_to_branch_payload(role_addresses, regions)

    # Раньше здесь был fallback через `_ensure_procedure_rows_loaded` →
    # `doctor_prices.regionName`. Этот источник возвращает филиал ПРИЁМА
    # врача, а не место оказания услуги — тот же баг, что починили в
    # /priceByRegion-ветке (см. PR #14). Поскольку Path 1 (priceByRegion +
    # priceUnit override) уже отрабатывает корректно, здесь оставляем
    # только статический справочник для diagnostic-процедур, по которым
    # /priceByRegion ничего не отдаёт.
    addresses = _static_procedure_addresses(service_q)
    return _addresses_to_branch_payload(addresses, regions)
