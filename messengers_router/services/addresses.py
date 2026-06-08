"""Модуль домена адресов и филиалов.

Содержит pilot-миграцию address-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent_logic_2.nayka_api import api_price

from ..policies import handoff_message
from .. import resilience as _R
from ..service_phrase import extract_service_phrase
from ._addresses_helpers import (
    _addresses_to_branch_payload,
    _branch_query_matches,
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
    _service_procedure_flag,
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

    # /regions tech-failure can surface two ways: the call raising here (e.g. a test
    # double, or a future direct raise), OR the real _ensure_regions_loaded swallowing
    # the exception under its lock and setting self._regions_last_failed. Treat either as degraded.
    regions_tech_failed = False
    try:
        regions = await self._ensure_regions_loaded()
    except Exception:
        regions = []
        regions_tech_failed = True
    if self._regions_last_failed:
        regions_tech_failed = True
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
    # Track whether branch came from *only* the city entity (no specific branch/region/unit
    # was mentioned). When true we must NOT use branch_q as a substring filter on doctor
    # addresses — those addresses don't carry the city prefix (e.g. «пр.Ленина, 5» has no
    # «самара») so city-as-branch filtering drops ALL valid doctor branches.
    # City-scoping relies on the existing Samara-token logic instead.
    _branch_is_city_only = (
        not _get_first_present(entities, ["region", "branch", "company_unit", "unit"])
        and bool(_get_first_present(entities, ["city"]))
        and bool(branch)
    )
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
    # Место оказания процедуры (УЗИ/анализы/ЭКГ) определяется флагом филиала из
    # /site/regions (usi/analysis/ecg) — авторитетный источник по словам
    # разработчика Наяки. Для таких услуг адреса берём фильтром регионов по
    # флагу (ниже, Path live regions), а priceUnit care-setting хардкод и
    # пересечение с локациями врачей пропускаем (они давали единственный
    # дефолтный адрес «Ленина 5» вместо всех филиалов с флагом).
    procedure_flag = _service_procedure_flag(service_q) if service_q else None
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
                if _branch_query_matches(branch_q, _normalise_input(addr))
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
        # При записи ПО СПЕЦИАЛЬНОСТИ ограничиваем допустимые адреса врачами ЭТОЙ
        # специальности (как Fix B в doctors-cache fallback ниже). Иначе live-regions
        # путь, отфильтровав «doctor-capable» по ЛЮБОМУ врачу, вернёт самарские клиники
        # без нужной специальности (регресс инварианта BUG-2026-06-04-04 — оффер
        # филиала, где нет врача запрошенной специальности). Без specialty — поведение
        # прежнее (любые врачебные адреса).
        appt_specialty_q = _normalise_input(_get_first_present(entities, ["specialty"]) or "")
        for d in doctors:
            if not isinstance(d, dict):
                continue
            if appt_specialty_q and _doctor_role_specialty_match_level(d, appt_specialty_q) <= 0:
                continue
            for addr in (d.get("regions") or d.get("addresses") or []):
                a = str(addr).strip()
                if not a or not _looks_like_real_address(a):
                    continue
                allowed_doctor_addresses_norm.add(_normalise_input(a))

    # Запись ПО СПЕЦИАЛЬНОСТИ (приём врача) — НЕ процедура: priceUnits care-setting
    # (Path 2 ниже) и procedure-index (Path 3) её перехватывали и сужали до одного
    # care-setting адреса («Ленина 5») вместо ВСЕХ филиалов спец-ти (дезинформация —
    # будто врачи только там). Такие запросы ведём на doctor-capable путь, который
    # отдаёт все филиалы этой специальности (диагностика 2026-06-08, прод-probe).
    appt_by_specialty = appointment_mode and bool(
        _normalise_input(_get_first_present(entities, ["specialty"]) or "")
    )

    if service_q and not procedure_flag and not appt_by_specialty and (appointment_mode or _is_procedure_branch_lookup_query(query, service_q)):
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
                    if _branch_query_matches(branch_q, _normalise_input(addr))
                ]
            if care_addresses:
                return {
                    "addresses": care_addresses,
                    "branches": _addresses_to_branch_payload(care_addresses, regions),
                    "note": "address_info: priceUnits care-setting",
                    "entities_used": entities,
                }

    if service_q and not procedure_flag and not appt_by_specialty and _is_procedure_branch_lookup_query(query, service_q):
        procedure_branches = await self._procedure_branches_from_index(service_q, regions)
        if procedure_branches:
            if branch_q:
                procedure_branches = [
                    b
                    for b in procedure_branches
                    if _branch_query_matches(branch_q, _normalise_input(str(b.get("address") or "")))
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
        if branch_q and not _branch_query_matches(branch_q, hay):
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

        if appointment_mode and not procedure_flag and allowed_doctor_addresses_norm:
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
        if appointment_mode and not procedure_flag:
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
    # Fix A (BUG-2026-06-04-04): when branch_q is merely the CITY name (no specific
    # street/unit was mentioned), do NOT use it as a substring filter on doctor
    # addresses. Doctor region strings like «пр.Ленина, 5» carry no city prefix, so
    # «самара» not in «пр.ленина, 5» would drop every valid branch. City scoping is
    # handled by the Samara-region-token path above; here we rely on that filtering
    # having already happened (regions were empty → we're in fallback).
    effective_branch_q = "" if _branch_is_city_only else branch_q

    # Fix B (BUG-2026-06-04-04): when specialty is set + appointment_mode, restrict
    # fallback to doctors whose unit/specialization matches the requested specialty.
    # This ensures only e.g. gynecologist branches appear, never a lab-only branch.
    specialty_q = _normalise_input(
        _get_first_present(entities, ["specialty"]) or ""
    ) if appointment_mode else ""

    fallback: list[str] = []
    for d in doctors:
        if not isinstance(d, dict):
            continue
        if specialty_q and _doctor_role_specialty_match_level(d, specialty_q) <= 0:
            continue
        regions_src = [str(x).strip() for x in (d.get("regions") or d.get("addresses") or []) if str(x).strip()]
        # Skip doctors whose entire regions list belongs explicitly to another city.
        if _has_explicit_non_samara_regions(regions_src):
            continue
        for a in regions_src:
            if not _looks_like_real_address(a):
                continue
            if effective_branch_q and not _branch_query_matches(effective_branch_q, _normalise_input(a)):
                continue
            fallback.append(a)
    deduped = sorted(set(fallback))
    if regions_tech_failed and not deduped:
        # No safe best-effort data AND /regions tech-failed → honest tech_unavailable,
        # never misleading addresses. No auto-handoff (user can type «оператор»).
        # failure_mode is FM_EXCEPTION (not granular http_5xx/timeout) by design: site_regions
        # returns a bare list and RAISES on HTTP error rather than an {ok,status_code} envelope,
        # so an exception is the only failure signal available at this layer (Phase 1).
        _R.log_degraded(upstream="regions", failure_mode=_R.FM_EXCEPTION, fallback_used=False)
        tech_payload = {
            "addresses": [],
            "branches": [],
            "note": "address_info: regions tech_unavailable",
            "handoff_reason": "tech_unavailable",
            "handoff_message": _R.tech_unavailable_text("список филиалов"),
            "entities_used": entities,
        }
        return _R.mark_degraded(
            tech_payload, upstream="regions", failure_mode=_R.FM_EXCEPTION, fallback_used=False
        )
    payload = {
        "addresses": deduped,
        "branches": [{"address": a, "phone": "", "work_time": ""} for a in deduped],
        "note": "address_info: doctors cache fallback",
        "entities_used": entities,
    }
    if regions_tech_failed:
        # Best-effort: doctors-cache addresses served, but mark the degradation.
        _R.log_degraded(upstream="regions", failure_mode=_R.FM_EXCEPTION, fallback_used=True)
        payload = _R.mark_degraded(payload, upstream="regions", failure_mode=_R.FM_EXCEPTION, fallback_used=True)
    return payload


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
