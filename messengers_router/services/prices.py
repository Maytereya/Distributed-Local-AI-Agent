"""Модуль домена цен.

Содержит pilot-миграцию price-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Services


def _legacy_module():
    """Лениво импортирует legacy-модуль, чтобы не создать цикл импортов.

    :return: модуль ``messengers_router.services_legacy``
    """

    from . import core as legacy

    return legacy


async def price_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает информацию о ценах на услуги или приём у врача.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: словарь с ценовыми строками или сообщением об ошибке
    """

    legacy = _legacy_module()

    async def _load_with_retry(fn: Any, *args: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.to_thread(fn, *args)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.12)
                    continue
        if last_exc is not None:
            raise last_exc
        return []

    doctor_id = legacy._as_int(entities.get("doctor_id"))
    doctor_name = legacy._get_first_present(entities, ["doctor_name", "doctor", "fio", "last_name", "doctor_last_name"]) or ""
    resolved_doctor_fio: str | None = None
    if not doctor_id and doctor_name:
        doctor_id, resolved_doctor_fio = await self._resolve_doctor_id_from_name(doctor_name)
    if not doctor_id and query and legacy._DOCTOR_PRICE_HINT_RE.search(str(query or "")):
        # Fallback для фраз вида "сколько стоит ... у Белохвостиковой":
        # извлекаем врача из полного текста запроса, даже если classifier не выделил doctor_name.
        doctor_id, q_resolved_fio = await self._resolve_doctor_id_from_name(str(query))
        if doctor_id and q_resolved_fio:
            resolved_doctor_fio = q_resolved_fio
            doctor_name = q_resolved_fio
    entity_service_name = legacy._get_first_present(entities, ["service_name", "test_name"]) or ""
    query_text = str(query or "").strip()
    has_price_request = bool(query_text and legacy._PRICE_REQUEST_RE.search(query_text))
    if legacy._is_price_show_all_request(query_text):
        family_payload = legacy._price_family_payload_from_context(entities, show_all=True)
        if family_payload:
            family_payload["entities_used"] = entities
            return family_payload
    if legacy._is_generic_uzi_price_request(query_text):
        return {
            "prices": [],
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
            "note": "price_info: generic_uzi_clarify",
            "entities_used": entities,
        }
    doctor_query_specialty = legacy._extract_specialty_from_text(query_text) if (doctor_id and has_price_request) else ""

    if doctor_id:
        # Для doctor-specific PRICE не приземляемся в городский retail-catalog:
        # иначе вопрос "у Иванова" может маппиться в случайную услугу по всему прайсу.
        query_service_name = legacy._extract_price_service_from_query(query_text) if query_text else None
        if not query_service_name:
            query_service_name = entity_service_name
        # "сколько стоит прием у <врач>" без специальности:
        # принудительно удерживаем консультационный контекст вместо stale service_name.
        if has_price_request and legacy._PRICE_CONSULT_HINT_RE.search(query_text) and not doctor_query_specialty:
            if not legacy._is_consultation_service_query(str(query_service_name or "")):
                query_service_name = "консультация"
        service_name = query_service_name or ""
        # Если это doctor-specific price без явной услуги, оставляем query как fallback
        # (для редких строк doctor_price, где нет стандартных маркеров).
        if not service_name and has_price_request and legacy._DOCTOR_PRICE_HINT_RE.search(query_text):
            service_name = query_text
    else:
        if entity_service_name and legacy._is_city_only_reply(query_text):
            query_service_name = None
        else:
            query_service_name = legacy.resolve_price_service_name_from_catalog(
                query_text,
                current_service_name=entity_service_name,
            ) or legacy._extract_price_service_from_query(query_text)
        service_name = legacy._select_effective_price_service_name(
            entity_service_name,
            query_service_name,
        )
    needle = legacy._normalise_input(service_name)

    if doctor_id:
        # doctor prices: branch-level regionId из /doctorServicePricesByRegion cache
        try:
            prices = await _load_with_retry(legacy.api_price.load_doctor_prices)
        except Exception:
            return legacy._service_fallback(
                note="price_info source unavailable",
                handoff_message=legacy.handoff_message("service_error_prices"),
                entities=entities,
                extra={"prices": []},
            )
        doc_prices = [p for p in prices if legacy._as_int(p.get("doctorId")) == doctor_id]
        if needle:
            ranked = legacy._rank_price_rows(doc_prices, service_name, limit=10)
            if (
                not ranked
                and has_price_request
                and legacy._PRICE_CONSULT_HINT_RE.search(query_text)
            ):
                ranked = [
                    p for p in doc_prices
                    if legacy._is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                ]
            doc_prices = ranked
        if not needle:
            if has_price_request and legacy._PRICE_CONSULT_HINT_RE.search(query_text):
                consult_rows = [
                    p for p in doc_prices
                    if legacy._is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                ]
                if consult_rows:
                    doc_prices = consult_rows
            doc_prices = sorted(
                [p for p in doc_prices if isinstance(p, dict)],
                key=lambda p: (legacy._normalise_input(str(p.get("serviceName") or "")), legacy._as_int(p.get("cost")) or 0),
            )
        doc_prices = legacy._annotate_price_rows_with_care_context(doc_prices[:10])
        return {
            "prices": doc_prices,
            "note": "price_info: doctorServicePricesByRegion (branch-level regionId)",
            "entities_used": {
                **entities,
                "doctor_id_resolved": doctor_id,
                "doctor_name_resolved": resolved_doctor_fio or doctor_name or "",
                "service_name_effective": service_name,
            },
        }

    # retail prices: city-level regionId в /priceByRegion/{cityRegionId}
    try:
        price_rows = await _load_with_retry(legacy.api_price.load_price_by_region, legacy.SAMARA_PRICE_REGION_ID)
    except Exception:
        return legacy._service_fallback(
            note="price_info source unavailable",
            handoff_message=legacy.handoff_message("service_error_prices"),
            entities=entities,
            extra={"prices": []},
        )
    retail_rows_clean = [p for p in price_rows if isinstance(p, dict)]
    # Мульти-услуговый запрос («ВИЧ, гепатит, ОАК»): активируется только если
    # ≥2 фрагментов приземлились на реальные услуги каталога. Иначе — fallback
    # на существующий single-service путь, старые кейсы остаются без изменений.
    multi_payload = legacy._build_multi_price_payload(query_text, retail_rows_clean)
    if multi_payload is not None:
        multi_payload["entities_used"] = {
            **entities,
            "service_name_effective": str(multi_payload.get("service_name") or "").strip(),
        }
        return multi_payload
    family_payload = legacy._build_price_family_payload(
        query_text,
        retail_rows_clean,
        show_all=False,
        visible_limit=10,
    )
    if family_payload and not doctor_id:
        family_payload["entities_used"] = {
            **entities,
            "service_name_effective": str(family_payload.get("service_name") or "").strip(),
        }
        return family_payload
    if not needle:
        return {"prices": [], "note": "no service query", "entities_used": entities}
    retail_query = service_name
    if query_text and not legacy._is_city_only_reply(query_text):
        query_candidate = legacy._extract_price_service_from_query(query_text)
        if legacy._should_prefer_retail_query_candidate(query_candidate or "", service_name):
            retail_query = query_candidate or service_name
    matches = legacy._select_patient_price_rows(
        [p for p in price_rows if isinstance(p, dict)],
        retail_query,
        limit=10,
    )
    matches = legacy._annotate_price_rows_with_care_context(matches)
    return {
        "prices": matches,
        "service_kind": "lab" if legacy._classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=matches,
            has_exact_doctor_link=False,
            is_consult_query=False,
        ) == "lab" else "",
        "note": f"price_info: priceByRegion({legacy.SAMARA_PRICE_REGION_ID})",
        "entities_used": {
            **entities,
            "service_name_effective": service_name,
        },
    }


async def service_bundle_info(
    self: "Services",
    query: str,
    entities: dict[str, Any],
    *,
    top_n: int | None = None,
) -> dict[str, Any]:
    legacy = _legacy_module()

    top_limit = legacy._coerce_top_n(top_n, default=legacy.DOCTORS_TOP_N)
    entity_service_name = legacy._get_first_present(entities, ["service_name", "test_name"]) or ""
    query_text = str(query or "").strip()
    if legacy._is_price_show_all_request(query_text):
        family_payload = legacy._price_family_payload_from_context(entities, show_all=True)
        if family_payload:
            family_payload["entities_used"] = entities
            return family_payload
    if legacy._is_generic_uzi_price_request(query_text):
        return {
            "service_name": "УЗИ",
            "retail_prices": [],
            "doctors": [],
            "prepare": "",
            "show_prepare": False,
            "top_n_applied": top_limit,
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
            "note": "service_bundle_info: generic_uzi_clarify",
            "entities_used": entities,
        }
    try:
        retail_rows = await asyncio.to_thread(legacy.api_price.load_price_by_region, legacy.SAMARA_PRICE_REGION_ID)
    except Exception:
        retail_rows = []
    retail_rows = [p for p in retail_rows if isinstance(p, dict)]

    family_payload = legacy._build_price_family_payload(
        query_text,
        retail_rows,
        show_all=False,
        visible_limit=10,
    )
    if family_payload:
        family_payload["top_n_applied"] = top_limit
        family_payload["entities_used"] = entities
        return family_payload
    if entity_service_name and legacy._is_city_only_reply(query_text):
        query_service_name = None
    else:
        query_service_name = legacy.resolve_price_service_name_from_catalog(
            query_text,
            current_service_name=entity_service_name,
        ) or legacy._extract_price_service_from_query(query_text)
    service_name = legacy._select_effective_price_service_name(
        entity_service_name,
        query_service_name,
    )
    needle = legacy._normalise_input(service_name)

    out: dict[str, Any] = {
        "service_name": service_name,
        "retail_prices": [],
        "doctors": [],
        "prepare": "",
        "show_prepare": False,
        "top_n_applied": top_limit,
        "note": "service_bundle_info",
        "entities_used": {
            **entities,
            "service_name_effective": service_name,
        },
    }
    if not needle:
        out["note"] = "service_bundle_info: no service query"
        return out

    # 1) Retail price by city-level regionId (Самара = 3).
    retail_query = service_name
    retail_prefers_query_candidate = False
    if query_text and not legacy._is_city_only_reply(query_text):
        query_candidate = legacy._extract_price_service_from_query(query_text)
        retail_prefers_query_candidate = legacy._should_prefer_retail_query_candidate(
            query_candidate or "",
            service_name,
        )
        if retail_prefers_query_candidate:
            retail_query = query_candidate or service_name
    try:
        out["retail_prices"] = legacy._select_patient_price_rows(
            retail_rows,
            retail_query,
            limit=5,
        )
        out["retail_prices"] = legacy._annotate_price_rows_with_care_context(out["retail_prices"])
    except Exception:
        out["retail_prices"] = []
        out["note"] = "service_bundle_info: retail source unavailable"
    if retail_prefers_query_candidate and retail_query:
        out["service_name"] = retail_query

    compound_payload = legacy._build_compound_price_clarify_payload(
        query_text=query_text,
        entities=entities,
        primary_service_name=service_name,
        retail_rows=retail_rows,
        primary_retail_prices=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
    )
    if compound_payload:
        compound_payload["top_n_applied"] = top_limit
        compound_payload["entities_used"] = {
            **entities,
            "service_name_effective": service_name,
        }
        return compound_payload

    # 2) Top-N doctors by ord among doctors that have the matched service in doctor prices.
    top_retail = out["retail_prices"][0] if isinstance(out.get("retail_prices"), list) and out["retail_prices"] else {}
    target_homecode = legacy._normalise_input(
        str(top_retail.get("serviceHomecode") or top_retail.get("homecode") or "")
    )
    is_consult_query = legacy._is_consultation_service_query(service_name)
    preliminary_kind = legacy._classify_catalog_service_kind(
        service_name,
        query_text=query_text,
        retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
        has_exact_doctor_link=False,
        is_consult_query=is_consult_query,
    )
    query_norm = legacy._normalise_input(service_name)
    query_tokens = legacy._price_query_tokens(service_name)
    homecode_query = legacy._extract_homecode_query(service_name)
    matched_price_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
    exact_link_rows: list[tuple[int, int, int, int, dict[str, Any]]] = []
    samara_tokens: set[str] = set()
    by_id: dict[int, dict[str, Any]] = {}
    doctor_prices: list[dict[str, Any]] = []
    service_kind = preliminary_kind
    if preliminary_kind not in {"lab", "diagnostic_no_doctor"}:
        samara_tokens = await self._samara_region_tokens()
        doctors = await self._ensure_doctors_cache_loaded()
        for doc in doctors:
            if not isinstance(doc, dict):
                continue
            doc_id = legacy._as_int(doc.get("id"))
            if doc_id is None:
                continue
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            if legacy._has_explicit_non_samara_regions(raw_regions):
                continue
            if samara_tokens and raw_regions and not any(legacy._region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
            by_id[doc_id] = doc

        try:
            doctor_prices = await asyncio.to_thread(legacy.api_price.load_doctor_prices)
        except Exception:
            doctor_prices = []

        for row in doctor_prices:
            if not isinstance(row, dict):
                continue
            doctor_id = legacy._as_int(row.get("doctorId"))
            if doctor_id is None or doctor_id not in by_id:
                continue
            row_name_norm = legacy._normalise_input(str(row.get("serviceName") or row.get("name") or ""))
            row_homecode = legacy._normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
            score, matched = legacy._price_row_score(
                row,
                query=query_norm,
                tokens=query_tokens,
                homecode_query=homecode_query,
            )
            if not is_consult_query and target_homecode and row_homecode and target_homecode == row_homecode:
                score = max(score, 260)
                matched = max(matched, 1)
            if score <= 0:
                continue
            if not is_consult_query and not legacy._is_strong_doctor_price_match(
                query_norm=query_norm,
                query_tokens=query_tokens,
                row_name_norm=row_name_norm,
                matched_tokens=matched,
                target_homecode=target_homecode,
                row_homecode=row_homecode,
            ):
                continue
            cost = legacy._as_int(row.get("cost")) or 0
            item = (score, matched, -cost, doctor_id, row)
            matched_price_rows.append(item)
            if target_homecode and row_homecode and target_homecode == row_homecode:
                exact_link_rows.append(item)

        has_reliable_doctor_link = bool(exact_link_rows) or legacy._has_reliable_doctor_service_link(
            matched_price_rows,
            query_norm,
        )
        service_kind = legacy._classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=has_reliable_doctor_link,
            is_consult_query=is_consult_query,
        )
    if service_kind == "ambiguous":
        service_kind = await legacy._resolve_ambiguous_price_kind_with_llm(
            query_text,
            out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=bool(exact_link_rows) or legacy._has_reliable_doctor_service_link(
                matched_price_rows,
                query_norm,
            ),
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
        )
    if service_kind == "operator":
        return legacy._service_fallback(
            note="service_bundle_info ambiguous operator fallback",
            handoff_message=legacy.handoff_message("ambiguous_price_service"),
            entities=entities,
            reason="ambiguous_price_service",
            extra={
                "retail_prices": out.get("retail_prices") or [],
                "service_name": service_name,
            },
        )
    if service_kind == "family_query":
        family_payload = legacy._build_price_family_payload(
            query_text,
            retail_rows,
            show_all=False,
            visible_limit=10,
        )
        if family_payload:
            family_payload["entities_used"] = entities
            family_payload["top_n_applied"] = top_limit
            return family_payload
    out["service_kind"] = service_kind

    if service_kind in {"doctor_consult", "procedure_with_doctor"}:
        candidate_rows = (
            exact_link_rows
            if service_kind == "procedure_with_doctor" and exact_link_rows
            else matched_price_rows
        )
        allow_soft_substring_fallback = service_kind == "doctor_consult" and len(query_tokens) <= 1
        if not candidate_rows and query_norm and allow_soft_substring_fallback:
            for row in doctor_prices:
                if not isinstance(row, dict):
                    continue
                doctor_id = legacy._as_int(row.get("doctorId"))
                if doctor_id is None or doctor_id not in by_id:
                    continue
                service_row_name = legacy._normalise_input(str(row.get("serviceName") or ""))
                if query_norm and query_norm in service_row_name:
                    cost = legacy._as_int(row.get("cost")) or 0
                    candidate_rows.append((1, 1, -cost, doctor_id, row))

        candidate_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        best_row_by_doctor: dict[int, dict[str, Any]] = {}
        for _, _, _, doctor_id, row in candidate_rows:
            if doctor_id not in best_row_by_doctor:
                best_row_by_doctor[doctor_id] = row

        doctor_cards = sorted(
            [by_id[doctor_id] for doctor_id in best_row_by_doctor if doctor_id in by_id],
            key=legacy._doctor_sort_key,
        )[:top_limit]

        out_doctors: list[dict[str, Any]] = []
        query_specialty = legacy._extract_specialty_from_text(query_text) or legacy._extract_specialty_from_text(service_name)
        for doc in doctor_cards:
            doctor_id = legacy._as_int(doc.get("id"))
            if doctor_id is None:
                continue
            if service_kind == "doctor_consult" and query_specialty and not legacy._doctor_matches_primary_specialty(doc, query_specialty):
                continue
            price_row = best_row_by_doctor.get(doctor_id, {})
            availability = await self._doctor_availability_snapshot(
                str(doc.get("fio") or ""),
                samara_tokens=samara_tokens,
            )
            out_doctors.append(
                {
                    "id": doctor_id,
                    "fio": str(doc.get("fio") or "").strip(),
                    "ord": legacy._as_int(doc.get("ord")),
                    "specialization": legacy._compact_specialization(
                        legacy._pick_display_specialization(
                            doc,
                            preferred_specialty=query_specialty,
                            preferred_service=service_name,
                        )
                    ),
                    "specialty_label": legacy._specialty_label_for_doctor(
                        doc,
                        preferred_specialty=query_specialty,
                    ),
                    "regions": [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()],
                    "service_price": legacy._as_int(price_row.get("cost")),
                    "available": bool(availability.get("available")),
                    "nearest_slot": str(availability.get("nearest_slot") or ""),
                    "regions_with_slots": list(availability.get("regions_with_slots") or []),
                    "availability_note": str(availability.get("note") or ""),
                }
            )
        out["doctors"] = out_doctors
    else:
        out["doctors"] = []
        out["note"] = (
            f"{out['note']}; " if str(out.get("note") or "").strip() else ""
        ) + f"service_bundle_info: {service_kind or 'no_doctors'}"

    # 3) Preparation guidance by service/test name.
    # В PRICE показываем подготовку только по явному запросу пациента.
    # Иначе блок шумит и мешает основной задаче (цена/врач/расписание).
    show_prepare = legacy._is_prepare_requested_in_price_query(query_text)
    out["show_prepare"] = show_prepare
    if show_prepare and not is_consult_query:
        prepare_payload = await self.test_prepare(service_name, {"service_name": service_name})
        if isinstance(prepare_payload, dict) and not prepare_payload.get("handoff_required"):
            out["prepare"] = str(prepare_payload.get("prepare") or "").strip()

    return out
