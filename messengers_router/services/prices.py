"""Модуль домена цен.

Содержит pilot-миграцию price-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent_logic_2.nayka_api import api_price

from ..policies import handoff_message
from ._addresses_helpers import _extract_homecode_query
from ._common import (
    DOCTORS_TOP_N,
    _as_int,
    _coerce_top_n,
    _get_first_present,
    _normalise_input,
    _service_fallback,
)
from ._doctors_helpers import (
    _classify_catalog_service_kind,
    _compact_specialization,
    _doctor_matches_primary_specialty,
    _doctor_sort_key,
    _extract_specialty_from_text,
    _has_reliable_doctor_service_link,
    _is_clean_consultation_row_name,
    _is_consultation_service_query,
    _pick_display_specialization,
    _select_effective_price_service_name,
    _should_prefer_retail_query_candidate,
    _specialty_label_for_doctor,
)
from ._prepare import _is_prepare_requested_in_price_query
from ._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _DOCTOR_PRICE_HINT_RE,
    _PRICE_CONSULT_HINT_RE,
    _PRICE_REQUEST_RE,
    _annotate_price_rows_with_care_context,
    _apply_service_synonyms,
    _build_compound_price_clarify_payload,
    _build_multi_price_payload,
    _build_price_family_payload,
    _extract_price_service_from_query,
    _is_city_only_reply,
    _is_generic_uzi_price_request,
    _is_price_show_all_request,
    _is_strong_doctor_price_match,
    _price_family_payload_from_context,
    _price_query_tokens,
    _price_row_score,
    _rank_price_rows,
    _resolve_ambiguous_price_kind_with_llm,
    _select_patient_price_rows,
    resolve_price_service_name_from_catalog,
    restrict_rows_to_family_letter,
)
from ._regions import (
    _has_explicit_non_samara_regions,
    _region_matches_samara_tokens,
)

if TYPE_CHECKING:
    from .core import Services


async def price_info(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает информацию о ценах на услуги или приём у врача.

    :param self: экземпляр сервисного слоя
    :param query: текст запроса пользователя
    :param entities: извлечённые NLU-сущности
    :return: словарь с ценовыми строками или сообщением об ошибке
    """

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

    doctor_id = _as_int(entities.get("doctor_id"))
    doctor_name = _get_first_present(entities, ["doctor_name", "doctor", "fio", "last_name", "doctor_last_name"]) or ""
    resolved_doctor_fio: str | None = None
    if not doctor_id and doctor_name:
        doctor_id, resolved_doctor_fio = await self._resolve_doctor_id_from_name(doctor_name)
    if not doctor_id and query and _DOCTOR_PRICE_HINT_RE.search(str(query or "")):
        # Fallback для фраз вида "сколько стоит ... у Белохвостиковой":
        # извлекаем врача из полного текста запроса, даже если classifier не выделил doctor_name.
        doctor_id, q_resolved_fio = await self._resolve_doctor_id_from_name(str(query))
        if doctor_id and q_resolved_fio:
            resolved_doctor_fio = q_resolved_fio
            doctor_name = q_resolved_fio
    # Нормализуем пациентские синонимы к терминам каталога (BUG-2026-06-23-01:
    # «забор крови»→«взятие крови», «электромиография»→«ЭМГ») до матчинга услуги.
    entity_service_name = _apply_service_synonyms(_get_first_present(entities, ["service_name", "test_name"]) or "")
    query_text = _apply_service_synonyms(str(query or "").strip())
    has_price_request = bool(query_text and _PRICE_REQUEST_RE.search(query_text))
    if _is_price_show_all_request(query_text):
        family_payload = _price_family_payload_from_context(entities, show_all=True)
        if family_payload:
            family_payload["entities_used"] = entities
            return family_payload
    if _is_generic_uzi_price_request(query_text):
        return {
            "prices": [],
            "clarify_text": (
                "Введите конкретное название процедуры, например: "
                "стоимость УЗИ брюшной полости или цена УЗИ молочной железы."
            ),
            "note": "price_info: generic_uzi_clarify",
            "entities_used": entities,
        }
    doctor_query_specialty = _extract_specialty_from_text(query_text) if (doctor_id and has_price_request) else ""

    if doctor_id:
        # Для doctor-specific PRICE не приземляемся в городский retail-catalog:
        # иначе вопрос "у Иванова" может маппиться в случайную услугу по всему прайсу.
        query_service_name = _extract_price_service_from_query(query_text) if query_text else None
        if not query_service_name:
            query_service_name = entity_service_name
        # "сколько стоит прием у <врач>" без специальности:
        # принудительно удерживаем консультационный контекст вместо stale service_name.
        if has_price_request and _PRICE_CONSULT_HINT_RE.search(query_text) and not doctor_query_specialty:
            if not _is_consultation_service_query(str(query_service_name or "")):
                query_service_name = "консультация"
        service_name = query_service_name or ""
        # Если это doctor-specific price без явной услуги, оставляем query как fallback
        # (для редких строк doctor_price, где нет стандартных маркеров).
        if not service_name and has_price_request and _DOCTOR_PRICE_HINT_RE.search(query_text):
            service_name = query_text
    else:
        if entity_service_name and _is_city_only_reply(query_text):
            query_service_name = None
        else:
            query_service_name = resolve_price_service_name_from_catalog(
                query_text,
                current_service_name=entity_service_name,
            ) or _extract_price_service_from_query(query_text)
        service_name = _select_effective_price_service_name(
            entity_service_name,
            query_service_name,
        )
    needle = _normalise_input(service_name)

    if doctor_id:
        # doctor prices: branch-level regionId из /doctorServicePricesByRegion cache
        try:
            prices = await _load_with_retry(api_price.load_doctor_prices)
        except Exception:
            return _service_fallback(
                note="price_info source unavailable",
                handoff_message=handoff_message("service_error_prices"),
                entities=entities,
                extra={"prices": []},
            )
        doc_prices = [p for p in prices if _as_int(p.get("doctorId")) == doctor_id]
        if needle:
            ranked = _rank_price_rows(doc_prices, service_name, limit=10)
            if (
                not ranked
                and has_price_request
                and _PRICE_CONSULT_HINT_RE.search(query_text)
            ):
                ranked = [
                    p for p in doc_prices
                    if _is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                ]
            doc_prices = ranked
        if not needle:
            if has_price_request and _PRICE_CONSULT_HINT_RE.search(query_text):
                consult_rows = [
                    p for p in doc_prices
                    if _is_clean_consultation_row_name(str(p.get("serviceName") or p.get("name") or ""))
                ]
                if consult_rows:
                    doc_prices = consult_rows
            doc_prices = sorted(
                [p for p in doc_prices if isinstance(p, dict)],
                key=lambda p: (_normalise_input(str(p.get("serviceName") or "")), _as_int(p.get("cost")) or 0),
            )
        doc_prices = _annotate_price_rows_with_care_context(doc_prices[:10])
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
        price_rows = await _load_with_retry(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
    except Exception:
        return _service_fallback(
            note="price_info source unavailable",
            handoff_message=handoff_message("service_error_prices"),
            entities=entities,
            extra={"prices": []},
        )
    retail_rows_clean = [p for p in price_rows if isinstance(p, dict)]
    # Мульти-услуговый запрос («ВИЧ, гепатит, ОАК»): активируется только если
    # ≥2 фрагментов приземлились на реальные услуги каталога. Иначе — fallback
    # на существующий single-service путь, старые кейсы остаются без изменений.
    multi_payload = _build_multi_price_payload(query_text, retail_rows_clean)
    if multi_payload is not None:
        multi_payload["entities_used"] = {
            **entities,
            "service_name_effective": str(multi_payload.get("service_name") or "").strip(),
        }
        return multi_payload
    family_payload = _build_price_family_payload(
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
    if query_text and not _is_city_only_reply(query_text):
        query_candidate = _extract_price_service_from_query(query_text)
        if _should_prefer_retail_query_candidate(query_candidate or "", service_name):
            retail_query = query_candidate or service_name
    # Гард различающего токена семейства для МАЛЫХ семейств (гепатит А/D/E —
    # <3 вариантов, family-mode их не берёт): если запрос несёт букву, ранжируем
    # ТОЛЬКО по строкам этой буквы, иначе всплывёт чужой гепатит (В за запрос про А).
    # Для запросов без дискриминатора список не меняется.
    family_scoped_rows = restrict_rows_to_family_letter(
        query_text, [p for p in price_rows if isinstance(p, dict)]
    )
    matches = _select_patient_price_rows(
        family_scoped_rows,
        retail_query,
        limit=10,
    )
    # BUG-H sibling: query-кандидат из текста мог нести framing-шум («анализ ттг»),
    # который даёт ПУСТУЮ выборку (AND-семантика: нет строки и с «анализ», и с «ттг»),
    # хотя резолвленная услуга (service_name) находится. Откатываемся на неё, чтобы
    # не терять цену на «сколько стоит анализ на <X>». Срабатывает только на пустом
    # результате — рабочие кейсы не затрагиваются.
    if not matches and retail_query != service_name:
        matches = _select_patient_price_rows(
            family_scoped_rows,
            service_name,
            limit=10,
        )
    matches = _annotate_price_rows_with_care_context(matches)
    return {
        "prices": matches,
        "service_kind": "lab" if _classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=matches,
            has_exact_doctor_link=False,
            is_consult_query=False,
        ) == "lab" else "",
        "note": f"price_info: priceByRegion({SAMARA_PRICE_REGION_ID})",
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
    top_limit = _coerce_top_n(top_n, default=DOCTORS_TOP_N)
    # Нормализуем пациентские синонимы к терминам каталога (BUG-2026-06-23-01:
    # «забор крови»→«взятие крови», «электромиография»→«ЭМГ») до матчинга услуги.
    entity_service_name = _apply_service_synonyms(_get_first_present(entities, ["service_name", "test_name"]) or "")
    query_text = _apply_service_synonyms(str(query or "").strip())
    if _is_price_show_all_request(query_text):
        family_payload = _price_family_payload_from_context(entities, show_all=True)
        if family_payload:
            family_payload["entities_used"] = entities
            return family_payload
    if _is_generic_uzi_price_request(query_text):
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
        retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
    except Exception:
        retail_rows = []
    retail_rows = [p for p in retail_rows if isinstance(p, dict)]

    family_payload = _build_price_family_payload(
        query_text,
        retail_rows,
        show_all=False,
        visible_limit=10,
    )
    if family_payload:
        family_payload["top_n_applied"] = top_limit
        family_payload["entities_used"] = entities
        return family_payload
    if entity_service_name and _is_city_only_reply(query_text):
        query_service_name = None
    else:
        query_service_name = resolve_price_service_name_from_catalog(
            query_text,
            current_service_name=entity_service_name,
        ) or _extract_price_service_from_query(query_text)
    service_name = _select_effective_price_service_name(
        entity_service_name,
        query_service_name,
    )
    needle = _normalise_input(service_name)

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
    if query_text and not _is_city_only_reply(query_text):
        query_candidate = _extract_price_service_from_query(query_text)
        retail_prefers_query_candidate = _should_prefer_retail_query_candidate(
            query_candidate or "",
            service_name,
        )
        if retail_prefers_query_candidate:
            retail_query = query_candidate or service_name
    try:
        out["retail_prices"] = _select_patient_price_rows(
            retail_rows,
            retail_query,
            limit=5,
        )
        # BUG-H sibling: framing-шум в query-кандидате («анализ ттг») даёт ПУСТУЮ
        # выборку, хотя резолвленная услуга находится. Откат на неё (только на пустом
        # результате), и не подменяем service_name на шумный кандидат.
        if not out["retail_prices"] and retail_query != service_name:
            out["retail_prices"] = _select_patient_price_rows(retail_rows, service_name, limit=5)
            if out["retail_prices"]:
                retail_prefers_query_candidate = False
                retail_query = service_name
        out["retail_prices"] = _annotate_price_rows_with_care_context(out["retail_prices"])
    except Exception:
        out["retail_prices"] = []
        out["note"] = "service_bundle_info: retail source unavailable"
    if retail_prefers_query_candidate and retail_query:
        out["service_name"] = retail_query

    compound_payload = _build_compound_price_clarify_payload(
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
    target_homecode = _normalise_input(
        str(top_retail.get("serviceHomecode") or top_retail.get("homecode") or "")
    )
    is_consult_query = _is_consultation_service_query(service_name)
    preliminary_kind = _classify_catalog_service_kind(
        service_name,
        query_text=query_text,
        retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
        has_exact_doctor_link=False,
        is_consult_query=is_consult_query,
    )
    query_norm = _normalise_input(service_name)
    query_tokens = _price_query_tokens(service_name)
    homecode_query = _extract_homecode_query(service_name)
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
            doc_id = _as_int(doc.get("id"))
            if doc_id is None:
                continue
            raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
            if _has_explicit_non_samara_regions(raw_regions):
                continue
            if samara_tokens and raw_regions and not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
            by_id[doc_id] = doc

        try:
            doctor_prices = await asyncio.to_thread(api_price.load_doctor_prices)
        except Exception:
            doctor_prices = []

        for row in doctor_prices:
            if not isinstance(row, dict):
                continue
            # Пропускаем «нулевые» doctor_prices — это, как правило,
            # незаполненные админом записи (data drafts). Пациент
            # видел в выдаче «Саушкина (Лор) — 0 руб», «Джовмардов
            # (Лор) — 0 руб» — заведомо неверная цена для приёма
            # ЛОР-врача (розница 2 000 руб). Если для специальности
            # есть retail-цена, она и так показана сверху ответа,
            # а конкретный врач без заполненной цены пусть всплывает
            # через doctors_info, но не клеймится «0 руб».
            row_cost = _as_int(row.get("cost")) or 0
            if row_cost <= 0:
                continue
            doctor_id = _as_int(row.get("doctorId"))
            if doctor_id is None or doctor_id not in by_id:
                continue
            row_name_norm = _normalise_input(str(row.get("serviceName") or row.get("name") or ""))
            row_homecode = _normalise_input(str(row.get("serviceHomecode") or row.get("homecode") or ""))
            score, matched = _price_row_score(
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
            if not is_consult_query and not _is_strong_doctor_price_match(
                query_norm=query_norm,
                query_tokens=query_tokens,
                row_name_norm=row_name_norm,
                matched_tokens=matched,
                target_homecode=target_homecode,
                row_homecode=row_homecode,
            ):
                continue
            item = (score, matched, -row_cost, doctor_id, row)
            matched_price_rows.append(item)
            if target_homecode and row_homecode and target_homecode == row_homecode:
                exact_link_rows.append(item)

        has_reliable_doctor_link = bool(exact_link_rows) or _has_reliable_doctor_service_link(
            matched_price_rows,
            query_norm,
        )
        service_kind = _classify_catalog_service_kind(
            service_name,
            query_text=query_text,
            retail_rows=out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=has_reliable_doctor_link,
            is_consult_query=is_consult_query,
        )
    if service_kind == "ambiguous":
        service_kind = await _resolve_ambiguous_price_kind_with_llm(
            query_text,
            out["retail_prices"] if isinstance(out.get("retail_prices"), list) else [],
            has_exact_doctor_link=bool(exact_link_rows) or _has_reliable_doctor_service_link(
                matched_price_rows,
                query_norm,
            ),
            runtime_llm_mode=str(entities.get("__runtime_llm_mode") or ""),
        )
    if service_kind == "operator":
        return _service_fallback(
            note="service_bundle_info ambiguous operator fallback",
            handoff_message=handoff_message("ambiguous_price_service"),
            entities=entities,
            reason="ambiguous_price_service",
            extra={
                "retail_prices": out.get("retail_prices") or [],
                "service_name": service_name,
            },
        )
    if service_kind == "family_query":
        family_payload = _build_price_family_payload(
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
                doctor_id = _as_int(row.get("doctorId"))
                if doctor_id is None or doctor_id not in by_id:
                    continue
                service_row_name = _normalise_input(str(row.get("serviceName") or ""))
                if query_norm and query_norm in service_row_name:
                    cost = _as_int(row.get("cost")) or 0
                    candidate_rows.append((1, 1, -cost, doctor_id, row))

        candidate_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        best_row_by_doctor: dict[int, dict[str, Any]] = {}
        for _, _, _, doctor_id, row in candidate_rows:
            if doctor_id not in best_row_by_doctor:
                best_row_by_doctor[doctor_id] = row

        doctor_cards = sorted(
            [by_id[doctor_id] for doctor_id in best_row_by_doctor if doctor_id in by_id],
            key=_doctor_sort_key,
        )[:top_limit]

        # Параллелизируем availability_snapshot для всех кандидатов.
        # Раньше последовательный цикл `for doc ...: await snapshot(...)`
        # подвешивал PRICE-консультацию специалиста на 240+ секунд при
        # медленном Nayka /doctorSchedule (один surname Трубин у нас
        # отдавался 224с). Теперь все 4-8 кандидатов едут параллельно,
        # верхняя граница на availability-фазу ≈ 30с (per-doctor timeout
        # внутри `_schedule_by_specialty` уже 60с, для чисто
        # availability-проверки ставим 25с — мы не показываем дни/слоты,
        # а только bool «есть/нет окон»).
        # Жалоба заказчика 2026-05-05: «сколько стоит приём кардиолога»
        # уходил в 4-минутный таймаут (eval P3 / CRIT_PRICE_CONSULT_UROLOGIST_001).
        query_specialty = _extract_specialty_from_text(query_text) or _extract_specialty_from_text(service_name)
        _AVAILABILITY_TIMEOUT_S = 25.0
        _AVAILABILITY_CONCURRENCY = 8
        _availability_sem = asyncio.Semaphore(_AVAILABILITY_CONCURRENCY)
        _empty_availability = {
            "available": False,
            "nearest_slot": "",
            "regions_with_slots": [],
            "note": "availability_timeout",
        }

        async def _fetch_doctor_availability(doc: dict[str, Any]) -> dict[str, Any] | None:
            """Возвращает payload для одного врача либо None, если врача надо
            пропустить (specialty mismatch / no doctor_id)."""
            doctor_id = _as_int(doc.get("id"))
            if doctor_id is None:
                return None
            if service_kind == "doctor_consult" and query_specialty and not _doctor_matches_primary_specialty(doc, query_specialty):
                return None
            async with _availability_sem:
                try:
                    availability = await asyncio.wait_for(
                        self._doctor_availability_snapshot(
                            str(doc.get("fio") or ""),
                            samara_tokens=samara_tokens,
                        ),
                        timeout=_AVAILABILITY_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    availability = dict(_empty_availability)
                except Exception:
                    availability = dict(_empty_availability, note="availability_error")
            price_row = best_row_by_doctor.get(doctor_id, {})
            return {
                "id": doctor_id,
                "fio": str(doc.get("fio") or "").strip(),
                "ord": _as_int(doc.get("ord")),
                "specialization": _compact_specialization(
                    _pick_display_specialization(
                        doc,
                        preferred_specialty=query_specialty,
                        preferred_service=service_name,
                    )
                ),
                "specialty_label": _specialty_label_for_doctor(
                    doc,
                    preferred_specialty=query_specialty,
                ),
                "regions": [str(x).strip() for x in (doc.get("regions") or []) if str(x).strip()],
                "service_price": _as_int(price_row.get("cost")),
                "available": bool(availability.get("available")),
                "nearest_slot": str(availability.get("nearest_slot") or ""),
                "regions_with_slots": list(availability.get("regions_with_slots") or []),
                "availability_note": str(availability.get("note") or ""),
            }

        availability_results = await asyncio.gather(
            *(_fetch_doctor_availability(doc) for doc in doctor_cards)
        )
        out_doctors = [r for r in availability_results if r is not None]
        out["doctors"] = out_doctors
    else:
        out["doctors"] = []
        out["note"] = (
            f"{out['note']}; " if str(out.get("note") or "").strip() else ""
        ) + f"service_bundle_info: {service_kind or 'no_doctors'}"

    # 3) Preparation guidance by service/test name.
    # В PRICE показываем подготовку только по явному запросу пациента.
    # Иначе блок шумит и мешает основной задаче (цена/врач/расписание).
    show_prepare = _is_prepare_requested_in_price_query(query_text)
    out["show_prepare"] = show_prepare
    if show_prepare and not is_consult_query:
        prepare_payload = await self.test_prepare(service_name, {"service_name": service_name})
        if isinstance(prepare_payload, dict) and not prepare_payload.get("handoff_required"):
            out["prepare"] = str(prepare_payload.get("prepare") or "").strip()

    return out
