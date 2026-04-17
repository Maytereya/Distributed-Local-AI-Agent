"""Модуль домена цен.

Содержит pilot-миграцию price-related методов из ``services_legacy``.
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
                handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
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
            handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
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
