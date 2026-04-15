"""Модуль домена врачей.

Содержит pilot-миграцию doctor-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..services_legacy import Services


def _legacy_module():
    """Лениво импортирует legacy-модуль, чтобы не создать цикл импортов.

    :return: модуль ``messengers_router.services_legacy``
    """

    from .. import services_legacy as legacy

    return legacy


async def match_catalog_doctor(self: "Services", raw_text_or_name: str) -> dict[str, Any]:
    """Матчит врача по кэшу каталога врачей.

    :param self: экземпляр сервисного слоя
    :param raw_text_or_name: исходный текст пользователя или фамилия врача
    :return: словарь со статусом матчинга и каноническим именем
    """

    legacy = _legacy_module()
    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return {"status": "miss", "query": "", "canonical": ""}

    queries = legacy._doctor_catalog_query_candidates(raw_text_or_name)
    if not queries:
        return {"status": "miss", "query": "", "canonical": ""}

    for query in queries:
        exact = legacy.resolve_schedule_surname(query, doctors)
        if exact:
            return {
                "status": "exact",
                "query": query,
                "canonical": str(exact).strip(),
            }

    surname_map: dict[str, str] = {}
    for doc in doctors:
        if not isinstance(doc, dict):
            continue
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            continue
        surname = str(fio.split()[0] or "").strip()
        norm = legacy._normalise_catalog_text(surname)
        if norm and norm not in surname_map:
            surname_map[norm] = surname
    surname_keys = list(surname_map.keys())
    if not surname_keys:
        return {"status": "miss", "query": "", "canonical": ""}

    for query in queries:
        norm = legacy._normalise_catalog_text(query)
        if len(norm) < 4:
            continue
        hit = legacy.get_close_matches(norm, surname_keys, n=1, cutoff=0.84)
        if not hit:
            continue
        canonical = surname_map.get(hit[0], "").strip()
        if canonical and legacy._normalise_catalog_text(canonical) != norm:
            return {
                "status": "fuzzy",
                "query": query,
                "canonical": canonical,
                "matched_key": hit[0],
            }
    return {"status": "miss", "query": queries[0], "canonical": ""}


async def _schedule_by_specialty(
    self: "Services",
    specialty: str,
    entities: dict[str, Any],
    *,
    nearest_only: bool,
    query_text: str = "",
) -> tuple[list[dict[str, Any]], str | None]:
    """Ищет расписание по специальности.

    :param self: экземпляр сервисного слоя
    :param specialty: каноническая специальность
    :param entities: текущие сущности диалога
    :param nearest_only: вернуть только ближайшего врача
    :param query_text: исходный текст пользователя
    :return: кортеж ``(список расписаний, причина пустого результата)``
    """

    legacy = _legacy_module()
    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return [], None
    spec = legacy._normalise_input(specialty)
    if not spec:
        return [], None

    samara_tokens = await self._samara_region_tokens()
    role_query = legacy._is_role_specialty_query(query_text or specialty, spec)
    role_levels = {
        id(d): legacy._doctor_role_specialty_match_level(d, spec)
        for d in doctors
    } if role_query else {}
    candidates = sorted(
        [
            d for d in doctors
            if (
                (role_levels.get(id(d), 0) > 0)
                if role_query
                else legacy._doctor_matches_specialty(d, spec, query_text or specialty)
            )
            and not legacy._has_explicit_non_samara_regions([str(x) for x in (d.get("regions") or []) if str(x).strip()])
            and (
                not samara_tokens
                or any(
                    legacy._region_matches_samara_tokens(str(x), samara_tokens)
                    for x in (d.get("regions") or [])
                    if str(x).strip()
                )
            )
        ],
        key=(
            (lambda d: (
                -role_levels.get(id(d), 0),
                legacy._specialty_priority_rank(d, spec),
                *legacy._doctor_sort_key(d),
            ))
            if role_query
            else legacy._doctor_sort_key
        ),
    )[:8]
    if not candidates:
        return [], None

    out_rows: list[dict[str, Any]] = []
    matched_but_without_slots = False
    for doc in candidates:
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            continue
        surname = fio.split()[0]
        try:
            data = await self._get_schedule_payload_cached(surname)
        except Exception:
            continue
        if legacy._is_schedule_no_slots_text(data):
            matched_but_without_slots = True
            continue
        if not isinstance(data, list) or not data:
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            row_fio = str(row.get("fio") or "").strip()
            if row_fio and legacy._normalise_input(row_fio) != legacy._normalise_input(fio):
                continue
            item = dict(row)
            display_spec = legacy._pick_display_specialization(
                doc,
                preferred_specialty=spec,
            )
            item["specialization"] = legacy._compact_specialization(display_spec)
            slots = legacy._iter_slot_datetimes(item.get("schedule") or {})
            if slots:
                item["_nearest_slot"] = min(slots)
            out_rows.append(item)
            break

    if not out_rows:
        if matched_but_without_slots:
            return [], "no_free_slots_2_weeks"
        return [], None
    with_slots = [x for x in out_rows if isinstance(x.get("_nearest_slot"), legacy.datetime)]
    if with_slots:
        with_slots.sort(key=lambda x: x["_nearest_slot"])
        chosen = with_slots[:1] if nearest_only else with_slots[:3]
    else:
        chosen = out_rows[:1] if nearest_only else out_rows[:3]

    for item in chosen:
        item.pop("_nearest_slot", None)
    return chosen, None


async def _doctor_availability_snapshot(
    self: "Services",
    fio: str,
    *,
    samara_tokens: set[str],
) -> dict[str, Any]:
    """Проверяет наличие слотов у врача в расписании.

    :param self: экземпляр сервисного слоя
    :param fio: ФИО врача
    :param samara_tokens: набор самарских токенов филиалов
    :return: словарь со статусом доступности и ближайшим слотом
    """

    legacy = _legacy_module()
    fio_clean = str(fio or "").strip()
    surname = fio_clean.split()[0] if fio_clean else ""
    if not surname:
        return {
            "available": False,
            "nearest_slot": "",
            "regions_with_slots": [],
            "note": "availability_missing_surname",
        }

    try:
        data = await self._get_schedule_payload_cached(surname)
    except Exception:
        return {
            "available": False,
            "nearest_slot": "",
            "regions_with_slots": [],
            "note": "availability_source_unavailable",
        }

    if not isinstance(data, list) or not data:
        return {
            "available": False,
            "nearest_slot": "",
            "regions_with_slots": [],
            "note": "availability_empty",
        }

    target_norm = legacy._normalise_input(fio_clean)
    chosen: dict[str, Any] | None = None
    for row in data:
        if not isinstance(row, dict):
            continue
        row_fio = str(row.get("fio") or "").strip()
        if not row_fio:
            continue
        row_norm = legacy._normalise_input(row_fio)
        if target_norm and row_norm == target_norm:
            chosen = row
            break
        if legacy._doctor_matches_fio(row_fio, fio_clean, resolved_surname=surname):
            chosen = row
            break
    if chosen is None:
        chosen = next((row for row in data if isinstance(row, dict)), None)
    if not isinstance(chosen, dict):
        return {
            "available": False,
            "nearest_slot": "",
            "regions_with_slots": [],
            "note": "availability_unmatched",
        }

    schedule_raw = chosen.get("schedule")
    schedule: dict[str, Any] = {}
    if isinstance(schedule_raw, dict):
        if samara_tokens:
            for region_name, days in schedule_raw.items():
                region = str(region_name or "").strip()
                if not region:
                    continue
                if legacy._region_matches_samara_tokens(region, samara_tokens):
                    schedule[region] = days
        else:
            schedule = {str(k): v for k, v in schedule_raw.items()}

    slots = legacy._iter_slot_datetimes(schedule)
    nearest_slot = min(slots).isoformat(timespec="minutes") if slots else ""
    return {
        "available": bool(slots),
        "nearest_slot": nearest_slot,
        "regions_with_slots": legacy._schedule_regions_with_free_slots(schedule),
        "note": "availability_checked",
    }


async def resolve_doctor_name(self: "Services", raw_text_or_name: str) -> str | None:
    """Валидирует фамилию/ФИО врача по актуальному doctor-cache.

    :param self: экземпляр сервисного слоя
    :param raw_text_or_name: исходный текст пользователя
    :return: каноническая фамилия врача или ``None``
    """

    legacy = _legacy_module()
    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return None
    value = str(raw_text_or_name or "").strip()
    if not value:
        return None
    resolved = legacy.resolve_schedule_surname(value, doctors)
    if resolved:
        return resolved

    candidate = legacy.extract_doctor_name_candidate(value, prefer_schedule=True)
    if candidate and legacy._normalise_input(candidate) != legacy._normalise_input(value):
        return legacy.resolve_schedule_surname(candidate, doctors)
    return None


async def _resolve_doctor_id_from_name(
    self: "Services",
    raw_text_or_name: str,
) -> tuple[int | None, str | None]:
    """Находит ``doctor_id`` и каноническое ФИО по имени врача.

    :param self: экземпляр сервисного слоя
    :param raw_text_or_name: исходное имя/фамилия врача
    :return: кортеж ``(doctor_id, canonical_fio)``
    """

    legacy = _legacy_module()
    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return None, None

    raw = str(raw_text_or_name or "").strip()
    if not raw:
        return None, None

    resolved_surname = legacy.resolve_schedule_surname(raw, doctors)
    samara_tokens = await self._samara_region_tokens()
    matched: list[dict[str, Any]] = []
    for doc in doctors:
        if not isinstance(doc, dict):
            continue
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            continue
        raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
        if legacy._has_explicit_non_samara_regions(raw_regions):
            continue
        if samara_tokens and raw_regions and not any(legacy._region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
            continue
        if legacy._doctor_matches_fio(fio, raw, resolved_surname):
            matched.append(doc)

    if not matched:
        return None, None
    matched = sorted(matched, key=legacy._doctor_sort_key)
    first = matched[0]
    return legacy._as_int(first.get("id")), str(first.get("fio") or "").strip() or None


async def doctors_info(
    self: "Services",
    query: str,
    entities: dict[str, Any],
    output_max: int | None = None,
) -> dict[str, Any]:
    """Возвращает список врачей из doctor-cache без realtime API.

    :param self: экземпляр сервисного слоя
    :param query: исходный запрос пользователя
    :param entities: сущности роутера
    :param output_max: верхний предел выдачи
    :return: payload с карточками врачей или fallback-ответом
    """

    legacy = _legacy_module()
    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return legacy._service_fallback(
            note="doctors_info source unavailable",
            handoff_message="Сейчас не удалось получить список врачей автоматически. Соединяю с оператором.",
            entities=entities,
            extra={"doctors": []},
        )

    q = legacy._normalise_input(query)
    doctor_raw = legacy._get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
    fio_q = legacy._normalise_input(doctor_raw)
    spec_q = legacy._normalise_input(legacy._get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
    if not spec_q:
        spec_q = legacy._extract_specialty_from_text(query)
    service_q = legacy._normalise_input(legacy._get_first_present(entities, ["service_name", "test_name"]) or "")
    if not service_q:
        if legacy._SERVICE_QUERY_SIGNAL_RE.search(legacy._normalise_input(query)):
            extracted_service = legacy.extract_service_phrase(query)
            if extracted_service:
                service_q = legacy._normalise_input(extracted_service)
    region_q = legacy._normalise_input(legacy._get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")
    resolved_surname = legacy.resolve_schedule_surname(doctor_raw, doctors) if doctor_raw else None

    query_candidate = legacy.extract_doctor_name_candidate(query, prefer_schedule=True) if query else None
    query_resolved = legacy.resolve_schedule_surname(query_candidate, doctors) if query_candidate else None
    if not resolved_surname:
        resolved_surname = query_resolved

    if spec_q and not query_resolved:
        fio_q = ""
        resolved_surname = None

    samara_tokens = await self._samara_region_tokens()
    role_query = bool(spec_q and legacy._is_role_specialty_query(query, spec_q))
    role_levels = {
        id(d): legacy._doctor_role_specialty_match_level(d, spec_q)
        for d in doctors
    } if role_query else {}

    keyword = ""
    if fio_q:
        keyword = fio_q
    elif spec_q:
        keyword = spec_q
    elif region_q:
        keyword = region_q
    else:
        keyword = q

    keyword = keyword.strip()

    def match_doc(doc: dict[str, Any]) -> bool:
        fio = legacy._normalise_input(str(doc.get("fio", "")))
        spec_text = legacy._normalise_input(str(doc.get("specialization", "")))
        raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
        regions = " ".join([legacy._normalise_input(x) for x in raw_regions])
        units = " ".join([legacy._normalise_input(str(x)) for x in (doc.get("units") or [])])

        hay = " | ".join([fio, spec_text, regions, units])
        if legacy._has_explicit_non_samara_regions(raw_regions):
            return False
        if samara_tokens:
            if not any(legacy._region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                return False
        if fio_q:
            if not legacy._doctor_matches_fio(fio, fio_q, resolved_surname):
                return False
        if spec_q:
            if role_query:
                if role_levels.get(id(doc), 0) <= 0:
                    return False
            elif not legacy._doctor_matches_specialty(doc, spec_q, query):
                return False
        if service_q:
            if not legacy._doctor_matches_service(doc, service_q):
                return False
        if region_q and region_q not in hay:
            return False

        if not (fio_q or spec_q or region_q or service_q):
            if len(keyword) < 3:
                return False
            return keyword in hay

        return True

    filtered = [d for d in doctors if match_doc(d)]
    filtered = legacy._dedupe_doctors_by_fio(filtered)
    if role_query:
        filtered = sorted(
            filtered,
            key=lambda d: (
                -role_levels.get(id(d), 0),
                legacy._specialty_priority_rank(d, spec_q),
                *legacy._doctor_sort_key(d),
            ),
        )
    else:
        filtered = sorted(filtered, key=legacy._doctor_sort_key)

    limit = legacy._coerce_top_n(output_max, default=legacy.DOCTORS_TOP_N)
    if resolved_surname:
        limit = min(limit, 3)

    filtered = filtered[:limit]
    compact: list[dict[str, Any]] = []
    for d in filtered:
        row = dict(d)
        display_spec = legacy._pick_display_specialization(
            row,
            preferred_specialty=spec_q,
            preferred_service=service_q,
        )
        row["specialization"] = legacy._compact_specialization(display_spec)
        compact.append(row)

    return {
        "doctors": compact,
        "note": "doctors_info: from cached registry (jsonl)",
        "cache_file": self._doctors_cache_path,
        "entities_used": {
            "doctor_query": fio_q,
            "doctor_resolved": resolved_surname,
            "specialty_query": spec_q,
            "service_query": service_q,
            "region_query": region_q,
            "output_limit": limit,
        },
    }


async def doctors_schedule_week(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает realtime-расписание врача или специальности.

    :param self: экземпляр сервисного слоя
    :param query: исходный запрос пользователя
    :param entities: сущности роутера
    :return: payload с расписанием или fallback-ответом
    """

    legacy = _legacy_module()
    raw_name = legacy._get_first_present(
        entities,
        ["last_name", "doctor_last_name", "doctor", "doctor_name", "fio"],
    )
    specialty = legacy._normalise_input(legacy._get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
    query_specialty = legacy._extract_specialty_from_text(query)
    query_procedure_specialty = legacy._procedure_query_role_specialty(query or "")
    if query_specialty:
        specialty = query_specialty
    elif query_procedure_specialty:
        specialty = query_procedure_specialty
    if raw_name and legacy._looks_like_schedule_specialty_token(str(raw_name)):
        raw_name = ""

    doctors = await self._ensure_doctors_cache_loaded()
    raw_for_match = str(raw_name or "").strip()
    if " " in raw_for_match:
        first = raw_for_match.split()[0].strip()
        raw_for_match = first or raw_for_match

    query_doctor_candidate = legacy.extract_doctor_name_candidate(str(query or ""), prefer_schedule=True) if query else None
    if query_doctor_candidate and legacy._looks_like_schedule_specialty_token(str(query_doctor_candidate)):
        query_doctor_candidate = None
    query_name = legacy.resolve_schedule_surname(str(query_doctor_candidate), doctors) if query_doctor_candidate else None
    last_name = legacy.resolve_schedule_surname(raw_for_match, doctors)
    has_schedule_signal = bool(legacy._SCHEDULE_QUERY_RE.search(str(query or "")))
    if query_name and (
        not last_name
        or has_schedule_signal
        or legacy._normalise_input(str(query_name)) != legacy._normalise_input(str(last_name))
    ):
        last_name = query_name
    elif not last_name and query and query != raw_name:
        if not specialty:
            last_name = query_name or legacy.resolve_schedule_surname(query, doctors)

    query_has_specialty_signal = bool(query_specialty or query_procedure_specialty)
    if query_has_specialty_signal and specialty and not query_name:
        last_name = None

    if not last_name and specialty:
        schedule_by_spec, schedule_unavailable_reason = await self._schedule_by_specialty(
            specialty,
            entities,
            nearest_only=legacy._has_nearest_hint(query),
            query_text=query,
        )
        return {
            "schedule": schedule_by_spec,
            "note": "doctors_schedule_week: by specialty",
            "schedule_unavailable_reason": schedule_unavailable_reason,
            "entities_used": {"specialty": specialty, "raw_name": raw_name},
        }

    if not last_name:
        return {
            "schedule": [],
            "note": "doctors_schedule_week: missing doctor last name",
            "entities_used": entities,
        }

    region_name = legacy._get_first_present(entities, ["region", "branch", "company_unit", "unit", "city", "branch_name"])
    if region_name and legacy._is_non_samara_city_value(region_name):
        return legacy._service_fallback(
            note=f"doctors_schedule_week unsupported city: {region_name}",
            handoff_message="Сейчас могу помочь только по Самаре. Соединяю с оператором.",
            entities=entities,
            reason="city_not_supported",
            extra={"schedule": []},
        )
    if region_name and legacy._is_samara_city_value(region_name):
        region_name = None

    data = None
    schedule_unavailable_reason: str | None = None
    candidates = legacy.surname_variants(str(last_name))
    if not candidates:
        candidates = [str(last_name)]
    if query_name:
        for qv in legacy.surname_variants(str(query_name)):
            if qv not in candidates:
                candidates.append(qv)

    try:
        for candidate in candidates:
            data = await self._get_schedule_payload_cached(candidate, region_name)
            if isinstance(data, list) and data and legacy._schedule_payload_matches_doctor(data, candidate):
                last_name = candidate
                break
            if legacy._is_schedule_no_slots_text(data):
                schedule_unavailable_reason = "no_free_slots_2_weeks"
            if region_name:
                data = await self._get_schedule_payload_cached(candidate, None)
                if isinstance(data, list) and data and legacy._schedule_payload_matches_doctor(data, candidate):
                    last_name = candidate
                    break
                if legacy._is_schedule_no_slots_text(data):
                    schedule_unavailable_reason = "no_free_slots_2_weeks"
    except Exception:
        return legacy._service_fallback(
            note="doctors_schedule_week unavailable",
            handoff_message="Сейчас не удалось получить расписание автоматически. Соединяю с оператором.",
            entities=entities,
            extra={"schedule": []},
        )

    if isinstance(data, list):
        compact_data: list[dict[str, Any]] = []
        samara_tokens = await self._samara_region_tokens()
        doctor_by_fio = {
            legacy._normalise_input(str(d.get("fio") or "")): d
            for d in doctors
            if isinstance(d, dict) and str(d.get("fio") or "").strip()
        }
        for row in data:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            if last_name:
                row_fio = str(item.get("fio") or "").strip()
                if row_fio and not legacy._doctor_matches_fio(row_fio, str(last_name), resolved_surname=str(last_name)):
                    continue
            row_fio_key = legacy._normalise_input(str(item.get("fio") or ""))
            cache_doc = doctor_by_fio.get(row_fio_key)
            if cache_doc:
                display_spec = legacy._pick_display_specialization(
                    cache_doc,
                    preferred_specialty=specialty,
                )
            else:
                display_spec = str(item.get("specialization") or "")
            item["specialization"] = legacy._compact_specialization(display_spec)
            regions_src = [str(x) for x in (item.get("regions") or []) if str(x).strip()]
            if legacy._has_explicit_non_samara_regions(regions_src):
                continue
            if samara_tokens:
                if regions_src and not any(legacy._region_matches_samara_tokens(x, samara_tokens) for x in regions_src):
                    continue
                sched = item.get("schedule")
                if isinstance(sched, dict) and sched:
                    sched_filtered: dict[str, Any] = {}
                    for k, v in sched.items():
                        if legacy._region_matches_samara_tokens(str(k), samara_tokens):
                            sched_filtered[k] = v
                    if sched_filtered:
                        item["schedule"] = sched_filtered
                    elif regions_src:
                        continue
            compact_data.append(item)
        data = compact_data

    return {
        "schedule": data or [],
        "note": "doctors_schedule_week: realtime from Nayka API",
        "schedule_unavailable_reason": schedule_unavailable_reason,
        "entities_used": {"last_name": last_name, "raw_name": raw_name, "region_name": region_name},
    }
