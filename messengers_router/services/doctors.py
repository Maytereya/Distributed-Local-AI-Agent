"""Модуль домена врачей.

Содержит pilot-миграцию doctor-related методов из ``services_legacy``.
Публичный API для внешних вызовов по-прежнему идёт через ``Services``.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

from agent_logic_2.nayka_api import api_price

from ..doctor_name_port import (
    extract_doctor_name_candidate,
    resolve_schedule_surname,
    surname_variants,
)
from .. import resilience as _R
from ..policies import handoff_message
from ..russian_nlu import normalize_ru
from ..service_phrase import extract_service_phrase
from ..specialty_parser import PROCEDURE_TO_SPECIALTY, PROCEDURE_TO_SPECIALTY_RE
from ._common import (
    DOCTORS_TOP_N,
    _as_int,
    _coerce_top_n,
    _get_first_present,
    _has_nearest_hint,
    _normalise_catalog_text,
    _normalise_input,
    _service_fallback,
)
from ._doctors_helpers import (
    _SCHEDULE_QUERY_RE,
    _SERVICE_QUERY_SIGNAL_RE,
    _compact_specialization,
    _dedupe_doctors_by_fio,
    _doctor_catalog_query_candidates,
    _doctor_matches_fio,
    _doctor_matches_service,
    _doctor_matches_specialty,
    _doctor_role_specialty_match_level,
    _doctor_sort_key,
    _extract_specialty_from_text,
    _is_role_specialty_query,
    _is_schedule_no_slots_text,
    _iter_slot_datetimes,
    _looks_like_schedule_specialty_token,
    _pick_display_specialization,
    _procedure_query_role_specialty,
    _schedule_payload_matches_doctor,
    _service_catalog_query_candidates,
    _specialty_priority_rank,
)
from ._prices_helpers import resolve_price_service_name_from_catalog
from ._regions import (
    _extract_region_phone,
    _has_explicit_non_samara_regions,
    _is_non_samara_city_value,
    _is_samara_city_value,
    _region_matches_samara_tokens,
    _schedule_regions_with_free_slots,
)


# Регэксп для поиска филиала на пр. Ленина, 5 в `addressForSite`/`name` /regions.
# Дом «5» отделён границей слова, чтобы НЕ ловить «Ленина 50», «Ленина 55».
_FIXED_EQUIPMENT_LENINA5_RE = re.compile(r"ленина[,\s]+5\b", re.I)

if TYPE_CHECKING:
    from .core import Services


def _normalise_text(value: str | None) -> str:
    """Нормализует строку для безопасного текстового сравнения.

    :param value: исходное значение
    :return: нормализованная строка без лишних пробелов
    """

    return " ".join(normalize_ru(value).split())


def _is_mapped_procedure_specialty_query(query: str, specialty: str) -> bool:
    """Проверяет, что специальность получена из известного маппинга процедуры.

    Нужен для безопасного fallback: если процедурный запрос уже был
    приземлён в специальность (`уретроскопия -> уролог`), но профиль врача
    не содержит текст процедуры, можно показать врачей этой специальности
    вместо пустой выдачи.

    :param query: исходный текст пользователя
    :param specialty: каноническая специальность
    :return: True, если запрос содержит процедуру из утвержденного маппинга
    """

    match = PROCEDURE_TO_SPECIALTY_RE.search(str(query or ""))
    if not match:
        return False
    mapped = PROCEDURE_TO_SPECIALTY.get(match.group(1).lower(), "")
    return _normalise_text(mapped) == _normalise_text(specialty)


async def match_catalog_doctor(self: "Services", raw_text_or_name: str) -> dict[str, Any]:
    """Матчит врача по кэшу каталога врачей.

    :param self: экземпляр сервисного слоя
    :param raw_text_or_name: исходный текст пользователя или фамилия врача
    :return: словарь со статусом матчинга и каноническим именем
    """

    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return {"status": "miss", "query": "", "canonical": ""}

    queries = _doctor_catalog_query_candidates(raw_text_or_name)
    if not queries:
        return {"status": "miss", "query": "", "canonical": ""}

    for query in queries:
        exact = resolve_schedule_surname(query, doctors)
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
        norm = _normalise_catalog_text(surname)
        if norm and norm not in surname_map:
            surname_map[norm] = surname
    surname_keys = list(surname_map.keys())
    if not surname_keys:
        return {"status": "miss", "query": "", "canonical": ""}

    for query in queries:
        norm = _normalise_catalog_text(query)
        if len(norm) < 4:
            continue
        hit = get_close_matches(norm, surname_keys, n=1, cutoff=0.84)
        if not hit:
            continue
        canonical = surname_map.get(hit[0], "").strip()
        if canonical and _normalise_catalog_text(canonical) != norm:
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

    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return [], None
    spec = _normalise_input(specialty)
    if not spec:
        return [], None

    samara_tokens = await self._samara_region_tokens()
    role_query = _is_role_specialty_query(query_text or specialty, spec)
    role_levels = {
        id(d): _doctor_role_specialty_match_level(d, spec)
        for d in doctors
    } if role_query else {}
    candidates = sorted(
        [
            d for d in doctors
            if (
                (role_levels.get(id(d), 0) > 0)
                if role_query
                else _doctor_matches_specialty(d, spec, query_text or specialty)
            )
            and (
                any(
                    _region_matches_samara_tokens(str(x), samara_tokens)
                    for x in (d.get("regions") or [])
                    if str(x).strip()
                )
                if samara_tokens
                else not _has_explicit_non_samara_regions(
                    [str(x) for x in (d.get("regions") or []) if str(x).strip()]
                )
            )
        ],
        key=(
            (lambda d: (
                -role_levels.get(id(d), 0),
                _specialty_priority_rank(d, spec),
                *_doctor_sort_key(d),
            ))
            if role_query
            else _doctor_sort_key
        ),
    )[:8]
    if not candidates:
        return [], None

    # Параллельно собираем расписания для всех 8 кандидатов.
    # Раньше цикл был последовательным: каждый surname блокировал
    # вызов следующего, плюс каждый `find_doctor_schedule` под капотом
    # делает каскад HTTP-вызовов к Nayka (/doctors → /doctorCompanyUnits
    # → /doctorRegions → /doctorSchedule × дни → /doctorScheduleCells).
    # Для специальностей с 8 кандидатами и многими повторными походами
    # в Nayka это давало >100 секунд (см. кейс «Расписание уролог»
    # 2026-04-30). Параллелизация с per-doctor timeout снимает узкое
    # место: один залипший врач больше не блокирует остальных.
    # Per-doctor timeout 60s — компромисс под текущую деградацию
    # Nayka schedule API. Замер 2026-04-30: одиночный
    # find_doctor_schedule сейчас занимает 38-224с (Трубин — 224с,
    # Тюрин — 60с, Михлик — 38с). 60с пропускает большинство врачей,
    # а аномально медленные (Трубин 224с) обрезаются — без них
    # расписание остальных всё равно вернётся.
    # Concurrency 8 = все 8 кандидатов едут разом → верхняя граница
    # на schedule-фазу = 60с (max одного fetch), а не 60с × batch.
    # Итого pipeline: LLM ≈ 14с + schedule ≤ 60с = до 74с в худшем
    # случае; типично 35-50с при текущей нагрузке Nayka.
    _PER_DOCTOR_TIMEOUT_S = 60.0
    _MAX_CONCURRENT_DOCTORS = 8
    _doctor_sem = asyncio.Semaphore(_MAX_CONCURRENT_DOCTORS)

    async def _fetch_one(doc: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        """Вернуть ``(row | None, matched_but_without_slots)`` для одного врача."""
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            return None, False
        surname = fio.split()[0]
        async with _doctor_sem:
            try:
                data = await asyncio.wait_for(
                    self._get_schedule_payload_cached(surname),
                    timeout=_PER_DOCTOR_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return None, False
            except Exception:
                return None, False
        if _is_schedule_no_slots_text(data):
            return None, True
        if not isinstance(data, list) or not data:
            return None, False
        for row in data:
            if not isinstance(row, dict):
                continue
            row_fio = str(row.get("fio") or "").strip()
            if row_fio and _normalise_input(row_fio) != _normalise_input(fio):
                continue
            item = dict(row)
            display_spec = _pick_display_specialization(
                doc,
                preferred_specialty=spec,
            )
            item["specialization"] = _compact_specialization(display_spec)
            slots = _iter_slot_datetimes(item.get("schedule") or {})
            if slots:
                item["_nearest_slot"] = min(slots)
            return item, False
        return None, False

    fetched = await asyncio.gather(*(_fetch_one(d) for d in candidates))

    out_rows: list[dict[str, Any]] = []
    matched_but_without_slots = False
    for row, no_slots in fetched:
        if no_slots:
            matched_but_without_slots = True
        if row is not None:
            out_rows.append(row)

    if not out_rows:
        if matched_but_without_slots:
            return [], "no_free_slots_2_weeks"
        return [], None
    with_slots = [x for x in out_rows if isinstance(x.get("_nearest_slot"), datetime)]
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

    target_norm = _normalise_input(fio_clean)
    chosen: dict[str, Any] | None = None
    for row in data:
        if not isinstance(row, dict):
            continue
        row_fio = str(row.get("fio") or "").strip()
        if not row_fio:
            continue
        row_norm = _normalise_input(row_fio)
        if target_norm and row_norm == target_norm:
            chosen = row
            break
        if _doctor_matches_fio(row_fio, fio_clean, resolved_surname=surname):
            chosen = row
            break
    # Без first-row fallback на FIO-промахе: кэш может вернуть «общий» список
    # ДРУГИХ врачей (ложный позитив API), и взятие первой строки показало бы
    # доступность не того врача (омоним). Честно отдаём unmatched.
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
                if _region_matches_samara_tokens(region, samara_tokens):
                    schedule[region] = days
        else:
            schedule = {str(k): v for k, v in schedule_raw.items()}

    slots = _iter_slot_datetimes(schedule)
    nearest_slot = min(slots).isoformat(timespec="minutes") if slots else ""
    return {
        "available": bool(slots),
        "nearest_slot": nearest_slot,
        "regions_with_slots": _schedule_regions_with_free_slots(schedule),
        "note": "availability_checked",
    }


async def resolve_doctor_name(self: "Services", raw_text_or_name: str) -> str | None:
    """Валидирует фамилию/ФИО врача по актуальному doctor-cache.

    :param self: экземпляр сервисного слоя
    :param raw_text_or_name: исходный текст пользователя
    :return: каноническая фамилия врача или ``None``
    """

    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return None
    value = str(raw_text_or_name or "").strip()
    if not value:
        return None
    resolved = resolve_schedule_surname(value, doctors)
    if resolved:
        return resolved

    candidate = extract_doctor_name_candidate(value, prefer_schedule=True)
    if candidate and _normalise_input(candidate) != _normalise_input(value):
        return resolve_schedule_surname(candidate, doctors)
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

    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return None, None

    raw = str(raw_text_or_name or "").strip()
    if not raw:
        return None, None

    resolved_surname = resolve_schedule_surname(raw, doctors)
    samara_tokens = await self._samara_region_tokens()
    matched: list[dict[str, Any]] = []
    for doc in doctors:
        if not isinstance(doc, dict):
            continue
        fio = str(doc.get("fio") or "").strip()
        if not fio:
            continue
        raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
        # Врач с практикой в нескольких городах (Самара + другой) не должен
        # выпадать из резолва — иначе нельзя оформить запись к нему.
        if samara_tokens:
            if raw_regions and not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                continue
        elif _has_explicit_non_samara_regions(raw_regions):
            continue
        if _doctor_matches_fio(fio, raw, resolved_surname):
            matched.append(doc)

    if not matched:
        return None, None
    matched = sorted(matched, key=_doctor_sort_key)
    first = matched[0]
    return _as_int(first.get("id")), str(first.get("fio") or "").strip() or None


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

    doctors = await self._ensure_doctors_cache_loaded()
    if not doctors:
        return _service_fallback(
            note="doctors_info source unavailable",
            handoff_message=handoff_message("service_error_doctors_list"),
            entities=entities,
            extra={"doctors": []},
        )

    q = _normalise_input(query)
    doctor_raw = _get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
    fio_q = _normalise_input(doctor_raw)
    spec_q = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
    if not spec_q:
        spec_q = _extract_specialty_from_text(query)
    service_q = _normalise_input(_get_first_present(entities, ["service_name", "test_name"]) or "")
    if not service_q:
        if _SERVICE_QUERY_SIGNAL_RE.search(_normalise_input(query)):
            extracted_service = extract_service_phrase(query)
            if extracted_service:
                service_q = _normalise_input(extracted_service)
    region_q = _normalise_input(_get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")
    resolved_surname = resolve_schedule_surname(doctor_raw, doctors) if doctor_raw else None

    query_candidate = extract_doctor_name_candidate(query, prefer_schedule=True) if query else None
    query_resolved = resolve_schedule_surname(query_candidate, doctors) if query_candidate else None
    if not resolved_surname:
        resolved_surname = query_resolved

    if spec_q and not query_resolved:
        fio_q = ""
        resolved_surname = None

    samara_tokens = await self._samara_region_tokens()
    role_query = bool(spec_q and _is_role_specialty_query(query, spec_q))
    role_levels = {
        id(d): _doctor_role_specialty_match_level(d, spec_q)
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

    def match_doc(doc: dict[str, Any], *, require_service: bool = True) -> bool:
        fio = _normalise_input(str(doc.get("fio", "")))
        spec_text = _normalise_input(str(doc.get("specialization", "")))
        raw_regions = [str(x) for x in (doc.get("regions") or []) if str(x).strip()]
        regions = " ".join([_normalise_input(x) for x in raw_regions])
        units = " ".join([_normalise_input(str(x)) for x in (doc.get("units") or [])])

        hay = " | ".join([fio, spec_text, regions, units])
        # Врач может вести приём и в Самаре, и в другом городе — оставляем его
        # по самарскому присутствию, а не отбрасываем по любому несамарскому
        # филиалу. «Явный несамарский» дроп — fallback без samara_tokens.
        if samara_tokens:
            if not any(_region_matches_samara_tokens(x, samara_tokens) for x in raw_regions):
                return False
        elif _has_explicit_non_samara_regions(raw_regions):
            return False
        if fio_q:
            if not _doctor_matches_fio(fio, fio_q, resolved_surname):
                return False
        if spec_q:
            if role_query:
                if role_levels.get(id(doc), 0) <= 0:
                    return False
            elif not _doctor_matches_specialty(doc, spec_q, query):
                return False
        if require_service and service_q:
            if not _doctor_matches_service(doc, service_q):
                return False
        if region_q and region_q not in hay:
            return False

        if not (fio_q or spec_q or region_q or service_q):
            if len(keyword) < 3:
                return False
            return keyword in hay

        return True

    filtered = [d for d in doctors if match_doc(d)]
    if (
        not filtered
        and service_q
        and spec_q
        and role_query
        and _is_mapped_procedure_specialty_query(query, spec_q)
    ):
        filtered = [d for d in doctors if match_doc(d, require_service=False)]
    filtered = _dedupe_doctors_by_fio(filtered)
    if role_query:
        filtered = sorted(
            filtered,
            key=lambda d: (
                -role_levels.get(id(d), 0),
                _specialty_priority_rank(d, spec_q),
                *_doctor_sort_key(d),
            ),
        )
    else:
        filtered = sorted(filtered, key=_doctor_sort_key)

    limit = _coerce_top_n(output_max, default=DOCTORS_TOP_N)
    if resolved_surname:
        limit = min(limit, 3)

    filtered = filtered[:limit]
    compact: list[dict[str, Any]] = []
    for d in filtered:
        row = dict(d)
        display_spec = _pick_display_specialization(
            row,
            preferred_specialty=spec_q,
            preferred_service=service_q,
        )
        row["specialization"] = _compact_specialization(display_spec)
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


async def _fixed_equipment_handoff_payload(self: "Services") -> tuple[str, str]:
    """Возвращает ``(address, phone)`` регистратуры филиала с фиксированным оборудованием.

    Используется для запроса расписания процедур, у которых нет приёмного
    врача с расписанием в Naika (флюорограф/маммограф на пр. Ленина, 5).
    Запись таких процедур всегда ведёт регистратура филиала.

    :param self: экземпляр сервисного слоя
    :return: кортеж ``(адрес, телефон)``; телефон может быть пустой строкой,
             если в /regions нет соответствующей записи или у неё нет телефона
    """

    fixed_addr = "г. Самара, пр. Ленина, 5"
    try:
        regions = await self._ensure_regions_loaded()
    except Exception:
        regions = []
    for r in regions:
        if not isinstance(r, dict):
            continue
        for key in ("addressForSite", "name", "address"):
            raw = str(r.get(key) or "")
            if _FIXED_EQUIPMENT_LENINA5_RE.search(raw):
                phone = _extract_region_phone(r)
                if phone:
                    return fixed_addr, phone
                # продолжаем — вдруг другая запись /regions содержит телефон
                break
    return fixed_addr, ""


def _fixed_equipment_handoff_message(address: str, phone: str) -> str:
    """Формирует patient-facing handoff-сообщение для процедур с фиксированным оборудованием.

    :param address: адрес филиала с оборудованием
    :param phone: телефон регистратуры (может быть пустой строкой)
    :return: текст сообщения для пациента
    """
    if phone:
        return (
            f"Запись на флюорографию и маммографию ведётся через регистратуру "
            f"филиала {address}, телефон: {phone}. Соединяю с оператором, "
            f"чтобы подтвердить удобное время."
        )
    return (
        f"Запись на флюорографию и маммографию ведётся через регистратуру "
        f"филиала {address}. Соединяю с оператором."
    )


async def doctors_schedule_week(self: "Services", query: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Возвращает realtime-расписание врача или специальности.

    :param self: экземпляр сервисного слоя
    :param query: исходный запрос пользователя
    :param entities: сущности роутера
    :return: payload с расписанием или fallback-ответом
    """

    # Short-circuit для процедур, привязанных к фиксированному оборудованию
    # (флюорограф/маммограф). У клиники нет приёмного врача-радиолога с
    # расписанием в Naika — рентгенолог только пишет заключения и помечен
    # «Не выгружать на сайт». Поэтому любые попытки найти расписание ниже
    # приводят либо к пустой выдаче, либо к подмене на УЗИ-врача (см. кейс
    # «расписание флюорографии» 22.04.2026). Временное решение — сразу
    # отдавать handoff с телефоном регистратуры филиала на Ленина 5.
    if api_price.resolve_diagnostic_fixed_addresses(query or ""):
        addr, phone = await _fixed_equipment_handoff_payload(self)
        return _service_fallback(
            note="doctors_schedule_week: fixed equipment → registry handoff",
            handoff_message=handoff_message(
                "schedule_via_registry_fixed_equipment",
                override=_fixed_equipment_handoff_message(addr, phone),
            ),
            entities=entities,
            reason="schedule_via_registry_fixed_equipment",
            extra={"schedule": [], "fixed_equipment_branch": {"address": addr, "phone": phone}},
        )

    raw_name = _get_first_present(
        entities,
        ["last_name", "doctor_last_name", "doctor", "doctor_name", "fio"],
    )
    specialty = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
    query_specialty = _extract_specialty_from_text(query)
    query_procedure_specialty = _procedure_query_role_specialty(query or "")
    if query_specialty:
        specialty = query_specialty
    elif query_procedure_specialty:
        specialty = query_procedure_specialty
    if raw_name and _looks_like_schedule_specialty_token(str(raw_name)):
        raw_name = ""

    doctors = await self._ensure_doctors_cache_loaded()
    raw_for_match = str(raw_name or "").strip()
    if " " in raw_for_match:
        first = raw_for_match.split()[0].strip()
        raw_for_match = first or raw_for_match

    query_doctor_candidate = extract_doctor_name_candidate(str(query or ""), prefer_schedule=True) if query else None
    if query_doctor_candidate and _looks_like_schedule_specialty_token(str(query_doctor_candidate)):
        query_doctor_candidate = None
    query_name = resolve_schedule_surname(str(query_doctor_candidate), doctors) if query_doctor_candidate else None
    last_name = resolve_schedule_surname(raw_for_match, doctors)
    has_schedule_signal = bool(_SCHEDULE_QUERY_RE.search(str(query or "")))
    if query_name and (
        not last_name
        or has_schedule_signal
        or _normalise_input(str(query_name)) != _normalise_input(str(last_name))
    ):
        last_name = query_name
    elif not last_name and query and query != raw_name:
        if not specialty:
            last_name = query_name or resolve_schedule_surname(query, doctors)

    query_has_specialty_signal = bool(query_specialty or query_procedure_specialty)
    if query_has_specialty_signal and specialty and not query_name:
        last_name = None

    if not last_name and specialty:
        schedule_by_spec, schedule_unavailable_reason = await self._schedule_by_specialty(
            specialty,
            entities,
            nearest_only=_has_nearest_hint(query),
            query_text=query,
        )
        return {
            "schedule": schedule_by_spec,
            "note": "doctors_schedule_week: by specialty",
            "schedule_unavailable_reason": schedule_unavailable_reason,
            "entities_used": {"specialty": specialty, "raw_name": raw_name},
        }

    if not last_name:
        # Имя врача было названо («Записаться к <ФИО>»), но фамилия не
        # резолвится в самарском каталоге врачей — значит врача нет в нашей
        # системе онлайн-записи (напр. принимает только в Оренбурге, регион
        # исключён из кэша через EXCLUDED_REGION_ROOTS). Помечаем явным
        # сигналом `doctor_lookup=unresolved`, чтобы response_builder не
        # предлагал самарские филиалы вслепую, а честно сообщил об
        # ограничении и предложил оператора.
        return {
            "schedule": [],
            "note": "doctors_schedule_week: missing doctor last name",
            "doctor_lookup": "unresolved" if str(raw_name or "").strip() else "",
            "entities_used": entities,
        }

    region_name = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "branch_name"])
    city_name = _get_first_present(entities, ["city"])
    if city_name and _is_non_samara_city_value(city_name):
        return _service_fallback(
            note=f"doctors_schedule_week unsupported city: {city_name}",
            handoff_message=handoff_message("city_not_supported"),
            entities=entities,
            reason="city_not_supported",
            extra={"schedule": []},
        )
    if not region_name and city_name:
        region_name = city_name
    if region_name and _is_samara_city_value(region_name):
        region_name = None

    data = None
    schedule_unavailable_reason: str | None = None
    candidates = surname_variants(str(last_name))
    if not candidates:
        candidates = [str(last_name)]
    if query_name:
        for qv in surname_variants(str(query_name)):
            if qv not in candidates:
                candidates.append(qv)

    try:
        for candidate in candidates:
            data = await self._get_schedule_payload_cached(candidate, region_name)
            # ВАЖНО: проверяем «свободных слотов нет» ДО positive-match.
            # Synthetic-list `[{"_no_free_slots": True, ...}]` из
            # `_fetch_schedule_source` тоже non-empty, и `_schedule_payload_matches_doctor`
            # может его принять за валидное расписание (если в нём есть
            # `fio`). Без раннего ветвления получим пустое расписание
            # вместо корректного handoff на оператора по «нет слотов».
            if _is_schedule_no_slots_text(data):
                schedule_unavailable_reason = "no_free_slots_2_weeks"
                last_name = candidate
                data = []
                continue
            if isinstance(data, list) and data and _schedule_payload_matches_doctor(data, candidate):
                last_name = candidate
                # Нашли реальное расписание — гасим протухший «нет слотов»,
                # выставленный более ранним кандидатом-вариантом фамилии.
                schedule_unavailable_reason = None
                break
            if region_name:
                data = await self._get_schedule_payload_cached(candidate, None)
                if _is_schedule_no_slots_text(data):
                    schedule_unavailable_reason = "no_free_slots_2_weeks"
                    last_name = candidate
                    data = []
                    continue
                if isinstance(data, list) and data and _schedule_payload_matches_doctor(data, candidate):
                    last_name = candidate
                    schedule_unavailable_reason = None
                    break
    except Exception:
        # Hard schedule-source failure with no best-effort data at this layer →
        # honest operator handoff (behavior unchanged); fallback_used=False accordingly.
        _R.log_degraded(
            upstream="doctor_schedule",
            failure_mode=_R.FM_EXCEPTION,
            fallback_used=False,
        )
        payload = _service_fallback(
            note="doctors_schedule_week unavailable",
            handoff_message=handoff_message("service_error_schedule"),
            entities=entities,
            extra={"schedule": []},
        )
        return _R.mark_degraded(
            payload, upstream="doctor_schedule", failure_mode=_R.FM_EXCEPTION, fallback_used=False
        )

    if isinstance(data, list):
        compact_data: list[dict[str, Any]] = []
        samara_tokens = await self._samara_region_tokens()
        doctor_by_fio = {
            _normalise_input(str(d.get("fio") or "")): d
            for d in doctors
            if isinstance(d, dict) and str(d.get("fio") or "").strip()
        }
        for row in data:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            if last_name:
                row_fio = str(item.get("fio") or "").strip()
                if row_fio and not _doctor_matches_fio(row_fio, str(last_name), resolved_surname=str(last_name)):
                    continue
            row_fio_key = _normalise_input(str(item.get("fio") or ""))
            cache_doc = doctor_by_fio.get(row_fio_key)
            if cache_doc:
                display_spec = _pick_display_specialization(
                    cache_doc,
                    preferred_specialty=specialty,
                )
            else:
                display_spec = str(item.get("specialization") or "")
            item["specialization"] = _compact_specialization(display_spec)
            regions_src = [str(x) for x in (item.get("regions") or []) if str(x).strip()]
            # ВАЖНО: врач может вести приём и в Самаре, и в другом городе
            # (напр. Лунев: «Ленина 5» + Оренбург). Нельзя отбрасывать его
            # целиком по `_has_explicit_non_samara_regions` — иначе теряем
            # самарское расписание. При наличии samara_tokens отбираем по
            # самарскому присутствию и обрезаем регионы/расписание до
            # самарских; «явный несамарский» дроп оставляем как fallback на
            # случай недоступности /regions.
            if samara_tokens:
                if regions_src and not any(_region_matches_samara_tokens(x, samara_tokens) for x in regions_src):
                    continue
                sched = item.get("schedule")
                if isinstance(sched, dict) and sched:
                    sched_filtered: dict[str, Any] = {}
                    for k, v in sched.items():
                        if _region_matches_samara_tokens(str(k), samara_tokens):
                            sched_filtered[k] = v
                    if sched_filtered:
                        item["schedule"] = sched_filtered
                    elif regions_src:
                        continue
                samara_regions = [
                    x for x in regions_src
                    if _region_matches_samara_tokens(x, samara_tokens)
                ]
                if samara_regions:
                    item["regions"] = samara_regions
            elif _has_explicit_non_samara_regions(regions_src):
                continue
            compact_data.append(item)
        data = compact_data

    return {
        "schedule": data or [],
        "note": "doctors_schedule_week: realtime from Nayka API",
        "schedule_unavailable_reason": schedule_unavailable_reason,
        "entities_used": {"last_name": last_name, "raw_name": raw_name, "region_name": region_name},
    }


async def match_catalog_service(
    self: "Services",
    raw_text_or_name: str,
    *,
    current_service_name: str = "",
) -> dict[str, Any]:
    """
    Матчит услугу по объединенному каталогу услуг клиники:
    1) exact через resolver price-catalog
    2) fuzzy через difflib по нормализованным названиям услуг
    """

    queries = _service_catalog_query_candidates(
        raw_text_or_name,
        current_service_name=current_service_name,
    )
    if not queries:
        return {"status": "miss", "query": "", "canonical": ""}

    catalog_rows = await self._ensure_service_catalog_rows_loaded()
    if not catalog_rows:
        return {
            "status": "unavailable",
            "query": queries[0],
            "canonical": "",
            "reason": self._service_catalog_last_error or "service_catalog_empty",
        }

    for query in queries:
        exact = resolve_price_service_name_from_catalog(
            query,
            current_service_name="",
            rows=catalog_rows,
        )
        if exact:
            return {
                "status": "exact",
                "query": query,
                "canonical": str(exact).strip(),
            }

    name_map: dict[str, str] = {}
    for row in catalog_rows:
        name = str(row.get("serviceName") or row.get("name") or "").strip()
        norm = _normalise_catalog_text(name)
        if norm and norm not in name_map:
            name_map[norm] = name
    name_keys = list(name_map.keys())
    if not name_keys:
        return {"status": "miss", "query": "", "canonical": ""}

    for query in queries:
        norm = _normalise_catalog_text(query)
        if len(norm) < 4:
            continue
        hit = get_close_matches(norm, name_keys, n=1, cutoff=0.86)
        if not hit:
            continue
        canonical = str(name_map.get(hit[0]) or "").strip()
        if canonical and _normalise_catalog_text(canonical) != norm:
            return {
                "status": "fuzzy",
                "query": query,
                "canonical": canonical,
                "matched_key": hit[0],
            }

    return {"status": "miss", "query": queries[0], "canonical": ""}
