"""Сервисный слой интеграций для роутера пациентов.

Содержит вызовы внешних источников (Nayka API, price, meili), кэш врачей,
поиск расписания/цен/адресов и fallback-контракты для handoff при сбоях.

Ответственность модуля:
1) Доступ к внешним данным и их нормализация к стабильному внутреннему формату.
2) Локальный кэш/дедупликация/ограничение объема данных для рендера.
3) Прозрачный graceful degradation (note/reason/handoff flags) при сбоях.

Модуль не должен принимать state-machine решения по диалогу.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime  # noqa: F401 — re-exported for services/doctors.py via legacy.datetime
from dataclasses import dataclass, field
from difflib import get_close_matches  # noqa: F401 — re-exported for services/doctors.py via legacy.get_close_matches
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch  # noqa: F401 — re-exported for services/main_index.py, services/news.py
from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info
from converters import html_cleaner  # noqa: F401 — re-exported for services/main_index.py via legacy.html_cleaner
from schedule_ttl_cache import AsyncListTTLStaleCache

# Re-exported for domain modules (services/doctors.py etc.) that access these via
# ``_legacy_module().<name>``. Ruff cannot see the dynamic access, hence noqa.
from .doctor_name_port import (  # noqa: F401
    extract_doctor_name_candidate,
    resolve_schedule_surname,
    surname_variants,
)
from .llm_doesnt_work_fallback import build_prepare_fallback_answer  # noqa: F401 — re-exported for services/prepare.py
from .llm_runtime import generate_text  # noqa: F401 — re-exported for services/*.py via legacy.generate_text (tests monkey-patch svc_mod.generate_text)
from .policies import handoff_message  # noqa: F401 — re-exported for services/*.py via legacy.handoff_message
from .service_phrase import extract_service_phrase  # noqa: F401 — re-exported for services/doctors.py, services/addresses.py

# Stage 20 cluster 1 — common helpers moved to services/_common.py.
# Re-exported here so internal references and external imports
# (e.g. ``services_legacy.DOCTORS_TOP_N`` from services/doctors.py) keep working.
from .services._common import (  # noqa: F401
    DOCTORS_TOP_N,
    _as_int,
    _coerce_top_n,
    _dedupe_str,
    _get_first_present,
    _has_nearest_hint,
    _is_main_index_relevant,
    _is_meili_error_text,
    _is_meili_no_matches_text,
    _normalise_catalog_text,
    _normalise_input,
    _runtime_bool,
    _runtime_float,
    _runtime_int,
    _service_fallback,
    _tax_doc_guidance_response,
)

# Stage 20 cluster 2 — prepare helpers moved to services/_prepare.py.
# Re-exported here so internal references keep working.
from .services._prepare import (  # noqa: E402, F401
    _PREPARE_RELEVANCE_VERDICT_IRRELEVANT,
    _PREPARE_RELEVANCE_VERDICT_RELEVANT,
    _PrepareCandidate,
    _dedupe_prepare_candidates,
    _has_prepare_strong_hints,
    _is_prepare_content_actionable,
    _is_prepare_requested_in_price_query,
    _is_prepare_service_info_usable,
    _is_prepare_wrap_output_usable,
    _parse_prepare_relevance_validator,
    _prepare_clarify_response,
    _prepare_fast_relevance_score,
    _prepare_query_variants,
    _prepare_relevance_gate,
    _prepare_relevance_prompt,
    _prepare_roots_coverage,
    _prepare_roots_match,
    _prepare_service_info_queries,
    _prepare_subject_hint,
    _prepare_term_roots,
    _prepare_wrap_clean,
    _prepare_wrap_prompt,
)

# Stage 20 cluster 3 — region/city helpers moved to services/_regions.py.
# Re-exported here so internal references and external callers keep working.
from .services._regions import (  # noqa: E402, F401
    _ADDRESS_HINT_RE,
    _CITY_PREFIX_RE,
    _PHONE_EXTRACT_RE,
    _compact_region_text,
    _extract_city_token,
    _extract_region_phone,
    _extract_region_work_time,
    _filter_regions_by_service_flags,
    _has_explicit_non_samara_regions,
    _is_explicit_non_samara_region,
    _is_non_samara_city_value,
    _is_samara_city_value,
    _norm_city,
    _normalize_region_text,
    _region_display_name,
    _region_matches_samara_tokens,
    _schedule_regions_with_free_slots,
)

# Stage 20 cluster 4 — address/branch helpers moved to services/_addresses_helpers.py.
# Re-exported here so internal references and external callers keep working.
from .services._addresses_helpers import (  # noqa: E402, F401
    _NONBOOKABLE_POINTS_PATH,
    _PRICE_HOMECODE_DOTTED_RE,
    _PRICE_HOMECODE_NUM_RE,
    _PROCEDURE_BRANCH_LOOKUP_RE,
    _STATIC_PROCEDURE_BRANCH_OVERRIDES,
    _addresses_to_branch_payload,
    _extract_homecode_query,
    _is_procedure_branch_lookup_query,
    _load_nonbookable_points,
    _looks_like_real_address,
    _nonbookable_needs,
    _soft_address_match,
    _static_nonbookable_branches,
    _static_procedure_addresses,
)

# Stage 20 cluster 6 — doctor/specialty/service helpers moved to
# services/_doctors_helpers.py. Re-exported here so internal references and
# external callers keep working.
from .services._doctors_helpers import (  # noqa: E402, F401
    _CATALOG_DOCTOR_STOPWORDS,
    _CATALOG_SERVICE_LEADIN_RE,
    _CATALOG_SERVICE_SIGNAL_RE,
    _CATALOG_SERVICE_STOPWORDS,
    _CATALOG_SERVICE_TRAILING_TIME_RE,
    _CATALOG_WORD_RE,
    _FIO_TOKEN_RE,
    _SCHEDULE_QUERY_RE,
    _SCHEDULE_SPECIALTY_TOKENS,
    _SERVICE_FILTER_STOPWORDS,
    _SERVICE_QUERY_SIGNAL_RE,
    _SPECIALTY_PRIORITY_SURNAMES,
    _UZI_FALSE_POSITIVE_RE,
    _UZI_LINE_RE,
    _UZI_PROCEDURE_HINT_RE,
    _UZI_ROLE_HINT_RE,
    _classify_catalog_service_kind,
    _collect_role_unit_names,
    _compact_specialization,
    _dedupe_doctors_by_fio,
    _detect_service_kind,
    _doctor_catalog_query_candidates,
    _doctor_main_payload,
    _doctor_matches_fio,
    _doctor_matches_primary_specialty,
    _doctor_matches_service,
    _doctor_matches_specialty,
    _doctor_role_specialty_match_level,
    _doctor_sort_key,
    _extract_specialties_from_text,
    _extract_specialty_from_text,
    _fio_tokens,
    _has_reliable_doctor_service_link,
    _is_clean_consultation_row_name,
    _is_consultation_service_query,
    _is_direct_specialty_text_match,
    _is_lab_like_service_name,
    _is_price_service_noise_token,
    _is_role_specialty_query,
    _is_schedule_no_slots_text,
    _is_uzi_query_text,
    _iter_slot_datetimes,
    _iter_unit_link_specs,
    _looks_like_schedule_specialty_token,
    _matches_specialty_terms,
    _matches_uzi_doctor_profile,
    _meaningful_price_service_tokens,
    _pick_display_specialization,
    _procedure_query_role_specialty,
    _schedule_payload_matches_doctor,
    _select_effective_price_service_name,
    _service_catalog_query_candidates,
    _service_name_allows_specialty,
    _service_name_matches_specialty,
    _service_query_matches,
    _service_tokens,
    _should_prefer_retail_query_candidate,
    _specialization_matches_specialty,
    _specialty_equivalent,
    _specialty_label_for_doctor,
    _specialty_norm,
    _specialty_priority_rank,
    _specialty_terms,
    _split_spec_lines,
    _stem_service_token,
)

# Stage 20 cluster 5 — price helpers moved to services/_prices_helpers.py.
# Re-exported here so internal references and external callers keep working.
from .services._prices_helpers import (  # noqa: E402, F401
    SAMARA_PRICE_REGION_ID,
    _DOCTOR_PRICE_HINT_RE,
    _DOCTOR_SERVICE_HINT_RE,
    _LAB_DEADLINE_HINT_RE,
    _LAB_SERVICE_HINT_RE,
    _MULTI_PRICE_SERVICE_HINT_RE,
    _MULTI_PRICE_SPLIT_RE,
    _OAK_ALIAS_RE,
    _OAK_CANONICAL_BASE_RE,
    _OAK_HOMEVISIT_RE,
    _OAK_PHRASE_RE,
    _PRICE_CAPILLARY_QUERY_RE,
    _PRICE_CAPILLARY_ROW_RE,
    _PRICE_CHILD_QUERY_RE,
    _PRICE_CHILD_ROW_RE,
    _PRICE_CITO_QUERY_RE,
    _PRICE_CITO_ROW_RE,
    _PRICE_COMPOUND_LAB_FRAGMENT_RE,
    _PRICE_CONSULT_EXCLUDE_RE,
    _PRICE_CONSULT_HINT_RE,
    _PRICE_DIAGNOSTIC_NO_DOCTOR_RE,
    _PRICE_DOCTOR_SUFFIX_RE,
    _PRICE_GENERIC_FAMILY_ROOT_TOKENS,
    _PRICE_GENERIC_SERVICE_TOKENS,
    _PRICE_GENETIC_QUERY_RE,
    _PRICE_GENETIC_ROW_RE,
    _PRICE_HOME_QUERY_RE,
    _PRICE_HOME_ROW_RE,
    _PRICE_KMN_QUERY_RE,
    _PRICE_KMN_ROW_RE,
    _PRICE_PACKAGE_QUERY_RE,
    _PRICE_PACKAGE_ROW_RE,
    _PRICE_PROCEDURE_LIKE_RE,
    _PRICE_QUERY_CANONICAL_TOKENS,
    _PRICE_QUERY_SERVICE_NOISE_TOKENS,
    _PRICE_QUERY_STOPWORDS,
    _PRICE_REPEAT_QUERY_RE,
    _PRICE_REPEAT_ROW_RE,
    _PRICE_REQUEST_RE,
    _PRICE_SERVICE_ALIASES,
    _PRICE_SERVICE_PREFIX_RE,
    _PRICE_SHORT_TOKEN_WHITELIST,
    _PRICE_SHOW_ALL_RE,
    _PRICE_TOKEN_RE,
    _VITAMIN_CODE_MAP,
    _annotate_price_rows_with_care_context,
    _augment_price_tokens,
    _build_compound_price_clarify_payload,
    _build_family_candidate_rows,
    _build_multi_price_payload,
    _build_oak_canonical_payload,
    _build_price_catalog_queries,
    _build_price_family_payload,
    _build_price_kind_ambiguous_prompt,
    _care_setting_addresses_from_price_rows,
    _compound_price_secondary_lab_service,
    _dedupe_price_queries,
    _dedupe_price_rows,
    _expand_family_rows_by_root_token,
    _extract_price_service_from_query,
    _extract_vitamin_designator,
    _family_query_root_tokens,
    _family_variant_base_names,
    _has_specific_price_tokens,
    _is_city_only_reply,
    _is_family_query_candidate,
    _is_generic_uzi_price_request,
    _is_lab_price_query_for_catalog,
    _is_oak_base_query,
    _is_price_match_strong,
    _is_price_show_all_request,
    _is_strong_doctor_price_match,
    _lab_price_variant_flags,
    _normalise_family_variant_name,
    _normalise_price_token,
    _parse_price_kind_ambiguous_result,
    _price_alias_candidates,
    _price_family_payload_from_context,
    _price_query_tokens,
    _price_row_modifier_penalty,
    _price_row_score,
    _query_nonbase_price_flags,
    _query_price_variant_flags,
    _rank_price_rows,
    _resolve_ambiguous_price_kind_with_llm,
    _resolve_best_price_row_from_queries,
    _resolve_multi_price_items,
    _resolve_price_alias_from_catalog,
    _row_nonbase_price_flags,
    _score_price_rows,
    _select_address_price_rows,
    _select_family_variant_rows,
    _select_oak_canonical_rows,
    _select_patient_price_rows,
    _should_prefer_current_price_query_over_context,
    _split_price_query_items,
    match_compound_price_service_option,
    resolve_price_service_name_from_catalog,
)

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """
    Сервисный слой для patient/messenger router.

    - doctors_info: использует ФАЙЛОВЫЙ кэш врачей (JSONL) + in-memory кэш.
    - doctors_schedule_week: realtime с коротким TTL-кэшем и stale-fallback при сбоях API.
    """

    # in-memory кэш врачей
    # Приходится использовать идентификатор полей "field" и его свойство default_factory=list из @dataclass, так как list
    # относится к изменяемым типам данных. В противном случае переменная спика будет
    # рандомно перезаписываться в неожиданных местах (база).
    _doctors_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    # Тут неизменяемый тип str, не усложняем:
    _doctors_cache_path: Optional[str] = field(default=None, init=False)
    _doctors_cache_loaded_at: float = field(default=0.0, init=False)
    _regions_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _regions_cache_loaded_at: float = field(default=0.0, init=False)
    _procedure_rows_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _procedure_rows_loaded_at: float = field(default=0.0, init=False)
    _service_catalog_rows_cache: list[dict[str, Any]] = field(default_factory=list, init=False)
    _service_catalog_rows_loaded_at: float = field(default=0.0, init=False)
    _service_catalog_last_error: str = field(default="", init=False)
    _service_catalog_last_source_counts: dict[str, int] = field(default_factory=dict, init=False)
    _schedule_cache_client: AsyncListTTLStaleCache = field(init=False)

    # блокировка, чтобы несколько запросов параллельно не перегенерировали кэш
    _doctors_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _regions_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _procedure_rows_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _service_catalog_rows_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # TTL in-memory кэша (латентность, сек, определят свежесть кэша)
    doctors_mem_ttl_seconds: int = 300
    regions_mem_ttl_seconds: int = 300
    procedure_rows_mem_ttl_seconds: int = 300
    service_catalog_mem_ttl_seconds: int = 300
    schedule_fresh_ttl_seconds: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_FRESH_TTL_SECONDS",
            30,
            min_value=1,
            max_value=300,
        )
    )
    schedule_stale_ttl_seconds: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_STALE_TTL_SECONDS",
            600,
            min_value=1,
            max_value=3600,
        )
    )
    schedule_negative_ttl_seconds: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_NEGATIVE_TTL_SECONDS",
            15,
            min_value=1,
            max_value=120,
        )
    )
    schedule_cache_max_keys: int = field(
        default_factory=lambda: _runtime_int(
            "MR_SCHEDULE_CACHE_MAX_KEYS",
            1000,
            min_value=50,
            max_value=10000,
        )
    )
    schedule_cache_log_events: bool = field(
        default_factory=lambda: _runtime_bool("MR_SCHEDULE_CACHE_LOG_EVENTS", False)
    )

    def __post_init__(self) -> None:
        self._schedule_cache_client = AsyncListTTLStaleCache(
            fresh_ttl_seconds=self.schedule_fresh_ttl_seconds,
            stale_ttl_seconds=self.schedule_stale_ttl_seconds,
            negative_ttl_seconds=self.schedule_negative_ttl_seconds,
            max_keys=self.schedule_cache_max_keys,
            logger=logger,
            log_events=self.schedule_cache_log_events,
            name="schedule_cache",
            time_func=lambda: time.time(),
        )

    # -----------------------------
    # Low-level helpers (async)
    # -----------------------------

    def ensure_background_refresh_started(self) -> None:
        """Запускает фоновые refresh-задачи кэшей (idempotent)."""
        try:
            api_price.ensure_daily_price_refresh_started()
        except Exception:
            pass
        try:
            api_service_info.ensure_daily_service_info_refresh_started()
        except Exception:
            pass

    @staticmethod
    def _schedule_cache_key(last_name: str, region_name: str | None = None) -> tuple[str, str]:
        return _normalise_input(last_name), _normalise_input(region_name or "")

    async def _fetch_schedule_source(self, last_name: str, region_name: str | None = None) -> Any:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                if region_name:
                    try:
                        return await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name, region_name)
                    except TypeError:
                        return await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name)
                return await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    await asyncio.sleep(0.12)
                    continue
        if last_exc is not None:
            raise last_exc
        return []

    async def _get_schedule_payload_cached(self, last_name: str, region_name: str | None = None) -> Any:
        key = self._schedule_cache_key(last_name, region_name)
        return await self._schedule_cache_client.get_or_fetch(
            key,
            lambda: self._fetch_schedule_source(last_name, region_name),
            key_details={"last_name": key[0], "region": key[1]},
        )

    async def _ensure_doctors_cache_loaded(self) -> list[dict[str, Any]]:
        """
        1) Пытаемся получить свежий doctor cache через api_nayka.get_cached_doctors_data()
        2) При неуспехе откатываемся на последний непустой файл
        3) Держим in-memory-кэш поверх файлового кэша
        """
        now = time.time()

        # быстрый путь: ещё не протухло
        if self._doctors_cache and (now - self._doctors_cache_loaded_at) < self.doctors_mem_ttl_seconds:
            return self._doctors_cache

        async with self._doctors_lock:
            # повторная проверка под локом
            now = time.time()
            if self._doctors_cache and (now - self._doctors_cache_loaded_at) < self.doctors_mem_ttl_seconds:
                return self._doctors_cache

            try:
                doctors_loaded = await asyncio.to_thread(api_nayka.get_cached_doctors_data)
                file_path = await asyncio.to_thread(api_nayka.find_existing_doctors_file)
            except Exception:
                return self._doctors_cache or []

            self._doctors_cache = doctors_loaded
            self._doctors_cache_path = str(file_path) if file_path is not None else None
            self._doctors_cache_loaded_at = time.time()
            return self._doctors_cache

    async def _ensure_regions_loaded(self) -> list[dict[str, Any]]:
        """
        Live-список регионов/филиалов из Nayka API (/regions) с коротким in-memory TTL.
        """
        now = time.time()
        if self._regions_cache and (now - self._regions_cache_loaded_at) < self.regions_mem_ttl_seconds:
            return self._regions_cache

        async with self._regions_lock:
            now = time.time()
            if self._regions_cache and (now - self._regions_cache_loaded_at) < self.regions_mem_ttl_seconds:
                return self._regions_cache

            try:
                regions = await asyncio.to_thread(api_nayka.site_regions)
                if not isinstance(regions, list):
                    regions = []
            except Exception:
                regions = []

            self._regions_cache = regions
            self._regions_cache_loaded_at = time.time()
            return self._regions_cache

    async def _samara_region_tokens(self) -> set[str]:
        regions = await self._ensure_regions_loaded()
        tokens: set[str] = set()
        for r in regions:
            if not isinstance(r, dict):
                continue
            city = str(r.get("city") or "").strip()
            name = str(r.get("name") or "").strip()
            addr = str(r.get("addressForSite") or "").strip()
            if not (
                _is_samara_city_value(city)
                or "самара" in _normalise_input(name)
                or "самара" in _normalise_input(addr)
            ):
                continue
            for raw in (name, addr):
                n = _normalise_input(raw)
                if n:
                    tokens.add(n)
        return tokens

    async def _ensure_procedure_rows_loaded(self) -> list[dict[str, Any]]:
        """
        Готовит in-memory индекс процедур по филиалам из doctor_prices.

        Источник: `api_price.load_doctor_prices()` (кэш по branch-level regionId).
        Для защиты от мусора оставляем только самарские строки с реальными адресами.

        :return: строки вида {"serviceName": str, "regionName": str}
        """
        now = time.time()
        if self._procedure_rows_cache and (now - self._procedure_rows_loaded_at) < self.procedure_rows_mem_ttl_seconds:
            return self._procedure_rows_cache

        async with self._procedure_rows_lock:
            now = time.time()
            if self._procedure_rows_cache and (now - self._procedure_rows_loaded_at) < self.procedure_rows_mem_ttl_seconds:
                return self._procedure_rows_cache

            try:
                rows = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception:
                rows = []
            if not isinstance(rows, list):
                rows = []

            samara_tokens = await self._samara_region_tokens()
            filtered: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                service_name = str(row.get("serviceName") or "").strip()
                region_name = str(row.get("regionName") or "").strip()
                if not service_name or not region_name:
                    continue
                if not _looks_like_real_address(region_name):
                    continue
                if _is_explicit_non_samara_region(region_name):
                    continue
                if samara_tokens and not _region_matches_samara_tokens(region_name, samara_tokens):
                    continue
                filtered.append({"serviceName": service_name, "regionName": region_name})

            self._procedure_rows_cache = filtered
            self._procedure_rows_loaded_at = time.time()
            return self._procedure_rows_cache

    async def _ensure_service_catalog_rows_loaded(self) -> list[dict[str, Any]]:
        """
        Готовит объединенный каталог услуг клиники для exact/fuzzy матчинга.

        Источники:
        - city retail прайс `priceByRegion` (анализы и общие услуги);
        - doctor prices (doctorServicePricesByRegion) для услуг, которые бывают
          только в doctor-строках.
        """

        now = time.time()
        if self._service_catalog_rows_loaded_at and (now - self._service_catalog_rows_loaded_at) < self.service_catalog_mem_ttl_seconds:
            return self._service_catalog_rows_cache

        async with self._service_catalog_rows_lock:
            now = time.time()
            if self._service_catalog_rows_loaded_at and (now - self._service_catalog_rows_loaded_at) < self.service_catalog_mem_ttl_seconds:
                return self._service_catalog_rows_cache

            merged_rows: list[dict[str, Any]] = []
            source_errors: list[str] = []
            try:
                retail_rows = await asyncio.to_thread(api_price.load_price_by_region, SAMARA_PRICE_REGION_ID)
            except Exception as exc:
                source_errors.append(f"price_by_region:{type(exc).__name__}")
                retail_rows = []
            try:
                doctor_rows = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception as exc:
                source_errors.append(f"doctor_prices:{type(exc).__name__}")
                doctor_rows = []

            for src in (retail_rows, doctor_rows):
                if not isinstance(src, list):
                    continue
                for row in src:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("serviceName") or row.get("name") or "").strip()
                    if len(name) < 3:
                        continue
                    merged_rows.append(
                        {
                            "serviceName": name,
                            "name": name,
                            "serviceHomecode": str(row.get("serviceHomecode") or row.get("homecode") or "").strip(),
                            "cost": _as_int(row.get("cost")) or 0,
                        }
                    )

            deduped: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for row in merged_rows:
                name_norm = _normalise_catalog_text(str(row.get("serviceName") or row.get("name") or ""))
                if not name_norm:
                    continue
                code_norm = _normalise_input(str(row.get("serviceHomecode") or ""))
                key = (name_norm, code_norm)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)

            self._service_catalog_rows_cache = deduped
            self._service_catalog_rows_loaded_at = time.time()
            self._service_catalog_last_error = ";".join(source_errors)
            self._service_catalog_last_source_counts = {
                "retail_rows": len(retail_rows) if isinstance(retail_rows, list) else 0,
                "doctor_rows": len(doctor_rows) if isinstance(doctor_rows, list) else 0,
                "catalog_rows": len(deduped),
            }
            return self._service_catalog_rows_cache

    async def get_catalog_health(self) -> dict[str, Any]:
        """
        Централизованный health-check каталога для роутера.

        Возвращает сводный статус каталогов, которые используются для
        service/doctor grounding в мессенджерном контуре.
        """

        service_rows = await self._ensure_service_catalog_rows_loaded()
        doctors = await self._ensure_doctors_cache_loaded()
        service_ok = len(service_rows) > 0
        doctors_ok = len(doctors) > 0
        ok = service_ok and doctors_ok

        reasons: list[str] = []
        if not service_ok:
            reasons.append(self._service_catalog_last_error or "service_catalog_empty")
        if not doctors_ok:
            reasons.append("doctors_catalog_empty")

        return {
            "ok": ok,
            "status": "ok" if ok else "degraded",
            "service_catalog_ok": service_ok,
            "doctors_catalog_ok": doctors_ok,
            "reason": ";".join(reasons),
            "checked_at": int(time.time()),
            "source_counts": dict(self._service_catalog_last_source_counts or {}),
        }

    # -----------------------------
    # NAUKA API used by router
    # -----------------------------

    def get_branches(self) -> list[dict[str, str]]:
        """
        Возвращает справочник филиалов.
        Формат:
          [{"id":"branch_1","name":"Филиал на Проспекте Ленина","aliases":"Ленина, Ленинская,Проспект Ленина 5"}]
        Пока заглушка.
        """
        regions = self._regions_cache or []
        if not regions:
            try:
                data = api_nayka.site_regions()
                if isinstance(data, list):
                    regions = data
                    self._regions_cache = data
                    self._regions_cache_loaded_at = time.time()
            except Exception:
                regions = []
        out: list[dict[str, str]] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            if not (
                _is_samara_city_value(str(r.get("city") or ""))
                or "самара" in _normalise_input(str(r.get("name") or ""))
                or "самара" in _normalise_input(str(r.get("addressForSite") or ""))
            ):
                continue
            rid = r.get("id")
            disp = _region_display_name(r)
            if not disp:
                continue
            if not _looks_like_real_address(disp):
                continue
            bid = f"branch_{rid}" if rid is not None else f"branch_{len(out) + 1}"
            aliases = ", ".join(
                [
                    _normalise_input(disp),
                    _normalise_input(str(r.get("name") or "")),
                    _normalise_input(str(r.get("city") or "")),
                ]
            )
            out.append({"id": bid, "name": disp, "aliases": aliases})

        if out:
            # убираем дубли по имени
            uniq_by_name: dict[str, dict[str, str]] = {}
            for b in out:
                uniq_by_name.setdefault(b["name"], b)
            return list(uniq_by_name.values())

        return []


from .services.doctors import (  # noqa: E402
    _doctor_availability_snapshot as _doctor_availability_snapshot_impl,
    _resolve_doctor_id_from_name as _resolve_doctor_id_from_name_impl,
    _schedule_by_specialty as _schedule_by_specialty_impl,
    doctors_info as _doctors_info_impl,
    doctors_schedule_week as _doctors_schedule_week_impl,
    match_catalog_doctor as _match_catalog_doctor_impl,
    match_catalog_service as _match_catalog_service_impl,
    resolve_doctor_name as _resolve_doctor_name_impl,
)

Services.match_catalog_doctor = _match_catalog_doctor_impl
Services.match_catalog_service = _match_catalog_service_impl
Services._schedule_by_specialty = _schedule_by_specialty_impl
Services._doctor_availability_snapshot = _doctor_availability_snapshot_impl
Services.resolve_doctor_name = _resolve_doctor_name_impl
Services._resolve_doctor_id_from_name = _resolve_doctor_id_from_name_impl
Services.doctors_info = _doctors_info_impl
Services.doctors_schedule_week = _doctors_schedule_week_impl

from .services.prices import (  # noqa: E402
    price_info as _price_info_impl,
    service_bundle_info as _service_bundle_info_impl,
)
Services.price_info = _price_info_impl
Services.service_bundle_info = _service_bundle_info_impl

from .services.addresses import (  # noqa: E402
    _procedure_branches_from_index as _procedure_branches_from_index_impl,
    address_info as _address_info_impl,
)
Services.address_info = _address_info_impl
Services._procedure_branches_from_index = _procedure_branches_from_index_impl

from .services.appointments import appointment_help as _appointment_help_impl  # noqa: E402
Services.appointment_help = _appointment_help_impl

from .services.main_index import main_index_info as _main_index_info_impl  # noqa: E402
Services.main_index_info = _main_index_info_impl

from .services.news import news_info as _news_info_impl  # noqa: E402
Services.news_info = _news_info_impl

from .services.lab_tests import (  # noqa: E402
    test_assist as _test_assist_impl,
    test_result_status as _test_result_status_impl,
)
Services.test_assist = _test_assist_impl
Services.test_result_status = _test_result_status_impl

from .services.prepare import (  # noqa: E402
    _maybe_compact_prepare_text as _maybe_compact_prepare_text_impl,
    _pick_prepare_candidate as _pick_prepare_candidate_impl,
    _prepare_candidates_from_analysis_api_cache as _prepare_candidates_from_analysis_api_cache_impl,
    _prepare_from_analysis_api_cache as _prepare_from_analysis_api_cache_impl,
    _prepare_llm_validate_candidate as _prepare_llm_validate_candidate_impl,
    test_prepare as _test_prepare_impl,
)

Services._prepare_llm_validate_candidate = _prepare_llm_validate_candidate_impl
Services._pick_prepare_candidate = _pick_prepare_candidate_impl
Services._prepare_candidates_from_analysis_api_cache = _prepare_candidates_from_analysis_api_cache_impl
Services._maybe_compact_prepare_text = _maybe_compact_prepare_text_impl
Services.test_prepare = _test_prepare_impl
Services._prepare_from_analysis_api_cache = _prepare_from_analysis_api_cache_impl


if __name__ == "__main__":
    async def main():
        s = Services()
        print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
        print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

    asyncio.run(main())
