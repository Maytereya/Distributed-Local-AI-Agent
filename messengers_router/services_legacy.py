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
import re
import time
from datetime import datetime  # noqa: F401 — re-exported for services/doctors.py via legacy.datetime
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info
from converters import html_cleaner
from schedule_ttl_cache import AsyncListTTLStaleCache

# Re-exported for domain modules (services/doctors.py etc.) that access these via
# ``_legacy_module().<name>``. Ruff cannot see the dynamic access, hence noqa.
from .doctor_name_port import (  # noqa: F401
    extract_doctor_name_candidate,
    resolve_schedule_surname,
    surname_variants,
)
from .llm_doesnt_work_fallback import build_prepare_fallback_answer
from .llm_runtime import generate_text
from .policies import handoff_message
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
    _STATIC_PROCEDURE_BRANCH_OVERRIDES,
    _addresses_to_branch_payload,
    _extract_homecode_query,
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
    _SCHEDULE_SPECIALTY_TOKENS,
    _SERVICE_FILTER_STOPWORDS,
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

_SCHEDULE_QUERY_RE = re.compile(r"\b(расписани\w*|график|когда\b.*\bпринима\w*|принима\w*)\b", re.I)
_DOCTOR_PRICE_HINT_RE = re.compile(r"\b(?:у|врач\w*|доктор\w*)\s+[а-яё\-]{3,}\b", re.I)
_SERVICE_QUERY_SIGNAL_RE = re.compile(
    r"\b(услуг\w*|процедур\w*|исследован\w*|анализ\w*|сда[тч]\w*|"
    r"сдела\w*|провед\w*|провод\w*|выполня\w*|дела\w*|удали\w*|удалени\w*|"
    r"узи|экг|мрт|кт|фгдс|фкс|эндоскоп\w*|гастроскоп\w*|"
    r"кольпоскоп\w*|колоноскоп\w*|рентген\w*|холтер\w*)\b",
    re.I,
)
_PROCEDURE_BRANCH_LOOKUP_RE = re.compile(
    r"\b(где|сдела\w*|пройти|провест\w*|выполня\w*|дела\w*|можно|пройти\s+диагностик\w*)\b",
    re.I,
)


def _is_procedure_branch_lookup_query(query_text: str, service_q: str) -> bool:
    """
    Определяет, что пользователь ищет филиал под конкретную процедуру.

    :param query_text: исходный текст запроса
    :param service_q: нормализованная процедура/услуга
    :return: True, если это адресный lookup по процедуре
    """
    if not str(service_q or "").strip():
        return False
    q = _normalise_input(query_text or "")
    if not q:
        return False
    return bool(_PROCEDURE_BRANCH_LOOKUP_RE.search(q))


async def _resolve_ambiguous_price_kind_with_llm(
    query_text: str,
    retail_rows: list[dict[str, Any]],
    *,
    has_exact_doctor_link: bool,
    runtime_llm_mode: str = "",
) -> str:
    """
    Запускает LLM fallback только для ambiguous PRICE-кейсов.

    :param query_text: исходный пользовательский запрос
    :param retail_rows: top retail rows
    :param has_exact_doctor_link: найден ли надежный doctor linkage
    :param runtime_llm_mode: текущий llm_mode (`strict|hybrid|rich`)
    :return: выбранный kind либо `ambiguous`
    """

    mode = str(runtime_llm_mode or "").strip().lower()
    if mode not in {"hybrid", "rich"}:
        return "ambiguous"
    prompt = _build_price_kind_ambiguous_prompt(
        query_text,
        retail_rows,
        has_exact_doctor_link=has_exact_doctor_link,
    )
    if not prompt:
        return "ambiguous"
    try:
        raw = await generate_text(
            prompt,
            timeout_s=20,
            queue_timeout_ms=4000,
            fmt="json",
            think=False,
        )
    except Exception:
        return "ambiguous"
    kind = _parse_price_kind_ambiguous_result(raw)
    if not kind:
        return "ambiguous"
    if kind == "procedure_with_doctor" and not has_exact_doctor_link:
        return "ambiguous"
    return kind


_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT = handoff_message("knowledge_not_found")


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

    async def match_catalog_service(
        self,
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

    async def _procedure_branches_from_index(
        self,
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

        rows = await self._ensure_procedure_rows_loaded()
        ranked = _rank_price_rows(rows, service_q, limit=200)

        addresses: list[str] = []
        for row in ranked:
            addr = str(row.get("regionName") or "").strip()
            if not addr or not _looks_like_real_address(addr):
                continue
            if addr not in addresses:
                addresses.append(addr)

        if not addresses:
            addresses = _static_procedure_addresses(service_q)
        return _addresses_to_branch_payload(addresses, regions)

    async def service_bundle_info(
        self,
        query: str,
        entities: dict[str, Any],
        *,
        top_n: int | None = None,
    ) -> dict[str, Any]:
        top_limit = _coerce_top_n(top_n, default=DOCTORS_TOP_N)
        entity_service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        query_text = str(query or "").strip()
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
                cost = _as_int(row.get("cost")) or 0
                item = (score, matched, -cost, doctor_id, row)
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

            out_doctors: list[dict[str, Any]] = []
            query_specialty = _extract_specialty_from_text(query_text) or _extract_specialty_from_text(service_name)
            for doc in doctor_cards:
                doctor_id = _as_int(doc.get("id"))
                if doctor_id is None:
                    continue
                if service_kind == "doctor_consult" and query_specialty and not _doctor_matches_primary_specialty(doc, query_specialty):
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
        show_prepare = _is_prepare_requested_in_price_query(query_text)
        out["show_prepare"] = show_prepare
        if show_prepare and not is_consult_query:
            prepare_payload = await self.test_prepare(service_name, {"service_name": service_name})
            if isinstance(prepare_payload, dict) and not prepare_payload.get("handoff_required"):
                out["prepare"] = str(prepare_payload.get("prepare") or "").strip()

        return out

    # -----------------------------
    # NAUKA API used by router
    # -----------------------------

    async def main_index_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {
                "content": "",
                "note": "main_index_info: no query",
                "entities_used": entities,
            }
        doc_kind = str(entities.get("doc_request_kind") or "").strip().lower()
        if doc_kind not in {"tax", "generic"}:
            norm_q = _normalise_input(q)
            doc_kind = "tax" if any(k in norm_q for k in ("налог", "вычет", "фнс")) else "generic"

        if doc_kind == "tax":
            return _tax_doc_guidance_response(entities, note="main_index_info: tax direct link")

        normalized_q = _normalise_input(q)
        fallback_queries: list[str] = []
        if doc_kind == "tax" and any(k in normalized_q for k in ("налог", "фнс", "вычет", "справк")):
            fallback_queries = [
                "справка для налоговой",
                "налоговый вычет",
                "справка об оплате медицинских услуг",
            ]

        queries = [q]
        for fq in fallback_queries:
            if _normalise_input(fq) != normalized_q:
                queries.append(fq)

        cleaned = ""
        relevant_hit = False
        try:
            for qq in queries:
                raw = await asyncio.to_thread(
                    meilisearch.search_meili,
                    "main_index",
                    qq,
                    output_mode="content_only",
                    max_chars=12000,
                )
                cleaned = html_cleaner.strip_html(raw).strip()
                if _is_meili_error_text(cleaned):
                    continue
                if _is_meili_no_matches_text(cleaned):
                    continue
                if _is_main_index_relevant(qq, cleaned, doc_kind=doc_kind):
                    relevant_hit = True
                    break
        except Exception:
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback unavailable")
            return _service_fallback(
                note="main_index_info source unavailable",
                handoff_message=handoff_message("service_error_doctor_info"),
                entities=entities,
                extra={"content": ""},
            )

        if _is_meili_error_text(cleaned):
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback error")
            return _service_fallback(
                note="main_index_info source unavailable",
                handoff_message=handoff_message("service_error_doctor_info"),
                entities=entities,
                extra={"content": ""},
            )

        if _is_meili_no_matches_text(cleaned):
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback no matches")
            return _service_fallback(
                note="main_index_info: no matches",
                handoff_message=_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT,
                entities=entities,
                reason="knowledge_not_found",
                extra={"content": ""},
            )

        if not relevant_hit:
            if doc_kind == "tax":
                return _tax_doc_guidance_response(entities, note="main_index_info: tax fallback weak relevance")
            return _service_fallback(
                note=f"main_index_info: weak relevance ({doc_kind})",
                handoff_message=_KNOWLEDGE_NOT_FOUND_HANDOFF_TEXT,
                entities=entities,
                reason="knowledge_not_found",
                extra={"content": ""},
            )

        return {
            "content": cleaned,
            "note": f"main_index_info: main_index ({doc_kind})",
            "entities_used": entities,
        }

    async def appointment_help(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        if query:
            try:
                raw = await asyncio.to_thread(
                    meilisearch.search_meili,
                    "main_index",
                    query,
                    output_mode="content_only",
                    max_chars=12000,
                )
                cleaned = html_cleaner.strip_html(raw)
            except Exception:
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message=handoff_message("service_error_appointments"),
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            if _is_meili_error_text(cleaned):
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message=handoff_message("service_error_appointments"),
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            return {"instructions": cleaned, "entities_used": entities}
        return {
            "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
            "entities_used": entities,
        }

    async def _prepare_llm_validate_candidate(
        self,
        query: str,
        candidate: _PrepareCandidate,
    ) -> tuple[bool, float, str]:
        """
        LLM-валидация релевантности для кандидата PREPARE в серой зоне score.

        :param query: запрос пользователя
        :param candidate: кандидат из API/Meili
        :return: (релевантно, confidence, reason)
        """

        if not _runtime_bool("MR_PREPARE_RELEVANCE_LLM_ENABLED", True):
            return False, 0.0, "llm_disabled"

        prompt = _prepare_relevance_prompt(
            query,
            candidate.text,
            source_kind=candidate.source,
            service_title=candidate.service_title,
        )
        if not prompt:
            return False, 0.0, "empty_prompt"

        timeout_s = _runtime_int(
            "MR_PREPARE_RELEVANCE_LLM_TIMEOUT_S",
            15,
            min_value=1,
            max_value=60,
        )
        queue_timeout_ms = _runtime_int(
            "MR_PREPARE_RELEVANCE_LLM_QUEUE_TIMEOUT_MS",
            3000,
            min_value=200,
            max_value=20000,
        )
        try:
            raw = await generate_text(
                prompt,
                timeout_s=timeout_s,
                queue_timeout_ms=queue_timeout_ms,
                fmt="json",
                think=False,
            )
        except Exception as e:
            logger.info("prepare relevance llm skipped: %s", e.__class__.__name__)
            return False, 0.0, "llm_unavailable"

        return _parse_prepare_relevance_validator(str(raw or ""))

    async def _pick_prepare_candidate(
        self,
        query: str,
        candidates: list[_PrepareCandidate],
    ) -> _PrepareCandidate | None:
        """
        Выбирает лучший кандидат PREPARE по fast-score + LLM в серой зоне.

        :param query: запрос пользователя
        :param candidates: кандидаты из источников
        :return: лучший релевантный кандидат или None
        """

        ranked = _dedupe_prepare_candidates(candidates, limit=12)
        if not ranked:
            return None

        max_llm_checks = _runtime_int(
            "MR_PREPARE_RELEVANCE_LLM_MAX_CHECKS",
            2,
            min_value=1,
            max_value=8,
        )
        llm_checks = 0
        query_roots = _prepare_term_roots(query)
        for idx, cand in enumerate(ranked):
            next_score = ranked[idx + 1].score if idx + 1 < len(ranked) else 0.0
            margin = max(0.0, float(cand.score) - float(next_score))
            cand.margin = margin
            gate = _prepare_relevance_gate(cand.score, cand.margin)
            if gate == "accept":
                cand.note = (cand.note + "; " if cand.note else "") + "prepare_fast_gate=accept"
                return cand
            if gate == "reject":
                cand.note = (cand.note + "; " if cand.note else "") + "prepare_fast_gate=reject"
                continue

            if cand.source == "serviceInfoAll" and query_roots:
                title_roots = _prepare_term_roots(cand.service_title)
                title_cov = _prepare_roots_coverage(query_roots, title_roots)
                if title_cov >= 0.99 and _has_prepare_strong_hints(cand.text):
                    cand.note = (
                        (cand.note + "; " if cand.note else "")
                        + "prepare_fast_gate=accept_service_title_anchor"
                    )
                    return cand

            if llm_checks >= max_llm_checks:
                cand.note = (cand.note + "; " if cand.note else "") + "prepare_llm_skipped=max_checks"
                continue

            llm_checks += 1
            ok, confidence, reason = await self._prepare_llm_validate_candidate(query, cand)
            cand.note = (
                (cand.note + "; " if cand.note else "")
                + f"prepare_llm={_PREPARE_RELEVANCE_VERDICT_RELEVANT if ok else _PREPARE_RELEVANCE_VERDICT_IRRELEVANT}"
                + f"({confidence:.2f})"
                + (f":{reason}" if reason else "")
            )
            if ok:
                return cand

            # Quality-first override for Meili mixed-docs:
            # если LLM отверг из-за "смешанности", но у кандидата высокий fast-score,
            # полное покрытие корней запроса и actionable-текст, пропускаем в wrapper.
            if cand.source == "main_index" and query_roots:
                body_roots = _prepare_term_roots(cand.text)
                body_cov = _prepare_roots_coverage(query_roots, body_roots)
                override_score = _runtime_float(
                    "MR_PREPARE_MAIN_INDEX_OVERRIDE_SCORE",
                    0.70,
                    min_value=0.30,
                    max_value=0.95,
                )
                if (
                    cand.score >= override_score
                    and body_cov >= 0.99
                    and _is_prepare_content_actionable(cand.text)
                ):
                    cand.note = (
                        (cand.note + "; " if cand.note else "")
                        + "prepare_llm_reject_override_main_index"
                    )
                    return cand

            if reason in {"llm_unavailable", "llm_non_json", "llm_disabled", "empty_prompt"}:
                fallback_score = _runtime_float(
                    "MR_PREPARE_RELEVANCE_LLM_UNAVAILABLE_ACCEPT_SCORE",
                    0.40,
                    min_value=0.10,
                    max_value=0.95,
                )
                if cand.score >= fallback_score:
                    cand.note = (cand.note + "; " if cand.note else "") + "prepare_llm_fallback_fast_accept"
                    return cand
        return None

    async def _prepare_candidates_from_analysis_api_cache(
        self,
        query: str,
        entities: dict[str, Any],
    ) -> list[_PrepareCandidate]:
        """
        Возвращает отсортированные prepare-кандидаты из serviceInfoAll.

        :param query: исходный запрос пользователя
        :param entities: сущности роутера
        :return: список кандидатов
        """

        entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
        if not _prepare_term_roots(" ".join(x for x in (query, entity_query) if x)):
            return []
        queries = _prepare_service_info_queries(query, entity_query)
        if not queries:
            return []

        try:
            rows = await asyncio.to_thread(api_service_info.load_service_info)
        except Exception:
            return []

        candidates: list[_PrepareCandidate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            service_name = str(row.get("serviceName") or "").strip()
            preparation = str(row.get("preparation") or "").strip()
            if not service_name or not preparation:
                continue

            best_score = 0.0
            best_query = ""
            for query_variant in queries:
                score = _prepare_fast_relevance_score(query_variant, preparation, title=service_name)
                if score > best_score:
                    best_score = score
                    best_query = query_variant

            if best_score <= 0.0:
                continue

            candidates.append(
                _PrepareCandidate(
                    source="serviceInfoAll",
                    text=preparation,
                    query_variant=best_query or query,
                    service_title=service_name,
                    score=best_score,
                    note="prepare: serviceInfoAll candidate",
                )
            )

        return _dedupe_prepare_candidates(candidates, limit=16)

    async def _maybe_compact_prepare_text(self, query: str, source_text: str) -> tuple[str, str, str]:
        """
        Компактирует длинный PREPARE-текст через LLM с безопасным fallback.

        :param query: исходный запрос пациента
        :param source_text: текст подготовки из источника
        :return: (итоговый текст, статус wrap, причина/диагностика)
        """

        text = str(source_text or "").strip()
        if not text:
            return "", "empty_source", "no_source_text"
        if not _runtime_bool("MR_PREPARE_LLM_WRAP_ENABLED", True):
            return text, "disabled", "llm_wrap_disabled"

        min_chars = _runtime_int(
            "MR_PREPARE_LLM_WRAP_MIN_CHARS",
            700,
            min_value=120,
            max_value=12000,
        )
        if len(text) < min_chars:
            return text, "short_source", "below_min_chars"

        source_max_chars = _runtime_int(
            "MR_PREPARE_LLM_WRAP_SOURCE_MAX_CHARS",
            9000,
            min_value=500,
            max_value=30000,
        )
        source_for_prompt = text[:source_max_chars].strip()

        def _fallback_or_source(reason: str) -> tuple[str, str, str]:
            compacted = build_prepare_fallback_answer(
                query,
                source_for_prompt,
                max_chars=_runtime_int(
                    "MR_PREPARE_FALLBACK_MAX_CHARS",
                    1600,
                    min_value=400,
                    max_value=4000,
                ),
                max_points=_runtime_int(
                    "MR_PREPARE_FALLBACK_MAX_POINTS",
                    7,
                    min_value=3,
                    max_value=10,
                ),
            )
            if compacted and len(compacted) < len(text):
                return compacted, "fallback_compact", reason
            if compacted:
                return text, "fallback_not_shorter", reason
            return text, "fallback_failed", reason

        prompt = _prepare_wrap_prompt(query, source_for_prompt)
        if not prompt:
            return _fallback_or_source("empty_prompt")

        timeout_s = _runtime_int(
            "MR_PREPARE_LLM_WRAP_TIMEOUT_S",
            30,
            min_value=3,
            max_value=90,
        )
        queue_timeout_ms = _runtime_int(
            "MR_PREPARE_LLM_WRAP_QUEUE_TIMEOUT_MS",
            6000,
            min_value=300,
            max_value=30000,
        )
        try:
            raw = await generate_text(
                prompt,
                timeout_s=timeout_s,
                queue_timeout_ms=queue_timeout_ms,
                think=False,
            )
        except Exception as e:
            logger.info("prepare llm wrap skipped: %s", e.__class__.__name__)
            return _fallback_or_source(f"llm_wrap_error:{e.__class__.__name__}")

        wrapped = _prepare_wrap_clean(str(raw or ""))
        if not _is_prepare_wrap_output_usable(query, source_for_prompt, wrapped):
            return _fallback_or_source("llm_wrap_invalid_output")
        return wrapped, "llm_wrapped", "ok"

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        raw_query = str(query or "").strip()
        entity_query = _get_first_present(entities, ["test_name", "service_name"]) or ""
        # Для нового вопроса берем текст пользователя, чтобы не залипала старая услуга из контекста.
        q = raw_query or entity_query
        if not q:
            return {"prepare": "", "note": "no query", "entities_used": entities}

        api_candidates = await self._prepare_candidates_from_analysis_api_cache(q, entities)
        api_best = await self._pick_prepare_candidate(q, api_candidates)
        if api_best:
            api_cached_cleaned = html_cleaner.strip_html(api_best.text).strip()
            if not _is_prepare_service_info_usable(q, api_cached_cleaned, title=api_best.service_title):
                api_best = None
            else:
                api_best.text = api_cached_cleaned
        if api_best:
            compacted, wrap_status, wrap_reason = await self._maybe_compact_prepare_text(q, api_best.text)
            return {
                "prepare": compacted,
                "note": "prepare: serviceInfoAll",
                "entities_used": entities,
                "prepare_wrap_status": wrap_status,
                "prepare_wrap_reason": wrap_reason,
            }

        variants = _prepare_query_variants(q, entity_query)
        if not variants:
            variants = [q]

        meili_candidates: list[_PrepareCandidate] = []
        saw_no_matches = False
        saw_non_empty = False
        saw_service_error = False
        for candidate in variants:
            try:
                raw = await asyncio.to_thread(
                    meilisearch.search_meili,
                    "main_index",
                    candidate,
                    output_mode="content_only",
                    max_chars=12000,
                )
                cleaned = html_cleaner.strip_html(raw).strip()
            except Exception:
                saw_service_error = True
                continue

            if _is_meili_error_text(cleaned):
                saw_service_error = True
                continue

            if _is_meili_no_matches_text(cleaned):
                saw_no_matches = True
                continue

            if not cleaned:
                continue

            saw_non_empty = True
            score = max(
                _prepare_fast_relevance_score(q, cleaned),
                _prepare_fast_relevance_score(candidate, cleaned),
            )
            if score <= 0.0:
                continue
            meili_candidates.append(
                _PrepareCandidate(
                    source="main_index",
                    text=cleaned,
                    query_variant=candidate,
                    score=score,
                    note="prepare: main_index candidate",
                )
            )

        meili_best = await self._pick_prepare_candidate(q, meili_candidates)
        if meili_best:
            compacted, wrap_status, wrap_reason = await self._maybe_compact_prepare_text(q, meili_best.text)
            return {
                "prepare": compacted,
                "note": "prepare: main_index",
                "entities_used": entities,
                "prepare_wrap_status": wrap_status,
                "prepare_wrap_reason": wrap_reason,
            }

        if saw_non_empty:
            return _prepare_clarify_response(q, entities, note="prepare: weak relevance")

        if saw_no_matches or not saw_service_error:
            return _prepare_clarify_response(q, entities, note="prepare: no matches")

        return _prepare_clarify_response(q, entities, note="prepare source unavailable")

    async def _prepare_from_analysis_api_cache(self, query: str, entities: dict[str, Any]) -> str | None:
        """
        Ищет подготовку к анализу в кэше `serviceInfoAll`.

        :param query: текст запроса пользователя
        :param entities: текущие сущности роутера
        :return: текст поля `preparation` или None
        """

        candidates = await self._prepare_candidates_from_analysis_api_cache(query, entities)
        best = await self._pick_prepare_candidate(query, candidates)
        if not best:
            return None
        return str(best.text or "").strip() or None

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        try:
            hits = await asyncio.to_thread(
                meilisearch.search_news_active,
                index_name="news",
                keyword=query or None,
                limit=10,
                sort=["from_ts:desc"],
            )
        except Exception:
            # Для новостей деградация источника не критична: возвращаем пустой ответ
            # без принудительного handoff.
            return {
                "news": [],
                "note": "news source unavailable",
                "entities_used": entities,
            }
        return {"news": hits, "entities_used": entities}

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
    resolve_doctor_name as _resolve_doctor_name_impl,
)

Services.match_catalog_doctor = _match_catalog_doctor_impl
Services._schedule_by_specialty = _schedule_by_specialty_impl
Services._doctor_availability_snapshot = _doctor_availability_snapshot_impl
Services.resolve_doctor_name = _resolve_doctor_name_impl
Services._resolve_doctor_id_from_name = _resolve_doctor_id_from_name_impl
Services.doctors_info = _doctors_info_impl
Services.doctors_schedule_week = _doctors_schedule_week_impl

from .services.prices import price_info as _price_info_impl  # noqa: E402
Services.price_info = _price_info_impl

from .services.addresses import address_info as _address_info_impl  # noqa: E402
Services.address_info = _address_info_impl

from .services.lab_tests import (  # noqa: E402
    test_assist as _test_assist_impl,
    test_result_status as _test_result_status_impl,
)
Services.test_assist = _test_assist_impl
Services.test_result_status = _test_result_status_impl


if __name__ == "__main__":
    async def main():
        s = Services()
        print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
        print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

    asyncio.run(main())
