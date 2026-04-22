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
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch  # noqa: F401 — tests patch svc_mod.meilisearch
from agent_logic_2.nayka_api import api_nayka, api_price, api_service_info
from converters import html_cleaner  # noqa: F401 — tests patch svc_mod.html_cleaner
from schedule_ttl_cache import AsyncListTTLStaleCache

from ._common import (
    _as_int,
    _normalise_catalog_text,
    _normalise_input,
    _runtime_bool,
    _runtime_int,
)
from ._addresses_helpers import _looks_like_real_address
from ._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    match_compound_price_service_option,  # noqa: F401 — public API re-export for messengers_router.services
    resolve_price_service_name_from_catalog,  # noqa: F401 — public API re-export for messengers_router.services
)
from ._regions import (
    _is_explicit_non_samara_region,
    _is_samara_city_value,
    _region_display_name,
    _region_matches_samara_tokens,
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


from .doctors import (  # noqa: E402
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

from .prices import (  # noqa: E402
    price_info as _price_info_impl,
    service_bundle_info as _service_bundle_info_impl,
)
Services.price_info = _price_info_impl
Services.service_bundle_info = _service_bundle_info_impl

from .addresses import (  # noqa: E402
    _procedure_branches_from_index as _procedure_branches_from_index_impl,
    address_info as _address_info_impl,
)
Services.address_info = _address_info_impl
Services._procedure_branches_from_index = _procedure_branches_from_index_impl

from .appointments import appointment_help as _appointment_help_impl  # noqa: E402
Services.appointment_help = _appointment_help_impl

from .main_index import main_index_info as _main_index_info_impl  # noqa: E402
Services.main_index_info = _main_index_info_impl

from .news import news_info as _news_info_impl  # noqa: E402
Services.news_info = _news_info_impl

from .lab_tests import (  # noqa: E402
    test_assist as _test_assist_impl,
    test_result_status as _test_result_status_impl,
)
Services.test_assist = _test_assist_impl
Services.test_result_status = _test_result_status_impl

from .prepare import (  # noqa: E402
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
