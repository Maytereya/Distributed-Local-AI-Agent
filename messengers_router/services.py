"""Сервисный слой интеграций для роутера пациентов.

Содержит вызовы внешних источников (Nayka API, price, meili), кэш врачей,
поиск расписания/цен/адресов и fallback-контракты для handoff при сбоях.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.doctor_name_matching import resolve_schedule_surname, surname_variants
from agent_logic_2.nayka_api import api_nayka, api_price
from converters import html_cleaner

_ADDRESS_HINT_RE = re.compile(
    r"\b(ул\.?|улица|пр\.?|проспект|пр-?т|тракт|б-р|бульвар|шоссе|пер\.?|переулок|наб\.?|площадь|дом|д\.|корп\.?|к\.|пом\.?)\b",
    re.I,
)
_SCHEDULE_QUERY_RE = re.compile(r"\b(расписани\w*|график)\b", re.I)


def _normalise_input(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _get_first_present(d: dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _as_int(val: Any) -> int | None:
    try:
        return int(val)
    except Exception:
        return None


def _region_display_name(region: dict[str, Any]) -> str:
    """
    Берем человекочитаемый адрес из live /regions.
    Приоритет: addressForSite -> name.
    """
    addr = str(region.get("addressForSite") or "").strip()
    name = str(region.get("name") or "").strip()
    return addr or name


def _looks_like_real_address(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if re.fullmatch(r"ID\s+\d+", s, flags=re.I):
        return False
    if _ADDRESS_HINT_RE.search(s):
        return True
    if re.search(r"\d", s):
        return True
    return False


def _service_fallback(
    *,
    note: str,
    handoff_message: str,
    entities: dict[str, Any],
    reason: str = "service_error",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "note": note,
        "handoff_required": True,
        "handoff_reason": reason,
        "handoff_message": handoff_message,
        "entities_used": entities,
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _is_meili_error_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return "Ошибка поисковой системы" in text or "Meilisearch" in text


@dataclass
class Services:
    """
    Сервисный слой для patient/messenger router.

    - doctors_info: использует ФАЙЛОВЫЙ кэш врачей (JSONL) + in-memory кэш.
    - doctors_schedule_week: всегда ходит в API (реалтайм расписание).
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

    # блокировка, чтобы несколько запросов параллельно не перегенерировали кэш
    _doctors_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _regions_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # TTL in-memory кэша (латентность, сек, определят свежесть кэша)
    doctors_mem_ttl_seconds: int = 300
    regions_mem_ttl_seconds: int = 300

    # -----------------------------
    # Low-level helpers (async)
    # -----------------------------

    async def _ensure_doctors_cache_loaded(self) -> list[dict[str, Any]]:
        """
        1) Проверяем актуальный файл doctors_YYYYMMDD.jsonl
        2) Если файла нет — собираем через API и сохраняем
        3) Держим in-memory-кэш поверх файла
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
                # 1) ищем актуальный файл
                file_path = await asyncio.to_thread(api_nayka.find_existing_doctors_file)

                # 2) если нет — обновляем
                if file_path is None:
                    doctors = await asyncio.to_thread(api_nayka.get_all_doctors)
                    await asyncio.to_thread(api_nayka.save_doctors_data, doctors)
                    file_path = await asyncio.to_thread(api_nayka.find_existing_doctors_file)

                # 3) загружаем файл
                doctors_loaded: list[dict[str, Any]] = []
                if file_path is not None:
                    doctors_loaded = await asyncio.to_thread(api_nayka.load_doctors_data, file_path)
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

    # -----------------------------
    # NAUKA API used by router
    # -----------------------------

    async def doctors_info(self, query: str, entities: dict[str, Any], output_max: int = 5) -> dict[str, Any]:
        """
        Возвращает список врачей из кэша (без real-time API).
        Фильтрация делается программно:
        - по фамилии/ФИО
        - по специализации
        - по региону/филиалу (по строкам regions/units если есть)
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return _service_fallback(
                note="doctors_info source unavailable",
                handoff_message="Сейчас не удалось получить список врачей автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"doctors": []},
            )

        q = _normalise_input(query)
        fio_q = _normalise_input(
            _get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
        )
        spec_q = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
        region_q = _normalise_input(_get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")

        # если из entities пусто — попробуем хотя бы query как ключ
        # (но аккуратно: не хотим показывать всех врачей по любому вопросу)
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

        def match_doc(doc: dict[str, Any], ) -> bool:
            fio = _normalise_input(str(doc.get("fio", "")))
            spec = _normalise_input(str(doc.get("specialization", "")))
            regions = " ".join([_normalise_input(str(x)) for x in (doc.get("regions") or [])])
            units = " ".join([_normalise_input(str(x)) for x in (doc.get("units") or [])])

            hay = " | ".join([fio, spec, regions, units])
            if fio_q and fio_q not in hay:
                return False
            if spec_q and spec_q not in hay:
                return False
            if region_q and region_q not in hay:
                return False

            # если ничего конкретного не задано — используем keyword, но требуем хотя бы 3 символа
            if not (fio_q or spec_q or region_q):
                if len(keyword) < 3:
                    return False
                return keyword in hay

            return True

        filtered = [d for d in doctors if match_doc(d)]

        # ограничим размер, чтобы не отправлять сотни карточек в LLM
        # (далее LLM/рендерер красиво завернёт)
        # Определить сколько тут карточек нужно в выводе обычно
        filtered = filtered[:output_max]

        return {
            "doctors": filtered,
            "note": "doctors_info: from cached registry (jsonl)",
            "cache_file": self._doctors_cache_path,
            "entities_used": {
                "doctor_query": fio_q,
                "specialty_query": spec_q,
                "region_query": region_q,
            },
        }

    async def doctors_schedule_week(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Реалтайм расписание на неделю.

        Используем high-level функцию api_nayka.find_doctor_schedule(last_name, region_name=None)
        Она сама ходит в:
        - /doctors, /regions, /doctorRegions, /doctorCompanyUnits
        - /doctorSchedule + /doctorScheduleCells

        Возвращаем как есть (агрегированный список).
        """
        raw_name = _get_first_present(
            entities,
            ["last_name", "doctor_last_name", "doctor", "doctor_name", "fio"],
        )
        if not raw_name:
            raw_name = query

        doctors = await self._ensure_doctors_cache_loaded()
        raw_for_match = str(raw_name or "").strip()
        # Если прилетело полное ФИО, для расписания берем фамилию (1-е слово),
        # иначе fuzzy-резолвер может схватить отчество и вернуть ложные матчи.
        if " " in raw_for_match:
            first = raw_for_match.split()[0].strip()
            raw_for_match = first or raw_for_match

        query_name = resolve_schedule_surname(str(query or ""), doctors) if query else None
        last_name = resolve_schedule_surname(raw_for_match, doctors)
        # Если в текущей реплике явно фигурирует другая фамилия по расписанию,
        # приоритет отдаем ей (сброс от залипшего doctor_name из state).
        if query_name and (
            not last_name
            or _SCHEDULE_QUERY_RE.search(str(query or ""))
            and _normalise_input(str(query_name)) != _normalise_input(str(last_name))
        ):
            last_name = query_name
        elif not last_name and query and query != raw_name:
            last_name = query_name or resolve_schedule_surname(query, doctors)

        if not last_name:
            return {
                "schedule": [],
                "note": "doctors_schedule_week: missing doctor last name",
                "entities_used": entities,
            }

        # необязательный фильтр региона/филиала/города
        region_name = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city", "branch_name"])

        # api_nayka.find_doctor_schedule блокирующая (requests) — уводим в thread.
        # Пробуем несколько вариантов фамилии (родительный падеж -> именительный).
        data = None
        candidates = surname_variants(str(last_name))
        if not candidates:
            candidates = [str(last_name)]
        if query_name:
            for qv in surname_variants(str(query_name)):
                if qv not in candidates:
                    candidates.append(qv)
        try:
            for candidate in candidates:
                try:
                    data = await asyncio.to_thread(api_nayka.find_doctor_schedule, candidate, region_name)
                except TypeError:
                    data = await asyncio.to_thread(api_nayka.find_doctor_schedule, candidate)
                if isinstance(data, list) and data:
                    last_name = candidate
                    break
                # fallback: если регионный фильтр дал пусто, пробуем без региона
                if region_name:
                    try:
                        data = await asyncio.to_thread(api_nayka.find_doctor_schedule, candidate)
                    except Exception:
                        data = []
                    if isinstance(data, list) and data:
                        last_name = candidate
                        break
        except Exception:
            return _service_fallback(
                note="doctors_schedule_week unavailable",
                handoff_message="Сейчас не удалось получить расписание автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"schedule": []},
            )

        return {
            "schedule": data or [],
            "note": "doctors_schedule_week: realtime from Nayka API",
            "entities_used": {"last_name": last_name, "raw_name": raw_name, "region_name": region_name},
        }

    # Остальные методы пока как заглушки
    async def appointment_help(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        if query:
            try:
                raw = await asyncio.to_thread(meilisearch.search_meili, "main_index", query)
                cleaned = html_cleaner.strip_html(raw)
            except Exception:
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message="Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            if _is_meili_error_text(cleaned):
                return _service_fallback(
                    note="appointment_help source unavailable",
                    handoff_message="Сейчас не удалось получить данные для записи автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"instructions": "Сейчас не удалось получить данные для записи автоматически."},
                )
            return {"instructions": cleaned, "entities_used": entities}
        return {
            "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
            "entities_used": entities,
        }

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        test_name = _get_first_present(entities, ["test_name", "service_name"]) or query
        needle = _normalise_input(test_name)
        if not needle:
            return {"tests": [], "promos": [], "note": "no test query", "entities_used": entities}

        try:
            price_all = await asyncio.to_thread(api_price.load_price_all)
        except Exception:
            return _service_fallback(
                note="test_assist source unavailable",
                handoff_message="Сейчас не удалось подобрать анализы автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"tests": [], "promos": []},
            )
        matches = [p for p in price_all if needle in _normalise_input(p.get("serviceName"))][:10]

        return {"tests": matches, "promos": [], "entities_used": entities}

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        q = _get_first_present(entities, ["test_name", "service_name"]) or query
        if not q:
            return {"prepare": "", "note": "no query", "entities_used": entities}
        try:
            raw = await asyncio.to_thread(meilisearch.search_meili, "main_index", q)
            cleaned = html_cleaner.strip_html(raw)
        except Exception:
            return _service_fallback(
                note="prepare source unavailable",
                handoff_message="Сейчас не удалось получить правила подготовки автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prepare": ""},
            )
        if _is_meili_error_text(cleaned):
            return _service_fallback(
                note="prepare source unavailable",
                handoff_message="Сейчас не удалось получить правила подготовки автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prepare": ""},
            )
        return {"prepare": cleaned, "entities_used": entities}

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {
            "ready": False,
            "note": "no result API",
            "handoff_required": True,
            "handoff_reason": "test_result_fallback",
            "entities_used": entities,
        }

    async def test_result_pdf(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {
            "pdf": None,
            "note": "no result PDF service",
            "handoff_required": True,
            "handoff_reason": "test_result_fallback",
            "entities_used": entities,
        }

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        doctor_id = _as_int(entities.get("doctor_id"))
        service_name = _get_first_present(entities, ["service_name", "test_name"]) or query
        needle = _normalise_input(service_name)

        if doctor_id:
            try:
                prices = await asyncio.to_thread(api_price.load_doctor_prices)
            except Exception:
                return _service_fallback(
                    note="price_info source unavailable",
                    handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
                    entities=entities,
                    extra={"prices": []},
                )
            doc_prices = [p for p in prices if _as_int(p.get("doctorId")) == doctor_id]
            if needle:
                doc_prices = [p for p in doc_prices if needle in _normalise_input(p.get("serviceName"))]
            return {"prices": doc_prices[:10], "entities_used": entities}

        try:
            price_all = await asyncio.to_thread(api_price.load_price_all)
        except Exception:
            return _service_fallback(
                note="price_info source unavailable",
                handoff_message="Сейчас не удалось получить цены автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"prices": []},
            )
        if not needle:
            return {"prices": [], "note": "no service query", "entities_used": entities}
        matches = [p for p in price_all if needle in _normalise_input(p.get("serviceName"))][:10]
        return {"prices": matches, "entities_used": entities}

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        try:
            regions = await self._ensure_regions_loaded()
        except Exception:
            regions = []
        branch = _get_first_present(entities, ["region", "branch", "company_unit", "unit", "city"]) or query
        branch_q = _normalise_input(branch)
        service_name = _get_first_present(entities, ["service_name", "test_name"]) or ""
        service_q = _normalise_input(service_name)

        allowed_region_ids: set[int] | None = None
        if service_q:
            try:
                doctor_prices = await asyncio.to_thread(api_price.load_doctor_prices)
                matched_region_ids: set[int] = set()
                for row in doctor_prices:
                    if not isinstance(row, dict):
                        continue
                    svc = _normalise_input(str(row.get("serviceName") or ""))
                    if not svc or service_q not in svc:
                        continue
                    rid = _as_int(row.get("regionId"))
                    if rid is not None:
                        matched_region_ids.add(rid)
                if matched_region_ids:
                    allowed_region_ids = matched_region_ids
            except Exception:
                allowed_region_ids = None

        addresses: list[str] = []
        for r in regions:
            if not isinstance(r, dict):
                continue
            rid = _as_int(r.get("id"))
            if allowed_region_ids is not None and (rid is None or rid not in allowed_region_ids):
                continue
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

        uniq = sorted(set(addresses))
        if uniq:
            note = "address_info: live regions API"
            if service_q and allowed_region_ids is not None:
                note += " + filtered by service"
            return {
                "addresses": uniq,
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
            "note": "address_info: doctors cache fallback",
            "entities_used": entities,
        }

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
            return _service_fallback(
                note="news source unavailable",
                handoff_message="Сейчас не удалось получить новости автоматически. Соединяю с оператором.",
                entities=entities,
                extra={"news": []},
            )
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

        return [
            {"id": "branch_novo-sadovaya", "name": "Филиал на Ново - Садовой",
             "aliases": "ново - садовая, ул ново-садовая"},
            {"id": "branch_lenina", "name": "Филиал на Ленина", "aliases": "ленина,ул ленина,ленина 5"},
        ]


if __name__ == "__main__":
    async def main():
        s = Services()
        print(await s.doctors_info("уролог Дразнин", {"specialty": "уролог", "last_name": "Дразнин"}))
        print(await s.doctors_schedule_week("покажи расписание Дразнина", {"last_name": "Дразнин"}))

    asyncio.run(main())
