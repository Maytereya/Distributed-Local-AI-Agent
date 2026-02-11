"""Сервисный слой интеграций для роутера пациентов.

Содержит вызовы внешних источников (Nayka API, price, meili), кэш врачей,
поиск расписания/цен/адресов и fallback-контракты для handoff при сбоях.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_logic_1 import meilisearch_client as meilisearch
from agent_logic_2.doctor_name_matching import (
    extract_doctor_name_candidate,
    resolve_schedule_surname,
    surname_variants,
)
from agent_logic_2.nayka_api import api_nayka, api_price
from converters import html_cleaner

_ADDRESS_HINT_RE = re.compile(
    r"\b(ул\.?|улица|пр\.?|проспект|пр-?т|тракт|б-р|бульвар|шоссе|пер\.?|переулок|наб\.?|площадь|дом|д\.|корп\.?|к\.|пом\.?)\b",
    re.I,
)
_SCHEDULE_QUERY_RE = re.compile(r"\b(расписани\w*|график|когда\b.*\bпринима\w*|принима\w*)\b", re.I)
_FIO_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]{2,}")


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


def _stringify_json_preview(value: Any, max_len: int = 2500) -> str:
    try:
        txt = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        txt = str(value)
    if len(txt) > max_len:
        return txt[:max_len] + "\n... (обрезано)"
    return txt


def _fio_tokens(text: str) -> list[str]:
    return [t.lower() for t in _FIO_TOKEN_RE.findall(str(text or ""))]


def _doctor_matches_fio(fio: str, doctor_query: str, resolved_surname: str | None = None) -> bool:
    """
    Проверка фамилии/ФИО ТОЛЬКО по fio врача.
    Не ищем по specialization, чтобы "Ким" не матчился на "хроническим".
    """
    tokens = _fio_tokens(fio)
    if not tokens:
        return False

    candidates: list[str] = []
    if resolved_surname:
        candidates.extend([v for v in surname_variants(resolved_surname) if v])
    else:
        q_tokens = _fio_tokens(doctor_query)
        if q_tokens:
            candidates.extend([v for v in surname_variants(q_tokens[0]) if v])

    normalized = sorted({ _normalise_input(x) for x in candidates if len(_normalise_input(x)) >= 2 }, key=len, reverse=True)
    if not normalized:
        return False

    for token in tokens:
        for c in normalized:
            if token.startswith(c):
                return True
    return False


def _compact_specialization(text: str, max_lines: int = 16, max_chars: int = 900) -> str:
    """
    Сжимает повторяющиеся и слишком длинные блоки специализации для безопасного рендера.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines()]
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if not ln:
            continue
        key = re.sub(r"\s+", " ", ln).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
        if len(out) >= max_lines:
            break

    compact = "\n".join(out).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


def _dedupe_doctors_by_fio(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in docs:
        fio = _normalise_input(str(d.get("fio") or ""))
        if not fio or fio in seen:
            continue
        seen.add(fio)
        out.append(d)
    return out


def _extract_result_query_fields(entities: dict[str, Any], query: str) -> dict[str, Any]:
    surname = _get_first_present(entities, ["surname", "result_surname"])
    filial = _get_first_present(entities, ["filial", "result_filial"])
    year_raw = entities.get("year")
    number_raw = entities.get("number")

    if number_raw is None:
        number_raw = entities.get("order_id")

    # Мягкий fallback: если поле фамилии не заполнено, берем первое слово из query
    if not surname and isinstance(query, str):
        words = [w for w in re.findall(r"[A-Za-zА-Яа-яЁё\-]{2,}", query)]
        if words:
            surname = words[0]

    year = _as_int(year_raw)
    number = _as_int(number_raw)
    return {
        "surname": str(surname or "").strip(),
        "year": year,
        "filial": str(filial or "").strip(),
        "number": number,
        "lang": _get_first_present(entities, ["lang", "result_lang"]) or "ru",
    }


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

    async def resolve_doctor_name(self, raw_text_or_name: str) -> str | None:
        """
        Валидация кандидата фамилии/ФИО по актуальному кэшу врачей.
        Возвращает каноническую фамилию только если удалось сопоставить с кэшем.
        """
        doctors = await self._ensure_doctors_cache_loaded()
        if not doctors:
            return None
        value = str(raw_text_or_name or "").strip()
        if not value:
            return None
        return resolve_schedule_surname(value, doctors)

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
        doctor_raw = _get_first_present(entities, ["doctor", "doctor_name", "fio", "last_name", "doctor_last_name"]) or ""
        fio_q = _normalise_input(doctor_raw)
        spec_q = _normalise_input(_get_first_present(entities, ["specialty", "specialization", "spec"]) or "")
        region_q = _normalise_input(_get_first_present(entities, ["region", "branch", " филиал", "company_unit"]) or "")
        resolved_surname = resolve_schedule_surname(doctor_raw, doctors) if doctor_raw else None

        if not resolved_surname and query:
            candidate = extract_doctor_name_candidate(query, prefer_schedule=True)
            if candidate:
                resolved_surname = resolve_schedule_surname(candidate, doctors)

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
            if fio_q:
                if not _doctor_matches_fio(fio, fio_q, resolved_surname):
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
        filtered = _dedupe_doctors_by_fio(filtered)

        if resolved_surname:
            # при явной фамилии врача не раздуваем выдачу.
            output_max = min(output_max, 3)

        # ограничим размер, чтобы не отправлять сотни карточек в LLM
        # (далее LLM/рендерер красиво завернёт)
        # Определить сколько тут карточек нужно в выводе обычно
        filtered = filtered[:output_max]
        compact: list[dict[str, Any]] = []
        for d in filtered:
            row = dict(d)
            row["specialization"] = _compact_specialization(str(row.get("specialization") or ""))
            compact.append(row)

        return {
            "doctors": compact,
            "note": "doctors_info: from cached registry (jsonl)",
            "cache_file": self._doctors_cache_path,
            "entities_used": {
                "doctor_query": fio_q,
                "doctor_resolved": resolved_surname,
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

        query_doctor_candidate = extract_doctor_name_candidate(str(query or ""), prefer_schedule=True) if query else None
        query_name = resolve_schedule_surname(str(query_doctor_candidate), doctors) if query_doctor_candidate else None
        last_name = resolve_schedule_surname(raw_for_match, doctors)
        # Если в текущей реплике явно фигурирует другая фамилия по расписанию,
        # приоритет отдаем ей (сброс от залипшего doctor_name из state).
        has_schedule_signal = bool(_SCHEDULE_QUERY_RE.search(str(query or "")))
        if query_name and (
            not last_name
            or has_schedule_signal
            or _normalise_input(str(query_name)) != _normalise_input(str(last_name))
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
        fields = _extract_result_query_fields(entities, query)
        missing = [k for k in ("surname", "year", "filial", "number") if not fields.get(k)]
        if missing:
            return {
                "ready": False,
                "note": "missing_result_fields",
                "missing_fields": missing,
                "entities_used": entities,
            }

        try:
            api_resp = await asyncio.to_thread(
                api_nayka.site_result_for_patient,
                surname=fields["surname"],
                year=int(fields["year"]),
                filial=fields["filial"],
                number=int(fields["number"]),
                lang=fields["lang"],
                with_time=None,
            )
        except Exception as e:
            return _service_fallback(
                note=f"resultForPatient failed: {e}",
                handoff_message="Сейчас не удалось получить результаты автоматически. Соединяю с оператором.",
                entities=entities,
                reason="test_result_fallback",
                extra={"ready": False},
            )

        if not isinstance(api_resp, dict) or not api_resp.get("ok"):
            return _service_fallback(
                note=f"resultForPatient error: {api_resp}",
                handoff_message="Сейчас не удалось получить результаты автоматически. Соединяю с оператором.",
                entities=entities,
                reason="test_result_fallback",
                extra={"ready": False},
            )

        payload = api_resp.get("data")
        has_payload = bool(payload)
        if not has_payload:
            return {
                "ready": False,
                "note": "result_not_found_or_not_ready",
                "result_payload": payload,
                "result_preview": "По указанным данным результаты пока не найдены или еще не готовы.",
                "entities_used": entities,
            }

        return {
            "ready": True,
            "note": "resultForPatient success",
            "result_payload": payload,
            "result_preview": _stringify_json_preview(payload),
            "entities_used": entities,
        }

    async def test_result_pdf(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        # Пока отдельного PDF endpoint не подключали: отдаем пусто без handoff,
        # чтобы не ломать успешный TEST_RESULT статус.
        return {"pdf": None, "note": "no_result_pdf_endpoint", "entities_used": entities}

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
