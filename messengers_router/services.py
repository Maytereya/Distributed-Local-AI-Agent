from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_logic_2.nayka_api import api_nayka


def _normalise_input(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _extract_last_name(text: str) -> Optional[str]:
    """
    Простейшая эвристика: берём самое "похожее на фамилию" слово.
    Для расписания нам обычно нужен last_name.
    """
    words = re.findall(r"[A-Za-zА-Яа-яЁё\-]{3,}", text or "")
    if not words:
        return None
    # чаще фамилия — последнее "содержательное" слово
    return words[-1]


def _get_first_present(d: dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


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

    # блокировка, чтобы несколько запросов параллельно не перегенерировали кэш
    _doctors_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # TTL in-memory кэша (латентность, сек, определят свежесть кэша)
    doctors_mem_ttl_seconds: int = 300

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

            self._doctors_cache = doctors_loaded
            self._doctors_cache_path = str(file_path) if file_path is not None else None
            self._doctors_cache_loaded_at = time.time()
            return self._doctors_cache

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
        last_name = _get_first_present(
            entities,
            ["last_name", "doctor_last_name", "doctor", "doctor_name", "fio"],
        )
        if not last_name:
            last_name = _extract_last_name(query)

        if not last_name:
            return {
                "schedule": [],
                "note": "doctors_schedule_week: missing doctor last name",
                "entities_used": entities,
            }

        # необязательный фильтр региона/филиала
        region_name = _get_first_present(entities, ["region", "branch", "company_unit", "unit"])

        # api_nayka.find_doctor_schedule блокирующая (requests) — уводим в thread
        try:
            data = await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name, region_name)
        except TypeError:
            # если сигнатура find_doctor_schedule(last_name) без region_name
            data = await asyncio.to_thread(api_nayka.find_doctor_schedule, last_name)

        return {
            "schedule": data or [],
            "note": "doctors_schedule_week: realtime from Nayka API",
            "entities_used": {"last_name": last_name, "region_name": region_name},
        }

    # Остальные методы пока как заглушки
    async def appointment_help(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {
            "instructions": "Чтобы записаться, уточните врача/специальность/услугу и удобные даты.",
            "entities_used": entities,
        }

    async def test_assist(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"tests": [], "promos": [], "note": "stub test_assist", "entities_used": entities}

    async def test_prepare(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"prepare": "", "note": "stub test_prepare", "entities_used": entities}

    async def test_result_status(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"ready": False, "note": "stub test_result_status", "entities_used": entities}

    async def test_result_pdf(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"pdf": None, "note": "stub test_result_pdf", "entities_used": entities}

    async def price_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"prices": [], "note": "stub price_info", "entities_used": entities}

    async def address_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"addresses": [], "note": "stub address_info", "entities_used": entities}

    async def news_info(self, query: str, entities: dict[str, Any]) -> dict[str, Any]:
        return {"news": [], "note": "stub news_info", "entities_used": entities}

    def get_branches(self) -> list[dict[str, str]]:
        """
        Возвращает справочник филиалов.
        Формат:
          [{"id":"branch_1","name":"Филиал на Проспекте Ленина","aliases":"Ленина, Ленинская,Проспект Ленина 5"}]
        Пока заглушка.
        """
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
