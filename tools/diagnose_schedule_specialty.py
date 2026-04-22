"""Диагностика выбора врача по специальности для запросов вида "расписание <процедура>".

Скрипт повторяет логику `services/doctors.py::_schedule_by_specialty`:
1. Извлекает specialty из текста (через текущий код).
2. Если код не отдал specialty — прогоняет вручную набор гипотез
   (по умолчанию для "флюорография" — `рентгенолог`,
   `функциональная диагностика`, `узи`).
3. Для каждой гипотезы печатает top-8 кандидатов из doctors-кэша:
   FIO, units, regions, role-match level и обычный sort_key.
4. Опционально (`--live-schedule`) дёргает live API расписания
   для каждого кандидата и печатает ближайший свободный слот.
5. Опционально (`--hit-bot URL`) дёргает живой messenger-endpoint
   и печатает финальный ответ бота.

Запуск:
    PYTHONPATH=. python3 tools/diagnose_schedule_specialty.py "расписание флюорография"
    PYTHONPATH=. python3 tools/diagnose_schedule_specialty.py "расписание флюорография" --refresh-doctors
    PYTHONPATH=. python3 tools/diagnose_schedule_specialty.py "расписание флюорография" --live-schedule
    PYTHONPATH=. python3 tools/diagnose_schedule_specialty.py "расписание флюорография" --hit-bot http://172.16.0.16/api/messenger-generate-once

При запуске без --refresh-doctors используется локальный JSONL-кэш (может отставать
от прода). Все live-вызовы требуют корректно настроенных env-переменных Nayka API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from typing import Any

# Имитируем "from project root"
from agent_logic_2.nayka_api import api_nayka
from messengers_router.services._doctors_helpers import (
    _doctor_matches_specialty,
    _doctor_role_specialty_match_level,
    _doctor_sort_key,
    _is_role_specialty_query,
    _iter_slot_datetimes,
    _procedure_query_role_specialty,
    _is_schedule_no_slots_text,
)
from messengers_router.services._regions import (
    _has_explicit_non_samara_regions,
    _is_samara_city_value,
    _region_matches_samara_tokens,
)
from messengers_router.services._common import _normalise_input
from messengers_router.specialty_parser import extract_specialty_from_text


DEFAULT_HYPOTHESES_BY_TOKEN: dict[str, tuple[str, ...]] = {
    "флюорограф": ("рентгенолог", "функциональная диагностика", "узи", "терапевт"),
    "маммограф": ("рентгенолог", "гинеколог-маммолог", "функциональная диагностика"),
    "рентген": ("рентгенолог", "функциональная диагностика"),
    "экг": ("функциональная диагностика", "кардиолог"),
    "фгдс": ("эндоскопист", "гастроэнтеролог"),
    "колоноскоп": ("эндоскопист", "колопроктолог"),
    "узи": ("узи", "функциональная диагностика"),
}


def _default_hypotheses(query: str) -> list[str]:
    qn = _normalise_input(query)
    out: list[str] = []
    for token, hypotheses in DEFAULT_HYPOTHESES_BY_TOKEN.items():
        if token in qn:
            for hyp in hypotheses:
                if hyp not in out:
                    out.append(hyp)
    return out


def _load_doctors(refresh: bool) -> list[dict[str, Any]]:
    if refresh:
        try:
            doctors = api_nayka.get_all_doctors()
            api_nayka.save_doctors_data(doctors)
            print(f"[refresh-doctors] OK, всего врачей: {len(doctors)}")
            return doctors
        except Exception as exc:  # pragma: no cover
            print(f"[refresh-doctors] FAIL: {exc!r} — fallback на локальный кэш")
    return api_nayka.get_cached_doctors_data() or []


def _samara_tokens_from_regions(regions: list[dict[str, Any]]) -> set[str]:
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
        for raw in (city, name, addr):
            for tok in _normalise_input(raw).split():
                if len(tok) >= 4:
                    tokens.add(tok)
    return tokens


def _filter_and_sort_candidates(
    doctors: list[dict[str, Any]],
    specialty: str,
    query: str,
    samara_tokens: set[str],
) -> list[dict[str, Any]]:
    spec_norm = _normalise_input(specialty)
    role_query = _is_role_specialty_query(query or specialty, spec_norm)
    role_levels = (
        {id(d): _doctor_role_specialty_match_level(d, spec_norm) for d in doctors}
        if role_query
        else {}
    )

    def _passes(d: dict[str, Any]) -> bool:
        if not isinstance(d, dict):
            return False
        if role_query:
            if role_levels.get(id(d), 0) <= 0:
                return False
        else:
            if not _doctor_matches_specialty(d, spec_norm, query or specialty):
                return False
        regions_src = [str(x) for x in (d.get("regions") or []) if str(x).strip()]
        if _has_explicit_non_samara_regions(regions_src):
            return False
        if samara_tokens and regions_src and not any(
            _region_matches_samara_tokens(rs, samara_tokens) for rs in regions_src
        ):
            return False
        return True

    candidates = [d for d in doctors if _passes(d)]
    candidates.sort(
        key=(
            (lambda d: (-role_levels.get(id(d), 0), *_doctor_sort_key(d)))
            if role_query
            else _doctor_sort_key
        )
    )
    return candidates[:8]


def _print_candidate(idx: int, doc: dict[str, Any], specialty: str, query: str) -> None:
    fio = str(doc.get("fio") or "").strip()
    units = doc.get("units") or []
    regions = doc.get("regions") or []
    role_level = _doctor_role_specialty_match_level(doc, _normalise_input(specialty))
    sort_key = _doctor_sort_key(doc)
    print(f"  #{idx:>2} [{role_level}] {fio}")
    print(f"        units:   {units}")
    print(f"        regions: {regions}")
    print(f"        sort_key: {sort_key}")


async def _fetch_schedule_for_candidate(
    doc: dict[str, Any],
) -> tuple[str | None, list[str], dict[str, Any] | None]:
    fio = str(doc.get("fio") or "").strip()
    surname = fio.split()[0] if fio else ""
    if not surname:
        return None, [], None
    try:
        data = await asyncio.to_thread(api_nayka.find_doctor_schedule, surname)
    except Exception as exc:
        return f"FETCH_FAIL: {exc!r}", [], None
    if _is_schedule_no_slots_text(data):
        return "no_free_slots_text", [], None
    if not isinstance(data, list) or not data:
        return "empty_or_invalid", [], data if isinstance(data, dict) else None

    nearest_slots: list[str] = []
    matched_row: dict[str, Any] | None = None
    for row in data:
        if not isinstance(row, dict):
            continue
        row_fio = str(row.get("fio") or "").strip()
        if row_fio and _normalise_input(row_fio) != _normalise_input(fio):
            continue
        slots = _iter_slot_datetimes(row.get("schedule") or {})
        if slots:
            slots_sorted = sorted(slots)
            nearest_slots = [s.isoformat() for s in slots_sorted[:3]]
        matched_row = row
        break
    return None, nearest_slots, matched_row


async def _hit_bot(url: str, query: str) -> dict[str, Any]:
    import urllib.request

    payload = {
        "session_id": f"diag_{uuid.uuid4().hex[:8]}",
        "user_id": "diagnose_specialty",
        "text": query,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw}

    return await loop.run_in_executor(None, _do)


async def main_async(args: argparse.Namespace) -> int:
    print(f"\n=== Diagnose schedule-by-specialty ===")
    print(f"query: {args.query!r}")
    print(f"timestamp: {datetime.now().isoformat()}")

    print("\n[1/4] Извлекаем specialty из текста через текущий код:")
    code_specialty = extract_specialty_from_text(args.query)
    code_role_specialty = _procedure_query_role_specialty(args.query)
    print(f"  extract_specialty_from_text() -> {code_specialty!r}")
    print(f"  _procedure_query_role_specialty() -> {code_role_specialty!r}")

    hypotheses: list[str] = []
    if args.hypothesis:
        hypotheses = list(args.hypothesis)
    else:
        if code_specialty:
            hypotheses.append(code_specialty)
        if code_role_specialty and code_role_specialty not in hypotheses:
            hypotheses.append(code_role_specialty)
        for h in _default_hypotheses(args.query):
            if h not in hypotheses:
                hypotheses.append(h)
    if not hypotheses:
        print("\n[!] Не удалось определить ни одну специальность. "
              "Передайте --hypothesis <spec> вручную.")
        return 2
    print(f"  Гипотезы для прогона: {hypotheses}")

    print("\n[2/4] Загружаем doctors cache...")
    doctors = _load_doctors(refresh=args.refresh_doctors)
    print(f"  Всего врачей в кэше: {len(doctors)}")

    print("\n[3/4] Загружаем regions для samara_tokens...")
    try:
        regions = api_nayka.site_regions() or []
    except Exception as exc:
        print(f"  [!] regions FAIL: {exc!r} — продолжаем без samara filter")
        regions = []
    samara_tokens = _samara_tokens_from_regions(regions)
    print(f"  samara_tokens: {sorted(samara_tokens)[:8]}{'...' if len(samara_tokens) > 8 else ''}")

    print("\n[4/4] Прогон гипотез:")
    for spec in hypotheses:
        print(f"\n--- specialty: {spec!r} ---")
        role_query = _is_role_specialty_query(args.query, _normalise_input(spec))
        print(f"  role_query={role_query}")
        candidates = _filter_and_sort_candidates(doctors, spec, args.query, samara_tokens)
        if not candidates:
            print("  candidates: (нет)")
            continue
        print(f"  top-{len(candidates)} candidates:")
        for idx, doc in enumerate(candidates, start=1):
            _print_candidate(idx, doc, spec, args.query)

        if args.live_schedule:
            print("\n  [live schedule fetch]")
            for idx, doc in enumerate(candidates, start=1):
                fio = str(doc.get("fio") or "").strip()
                err, nearest, _row = await _fetch_schedule_for_candidate(doc)
                if err:
                    print(f"    #{idx} {fio}: {err}")
                elif nearest:
                    print(f"    #{idx} {fio}: nearest_slots={nearest}")
                else:
                    print(f"    #{idx} {fio}: no_slots_in_payload")

    if args.hit_bot:
        print(f"\n=== Live bot call: {args.hit_bot} ===")
        try:
            resp = await _hit_bot(args.hit_bot, args.query)
            print(json.dumps(resp, ensure_ascii=False, indent=2)[:4000])
        except Exception as exc:
            print(f"[hit-bot] FAIL: {exc!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Исходный текст пользователя")
    parser.add_argument("--hypothesis", action="append", default=[],
                        help="Specialty-гипотеза (повторяемый аргумент)")
    parser.add_argument("--refresh-doctors", action="store_true",
                        help="Перед стартом обновить doctors cache через live API")
    parser.add_argument("--live-schedule", action="store_true",
                        help="Дёргать live расписание для каждого top-кандидата")
    parser.add_argument("--hit-bot", default=None,
                        help="URL messenger-endpoint, например http://172.16.0.16/api/messenger-generate-once")
    args = parser.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
