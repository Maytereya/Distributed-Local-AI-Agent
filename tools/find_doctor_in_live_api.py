"""Поиск врача в сырых эндпоинтах Nayka API без локальных фильтров.

Зачем: убедиться, что отсутствие врача в локальном `doctors_*.jsonl` —
это не снапшот-лаг, а отсутствие в самом CRM (или попадание под
фильтр Оренбурга/пустых regions).

Дёргает напрямую:
- /doctors            — список всех врачей
- /doctorRegions      — связи доктор↔регион
- /doctorCompanyUnits — связи доктор↔подразделение
- /companyUnits       — справочник подразделений
- /regions            — справочник регионов

Запуск:
    PYTHONPATH=. python3 tools/find_doctor_in_live_api.py Тагиров
    PYTHONPATH=. python3 tools/find_doctor_in_live_api.py Тагиров --also Казаков

Печатает по каждому совпадению: ФИО, id, units (с companyUnitId/spec/main),
regions (с companyUnitId), excluded-status (Оренбург или нет).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

from agent_logic_2.nayka_api import api_nayka


def _norm_lower(s: Any) -> str:
    return str(s or "").strip().lower()


def _collect_excluded(regions: list[dict[str, Any]]) -> set[int]:
    return api_nayka._collect_region_descendants(regions, api_nayka.EXCLUDED_REGION_ROOTS)


def _print_match(
    doctor: dict[str, Any],
    *,
    doctor_units: list[dict[str, Any]],
    doctor_regions: list[dict[str, Any]],
    units_dict: dict[int, str],
    regions_dict: dict[int, str],
    excluded_region_ids: set[int],
) -> None:
    did = doctor.get("id")
    fio = doctor.get("fio") or doctor.get("name") or ""
    print(f"\n=== MATCH: {fio} (id={did}) ===")
    print(f"  raw doctor row: {json.dumps({k: v for k, v in doctor.items() if k not in ('photo', 'image')}, ensure_ascii=False)[:500]}")

    my_units = [u for u in doctor_units if u.get("worker") == did]
    my_regions = [r for r in doctor_regions if r.get("worker") == did]

    print(f"  /doctorCompanyUnits ({len(my_units)} links):")
    for link in my_units:
        cuid = link.get("companyUnit")
        unit_name = units_dict.get(cuid, "??")
        spec = link.get("specialization") or ""
        main = link.get("main")
        print(f"    cu_id={cuid} '{unit_name}' main={main} spec={spec!r}")

    print(f"  /doctorRegions ({len(my_regions)} links):")
    excluded_count = 0
    for link in my_regions:
        rid = link.get("region")
        cuid = link.get("companyUnit")
        rname = regions_dict.get(rid, "??")
        is_excluded = rid in excluded_region_ids
        if is_excluded:
            excluded_count += 1
        flag = " [EXCLUDED-Оренбург]" if is_excluded else ""
        print(f"    region_id={rid} '{rname}' company_unit={cuid}{flag}")

    if my_regions and excluded_count == len(my_regions):
        print(f"  ⚠️  ВСЕ regions у врача — в EXCLUDED_REGION_ROOTS (Оренбург). "
              f"get_all_doctors() ОТБРОСИТ его (см. api_nayka.py:402).")
    if not my_regions:
        print(f"  ⚠️  У врача 0 связей в /doctorRegions. "
              f"get_all_doctors() ОТБРОСИТ его (valid_region_links пуст).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("surname", help="Часть фамилии (case-insensitive substring)")
    parser.add_argument("--also", action="append", default=[], help="Доп. фамилии для поиска")
    args = parser.parse_args()

    needles = [_norm_lower(args.surname)] + [_norm_lower(s) for s in args.also]
    needles = [n for n in needles if n]

    print(f"Searching live /doctors for substrings: {needles}")
    print("=" * 60)

    print("[1/5] GET /doctors ...")
    doctors = api_nayka.site_doctors() or []
    print(f"  total: {len(doctors)}")

    print("[2/5] GET /companyUnits ...")
    units = api_nayka.site_company_units() or []
    units_dict = {u.get("id"): u.get("name") for u in units if isinstance(u, dict)}
    print(f"  total units: {len(units)}")

    print("[3/5] GET /doctorCompanyUnits ...")
    doctor_units = api_nayka.site_doctor_company_units() or []
    print(f"  total links: {len(doctor_units)}")

    print("[4/5] GET /doctorRegions ...")
    doctor_regions = api_nayka.site_doctor_regions() or []
    print(f"  total links: {len(doctor_regions)}")

    print("[5/5] GET /regions ...")
    regions = api_nayka.site_regions() or []
    regions_dict = {r.get("id"): api_nayka._region_display_name(r) for r in regions if isinstance(r, dict)}
    excluded_region_ids = _collect_excluded(regions)
    print(f"  total regions: {len(regions)}, excluded(Оренбург+потомки): {len(excluded_region_ids)} ids")

    print("\n--- SCAN /doctors ---")
    found_any = False
    for doc in doctors:
        if not isinstance(doc, dict):
            continue
        fio = _norm_lower(doc.get("fio") or doc.get("name"))
        if not fio:
            continue
        if any(needle in fio for needle in needles):
            found_any = True
            _print_match(
                doc,
                doctor_units=doctor_units,
                doctor_regions=doctor_regions,
                units_dict=units_dict,
                regions_dict=regions_dict,
                excluded_region_ids=excluded_region_ids,
            )

    if not found_any:
        print(f"\n[!] В /doctors ({len(doctors)} записей) не найдено ни одного врача с фамилией {needles}")
        print(f"    Возможные причины:")
        print(f"     - врача нет в CRM Naika (только в другой системе расписания)")
        print(f"     - врач — медсестра/лаборант (не попадает в /doctors)")
        print(f"     - опечатка в фамилии (попробуйте сократить needle)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
