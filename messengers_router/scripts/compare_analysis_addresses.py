"""Сравнение адресов "сдать анализы/ЭКГ" между API и эталонным списком.

Запуск:
  venv/bin/python messengers_router/scripts/compare_analysis_addresses.py \
    --reference-file /path/to/samara_reference.txt \
    --city Самара

Файл reference-file:
  одна строка = один адрес (без телефонов/графика).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from agent_logic_2.nayka_api import api_nayka


def _norm_addr(s: str) -> str:
    t = (s or "").lower().strip()
    t = t.replace("ё", "е")
    t = t.replace("кор.", "корпус").replace("к.", "корпус")
    t = re.sub(r"\bг\.\s*", "", t)
    t = re.sub(r"\bулица\b", "ул", t)
    t = re.sub(r"\bпроспект\b", "пр", t)
    t = re.sub(r"\bпр-т\b", "пр", t)
    t = re.sub(r"[«»\"()]", " ", t)
    t = re.sub(r"[^a-zа-я0-9\s\-/.,]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" ,.")


def _looks_like_address(s: str) -> bool:
    t = s.lower()
    return bool(
        re.search(r"\d", t)
        and re.search(r"\b(ул|пр|просп|тракт|бульвар|б-р|шоссе|квартал|пос|п\.)\b", t)
    )


def _extract_api_city_addresses(city: str) -> list[str]:
    regions = api_nayka.site_regions()
    if not isinstance(regions, list):
        return []
    out: list[str] = []
    city_norm = _norm_addr(city)
    for row in regions:
        if not isinstance(row, dict):
            continue
        addr = str(row.get("addressForSite") or row.get("name") or "").strip()
        if not addr or not _looks_like_address(addr):
            continue
        city_val = str(row.get("city") or "").strip()
        hay = _norm_addr(f"{city_val} {addr}")
        # Южный город часто относится к Самаре в бизнес-сценарии.
        if city_norm in hay or ("южный город" in hay and city_norm == "самара"):
            out.append(addr)
    uniq = []
    seen = set()
    for a in out:
        n = _norm_addr(a)
        if n not in seen:
            seen.add(n)
            uniq.append(a)
    return sorted(uniq)


def _load_reference(path: Path) -> list[str]:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    lines = [x for x in lines if x and _looks_like_address(x)]
    uniq = []
    seen = set()
    for a in lines:
        n = _norm_addr(a)
        if n not in seen:
            seen.add(n)
            uniq.append(a)
    return sorted(uniq)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-file", required=True, help="txt с эталонными адресами (по одному на строке)")
    p.add_argument("--city", default="Самара", help="город для фильтра в API")
    args = p.parse_args()

    ref_path = Path(args.reference_file)
    if not ref_path.exists():
        print(f"[error] reference file not found: {ref_path}")
        return 2

    ref = _load_reference(ref_path)
    api = _extract_api_city_addresses(args.city)

    ref_norm = {_norm_addr(x): x for x in ref}
    api_norm = {_norm_addr(x): x for x in api}

    only_ref = sorted([ref_norm[k] for k in ref_norm.keys() - api_norm.keys()])
    only_api = sorted([api_norm[k] for k in api_norm.keys() - ref_norm.keys()])
    both = sorted([api_norm[k] for k in api_norm.keys() & ref_norm.keys()])

    print(f"City: {args.city}")
    print(f"Reference addresses: {len(ref)}")
    print(f"API addresses: {len(api)}")
    print(f"Matched: {len(both)}")
    print(f"Only in reference: {len(only_ref)}")
    print(f"Only in API: {len(only_api)}")
    print("")

    if only_ref:
        print("=== Only in reference ===")
        for a in only_ref:
            print(f"- {a}")
        print("")

    if only_api:
        print("=== Only in API ===")
        for a in only_api:
            print(f"- {a}")
        print("")

    if both:
        print("=== Matched ===")
        for a in both:
            print(f"- {a}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

