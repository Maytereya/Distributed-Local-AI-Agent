#!/usr/bin/env python3
"""Live-аудит site API Наука и генерация сводного markdown-отчета.

Скрипт проверяет ключевые endpoint-ы `/api/v1/site/*`, собирает примеры ответов
и формирует отчет с эвристическим поиском мест использования endpoint-ов в коде.
Ответственность скрипта: технический аудит доступности/формата site API.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from agent_logic_2.nayka_api import api_nayka


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "nayka_site_api_live_audit_latest.md"
SAMARA_CITY_REGION_ID = 3

USAGE_PATTERNS: dict[str, tuple[str, ...]] = {
    "/regions": ("site_regions(", "_ensure_regions_loaded(", "address_info(", "get_branches("),
    "/doctorServicePricesByRegion": ("load_doctor_prices(", "fetch_doctor_prices(", "price_info("),
    "/doctorSchedule + /doctorScheduleCells": ("find_doctor_schedule(", "doctorScheduleCells", "doctors_schedule_week("),
    "/doctors": ("site_doctors(", "get_all_doctors(", "find_doctor_schedule("),
    "/priceByRegion/{regionId}": ("load_price_by_region(", "fetch_price_by_region("),
}


def _build_session() -> requests.Session:
    session = requests.Session()
    session.auth = api_nayka.auth
    return session


def _get_json(session: requests.Session, path: str) -> tuple[int, Any]:
    resp = session.get(
        f"{api_nayka.base_url.rstrip('/')}{path}",
        verify=api_nayka.VERIFY_ARG,
        timeout=api_nayka.DEFAULT_TIMEOUT,
    )
    status = resp.status_code
    if not resp.ok:
        return status, (resp.text or "").strip()
    try:
        return status, resp.json()
    except ValueError:
        return status, (resp.text or "").strip()


def _find_samara_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in regions:
        if not isinstance(row, dict):
            continue
        blob = " ".join(
            str(row.get(key) or "")
            for key in ("city", "name", "addressForSite", "address", "fullName")
        ).lower()
        if "самара" in blob or row.get("parent") == SAMARA_CITY_REGION_ID or row.get("id") == SAMARA_CITY_REGION_ID:
            out.append(row)
    return out


def _pretty_json(data: Any, max_len: int = 1600) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def _collect_usage_hits() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    skip_dirs = {"venv", "__pycache__", ".git"}
    py_files = [p for p in REPO_ROOT.rglob("*.py") if not any(part in skip_dirs for part in p.parts)]

    for label, patterns in USAGE_PATTERNS.items():
        found: list[str] = []
        for path in py_files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(lines, 1):
                if any(pattern in line for pattern in patterns):
                    rel = path.relative_to(REPO_ROOT)
                    found.append(f"{rel}:{lineno}")
                    if len(found) >= 6:
                        break
            if len(found) >= 6:
                break
        hits[label] = found
    return hits


def build_report() -> str:
    session = _build_session()

    status_regions, regions_raw = _get_json(session, "/regions")
    regions = regions_raw if isinstance(regions_raw, list) else []
    samara_regions = _find_samara_regions(regions)

    status_doctors, doctors_raw = _get_json(session, "/doctors")
    doctors = doctors_raw if isinstance(doctors_raw, list) else []

    status_doctor_regions, doctor_regions_raw = _get_json(session, "/doctorRegions")
    doctor_regions = doctor_regions_raw if isinstance(doctor_regions_raw, list) else []

    samara_region_ids = {row.get("id") for row in samara_regions if isinstance(row, dict)}
    sample_pair: dict[str, Any] | None = None
    for row in doctor_regions:
        if not isinstance(row, dict):
            continue
        if row.get("region") in samara_region_ids:
            sample_pair = row
            break
    if sample_pair is None:
        for row in doctor_regions:
            if isinstance(row, dict):
                sample_pair = row
                break

    schedule_status = None
    schedule_sample: Any = None
    cells_status = None
    cells_sample: Any = None
    if sample_pair:
        doctor_id = sample_pair.get("worker")
        company_unit = sample_pair.get("companyUnit")
        region_id = sample_pair.get("region")
        start = date.today().isoformat()
        end = (date.today() + timedelta(days=7)).isoformat()
        schedule_path = (
            f"/doctorSchedule?doctor={doctor_id}&companyUnit={company_unit}"
            f"&region={region_id}&startDate={start}&endDate={end}"
        )
        schedule_status, schedule_raw = _get_json(session, schedule_path)
        if isinstance(schedule_raw, list) and schedule_raw:
            schedule_sample = schedule_raw[0]
            schedule_id = schedule_sample.get("id")
            if schedule_id is not None:
                cells_status, cells_raw = _get_json(session, f"/doctorScheduleCells?doctorSchedule={schedule_id}")
                if isinstance(cells_raw, list) and cells_raw:
                    cells_sample = cells_raw[0]
        else:
            schedule_sample = schedule_raw

    doctor_price_status, doctor_price_raw = _get_json(session, "/doctorServicePricesByRegion?doctorId=4&regionId=8882")
    doctor_price_sample = doctor_price_raw[0] if isinstance(doctor_price_raw, list) and doctor_price_raw else doctor_price_raw

    doctor_price_plural_status, doctor_price_plural_raw = _get_json(
        session, "/doctorServicesPricesByRegion?doctorId=4&regionId=8882"
    )

    branch_price_checks: list[dict[str, Any]] = []
    for row in samara_regions[:10]:
        region_id = row.get("id")
        if region_id is None:
            continue
        status, payload = _get_json(session, f"/priceByRegion/{region_id}")
        count = len(payload) if isinstance(payload, list) else None
        branch_price_checks.append(
            {
                "id": region_id,
                "name": row.get("name"),
                "addressForSite": row.get("addressForSite"),
                "status": status,
                "count": count,
            }
        )

    price_status, price_raw = _get_json(session, f"/priceByRegion/{SAMARA_CITY_REGION_ID}")
    price_sample = price_raw[0] if isinstance(price_raw, list) and price_raw else price_raw
    analysis_hits: list[dict[str, Any]] = []
    uzi_hits: list[dict[str, Any]] = []
    if isinstance(price_raw, list):
        for item in price_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("serviceName") or "")
            low = name.lower()
            if len(analysis_hits) < 3 and ("анализ" in low or "кров" in low or "моч" in low):
                analysis_hits.append(
                    {
                        "serviceName": name,
                        "cost": item.get("cost"),
                        "serviceHomecode": item.get("serviceHomecode"),
                    }
                )
            if len(uzi_hits) < 3 and "узи" in low:
                uzi_hits.append(
                    {
                        "serviceName": name,
                        "cost": item.get("cost"),
                        "serviceHomecode": item.get("serviceHomecode"),
                    }
                )
            if len(analysis_hits) >= 3 and len(uzi_hits) >= 3:
                break

    doctors_ord_examples = [
        {
            "id": row.get("id"),
            "fio": row.get("fio"),
            "ord": row.get("ord"),
        }
        for row in doctors
        if isinstance(row, dict) and row.get("ord") is not None
    ][:5]

    region_8502_present = any(isinstance(row, dict) and row.get("id") == 8502 for row in regions)
    usage_hits = _collect_usage_hits()

    lines: list[str] = []
    lines.append("# Nayka Site API Live Audit")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append("- Generated by `messengers_router/scripts/audit_nayka_site_api.py`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- `/regions`: HTTP `{status_regions}`, items `{len(regions)}`")
    lines.append(f"- `/doctors`: HTTP `{status_doctors}`, items `{len(doctors)}`")
    lines.append(f"- `/doctorRegions`: HTTP `{status_doctor_regions}`, items `{len(doctor_regions)}`")
    lines.append(f"- `/doctorServicePricesByRegion`: HTTP `{doctor_price_status}`")
    lines.append(f"- `/doctorServicesPricesByRegion`: HTTP `{doctor_price_plural_status}`")
    lines.append(f"- `/priceByRegion/{SAMARA_CITY_REGION_ID}`: HTTP `{price_status}`, items `{len(price_raw) if isinstance(price_raw, list) else 0}`")
    lines.append(f"- Region `8502` present in `/regions`: `{region_8502_present}`")
    lines.append("")
    lines.append("## /regions")
    lines.append("")
    lines.append(f"- Samara regions detected: `{len(samara_regions)}`")
    lines.append("```json")
    lines.append(_pretty_json(samara_regions[0] if samara_regions else (regions[0] if regions else regions_raw)))
    lines.append("```")
    lines.append("")
    lines.append("## /doctorServicePricesByRegion")
    lines.append("")
    lines.append("```json")
    lines.append(_pretty_json(doctor_price_sample))
    lines.append("```")
    lines.append("")
    lines.append("## /doctorServicesPricesByRegion (plural form)")
    lines.append("")
    lines.append("```json")
    lines.append(_pretty_json(doctor_price_plural_raw))
    lines.append("```")
    lines.append("")
    lines.append("## /doctorSchedule and /doctorScheduleCells")
    lines.append("")
    if sample_pair:
        lines.append(f"- Sample pair: `{json.dumps(sample_pair, ensure_ascii=False)}`")
    if schedule_status is not None:
        lines.append(f"- `doctorSchedule` HTTP `{schedule_status}`")
    lines.append("```json")
    lines.append(_pretty_json(schedule_sample))
    lines.append("```")
    if cells_status is not None:
        lines.append(f"- `doctorScheduleCells` HTTP `{cells_status}`")
        lines.append("```json")
        lines.append(_pretty_json(cells_sample))
        lines.append("```")
    lines.append("")
    lines.append("## /doctors")
    lines.append("")
    lines.append("```json")
    lines.append(_pretty_json(doctors_ord_examples))
    lines.append("```")
    lines.append("")
    lines.append("## /priceByRegion/{regionId}")
    lines.append("")
    lines.append("- Branch-level checks (first Samara regions):")
    lines.append("```json")
    lines.append(_pretty_json(branch_price_checks))
    lines.append("```")
    lines.append("- City-level retail price (`regionId=3`):")
    lines.append("```json")
    lines.append(_pretty_json(price_sample))
    lines.append("```")
    lines.append("- Analysis-like examples:")
    lines.append("```json")
    lines.append(_pretty_json(analysis_hits))
    lines.append("```")
    lines.append("- UZI-like examples:")
    lines.append("```json")
    lines.append(_pretty_json(uzi_hits))
    lines.append("```")
    lines.append("")
    lines.append("## Heuristic Code Usage")
    lines.append("")
    lines.append("Это быстрый grep по проекту, не строгий call graph.")
    lines.append("")
    for label, refs in usage_hits.items():
        lines.append(f"### {label}")
        if refs:
            for ref in refs:
                lines.append(f"- `{ref}`")
        else:
            lines.append("- not found")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Nayka site endpoints and write a Markdown report.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output markdown path")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
