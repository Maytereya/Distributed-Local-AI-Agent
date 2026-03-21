#!/usr/bin/env python3
"""Coverage gate for extended eval assets (P2 scope).

Checks that extension datasets include required dimensions:
- labels: DOCTOR_INFO, DOCTOR_SCHEDULE, PREPARE, OTHER
- non-Samara cases
- follow-up (multi-turn) scenarios
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE5_EXT = (
    ROOT
    / "messengers_router"
    / "messengers_mds_to_collect_thoughts"
    / "analysis"
    / "golden_versions"
    / "stage5_golden_extension_v3.jsonl"
)
DEFAULT_CRITICAL_EXT = ROOT / "messengers_router" / "eval_suite" / "critical_cases_extended.jsonl"

CITY_KEYWORDS = (
    "тольятти",
    "ульяновск",
    "оренбург",
    "кувандык",
    "казань",
    "москва",
    "санкт",
    "спб",
    "питер",
)

MIN_STAGE5_LABELS = {
    "DOCTOR_INFO": 2,
    "DOCTOR_SCHEDULE": 2,
    "PREPARE": 3,
    "OTHER": 3,
}
MIN_NON_SAMARA = 3
MIN_FOLLOW_UP_SCENARIOS = 4


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _is_non_samara(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CITY_KEYWORDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check coverage of extended eval datasets")
    parser.add_argument("--stage5-ext", default=str(DEFAULT_STAGE5_EXT), help="Path to extended stage5 golden JSONL")
    parser.add_argument("--critical-ext", default=str(DEFAULT_CRITICAL_EXT), help="Path to extended critical JSONL")
    args = parser.parse_args()

    stage5_path = Path(args.stage5_ext).resolve()
    critical_path = Path(args.critical_ext).resolve()

    if not stage5_path.exists():
        print(f"[error] stage5 extension file not found: {stage5_path}")
        return 2
    if not critical_path.exists():
        print(f"[error] critical extension file not found: {critical_path}")
        return 2

    stage5_cases = _load_jsonl(stage5_path)
    critical_cases = _load_jsonl(critical_path)

    stage5_label_counts: Counter[str] = Counter()
    non_samara_count = 0
    follow_up_scenarios = 0

    for case in stage5_cases:
        label = str(case.get("expected_label") or "").strip()
        if label:
            stage5_label_counts[label] += 1
        if _is_non_samara(str(case.get("text") or "")):
            non_samara_count += 1

    for case in critical_cases:
        turns = case.get("turns")
        if isinstance(turns, list) and turns:
            if len(turns) > 1:
                follow_up_scenarios += 1
            for turn in turns:
                if isinstance(turn, dict):
                    label = str(turn.get("expected_label") or "").strip()
                    if label:
                        stage5_label_counts[label] += 1
                    if _is_non_samara(str(turn.get("text") or "")):
                        non_samara_count += 1
        else:
            label = str(case.get("expected_label") or "").strip()
            if label:
                stage5_label_counts[label] += 1
            if _is_non_samara(str(case.get("text") or "")):
                non_samara_count += 1

    errors: list[str] = []

    for label, min_count in MIN_STAGE5_LABELS.items():
        actual = int(stage5_label_counts.get(label, 0))
        if actual < min_count:
            errors.append(f"{label}: expected >= {min_count}, got {actual}")

    if non_samara_count < MIN_NON_SAMARA:
        errors.append(f"non-Samara coverage: expected >= {MIN_NON_SAMARA}, got {non_samara_count}")
    if follow_up_scenarios < MIN_FOLLOW_UP_SCENARIOS:
        errors.append(
            f"follow-up scenarios: expected >= {MIN_FOLLOW_UP_SCENARIOS}, got {follow_up_scenarios}"
        )

    print("Extended eval coverage summary:")
    print(f"- stage5 extension cases: {len(stage5_cases)}")
    print(f"- critical extension scenarios: {len(critical_cases)}")
    print(f"- follow-up scenarios: {follow_up_scenarios}")
    print(f"- non-Samara turns: {non_samara_count}")
    for label in sorted(MIN_STAGE5_LABELS):
        print(f"- {label}: {stage5_label_counts.get(label, 0)}")

    if errors:
        print("")
        print("COVERAGE CHECK FAILED")
        for i, err in enumerate(errors, start=1):
            print(f"{i}. {err}")
        return 1

    print("")
    print("COVERAGE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
