"""Генератор golden-корпуса Stage 5 из анализированных чат-сэмплов.

Читает агрегированный JSON с примерами, маппит source intent в канонические labels
и сохраняет JSONL-набор кейсов для регрессионного eval.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "messengers_mds_to_collect_thoughts" / "analysis" / "_tmp_intent_samples.json"
DEFAULT_OUT = ROOT / "messengers_mds_to_collect_thoughts" / "analysis" / "stage5_golden_cases.jsonl"


@dataclass(frozen=True)
class LabelMap:
    source: str
    target: str
    handoff: bool


MAPPING: dict[str, LabelMap] = {
    "PRICE": LabelMap("PRICE", "PRICE", False),
    "APPOINTMENT": LabelMap("APPOINTMENT", "APPOINTMENT", False),
    "TEST_ASSIST": LabelMap("TEST_ASSIST", "TEST_ASSIST", False),
    "RESULTS": LabelMap("RESULTS", "TEST_RESULT", False),
    "ADDRESS": LabelMap("ADDRESS", "ADDRESS", False),
    "DISCOUNT": LabelMap("DISCOUNT", "NEWS", False),
    "CERTIFICATE": LabelMap("CERTIFICATE", "OTHER", True),
}


def _is_meaningful_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # отсекаем приветствия/прощания без смысла
    if re.fullmatch(r"(привет|здравствуйте|добрый день|добрый вечер|доброе утро|спасибо|хорошо)[!.,\s]*", t, re.I):
        return False
    # отсекаем совсем короткое
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", t)
    return len(words) >= 3


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        key = re.sub(r"\s+", " ", x.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x.strip())
    return out


def build_cases(source_path: Path, per_intent: int) -> list[dict]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    cases: list[dict] = []
    cid = 1

    for src_label, lm in MAPPING.items():
        bucket = raw.get(src_label, {})
        patient_items = bucket.get("patient", []) if isinstance(bucket, dict) else []
        if not isinstance(patient_items, list):
            continue
        patient_texts = [str(x) for x in patient_items if isinstance(x, str)]
        patient_texts = _dedupe_keep_order(patient_texts)
        patient_texts = [x for x in patient_texts if _is_meaningful_text(x)]
        picked = patient_texts[:per_intent]

        for text in picked:
            cases.append(
                {
                    "case_id": f"G{cid:03d}",
                    "source_label": lm.source,
                    "expected_label": lm.target,
                    "expected_handoff": lm.handoff,
                    "text": text,
                }
            )
            cid += 1

    return cases


def write_jsonl(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Build Stage-5 golden corpus from analyzed chat samples")
    p.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to _tmp_intent_samples.json")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    p.add_argument("--per-intent", type=int, default=12, help="Max cases per source intent")
    args = p.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    cases = build_cases(source, max(1, int(args.per_intent)))
    write_jsonl(out, cases)
    print(f"Built {len(cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
