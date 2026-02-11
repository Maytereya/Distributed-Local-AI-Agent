"""Корпусный eval Stage 5 по golden JSONL.

Считает ключевые метрики качества роутинга (intent/handoff/slot/unsafe),
печатает by-label статистику и причины провалов для сравнения baseline/итераций.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN = ROOT / "messengers_router" / "messengers_mds_to_collect_thoughts" / "analysis" / "stage5_golden_cases.jsonl"

# Для прокси-оценки заполнения слотов на 1-й реплике
REQUIRED_SLOTS: dict[str, list[str]] = {
    "APPOINTMENT": ["_any_of:doctor_id,doctor_name,specialty,service_name", "_any_of:city,branch_name,branch_id"],
    "TEST_ASSIST": ["_any_of:city,branch_name,branch_id", "_any_of:test_goal,test_name"],
    "TEST_RESULT": ["surname", "year", "filial", "number"],
    "DOCTOR_INFO": ["_any_of:specialty,doctor_id,doctor_name"],
    "DOCTOR_SCHEDULE": ["_any_of:doctor_id,doctor_name"],
    "PRICE": ["_any_of:city,branch_name,branch_id", "service_name"],
    "ADDRESS": ["_any_of:city,branch_name,branch_id"],
    "PREPARE": ["_any_of:test_name,service_name"],
}

UNSAFE_LABELS = {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}


def post_json(url: str, payload: dict, timeout_sec: int = 50) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def preflight_endpoint(url: str) -> tuple[bool, str]:
    payload = {
        "session_id": f"s_eval_stage5_preflight_{int(time.time())}",
        "text": "привет",
        "debug": True,
    }
    try:
        data = post_json(url, payload, timeout_sec=8)
    except urllib.error.HTTPError as e:
        return False, f"http_error={e.code}"
    except urllib.error.URLError as e:
        return False, f"transport_error={e.reason}"
    except TimeoutError:
        return False, "timeout_error"
    except json.JSONDecodeError:
        return False, "json_decode_error"
    except Exception as e:  # noqa: BLE001
        return False, f"unexpected_error={type(e).__name__}"

    if not isinstance(data, dict):
        return False, "response_not_json_object"
    if "text" not in data or "handoff" not in data:
        return False, "response_missing_required_fields"
    return True, "ok"


def load_golden(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _slots_satisfied(label: str, entities: dict[str, Any]) -> tuple[int, int]:
    req = REQUIRED_SLOTS.get(label, [])
    if not req:
        return 0, 0
    sat = 0
    for r in req:
        if r.startswith("_any_of:"):
            keys = [k.strip() for k in r.split(":", 1)[1].split(",") if k.strip()]
            if any(entities.get(k) for k in keys):
                sat += 1
        elif entities.get(r):
            sat += 1
    return sat, len(req)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate stage-5 golden corpus")
    p.add_argument("--url", default="http://localhost:8000/api/messenger-generate-once", help="Debug endpoint URL")
    p.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="Golden JSONL path")
    p.add_argument("--session-prefix", default="s_eval_stage5", help="Session prefix")
    p.add_argument("--run-id", default="", help="Optional run id (default unix ts)")
    args = p.parse_args()

    golden_path = Path(args.golden).resolve()
    if not golden_path.exists():
        print(f"[error] golden file not found: {golden_path}")
        return 2
    cases = load_golden(golden_path)
    if not cases:
        print(f"[error] golden file is empty: {golden_path}")
        return 2

    preflight_ok, preflight_note = preflight_endpoint(args.url)
    if not preflight_ok:
        print(f"[error] endpoint preflight failed: {preflight_note}")
        print(f"[hint] start API first, e.g. python -m uvicorn agent_api:app --host 0.0.0.0 --port 8000 --reload")
        return 2

    run_id = args.run_id.strip() or str(int(time.time()))
    total = len(cases)
    intent_ok = 0
    handoff_ok = 0
    false_handoff = 0
    unsafe_total = 0
    unsafe_miss = 0
    slot_sat_total = 0
    slot_req_total = 0
    transport_errors = 0

    by_label_total: Counter[str] = Counter()
    by_label_pass: Counter[str] = Counter()
    fail_reasons: Counter[str] = Counter()

    print(f"Evaluating stage-5 corpus: {total} cases against {args.url}")
    print(f"Golden: {golden_path}")
    print(f"Run id: {run_id}")
    print("-" * 180)
    print(
        f"{'ID':<6} {'ExpLabel':<14} {'ActLabel':<14} {'ExpHO':<6} {'ActHO':<6} "
        f"{'Conf':<6} {'Slot':<8} {'Pass':<5} {'TopFlags'}"
    )
    print("-" * 180)

    for i, c in enumerate(cases, 1):
        case_id = str(c.get("case_id") or f"C{i:03d}")
        text = str(c.get("text") or "")
        exp_label = str(c.get("expected_label") or "OTHER")
        exp_handoff = bool(c.get("expected_handoff", False))
        by_label_total[exp_label] += 1

        payload = {
            "session_id": f"{args.session_prefix}_{run_id}_{case_id}",
            "text": text,
            "debug": True,
        }
        act_label = "ERROR"
        act_handoff = False
        conf = 0.0
        top_flags: list[str] = []
        slot_info = "-"

        try:
            data = post_json(args.url, payload)
            act_handoff = bool(data.get("handoff", False))
            debug = data.get("state_update", {}).get("debug", {})
            decision = debug.get("decision", {}) if isinstance(debug, dict) else {}
            act_label = str(decision.get("label") or "UNKNOWN")
            try:
                conf = float(decision.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            flags = decision.get("flags") if isinstance(decision, dict) else []
            if isinstance(flags, list):
                top_flags = [str(x) for x in flags[:5]]
            ents = debug.get("last_entities", {}) if isinstance(debug, dict) else {}
            if not isinstance(ents, dict):
                ents = {}
            sat, req = _slots_satisfied(exp_label, ents)
            if req > 0:
                slot_sat_total += sat
                slot_req_total += req
                slot_info = f"{sat}/{req}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            transport_errors += 1
            fail_reasons["transport_or_json_error"] += 1

        label_match = act_label == exp_label
        handoff_match = act_handoff == exp_handoff
        case_pass = label_match and handoff_match
        if case_pass:
            by_label_pass[exp_label] += 1
        else:
            if not label_match:
                fail_reasons["label_mismatch"] += 1
            if not handoff_match:
                fail_reasons["handoff_mismatch"] += 1

        if label_match:
            intent_ok += 1
        if handoff_match:
            handoff_ok += 1
        if (not exp_handoff) and act_handoff:
            false_handoff += 1

        if exp_label in UNSAFE_LABELS:
            unsafe_total += 1
            if act_label != exp_label:
                unsafe_miss += 1

        print(
            f"{case_id:<6} {exp_label:<14} {act_label:<14} {str(exp_handoff):<6} {str(act_handoff):<6} "
            f"{conf:<6.2f} {slot_info:<8} {str(case_pass):<5} {','.join(top_flags) if top_flags else '-'}"
        )

    intent_accuracy = (intent_ok / total) * 100
    handoff_accuracy = (handoff_ok / total) * 100
    false_handoff_rate = (false_handoff / total) * 100
    unsafe_miss_rate = (unsafe_miss / unsafe_total * 100) if unsafe_total else 0.0
    slot_fill_rate = (slot_sat_total / slot_req_total * 100) if slot_req_total else 0.0

    print("-" * 180)
    print(f"intent_accuracy:     {intent_ok}/{total} = {intent_accuracy:.1f}%")
    print(f"handoff_accuracy:    {handoff_ok}/{total} = {handoff_accuracy:.1f}%")
    print(f"false_handoff_rate:  {false_handoff}/{total} = {false_handoff_rate:.1f}%")
    print(f"slot_fill_rate:      {slot_sat_total}/{slot_req_total} = {slot_fill_rate:.1f}%")
    if unsafe_total:
        print(f"unsafe_miss_rate:    {unsafe_miss}/{unsafe_total} = {unsafe_miss_rate:.1f}%")
    else:
        print("unsafe_miss_rate:    n/a (no unsafe labels in current golden)")
    print("")
    print("By-label accuracy:")
    for lbl, cnt in sorted(by_label_total.items(), key=lambda x: x[0]):
        ok = by_label_pass.get(lbl, 0)
        acc = (ok / cnt) * 100 if cnt else 0.0
        print(f"- {lbl}: {ok}/{cnt} ({acc:.1f}%)")
    print("")
    print("Failure reasons:")
    if fail_reasons:
        for k, v in fail_reasons.most_common():
            print(f"- {k}: {v}")
    else:
        print("- none")
    if transport_errors:
        print(f"transport_errors: {transport_errors}")

    # Базовый gate: intent >= 85%, handoff >= 85%, transport/json ошибок нет.
    gate_ok = intent_accuracy >= 85.0 and handoff_accuracy >= 85.0 and transport_errors == 0
    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
