"""Критичные turn-by-turn проверки для удаленного production endpoint.

Проверяет сценарии, где нельзя допускать отклонений:
- целевой label/handoff,
- обязательные паттерны в ответе,
- запрещенные паттерны (например, уход в ADDRESS-flow).

Назначение: быстрый quality-gate для "серьезных" регрессий логики.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int = 50, host_header: str = "") -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if host_header:
        headers["Host"] = host_header
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _contains_any(text: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    low = (text or "").lower()
    for p in patterns:
        needle = str(p or "").strip().lower()
        if needle and needle in low:
            return True
    return False


def _contains_forbidden(text: str, patterns: list[str]) -> str | None:
    if not patterns:
        return None
    low = (text or "").lower()
    for p in patterns:
        needle = str(p or "").strip().lower()
        if needle and needle in low:
            return needle
    return None


def _load_cases(path: Path) -> list[dict[str, Any]]:
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


def _build_turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    case_expected_label = case.get("expected_label")
    case_expected_handoff = case.get("expected_handoff")
    case_required = case.get("required_any") if isinstance(case.get("required_any"), list) else []
    case_forbidden = case.get("forbidden_any") if isinstance(case.get("forbidden_any"), list) else []

    turns_raw = case.get("turns")
    if isinstance(turns_raw, list) and turns_raw:
        turns: list[dict[str, Any]] = []
        for t in turns_raw:
            if not isinstance(t, dict):
                continue
            turn = dict(t)
            if "expected_label" not in turn and case_expected_label is not None:
                turn["expected_label"] = case_expected_label
            if "expected_handoff" not in turn and case_expected_handoff is not None:
                turn["expected_handoff"] = case_expected_handoff
            if "required_any" not in turn and case_required:
                turn["required_any"] = list(case_required)
            if "forbidden_any" not in turn and case_forbidden:
                turn["forbidden_any"] = list(case_forbidden)
            turns.append(turn)
        return turns

    # one-turn shorthand
    text = str(case.get("text") or "").strip()
    if not text:
        return []
    return [
        {
            "text": text,
            "expected_label": case_expected_label,
            "expected_handoff": case_expected_handoff,
            "required_any": list(case_required),
            "forbidden_any": list(case_forbidden),
        }
    ]


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run critical turn-by-turn checks for messenger endpoint")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/messenger-generate-once",
        help="Endpoint URL (recommended: /api/messenger-generate-once)",
    )
    parser.add_argument(
        "--cases",
        default=str((root / "critical_cases.jsonl").resolve()),
        help="Path to JSONL with critical cases",
    )
    parser.add_argument("--session-prefix", default="s_eval_critical", help="Session id prefix")
    parser.add_argument("--run-id", default="", help="Optional run id, default unix timestamp")
    parser.add_argument("--timeout-sec", type=int, default=50, help="HTTP timeout in seconds")
    parser.add_argument("--host-header", default="", help="Optional Host header")
    parser.add_argument("--llm-mode", default="hybrid", choices=["strict", "hybrid", "rich"], help="llm_mode in payload")
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    if not cases_path.exists():
        print(f"[error] cases file not found: {cases_path}")
        return 2

    cases = _load_cases(cases_path)
    if not cases:
        print(f"[error] cases file is empty: {cases_path}")
        return 2

    run_id = args.run_id.strip() or str(int(time.time()))
    total_turns = 0
    passed_turns = 0
    fail_reasons: Counter[str] = Counter()

    print(f"Critical eval: {len(cases)} cases against {args.url}")
    print(f"Cases file: {cases_path}")
    print(f"Run id: {run_id}")
    print("-" * 180)
    print(
        f"{'Case':<28} {'Turn':<4} {'ExpLabel':<12} {'ActLabel':<12} {'ExpHO':<6} {'ActHO':<6} "
        f"{'Pass':<5} {'Reason':<32} {'BOT (trim)'}"
    )
    print("-" * 180)

    for idx, case in enumerate(cases, 1):
        case_id = str(case.get("case_id") or f"CASE_{idx:03d}")
        turns = _build_turns(case)
        if not turns:
            fail_reasons["invalid_case_format"] += 1
            print(f"{case_id:<28} {'-':<4} {'-':<12} {'-':<12} {'-':<6} {'-':<6} {'False':<5} invalid_case_format {'-'}")
            continue

        session_id = f"{args.session_prefix}_{run_id}_{case_id}"

        for turn_no, turn in enumerate(turns, 1):
            total_turns += 1
            text = str(turn.get("text") or "").strip()
            exp_label_raw = turn.get("expected_label")
            exp_label = str(exp_label_raw).strip() if exp_label_raw is not None else ""
            exp_handoff = turn.get("expected_handoff")
            required_any = turn.get("required_any") if isinstance(turn.get("required_any"), list) else []
            forbidden_any = turn.get("forbidden_any") if isinstance(turn.get("forbidden_any"), list) else []

            payload: dict[str, Any] = {
                "session_id": session_id,
                "text": text,
                "debug": True,
                "llm_mode": args.llm_mode,
            }

            act_label = ""
            act_handoff = False
            bot_text = ""
            reason = ""

            try:
                data = _post_json(args.url, payload, timeout_sec=args.timeout_sec, host_header=args.host_header)
                bot_text = str(data.get("text") or "")
                act_handoff = bool(data.get("handoff", False))
                debug = data.get("state_update", {}).get("debug", {})
                decision = debug.get("decision", {}) if isinstance(debug, dict) else {}
                act_label = str(decision.get("label") or "").strip()
            except urllib.error.HTTPError as e:
                reason = f"http_error_{e.code}"
                fail_reasons[reason] += 1
            except urllib.error.URLError:
                reason = "transport_error"
                fail_reasons[reason] += 1
            except TimeoutError:
                reason = "timeout_error"
                fail_reasons[reason] += 1
            except json.JSONDecodeError:
                reason = "json_decode_error"
                fail_reasons[reason] += 1

            ok = True
            if reason:
                ok = False
            if ok and exp_label and act_label != exp_label:
                ok = False
                reason = f"label_mismatch({act_label or '-'})"
                fail_reasons["label_mismatch"] += 1
            if ok and isinstance(exp_handoff, bool) and act_handoff != exp_handoff:
                ok = False
                reason = f"handoff_mismatch({act_handoff})"
                fail_reasons["handoff_mismatch"] += 1
            if ok and required_any and not _contains_any(bot_text, required_any):
                ok = False
                reason = "required_any_missing"
                fail_reasons["required_any_missing"] += 1
            if ok:
                forbidden_hit = _contains_forbidden(bot_text, forbidden_any)
                if forbidden_hit is not None:
                    ok = False
                    reason = f"forbidden_hit({forbidden_hit})"
                    fail_reasons["forbidden_pattern_hit"] += 1

            if ok:
                passed_turns += 1
                reason = "-"

            trimmed = " ".join(bot_text.split())[:72]
            exp_handoff_str = str(exp_handoff) if isinstance(exp_handoff, bool) else "-"
            print(
                f"{case_id:<28} {turn_no:<4} {exp_label or '-':<12} {act_label or '-':<12} "
                f"{exp_handoff_str:<6} {str(act_handoff):<6} {str(ok):<5} {reason:<32} {trimmed}"
            )

    print("-" * 180)
    pct = (passed_turns / total_turns * 100.0) if total_turns else 0.0
    print(f"Passed: {passed_turns}/{total_turns} ({pct:.1f}%)")
    print("Gate: 100% pass for critical checks")
    print("")
    print("Failure reasons:")
    if fail_reasons:
        for name, count in fail_reasons.most_common():
            print(f"- {name}: {count}")
    else:
        print("- none")

    return 0 if total_turns > 0 and passed_turns == total_turns else 2


if __name__ == "__main__":
    sys.exit(main())
