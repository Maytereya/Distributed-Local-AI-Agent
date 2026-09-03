#!/usr/bin/env python3
"""Remote turn-by-turn eval for FreeTalk debug API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _default_api_key() -> str:
    env_value = os.getenv("AGENT_API_KEY", "").strip()
    if env_value:
        return env_value
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        import agent_logic_2.config as c
    except Exception:
        return ""
    return str(getattr(c, "AGENT_API_KEY", "") or "").strip()


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_sec: int,
    api_key: str,
    host_header: str,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    if host_header:
        headers["Host"] = host_header
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                items.append(value)
    return items


def _norm(value: str) -> str:
    return str(value or "").strip().lower()


def _contains_all(text: str, patterns: list[str]) -> bool:
    hay = _norm(text)
    for pattern in patterns:
        needle = _norm(pattern)
        if needle and needle not in hay:
            return False
    return True


def _contains_any(text: str, patterns: list[str]) -> bool:
    hay = _norm(text)
    for pattern in patterns:
        needle = _norm(pattern)
        if needle and needle in hay:
            return True
    return False


def _metric_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 1.0
    return float(num) / float(den)


def _session_tag_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_case[str(row["case_id"])].append(row)

    tag_totals: Counter[str] = Counter()
    tag_pass: Counter[str] = Counter()
    for case_rows in by_case.values():
        tags = set()
        for row in case_rows:
            tags.update(row.get("tags") or [])
        passed = all(bool(row.get("pass_step")) for row in case_rows)
        for tag in tags:
            tag_totals[tag] += 1
            if passed:
                tag_pass[tag] += 1

    return {
        "multiturn_session_rate": _metric_ratio(tag_pass["multiturn"], tag_totals["multiturn"]),
        "clarification_session_rate": _metric_ratio(tag_pass["clarification"], tag_totals["clarification"]),
        "memory_session_rate": _metric_ratio(tag_pass["memory"], tag_totals["memory"]),
        "selferrorcorrection_session_rate": _metric_ratio(
            tag_pass["selferrorcorrection"],
            tag_totals["selferrorcorrection"],
        ),
        "flexibility_session_rate": _metric_ratio(tag_pass["flexibility"], tag_totals["flexibility"]),
    }


def _compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    step_pass = sum(1 for row in results if bool(row.get("pass_step")))
    clarification_cases = [
        row for row in results if row.get("clarification_expected") is not None
    ]
    clarification_ok = [
        row
        for row in clarification_cases
        if bool(row.get("clarification")) == bool(row.get("clarification_expected"))
    ]

    session_status: dict[str, bool] = {}
    for row in results:
        case_id = str(row["case_id"])
        session_status[case_id] = bool(row.get("pass_step")) and session_status.get(case_id, True)

    metrics = {
        "steps_total": float(len(results)),
        "sessions_total": float(len(session_status)),
        "step_pass_rate": _metric_ratio(step_pass, len(results)),
        "session_pass_rate": _metric_ratio(sum(1 for ok in session_status.values() if ok), len(session_status)),
        "clarification_success_rate": _metric_ratio(len(clarification_ok), len(clarification_cases)),
    }
    metrics.update(_session_tag_metrics(results))
    return metrics


def _load_thresholds(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _check_gate(metrics: dict[str, float], thresholds: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    min_steps = int(thresholds.get("min_steps", 1))
    if int(metrics.get("steps_total", 0)) < min_steps:
        errors.append(f"steps_total < min_steps ({int(metrics.get('steps_total', 0))} < {min_steps})")

    for key, threshold_key in (
        ("step_pass_rate", "step_pass_rate_min"),
        ("session_pass_rate", "session_pass_rate_min"),
        ("clarification_success_rate", "clarification_success_rate_min"),
        ("multiturn_session_rate", "multiturn_session_rate_min"),
        ("clarification_session_rate", "clarification_session_rate_min"),
        ("memory_session_rate", "memory_session_rate_min"),
        ("selferrorcorrection_session_rate", "selferrorcorrection_session_rate_min"),
        ("flexibility_session_rate", "flexibility_session_rate_min"),
    ):
        minimum = float(thresholds.get(threshold_key, 0.0))
        if metrics.get(key, 0.0) < minimum:
            errors.append(f"{key} {metrics.get(key, 0.0):.3f} < {minimum:.3f}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run remote eval against FT debug API")
    parser.add_argument("--url", default="http://172.16.0.28/v1/freetalk/generate-once", help="FT endpoint URL")
    parser.add_argument("--api-key", default=_default_api_key(), help="X-API-Key value")
    parser.add_argument("--host-header", default="", help="Optional Host header")
    parser.add_argument("--timeout-sec", type=int, default=60, help="HTTP timeout per turn")
    parser.add_argument(
        "--cases",
        default=str((root / "remote_cases.jsonl").resolve()),
        help="JSONL with FT remote eval cases",
    )
    parser.add_argument(
        "--thresholds",
        default=str((root / "thresholds.json").resolve()),
        help="Quality gate thresholds JSON",
    )
    parser.add_argument("--session-prefix", default="ft_remote_eval", help="Session id prefix")
    parser.add_argument("--run-id", default="", help="Optional run id")
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    thresholds_path = Path(args.thresholds).resolve()
    if not cases_path.exists():
        print(f"[error] cases file not found: {cases_path}")
        return 2
    if not thresholds_path.exists():
        print(f"[error] thresholds file not found: {thresholds_path}")
        return 2
    if not args.api_key:
        print("[error] api key is empty; pass --api-key or set AGENT_API_KEY")
        return 2

    sessions = _load_jsonl(cases_path)
    thresholds = _load_thresholds(thresholds_path)
    run_id = args.run_id.strip() or str(int(time.time()))

    print(f"FT remote eval: {len(sessions)} cases against {args.url}")
    print(f"Cases file: {cases_path}")
    print(f"Thresholds: {thresholds_path}")
    print(f"Run id: {run_id}")
    print("-" * 180)
    print(
        f"{'Case':<30} {'Step':<4} {'ReplyKind':<13} {'Source':<18} {'Tool':<24} "
        f"{'Pass':<5} {'Reason':<28} {'BOT (trim)'}"
    )
    print("-" * 180)

    results: list[dict[str, Any]] = []
    fail_reasons: Counter[str] = Counter()

    for idx, session in enumerate(sessions, start=1):
        case_id = str(session.get("case_id") or session.get("session_id") or f"FT_CASE_{idx:03d}")
        tags = [str(x).strip() for x in (session.get("tags") or []) if str(x).strip()]
        steps = session.get("steps") if isinstance(session.get("steps"), list) else []
        current_session_id = f"{args.session_prefix}_{run_id}_{case_id}"

        for step_index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            message = str(step.get("message") or "").strip()
            expected = step.get("expected") if isinstance(step.get("expected"), dict) else {}
            required_all = [str(x) for x in (expected.get("must_contain") or []) if str(x).strip()]
            forbidden_any = [str(x) for x in (expected.get("must_not_contain") or []) if str(x).strip()]
            expected_source = _norm(str(expected.get("source") or ""))
            expected_tool_called = expected.get("tool_called_expected")
            clarification_expected = expected.get("clarification_expected")
            expected_reply_kind = _norm(str(expected.get("reply_kind") or ""))

            reason = "-"
            response_text = ""
            source = ""
            tool_name = ""
            reply_kind = ""
            clarification = False
            step_pass = True

            payload = {
                "session_id": current_session_id,
                "text": message,
                "debug": True,
            }
            try:
                data = _post_json(
                    args.url,
                    payload,
                    timeout_sec=args.timeout_sec,
                    api_key=args.api_key,
                    host_header=args.host_header,
                )
                response_text = str(data.get("text") or "")
                source = str(data.get("source") or "")
                tool_name = str(data.get("tool_name") or "")
                reply_kind = str(data.get("reply_kind") or "")
                clarification = _norm(reply_kind) == "clarify"
                next_session_id = str(data.get("next_session_id") or "").strip()
                if next_session_id:
                    current_session_id = next_session_id
            except urllib.error.HTTPError as exc:
                step_pass = False
                reason = f"http_error_{exc.code}"
                fail_reasons[reason] += 1
            except urllib.error.URLError:
                step_pass = False
                reason = "transport_error"
                fail_reasons[reason] += 1
            except TimeoutError:
                step_pass = False
                reason = "timeout_error"
                fail_reasons[reason] += 1
            except json.JSONDecodeError:
                step_pass = False
                reason = "json_decode_error"
                fail_reasons[reason] += 1

            if step_pass and required_all and not _contains_all(response_text, required_all):
                step_pass = False
                reason = "must_contain_missing"
                fail_reasons[reason] += 1
            if step_pass and forbidden_any and _contains_any(response_text, forbidden_any):
                step_pass = False
                reason = "forbidden_phrase_hit"
                fail_reasons[reason] += 1
            if step_pass and expected_source and _norm(source) != expected_source:
                step_pass = False
                reason = f"source_mismatch({_norm(source) or '-'})"
                fail_reasons["source_mismatch"] += 1
            if step_pass and expected_tool_called is not None and bool(tool_name) != bool(expected_tool_called):
                step_pass = False
                reason = f"tool_called_mismatch({bool(tool_name)})"
                fail_reasons["tool_called_mismatch"] += 1
            if step_pass and clarification_expected is not None and clarification != bool(clarification_expected):
                step_pass = False
                reason = f"clarification_mismatch({clarification})"
                fail_reasons["clarification_mismatch"] += 1
            if step_pass and expected_reply_kind and _norm(reply_kind) != expected_reply_kind:
                step_pass = False
                reason = f"reply_kind_mismatch({_norm(reply_kind) or '-'})"
                fail_reasons["reply_kind_mismatch"] += 1

            trim = response_text.replace("\n", " ")[:100]
            print(
                f"{case_id:<30} {step_index:<4} {reply_kind or '-':<13} {source or '-':<18} "
                f"{tool_name or '-':<24} {str(step_pass):<5} {reason:<28} {trim}"
            )
            results.append(
                {
                    "case_id": case_id,
                    "step_index": step_index,
                    "tags": tags,
                    "pass_step": step_pass,
                    "clarification": clarification,
                    "clarification_expected": clarification_expected,
                }
            )

    print("-" * 180)
    metrics = _compute_metrics(results)
    for key in (
        "steps_total",
        "sessions_total",
        "step_pass_rate",
        "session_pass_rate",
        "clarification_success_rate",
        "multiturn_session_rate",
        "clarification_session_rate",
        "memory_session_rate",
        "selferrorcorrection_session_rate",
        "flexibility_session_rate",
    ):
        value = metrics.get(key, 0.0)
        if key.endswith("_total"):
            print(f"{key}={int(value)}")
        else:
            print(f"{key}={value:.3f}")

    errors = _check_gate(metrics, thresholds)
    print("")
    if fail_reasons:
        print("Diagnostics:")
        for name, count in fail_reasons.most_common():
            print(f"- {name}: {count}")
        print("")

    if errors:
        print("QUALITY GATE: FAILED")
        for line in errors:
            print(f"- {line}")
        return 2

    print("QUALITY GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
