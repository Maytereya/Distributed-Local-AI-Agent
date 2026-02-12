"""Multi-turn regression для Stage 3 APPOINTMENT flow.

Эмулирует последовательность сообщений пациента по записи/переносу
и проверяет, что бот проходит шаги ветки без зацикливания.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Step:
    user_text: str
    expect_any: tuple[str, ...]
    expect_handoff: bool


@dataclass
class Flow:
    flow_id: str
    steps: list[Step]


FLOWS: list[Flow] = [
    Flow(
        "APPT_ECG_STD",
        steps=[
            Step("Я хочу записаться на ЭКГ", ("город", "из какого города"), False),
            Step("Самара", ("по адресам", "какой филиал"), False),
            Step("Победы 83", ("дату и время", "на какую дату"), False),
            Step("10 февраля 14:00", ("фио пациента",), False),
            Step("Иванов Иван Иванович", ("подтверждаете", "запись:"), False),
            Step("да", ("передаю заявку оператору",), True),
        ],
    ),
    Flow(
        "APPT_HOLTER_RESCHEDULE",
        steps=[
            Step("Можно перенести запись на холтер?", ("город", "из какого города"), False),
            Step("Самара", ("по адресам", "какой филиал"), False),
            Step("Ленина 5", ("дату и время", "на какую дату"), False),
            Step("завтра после 16:00", ("фио пациента",), False),
            Step("Петров Петр Петрович", ("подтверждаете", "запись:"), False),
            Step("нет", ("уточните новую дату",), False),
        ],
    ),
]


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=50) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    for p in patterns:
        if re.search(re.escape(p.lower()), t):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate stage-3 APPOINTMENT multi-turn flow")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/messenger-generate-once",
        help="Debug endpoint URL",
    )
    parser.add_argument(
        "--session-prefix",
        default="s_eval_stage3_appt",
        help="Session prefix for requests",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id to isolate sessions across repeated launches. Default: current unix timestamp.",
    )
    args = parser.parse_args()
    run_id = args.run_id.strip() or str(int(time.time()))

    total = 0
    passed = 0

    print(f"Evaluating {len(FLOWS)} APPOINTMENT flows against {args.url}")
    print(f"Run id: {run_id}")
    print("-" * 140)
    print(f"{'Flow':<24} {'Step':<4} {'Handoff':<9} {'Pass':<5} {'BOT (trim)'}")
    print("-" * 140)

    for flow in FLOWS:
        session_id = f"{args.session_prefix}_{run_id}_{flow.flow_id}"
        for idx, step in enumerate(flow.steps, 1):
            total += 1
            payload = {"session_id": session_id, "text": step.user_text, "debug": True}
            try:
                data = post_json(args.url, payload)
                bot_text = str(data.get("text") or "")
                handoff = bool(data.get("handoff", False))
                ok = _contains_any(bot_text, step.expect_any) and handoff == step.expect_handoff
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                bot_text = f"ERROR:{type(e).__name__}"
                handoff = False
                ok = False

            if ok:
                passed += 1

            trimmed = re.sub(r"\s+", " ", bot_text).strip()[:80]
            print(f"{flow.flow_id:<24} {idx:<4} {str(handoff):<9} {str(ok):<5} {trimmed}")

    pct = (passed / total) * 100 if total else 0.0
    print("-" * 140)
    print(f"Passed: {passed}/{total} ({pct:.1f}%)")
    print("Target: >= 90% for stage-3 flow stability")

    return 0 if pct >= 90.0 else 2


if __name__ == "__main__":
    sys.exit(main())
