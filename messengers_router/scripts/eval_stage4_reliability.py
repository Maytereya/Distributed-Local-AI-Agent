"""Smoke-тест надежности Stage 4.

Проверяет, что endpoint стабильно возвращает HTTP 200 и валидный JSON envelope
на наборе реплик даже при частичной деградации внешних зависимостей.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass


@dataclass
class Case:
    case_id: str
    text: str


CASES: list[Case] = [
    Case("RLY_01", "Привет"),
    Case("RLY_02", "Запишите на ЭКГ в Самаре"),
    Case("RLY_03", "Самара"),
    Case("RLY_04", "Победы 83"),
    Case("RLY_05", "10 февраля 14:00"),
    Case("RLY_06", "Сколько стоит УЗИ печени в Самаре?"),
    Case("RLY_07", "Адрес филиала в Оренбурге"),
    Case("RLY_08", "Нужна подготовка к ФГДС"),
    Case("RLY_09", "Результаты анализов готовы?"),
    Case("RLY_10", "Пришлите PDF результатов"),
    Case("RLY_11", "Мне нужна справка для налогового вычета"),
    Case("RLY_12", "Хочу оставить жалобу на администратора"),
    Case("RLY_13", "Я задыхаюсь и сильная боль в груди"),
    Case("RLY_14", "Можно перенести запись на холтер?"),
    Case("RLY_15", "нет"),
]


def post_json(url: str, payload: dict, timeout_sec: int = 45) -> tuple[int, dict]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        status = getattr(resp, "status", 200)
        raw = resp.read().decode("utf-8")
    return status, json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-4 reliability smoke for messenger endpoint")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/messenger-generate-once",
        help="Debug endpoint URL",
    )
    parser.add_argument(
        "--session-prefix",
        default="s_eval_stage4_rel",
        help="Session prefix for requests",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id to isolate sessions across repeated launches. Default: unix timestamp.",
    )
    args = parser.parse_args()

    run_id = args.run_id.strip() or str(int(time.time()))
    total = len(CASES)
    passed = 0
    reason_counts: Counter[str] = Counter()

    print(f"Reliability smoke: {total} cases against {args.url}")
    print(f"Run id: {run_id}")
    print("-" * 140)
    print(f"{'ID':<8} {'HTTP':<6} {'JSON':<6} {'Envelope':<9} {'Pass':<6} {'Notes'}")
    print("-" * 140)

    for idx, case in enumerate(CASES, 1):
        session_id = f"{args.session_prefix}_{run_id}_{idx}"
        payload = {"session_id": session_id, "text": case.text, "debug": True}

        http_status = 0
        json_ok = False
        envelope_ok = False
        note = "-"

        try:
            http_status, data = post_json(args.url, payload)
            json_ok = isinstance(data, dict)
            if json_ok:
                has_text = "text" in data
                has_handoff = "handoff" in data
                has_state_update = "state_update" in data
                envelope_ok = has_text and has_handoff and has_state_update
                if not envelope_ok:
                    note = "missing envelope keys"
                    reason_counts["envelope_shape_error"] += 1
                debug = data.get("state_update", {}).get("debug", {})
                if isinstance(debug, dict) and debug.get("route_error"):
                    reason_counts["route_error_fallback"] += 1
                    note = f"route_error={debug.get('route_error')}"
            else:
                reason_counts["json_not_object"] += 1
                note = "json_not_object"
        except urllib.error.HTTPError as e:
            http_status = e.code
            reason_counts[f"http_{e.code}"] += 1
            note = f"http_error={e.code}"
        except urllib.error.URLError:
            reason_counts["transport_error"] += 1
            note = "transport_error"
        except TimeoutError:
            reason_counts["timeout_error"] += 1
            note = "timeout_error"
        except json.JSONDecodeError:
            reason_counts["json_decode_error"] += 1
            note = "json_decode_error"

        ok = http_status == 200 and json_ok and envelope_ok
        if ok:
            passed += 1

        print(
            f"{case.case_id:<8} {str(http_status):<6} {str(json_ok):<6} "
            f"{str(envelope_ok):<9} {str(ok):<6} {note}"
        )

    pct = (passed / total) * 100 if total else 0.0
    print("-" * 140)
    print(f"Passed: {passed}/{total} ({pct:.1f}%)")
    print("Target: 100% (no 500/transport/json envelope failures)")
    print("")
    print("Diagnostics:")
    if reason_counts:
        for name, count in reason_counts.most_common():
            print(f"- {name}: {count}")
    else:
        print("- none")

    return 0 if passed == total else 2


if __name__ == "__main__":
    sys.exit(main())
