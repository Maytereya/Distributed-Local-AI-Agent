"""Быстрый регрессионный eval для Stage 1 (20 контрольных кейсов).

Проверяет ожидаемый label/handoff, агрегирует причины провалов
и помогает контролировать стабильность базовой маршрутизации.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass


@dataclass
class Case:
    case_id: str
    text: str
    expect_label: str
    expect_handoff: bool


CASES: list[Case] = [
    Case("A1", "Хочу записаться на ЭКГ в Самаре", "APPOINTMENT", False),
    Case("A2", "Можно перенести запись на холтер на завтра?", "APPOINTMENT", False),
    Case("A3", "Отмените прием у невролога", "APPOINTMENT", False),
    Case("A4", "Нужна запись к кардиологу ребенку 4 года", "APPOINTMENT", False),
    Case("A5", "Запишите на УЗИ печени на Ленина 5", "APPOINTMENT", False),
    Case("A6", "Запись к Стрежневу на следующей неделе", "APPOINTMENT", False),
    Case("P1", "Сколько стоит ЭКГ в Самаре?", "PRICE", False),
    Case("P2", "Подскажите цену на ФГДС", "PRICE", False),
    Case("P3", "Какая стоимость приема уролога?", "PRICE", False),
    Case("P4", "Сколько стоит анализ на витамин Д?", "PRICE", False),
    Case("T1", "Какие анализы сдать на щитовидку?", "TEST_ASSIST", False),
    Case("T2", "Нужен чекап по анемии", "TEST_ASSIST", False),
    Case("T3", "Можно сдать ОАК и ферритин завтра?", "TEST_ASSIST", False),
    Case("D1", "Адрес филиала на Победы 83 и режим работы", "ADDRESS", False),
    Case("D2", "Где вы находитесь в Оренбурге?", "ADDRESS", False),
    Case("R1", "Результаты анализов готовы?", "TEST_RESULT", False),
    Case("R2", "Пришлите PDF бланк анализов", "TEST_RESULT", False),
    Case("C1", "Мне нужна справка для налогового вычета", "OTHER", True),
    Case("C2", "Нужна копия договора с печатью", "OTHER", True),
    Case("C3", "Как получить копию амбулаторной карты?", "OTHER", True),
]


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 messenger routing cases")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/messenger-generate-once",
        help="Debug endpoint URL",
    )
    parser.add_argument(
        "--session-prefix",
        default="s_eval_stage1",
        help="Session prefix for requests",
    )
    args = parser.parse_args()

    passed = 0
    total = len(CASES)
    reason_counts: Counter[str] = Counter()

    print(f"Evaluating {total} cases against {args.url}")
    print("-" * 168)
    print(
        f"{'ID':<4} {'Expected Label':<16} {'Actual Label':<16} "
        f"{'Exp Handoff':<11} {'Act Handoff':<11} {'Conf':<6} {'Flags':<70} {'Pass'}"
    )
    print("-" * 168)

    for i, case in enumerate(CASES, 1):
        session_id = f"{args.session_prefix}_{case.case_id}_{i}"
        payload = {"session_id": session_id, "text": case.text, "debug": True}
        try:
            data = post_json(args.url, payload)
            debug = data.get("state_update", {}).get("debug", {})
            decision = debug.get("decision", {})
            actual_label = (
                decision.get("label", "UNKNOWN")
            )
            actual_handoff = bool(data.get("handoff", False))
            conf_raw = decision.get("confidence", 0.0)
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = 0.0
            flags = decision.get("flags", [])
            if not isinstance(flags, list):
                flags = []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            actual_label = f"ERROR:{type(e).__name__}"
            actual_handoff = False
            confidence = 0.0
            flags = []
            reason_counts["transport_or_json_error"] += 1

        ok = actual_label == case.expect_label and actual_handoff == case.expect_handoff
        if ok:
            passed += 1
        else:
            if actual_label != case.expect_label:
                reason_counts["label_mismatch"] += 1
            if actual_handoff != case.expect_handoff:
                reason_counts["handoff_mismatch"] += 1
            for marker in ("ollama_timeout", "low_confidence", "doc_request_handoff", "test_result_fallback"):
                if marker in flags:
                    reason_counts[marker] += 1
        flags_str = ",".join(str(f) for f in flags[:6]) if flags else "-"
        print(
            f"{case.case_id:<4} {case.expect_label:<16} {actual_label:<16} "
            f"{str(case.expect_handoff):<11} {str(actual_handoff):<11} "
            f"{confidence:<6.2f} {flags_str:<70} {str(ok)}"
        )

    pct = (passed / total) * 100 if total else 0.0
    print("-" * 168)
    print(f"Passed: {passed}/{total} ({pct:.1f}%)")
    print("KPI target: >= 85.0%")
    print("")
    print("Failure reasons:")
    if reason_counts:
        fail_total = max(total - passed, 1)
        for name, count in reason_counts.most_common():
            share = (count / fail_total) * 100
            print(f"- {name}: {count} ({share:.1f}% of failed cases)")
    else:
        print("- none")

    return 0 if pct >= 85.0 else 2


if __name__ == "__main__":
    sys.exit(main())
