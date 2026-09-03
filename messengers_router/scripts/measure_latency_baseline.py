"""Замер клиентской латентности прод-эндпоинта по фиксированному набору проб.

Зачем (П3 дорожной карты, architecture_review_2026-06-26 §7): снять p50/p95
ДО инфра-изменений (квант ollama, NUM_PARALLEL — П4) и ПОСЛЕ них — честное
сравнение на одном и том же наборе. Пробы делятся на два класса:
  - rule-путь: детерминированное правило короткозамыкает LLM (быстро);
  - llm-путь: rule=None → LLM classify + LLM render (медленно; именно его чинит квант).

Запуск с Mac (сервер режет по source-IP, эндпоинт с Mac доступен):
    ./venv/bin/python messengers_router/scripts/measure_latency_baseline.py \
        --url http://172.16.0.28/api/messenger-generate-once --reps 3 \
        --out /tmp/latency_baseline.json

Последовательно (не грузим прод), свежий session_id на пробу (без pending-заражения).
Пер-стейдж разбивка (stage_timings) появится в debug-ответе после деплоя коммита,
прокидывающего ctx.stage_timings, — скрипт подхватит её автоматически.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

import requests

# (класс, текст) — класс по архитектуре NLU: rule-путь = правило не-None (см.
# classifier.deterministic_rule_decision), llm-путь = rule=None → classify+render LLM.
PROBES: list[tuple[str, str]] = [
    ("rule", "адреса филиалов"),
    ("rule", "график работы филиалов"),
    ("rule", "сколько стоит общий анализ крови"),
    ("rule", "где сдать анализы"),
    ("rule", "Код 5437"),
    ("llm", "какую модель ты используешь"),
    ("llm", "у вас есть парковка возле клиники?"),
    ("llm", "расскажите о вашей клинике"),
]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://172.16.0.28/api/messenger-generate-once")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows: list[dict] = []
    for rep in range(args.reps):
        for kind, text in PROBES:
            sid = f"lat_{uuid.uuid4().hex[:10]}"
            t0 = time.monotonic()
            stage_timings = None
            ok = False
            try:
                r = requests.post(
                    args.url,
                    json={"session_id": sid, "text": text, "debug": True},
                    timeout=args.timeout,
                )
                ok = r.status_code == 200
                try:
                    dbg = ((r.json().get("state_update") or {}).get("debug") or {})
                    stage_timings = dbg.get("stage_timings")
                except Exception:
                    pass
            except Exception:
                pass
            dt = time.monotonic() - t0
            rows.append(
                {
                    "rep": rep,
                    "kind": kind,
                    "text": text,
                    "ok": ok,
                    "seconds": round(dt, 2),
                    "stage_timings": stage_timings,
                }
            )
            print(f"[{rep}] {kind:<4} {dt:6.2f}s ok={ok} {text}")
            time.sleep(0.5)

    print("\n=== СВОДКА (сек) ===")
    summary: dict[str, dict[str, float]] = {}
    for kind in ("rule", "llm"):
        vals = [r["seconds"] for r in rows if r["kind"] == kind and r["ok"]]
        if not vals:
            print(f"{kind}: нет успешных проб!")
            continue
        summary[kind] = {
            "n": len(vals),
            "p50": round(statistics.median(vals), 2),
            "p95": round(_pct(vals, 95), 2),
            "max": round(max(vals), 2),
        }
        print(f"{kind:<4} n={len(vals):<3} p50={summary[kind]['p50']:<7} "
              f"p95={summary[kind]['p95']:<7} max={summary[kind]['max']}")

    # Пер-стейдж сводка — когда прод начнёт отдавать stage_timings.
    stage_rows = [r["stage_timings"] for r in rows if isinstance(r.get("stage_timings"), dict)]
    if stage_rows:
        stages = sorted({k for st in stage_rows for k in st})
        print("\n=== stage_timings, ms (p50) ===")
        for s in stages:
            vals = [float(st[s]) for st in stage_rows if s in st]
            print(f"  {s:<20} p50={statistics.median(vals):.0f}")
    else:
        print("\n(stage_timings прод пока не отдаёт — появится после деплоя debug-обвязки)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "summary": summary, "ts": time.time()}, fh, ensure_ascii=False, indent=1)
        print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
