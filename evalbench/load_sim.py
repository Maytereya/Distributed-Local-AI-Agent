"""Модель очереди к слотам LLM по реальным ходам прода: k ходов приходят разом (BUG-2026-10-27-LLM-QUEUE-GIVES-UP).

    python3 -I -m evalbench.load_sim <папка с turns_*.jsonl> [--slots 2] [--deadline 50] [--trials 3000]

Ходы — последовательности вызовов LLM с длительностями из журнала хода (`endpoint = stream`, без теней; копии журнала — вне
репо, без реплик). Политики:
- `сейчас` — ожидание слота по месту вызова (`queue_timeout_ms`: классификатор 30 с, решения 1,5–4 с), очередь FIFO; не дождался —
  вызов сдаётся, ход идёт путём сбоя (правила);
- `оракул` — дедлайн хода: ждать до «дедлайн − длительность этого вызова» (верхняя граница: длительность заранее известна), FIFO;
- `код` — как в `llm_runtime` под `llm_turn_deadline_s`: ждать до «дедлайн − p95 задачи» (`_P95_S`), слот — ходу, начатому раньше
  (EDF); генерация, не уложившаяся в остаток, обрывается дедлайном (тоже сбой).
Время вне LLM (МИС, рендер) не учтено; вызовы внутри хода — строго последовательно. Модель грубая — для порядка величин.
Первый прогон — 27.10 (`~/Dev/handoffs/2026-10-27_llm_load_model.md`), политика `код` — С1, 27.10.
"""

from __future__ import annotations

import argparse
import collections
import glob
import heapq
import json
import random

# Ожидание слота по месту вызова (`queue_timeout_ms` кода на 27.10); прочие — 2 с.
QUEUE_S = collections.defaultdict(
    lambda: 2.0,
    {
        "classifier": 30.0, "turn_spans": 1.5, "choose_role_pick": 2.0, "choose_pending_reply": 3.0,
        "choose_pending_free_reply": 3.0, "_service_normalizer": 2.0, "_empty_evidence_reply": 2.0, "_prepare_select_llm": 3.0,
        "choose_price_service": 3.0, "choose_price_service_general": 3.0, "choose_price_service_qualifier_general": 3.0,
        "choose_price_basket": 3.0, "choose_basket_total": 3.0, "choose_address_followup": 3.0, "_service_slot_validator": 2.0,
    },
)
# Запас p95 задачи — как `llm_runtime._P95_S` / `_P95_DEFAULT_S`.
P95_S = collections.defaultdict(lambda: 4.0, {"classifier": 12.5, "renderer": 9.0})


def load_turns(folder: str) -> list[list[tuple[str, float]]]:
    turns = []
    for path in sorted(glob.glob(folder + "/turns_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if "kind" in row or row.get("endpoint") != "stream":
                continue
            calls = [
                (c.get("site"), max(0.0, (c.get("ms", 0) - c.get("queue_ms", 0)) / 1000.0))
                for c in row.get("llm") or []
                if isinstance(c, dict) and not c.get("shadow")
            ]
            if calls:
                turns.append(calls)
    return turns


def simulate(turns, k: int, policy: str, *, slots: int = 2, deadline: float = 50.0, trials: int = 3000, seed: int = 7):
    """(ход p50, p95, max, доля сдавшихся вызовов, доля ходов со сдавшимся вызовом).

    События: вызов готов встать в очередь, слот освободился, ждущий сдался. Ходы пришли разом в порядке номера (ход 0 начат
    раньше всех): `код` отдаёт освободившийся слот ждущему с меньшим номером хода (EDF), прочие — первому пришедшему (FIFO).
    """

    rng = random.Random(seed)
    gave_up = calls_n = turns_failed = 0
    done: list[float] = []
    for _ in range(trials):
        sample = [rng.choice(turns) for _ in range(k)]
        free = slots
        waiting: dict[int, tuple[tuple[float, float], int, int]] = {}  # жетон → (приоритет, ход, вызов)
        events: list[tuple[float, int, str, tuple]] = []
        seq = 0

        def push(t: float, kind: str, data: tuple) -> None:
            nonlocal seq
            seq += 1
            heapq.heappush(events, (t, seq, kind, data))

        finish = [0.0] * k
        failed = [False] * k

        def begin(t: float, i: int, j: int) -> None:
            nonlocal free
            free -= 1
            dur = sample[i][j][1]
            end = t + dur
            if policy == "code" and end > deadline:  # генерацию обрывает дедлайн хода
                end = max(t, deadline)
                failed[i] = True
                counts["gave"] += 1
            push(end, "free", ())
            push(end, "ready", (i, j + 1))

        counts = {"gave": 0, "calls": 0}
        for i in range(k):
            push(0.0, "ready", (i, 0))
        while events:
            t, _seq, kind, data = heapq.heappop(events)
            if kind == "ready":
                i, j = data
                if j >= len(sample[i]):
                    finish[i] = t
                    continue
                site, dur = sample[i][j]
                counts["calls"] += 1
                if policy == "now":
                    limit = QUEUE_S[site]
                elif policy == "oracle":
                    limit = deadline - t - dur
                else:
                    limit = deadline - t - P95_S[site]
                if limit <= 0 and policy != "now":
                    counts["gave"] += 1
                    failed[i] = True
                    push(t, "ready", (i, j + 1))
                    continue
                if free > 0 and not waiting:
                    begin(t, i, j)
                    continue
                prio = (float(i), t) if policy == "code" else (t, float(i))
                waiting[seq] = (prio, i, j)
                push(t + limit, "giveup", (seq,))
            elif kind == "free":
                free += 1
                if waiting:
                    token = min(waiting, key=lambda tok: waiting[tok][0])
                    _prio, i, j = waiting.pop(token)
                    begin(t, i, j)
            else:
                (token,) = data
                if token in waiting:
                    _prio, i, j = waiting.pop(token)
                    counts["gave"] += 1
                    failed[i] = True
                    push(t, "ready", (i, j + 1))
        gave_up += counts["gave"]
        calls_n += counts["calls"]
        done += finish
        turns_failed += sum(failed)
    done.sort()

    def pq(x: float) -> float:
        return done[min(len(done) - 1, int(x * len(done)))]

    return pq(0.5), pq(0.95), done[-1], gave_up / max(1, calls_n), turns_failed / (trials * k)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--deadline", type=float, default=50.0)
    ap.add_argument("--trials", type=int, default=3000)
    args = ap.parse_args(argv)
    turns = load_turns(args.folder)
    print(f"ходов с вызовами LLM: {len(turns)}; слотов {args.slots}; дедлайн {args.deadline:g} с; {args.trials} прогонов на k")
    print("| k | политика | ход p50 / p95 / max, с | вызовов сдались | ходов со сдавшимся (→ правила / оффер) |")
    print("|---|---|---|---|---|")
    for k in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20):
        for policy, title in (("now", "сейчас"), ("oracle", "оракул"), ("code", "код")):
            p50, p95, mx, gave, failed = simulate(turns, k, policy, slots=args.slots, deadline=args.deadline, trials=args.trials)
            print(f"| {k} | {title} | {p50:.1f} / {p95:.1f} / {mx:.1f} | {gave * 100:.1f} % | {failed * 100:.1f} % |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
