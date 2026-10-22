"""Сводка телеметрии хода за окно — пороги «годен» нагрузочной проверки выбора (§8 runbook деплоя 2).

Зачем скрипт. 15.10 сводку на сервере набирали однострочником `python -c "..."`; длинная вставка в терминал дважды
теряла кусок строки (SyntaxError посреди проверки после включения `turn_parse`). Скрипт лежит в образе (`COPY . .`),
команда — одна короткая строка:

    docker exec agent-api python scripts/turn_telemetry_summary.py --since 2026-10-15T07:42:26

Читает `<APP_DATA_DIR>/journals/turns_*.jsonl` (последние два дня; `--journals` — другая папка), берёт ходы с `ts` не
раньше `--since` (UTC, Самара −4 ч) и печатает: ходы и p95 хода; решения выбора (`choose`) по задаче и статусу; p95
выбора не из кэша; UNAVAILABLE+INVALID; исходы и корневые причины вызовов LLM; p95 ожидания очереди; разбор реплики
(`tu_parse_status`) и долю запасного пути правил (T7); ценовые ходы — решения политики силы совпадения (L-13,
`price_policy`), кто судил уточнение П2 (`price_guard`, `strong_phrase_guard`, `p2_llm_judge`) и p95 ценового хода. Решения тени (`shadow`) в счёт не идут. Последняя строка —
вердикт по порогам «годен» 14.10: UNAVAILABLE+INVALID среди некэшированных < 2%, p95 выбора < 4 с, p95 `queue_ms` < 1 с,
нет `circuit_open`, пустой `err`, исход `budget` ≤ 2%.

Текста пациента в журнале ходов нет — только ключи и числа (`turn_telemetry`), вывод можно пересылать.
Коды возврата: 0 — годен, 1 — не годен, 2 — ходов в окне нет.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Iterable

# Поля телеметрии ценового хода: решение политики L-13 и судьи уточнения П2 (класс A, сильная фраза, сущность NLU).
_PRICE_FIELDS = ("price_policy", "price_guard", "strong_phrase_guard", "p2_llm_judge")

# Статусы разбора, при которых читатели идут путём правил (`turn_parse.acting_for` → None или сбой).
_FALLBACK = frozenset({"empty", "too_many", "merged", "unavailable", "invalid", "late", "error"})


def p95(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None


def load_rows(journals: str, since: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(journals, "turns_*.jsonl")))[-2:]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "kind" not in row and str(row.get("ts", "")) >= since:
                    rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """Строки сводки и вердикт «годен»."""

    choices = [c for r in rows for c in r.get("choose") or [] if not c.get("shadow")]
    calls = [x for r in rows for x in r.get("llm") or [] if not x.get("shadow")]
    live = [c for c in choices if not c.get("cached")]
    failed = sum(c["status"] in ("unavailable", "invalid") for c in live)
    errors = Counter(x["err"] for x in calls if x.get("err"))
    outcomes = Counter(x["outcome"] for x in calls)
    budget = outcomes.get("budget", 0)  # лимит решений хода: вызов не сделан (`llm_runtime`, исход `budget`)
    parse = Counter(r.get("tu_parse_status") for r in rows if r.get("tu_parse_status"))
    fallback = sum(n for status, n in parse.items() if status in _FALLBACK)
    choose_p95 = p95(c["ms"] for c in live)
    queue_p95 = p95(x.get("queue_ms", 0) for x in calls)

    def share(n: int, total: int) -> str:
        return f"{n} из {total} ({100 * n / total:.1f}%)" if total else "0 из 0"

    lines = [
        f"ходов {len(rows)} | p95 хода, мс {p95(r.get('ms', 0) for r in rows)}",
        f"выбор (задача, статус): {dict(Counter((c['task'], c['status']) for c in choices))}",
        f"p95 выбора не из кэша, мс {choose_p95} | UNAVAILABLE+INVALID {share(failed, len(live))} | budget {share(budget, len(calls))}",
        f"p95 выбора фраз (turn_phrases), мс {p95(c['ms'] for c in live if c['task'] == 'turn_phrases')}",
        f"llm исходы: {dict(outcomes)} | err: {dict(errors)} | queue_ms p95 {queue_p95}",
        f"разбор: {dict(parse)} | запасной путь правил {share(fallback, sum(parse.values()))}",
    ]
    tools: dict[str, list[float]] = {}
    for r in rows:
        for name, ms in (r.get("tool_ms") or {}).items():
            tools.setdefault(name, []).append(ms)
    if tools:  # ход (е), 22.10: где секунды вне LLM — время шагов плана по инструментам
        per_tool = ", ".join(f"{name} n={len(v)} p95={p95(v)}" for name, v in sorted(tools.items(), key=lambda kv: -len(kv[1])))
        lines.append(f"инструменты, мс: {per_tool}")
    if any("tu_decisions_spent" in r for r in rows):
        denied = sum(1 for r in rows if r.get("tu_decisions_denied"))
        lines.append(
            f"решения хода: p95 потрачено {p95(r.get('tu_decisions_spent', 0) for r in rows)} | ходов с отказом лимита {share(denied, len(rows))}"
        )
    priced = [r for r in rows if any(r.get(field) for field in _PRICE_FIELDS)]
    if priced:
        fields = " | ".join(
            f"{field} {dict(Counter(str(r[field]) for r in priced if r.get(field)))}" for field in _PRICE_FIELDS if any(r.get(field) for r in priced)
        )
        lines.append(f"цена ({len(priced)} ходов, p95 мс {p95(r.get('ms', 0) for r in priced)}): {fields}")
    reasons = []
    if live and failed / len(live) >= 0.02:
        reasons.append("UNAVAILABLE+INVALID ≥ 2%")
    if choose_p95 is not None and choose_p95 >= 4000:
        reasons.append("p95 выбора ≥ 4 с")
    if queue_p95 is not None and queue_p95 >= 1000:
        reasons.append("p95 queue_ms ≥ 1 с")
    if outcomes.get("circuit_open"):
        reasons.append("circuit_open")
    if errors:
        reasons.append("err не пуст")
    if calls and budget / len(calls) > 0.02:
        reasons.append("budget > 2%")
    lines.append("ВЕРДИКТ: годен" if not reasons else "ВЕРДИКТ: не годен — " + ", ".join(reasons))
    return lines, not reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="начало окна, UTC, ГГГГ-ММ-ДДTЧЧ:ММ:СС")
    ap.add_argument("--journals", default=os.path.join(os.environ.get("APP_DATA_DIR", "/app_data"), "journals"))
    args = ap.parse_args(argv)
    rows = load_rows(args.journals, args.since)
    if not rows:
        print(f"ходов с {args.since} нет (папка {args.journals})")
        return 2
    lines, ok = summarize(rows)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
