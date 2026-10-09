"""Карточка замера «было / стало» по метрикам решения 08.10 (мастер-хэндофф §7.1).

    ./venv/bin/python -m evalbench.card <прогон_было> [<прогон_стало>] [--labels prices.jsonl]

Главные на эталоне: чужая цена (по разметке допустимых serviceId), срочное — полнота 100%
на синтетике, цена без источника. Ограничители: переводы и уточнения — раздельно
(день и ночь — разными прогонами), «нет данных» без перевода (класс 4), задержка p50/p95,
сбои LLM. Правило решения ≤3: одна предотвращённая чужая цена стоит не больше трёх
лишних уточнений или переводов.

Карточка печатается в Markdown; в ней только счётчики — без реплик пациентов, её можно
вставлять в журнал и хэндофф.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .prices import PricePair, price_verdict
from .world import PRIVATE_ROOT

OUTCOME_ORDER = (
    "price", "memo", "answer", "clarify", "not_understood", "operator_offer",
    "handoff", "handoff_tech", "urgent", "no_info", "tech_error",
)


def load_run(ref: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(ref)
    if not path.exists():
        path = PRIVATE_ROOT / "runs" / ref
    header = json.loads((path / "run.json").read_text(encoding="utf-8"))
    turns = [json.loads(line) for line in (path / "turns.jsonl").open(encoding="utf-8") if line.strip()]
    return header, turns


def load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    """Разметка ценовых вопросов: строка `{"id", "kind", "core", "ok", "ambiguous", ...}`."""

    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def metrics(turns: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(turns)
    outcomes = Counter(t["outcome"] for t in turns)
    verdicts = Counter()
    for t in turns:
        if t["id"] in labels:
            pairs = [PricePair(**p) for p in t.get("price_pairs") or []]
            verdict = price_verdict(pairs, labels[t["id"]])
            if verdict is not None:
                verdicts[verdict] += 1
    urgent_cases = [t for t in turns if t.get("expect_urgent") is True]
    twins = [t for t in turns if t.get("expect_urgent") is False]
    llm = Counter(c["outcome"] for t in turns for c in t.get("llm_calls") or [])
    sites = Counter(c["site"] for t in turns for c in t.get("llm_calls") or [])
    ms = [t["ms"] for t in turns]
    return {
        "n": n,
        "outcomes": dict(outcomes),
        "handoff": sum(1 for t in turns if t["handoff"] and not t["urgent"]),
        "handoff_tech": outcomes.get("handoff_tech", 0),
        "operator_offer": outcomes.get("operator_offer", 0),
        "clarify": outcomes.get("clarify", 0) + outcomes.get("not_understood", 0),
        "no_info": outcomes.get("no_info", 0),
        "unsourced_turns": sum(1 for t in turns if t.get("unsourced_prices")),
        "leak": sum(1 for t in turns if t.get("leak")),
        "long_1500": sum(1 for t in turns if t.get("chars", 0) > 1500),
        "price_labeled": sum(verdicts.values()),
        "price_verdicts": dict(verdicts),
        "foreign": verdicts.get("foreign", 0) + verdicts.get("list_miss", 0),
        **{f"price_{k}": verdicts.get(k, 0) for k in ("correct", "list_hit", "single_for_ambiguous", "no_price")},
        "urgent_recall": (sum(1 for t in urgent_cases if t["urgent"]), len(urgent_cases)),
        "urgent_false": (sum(1 for t in twins if t["urgent"]), len(twins)),
        "p50_ms": _pct(ms, 0.5),
        "p95_ms": _pct(ms, 0.95),
        "llm_calls": sum(llm.values()),
        "llm_outcomes": dict(llm),
        "llm_sites": dict(sites),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, tuple):
        hit, total = value
        return f"{hit}/{total}" if total else "—"
    return "—" if value is None else str(value)


def render(runs: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    lines = ["# Карточка замера", ""]
    for header, _m in runs:
        world = header.get("world", {})
        lines.append(
            f"- **{header['name']}**: код `{header['code_sha']}`, LLM `{header['mode']}` ({header.get('model')}), "
            f"мир {world.get('date')} {world.get('profile')}, вопросов {header['done']}/{header['cases']}, "
            f"синонимов {header.get('synonyms')}, valid={header.get('valid')}"
            + (f", **остановлен: {header['stopped']}**" if header.get("stopped") else "")
        )
        if not header.get("valid"):
            lines.append(
                f"  - недействителен: сеть {header.get('net_attempts')}, очистки {header.get('cleanup_calls')}, "
                f"мир {header.get('world_problems')}, неопознанные промпты {header.get('llm_outcomes', {}).get('unknown_site', 0)}, "
                f"неоднозначные {header.get('ambiguous_sites')}"
            )
        lines.append(f"  - недоступно в мире: {', '.join(world.get('unavailable', []))}")
    lines.append("")
    names = [h["name"] for h, _ in runs]
    lines.append("| Метрика | " + " | ".join(names) + (" | Δ |" if len(runs) == 2 else " |"))
    lines.append("|---|" + "---|" * len(runs) + ("---|" if len(runs) == 2 else ""))
    rows = [
        ("Чужая цена (разметка)", "foreign"),
        ("Размечено ценовых", "price_labeled"),
        ("  верная одна цена", "price_correct"),
        ("  список с нужной", "price_list_hit"),
        ("  одна цена вместо списка", "price_single_for_ambiguous"),
        ("  без цены", "price_no_price"),
        ("Срочное: полнота", "urgent_recall"),
        ("Срочное: ложное на двойниках", "urgent_false"),
        ("Цена без источника (ходов)", "unsourced_turns"),
        ("Перевод на оператора", "handoff"),
        ("  из них технический", "handoff_tech"),
        ("Оффер оператора", "operator_offer"),
        ("Уточнение / не понял", "clarify"),
        ("Нет данных без перевода (класс 4)", "no_info"),
        ("Утечка внутренностей", "leak"),
        ("Длиннее 1500 знаков", "long_1500"),
        ("p50, мс", "p50_ms"),
        ("p95, мс", "p95_ms"),
        ("Вызовов LLM", "llm_calls"),
    ]
    for title, key in rows:
        vals = [m[key] for _, m in runs]
        delta = ""
        if len(runs) == 2 and all(isinstance(v, int) for v in vals):
            delta = f" {vals[1] - vals[0]:+d} |"
        elif len(runs) == 2:
            delta = " |"
        lines.append(f"| {title} | " + " | ".join(_fmt(v) for v in vals) + (" |" + delta if delta else " |"))
    lines.append("")
    lines.append("**Исходы ходов:**")
    lines.append("")
    lines.append("| Исход | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for key in OUTCOME_ORDER:
        lines.append(f"| {key} | " + " | ".join(str(m["outcomes"].get(key, 0)) for _, m in runs) + " |")
    if len(runs) == 2:
        _, before = runs[0]
        _, after = runs[1]
        prevented = before["foreign"] - after["foreign"]
        extra = (after["handoff"] + after["operator_offer"] + after["clarify"]) - (
            before["handoff"] + before["operator_offer"] + before["clarify"]
        )
        lines.append("")
        lines.append(
            f"**Правило ≤3:** предотвращено чужих цен {prevented}, лишних уточнений и переводов {extra} → "
            + ("принять" if extra <= 3 * max(prevented, 0) else "НЕ проходит")
            + (" (чужие цены не размечены — правило не применимо)" if not before["price_labeled"] else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="имя прогона или путь к каталогу (1 или 2)")
    ap.add_argument("--labels", type=Path, default=PRIVATE_ROOT / "labels" / "price_allowed.jsonl")
    args = ap.parse_args(argv)
    labels = load_labels(args.labels)
    loaded = []
    for ref in args.runs[:2]:
        header, turns = load_run(ref)
        loaded.append((header, metrics(turns, labels)))
    print(render(loaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
