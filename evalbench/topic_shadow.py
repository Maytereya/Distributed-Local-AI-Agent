"""Матрица «правило × LLM» по теме хода — как часто правило `legacy_v2` ошибается (ход (б), решения владельца 20.10).

    ./venv/bin/python -m evalbench.topic_shadow <прогон> [--gold ~/Dev/dla-private/bench/labels/topic_test.jsonl]

Офлайн: читает прогон стенда, снятый с `topic_shadow_llm = on` (`--cfg MR_TOPIC_SHADOW_LLM=on`). На ходах, где тему решило
правило (`nlu_src = rule`), тень LLM-классификатора записала свою метку (`tu_topic_shadow`). Печатает:

- согласие правила и LLM по ветке правила (`nlu_rule`) и по категории эталона;
- матрицу «метка правила → метка LLM» (только расхождения);
- если есть разметка «правильной темы» (`--gold`): точность правила и LLM против разметки — всего, без спорных, по ветке.

Печатает Markdown только со счётчиками, именами правил и меток — без реплик пациентов.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .card import load_run
from .world import PRIVATE_ROOT

_GOLD = PRIVATE_ROOT / "labels" / "topic_test.jsonl"


def load_gold(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {row["id"]: row for row in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())}


def shadow_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ходы, где тему решило правило и тень дала метку: `{id, category, rule, rule_label, llm_label}`."""

    out = []
    for t in turns:
        tel = t.get("telemetry") or {}
        if tel.get("nlu_src") != "rule" or "tu_topic_shadow" not in tel:
            continue
        out.append({
            "id": t.get("id"),
            "category": t.get("category") or "—",
            "rule": tel.get("nlu_rule") or "?",
            "rule_label": tel.get("tu_topic_shadow_rule"),
            "llm_label": tel.get("tu_topic_shadow"),
            "final_label": tel.get("tu_label_final"),
        })
    return out


def measure(turns: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = shadow_turns(turns)
    decided = [r for r in rows if r["llm_label"] not in {"unavailable", "invalid", "late", "error"}]
    by_rule: dict[str, Counter] = defaultdict(Counter)
    by_category: dict[str, Counter] = defaultdict(Counter)
    matrix = Counter()
    for r in decided:
        agree = r["llm_label"] == r["rule_label"]
        for bucket, key in ((by_rule, r["rule"]), (by_category, r["category"])):
            bucket[key]["n"] += 1
            bucket[key]["agree"] += agree
        if not agree:
            matrix[(r["rule_label"], r["llm_label"])] += 1
    acc: dict[str, Counter] = defaultdict(Counter)
    for r in decided:
        g = gold.get(r["id"])
        if g is None:
            continue
        for scope in ("all", "clear") if not g.get("disputed") else ("all",):
            acc[scope]["n"] += 1
            acc[scope]["rule_ok"] += r["rule_label"] == g["label"]
            acc[scope]["llm_ok"] += r["llm_label"] == g["label"]
            acc[scope]["final_ok"] += r["final_label"] == g["label"]
        acc[f"rule:{r['rule']}"]["n"] += 1
        acc[f"rule:{r['rule']}"]["rule_ok"] += r["rule_label"] == g["label"]
        acc[f"rule:{r['rule']}"]["llm_ok"] += r["llm_label"] == g["label"]
        acc[f"rule:{r['rule']}"]["final_ok"] += r["final_label"] == g["label"]
    failures = Counter(r["llm_label"] for r in rows if r not in decided)
    return {"n": len(rows), "decided": len(decided), "failures": failures, "by_rule": by_rule,
            "by_category": by_category, "matrix": matrix, "acc": acc}


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "—"


def render(name: str, m: dict[str, Any]) -> str:
    agree = sum(c["agree"] for c in m["by_rule"].values())
    lines = [
        f"### {name}: тень LLM на ходах правила — {m['n']} ходов, решила {m['decided']}, сбоев {dict(m['failures']) or 0}",
        "",
        f"Согласие правила и LLM: {agree} из {m['decided']} ({_pct(agree, m['decided'])}).",
        "",
        "| Ветка правила | Ходов | Согласие |",
        "|---|---|---|",
    ]
    for rule, c in sorted(m["by_rule"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| `{rule}` | {c['n']} | {_pct(c['agree'], c['n'])} |")
    lines += ["", "| Категория эталона | Ходов | Согласие |", "|---|---|---|"]
    for cat, c in sorted(m["by_category"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {cat} | {c['n']} | {_pct(c['agree'], c['n'])} |")
    lines += ["", "| Правило → LLM (расхождения) | Ходов |", "|---|---|"]
    for (rule_label, llm_label), n in m["matrix"].most_common():
        lines.append(f"| {rule_label} → {llm_label} | {n} |")
    if m["acc"]:
        lines += ["", "| Против разметки темы | Ходов | Правило верно | LLM верно | Итог хода верно |", "|---|---|---|---|---|"]
        for scope in sorted(m["acc"], key=lambda s: (s.startswith("rule:"), -m["acc"][s]["n"])):
            c = m["acc"][scope]
            title = {"all": "все", "clear": "без спорных"}.get(scope, f"`{scope[5:]}`")
            lines.append(
                f"| {title} | {c['n']} | {_pct(c['rule_ok'], c['n'])} | {_pct(c['llm_ok'], c['n'])} | {_pct(c['final_ok'], c['n'])} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gold", type=Path, default=_GOLD)
    args = ap.parse_args(argv)
    gold = load_gold(args.gold)
    for ref in args.runs:
        header, turns = load_run(ref)
        print(render(str(header.get("name") or ref), measure(turns, gold)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
