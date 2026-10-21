"""Расхождения читателей «список ли» на ценовых ходах — перечень для карточки владельца (ход (б), консилиум 20.10 §1).

    ./venv/bin/python -m evalbench.list_readers <прогон> [<прогон> ...] [--world 20261008]

Офлайн, без LLM: на репликах ходов прогона с меткой PRICE / TEST_ASSIST считает в замороженном мире два вердикта:

- понимание хода — `price_list_services` (корзина + защиты K1: «одно название прайса покрывает все слова» и «сплошной
  список с нераспознанным остатком»);
- читатели корзины R1/R7 — `_multi_price_items` (без K1; K3 — гард уточнения — у обоих).

Расхождение — один видит список, другой нет. Печатает Markdown: счётчики по правилу понимания и поимённо id ходов
(без реплик пациентов). Это ≈4% разобранных ходов по консилиуму: где сходятся — этап 2 даёт 0 различий off/on, где
расходятся — решает карточка владельца.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from .card import load_run
from .world import PRIVATE_ROOT, FrozenWorld

_PRICE_LABELS = {"PRICE", "TEST_ASSIST"}


def verdicts(question: str, rows: list[dict[str, Any]]) -> tuple[str, bool, bool]:
    """(правило понимания, понимание — список, корзина — список)."""

    from messengers_router.services._prices_helpers import _multi_price_items, price_list_services

    names, rule = price_list_services(question, rows=rows)
    items, _unrecognized = _multi_price_items(question, rows)
    return rule, bool(names), bool(items)


def measure(turns: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rule: dict[str, Counter] = defaultdict(Counter)
    diverge: dict[str, list[str]] = defaultdict(list)
    n = 0
    for t in turns:
        if (t.get("label") or (t.get("telemetry") or {}).get("tu_label_final")) not in _PRICE_LABELS:
            continue
        question = str(t.get("question") or "")
        if not question.strip():
            continue
        n += 1
        rule, tu_list, basket_list = verdicts(question, rows)
        by_rule[rule]["n"] += 1
        if tu_list != basket_list:
            kind = "only_basket" if basket_list else "only_understanding"
            by_rule[rule][kind] += 1
            diverge[f"{rule}:{kind}"].append(str(t.get("id")))
    return {"n": n, "by_rule": by_rule, "diverge": diverge}


def render(name: str, m: dict[str, Any]) -> str:
    total = sum(len(v) for v in m["diverge"].values())
    lines = [
        f"### {name}: ценовых ходов {m['n']}, расхождений «список ли» {total} ({round(100 * total / m['n']) if m['n'] else 0}%)",
        "",
        "| Правило понимания | Ходов | Список только у корзины (R1/R7) | Список только у понимания |",
        "|---|---|---|---|",
    ]
    for rule, c in sorted(m["by_rule"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| `{rule}` | {c['n']} | {c['only_basket']} | {c['only_understanding']} |")
    if m["diverge"]:
        lines += ["", "Поимённо (id ходов):"]
        for key, ids in sorted(m["diverge"].items()):
            lines.append(f"- `{key}`: {', '.join(ids)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--world", default="20261008")
    args = ap.parse_args(argv)
    with FrozenWorld(PRIVATE_ROOT / "worlds" / args.world, "day"):
        from messengers_router.services._prices_helpers import _catalog_rows

        rows = _catalog_rows(None)
        for ref in args.runs:
            header, turns = load_run(ref)
            print(render(str(header.get("name") or ref), measure(turns, rows)))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
