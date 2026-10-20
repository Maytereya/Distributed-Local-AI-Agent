"""Кто решил тему хода — замер-основа хода (б) «одно понимание хода» (план 20.10).

    ./venv/bin/python -m evalbench.topic_source <прогон> [<прогон> ...] [--labels price_allowed.jsonl]

Офлайн, без LLM: читает сохранённые прогоны стенда (`turns.jsonl`) и раскладывает каждый ход по слою, который
решил метку (тему) реплики:

- `pending` — до NLU: предпроверка роутера, ответ на висящий вопрос, проверка существующей записи, кнопка;
- `rule` — правило классификатора `legacy_v2` (LLM-классификатор не звали: `nlu_llm` = false);
- `llm` — LLM-классификатор (`nlu_llm` = true, правило не сработало);
- `unrecorded` — ход завершился раньше, чем решение NLU попало в телеметрию (`router._note_decision`);
- поверх NLU — `override`: метку после NLU переписал реестр тем, guardrail или переклейка роутера (флаги ниже).

`decision_source` хода сам по себе этого не различает: решение правила `legacy_v2` проходит слияние
(`nlu_pipeline._merge`) с подписью `llm_primary`, как и решение LLM.

Имя правила — по флагам решения: правило ставит флаг `rule_*` или один из флагов безопасности/служебных без
префикса (`_UNPREFIXED`). Телеметрия хода `rule` — первый по алфавиту `rule_*`, при нескольких флагах неоднозначна.

Печатает Markdown только со счётчиками и именами правил — без реплик пациентов: можно вставлять в репо и журнал.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .card import load_labels, load_run
from .prices import PricePair, price_verdict
from .world import PRIVATE_ROOT

# Флаги правил `classifier.deterministic_rule_decision` без префикса `rule_` (ветки безопасности и служебные).
_UNPREFIXED = (
    "urgent", "complaint", "medical_advice", "smalltalk_greeting", "doc_request_main_index",
    "unsupported_catalog",
)
# Источники метки до NLU (ход не дошёл до классификатора или его решение заменено заранее).
_PENDING_SOURCES = {"router_precheck", "existing_appointment", "catalog_confirm", "guardrail", None}
# Переписали метку ПОСЛЕ NLU.
_OVERRIDE_SOURCES = {"topic_registry", "guardrail_post"}
_OVERRIDE_FLAGS = (
    "test_assist_catalog_hit_is_price", "understanding_list_is_price", "promoted_from_rule_hints",
    "rule_price_rescue",
)
_BAD_OUTCOMES = ("handoff", "operator_offer", "clarify", "not_understood", "no_info")


def rule_name(flags: list[str]) -> str | None:
    found = sorted(f for f in flags if f.startswith("rule_") and f not in {"rule_none", "rule_price_rescue"})
    if found:
        return "+".join(found)
    return next((f for f in _UNPREFIXED if f in flags), None)


def layer(turn: dict[str, Any]) -> str:
    tel = turn.get("telemetry") or {}
    if turn.get("label") is None and not turn.get("flags"):
        return "unrecorded"  # ход завершился до записи решения NLU (`router._note_decision`)
    source = tel.get("decision_source")
    if source in _PENDING_SOURCES:
        return "pending"
    if tel.get("nlu_llm"):
        return "llm"
    return "rule"


def overrides(turn: dict[str, Any]) -> list[str]:
    tel = turn.get("telemetry") or {}
    flags = turn.get("flags") or []
    out = [f for f in _OVERRIDE_FLAGS if f in flags]
    if tel.get("decision_source") in _OVERRIDE_SOURCES:
        out.append(str(tel["decision_source"]))
    return out


def _golden(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ref in header.get("sets") or []:
        path = Path(ref)
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    return out


def _classes(row: dict[str, Any] | None) -> list[str]:
    raw = (row or {}).get("classes")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = []
    return [str(x) for x in raw or []]


def measure(header: dict[str, Any], turns: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    golden = _golden(header)
    layers = Counter()
    by_rule: dict[str, Counter] = defaultdict(Counter)
    by_category: dict[str, Counter] = defaultdict(Counter)
    by_class: dict[str, Counter] = defaultdict(Counter)
    override = Counter()
    multi_rule = 0
    for t in turns:
        lay = layer(t)
        layers[lay] += 1
        flags = t.get("flags") or []
        for o in overrides(t):
            override[o] += 1
        by_category[t.get("category") or "—"][lay] += 1
        for cls in _classes(golden.get(t["id"])) or ["—"]:
            by_class[cls][lay] += 1
        if lay != "rule":
            continue
        name = rule_name(flags) or "(без флага)"
        multi_rule += "+" in name
        row = by_rule[name]
        row["n"] += 1
        row[f"cat:{t.get('category') or '—'}"] += 1
        row["bad"] += t.get("outcome") in _BAD_OUTCOMES
        row["overridden"] += bool(overrides(t))
        if t["id"] in labels:
            verdict = price_verdict([PricePair(**p) for p in t.get("price_pairs") or []], labels[t["id"]])
            if verdict is not None:
                row["price_labeled"] += 1
                row["foreign"] += verdict in {"foreign", "list_miss"}
    return {
        "name": header.get("name"), "sha": header.get("code_sha"), "n": len(turns), "layers": layers,
        "by_rule": by_rule, "by_category": by_category, "by_class": by_class, "override": override,
        "multi_rule": multi_rule,
    }


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "—"


def render(m: dict[str, Any]) -> str:
    n = m["n"]
    lay = m["layers"]
    lines = [
        f"### {m['name']} ({m['sha']}), ходов {n}",
        "",
        "| Слой, решивший тему | Ходов | Доля |",
        "|---|---|---|",
    ]
    for key in ("rule", "llm", "pending", "unrecorded"):
        lines.append(f"| {key} | {lay[key]} | {_pct(lay[key], n)} |")
    nlu = lay["rule"] + lay["llm"]
    lines += [
        "",
        f"Среди ходов, дошедших до NLU ({nlu}): правило — {_pct(lay['rule'], nlu)}, LLM — {_pct(lay['llm'], nlu)}. "
        f"Ходов с несколькими флагами правил: {m['multi_rule']}.",
        "",
        "Переписали метку после NLU: " + (", ".join(f"`{k}` {v}" for k, v in m["override"].most_common()) or "—"),
        "",
        "| Правило | Ходов | Главные категории | Перевод/уточнение | Переписано после | Чужая цена / размечено |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in sorted(m["by_rule"].items(), key=lambda kv: -kv[1]["n"]):
        cats = sorted(((k[4:], v) for k, v in row.items() if k.startswith("cat:")), key=lambda kv: -kv[1])[:3]
        cat_text = ", ".join(f"{k} {v}" for k, v in cats)
        price = f"{row['foreign']} / {row['price_labeled']}" if row["price_labeled"] else "—"
        lines.append(
            f"| `{name}` | {row['n']} | {cat_text} | {_pct(row['bad'], row['n'])} | {row['overridden']} | {price} |"
        )
    lines += ["", "| Категория реплики | Ходов | Правило | LLM | До NLU | Не записано |", "|---|---|---|---|---|---|"]
    for cat, c in sorted(m["by_category"].items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(c.values())
        lines.append(f"| {cat} | {total} | {_pct(c['rule'], total)} | {_pct(c['llm'], total)} | {_pct(c['pending'], total)} | {_pct(c['unrecorded'], total)} |")
    lines += ["", "| Класс провала (разбор 08.10) | Ходов | Правило | LLM | До NLU | Не записано |", "|---|---|---|---|---|---|"]
    for cls, c in sorted(m["by_class"].items(), key=lambda kv: kv[0]):
        total = sum(c.values())
        lines.append(f"| {cls} | {total} | {_pct(c['rule'], total)} | {_pct(c['llm'], total)} | {_pct(c['pending'], total)} | {_pct(c['unrecorded'], total)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", type=Path, default=PRIVATE_ROOT / "labels" / "price_allowed.jsonl")
    args = ap.parse_args(argv)
    labels = load_labels(args.labels)
    for ref in args.runs:
        header, turns = load_run(ref)
        print(render(measure(header, turns, labels)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
