"""Тень «слово → роль» по каталогу МИС — точный слой против LLM (ход (г), этап 2, решения владельца 20.10 (г) 1, 2, 6).

    ./venv/bin/python -m evalbench.role_shadow <прогон> [<прогон> ...] [--set свип.jsonl]

Офлайн: читает прогон стенда, снятый с `roles_catalog_llm = shadow`. Печатает по ходам с тенью: слой точного совпадения
(`role_layer`), исходы LLM, согласие LLM с точным слоем и с ролью правил. С `--set` (свип ролей: у вопроса поле `expect` —
роль МИС, `нет`, `qualifier`, `not_role`; несколько допустимых — через «|») — точность точного слоя и LLM против ожидания.
Названия ролей — данные МИС; реплик пациентов отчёт не печатает.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .card import load_run


def _ok(value: str | None, expect: str) -> bool:
    if not value:
        return False
    options = [e.strip() for e in expect.split("|") if e.strip()]
    if any(e.startswith("нет") for e in options):
        return value == "none"
    # Роль с несколькими специальностями («хирург, флеболог») — верна и по первой из них.
    return any(value == e or value.split(",")[0].strip() == e.split(",")[0].strip() for e in options)


def measure(turns: list[dict[str, Any]], expects: dict[str, str]) -> dict[str, Any]:
    rows = [t for t in turns if "role_layer" in (t.get("telemetry") or {})]
    layers = Counter(t["telemetry"]["role_layer"] for t in rows)
    outcomes = Counter(str(t["telemetry"].get("role_llm")) for t in rows)
    agree = Counter()
    for t in rows:
        tel = t["telemetry"]
        for key in ("role_llm_agree", "role_rules_agree"):
            if key in tel:
                agree[f"{key}:n"] += 1
                agree[f"{key}:yes"] += bool(tel[key])
        agree["role_rules_unmapped"] += bool(tel.get("role_rules_unmapped"))
    acc = Counter()
    for t in turns:
        expect = expects.get(str(t.get("id")))
        if expect is None:
            continue
        tel = t.get("telemetry") or {}
        acc["n"] += 1
        acc["no_shadow"] += "role_layer" not in tel
        acc["exact_ok"] += _ok(tel.get("role_exact"), expect)
        acc["llm_ok"] += _ok(tel.get("role_llm"), expect)
    return {"n": len(rows), "layers": layers, "outcomes": outcomes, "agree": agree, "acc": acc}


def render(name: str, m: dict[str, Any]) -> str:
    a = m["agree"]
    lines = [
        f"### {name}: ходов с тенью ролей {m['n']}",
        "",
        "Слой точного совпадения: " + ", ".join(f"{k} {v}" for k, v in m["layers"].most_common()),
        "",
        "Исходы LLM: " + ", ".join(f"`{k}` {v}" for k, v in m["outcomes"].most_common(12)),
        "",
        f"LLM согласна с точным слоем: {a['role_llm_agree:yes']} из {a['role_llm_agree:n']}; "
        f"с ролью правил: {a['role_rules_agree:yes']} из {a['role_rules_agree:n']} "
        f"(имя правил не легло в роль каталога: {a['role_rules_unmapped']}).",
    ]
    acc = m["acc"]
    if acc["n"]:
        lines += [
            "",
            f"Против ожидания свипа: фраз {acc['n']}, тень не включилась {acc['no_shadow']}; "
            f"точный слой верно {acc['exact_ok']}, LLM верно {acc['llm_ok']}.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--set", type=Path, default=None, help="свип ролей с полем expect")
    args = ap.parse_args(argv)
    expects: dict[str, str] = {}
    if args.set is not None:
        for line in args.set.expanduser().open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("expect"):
                    expects[str(row["id"])] = str(row["expect"])
    for ref in args.runs:
        header, turns = load_run(ref)
        print(render(str(header.get("name") or ref), measure(turns, expects)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
