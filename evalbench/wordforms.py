"""Детектор «один смысл — разные словоформы»: согласны ли ответы внутри группы набора `sets/wordforms.jsonl`.

    ./venv/bin/python -m evalbench.wordforms wf_real            # одна группа строк на прогон
    ./venv/bin/python -m evalbench.wordforms wf_real wf_down    # два прогона рядом (например, LLM как на проде и без LLM)

Зачем (решение владельца 22.10, `docs/OWNER_DECISIONS.md`): «анализ» и «анализы», «услуга» и «услугу» разбирались разными
списками стоп-слов, и ответ зависел от падежа — переспрос «аминокислоты?» (BUG-2026-10-22-CATALOG-CONFIRM-BY-LETTERS), памятка
Нечипоренко на «подготовка к анализам крови». Вопросы группы — один смысл в разных словоформах; бот, который понимает смысл,
отвечает им одинаково. Детектор не знает правильного ответа — он ловит расхождение, а правильность — карточка и разметка.

Подпись ответа — то, что пациент получил по сути, а не формулировка: исход хода; строки цены (`service_id` найденных строк);
название памятки клиники; иначе начало текста ответа без цифр и пунктуации.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .world import PRIVATE_ROOT

_MEMO_RE = re.compile(r"Памятка клиники «([^»]+)»")
_WORD_RE = re.compile(r"[а-яёa-z]+", re.I)


def _set_groups(header: dict[str, Any]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for raw in header.get("sets") or []:
        path = Path(raw)
        if not path.is_file():
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                if row.get("group"):
                    groups[str(row["id"])] = str(row["group"])
    return groups


def signature(turn: dict[str, Any]) -> str:
    outcome = str(turn.get("outcome") or "")
    rows = sorted({str(p.get("service_id")) for p in turn.get("price_pairs") or [] if p.get("service_id")})
    if rows:
        return f"{outcome}: строки {', '.join(rows)}"
    answer = str(turn.get("answer") or "")
    memo = _MEMO_RE.search(answer)
    if memo:
        return f"{outcome}: памятка «{memo.group(1)}»"
    return f"{outcome}: {' '.join(_WORD_RE.findall(answer.lower())[:8])}"


def load(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = PRIVATE_ROOT / "runs" / name
    header = json.loads((run / "run.json").read_text(encoding="utf-8"))
    turns = [json.loads(line) for line in (run / "turns.jsonl").open(encoding="utf-8") if line.strip()]
    return header, turns


def agreement(header: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    """Группа → [(вопрос, подпись)]; группы берутся из набора прогона (поле `group`)."""

    groups = _set_groups(header)
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for turn in turns:
        group = groups.get(str(turn.get("id")))
        if group:
            out[group].append((str(turn.get("question") or ""), signature(turn)))
    return dict(out)


def render(names: list[str]) -> str:
    lines = ["# Словоформы: согласие ответов внутри группы", ""]
    results = []
    for name in names:
        header, turns = load(name)
        results.append((name, header, agreement(header, turns)))
    lines.append("| Прогон | Режим | Групп | Согласны | Расходятся |")
    lines.append("|---|---|---|---|---|")
    for name, header, groups in results:
        split = [g for g, rows in groups.items() if len({sig for _, sig in rows}) > 1]
        lines.append(f"| {name} | {header.get('mode')} | {len(groups)} | {len(groups) - len(split)} | {len(split)} |")
    for name, _, groups in results:
        split = {g: rows for g, rows in groups.items() if len({sig for _, sig in rows}) > 1}
        if not split:
            continue
        lines += ["", f"## {name}: расходятся"]
        for group, rows in split.items():
            lines.append(f"- **{group}**")
            lines += [f"  - {question} → {sig}" for question, sig in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="имена прогонов в ~/Dev/dla-private/bench/runs")
    args = ap.parse_args(argv)
    print(render(args.runs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
