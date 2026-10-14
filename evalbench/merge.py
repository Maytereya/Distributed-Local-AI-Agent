"""Склейка частей прогона (`run --shard K/N`) в один прогон для карточки.

    ./venv/bin/python -m evalbench.merge base_2b9278d_real_day

Ищет `<имя>.part*of*` в каталоге прогонов, складывает `turns.jsonl` в исходном порядке
набора, шапку — общую: счётчики сети, очисток, недоступных вызовов и живых вызовов LLM
суммируются; прогон действителен, только если действительна каждая часть.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .world import PRIVATE_ROOT


def merge(name: str) -> Path:
    runs = PRIVATE_ROOT / "runs"
    parts = sorted(runs.glob(f"{name}.part*of*"))
    if not parts:
        raise SystemExit(f"частей прогона {name} нет в {runs}")
    headers = [json.loads((p / "run.json").read_text(encoding="utf-8")) for p in parts]
    total = {int(p.name.rsplit("of", 1)[1]) for p in parts}
    if len(total) != 1 or len(parts) != total.pop():
        raise SystemExit(f"части неполные: {[p.name for p in parts]}")
    sets = [Path(x).stem for x in headers[0].get("sets") or []]
    rows = [json.loads(line) for p in parts for line in (p / "turns.jsonl").open(encoding="utf-8") if line.strip()]
    ordered = sorted(rows, key=lambda r: (sets.index(r["set"]) if r.get("set") in sets else len(sets), r["id"]))
    out = runs / name
    out.mkdir(exist_ok=True)
    with (out / "turns.jsonl").open("w", encoding="utf-8") as fh:
        for row in ordered:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Журнал синонимов — строки всех частей (у частей до 14.10 файла нет — тогда и склейки нет).
    journals = [p / "synonym_journal.jsonl" for p in parts if (p / "synonym_journal.jsonl").exists()]
    if journals:
        (out / "synonym_journal.jsonl").write_bytes(b"".join(j.read_bytes() for j in journals))

    def summed(key: str) -> list:
        c: Counter = Counter()
        for h in headers:
            c.update(dict((k, v) for k, v in h.get(key) or []))
        return c.most_common()

    head = dict(headers[0])
    head.update(
        name=name,
        parts=[p.name for p in parts],
        cases=sum(h["cases"] for h in headers),
        done=sum(h["done"] for h in headers),
        stopped="; ".join(h["stopped"] for h in headers if h.get("stopped")) or None,
        live_calls=sum(h.get("live_calls", 0) for h in headers),
        net_attempts=summed("net_attempts"),
        unavailable_calls=summed("unavailable_calls"),
        suppressed_writes=summed("suppressed_writes"),
        cleanup_calls=[c for h in headers for c in h.get("cleanup_calls") or []],
        world_problems=[x for h in headers for x in h.get("world_problems") or []],
        llm_outcomes=dict(sum((Counter(h.get("llm_outcomes") or {}) for h in headers), Counter())),
        synonym_journal_lines=sum(h.get("synonym_journal_lines", 0) for h in headers),
        seconds=max(h.get("seconds", 0) for h in headers),
        # Части с разными настройками (`--cfg`) — разные прогоны, склеивать нельзя.
        cfg_mismatch=sorted({json.dumps(h.get("cfg") or {}, sort_keys=True) for h in headers})[1:] or None,
        valid=all(h.get("valid") for h in headers) and len({json.dumps(h.get("cfg") or {}, sort_keys=True) for h in headers}) == 1,
    )
    (out / "run.json").write_text(json.dumps(head, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    args = ap.parse_args(argv)
    out = merge(args.name)
    print(f"склеено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
