"""Прогон набора вопросов через эндпоинт бота в замороженном мире.

    ./venv/bin/python -m evalbench.run --set ~/Dev/dla-private/golden_v1.jsonl \\
        --code-root /path/to/worktree-2b9278d --world 20261008 --profile day --mode down

Каждый вопрос — один ход в новой сессии (решение 09.10: одиночный ход честнее для
сравнения версий — второй ход зависит от первого ответа). Ход идёт через
`messenger_generate_once` с `debug=true`, в одном asyncio-цикле на весь прогон.

Выход — каталог прогона вне репо (`~/Dev/dla-private/bench/runs/<имя>/`):
- `turns.jsonl` — строка на вопрос: вопрос, ответ, исход, признаки, вызовы LLM, задержка;
- `run.json` — шапка: код, мир, модель, режим, отпечатки промптов, словарь синонимов,
  попытки сети, очистки, промахи; `valid=false`, если прогон нельзя принимать;
- `bot.log` — вывод бота (в нём бывают адреса МИС — поэтому тоже вне репо).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import card as _card  # noqa: F401 — модули стенда грузим до смены пути к дереву кода
from .detectors import Templates, classify_turn
from .llm import MODES, HarnessClient, LiveBudgetExhausted, ResponseCache, SiteIndex
from .prices import PriceCatalog
from .world import PRIVATE_ROOT, PROFILES, FrozenWorld, build_world, host_of, verify_world

REPO_ROOT = Path(__file__).resolve().parent.parent
TURN_TIMEOUT_S = 240


def load_cases(paths: list[Path], *, ids: set[str] | None, split: str | None, limit: int | None) -> list[dict[str, Any]]:
    cases = []
    for path in paths:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("set", path.stem)
            if ids and row["id"] not in ids:
                continue
            if split and row.get("split") != split:
                continue
            cases.append(row)
    return cases[:limit] if limit else cases


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _enter_code_root(code_root: Path) -> None:
    """Бот импортируется из `code_root`; относительные пути бота (app_data) — от него же."""

    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(1, str(REPO_ROOT))


async def _run_cases(cases, *, client: HarnessClient, world: FrozenWorld, tpl: Templates, catalog: PriceCatalog, llm_mode: str, run_name: str, out, progress) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    memory = MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)
    services = Services()
    stats: dict[str, Any] = {"done": 0, "stopped": None}
    for i, case in enumerate(cases, 1):
        sid = f"bench-{run_name}-{case['id']}"
        req = MessengerGenerateRequest(session_id=sid, text=str(case["question"]), debug=True, llm_mode=llm_mode, button_id=case.get("button") or "")
        started = time.monotonic()
        error = None
        try:
            resp = await asyncio.wait_for(messenger_generate_once(req, memory=memory, services=services), TURN_TIMEOUT_S)
            body = json.loads(resp.body)
        except LiveBudgetExhausted as exc:
            stats["stopped"] = str(exc)
            break
        except Exception as exc:  # ход не завершился — фиксируем, идём дальше
            body = {"text": "", "handoff": False, "state_update": {"debug": {"endpoint_error": f"{type(exc).__name__}: {exc}"}}}
            error = type(exc).__name__
        ms = int((time.monotonic() - started) * 1000)
        state = memory.get(sid)
        calls = client.take_turn_calls()
        if any(c.outcome == "budget" for c in calls):
            stats["stopped"] = "исчерпан бюджет живых вызовов LLM"
            break
        outcome = classify_turn(body, dict(state.last_entities or {}), tpl, catalog)
        row = {
            "id": case["id"],
            "set": case.get("set"),
            "split": case.get("split"),
            "category": case.get("category") or case.get("topic_llm"),
            "tier": case.get("tier"),
            "expect_urgent": case.get("expect_urgent"),
            "question": case["question"],
            "answer": body.get("text", ""),
            "ms": ms,
            "harness_error": error,
            **outcome.as_row(),
            "stage_ms": ((body.get("state_update") or {}).get("debug") or {}).get("stage_timings"),
            "llm_calls": [c.__dict__ for c in calls],
            "telemetry": world.take_telemetry(),
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        stats["done"] = i
        if i % 10 == 0 or i == len(cases):
            progress(f"{i}/{len(cases)} последний: {case['id']} → {outcome.outcome} ({ms} мс)")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", type=Path, required=True, help="JSONL набора (можно несколько)")
    ap.add_argument("--code-root", type=Path, default=REPO_ROOT, help="дерево кода бота (по умолчанию — этот репозиторий)")
    ap.add_argument("--world", default="20261008", help="дата мира ГГГГММДД (каталог ~/Dev/dla-private/bench/worlds/<дата>)")
    ap.add_argument("--build-world-from", type=Path, help="собрать мир из этого каталога apidata, если его ещё нет")
    ap.add_argument("--profile", choices=PROFILES, default="day")
    ap.add_argument("--mode", choices=MODES, default="down")
    ap.add_argument("--llm-mode", default="hybrid", help="llm_mode запроса (шлюз прода шлёт hybrid)")
    ap.add_argument("--cache", type=Path, default=PRIVATE_ROOT / "llm_cache.jsonl", help="запись ответов настоящей LLM")
    ap.add_argument("--live-pause", type=float, default=1.0, help="пауза между вызовами настоящей LLM, с")
    ap.add_argument("--max-live-calls", type=int, help="потолок вызовов настоящей LLM на прогон")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="только эти id через запятую")
    ap.add_argument("--split", choices=("dev", "test"))
    ap.add_argument("--name", help="имя прогона (по умолчанию <sha>_<режим>_<профиль>)")
    ap.add_argument("--shard", help="часть K/N набора (K с 1): N процессов параллельно; склейка — evalbench.merge")
    args = ap.parse_args(argv)

    code_root = args.code_root.resolve()
    world_root = PRIVATE_ROOT / "worlds" / args.world
    if not (world_root / "world.json").exists():
        if not args.build_world_from:
            ap.error(f"мира {world_root} нет: укажите --build-world-from <apidata>")
        build_world(args.build_world_from.resolve(), args.world, world_root)

    cases = load_cases(
        [p.expanduser() for p in args.sets],
        ids=set(args.ids.split(",")) if args.ids else None,
        split=args.split,
        limit=args.limit,
    )
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= k <= n:
            ap.error("--shard K/N: 1 ≤ K ≤ N")
        cases = cases[k - 1 :: n]
    sha = git_sha(code_root)
    name = args.name or f"{sha}_{args.mode}_{args.profile}"
    if args.shard:
        name = f"{name}.part{args.shard.replace('/', 'of')}"
    run_dir = PRIVATE_ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(PRIVATE_ROOT, 0o700)

    _enter_code_root(code_root)
    from messengers_router import llm_runtime
    from messengers_router.runtime_config import config as bot_config

    live = args.mode in ("real", "cached")
    allow = (host_of(bot_config.ollama_url),) if live else ()
    console = sys.__stderr__
    header: dict[str, Any] = {}
    with open(run_dir / "bot.log", "w", encoding="utf-8") as log_fh, open(run_dir / "turns.jsonl", "w", encoding="utf-8") as out:
        with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
            import logging

            logging.basicConfig(level=logging.INFO, stream=log_fh, force=True)
            with FrozenWorld(world_root, args.profile, allow_hosts=allow) as world:
                from agent_logic_2.ollama_settings import LLMName
                from messengers_router.services import _biomaterial

                sites = SiteIndex.from_code_root(code_root)
                client = HarnessClient(
                    mode=args.mode,
                    sites=sites,
                    real=llm_runtime._OLLAMA_CLIENT if live else None,
                    cache=ResponseCache(args.cache) if args.mode in ("real", "cached", "replay") else None,
                    live_pause_s=args.live_pause,
                    max_live_calls=args.max_live_calls,
                )
                real_client = llm_runtime._OLLAMA_CLIENT
                llm_runtime._OLLAMA_CLIENT = client
                try:
                    tpl = Templates.from_bot()
                    catalog = PriceCatalog.from_world(world_root / "apidata")
                    vocab = _biomaterial._vocabularies()
                    synonyms = len(getattr(vocab, "synonym_candidates", ()) or ())
                    print(f"[evalbench] {name}: {len(cases)} вопросов, словарь синонимов {synonyms}", file=console)

                    def progress(msg: str) -> None:
                        print(f"[evalbench] {msg}", file=console, flush=True)

                    started = time.time()
                    stats = asyncio.run(
                        _run_cases(cases, client=client, world=world, tpl=tpl, catalog=catalog, llm_mode=args.llm_mode, run_name=name, out=out, progress=progress)
                    )
                finally:
                    llm_runtime._OLLAMA_CLIENT = real_client
                facts = world.facts()
                header = {
                    "name": name,
                    "code_sha": sha,
                    "code_root": str(code_root),
                    "sets": [str(p) for p in args.sets],
                    "cases": len(cases),
                    "done": stats["done"],
                    "stopped": stats["stopped"],
                    "mode": args.mode,
                    "llm_mode": args.llm_mode,
                    "model": LLMName.get(),
                    "world": facts.__dict__,
                    "synonyms": synonyms,
                    "prompts": {k: _sha8("\n".join(v)) for k, v in sites.segments.items()},
                    "ambiguous_sites": sites.ambiguous_pairs(),
                    "live_calls": client.live_calls,
                    "net_attempts": Counter(world.net.attempts).most_common(),
                    "cleanup_calls": world.cleanup_calls,
                    "suppressed_writes": Counter(world.side_writes).most_common(),
                    "unavailable_calls": Counter(world.unavailable_calls).most_common(),
                    "synonym_journal_lines": world.journal_lines(),
                    "seconds": round(time.time() - started, 1),
                }
        problems = verify_world(world_root)
    turns = [json.loads(line) for line in (run_dir / "turns.jsonl").open(encoding="utf-8")]
    calls = Counter(c["outcome"] for t in turns for c in t["llm_calls"])
    header["llm_outcomes"] = dict(calls)
    header["world_problems"] = problems
    header["valid"] = not (header["net_attempts"] or header["cleanup_calls"] or problems or calls.get("unknown_site") or header["ambiguous_sites"])
    (run_dir / "run.json").write_text(json.dumps(header, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[evalbench] готово: {run_dir} valid={header['valid']} исходы={dict(Counter(t['outcome'] for t in turns))}", file=sys.__stderr__)
    return 0 if header["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
