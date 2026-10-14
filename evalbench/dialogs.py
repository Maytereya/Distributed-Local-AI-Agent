"""Диалоги из нескольких ходов через эндпоинт бота в замороженном мире — двухходовые проверки стенда.

    ./venv/bin/python -m evalbench.dialogs run --set evalbench/sets/two_turn_l14.jsonl \\
        --code-root ~/Dev/dla-private/bench/code_d2d6a67 --mode real \\
        --cache ~/Dev/dla-private/bench/llm_cache_dialogs.jsonl --cfg MR_CONTEXT_SHOWN=on --name dlg_on
    ./venv/bin/python -m evalbench.dialogs show dlg_off dlg_on > ~/Dev/dla-private/bench/cards/dlg_off__vs__dlg_on.md

Зачем (мастер-хэндофф §4 (б), «слепое пятно плана», решение 13.10): `evalbench.run` меряет один ход в новой
сессии, а контекст между ходами (`context_shown`, утечка L-13, висящие вопросы) виден только во втором ходе.
Здесь диалог — новая сессия, его ходы идут подряд в ней через `messenger_generate_once` (`debug=true`), как у
пациента; мир, режимы LLM, `--cfg` и проверки действительности — как у `evalbench.run`.

Набор — JSONL: `{"id", "turns": ["реплика" | {"text", "button"}], "expect": "что ждём", "note": "..."}`.
Выход — каталог прогона вне репо (`~/Dev/dla-private/bench/runs/<имя>/`):
- `dialogs.jsonl` — строка на диалог; на каждый ход: реплика, ответ, исход, задержка, вызовы LLM, телеметрия,
  контекст сессии после хода (`service_name`, `_shown_services`, висящие вопросы) и строки журнала синонимов,
  которые дописал этот ход;
- `run.json` — шапка как у `evalbench.run`; `bot.log`; `synonym_journal.jsonl`.

`show` печатает один или два прогона в Markdown — ответы по ходам рядом, для разбора глазами. В нём реплики
набора и ответы бота: класть в каталог карточек стенда (вне репо), не в журнал.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import card as _card  # noqa: F401 — модули стенда грузим до смены пути к дереву кода
from .detectors import Templates, classify_turn
from .llm import MODES, HarnessClient, LiveBudgetExhausted, ResponseCache, SiteIndex
from .prices import PriceCatalog
from .run import TURN_TIMEOUT_S, _enter_code_root, _llm_reachable, _parse_cfg, _sha8, git_sha
from .world import PRIVATE_ROOT, PROFILES, FrozenWorld, host_of, verify_world

SHOWN_KEY = "_shown_services"  # `messengers_router.shown_context.SHOWN_KEY` (L-14, шаг 3); в коде до 14.10 его нет


def load_dialogs(paths: list[Path], *, ids: set[str] | None = None) -> list[dict[str, Any]]:
    dialogs = []
    for path in paths:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("set", path.stem)
            turns = row.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"{path.name}: у диалога {row.get('id')!r} нет ходов")
            if ids and row["id"] not in ids:
                continue
            dialogs.append(row)
    return dialogs


def _turn(item: Any) -> tuple[str, str]:
    """Ход набора → (реплика, кнопка): строка — реплика; словарь — `{"text", "button"}`."""

    if isinstance(item, dict):
        return str(item.get("text") or ""), str(item.get("button") or "")
    return str(item), ""


def state_view(entities: dict[str, Any]) -> dict[str, Any]:
    """Контекст сессии после хода — то, что прочтёт следующий ход: услуга, показанное, висящие вопросы."""

    view: dict[str, Any] = {}
    for key in ("service_name", "test_name", SHOWN_KEY):
        value = entities.get(key)
        if value not in (None, "", [], ()):
            view[key] = list(value) if isinstance(value, (list, tuple)) else value
    waiting = sorted(key for key, value in entities.items() if key.endswith("_pending") and key != "_pending" and value)
    if waiting:
        view["waiting"] = waiting
    pending = entities.get("_pending")
    if isinstance(pending, dict):
        view["pending"] = {"label": pending.get("label"), "missing": pending.get("missing")}
    return view


def _decision(body: dict[str, Any]) -> dict[str, Any]:
    debug = (body.get("state_update") or {}).get("debug") or {}
    plan = debug.get("plan") or {}
    return {
        "plan": [plan.get("label")] + [step.get("tool") for step in plan.get("steps") or []] if plan else None,
        "route_error": debug.get("route_error") or debug.get("endpoint_error"),
    }


def _journal_rows(lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


async def _run_dialogs(dialogs, *, client, world, tpl: Templates, catalog: PriceCatalog, llm_mode: str, run_name: str, out, progress) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    memory = MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)
    services = Services()
    stats: dict[str, Any] = {"done": 0, "turns": 0, "stopped": None}
    journal_seen = world.journal_lines()
    for i, dialog in enumerate(dialogs, 1):
        sid = f"dlg-{run_name}-{dialog['id']}"
        turns: list[dict[str, Any]] = []
        for text, button in map(_turn, dialog["turns"]):
            req = MessengerGenerateRequest(session_id=sid, text=text, debug=True, llm_mode=llm_mode, button_id=button)
            started = time.monotonic()
            error = None
            try:
                resp = await asyncio.wait_for(messenger_generate_once(req, memory=memory, services=services), TURN_TIMEOUT_S)
                body = json.loads(resp.body)
            except LiveBudgetExhausted as exc:
                stats["stopped"] = str(exc)
                break
            except Exception as exc:  # ход не завершился — фиксируем, диалог идёт дальше
                body = {"text": "", "handoff": False, "state_update": {"debug": {"endpoint_error": f"{type(exc).__name__}: {exc}"}}}
                error = type(exc).__name__
            ms = int((time.monotonic() - started) * 1000)
            entities = dict(memory.get(sid).last_entities or {})
            calls = client.take_turn_calls()
            if any(c.outcome == "budget" for c in calls):
                stats["stopped"] = "исчерпан бюджет живых вызовов LLM"
                break
            journal = world.journal_tail(journal_seen)
            journal_seen += len(journal)
            outcome = classify_turn(body, entities, tpl, catalog)
            turns.append(
                {
                    "text": text,
                    "button": button or None,
                    "answer": body.get("text", ""),
                    "ms": ms,
                    "harness_error": error,
                    **outcome.as_row(),
                    **_decision(body),
                    "state": state_view(entities),
                    "journal": _journal_rows(journal),
                    "llm_calls": [c.__dict__ for c in calls],
                    "telemetry": world.take_telemetry(),
                }
            )
        if stats["stopped"]:
            break  # недоигранный диалог не пишем: второй ход без первого не о том
        row = {key: dialog.get(key) for key in ("id", "set", "expect", "note")}
        out.write(json.dumps({**row, "turns": turns}, ensure_ascii=False) + "\n")
        out.flush()
        stats["done"] = i
        stats["turns"] += len(turns)
        progress(f"{i}/{len(dialogs)} {dialog['id']}: " + " → ".join(t["outcome"] for t in turns))
    return stats


def run_main(args: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    try:
        overrides = dict(_parse_cfg(item) for item in args.cfg)
    except ValueError as exc:
        ap.error(str(exc))
    code_root = args.code_root.resolve()
    world_root = PRIVATE_ROOT / "worlds" / args.world
    if not (world_root / "world.json").exists():
        ap.error(f"мира {world_root} нет: соберите его через evalbench.run --build-world-from <apidata>")
    try:
        dialogs = load_dialogs([p.expanduser() for p in args.sets], ids=set(args.ids.split(",")) if args.ids else None)
    except ValueError as exc:
        ap.error(str(exc))
    sha = git_sha(code_root)
    name = args.name or f"dlg_{sha}_{args.mode}_{args.profile}"
    run_dir = PRIVATE_ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(PRIVATE_ROOT, 0o700)

    _enter_code_root(code_root)
    from messengers_router import llm_runtime
    from messengers_router.runtime_config import config as bot_config

    cfg_applied = {key: {"value": value, "known": hasattr(bot_config, key)} for key, value in overrides.items()}
    for key, value in overrides.items():
        setattr(bot_config, key, value)

    live = args.mode in ("real", "cached")
    allow = (host_of(bot_config.ollama_url),) if live else ()
    if live and not _llm_reachable(bot_config.ollama_url):
        print("[dialogs] прод-LLM не отвечает (VPN?) — живой прогон не начат", file=sys.__stderr__)
        return 3
    console = sys.__stderr__
    header: dict[str, Any] = {}
    with open(run_dir / "bot.log", "w", encoding="utf-8") as log_fh, open(run_dir / "dialogs.jsonl", "w", encoding="utf-8") as out:
        with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
            import logging

            logging.basicConfig(level=logging.INFO, stream=log_fh, force=True)
            with FrozenWorld(world_root, args.profile, allow_hosts=allow) as world:
                from agent_logic_2.ollama_settings import LLMName
                from messengers_router.services import _biomaterial

                from agent_logic_2 import ollama_settings

                sites = SiteIndex.from_code_root(code_root)
                client = HarnessClient(
                    mode=args.mode,
                    sites=sites,
                    real=llm_runtime._OLLAMA_CLIENT if live else None,
                    cache=ResponseCache(args.cache) if args.mode in ("real", "cached", "replay") else None,
                    live_pause_s=args.live_pause,
                    max_live_calls=args.max_live_calls,
                    default_options=ollama_settings.options_set().model_dump(exclude_none=True),
                )
                real_client = llm_runtime._OLLAMA_CLIENT
                llm_runtime._OLLAMA_CLIENT = client
                try:
                    tpl = Templates.from_bot()
                    catalog = PriceCatalog.from_world(world_root / "apidata")
                    vocab = _biomaterial._vocabularies()
                    synonyms = len(getattr(vocab, "synonym_candidates", ()) or ())
                    print(f"[dialogs] {name}: {len(dialogs)} диалогов, словарь синонимов {synonyms}", file=console)

                    def progress(msg: str) -> None:
                        print(f"[dialogs] {msg}", file=console, flush=True)

                    started = time.time()
                    stats = asyncio.run(
                        _run_dialogs(dialogs, client=client, world=world, tpl=tpl, catalog=catalog, llm_mode=args.llm_mode, run_name=name, out=out, progress=progress)
                    )
                finally:
                    llm_runtime._OLLAMA_CLIENT = real_client
                world.save_journal(run_dir / "synonym_journal.jsonl")
                facts = world.facts()
                header = {
                    "name": name,
                    "kind": "dialogs",
                    "code_sha": sha,
                    "code_root": str(code_root),
                    "sets": [str(p) for p in args.sets],
                    "dialogs": len(dialogs),
                    "done": stats["done"],
                    "turns": stats["turns"],
                    "stopped": stats["stopped"],
                    "mode": args.mode,
                    "llm_mode": args.llm_mode,
                    "cfg": cfg_applied,
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
    rows = [json.loads(line) for line in (run_dir / "dialogs.jsonl").open(encoding="utf-8") if line.strip()]
    calls = Counter(c["outcome"] for row in rows for t in row["turns"] for c in t["llm_calls"])
    header["llm_outcomes"] = dict(calls)
    header["world_problems"] = problems
    header["valid"] = not (header["net_attempts"] or header["cleanup_calls"] or problems or calls.get("unknown_site") or header["ambiguous_sites"])
    (run_dir / "run.json").write_text(json.dumps(header, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[dialogs] готово: {run_dir} valid={header['valid']} диалогов {stats['done']}/{len(dialogs)}, ходов {stats['turns']}", file=sys.__stderr__)
    return 0 if header["valid"] and not stats["stopped"] else 2


# --------------------------------------------------------------------------- show


def load_dialog_run(ref: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(ref).expanduser()
    if not path.exists():
        path = PRIVATE_ROOT / "runs" / ref
    header = json.loads((path / "run.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "dialogs.jsonl").open(encoding="utf-8") if line.strip()]
    return header, {row["id"]: row for row in rows}


def _short(text: Any, limit: int) -> str:
    flat = " ⏎ ".join(line.strip() for line in str(text or "").strip().splitlines() if line.strip())
    return flat if limit <= 0 or len(flat) <= limit else flat[: limit - 1] + "…"


def _state_line(state: dict[str, Any]) -> str:
    parts = []
    if state.get("service_name"):
        parts.append(f"service_name «{state['service_name']}»")
    if state.get("test_name"):
        parts.append(f"test_name «{state['test_name']}»")
    if state.get(SHOWN_KEY):
        parts.append("показано: " + "; ".join(f"«{name}»" for name in state[SHOWN_KEY]))
    if state.get("waiting"):
        parts.append("ждёт: " + ", ".join(state["waiting"]))
    if state.get("pending"):
        parts.append(f"pending {state['pending'].get('label')} {state['pending'].get('missing')}")
    return "; ".join(parts) or "—"


def _calls_line(turn: dict[str, Any]) -> str:
    calls = turn.get("llm_calls") or []
    if not calls:
        return "LLM: —"
    return "LLM: " + ", ".join(f"{c['site']} {c['outcome']} {c['ms'] / 1000:.1f} с" for c in calls)


def _journal_line(row: dict[str, Any]) -> str:
    if "raw" in row:
        return f"журнал: {row['raw']}"
    services = ", ".join(s.get("name") or "" for s in row.get("services") or [])
    return f"журнал: «{row.get('phrase')}» found={row.get('found')}" + (f" source={row['source']}" if row.get("source") else "") + (f" → {services}" if services else "")


def render_dialogs(runs: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]], *, limit: int = 400) -> str:
    tags = ["A", "B"][: len(runs)] if len(runs) == 2 else [""]
    lines = ["# Диалоги стенда", ""]
    for tag, (header, _rows) in zip(tags, runs):
        cfg = ", ".join(f"{k}={v.get('value')}" for k, v in (header.get("cfg") or {}).items())
        world = header.get("world") or {}
        lines.append(
            f"- {'**' + tag + '** ' if tag else ''}`{header['name']}`: код `{header.get('code_sha')}`, LLM `{header.get('mode')}`, "
            f"мир {world.get('date')} {world.get('profile')}, диалогов {header.get('done')}/{header.get('dialogs')}, "
            f"живых вызовов {header.get('live_calls')}, valid={header.get('valid')}"
            + (f", **остановлен: {header['stopped']}**" if header.get("stopped") else "")
            + (f", настройки {cfg}" if cfg else "")
        )
    order: list[str] = []
    for _header, rows in runs:
        order += [key for key in rows if key not in order]
    differ = 0
    for dialog_id in order:
        found = [rows.get(dialog_id) for _header, rows in runs]
        first = next(d for d in found if d)
        lines += ["", f"## {dialog_id}" + (f" — ждём: {first['expect']}" if first.get("expect") else "")]
        if first.get("note"):
            lines.append(f"_{first['note']}_")
        for k in range(max(len(d["turns"]) if d else 0 for d in found)):
            turn0 = next(d["turns"][k] for d in found if d and k < len(d["turns"]))
            lines += ["", f"**{k + 1}. «{turn0['text']}»**" + (f" (кнопка `{turn0['button']}`)" if turn0.get("button") else "")]
            answers = []
            for tag, dialog in zip(tags, found):
                prefix = f"[{tag}] " if tag else ""
                turn = dialog["turns"][k] if dialog and k < len(dialog["turns"]) else None
                if turn is None:
                    lines.append(f"- {prefix}хода нет")
                    continue
                answers.append(str(turn.get("answer") or "").strip())
                lines.append(f"- {prefix}{turn.get('outcome')}, {turn.get('ms')} мс: {_short(turn.get('answer'), limit)}")
                lines.append(f"  - контекст после хода: {_state_line(turn.get('state') or {})}")
                lines.append(f"  - {_calls_line(turn)}")
                lines += [f"  - {_journal_line(row)}" for row in turn.get("journal") or []]
            if len(runs) == 2 and len(answers) == 2:
                same = answers[0] == answers[1]
                differ += not same
                lines.append("- ответы совпадают" if same else "- **ответы различаются**")
    if len(runs) == 2:
        lines += ["", f"**Ходов с разным ответом:** {differ}"]
    return "\n".join(lines)


def show_main(args: argparse.Namespace) -> int:
    runs = [load_dialog_run(ref) for ref in args.runs[:2]]
    print(render_dialogs(runs, limit=args.limit))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run_ap = sub.add_parser("run", help="прогнать набор диалогов")
    run_ap.add_argument("--set", dest="sets", action="append", type=Path, required=True, help="JSONL диалогов (можно несколько)")
    run_ap.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parent.parent, help="дерево кода бота")
    run_ap.add_argument("--world", default="20261008", help="дата мира ГГГГММДД (каталог ~/Dev/dla-private/bench/worlds/<дата>)")
    run_ap.add_argument("--profile", choices=PROFILES, default="day")
    run_ap.add_argument("--mode", choices=MODES, default="down")
    run_ap.add_argument("--llm-mode", default="hybrid", help="llm_mode запроса (шлюз прода шлёт hybrid)")
    run_ap.add_argument("--cache", type=Path, default=PRIVATE_ROOT / "llm_cache.jsonl", help="запись ответов настоящей LLM")
    run_ap.add_argument("--live-pause", type=float, default=1.0, help="пауза между вызовами настоящей LLM, с")
    run_ap.add_argument("--max-live-calls", type=int, help="потолок вызовов настоящей LLM на прогон")
    run_ap.add_argument("--ids", help="только эти диалоги через запятую")
    run_ap.add_argument("--name", help="имя прогона (по умолчанию dlg_<sha>_<режим>_<профиль>)")
    run_ap.add_argument("--cfg", action="append", default=[], metavar="ИМЯ=ЗНАЧЕНИЕ", help="переопределить настройку бота на прогон, как в evalbench.run")
    show_ap = sub.add_parser("show", help="напечатать один или два прогона диалогов в Markdown")
    show_ap.add_argument("runs", nargs="+", help="имя прогона или путь к каталогу (1 или 2)")
    show_ap.add_argument("--limit", type=int, default=400, help="обрезать ответ до стольких знаков (0 — целиком)")
    args = ap.parse_args(argv)
    if args.command == "run":
        return run_main(args, run_ap)
    return show_main(args)


if __name__ == "__main__":
    sys.exit(main())
