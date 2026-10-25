"""Прогон набора вопросов через эндпоинт бота в замороженном мире.

    ./venv/bin/python -m evalbench.run --set ~/Dev/dla-private/golden_v1.jsonl \\
        --code-root /path/to/worktree-2b9278d --world 20261008 --profile day --mode down

Каждый вопрос — один ход в новой сессии (решение 09.10: одиночный ход честнее для
сравнения версий — второй ход зависит от первого ответа). Ход идёт через
`messenger_generate_once` с `debug=true`, в одном asyncio-цикле на весь прогон.

Вопрос набора с `"press": true` и листом меню в `button` идёт как у пациента в меню шлюза (25.10):
нажатие листа (`button_id` + подпись кнопки в `text`, «Таблица экранов» — `evalbench.buttons`),
затем вопрос свободным текстом без `button_id`, в той же сессии. Одним запросом кнопка-тема `slot`
вопрос не читает вовсе — тема и сущности берутся из кнопки, ответом была бы подсказка кнопки.
Исход и метрики — по второму ходу; нажатие — в поле `press`. После кнопки-перевода второго хода нет.

Выход — каталог прогона вне репо (`~/Dev/dla-private/bench/runs/<имя>/`):
- `turns.jsonl` — строка на вопрос: вопрос, ответ, исход, признаки, вызовы LLM, задержка;
- `run.json` — шапка: код, мир, модель, режим, отпечатки промптов, словарь синонимов,
  попытки сети, очистки, промахи; `valid=false`, если прогон нельзя принимать;
- `bot.log` — вывод бота (в нём бывают адреса МИС — поэтому тоже вне репо);
- `synonym_journal.jsonl` — что прогон записал бы в журнал синонимов (в нём реплики пациентов).
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
from .world import PRIVATE_ROOT, PROFILES, FrozenWorld, build_world, host_of, settle_shadows, verify_world

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
    """Короткий sha дерева кода; «-dirty», если в отслеживаемых файлах есть незакоммиченные правки
    (иначе прогон на правке подписывается коммитом, в котором её нет)."""

    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    except Exception:
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _llm_reachable(url: str, timeout_s: float = 8.0) -> bool:
    """Отвечает ли сервер LLM (`/api/version`) — до прогона, а не тайм-аутом на каждом вопросе."""

    import urllib.request

    try:
        urllib.request.urlopen(url.rstrip("/") + "/api/version", timeout=timeout_s).read()
    except Exception:
        return False
    return True


def _parse_cfg(item: str) -> tuple[str, Any]:
    """`ИМЯ=ЗНАЧЕНИЕ` → (ИМЯ, значение): true/false — bool, целое — int, иначе строка."""

    name, sep, raw = item.partition("=")
    if not sep or not name.strip():
        raise ValueError(f"--cfg ждёт ИМЯ=ЗНАЧЕНИЕ, получено {item!r}")
    value: Any = raw.strip()
    if value.lower() in ("true", "false"):
        value = value.lower() == "true"
    elif value.lstrip("-").isdigit():
        value = int(value)
    return name.strip(), value


def _enter_code_root(code_root: Path) -> None:
    """Бот импортируется из `code_root`; относительные пути бота (app_data) — от него же."""

    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(1, str(REPO_ROOT))


async def _send(req, *, memory, services) -> tuple[dict[str, Any], str | None, int]:
    """Один ход через эндпоинт: (тело ответа, имя исключения стенда, мс). `LiveBudgetExhausted` — наружу."""

    from messengers_router.endpoint import messenger_generate_once

    started = time.monotonic()
    error = None
    try:
        resp = await asyncio.wait_for(messenger_generate_once(req, memory=memory, services=services), TURN_TIMEOUT_S)
        body = json.loads(resp.body)
    except LiveBudgetExhausted:
        raise
    except Exception as exc:  # ход не завершился — фиксируем, идём дальше
        body = {"text": "", "handoff": False, "state_update": {"debug": {"endpoint_error": f"{type(exc).__name__}: {exc}"}}}
        error = type(exc).__name__
    return body, error, int((time.monotonic() - started) * 1000)


# Контекст сессии после хода, который читает критерий кнопки (запись дошла до нужной специальности, услуга хода).
_STATE_KEYS = ("service_name", "test_name", "specialty", "doctor_name", "doctor_id", "branch_name", "appointment_flow_active", "appointment_action", "_last_label")


def state_after(entities: dict[str, Any]) -> dict[str, Any]:
    view = {key: entities[key] for key in _STATE_KEYS if entities.get(key) not in (None, "", [], ())}
    pending = entities.get("_pending")
    if isinstance(pending, dict):
        view["pending"] = {"label": pending.get("label"), "missing": pending.get("missing")}
    return view


async def _exchange(case, sid: str, *, memory, services, llm_mode: str, tpl: Templates, catalog: PriceCatalog) -> tuple[dict[str, Any], str | None, int, dict[str, Any] | None]:
    """Вопрос набора → (тело ответа на вопрос, ошибка стенда, мс, нажатие или None).

    `press` — сначала нажатие листа `button` с подписью из таблицы шлюза, потом вопрос без `button_id`. Кнопка-перевод
    (`handoff`) отдаёт пациента оператору: вопрос не отправляется, ответом хода остаётся ответ нажатия.
    """

    from messengers_router.endpoint import MessengerGenerateRequest

    from .buttons import GATEWAY_CAPTIONS

    press = None
    if case.get("press") and case.get("button"):
        button = str(case["button"])
        req = MessengerGenerateRequest(session_id=sid, text=GATEWAY_CAPTIONS[button], debug=True, llm_mode=llm_mode, button_id=button)
        body, error, ms = await _send(req, memory=memory, services=services)
        outcome = classify_turn(body, dict(memory.get(sid).last_entities or {}), tpl, catalog)
        press = {"button": button, "answer": body.get("text", ""), "outcome": outcome.outcome, "label": outcome.label, "handoff": outcome.handoff, "ms": ms, "error": error}
        if outcome.handoff or error:
            press["question_sent"] = False
            return body, error, ms, press
        press["question_sent"] = True
        req = MessengerGenerateRequest(session_id=sid, text=str(case["question"]), debug=True, llm_mode=llm_mode)
    else:
        req = MessengerGenerateRequest(session_id=sid, text=str(case["question"]), debug=True, llm_mode=llm_mode, button_id=case.get("button") or "")
    body, error, ms = await _send(req, memory=memory, services=services)
    return body, error, ms, press


def _case_row(case, body: dict[str, Any], error: str | None, ms: int, press: dict[str, Any] | None, outcome, state) -> dict[str, Any]:
    """Общие поля строки хода у обоих режимов прогона: вопрос, ответ, исход, нажатие и контекст после хода."""

    return {
        "id": case["id"],
        "set": case.get("set"),
        "split": case.get("split"),
        "category": case.get("category") or case.get("topic_llm"),
        "tier": case.get("tier"),
        "expect_urgent": case.get("expect_urgent"),
        "question": case["question"],
        "button": case.get("button"),
        "press": press,
        "answer": body.get("text", ""),
        "ms": ms,
        "harness_error": error,
        **outcome.as_row(),
        "state": state_after(dict(state.last_entities or {})),
        "stage_ms": ((body.get("state_update") or {}).get("debug") or {}).get("stage_timings"),
    }


async def _run_cases(cases, *, client: HarnessClient, world: FrozenWorld, tpl: Templates, catalog: PriceCatalog, llm_mode: str, run_name: str, out, progress, concurrency: int = 1) -> dict[str, Any]:
    if concurrency > 1:
        return await _run_cases_concurrent(
            cases, client=client, world=world, tpl=tpl, catalog=catalog, llm_mode=llm_mode, run_name=run_name, out=out, progress=progress, concurrency=concurrency
        )
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    memory = MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)
    services = Services()
    stats: dict[str, Any] = {"done": 0, "stopped": None}
    for i, case in enumerate(cases, 1):
        sid = f"bench-{run_name}-{case['id']}"
        try:
            body, error, ms, press = await _exchange(case, sid, memory=memory, services=services, llm_mode=llm_mode, tpl=tpl, catalog=catalog)
        except LiveBudgetExhausted as exc:
            stats["stopped"] = str(exc)
            break
        await settle_shadows()  # тени, которые ход не ждёт, — после замера, до вызовов и телеметрии
        state = memory.get(sid)
        calls = client.take_turn_calls()
        if any(c.outcome == "budget" for c in calls):
            stats["stopped"] = "исчерпан бюджет живых вызовов LLM"
            break
        outcome = classify_turn(body, dict(state.last_entities or {}), tpl, catalog)
        row = {
            **_case_row(case, body, error, ms, press, outcome, state),
            "llm_calls": [c.__dict__ for c in calls],
            "telemetry": world.take_telemetry(),
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        stats["done"] = i
        if i % 10 == 0 or i == len(cases):
            progress(f"{i}/{len(cases)} последний: {case['id']} → {outcome.outcome} ({ms} мс)")
    return stats


async def _run_cases_concurrent(cases, *, client: HarnessClient, world: FrozenWorld, tpl: Templates, catalog: PriceCatalog, llm_mode: str, run_name: str, out, progress, concurrency: int) -> dict[str, Any]:
    """Нагрузочный режим (ход (е)): `concurrency` ходов разных сессий одновременно, в одном процессе бота.

    Меряется время хода под параллельной нагрузкой: очередь к LLM (`llm_max_concurrency`), деление CPU
    между ходами. Вызовы LLM стенда копятся на весь прогон, а не на ход, — при параллельных ходах их не
    разнести, поэтому в строке хода `llm_calls = null`; телеметрия хода — по сессии хода (`take_telemetry_for`:
    строка несёт хэш сессии), в ней и вызовы LLM хода (`llm`). Строки, дописанные после ответа и пришедшие
    позже, в ход не попадают. Ответ, исход и время — свои у каждого хода.
    """

    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    memory = MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)
    services = Services()
    stats: dict[str, Any] = {"done": 0, "stopped": None}
    queue: asyncio.Queue = asyncio.Queue()
    for item in enumerate(cases, 1):
        queue.put_nowait(item)
    t0 = time.monotonic()

    async def worker(slot: int) -> None:
        while not queue.empty() and not stats["stopped"]:
            i, case = queue.get_nowait()
            sid = f"bench-{run_name}-{case['id']}"
            started = time.monotonic()
            try:
                body, error, ms, press = await _exchange(case, sid, memory=memory, services=services, llm_mode=llm_mode, tpl=tpl, catalog=catalog)
            except LiveBudgetExhausted as exc:
                stats["stopped"] = str(exc)
                return
            state = memory.get(sid)
            outcome = classify_turn(body, dict(state.last_entities or {}), tpl, catalog)
            row = {
                **_case_row(case, body, error, ms, press, outcome, state),
                "llm_calls": None,
                "telemetry": world.take_telemetry_for(sid),
                "load": {"concurrency": concurrency, "slot": slot, "start_ms": int((started - t0) * 1000)},
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            stats["done"] += 1
            if stats["done"] % 10 == 0 or stats["done"] == len(cases):
                progress(f"{stats['done']}/{len(cases)} последний: {case['id']} → {outcome.outcome} ({ms} мс)")

    await asyncio.gather(*(worker(k) for k in range(concurrency)))
    await settle_shadows()
    stats["load_llm_outcomes"] = dict(Counter(c.outcome for c in client.take_turn_calls()))
    world.take_telemetry()
    stats["wall_s"] = round(time.monotonic() - t0, 1)
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
    ap.add_argument("--concurrency", type=int, default=1, help="ход (е): столько ходов разных сессий одновременно (нагрузка); >1 — без вызовов LLM по ходам")
    ap.add_argument("--ids", help="только эти id через запятую")
    ap.add_argument("--split", choices=("dev", "test"))
    ap.add_argument("--name", help="имя прогона (по умолчанию <sha>_<режим>_<профиль>)")
    ap.add_argument("--shard", help="часть K/N набора (K с 1): N процессов параллельно; склейка — evalbench.merge")
    ap.add_argument(
        "--cfg",
        action="append",
        default=[],
        metavar="ИМЯ=ЗНАЧЕНИЕ",
        help="переопределить настройку бота на прогон (атрибут agent_logic_2.config, например MR_TURN_UNDERSTANDING_PRICE=on); пишется в run.json",
    )
    args = ap.parse_args(argv)
    try:
        overrides = dict(_parse_cfg(item) for item in args.cfg)
    except ValueError as exc:
        ap.error(str(exc))

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

    # Выключатели — на весь прогон, до первого хода; в дереве без такой настройки она ни на что
    # не влияет, поэтому в run.json пишется и то, была ли она в коде.
    cfg_applied = {key: {"value": value, "known": hasattr(bot_config, key)} for key, value in overrides.items()}
    for key, value in overrides.items():
        setattr(bot_config, key, value)

    live = args.mode in ("real", "cached")
    allow = (host_of(bot_config.ollama_url),) if live else ()
    if live and not _llm_reachable(bot_config.ollama_url):
        print("[evalbench] прод-LLM не отвечает (VPN?) — живой прогон не начат", file=sys.__stderr__)
        return 3
    console = sys.__stderr__
    header: dict[str, Any] = {}
    with open(run_dir / "bot.log", "w", encoding="utf-8") as log_fh, open(run_dir / "turns.jsonl", "w", encoding="utf-8") as out:
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
                    print(f"[evalbench] {name}: {len(cases)} вопросов, словарь синонимов {synonyms}", file=console)

                    def progress(msg: str) -> None:
                        print(f"[evalbench] {msg}", file=console, flush=True)

                    started = time.time()
                    stats = asyncio.run(
                        _run_cases(cases, client=client, world=world, tpl=tpl, catalog=catalog, llm_mode=args.llm_mode, run_name=name, out=out, progress=progress, concurrency=args.concurrency)
                    )
                finally:
                    llm_runtime._OLLAMA_CLIENT = real_client
                world.save_journal(run_dir / "synonym_journal.jsonl")
                facts = world.facts()
                header = {
                    "name": name,
                    "code_sha": sha,
                    "code_root": str(code_root),
                    "sets": [str(p) for p in args.sets],
                    "cases": len(cases),
                    "done": stats["done"],
                    "stopped": stats["stopped"],
                    "concurrency": args.concurrency,
                    "load_llm_outcomes": stats.get("load_llm_outcomes"),
                    "wall_s": stats.get("wall_s") or round(time.time() - started, 1),
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
    turns = [json.loads(line) for line in (run_dir / "turns.jsonl").open(encoding="utf-8")]
    calls = Counter(c["outcome"] for t in turns for c in (t["llm_calls"] or []))
    header["llm_outcomes"] = dict(calls)
    header["world_problems"] = problems
    header["valid"] = not (header["net_attempts"] or header["cleanup_calls"] or problems or calls.get("unknown_site") or header["ambiguous_sites"])
    (run_dir / "run.json").write_text(json.dumps(header, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[evalbench] готово: {run_dir} valid={header['valid']} исходы={dict(Counter(t['outcome'] for t in turns))}", file=sys.__stderr__)
    return 0 if header["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
