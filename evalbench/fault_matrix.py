"""Матрица сбоев МИС: инструменты × сбои × день/ночь — замер «до» хода (д), шаг 2 (17.10).

    ./venv/bin/python -m evalbench.fault_matrix --world 20261008 --out ~/Dev/dla-private/bench/faults

Каждый сценарий — короткий диалог (1–2 хода) через `messenger_generate_once` в замороженном мире; LLM недоступна
(режим `down` — замер механики, без живых вызовов). Сбой ставит `evalbench.faults` на HTTP-границу клиентов МИС (живые
методы и скачивание срезов) или на срезы на диске. Исход хода — `detectors.classify_turn` (как у карточек стенда).

Что показывает: что пациент получает сегодня при каждом сбое — честный перевод (`handoff_tech` / `handoff`), оффер
оператора, или правдоподобную неправду (`no_info`, `clarify`, `answer` без данных); контр-метрика — доля переводов и
офферов оператора, отдельно день и ночь (правило ≤3).

Выход (вне репо — в ответах бывают адреса и цены мира): `<out>/<имя>.json` (строки ходов) и `<имя>.md` (сводка).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .detectors import Templates, classify_turn
from .faults import Faults
from .llm import HarnessClient, SiteIndex
from .prices import PriceCatalog
from .world import PRIVATE_ROOT, PROFILES, FrozenWorld, settle_shadows

ROOT = Path(__file__).resolve().parents[1]

# (имя, реплики): инструменты, у которых есть источник МИС.
SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("price", ("сколько стоит ферритин", "а цена?")),
    ("price_list", ("сколько стоит ОАК и глюкоза",)),
    ("doctor_price", ("сколько стоит прием уролога",)),
    ("prepare", ("как подготовиться к анализу крови на глюкозу",)),
    ("address", ("где сдать анализы",)),
    ("schedule_specialty", ("когда принимает уролог",)),
    ("schedule_surname", ("когда принимает Трубин",)),
    ("appointment", ("хочу записаться к урологу", "в любой филиал")),
    ("promotions", ("какие у вас акции",)),
)

# (имя, параметры Faults): «как есть» — мир без сбоя (живое расписание в мире недоступно всегда — это и есть его сбой).
FAULTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("as_is", {}),
    ("live_network", {"live": "network"}),
    ("live_timeout", {"live": "timeout"}),
    ("live_empty", {"live": "empty"}),
    ("live_http_5xx", {"live": "http_5xx"}),
    ("snap_stale_3", {"snapshots": "stale", "stale_days": 3}),
    ("snap_stale_8", {"snapshots": "stale", "stale_days": 8}),
    ("snap_none", {"snapshots": "none", "live": "network"}),
)

_HONEST = {"handoff_tech", "handoff", "operator_offer", "urgent"}


async def _play(turns: tuple[str, ...], session: str, tpl: Templates, catalog: PriceCatalog, world: FrozenWorld) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    memory, services, out = MemoryStore(), Services(), []
    for text in turns:
        req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id="")
        body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
        await settle_shadows()
        outcome = classify_turn(body, dict(memory.get(session).last_entities or {}), tpl, catalog)
        row = world.take_telemetry() or {}
        out.append(
            {
                "question": text,
                "answer": str(body.get("text") or ""),
                "outcome": outcome.outcome,
                "handoff": outcome.handoff,
                "handoff_reason": outcome.handoff_reason,
                "operator_offer": outcome.operator_offer,
                "answer_source": row.get("answer_source"),
                "src": row.get("src"),
            }
        )
    return out


def run(world_root: Path) -> list[dict[str, Any]]:
    from messengers_router import llm_runtime

    return _run(world_root, llm_runtime)


def _run(world_root: Path, llm_runtime: Any) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        for fault_name, params in FAULTS:
            with FrozenWorld(world_root, profile) as world:
                client = HarnessClient(mode="down", sites=SiteIndex.from_code_root(ROOT))
                real_client, llm_runtime._OLLAMA_CLIENT = llm_runtime._OLLAMA_CLIENT, client
                try:
                    tpl = Templates.from_bot()
                    catalog = PriceCatalog.from_world(world_root / "apidata")
                    with Faults(world, **params) as fault:

                        async def all_scenarios() -> list[tuple[str, list[dict[str, Any]]]]:
                            return [(name, await _play(turns, f"faults-{profile}-{fault_name}-{name}", tpl, catalog, world)) for name, turns in SCENARIOS]

                        played = asyncio.run(all_scenarios())
                        boundary_calls = Counter(call.rsplit("/", 1)[-1] for call in fault.calls)
                        written = fault.written()
                finally:
                    llm_runtime._OLLAMA_CLIENT = real_client
                attempts = list(world.net.attempts)
            for name, turns in played:
                rows.append(
                    {
                        "profile": profile,
                        "fault": fault_name,
                        "scenario": name,
                        "turns": turns,
                        "boundary_calls": dict(boundary_calls),
                        "net_attempts": attempts,
                        "written": written,
                    }
                )
    return rows


def summary(rows: list[dict[str, Any]], header: dict[str, Any]) -> str:
    lines = [
        f"# Матрица сбоев МИС — {header['name']}",
        "",
        f"Код `{header['code']}`, мир `{header['world']}`, LLM недоступна (режим down), {header['seconds']} с.",
        "Ячейка — исход последнего хода сценария (`detectors.classify_turn`); **жирным** — честный: перевод или оффер оператора.",
        "",
    ]
    for profile in PROFILES:
        lines += [f"## {profile}", "", "| сценарий | " + " | ".join(name for name, _ in FAULTS) + " |", "|---" * (len(FAULTS) + 1) + "|"]
        cell: dict[tuple[str, str], str] = {}
        for row in rows:
            if row["profile"] != profile:
                continue
            last = row["turns"][-1]["outcome"]
            cell[(row["scenario"], row["fault"])] = f"**{last}**" if last in _HONEST else last
        for name, _turns in SCENARIOS:
            lines.append(f"| {name} | " + " | ".join(cell.get((name, fault), "—") for fault, _ in FAULTS) + " |")
        lines.append("")
    lines += ["## Контр-метрика: доля честных исходов (перевод + оффер) среди всех ходов", "", "| сбой | день | ночь |", "|---|---|---|"]
    share: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        for turn in row["turns"]:
            key = (row["fault"], row["profile"])
            share[key][0] += turn["outcome"] in _HONEST
            share[key][1] += 1
    for fault, _ in FAULTS:
        cells = []
        for profile in PROFILES:
            honest, total = share[(fault, profile)]
            cells.append(f"{honest}/{total}")
        lines.append(f"| {fault} | " + " | ".join(cells) + " |")
    attempts = sorted({a for row in rows for a in row["net_attempts"]})
    written = sorted({w for row in rows for w in row["written"]})
    lines += ["", f"Попытки сети: {attempts or 'нет'}. Записано срезов в пустые папки: {written or 'ничего'}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", default="20261008", help="дата мира ГГГГММДД (каталог ~/Dev/dla-private/bench/worlds/<дата>)")
    ap.add_argument("--out", type=Path, default=PRIVATE_ROOT / "faults")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    world_root = PRIVATE_ROOT / "worlds" / args.world
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    code = f"{sha}{'+wip' if dirty else ''}"
    name = args.name or f"faults_{code.replace('+', '_')}_{args.world}"
    started = time.time()
    rows = run(world_root)
    header = {"name": name, "code": code, "world": args.world, "seconds": round(time.time() - started, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps({"header": header, "rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out / f"{name}.md").write_text(summary(rows, header), encoding="utf-8")
    print(f"[fault_matrix] {name}: {len(rows)} диалогов → {args.out / (name + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
