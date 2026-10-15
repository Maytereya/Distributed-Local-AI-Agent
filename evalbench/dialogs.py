"""Диалоги из нескольких ходов через эндпоинт бота в замороженном мире — двухходовые проверки стенда.

    ./venv/bin/python -m evalbench.dialogs run --set evalbench/sets/two_turn_l14.jsonl \\
        --code-root ~/Dev/dla-private/bench/code_d2d6a67 --mode real \\
        --cache ~/Dev/dla-private/bench/llm_cache_dialogs.jsonl --cfg MR_CONTEXT_SHOWN=on --name dlg_on
    ./venv/bin/python -m evalbench.dialogs show dlg_off dlg_on > ~/Dev/dla-private/bench/cards/dlg_off__vs__dlg_on.md
    ./venv/bin/python -m evalbench.dialogs card dlg_off dlg_on > ~/Dev/dla-private/bench/cards/dlg_off__vs__dlg_on_card.md

Зачем (мастер-хэндофф §4 (б), «слепое пятно плана», решение 13.10): `evalbench.run` меряет один ход в новой
сессии, а контекст между ходами (`context_shown`, утечка L-13, висящие вопросы) виден только во втором ходе.
Здесь диалог — новая сессия, его ходы идут подряд в ней через `messenger_generate_once` (`debug=true`), как у
пациента; мир, режимы LLM, `--cfg` и проверки действительности — как у `evalbench.run`.

Набор — JSONL: `{"id", "turns": ["реплика" | {"text", "button"}], "expect": "что ждём", "note": "...", "tags": [классы],
"about_shown": [номера ходов с 1]}`. `tags` — классы диалога для разбивки карточки (context_shown, prepare, yes_no, …);
`about_shown` — ходы без своей услуги (вопрос о показанном: «а цена?», «да», «1», «а как подготовиться?»): на них
карточка проверяет «контекст = показанное». Разметка читается из текущих файлов наборов — правка разметки не требует
перепрогона.
Выход — каталог прогона вне репо (`~/Dev/dla-private/bench/runs/<имя>/`):
- `dialogs.jsonl` — строка на диалог; на каждый ход: реплика, ответ, исход, задержка, вызовы LLM, телеметрия,
  контекст сессии после хода (`service_name`, `_shown_services`, висящие вопросы) и строки журнала синонимов,
  которые дописал этот ход;
- `run.json` — шапка как у `evalbench.run`; `bot.log`; `synonym_journal.jsonl`.

`show` печатает один или два прогона в Markdown — ответы по ходам рядом, сводка и отметки нарушений по ходам, для
разбора глазами. В нём реплики набора и ответы бота: класть в каталог карточек стенда (вне репо), не в журнал.
`card` — счётчики без реплик (как `evalbench.card`): служебный текст в ответе, «контекст = показанное» (цена или
название услуги, которой пациент не видел; слова вопроса названием услуги), петли (тот же вопрос бота или тот же
не-ответ два хода подряд), исходы продолжений, разбивка по классам — её можно вставлять в журнал и хэндофф.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import card as _card  # модули стенда грузим до смены пути к дереву кода
from .detectors import Templates, classify_turn, service_text
from .llm import MODES, HarnessClient, LiveBudgetExhausted, ResponseCache, SiteIndex
from .prices import PriceCatalog, PricePair
from .prices import _norm as _norm_text
from .run import TURN_TIMEOUT_S, _enter_code_root, _llm_reachable, _parse_cfg, _sha8, git_sha
from .world import PRIVATE_ROOT, PROFILES, FrozenWorld, host_of, settle_shadows, verify_world

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
            await settle_shadows()  # тени, которые ход не ждёт, — после замера, до вызовов и телеметрии
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
        row = {key: dialog.get(key) for key in ("id", "set", "expect", "note", "tags", "about_shown")}
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


# --------------------------------------------------------------------------- метрики диалогов
#
# Классы, которые видны только между ходами (решение 13.10, слепые зоны стенда), — без LLM-судьи: по тексту ответов,
# прайсу мира и разметке набора.
# - Служебный текст в ответе — `detectors.service_text`, тот же детектор, что у карточки одиночных ходов.
# - «Контекст = показанное» (принципы 2 и 3): ход без своей услуги (разметка `about_shown`) не показывает цен и не
#   называет услуг, которых пациент не видел в прошлых ответах диалога, и не печатает слова самого вопроса названием
#   услуги («По услуге «а»», «Памятки по подготовке к «а как подготовиться»…»). Показанное — названия строк прайса в
#   тексте прошлых ответов (с ценой или без: срок, варианты без цен) и их суммы; варианты ответа, которые бот
#   предлагает («Ответьте «да» или «нет»»), и фрагменты, которые он уже брал в кавычки раньше, — его словарь, не эхо.
# - Петли (инвариант 3bacf58 «шаг дважды без продвижения»): тот же вопрос бота два хода подряд или тот же не-ответ
#   (уточнение, «не понял», оффер, «нет данных») слово в слово. Повтор цены на повтор вопроса — не петля.

_NON_ANSWERS = frozenset({"clarify", "not_understood", "operator_offer", "no_info"})
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n")
_QUOTED_RE = re.compile(r"«([^«»]+)»")
_OPTIONS_RE = re.compile(r"«([^«»]+)»\s*или\s*«([^«»]+)»")  # варианты ответа, предложенные ботом: «да» или «нет»
_LIST_ITEM_RE = re.compile(r"^\s*(?:[—–-]|\d+[.)])\s*(.+?)[\s;,.]*$", re.M)
_WORD_RE = re.compile(r"[0-9a-zа-я]+")


def bot_questions(answer: Any) -> set[str]:
    """Вопросы бота пациенту — предложения ответа, которые кончаются «?» (у примеров в кавычках за «?» идёт «»»)."""

    return {_norm_text(part) for part in _SENTENCE_RE.split(str(answer or "")) if part.strip().endswith("?")}


def loop_kind(prev: dict[str, Any], turn: dict[str, Any]) -> str | None:
    """Петля на ходу: тот же вопрос бота, что ходом раньше (`question`), или тот же не-ответ слово в слово (`repeat`)."""

    if bot_questions(turn.get("answer")) & bot_questions(prev.get("answer")):
        return "question"
    text = _norm_text(turn.get("answer"))
    return "repeat" if text and text == _norm_text(prev.get("answer")) and turn.get("outcome") in _NON_ANSWERS else None


def echoed_words(answer: Any, said: Any, bot_quoted: set[str] = frozenset()) -> list[str]:
    """Фрагменты в кавычках, составленные только из слов реплики пациента: слова вопроса напечатаны названием услуги.

    Не эхо — словарь самого бота: варианты ответа, которые он предлагает («Ответьте «да» или «нет»» на «да,
    отмените»), и фрагменты, которые он уже брал в кавычки раньше в диалоге (`bot_quoted`, нормализованные).
    """

    text = str(answer or "")
    own = set(_WORD_RE.findall(_norm_text(said)))
    options = {_norm_text(option) for pair in _OPTIONS_RE.findall(text) for option in pair}
    out = []
    for fragment in _QUOTED_RE.findall(text):
        words = set(_WORD_RE.findall(_norm_text(fragment)))
        if words and words <= own and _norm_text(fragment) not in bot_quoted | options:
            out.append(fragment)
    return out


class CatalogNames:
    """Названия строк прайса мира (нормализованные): какие из них есть в тексте ответа."""

    def __init__(self, catalog: PriceCatalog) -> None:
        self.exact = {_norm_text(name) for name in catalog.names.values()} - {""}

    def found(self, text: Any) -> set[str]:
        norm = _norm_text(text)
        return {name for name in self.exact if name in norm}

    def named(self, text: Any) -> set[str]:
        """Что ответ называет услугой: фрагмент в кавычках или пункт списка — дословно название строки прайса."""

        raw = str(text or "")
        items = _QUOTED_RE.findall(raw) + _LIST_ITEM_RE.findall(raw)
        return {norm for norm in map(_norm_text, items) if norm in self.exact}


@dataclass
class TurnMarks:
    """Отметки хода диалога: что нарушено (пусто — чисто)."""

    service_text: list[str] = field(default_factory=list)
    about_shown: bool = False  # ход без своей услуги (разметка набора): проверен на «контекст = показанное»
    unseen_prices: list[str] = field(default_factory=list)  # «название — сумма» или сумма без строки прайса
    unseen_names: list[str] = field(default_factory=list)
    echo: list[str] = field(default_factory=list)
    loop: str | None = None

    def problems(self) -> list[str]:
        out = []
        if self.service_text:
            out.append("служебный текст: " + ", ".join(self.service_text))
        if self.unseen_prices:
            out.append("цена непоказанного: " + "; ".join(self.unseen_prices[:5]))
        if self.unseen_names:
            out.append("название непоказанного: " + "; ".join(f"«{name}»" for name in self.unseen_names[:5]))
        if self.echo:
            out.append("слова вопроса названием услуги: " + ", ".join(f"«{frag}»" for frag in self.echo))
        if self.loop:
            out.append({"question": "петля: тот же вопрос бота, что ходом раньше", "repeat": "петля: тот же не-ответ, что ходом раньше"}[self.loop])
        return out


def mark_dialog(turns: list[dict[str, Any]], about_shown: Any = (), catalog: PriceCatalog | None = None) -> list[TurnMarks]:
    """Отметки по ходам. Цены — заново по прайсу мира (`catalog`), иначе сохранённые `price_pairs`; без прайса
    «название непоказанного» не проверяется (не отличить название услуги от другого фрагмента в кавычках)."""

    names = CatalogNames(catalog) if catalog is not None else None
    checked = {int(k) for k in about_shown or ()}
    seen_names: set[str] = set()
    seen_amounts: set[int] = set()
    bot_quoted: set[str] = set()
    out: list[TurnMarks] = []
    for k, turn in enumerate(turns, 1):
        answer = str(turn.get("answer") or "")
        pairs = catalog.pairs(answer) if catalog is not None else [PricePair(**p) for p in turn.get("price_pairs") or []]
        marks = TurnMarks(service_text=service_text(answer))
        if k >= 2:
            marks.loop = loop_kind(turns[k - 2], turn)
        if k >= 2 and k in checked:
            marks.about_shown = True
            for pair in pairs:
                if pair.status == "matched" and _norm_text(pair.name) not in seen_names:
                    marks.unseen_prices.append(f"{pair.name} — {pair.amount}")
                elif pair.status != "matched" and pair.amount not in seen_amounts:
                    marks.unseen_prices.append(str(pair.amount))
            if names is not None:
                marks.unseen_names = sorted(names.named(answer) - seen_names)
            marks.echo = echoed_words(answer, turn.get("text"), bot_quoted)
        bot_quoted |= {_norm_text(fragment) for fragment in _QUOTED_RE.findall(answer)}
        seen_amounts |= {pair.amount for pair in pairs}
        seen_names |= {_norm_text(pair.name) for pair in pairs if pair.name}
        if names is not None:
            seen_names |= names.found(answer)
        out.append(marks)
    return out


def refresh_annotations(header: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    """Разметка (`tags`, `about_shown`) — из текущих файлов наборов: правка разметки применяется и к прогонам, снятым
    до неё (как ожидания срочного в `evalbench.card`)."""

    current: dict[tuple[str, str], dict[str, Any]] = {}
    for ref in header.get("sets") or []:
        path = Path(ref).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if path.is_file():
            for dialog in load_dialogs([path]):
                current[(path.stem, dialog["id"])] = dialog
    for row in rows.values():
        dialog = current.get((row.get("set"), row["id"]))
        for key in ("tags", "about_shown"):
            if dialog is not None and key in dialog:
                row[key] = dialog[key]


def world_catalog(header: dict[str, Any]) -> PriceCatalog | None:
    apidata = PRIVATE_ROOT / "worlds" / str((header.get("world") or {}).get("date")) / "apidata"
    return PriceCatalog.from_world(apidata) if apidata.is_dir() else None


def run_marks(rows: dict[str, dict[str, Any]], catalog: PriceCatalog | None) -> dict[str, list[TurnMarks]]:
    return {dialog_id: mark_dialog(row["turns"], row.get("about_shown"), catalog) for dialog_id, row in rows.items()}


_FLAGS = {
    "service_text": lambda m: bool(m.service_text),
    "unseen_price": lambda m: bool(m.unseen_prices),
    "unseen_name": lambda m: bool(m.unseen_names),
    "echo": lambda m: bool(m.echo),
    "loop": lambda m: bool(m.loop),
}


def dialog_metrics(rows: dict[str, dict[str, Any]], marks: dict[str, list[TurnMarks]]) -> dict[str, Any]:
    """Счётчики прогона диалогов — без реплик: ходы, нарушения по видам, исходы продолжений, разбивка по классам."""

    turns = [(row, k, turn, marks[dialog_id][k - 1]) for dialog_id, row in rows.items() for k, turn in enumerate(row["turns"], 1)]
    follow = [(row, k, turn, m) for row, k, turn, m in turns if k >= 2]
    outcomes = Counter(turn["outcome"] for _row, _k, turn, _m in follow)
    llm = Counter(c["outcome"] for _row, _k, turn, _m in turns for c in turn.get("llm_calls") or [])
    ms = [turn["ms"] for _row, _k, turn, _m in turns]
    by_tag: dict[str, Counter] = defaultdict(Counter)
    ids: dict[str, list[str]] = defaultdict(list)
    for dialog_id, row in rows.items():
        dialog_marks = marks[dialog_id]
        flags = {name: sum(1 for m in dialog_marks if test(m)) for name, test in _FLAGS.items()}
        for tag in row.get("tags") or [row.get("set") or "—"]:
            by_tag[tag]["dialogs"] += 1
            by_tag[tag].update(flags)
        for name, count in flags.items():
            if count:
                ids[name].append(dialog_id)
    return {
        "dialogs": len(rows),
        "turns": len(turns),
        "followups": len(follow),
        "service_text": sum(1 for *_x, m in turns if m.service_text),
        "service_text_kinds": dict(Counter(kind for *_x, m in turns for kind in m.service_text)),
        "about_shown": sum(1 for *_x, m in turns if m.about_shown),
        "unseen_price": sum(1 for *_x, m in turns if m.unseen_prices),
        "unseen_name": sum(1 for *_x, m in turns if m.unseen_names),
        "echo": sum(1 for *_x, m in turns if m.echo),
        "loop_question": sum(1 for *_x, m in turns if m.loop == "question"),
        "loop_repeat": sum(1 for *_x, m in turns if m.loop == "repeat"),
        "handoff": sum(1 for _row, _k, turn, _m in follow if turn.get("handoff") and not turn.get("urgent")),
        "operator_offer": outcomes.get("operator_offer", 0),
        "clarify": outcomes.get("clarify", 0) + outcomes.get("not_understood", 0),
        "urgent": sum(1 for _row, _k, turn, _m in turns if turn.get("urgent")),
        "tech_error": sum(1 for _row, _k, turn, _m in turns if turn.get("outcome") == "tech_error" or turn.get("harness_error")),
        "journal": sum(len(turn.get("journal") or []) for _row, _k, turn, _m in turns),
        "p50_ms": _card._pct(ms, 0.5),
        "p95_ms": _card._pct(ms, 0.95),
        "llm_calls": sum(llm.values()),
        "llm_outcomes": dict(llm),
        "outcomes_followup": dict(outcomes),
        "by_tag": {tag: dict(counts) for tag, counts in by_tag.items()},
        "ids": dict(ids),
    }


_CARD_ROWS = (
    ("Диалогов", "dialogs"),
    ("Ходов", "turns"),
    ("  продолжений (ход 2+)", "followups"),
    ("Служебный текст в ответе (ходов)", "service_text"),
    ("Контекст = показанное: ходов без своей услуги", "about_shown"),
    ("  цена непоказанного (ходов)", "unseen_price"),
    ("  название непоказанного (ходов)", "unseen_name"),
    ("  слова вопроса названием услуги (ходов)", "echo"),
    ("Петля: тот же вопрос бота подряд (ходов)", "loop_question"),
    ("Петля: тот же не-ответ подряд (ходов)", "loop_repeat"),
    ("Перевод на оператора (продолжения)", "handoff"),
    ("Оффер оператора (продолжения)", "operator_offer"),
    ("Уточнение / не понял (продолжения)", "clarify"),
    ("Срочное (ходов)", "urgent"),
    ("Сбой хода (ходов)", "tech_error"),
    ("Строк журнала синонимов", "journal"),
    ("p50, мс", "p50_ms"),
    ("p95, мс", "p95_ms"),
    ("Вызовов LLM", "llm_calls"),
)
_ID_LISTS = (
    ("служебный текст", "service_text"),
    ("цена непоказанного", "unseen_price"),
    ("название непоказанного", "unseen_name"),
    ("слова вопроса названием услуги", "echo"),
    ("петли", "loop"),
)


def _run_line(tag: str, header: dict[str, Any]) -> str:
    cfg = ", ".join(f"{k}={v.get('value')}" for k, v in (header.get("cfg") or {}).items())
    world = header.get("world") or {}
    return (
        f"- {'**' + tag + '** ' if tag else ''}`{header['name']}`: код `{header.get('code_sha')}`, LLM `{header.get('mode')}`, "
        f"мир {world.get('date')} {world.get('profile')}, диалогов {header.get('done')}/{header.get('dialogs')}, "
        f"живых вызовов {header.get('live_calls')}, valid={header.get('valid')}"
        + (f", **остановлен: {header['stopped']}**" if header.get("stopped") else "")
        + (f", настройки {cfg}" if cfg else "")
    )


def _summary_line(tag: str, m: dict[str, Any]) -> str:
    return (
        f"- {'**' + tag + '** ' if tag else ''}служебный текст {m['service_text']}; ходов без своей услуги {m['about_shown']}: "
        f"цена непоказанного {m['unseen_price']}, название непоказанного {m['unseen_name']}, слова вопроса {m['echo']}; "
        f"петли {m['loop_question'] + m['loop_repeat']} (вопрос {m['loop_question']}, не-ответ {m['loop_repeat']})"
    )


def _answers_differ(runs: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]) -> tuple[int, int, list[str]]:
    (_ha, a), (_hb, b) = runs
    shared = differ = 0
    dialogs: list[str] = []
    for dialog_id, row in a.items():
        other = b.get(dialog_id)
        if other is None:
            continue
        for left, right in zip(row["turns"], other["turns"]):
            shared += 1
            if str(left.get("answer") or "").strip() != str(right.get("answer") or "").strip():
                differ += 1
                if dialog_id not in dialogs:
                    dialogs.append(dialog_id)
    return differ, shared, dialogs


def render_card(runs: list[tuple[dict[str, Any], dict[str, Any]]], diff: tuple[int, int, list[str]] | None = None) -> str:
    """Карточка диалогов: одна или две колонки (с Δ), исходы продолжений, классы, id диалогов с нарушениями."""

    tags = ["A", "B"][: len(runs)] if len(runs) == 2 else [""]
    lines = ["# Карточка диалогов", ""]
    lines += [_run_line(tag, header) for tag, (header, _m) in zip(tags, runs)]
    for header, _m in runs:
        if not header.get("valid"):
            lines.append(
                f"  - `{header['name']}` недействителен: сеть {header.get('net_attempts')}, очистки {header.get('cleanup_calls')}, "
                f"мир {header.get('world_problems')}, неопознанные промпты {header.get('llm_outcomes', {}).get('unknown_site', 0)}"
            )
    names = [header["name"] for header, _m in runs]
    two = len(runs) == 2
    lines += ["", "| Метрика | " + " | ".join(names) + (" | Δ |" if two else " |"), "|---|" + "---|" * len(runs) + ("---|" if two else "")]
    for title, key in _CARD_ROWS:
        vals = [m[key] for _h, m in runs]
        delta = ""
        if two:
            delta = f" {vals[1] - vals[0]:+d} |" if all(isinstance(v, int) for v in vals) else " |"
        lines.append(f"| {title} | " + " | ".join(_card._fmt(v) for v in vals) + (" |" + delta if delta else " |"))
    if any(m["service_text"] for _h, m in runs):
        lines += ["", "**Служебный текст по видам** (ходов): " + "; ".join(
            f"{h['name']}: " + (", ".join(f"{k} {v}" for k, v in sorted(m["service_text_kinds"].items())) or "—") for h, m in runs
        )]
    if diff is not None:
        lines += ["", f"**Ходов с разным ответом:** {diff[0]} из {diff[1]} общих" + (f" (диалоги: {', '.join(diff[2])})" if diff[2] else "")]
    lines += ["", "**Исходы продолжений (ход 2+):**", "", "| Исход | " + " | ".join(names) + " |", "|---|" + "---|" * len(runs)]
    for key in _card.OUTCOME_ORDER:
        lines.append(f"| {key} | " + " | ".join(str(m["outcomes_followup"].get(key, 0)) for _h, m in runs) + " |")
    all_tags = sorted({tag for _h, m in runs for tag in m["by_tag"]})
    lines += [
        "",
        "**По классам** (ходов с нарушением" + (", A / B" if two else "") + "):",
        "",
        "| Класс | диалогов | служебный текст | цена непоказанного | название непоказанного | слова вопроса | петли |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag in all_tags:
        cells = []
        for key in ("dialogs", *_FLAGS):
            cells.append(" / ".join(str(m["by_tag"].get(tag, {}).get(key, 0)) for _h, m in runs))
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines += ["", "**Диалоги с нарушениями** (id):"]
    for title, key in _ID_LISTS:
        lines.append(f"- {title}: " + "; ".join((f"{tag}: " if tag else "") + (", ".join(m["ids"].get(key, [])) or "—") for tag, (_h, m) in zip(tags, runs)))
    return "\n".join(lines)


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


def render_dialogs(
    runs: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]], *, limit: int = 400, catalogs: list[PriceCatalog | None] | None = None
) -> str:
    """Ответы по ходам рядом; сводка нарушений прогона и отметки на ходах (`catalogs` — прайс мира каждого прогона)."""

    tags = ["A", "B"][: len(runs)] if len(runs) == 2 else [""]
    catalogs = catalogs or [None] * len(runs)
    marks = [run_marks(rows, catalog) for (_header, rows), catalog in zip(runs, catalogs)]
    lines = ["# Диалоги стенда", ""]
    lines += [_run_line(tag, header) for tag, (header, _rows) in zip(tags, runs)]
    lines += ["", "**Сводка** (ходов; разбор — отметки «нарушение» у ходов, счётчики — `evalbench.dialogs card`):"]
    lines += [_summary_line(tag, dialog_metrics(rows, run)) for tag, (_header, rows), run in zip(tags, runs, marks)]
    order: list[str] = []
    for _header, rows in runs:
        order += [key for key in rows if key not in order]
    differ = 0
    for dialog_id in order:
        found = [rows.get(dialog_id) for _header, rows in runs]
        first = next(d for d in found if d)
        lines += ["", f"## {dialog_id}" + (f" — ждём: {first['expect']}" if first.get("expect") else "")]
        if first.get("tags"):
            lines.append("классы: " + ", ".join(first["tags"]) + (f"; без своей услуги — ходы {', '.join(map(str, first['about_shown']))}" if first.get("about_shown") else ""))
        if first.get("note"):
            lines.append(f"_{first['note']}_")
        for k in range(max(len(d["turns"]) if d else 0 for d in found)):
            turn0 = next(d["turns"][k] for d in found if d and k < len(d["turns"]))
            lines += ["", f"**{k + 1}. «{turn0['text']}»**" + (f" (кнопка `{turn0['button']}`)" if turn0.get("button") else "")]
            answers = []
            for tag, dialog, run in zip(tags, found, marks):
                prefix = f"[{tag}] " if tag else ""
                turn = dialog["turns"][k] if dialog and k < len(dialog["turns"]) else None
                if turn is None:
                    lines.append(f"- {prefix}хода нет")
                    continue
                answers.append(str(turn.get("answer") or "").strip())
                lines.append(f"- {prefix}{turn.get('outcome')}, {turn.get('ms')} мс: {_short(turn.get('answer'), limit)}")
                lines += [f"  - **нарушение:** {problem}" for problem in run[dialog_id][k].problems()]
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


def _load_runs(refs: list[str]) -> tuple[list[tuple[dict[str, Any], dict[str, dict[str, Any]]]], list[PriceCatalog | None]]:
    runs = [load_dialog_run(ref) for ref in refs[:2]]
    for header, rows in runs:
        refresh_annotations(header, rows)
    return runs, [world_catalog(header) for header, _rows in runs]


def show_main(args: argparse.Namespace) -> int:
    runs, catalogs = _load_runs(args.runs)
    print(render_dialogs(runs, limit=args.limit, catalogs=catalogs))
    return 0


def card_main(args: argparse.Namespace) -> int:
    runs, catalogs = _load_runs(args.runs)
    loaded = [(header, dialog_metrics(rows, run_marks(rows, catalog))) for (header, rows), catalog in zip(runs, catalogs)]
    print(render_card(loaded, _answers_differ(runs) if len(runs) == 2 else None))
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
    card_ap = sub.add_parser("card", help="карточка диалогов: счётчики без реплик (1 или 2 прогона)")
    card_ap.add_argument("runs", nargs="+", help="имя прогона или путь к каталогу (1 или 2)")
    args = ap.parse_args(argv)
    if args.command == "run":
        return run_main(args, run_ap)
    if args.command == "card":
        return card_main(args)
    return show_main(args)


if __name__ == "__main__":
    sys.exit(main())
