"""Телеметрия хода: одна строка на ход — только числа и ключи.

Ход (а) рефакторинга, решения владельца 08.10 (повестка №3): на проде видно, откуда взялся
ответ, кто решил намерение (правило или LLM), сколько стоили вызовы LLM и на каких данных
бот работал. Без этого ни один следующий ход нельзя честно принять или отклонить.

Что пишется (`v=1`):
- `answer_source` — ветка, давшая текст: `precheck:<причина>`, `safety:<метка>`, `clarify`,
  `intro`, `recovery`, `pending`, `prebuilt`, `empty_evidence`, `llm_free`,
  `render_timeout`, `render_error` (LLM упала в свободном ответе, поле `error` — тип),
  `route_error`; поверх — `operator_offer_on_repeat`;
- `label`, `decision_source` (как подписал NLU), `nlu_llm` — звалась ли LLM-классификация
  на этом ходе (подпись `llm_primary` у решения правила врёт — DEBT-1, поэтому факт вызова
  пишется отдельно), `rule` — первый флаг `rule_*`;
- `evidence_empty`, `handoff`, `chars`, `ms`, `stage_ms`;
- `money_cut` — сколько предложений с денежными суммами снято со свободного текста LLM
  (`money_guard`, только при `llm_free` и только если снято);
- `llm` — вызовы: место (модуль), исход, мс, ожидание очереди, токены; у сбоя — `err`, корневая
  причина (errno сокета или класс корня, `llm_runtime.error_detail`): строка `outcome` прежняя, по
  ней считают карточка стенда и `HarnessClient`;
- `choose` — решения модуля выбора (ход (в)): задача, статус, сколько вариантов, мс, из кэша ли,
  код причины или исход — без текста реплики и вариантов;
- `shadow: true` у записей `llm` и `choose` — решение тени (L-15b, шаг 4): считалось параллельно ходу
  вне его бюджета и на ответ не влияло; карточки и сводки считают решения хода без них;
- `world` — модель и срезы (имена файлов) — раз в 5 минут, иначе тот же `world_id`.

Чего НЕ пишется: текст пациента и ответа, значения сущностей, сырой session_id (только
sha8). Пишется в логгер `mr.turn` и в `<APP_DATA_DIR>/journals/turns_<ГГГГММДД>.jsonl` (volume
на проде). Сбой записи не ломает ответ. Выключатель: `[MESSENGER_ROUTER] turn_telemetry`.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .runtime_config import config as _cfg

logger = logging.getLogger("mr.turn")

_TURN: contextvars.ContextVar["Turn | None"] = contextvars.ContextVar("mr_turn", default=None)
_WORLD_TTL_S = 300
_world_cache: dict[str, Any] = {"at": 0.0, "value": None}
_world_provider: Callable[[], dict[str, Any]] | None = None


def enabled() -> bool:
    return bool(getattr(_cfg, "MR_TURN_TELEMETRY", True))


@dataclass
class Turn:
    endpoint: str
    llm_mode: str
    button: str
    session: str
    started: float = field(default_factory=time.monotonic)
    fields: dict[str, Any] = field(default_factory=dict)
    llm: list[dict[str, Any]] = field(default_factory=list)
    choices: list[dict[str, Any]] = field(default_factory=list)


def begin(endpoint: str, *, session_id: str, llm_mode: str = "", button_id: str = "") -> contextvars.Token | None:
    """Открыть ход. Вызывать в эндпоинте до роутера; вернуть токен в `finish`."""

    if not enabled():
        return None
    sess = hashlib.sha256(str(session_id or "").encode("utf-8")).hexdigest()[:8]
    return _TURN.set(Turn(endpoint=endpoint, llm_mode=str(llm_mode or ""), button=str(button_id or ""), session=sess))


def note(**values: Any) -> None:
    """Записать поля хода. Первое значение `answer_source` не затирается последующими."""

    turn = _TURN.get()
    if turn is None:
        return
    for key, value in values.items():
        if key == "answer_source" and "answer_source" in turn.fields:
            continue
        turn.fields[key] = value


def override(**values: Any) -> None:
    """Записать поля, затирая прежние (оффер после повтора меняет источник ответа)."""

    turn = _TURN.get()
    if turn is not None:
        turn.fields.update(values)


def note_llm(
    site: str,
    outcome: str,
    ms: int,
    *,
    queue_ms: int = 0,
    usage: dict[str, int] | None = None,
    err: str | None = None,
    shadow: bool = False,
) -> None:
    """Вызов LLM в ходе; `err` — корневая причина сбоя (`EHOSTDOWN`…), отдельно от `outcome`;
    `shadow` — вызов тени (на ответ хода не влиял)."""

    turn = _TURN.get()
    if turn is None:
        return
    entry: dict[str, Any] = {"site": site, "outcome": outcome, "ms": ms}
    if queue_ms:
        entry["queue_ms"] = queue_ms
    if err:
        entry["err"] = err
    if usage:
        if "prompt_eval_count" in usage:
            entry["in_tok"] = usage["prompt_eval_count"]
        if "eval_count" in usage:
            entry["out_tok"] = usage["eval_count"]
    if shadow:
        entry["shadow"] = True
    turn.llm.append(entry)


def note_choice(
    task: str,
    status: str,
    *,
    options: int,
    ms: int,
    cached: bool = False,
    reason: str = "",
    outcome: str = "",
    shadow: bool = False,
) -> None:
    """Решение модуля выбора (`choose()`): задача, статус, число вариантов, мс — без текста; `shadow` —
    решение тени."""

    turn = _TURN.get()
    if turn is None:
        return
    entry: dict[str, Any] = {"task": task, "status": status, "options": options, "ms": ms}
    if cached:
        entry["cached"] = True
    if reason:
        entry["reason"] = reason
    if outcome:
        entry["outcome"] = outcome
    if shadow:
        entry["shadow"] = True
    turn.choices.append(entry)


def set_world_provider(provider: Callable[[], dict[str, Any]]) -> None:
    """Источник отпечатка мира (срезы, модель) — задаёт сервисный слой, чтобы ядро не
    импортировало клиента МИС напрямую (храповик переносимости)."""

    global _world_provider
    _world_provider = provider


def _world() -> tuple[str, dict[str, Any] | None]:
    now = time.monotonic()
    cached = _world_cache["value"]
    if cached is not None and now - float(_world_cache["at"]) < _WORLD_TTL_S:
        return cached[0], None
    try:
        world = dict(_world_provider()) if _world_provider is not None else {}
    except Exception as exc:  # отпечаток — справка, не повод ломать ход
        world = {"error": type(exc).__name__}
    world_id = hashlib.sha256(json.dumps(world, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    changed = cached is None or cached[0] != world_id
    _world_cache.update(at=now, value=(world_id, world))
    return world_id, (world if changed else None)


def finish(token: contextvars.Token | None, *, handoff: bool, chars: int, stage_ms: dict[str, float] | None = None) -> dict[str, Any] | None:
    """Закрыть ход и записать строку. Возвращает строку (для тестов и debug)."""

    if token is None:
        return None
    turn = _TURN.get()
    _TURN.reset(token)
    if turn is None:
        return None
    world_id, world = _world()
    row: dict[str, Any] = {
        "v": 1,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": turn.endpoint,
        "sess": turn.session,
        "llm_mode": turn.llm_mode,
        "button": turn.button or None,
        **turn.fields,
        "nlu_llm": any(call["site"] == "classifier" for call in turn.llm),
        "handoff": bool(handoff),
        "chars": int(chars),
        "ms": int((time.monotonic() - turn.started) * 1000),
        "llm": turn.llm,
        "world_id": world_id,
    }
    if turn.choices:
        row["choose"] = turn.choices
    if stage_ms:
        row["stage_ms"] = {k: int(v) for k, v in stage_ms.items()}
    if world is not None:
        row["world"] = world
    _emit(row)
    return row


def _journal_path() -> Path:
    from agent_logic_2.persist import DATA_DIR, ensure_dir

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return ensure_dir(DATA_DIR / "journals", "journals") / f"turns_{stamp}.jsonl"


def _emit(row: dict[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False, default=str)
    logger.info(line)
    try:
        with _journal_path().open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:  # телеметрия не ломает ответ пациенту
        logger.warning("turn_telemetry_write_failed: %s", exc)
