"""Дедлайн хода для LLM (BUG-2026-10-27-LLM-QUEUE-GIVES-UP, `llm_turn_deadline_s`; решение владельца 27.10: «лучше подождать LLM,
чем переключаться на одни правила»).

Слотов LLM два (= `OLLAMA_NUM_PARALLEL`), а задачи выбора и проверки ждали слот 1,5–3 с: пока классификатор соседнего хода
держал слот 11–14 с, они сдавались, и ход молча уходил на правила — при 3 одновременных ходах 9 % ходов, при 6 — 40 %
(модель очереди 27.10). Классификатор при этом ждал 30 с.

Инвариант класса: под дедлайном хода вызов LLM ждёт слот до «остаток дедлайна − p95 задачи», а не свои 1,5–3 с, — ход под
нагрузкой, уложившийся в дедлайн, совпадает с ходом при свободной очереди (тот же текст, тема, исходы решений; ни одного
сдавшегося вызова). Не хватило дедлайна — ответ хода честный оффер оператора, а не ответ правил с фактом; в журнале хода —
исход `deadline`, `llm_deadline_hit`, `llm_queue_gaveup`. Срочное — срочно. `0` — как было: задачи сдаются по своим тайм-аутам.
Через обработчик эндпоинта, LLM — подделка настоящего типа (`llm_runtime._OLLAMA_CLIENT`), соседи держат слоты пула.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_CLASSIFIER_MARKS = (_CLASSIFIER_MARK, "Ты уточняющий классификатор")
_URGENT = "у меня сильная боль в груди и мне трудно дышать"
_DIGITS = re.compile(r"\d")
# Реплика → места вызова на пути хода. Цена — разметка реплики и выбор строки (очередь 1,5–4 с: под нагрузкой без дедлайна
# сдаются); без правила — только классификатор (очередь 30 с).
_REPLIES = {
    "price": "цена анализа на гормоны щитовидки",
    "no_rule": "подскажите по одному вопросу",
}
_BUSY_S = 2.0  # соседи держат оба слота дольше очереди задач выбора (1,5–3 с), но меньше дедлайна теста
_DEADLINE_S = 10.0


def _minimal(schema: Any) -> Any:
    """Самый короткий ответ по JSON Schema места вызова: пустой выбор, первый из перечня исходов («none»)."""

    if not isinstance(schema, dict):
        return {}
    kind = schema.get("type")
    if kind == "object":
        return {key: _minimal(sub) for key, sub in (schema.get("properties") or {}).items()}
    if kind == "array":
        return []
    if "enum" in schema:
        return schema["enum"][0]
    return {"integer": 0, "number": 0, "boolean": False}.get(str(kind), "")


class _EchoLLM:
    """Каждому месту — внятный и всегда одинаковый ответ через `delay` секунд: классификатору — тема, решению выбора — пустой
    выбор по его схеме, тексту — короткая фраза."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    async def generate(self, *, prompt: str = "", system: str | None = None, format: Any = None, **_kwargs):
        await asyncio.sleep(self.delay)
        seen = f"{system}\n\n{prompt}" if system else prompt
        if any(mark in seen for mark in _CLASSIFIER_MARKS):
            response = json.dumps({"label": "OTHER", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
        elif isinstance(format, dict):
            response = json.dumps(_minimal(format))
        elif format == "json":
            response = "{}"
        else:
            response = "Уточните, пожалуйста, вопрос."
        return GenerateResponse(model="m", response=response, done=True)


def _turn(
    world_dir,
    monkeypatch,
    text: str,
    *,
    deadline: float,
    busy_s: float = 0.0,
    delay: float = 0.0,
    reserve: float = 0.3,
    memory: Any = None,
    session: str = "",
) -> dict[str, Any]:
    """Ход через эндпоинт; `busy_s` — соседи держат оба слота столько секунд от начала хода; `reserve` — p95 задач теста."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    async def run() -> dict[str, Any]:
        pool = llm_runtime._SEMAPHORE
        if busy_s:
            for _ in range(2):
                await pool.acquire()
            loop = asyncio.get_running_loop()
            for _ in range(2):
                loop.call_later(busy_s, pool.release)
        sid = session or f"deadline-{deadline}-{busy_s}-{delay}-{text}"
        req = MessengerGenerateRequest(session_id=sid, text=text, debug=True)
        started = time.monotonic()
        body = json.loads((await messenger_generate_once(req, memory=memory or MemoryStore(), services=Services())).body)
        took = time.monotonic() - started
        if busy_s:
            await asyncio.sleep(max(0.0, busy_s - took) + 0.05)  # соседи вернули слоты — пул цел для следующего хода
        dbg = (body.get("state_update") or {}).get("debug") or {}
        return {"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff")), "decision": dbg.get("decision") or {}, "s": took}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_LLM_TURN_DEADLINE_S", deadline, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_SEMAPHORE", llm_runtime._SlotPool(2))
        patch.setattr(llm_runtime, "_P95_S", {})
        patch.setattr(llm_runtime, "_P95_DEFAULT_S", reserve)
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _EchoLLM(delay))
        choice._CACHE.clear()
        out = asyncio.run(run())
        out["telemetry"] = world.take_telemetry() or {}
    out["label"] = out["decision"].get("label") or out["telemetry"].get("label")
    calls = [e for e in out["telemetry"].get("llm") or [] if not e.get("shadow")]
    out["outcomes"] = [(e.get("site"), e.get("outcome")) for e in calls]
    return out


def _same(a: dict[str, Any], b: dict[str, Any]) -> tuple:
    keys = ("tu_label_nlu", "tu_label_final", "answer_source", "handoff", "nlu_src")

    def view(t: dict[str, Any]) -> tuple:
        return (t["text"], t["label"], [o for o in t["outcomes"]], *(t["telemetry"].get(k) for k in keys))

    return view(a), view(b)


def _gave_up(turn: dict[str, Any]) -> list[tuple[str, str]]:
    return [o for o in turn["outcomes"] if o[1] in ("queue_timeout", "deadline")]


@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_under_load_the_turn_waits_for_llm_and_matches_the_free_queue(reply, mini_prod_world, monkeypatch):  # noqa: F811
    text = _REPLIES[reply]
    loaded = _turn(mini_prod_world, monkeypatch, text, deadline=_DEADLINE_S, busy_s=_BUSY_S)
    free = _turn(mini_prod_world, monkeypatch, text, deadline=_DEADLINE_S)

    got, want = _same(loaded, free)
    assert got == want, reply
    assert not _gave_up(loaded) and loaded["outcomes"], loaded["outcomes"]
    assert loaded["s"] >= _BUSY_S - 0.1, loaded["s"]  # ждал слот, а не ушёл на правила
    first = (loaded["telemetry"].get("llm") or [{}])[0]
    assert first.get("queue_ms", 0) >= (_BUSY_S - 0.2) * 1000, first
    assert loaded["telemetry"].get("llm_deadline_s") == _DEADLINE_S
    assert "llm_deadline_hit" not in loaded["telemetry"] and "llm_queue_gaveup" not in loaded["telemetry"], loaded["telemetry"]


def test_without_deadline_a_short_queue_gives_up_and_the_turn_goes_to_rules(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: выключатель off — как было, и это дефект класса: решение выбора сдаётся через 1,5 с очереди."""

    text = _REPLIES["price"]
    loaded = _turn(mini_prod_world, monkeypatch, text, deadline=0, busy_s=_BUSY_S)
    free = _turn(mini_prod_world, monkeypatch, text, deadline=0)

    assert ("choose_turn_phrases", "queue_timeout") in loaded["outcomes"], loaded["outcomes"]
    assert loaded["telemetry"].get("llm_queue_gaveup", 0) >= 1, loaded["telemetry"]
    assert "llm_deadline_s" not in loaded["telemetry"]
    assert _same(loaded, free)[0] != _same(loaded, free)[1]


@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_expired_deadline_answers_with_an_honest_operator_offer(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Слоты заняты дольше дедлайна: ход не ждёт соседей сверх дедлайна и не отвечает правилами — честный оффер."""

    from messengers_router.router import DEADLINE_OFFER_TEXT

    turn = _turn(mini_prod_world, monkeypatch, _REPLIES[reply], deadline=1.0, busy_s=3.0)

    assert turn["text"] == DEADLINE_OFFER_TEXT and not _DIGITS.search(turn["text"]), turn["text"]
    assert not turn["handoff"]
    assert any(outcome == "deadline" for _site, outcome in turn["outcomes"]), turn["outcomes"]
    assert turn["telemetry"].get("answer_source") == "llm_deadline_offer", turn["telemetry"]
    assert turn["telemetry"].get("llm_deadline_hit", 0) >= 1 and turn["telemetry"].get("llm_queue_gaveup", 0) >= 1
    assert turn["s"] < 2.0, turn["s"]


def test_generation_is_cut_at_the_deadline(mini_prod_world, monkeypatch):  # noqa: F811
    """Очередь свободна, модель медленная: генерация не дольше остатка дедлайна, ответ — тот же честный оффер."""

    from messengers_router.router import DEADLINE_OFFER_TEXT

    turn = _turn(mini_prod_world, monkeypatch, _REPLIES["no_rule"], deadline=1.0, delay=3.0)

    assert turn["outcomes"] == [("classifier", "timeout")], turn["outcomes"]
    assert turn["text"] == DEADLINE_OFFER_TEXT and turn["telemetry"].get("llm_deadline_hit") == 1, turn["telemetry"]
    assert "llm_queue_gaveup" not in turn["telemetry"]
    assert turn["s"] < 2.0, turn["s"]


def test_yes_to_the_deadline_offer_hands_off(mini_prod_world, monkeypatch):  # noqa: F811
    """Оффер — настоящий вопрос: «да» следующим ходом переводит на оператора."""

    from messengers_router.memory import MemoryStore

    memory = MemoryStore()
    first = _turn(mini_prod_world, monkeypatch, _REPLIES["price"], deadline=1.0, busy_s=3.0, memory=memory, session="dl-yes")
    second = _turn(mini_prod_world, monkeypatch, "да", deadline=_DEADLINE_S, memory=memory, session="dl-yes")

    assert first["telemetry"].get("answer_source") == "llm_deadline_offer"
    assert second["handoff"], second


def test_urgent_stays_urgent_when_the_deadline_expires(mini_prod_world, monkeypatch):  # noqa: F811
    turn = _turn(mini_prod_world, monkeypatch, _URGENT, deadline=1.0, busy_s=3.0)

    assert turn["label"] == "URGENT" and turn["handoff"], turn


def test_switch_defaults_to_off():
    from agent_logic_2 import config as host_config

    assert host_config.settings.messenger_router.llm_turn_deadline_s == 0
