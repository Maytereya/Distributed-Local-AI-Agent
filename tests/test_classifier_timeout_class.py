"""Предохранитель времени LLM-классификатора темы (BUG-2026-10-27-CLASSIFIER-NO-TIMEOUT, `classifier_timeout_s`).

У вызова классификатора не было своего потолка: генерация до 45 с плюс очередь до 30 с, и лимит решений хода (3 / 25 с) его
не касается. При зависшей LLM ход ждал классификатор до 45 с ради ответа «Не совсем понял…» — а перенос тем на LLM-first
(разбор правил 27.10) добавит вызовов классификатора на частых ходах.

Инвариант класса: под `classifier_timeout_s = N` вызов классификатора, не ответивший за N секунд (очередь и генерация вместе),
обрывается, и ход совпадает с ходом, где классификатор сразу упал по тайм-ауту: тему решает правило, если сработало (ветка
в `topic_by_llm` — запасной путь правил), иначе путь «LLM недоступна»; срочное — срочно. В журнале хода — `nlu_timeout_s = N` и
вызов `classifier` с исходом тайм-аута. Ответ, успевший в потолок, — тот же ход, что без предохранителя; `0` — как было.
Через обработчик эндпоинта, LLM — медленная подделка настоящего типа (`llm_runtime._OLLAMA_CLIENT`).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests._llm_helpers import _CLASSIFIER_MARK
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

# `topic_by_llm` прода (деплой 18).
_PROD_TOPIC_BY_LLM = (
    "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,"
    "rule_website_help,rule_address_work_hours,rule_test_result,rule_prepare"
)
# Реплика → (ветка правила; сколько раз зовётся классификатор). У `rule_schedule` два вызова: уточнение темы (refine) в
# ветке правила и разбор темы — ветка в `topic_by_llm`; потолок — на каждый вызов.
_REPLIES = {
    "no_rule": ("подскажите по одному вопросу", "", 1),
    "topic_by_llm": ("расписание кардиолога", "rule_schedule", 2),
    "rule_first": ("сколько стоит ферритин", "rule_price", 0),
}
# Оба промпта классификатора: разбор темы и уточнение (refine) — один вызов `ollama_classify_payload`, один потолок.
_CLASSIFIER_MARKS = (_CLASSIFIER_MARK, "Ты уточняющий классификатор")
_URGENT = "у меня сильная боль в груди и мне трудно дышать"
_ANSWER = json.dumps({"label": "DOCTOR_SCHEDULE", "confidence": 0.9, "context_action": "new_topic", "entities": {}})
_GUARD_S = 0.3  # потолок теста; в настройке — целые секунды
_SLOW_S = 1.5  # подделка отвечает позже потолка, но раньше 45 с


class _SlowLLM:
    """Классификатору отвечает через `delay` секунд (None — сразу тайм-аут, как упавшая LLM); прочим местам — отказ
    соединения: решения хода идут своим путём сбоя, одинаковым в обоих сравниваемых ходах."""

    def __init__(self, delay: float | None):
        self.delay = delay

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if any(mark in seen for mark in _CLASSIFIER_MARKS):
            if self.delay is None:
                raise asyncio.TimeoutError()
            await asyncio.sleep(self.delay)
            return GenerateResponse(model="m", response=_ANSWER, done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _turn(world_dir, monkeypatch, text: str, *, guard: float, delay: float | None, slots: int | None = None) -> dict[str, Any]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    async def run() -> dict[str, Any]:
        holders = []
        if slots is not None:  # очередь к LLM занята соседями: все слоты семафора взяты до хода
            for _ in range(slots):
                await llm_runtime._SEMAPHORE.acquire()
                holders.append(1)
        req = MessengerGenerateRequest(session_id=f"cls-timeout-{guard}-{delay}-{slots}-{text}", text=text, debug=True)
        started = time.monotonic()
        body = json.loads((await messenger_generate_once(req, memory=MemoryStore(), services=Services())).body)
        for _ in holders:
            llm_runtime._SEMAPHORE.release()
        dbg = (body.get("state_update") or {}).get("debug") or {}
        return {"text": str(body.get("text") or ""), "decision": dbg.get("decision") or {}, "s": time.monotonic() - started}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_TOPIC_BY_LLM", _PROD_TOPIC_BY_LLM, raising=False)
        patch.setattr(cfg, "MR_CLASSIFIER_TIMEOUT_S", guard, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_SEMAPHORE", asyncio.Semaphore(slots or 2))
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _SlowLLM(delay))
        choice._CACHE.clear()
        out = asyncio.run(run())
        out["telemetry"] = world.take_telemetry() or {}
    out["label"] = out["decision"].get("label") or out["telemetry"].get("label")
    out["classifier"] = [e["outcome"] for e in out["telemetry"].get("llm") or [] if e.get("site") == "classifier"]
    return out


def _same(a: dict[str, Any], b: dict[str, Any]) -> tuple:
    keys = ("tu_label_nlu", "tu_label_final", "answer_source", "handoff", "nlu_src", "nlu_topic_by_llm")
    return (a["text"], a["label"], *(a["telemetry"].get(k) for k in keys)), (b["text"], b["label"], *(b["telemetry"].get(k) for k in keys))


@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_slow_classifier_is_cut_and_turn_takes_the_llm_failure_path(reply, mini_prod_world, monkeypatch):  # noqa: F811
    text, branch, calls = _REPLIES[reply]

    cut = _turn(mini_prod_world, monkeypatch, text, guard=_GUARD_S, delay=_SLOW_S)
    failed = _turn(mini_prod_world, monkeypatch, text, guard=0, delay=None)

    assert branch in str(cut["telemetry"].get("nlu_rule") or "").split("+") if branch else not cut["telemetry"].get("nlu_rule")
    got, want = _same(cut, failed)
    assert got == want, reply
    if calls:
        assert cut["classifier"] == ["timeout"] * calls, cut["telemetry"].get("llm")
        assert cut["telemetry"].get("nlu_timeout_s") == _GUARD_S, cut["telemetry"]
        assert cut["s"] < _SLOW_S, cut["s"]  # ход не ждал ответа модели ни на одном вызове
    else:  # правило решило тему без LLM — предохранителю нечего обрывать
        assert cut["classifier"] == [] and "nlu_timeout_s" not in cut["telemetry"], cut["telemetry"]
    if reply == "topic_by_llm":
        assert cut["telemetry"].get("nlu_topic_by_llm") == "rules_fallback", cut["telemetry"]


def test_without_guard_the_turn_waits_for_the_slow_classifier(mini_prod_world, monkeypatch):  # noqa: F811
    """Выключатель off — как было: ход ждёт медленный классификатор и берёт его тему."""

    text = _REPLIES["no_rule"][0]
    off = _turn(mini_prod_world, monkeypatch, text, guard=0, delay=_SLOW_S)

    assert off["classifier"] == ["ok"] and "nlu_timeout_s" not in off["telemetry"], off["telemetry"]
    assert off["s"] >= _SLOW_S and off["telemetry"].get("tu_label_nlu") == "DOCTOR_SCHEDULE", off["telemetry"]


def test_answer_within_the_guard_is_the_same_turn_as_without_it(mini_prod_world, monkeypatch):  # noqa: F811
    text = _REPLIES["no_rule"][0]
    on = _turn(mini_prod_world, monkeypatch, text, guard=_SLOW_S * 3, delay=0.05)
    off = _turn(mini_prod_world, monkeypatch, text, guard=0, delay=0.05)

    got, want = _same(on, off)
    assert got == want
    assert on["classifier"] == ["ok"] and "nlu_timeout_s" not in on["telemetry"], on["telemetry"]


def test_queue_wait_counts_toward_the_guard(mini_prod_world, monkeypatch):  # noqa: F811
    """Слоты LLM заняты соседями: потолок — на вызов целиком, а не только на генерацию (иначе +30 с очереди)."""

    text = _REPLIES["no_rule"][0]
    queued = _turn(mini_prod_world, monkeypatch, text, guard=1, delay=0.05, slots=1)
    failed = _turn(mini_prod_world, monkeypatch, text, guard=0, delay=None)

    assert queued["classifier"] == ["queue_timeout"], queued["telemetry"].get("llm")
    assert queued["telemetry"].get("nlu_timeout_s") == 1, queued["telemetry"]
    assert queued["s"] < 2.5, queued["s"]
    got, want = _same(queued, failed)
    assert got == want


def test_instant_llm_timeout_under_the_guard_is_not_counted_as_cut(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: клиент LLM сам бросил тайм-аут сразу — потолок ничего не обрывал, `nlu_timeout_s` в журнале нет (иначе
    стенд `down_timeout` без зависания показал бы «сработал предохранитель» на каждом ходе)."""

    text = _REPLIES["no_rule"][0]
    instant = _turn(mini_prod_world, monkeypatch, text, guard=_SLOW_S, delay=None)
    failed = _turn(mini_prod_world, monkeypatch, text, guard=0, delay=None)

    assert instant["classifier"] == ["timeout"] and "nlu_timeout_s" not in instant["telemetry"], instant["telemetry"]
    got, want = _same(instant, failed)
    assert got == want


def test_urgent_stays_urgent_under_the_guard(mini_prod_world, monkeypatch):  # noqa: F811
    turn = _turn(mini_prod_world, monkeypatch, _URGENT, guard=_GUARD_S, delay=_SLOW_S)

    assert turn["label"] == "URGENT" and turn["telemetry"].get("handoff"), turn


def test_switch_defaults_to_off():
    from agent_logic_2 import config as host_config

    assert host_config.settings.messenger_router.classifier_timeout_s == 0
