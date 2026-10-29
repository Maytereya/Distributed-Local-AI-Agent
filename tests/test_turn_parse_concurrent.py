"""Под `turn_parse = on` правила понимания считаются, пока LLM выбирает фразы (L-15b, шаг 6, O3b).

Профиль шага 6: ожидание выбора фраз LLM (p95 3,6 с) и CPU правил понимания (S2-сегментация сплошного
текста — до секунд) шли друг за другом. Инварианты: (а) порядок — одновременно разметка LLM и правила → читатели:
правила стартуют до ответа LLM (генератора окон на пути прода нет — разметка `turn_spans`, прод с деплоя 17);
(б) ответ хода и понимание те же, что при правилах до разбора (класс «одна или несколько» под
`deploy_parse` — `tests/test_price_one_or_many_class.py`); (в) телеметрия прежняя: `tu_list`, `tu_rule`,
`tu_parse_agree`; (г) сбой правил не вешает разбор и ход; (д) два хода — второй короткий вопрос не берёт
услугу, которой пациент не видел.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_understanding
from tests.test_price_one_or_many_class import CASES, TRUE_CUT, _configure, _OracleLLM, _violations
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)


class _Probe:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.rules_started = threading.Event()
        self.lock = threading.Lock()

    def note(self, event: str) -> None:
        with self.lock:
            self.events.append(event)


class _WaitingOracle(_OracleLLM):
    """Оракул; на выборе фраз ждёт, пока стартуют правила понимания (до 2 с), — так видно, что они идут
    одновременно с вызовом LLM, а не после него."""

    def __init__(self, case, cut, probe: _Probe) -> None:
        super().__init__(case, cut)
        self.probe = probe
        self.saw_rules: bool | None = None

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        if "понять, какие услуги назвал пациент" in f"{system}\n{prompt}":
            deadline = time.monotonic() + 2.0
            while not self.probe.rules_started.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            self.saw_rules = self.probe.rules_started.is_set()
            self.probe.note("llm_phrases")
        return await super().generate(prompt=prompt, system=system, **kwargs)


def _instrument(monkeypatch, probe: _Probe, *, fail_rules: bool = False) -> None:
    from messengers_router.runtime_config import config as cfg

    # Параллель правил и разбора (O3b) — режим телеметрии согласия; умолчание с 16.10 — ленивые правила (T7 шаг 4).
    monkeypatch.setattr(cfg, "MR_PARSE_AGREE_TELEMETRY", "on", raising=False)
    real_rules = turn_understanding.understand_price_turn

    def rules(text):
        probe.note("rules_start")
        probe.rules_started.set()
        if fail_rules:
            raise RuntimeError("тест: правила упали")
        return real_rules(text)

    monkeypatch.setattr(turn_understanding, "understand_price_turn", rules)


def _priced(text: str) -> set[str]:
    return {" ".join(m.group("name").split()) for m in _PRICED.finditer(text)}


def test_rules_run_while_the_model_picks_phrases(mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES["list_true_three"]
    _configure(monkeypatch, "deploy_parse")
    probe = _Probe()
    _instrument(monkeypatch, probe)
    oracle = _WaitingOracle(case, TRUE_CUT["list_true_three"], probe)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog([case.text, "а сколько?"], "parse-on-concurrent")
        row = world.take_telemetry()
        assert list(world.net.attempts) == []
    assert oracle.saw_rules is True  # правила стартовали до ответа LLM
    violations = _violations(case, first["text"], "oracle", oracle.picked)
    assert not violations, "\n".join([*violations, "--- ответ:", first["text"]])
    assert sorted(_priced(second["text"]) - _priced(first["text"])) == []  # контекст = показанное
    del row  # телеметрия второго хода; первый — в тесте ниже


def test_turn_telemetry_is_unchanged(mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES["list_true_three"]
    _configure(monkeypatch, "deploy_parse")
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PARSE_AGREE_TELEMETRY", "on", raising=False)  # согласие — только в нетерпеливом режиме
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _OracleLLM(case, TRUE_CUT["list_true_three"]))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        run_dialog([case.text], "parse-on-telemetry")
        row = world.take_telemetry()
    assert (row.get("tu_list"), row.get("tu_rule"), row.get("tu_parse_status")) == (3, "parse_picked", "picked")
    assert row.get("tu_facet") == "price"
    assert row.get("tu_parse_agree") in ("same", "split", "merge", "other")


def test_rules_failure_does_not_hang_the_turn(mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES["list_true_three"]
    _configure(monkeypatch, "deploy_parse")
    probe = _Probe()
    _instrument(monkeypatch, probe, fail_rules=True)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _OracleLLM(case, TRUE_CUT["list_true_three"]))
    started = time.monotonic()
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        [turn] = run_dialog([case.text], "parse-on-rules-fail")
        row = world.take_telemetry()
    assert time.monotonic() - started < 30
    assert turn["text"]
    assert row.get("tu_rule") == "error" and row.get("tu_parse_status") == "picked"
