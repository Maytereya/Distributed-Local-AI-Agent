"""Тень разбора реплики на фразы (L-15b, шаг 4) ≡ off — через обработчик эндпоинта, мини-прод.

Выключатель `[MESSENGER_ROUTER] turn_parse = off | shadow | on` (по умолчанию off). Решение владельца
13.10 (Q1): тень — во время хода, параллельно ему, вне бюджета хода и предохранителя, очередь к LLM ≤1 с;
записи тени в `llm[]` / `choose[]` помечены `shadow: true`; конец хода — `drain(1.0)`. Q4: лимит хода —
2 решения, под `on` выбор фраз — одно из них.

Инварианты:
- тень ≡ off по ответам: эталон пути цены (44 диалога × режимы понимания хода) при недоступной LLM; 16
  случаев класса «одна или несколько» в наборе деплоя с LLM-оракулом, который отвечает и на выбор фраз
  (ровно разрез случая), — ответы, исходы и вызовы прочих мест LLM те же, что под off;
- тень пишет `tu_parse_*` в телеметрию хода без текста пациента; фразы — только в `debug.turn_parse`;
- под `on` решений хода не больше двух (выбор фраз — одно из них). Что под `on` меняется в ответах
  (читатели разбора, шаг 5), — `test_turn_parse_readers.py`.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_one_or_many_class import _DEPLOY, CASES, _OracleLLM
from tests.test_price_turns_golden import (  # noqa: F401 — mini_prod_world, synonym_fork — фикстуры
    DIALOGS,
    GOLDEN_ON,
    MODES,
    _golden,
    _load,
    mini_prod_world,
    run_dialog,
    synonym_fork,
)
from tests.test_turn_phrases import TRUE_CUT

_PHRASES_MARK = "понять, какие услуги назвал пациент"
_PARSE_FIELDS = ("tu_parse_status", "tu_parse_src", "tu_parse_n", "tu_parse_cands", "tu_parse_ms")


def _turn_parse(monkeypatch, value: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", value, raising=False)


def _deploy(monkeypatch) -> None:
    """Набор деплоя; кэш выбора выключен (как на стенде): off и shadow в одном тесте не делят решений."""

    from messengers_router.runtime_config import config as cfg

    for key, value in _DEPLOY.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_CACHE_TTL_S", 0, raising=False)


class _PhrasesOracle(_OracleLLM):
    """Оракул класса, который отвечает и на выбор фраз: отмечает ровно фразы разреза случая."""

    def __init__(self, name: str) -> None:
        super().__init__(CASES[name])
        self.cut = {_normalise_catalog_text(text) for text in TRUE_CUT[name]}
        self.phrase_calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        if _PHRASES_MARK not in (system or ""):
            return await super().generate(prompt=prompt, system=system, **kwargs)
        self.phrase_calls += 1
        lines = (line.partition(". ") for line in prompt.splitlines())
        picks = [int(head) for head, dot, rest in lines if dot and head.isdigit() and _normalise_catalog_text(rest.strip("«»")) in self.cut]
        return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)


def _acting(row: dict) -> list[tuple[str, str]]:
    return [(entry["task"], entry["status"]) for entry in row.get("choose") or [] if not entry.get("shadow")]


@pytest.mark.parametrize("understanding", MODES)
@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_shadow_equals_off_on_the_golden(case, understanding, mini_prod_world, request, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", understanding, raising=False)
    _turn_parse(monkeypatch, "shadow")
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        record = run_dialog(DIALOGS[case], f"shadow-{case}-{understanding}")
        rows, attempts = list(world.telemetry), list(world.net.attempts)
    base = _golden()[case]
    assert attempts == []
    assert record == (_load(GOLDEN_ON).get(case, base) if understanding == "on" else base)
    for row in rows:  # тень не тратит решений хода и не пишет ничего, кроме своих полей
        assert all(entry.get("shadow") for entry in row.get("choose") or [] if entry["task"] == "turn_phrases")


def test_shadow_runs_on_price_turns_of_the_golden(mini_prod_world, monkeypatch):  # noqa: F811
    """Встречная проверка: тень не молчит — на ценовых ходах эталона разбор есть, и конец хода его дождался."""

    _turn_parse(monkeypatch, "shadow")
    statuses = []
    for case in ("list_text_three", "one_ferritin_bare", "list_text_no_commas", "name_with_commas"):
        with ev_world.FrozenWorld(mini_prod_world, "day") as world:
            run_dialog(DIALOGS[case], f"shadow-runs-{case}")
            statuses += [row.get("tu_parse_status") for row in world.telemetry]
    assert None not in statuses and "late" not in statuses, statuses
    assert {"strong_whole", "unavailable"} <= set(statuses), statuses


def _class_turn(name: str, mode: str, world_dir, monkeypatch):
    _turn_parse(monkeypatch, mode)
    oracle = _PhrasesOracle(name)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        [turn] = run_dialog([CASES[name].text], f"shadow-live-{name}-{mode}")
        row = world.take_telemetry()
    return turn, row, oracle


@pytest.mark.parametrize("name", sorted(CASES))
def test_shadow_equals_off_with_a_live_llm(name, mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    off, off_row, off_oracle = _class_turn(name, "off", mini_prod_world, monkeypatch)
    shadow, shadow_row, shadow_oracle = _class_turn(name, "shadow", mini_prod_world, monkeypatch)
    assert shadow == off
    assert shadow_oracle.calls == off_oracle.calls and shadow_oracle.picked == off_oracle.picked  # прочие места
    assert _acting(shadow_row) == _acting(off_row)
    assert off_oracle.phrase_calls == 0 and "tu_parse_status" not in off_row
    # Тень разобрала реплику разрезом случая — по данным или выбором оракула.
    assert shadow_row["tu_parse_status"] in ("strong_whole", "strong_all", "single", "picked"), shadow_row
    assert shadow_row["tu_parse_n"] == len(TRUE_CUT[name])


def test_shadow_writes_parse_fields_without_patient_text(mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    turn, row, oracle = _class_turn("list_caps_run_on_priced", "shadow", mini_prod_world, monkeypatch)
    assert oracle.phrase_calls == 1
    assert {key: row[key] for key in _PARSE_FIELDS if key != "tu_parse_ms"} == {
        "tu_parse_status": "picked",
        "tu_parse_src": "llm",
        "tu_parse_n": 3,
        "tu_parse_cands": 4,
    }
    assert isinstance(row["tu_parse_ms"], int)
    assert row["tu_parse_agree"] == "split"  # правила видят одну услугу, LLM — три
    [entry] = [entry for entry in row["choose"] if entry["task"] == "turn_phrases"]
    assert (entry["status"], entry["options"], entry.get("shadow")) == ("picked", 4, True)
    assert all(call.get("shadow") for call in row["llm"] if call["site"] == "choose_turn_phrases")
    dumped = json.dumps(row, ensure_ascii=False).lower()
    assert not any(word in dumped for word in ("ферритин", "глюкоза", "холестерол"))


def test_debug_shows_the_phrases_only_in_debug(mini_prod_world, monkeypatch):  # noqa: F811
    import asyncio

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    _deploy(monkeypatch)
    _turn_parse(monkeypatch, "on")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PhrasesOracle("list_caps_run_on_priced"))

    async def run() -> dict:
        req = MessengerGenerateRequest(session_id="debug-turn-parse", text=CASES["list_caps_run_on_priced"].text, debug=True)
        resp = await messenger_generate_once(req, memory=MemoryStore(), services=Services())
        return json.loads(resp.body)

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        body = asyncio.run(run())
    parse = body["state_update"]["debug"]["turn_parse"]
    assert (parse["status"], parse["phrases"], parse["strong"]) == ("picked", ["ФЕРРИТИН", "ГЛЮКОЗА", "ХОЛЕСТЕРОЛ"], [True, True, True])


@pytest.mark.parametrize("name", ["list_true_three", "list_caps_run_on_priced", "phrase_smear_pharynx", "readiness_extra_words"])
def test_on_spends_at_most_two_decisions(name, mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    _turn, row, oracle = _class_turn(name, "on", mini_prod_world, monkeypatch)
    acting = _acting(row)
    assert len(acting) <= 2, acting
    if oracle.phrase_calls:
        assert acting[0] == ("turn_phrases", "picked")  # выбор фраз — первое решение хода, из его бюджета
    assert not any(entry.get("shadow") for entry in row.get("choose") or [])

