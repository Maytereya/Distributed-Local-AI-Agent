"""Разбор реплики на фразы (L-15b, контракт C) в ходе — через обработчик эндпоинта, мини-прод.

Разбор — единственный режим с T7 шага 3б (16.10): режимы `turn_parse = off | shadow` и машина тени (фоновая задача вне
бюджета хода, `drain`, статус `late`) сняты. Решение владельца 13.10, Q4: лимит хода — 2 решения, выбор фраз — одно из них.

Инварианты:
- на ценовых ходах эталона разбор есть (`tu_parse_*` в телеметрии хода, без текста пациента); фразы — только в
  `debug.turn_parse`;
- решений хода не больше двух, выбор фраз — первое из них; записей тени нет.
Что меняется в ответах от разбора (читатели, шаг 5), — `test_turn_parse_readers.py`.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.spans_oracle import is_spans
from tests.test_price_one_or_many_class import _DEPLOY, CASES, _OracleLLM
from tests.test_price_turns_golden import DIALOGS, mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля
from tests.test_price_one_or_many_class import TRUE_CUT

_PHRASES_MARK = "понять, какие услуги назвал пациент"
_PARSE_FIELDS = ("tu_parse_status", "tu_parse_src", "tu_parse_n", "tu_parse_cands", "tu_parse_ms")


def _deploy(monkeypatch) -> None:
    """Набор деплоя; кэш выбора выключен (как на стенде): ходы одного теста не делят решений."""

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
        if is_spans(prompt):
            self.phrase_calls += 1  # разметка слов `turn_spans` (прод с деплоя 17) — те же фразы разреза
        if _PHRASES_MARK not in (system or "") or is_spans(prompt):
            return await super().generate(prompt=prompt, system=system, **kwargs)
        self.phrase_calls += 1
        lines = (line.partition(". ") for line in prompt.splitlines())
        picks = [int(head) for head, dot, rest in lines if dot and head.isdigit() and _normalise_catalog_text(rest.strip("«»")) in self.cut]
        return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)


def _acting(row: dict) -> list[tuple[str, str]]:
    return [(entry["task"], entry["status"]) for entry in row.get("choose") or [] if not entry.get("shadow")]


def _class_turn(name: str, world_dir, monkeypatch):
    oracle = _PhrasesOracle(name)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        [turn] = run_dialog([CASES[name].text], f"parse-live-{name}")
        row = world.take_telemetry()
    return turn, row, oracle


def test_parse_runs_on_price_turns_of_the_golden(mini_prod_world):  # noqa: F811
    """Встречная проверка: на ценовых ходах эталона разбор есть и решён к концу хода (статуса `late` больше нет)."""

    statuses = []
    for case in ("list_text_three", "one_ferritin_bare", "list_text_no_commas", "name_with_commas"):
        with ev_world.FrozenWorld(mini_prod_world, "day") as world:
            run_dialog(DIALOGS[case], f"parse-runs-{case}")
            statuses += [row.get("tu_parse_status") for row in world.telemetry]
    assert None not in statuses and "late" not in statuses, statuses
    assert {"strong_whole", "unavailable"} <= set(statuses), statuses


def test_parse_writes_fields_without_patient_text(mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    _turn, row, oracle = _class_turn("list_caps_run_on_priced", mini_prod_world, monkeypatch)
    assert oracle.phrase_calls == 1
    # Разметка слов `turn_spans` (прод с деплоя 17): варианты разбора — слова реплики («сколько стоит ФЕРРИТИН ГЛЮКОЗА
    # ХОЛЕСТЕРОЛ» — 5), а не окна генератора (было 4).
    assert {key: row[key] for key in _PARSE_FIELDS if key != "tu_parse_ms"} == {
        "tu_parse_status": "picked",
        "tu_parse_src": "llm",
        "tu_parse_n": 3,
        "tu_parse_cands": 5,
    }
    assert isinstance(row["tu_parse_ms"], int)
    [entry] = [entry for entry in row["choose"] if entry["task"] == "turn_spans"]
    assert (entry["status"], entry["options"], bool(entry.get("shadow"))) == ("picked", 5, False)
    assert not any(call.get("shadow") for call in row["llm"] if call["site"] == "turn_spans")
    dumped = json.dumps(row, ensure_ascii=False).lower()
    assert not any(word in dumped for word in ("ферритин", "глюкоза", "холестерол"))


def test_debug_shows_the_phrases_only_in_debug(mini_prod_world, monkeypatch):  # noqa: F811
    import asyncio

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    _deploy(monkeypatch)
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
def test_spends_at_most_two_decisions(name, mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    _turn, row, oracle = _class_turn(name, mini_prod_world, monkeypatch)
    acting = _acting(row)
    # Итог корзины (`basket_total`, `basket_calc = on` — прод с 26.10) — третье решение хода корзины, поиск — не больше двух.
    search = [decision for decision in acting if decision[0] != "basket_total"]
    assert len(search) <= 2 and len(acting) - len(search) <= 1, acting
    if oracle.phrase_calls:
        assert acting[0] == ("turn_spans", "picked")  # разметка фраз — первое решение хода, из его бюджета
    assert not any(entry.get("shadow") for entry in row.get("choose") or [])
