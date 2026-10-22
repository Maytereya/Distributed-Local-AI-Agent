"""Запись пула выбора корзины в телеметрию хода (Recall@pool списков, 22.10): только запись.

Списки цен ведёт выбор корзины (`choose()` с задачей `price_basket`, `_basket_select._pick`), а не выбор строки
(`select_price_rows`): до этой записи 73 из 142 размеченных вопросов dev без пула были корзиной. Корзина пишет в то же
поле `price_pool` одну запись `{"r": 1, "src": "basket", "n", "ids", "strong"}`: `ids` — `serviceId` строк, которые
видела LLM корзины, в порядке вариантов подсказки; `strong` — строки сильных позиций, решённые по данным без LLM.

Инварианты класса (через обработчик эндпоинта, мини-прод, оракул LLM настоящего типа):
- запись — ровно то, что видела LLM корзины: `ids` = варианты подсказки по порядку, `n` = их число = `options` решения
  `price_basket` хода; LLM корзины не звалась (все позиции сильные) — `ids` пусто, `n` = 0, строки — в `strong`;
- ответ корзины не выходит за записанный пул: каждая строка с ценой в ответе — из `ids ∪ strong`;
- запись не меняет ход: с выключенной записью ответ, подсказки LLM и решения хода те же;
- LLM недоступна (корзина правил, выбор не удался) — запись есть, `n` = `options` решения корзины.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_telemetry
from tests.test_price_one_or_many_class import CASES
from tests.test_price_pool_record_class import _price_ids
from tests.test_price_turns_golden import DIALOGS, mini_prod_world, run_dialog  # noqa: F401 — фикстура мира
from tests.test_turn_parse_readers import _BASKET_MARK, _Recorder
from tests.test_turn_parse_slot import _deploy

_NOTE_PRICE_POOL = turn_telemetry.note_price_pool
_OPTION_RE = re.compile(r"^\d+\. «[^»]*»(?: \[добавлено: «[^»]*»\])? → (.+) — (\d+) руб\.$", re.MULTILINE)
_ANSWER_ROW_RE = re.compile(r"^\d+\. (.+) — (\d[\d ]*) руб\.", re.MULTILINE)
_LLM_CASES = ("list_true_three",)  # слабые позиции — выбор LLM корзины; «ферритин» — сильная
_STRONG_CASES = ("list_caps_run_on_priced",)  # все позиции сильные — LLM корзины не звалась


def _record(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(turn_telemetry, "note_price_pool", _NOTE_PRICE_POOL if on else (lambda *_a, **_kw: None))


def _turn(case: str, world, monkeypatch, *, record: bool):
    _deploy(monkeypatch)
    _record(monkeypatch, record)
    llm = _Recorder(case)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(world, "day") as frozen:
        [turn] = run_dialog([CASES[case].text], f"basket-pool-{case}-{'on' if record else 'off'}")
        row = frozen.take_telemetry()
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return llm, turn, row


def _basket(row: dict) -> dict:
    [entry] = [p for p in row.get("price_pool") or [] if p.get("src") == "basket"]
    assert set(entry) == {"r", "n", "ids", "src", "strong"} and entry["r"] == 1, entry
    return entry


def _answer_ids(text: str, ids: dict[tuple[str, int], int]) -> set[int]:
    found = {(name.strip(), int(cost.replace(" ", "").replace(" ", ""))) for name, cost in _ANSWER_ROW_RE.findall(text)}
    assert found, text
    return {ids[key] for key in found}


@pytest.mark.parametrize("case", _LLM_CASES)
def test_basket_record_is_exactly_what_the_llm_saw(case, mini_prod_world, monkeypatch):  # noqa: F811
    llm, turn, row = _turn(case, mini_prod_world, monkeypatch, record=True)
    ids = _price_ids()
    [prompt] = llm.asked(_BASKET_MARK)
    seen = [ids[(name.strip(), int(cost))] for name, cost in _OPTION_RE.findall(prompt)]
    entry = _basket(row)
    assert seen and entry["ids"] == seen and entry["n"] == len(seen), (entry, seen)
    [decision] = [c for c in row["choose"] if c["task"] == "price_basket"]
    assert decision["options"] == entry["n"]
    assert entry["strong"], entry  # «ферритин» — сильная: строки по данным, LLM их не видела
    assert not set(entry["strong"]) & set(entry["ids"])
    assert _answer_ids(turn["text"], ids) <= set(entry["ids"]) | set(entry["strong"])


@pytest.mark.parametrize("case", _STRONG_CASES)
def test_all_strong_basket_records_data_rows_without_llm(case, mini_prod_world, monkeypatch):  # noqa: F811
    llm, turn, row = _turn(case, mini_prod_world, monkeypatch, record=True)
    assert llm.asked(_BASKET_MARK) == []
    entry = _basket(row)
    assert (entry["n"], entry["ids"]) == (0, []) and entry["strong"], entry
    assert _answer_ids(turn["text"], _price_ids()) <= set(entry["strong"])


@pytest.mark.parametrize("case", _LLM_CASES + _STRONG_CASES)
def test_basket_record_does_not_change_the_turn(case, mini_prod_world, monkeypatch):  # noqa: F811
    off, turn_off, row_off = _turn(case, mini_prod_world, monkeypatch, record=False)
    on, turn_on, row_on = _turn(case, mini_prod_world, monkeypatch, record=True)
    assert turn_on["text"] == turn_off["text"]
    assert on.prompts == off.prompts
    assert "price_pool" not in row_off and _basket(row_on)
    skip = {"price_pool", "ts", "tid", "sess", "ms", "stage_ms", "llm", "tool_ms", "world", "world_id", "choose", "src"}

    def decisions(row: dict) -> dict:  # решения хода без времени и служебных полей
        return {k: v for k, v in row.items() if k not in skip and not k.endswith("_ms")}

    assert decisions(row_on) == decisions(row_off)
    assert [(c["task"], c["status"], c["options"]) for c in row_on.get("choose") or []] == [
        (c["task"], c["status"], c["options"]) for c in row_off.get("choose") or []
    ]


def test_llm_down_rules_basket_still_records_its_pool(mini_prod_world, monkeypatch):  # noqa: F811
    """LLM недоступна: разбора нет, корзину режут правила, выбор строк корзины не удался (политика сбоя) — пул, ушедший
    в `choose()`, записан, и его размер = число вариантов решения корзины."""

    _deploy(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day") as frozen:
        run_dialog(DIALOGS["list_text_three"], "basket-pool-down")
        rows = [r for r in frozen.telemetry if r.get("kind", "turn") == "turn"]
    pooled = [(r, p) for r in rows for p in r.get("price_pool") or [] if p.get("src") == "basket"]
    assert pooled, rows
    for row, entry in pooled:
        basket = [c for c in row.get("choose") or [] if c["task"] == "price_basket"]
        assert [c["options"] for c in basket] == ([entry["n"]] if entry["n"] else []), (entry, basket)
        assert all(c["status"] != "picked" for c in basket)
