"""Запись пула выбора строки прайса в телеметрию хода (Recall@pool, замер 22.10): только запись.

Recall@pool — доля вопросов, где верная строка вообще есть среди кандидатов, показанных LLM; это потолок точности
выбора. Старый замер 07.10 копировал логику сборки пула и устарел вместе с ней. Теперь пул пишет сам ход: `_ask` кладёт
`serviceId` строк, ушедших в `choose()`, в поле `price_pool` строки хода (`turn_telemetry.note_price_pool`), стенд
считает пересечение с разметкой (`evalbench.detectors.pool_recall`).

Инварианты класса (через обработчик эндпоинта, мини-прод, оракул LLM настоящего типа; оба круга — случай «глотка»,
где LLM отвергает строку в первом круге и называет термины для второго):
- запись — ровно то, что видела LLM: на каждый вызов выбора одна запись с кругом, размером пула и `serviceId` строк
  в том же порядке, что позиции в подсказке; без текста;
- запись не меняет ход: с выключенной записью ответ, пулы LLM и решения хода те же;
- выключенная телеметрия хода — записи нет и ход не падает.
"""

from __future__ import annotations

import json

import pytest

from messengers_router import turn_telemetry
from tests.test_price_option_mis_notes_class import (  # noqa: F401 — фикстуры мира и словаря МИС
    _CASES,
    _EXTRA_ROWS,
    _ByMaterial,
    mini_prod_world,
    mis_vocab,
)
from tests.test_price_turns_golden import FIXTURE
from tests.test_qualifier_round_two_class import _run

_NOTE_PRICE_POOL = turn_telemetry.note_price_pool


def _price_ids() -> dict[tuple[str, int], int]:
    """(название, цена) → serviceId по прайсу мира теста."""

    [price] = FIXTURE.glob("price_by_region/price_region_3_*.jsonl")
    rows = [json.loads(line) for line in price.open(encoding="utf-8") if line.strip()] + _EXTRA_ROWS
    return {(str(row["serviceName"]).strip(), int(float(row["cost"]))): int(row["serviceId"]) for row in rows}


def _option_id(text: str, ids: dict[tuple[str, int], int]) -> int:
    # «название — цена руб. (пометка МИС)»: в пометке бывает своё « — » («материал — …»), поэтому сначала отрезаем её.
    name, _dash, cost = text.split(" руб.", 1)[0].rpartition(" — ")
    return ids[(name.strip(), int(cost))]


def _turn(case: str, world, monkeypatch, *, record: bool, notes: str = "off", **switches):
    reply, phrase, pick, material = _CASES[case]
    oracle = _ByMaterial(phrase, pick, material)
    monkeypatch.setattr(turn_telemetry, "note_price_pool", _NOTE_PRICE_POOL if record else (lambda *_a, **_kw: None))
    session = f"pool-{case}-{'on' if record else 'off'}-{notes}-{'-'.join(f'{k}{v}' for k, v in switches.items())}"
    [turn], rows = _run([reply], world, monkeypatch, llm=oracle, session=session, MR_PRICE_OPTION_MIS_NOTES=notes, **switches)
    return oracle, turn, rows


@pytest.mark.parametrize("notes", ["off", "on"])  # on — позиции с пометкой МИС (в ней своё « — »)
@pytest.mark.parametrize("case", _CASES)
def test_pool_record_is_exactly_what_the_llm_saw(case, notes, mini_prod_world, mis_vocab, monkeypatch):  # noqa: F811
    oracle, _turn_record, [row] = _turn(case, mini_prod_world, monkeypatch, record=True, notes=notes)
    pools = row.get("price_pool")
    assert pools and len(pools) == len(oracle.pools), (pools, oracle.pools)
    ids = _price_ids()
    for entry, seen in zip(pools, oracle.pools):
        assert set(entry) == {"r", "n", "ids"}, entry
        assert entry["n"] == len(seen)
        assert entry["ids"] == [_option_id(text, ids) for text in seen], (entry, seen)
    assert [entry["r"] for entry in pools][0] == 1
    assert all(entry["r"] in (1, 2) for entry in pools)
    if case == "throat" and notes == "off":  # LLM не видит материала, отвергает строку и даёт термины: второй круг отдельно
        assert [entry["r"] for entry in pools] == [1, 2], pools


@pytest.mark.parametrize("case", _CASES)
def test_pool_record_does_not_change_the_turn(case, mini_prod_world, mis_vocab, monkeypatch):  # noqa: F811
    off, turn_off, [row_off] = _turn(case, mini_prod_world, monkeypatch, record=False)
    on, turn_on, [row_on] = _turn(case, mini_prod_world, monkeypatch, record=True)
    assert turn_on["text"] == turn_off["text"]
    assert on.pools == off.pools
    assert "price_pool" not in row_off and row_on.get("price_pool")
    skip = {"price_pool", "ts", "tid", "sess", "ms", "stage_ms", "llm", "tool_ms", "world", "world_id", "choose", "src"}
    def decisions(row: dict) -> dict:  # решения хода без времени и служебных полей
        return {k: v for k, v in row.items() if k not in skip and not k.endswith("_ms")}

    assert decisions(row_on) == decisions(row_off)
    assert [(c["task"], c["status"], c["options"]) for c in row_on.get("choose") or []] == [
        (c["task"], c["status"], c["options"]) for c in row_off.get("choose") or []
    ]


def test_telemetry_off_writes_no_pool_and_the_turn_answers(mini_prod_world, mis_vocab, monkeypatch):  # noqa: F811
    _on, turn_on, _rows = _turn("throat", mini_prod_world, monkeypatch, record=True)
    _off, turn_off, rows = _turn("throat", mini_prod_world, monkeypatch, record=True, MR_TURN_TELEMETRY=False)
    assert rows == []
    assert turn_off["text"] == turn_on["text"]


def test_note_price_pool_keeps_only_ids_outside_a_turn_is_silent():
    assert turn_telemetry.note_price_pool(1, [1, 2]) is None  # хода нет — ничего не пишется и не падает
    token = turn_telemetry.begin("t", session_id="s")
    try:
        turn_telemetry.note_price_pool(1, [5, "7", None, "x"])
        turn_telemetry.note_price_pool(2, [])
        fields = turn_telemetry._TURN.get().fields
    finally:
        turn_telemetry._TURN.reset(token)
    assert fields["price_pool"] == [{"r": 1, "n": 4, "ids": [5, 7]}, {"r": 2, "n": 0, "ids": []}]
