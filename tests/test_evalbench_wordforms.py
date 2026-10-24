"""Детектор «один смысл — разные словоформы» (`evalbench.wordforms`, решение владельца 22.10) — на синтетических ходах.

Инвариант: группы берутся из набора прогона (поле `group`); подпись ответа — суть (исход, строки цены, памятка), а не
формулировка: одна цена разными словами — согласие; другая памятка или другой исход — расхождение.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalbench import wordforms


def _run(tmp_path: Path, turns: list[dict]) -> dict:
    sets = tmp_path / "set.jsonl"
    sets.write_text(
        "\n".join(json.dumps({"id": t["id"], "group": t["id"].rsplit("_", 1)[0], "question": t["question"]}) for t in turns),
        encoding="utf-8",
    )
    return {"sets": [str(sets)], "mode": "real"}


def test_same_price_rows_in_other_words_agree(tmp_path: Path) -> None:
    turns = [
        {"id": "g_0", "question": "Цена анализа", "outcome": "price", "answer": "По услуге «Ферритин» — 530 руб.", "price_pairs": [{"service_id": 408}]},
        {"id": "g_1", "question": "Сколько стоят анализы", "outcome": "price", "answer": "Ферритин — 530 руб.", "price_pairs": [{"service_id": 408}]},
    ]
    groups = wordforms.agreement(_run(tmp_path, turns), turns)
    assert len({sig for _, sig in groups["g"]}) == 1


def test_other_memo_or_outcome_splits_group(tmp_path: Path) -> None:
    turns = [
        {"id": "p_0", "question": "Подготовка к анализу крови", "outcome": "answer", "answer": "Кровь сдается без записи, натощак."},
        {"id": "p_1", "question": "Подготовка к анализам крови", "outcome": "memo", "answer": "Памятка клиники «Анализ мочи по Нечипоренко»: …"},
        {"id": "q_0", "question": "Цена приёма", "outcome": "price", "answer": "…", "price_pairs": [{"service_id": 1363}]},
        {"id": "q_1", "question": "Цена приема", "outcome": "handoff", "answer": "Подключаю оператора."},
    ]
    groups = wordforms.agreement(_run(tmp_path, turns), turns)
    assert len({sig for _, sig in groups["p"]}) == 2
    assert "памятка «Анализ мочи по Нечипоренко»" in groups["p"][1][1]
    assert len({sig for _, sig in groups["q"]}) == 2


def test_turns_outside_set_groups_are_ignored(tmp_path: Path) -> None:
    turns = [{"id": "g_0", "question": "x", "outcome": "answer", "answer": "a"}]
    header = _run(tmp_path, turns)
    extra = [*turns, {"id": "other", "question": "y", "outcome": "answer", "answer": "b"}]
    assert set(wordforms.agreement(header, extra)) == {"g"}
