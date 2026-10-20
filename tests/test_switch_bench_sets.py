"""Наборы стенда для карточек LLM-выключателей (`evalbench/sets/sw_<выключатель>.jsonl`, облако 20.10, кикофф §3.3).

Каждый набор — диалоги `evalbench.dialogs` (один или несколько ходов) для карточки «off против on» одного выключателя:
случаи класса и контрольные (`control` — выключатель действовать не должен). Набор гоняют вместе с другими — id не
повторяются; синтетика — без телефонов, почты и длинных номеров (репозиторий публичный); в пометке — выключатель.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evalbench import dialogs as ev_dialogs

_SETS = sorted((Path(__file__).resolve().parents[1] / "evalbench" / "sets").glob("sw_*.jsonl"))


def test_there_are_switch_sets():
    assert len(_SETS) >= 8, _SETS


def _is_dialog_set(path: Path) -> bool:
    first = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
    return '"turns"' in first  # однотуровые наборы `evalbench.run` (`question`) — другой формат


def test_ids_are_unique_across_all_dialog_sets():
    dialogs = ev_dialogs.load_dialogs([path for path in sorted(_SETS[0].parent.glob("*.jsonl")) if _is_dialog_set(path)])
    ids = [d["id"] for d in dialogs]
    assert len(set(ids)) == len(ids), sorted({i for i in ids if ids.count(i) > 1})


@pytest.mark.parametrize("path", _SETS, ids=lambda p: p.stem)
def test_switch_set_is_well_formed(path):
    dialogs = ev_dialogs.load_dialogs([path])
    assert dialogs
    assert all(d.get("expect") and d.get("tags") and str(d.get("note", "")).startswith("выключатель ") for d in dialogs)
    assert all(set(d.get("about_shown") or ()) <= set(range(2, len(d["turns"]) + 1)) for d in dialogs)
    assert any("control" in d["tags"] for d in dialogs), "нужен хотя бы один контрольный диалог (выключатель не действует)"
    texts = [ev_dialogs._turn(turn)[0] for d in dialogs for turn in d["turns"]]
    assert all(text.strip() for text in texts)
    assert not [t for t in texts if re.search(r"\d{5,}|@|\+7", t)]
