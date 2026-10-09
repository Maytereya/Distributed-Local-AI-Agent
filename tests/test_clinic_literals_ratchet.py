"""Храповик переносимости: знания о конкретной клинике в ядре бота только убывают.

Решение владельца 09.10 (повестка рефакторинга №12): второй клиники пока нет, масштабирование —
отдельным треком после рефакторинга; до тех пор — дешёвая страховка, чтобы ядро не обрастало
новыми привязками к «Науке». Ядро — пакет `messengers_router` (без тестов).

Как работать с тестом: число выросло — уберите новую привязку (данные — из МИС или профиля, клиент
МИС — через существующий сервис). Число упало — опустите потолок здесь же, в том же коммите.
"""

from __future__ import annotations

import re
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"

# Потолки на 09.10.2026 (ревью DLA, ход (з): «235 / 44 / 29» — другая методика подсчёта).
_CEILINGS = {
    "упоминания Самары (samara / самар)": (re.compile(r"(?i)samara|самар"), 425),
    "литерал адреса одного филиала (Ленина, 5)": (re.compile(r"Ленина,? 5"), 27),
    "прямые импорты клиента МИС (agent_logic_2.nayka_api)": (
        re.compile(r"^\s*(?:from agent_logic_2\.nayka_api import|import agent_logic_2\.nayka_api)", re.M),
        14,
    ),
}


def _core_sources() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in _CORE.rglob("*.py") if "tests" not in p.parts]


def test_clinic_specific_knowledge_in_core_only_decreases():
    sources = _core_sources()
    grown = {}
    for name, (pattern, ceiling) in _CEILINGS.items():
        count = sum(len(pattern.findall(src)) for src in sources)
        if count > ceiling:
            grown[name] = f"{count} > потолка {ceiling}"
    assert grown == {}, f"в ядре прибавилось привязок к клинике: {grown}"
