"""Механизм имён для тестов, которые держат инвариант при обоих путях: `names_* = off` — форма и списки, `on` — сущности
LLM-классификатора хода (STEP-2026-10-24-NAMES-NO-STOPWORDS). Подделка LLM — настоящего типа: JSON-строка модели на границе
`classifier.generate_text`, дальше — настоящий разбор (`ollama_classify_payload` → `analyze` → `turn_names.note_llm`)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from messengers_router import classifier, turn_understanding
from messengers_router.runtime_config import config

NAMES_SWITCHES = (
    "names_patient_fio",
    "names_patient_rule",
    "names_patient_fill",
    "names_result_surname",
    "names_doctor_followup",
    "names_doctor_noise",
    "names_doctor_catalog",
)
MODES = ("off", "on")


def set_names(monkeypatch, value: str) -> None:
    for name in NAMES_SWITCHES:
        monkeypatch.setattr(config, "MR_" + name.upper(), value, raising=False)


def llm_answers(monkeypatch, *, label: str, entities: dict[str, Any], confidence: float = 0.9) -> None:
    """LLM-классификатор отвечает так (как модель — JSON-строкой)."""

    async def model(*_args, **_kwargs) -> str:
        return json.dumps(
            {"label": label, "confidence": confidence, "context_action": "continue", "entities": entities, "flags": []},
            ensure_ascii=False,
        )

    monkeypatch.setattr(classifier, "generate_text", model)


def after_classifier(text: str, last_entities: dict[str, Any], then: Callable[[], Any]) -> Any:
    """Ход: LLM-классификатор разобрал реплику (`analyze`, путь LLM), затем `then()` в том же ходе."""

    async def turn() -> Any:
        turn_understanding.begin_turn()
        await classifier.analyze(text, last_entities, skip_rule=True)
        return then()

    return asyncio.run(turn())
