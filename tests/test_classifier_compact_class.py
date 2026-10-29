"""Короче ответ LLM-классификатора темы (`classifier_compact`; BUG-2026-10-27-LLM-QUEUE-GIVES-UP, шаг 5).

Журнал прода 14–23.10: ответы классификатора двух видов — ≈ 70 токенов (3,4 с) и ≈ 227 токенов (11,6 с); длинные печатают все
~30 полей шаблона `entities`, почти все — null (кэш стенда: 1 077 из 2 745 ответов с 25–29 полями). Длинный классификатор держит
слот LLM, пока задачи соседних ходов ждут (шаг 1), и задаёт p95 хода.

Инвариант класса: под `on` промпт просит только найденные поля (строка `classifier_compact_note`, одинаковая в обеих локациях,
сразу после «верни только JSON»), и короткий ответ разбирается так же, как длинный с теми же значениями: та же метка, тот же ход;
off — промпт байт в байт прежний. Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from messengers_router import classifier
from messengers_router.runtime_config import config as cfg
from tests._llm_helpers import _turn
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_ROOT = Path(__file__).resolve().parents[1]
_KEYS = (
    "doctor_name", "doctor_id", "specialty", "branch_name", "branch_id", "city", "service_name", "appointment_action",
    "patient_name", "test_name", "test_goal", "surname", "year", "filial", "number", "order_id", "result_action", "lang",
    "insurance_type", "accepts_children", "child_age", "date_hint", "date_from", "date_to", "time_from", "time_to",
)
# Реплика → (метка LLM, найденные сущности); ни у одной правило не находит своих сущностей — дарить нечего.
_REPLIES = {
    "no_rule": ("подскажите по одному вопросу", "OTHER", {}),
    "schedule": ("когда принимает кардиолог", "DOCTOR_SCHEDULE", {"specialty": "кардиолог"}),
}


def _answer(label: str, found: dict, *, compact: bool) -> str:
    entities = dict(found) if compact else {**{k: None for k in _KEYS}, **found}
    return json.dumps({"label": label, "confidence": 0.9, "context_action": "new_topic", "entities": entities, "flags": []})


def _note() -> list[str]:
    return [
        (_ROOT / "messengers_router" / "prompts" / "classifier_compact_note.txt").read_text(encoding="utf-8").strip(),
        (_ROOT / "app_data" / "prompts" / "mr_classifier_compact_note.txt").read_text(encoding="utf-8").strip(),
    ]


def test_off_prompt_is_unchanged(monkeypatch):
    monkeypatch.setattr(cfg, "MR_CLASSIFIER_COMPACT", "off", raising=False)
    built = classifier._build_classify_prompt("когда принимает кардиолог", {})
    template = (_ROOT / "messengers_router" / "prompts" / "classifier_patient.txt").read_text(encoding="utf-8")
    assert built.splitlines()[:3] == template.splitlines()[:3] and _note()[0] not in built


def test_on_prompt_asks_for_found_fields_only_right_after_json_only(monkeypatch):
    bundle, host = _note()
    assert bundle == host and "null" in bundle
    monkeypatch.setattr(cfg, "MR_CLASSIFIER_COMPACT", "on", raising=False)
    lines = classifier._build_classify_prompt("когда принимает кардиолог", {}).splitlines()
    assert lines[1].startswith("Верни только JSON") and lines[2] == bundle and lines.count(bundle) == 1
    assert "<<" not in "\n".join(lines)


@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_compact_answer_is_the_same_turn_as_the_long_one(reply, mini_prod_world, monkeypatch):  # noqa: F811
    text, label, found = _REPLIES[reply]
    monkeypatch.setattr(cfg, "MR_CLASSIFIER_COMPACT", "on", raising=False)
    short = _turn(mini_prod_world, monkeypatch, text, answer=_answer(label, found, compact=True))
    long = _turn(mini_prod_world, monkeypatch, text, answer=_answer(label, found, compact=False))

    assert (short["text"], short["label"]) == (long["text"], long["label"])
    assert short["telemetry"].get("tu_label_nlu") == long["telemetry"].get("tu_label_nlu")


def test_switch_defaults_to_off():
    from agent_logic_2 import config as host_config

    assert host_config.settings.messenger_router.classifier_compact == "off"
