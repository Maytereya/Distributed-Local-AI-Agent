"""LLM-различитель «отменить / перенести» без слова «запись» (05.10).

Ответ разбирается строго: «ЗАПИСЬ» — да, «ДРУГОЕ» — нет, всё остальное и сбой — None, и
тогда вызывающий ведёт себя как до 04.10. Промпт — в обеих локациях одинаковый.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from messengers_router.services import _appointment_change_validator as V

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(V, "_CACHE", {})


def _llm_answers(monkeypatch, answer) -> list[str]:
    prompts: list[str] = []

    async def fake_generate_text(prompt, **_kwargs):
        prompts.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(V, "generate_text", fake_generate_text)
    return prompts


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        ("ЗАПИСЬ", True),
        ("Запись.", True),
        ("«ДРУГОЕ»\nпотому что речь о лекарствах", False),
        ("Не уверен", None),
        ("", None),
        (TimeoutError(), None),
    ],
    ids=["yes", "yes_punct", "no_with_reason", "unclear", "empty", "llm_failure"],
)
def test_verdict_is_parsed_strictly(monkeypatch, answer, verdict):
    prompts = _llm_answers(monkeypatch, answer)

    assert asyncio.run(V.is_own_appointment_change("хочу отменить, заболела")) is verdict
    assert "«хочу отменить, заболела»" in prompts[0]


def test_answer_is_cached_per_phrase(monkeypatch):
    prompts = _llm_answers(monkeypatch, "ЗАПИСЬ")

    for text in ("Хочу отменить,  заболела", "хочу отменить, заболела"):
        assert asyncio.run(V.is_own_appointment_change(text)) is True
    assert len(prompts) == 1


def test_kill_switch_skips_llm(monkeypatch):
    prompts = _llm_answers(monkeypatch, "ДРУГОЕ")
    monkeypatch.setattr(V._cfg, "MR_LLM_APPOINTMENT_CHANGE_VALIDATION", False)

    assert asyncio.run(V.is_own_appointment_change("хочу отменить, заболела")) is None
    assert prompts == []


def test_prompt_copies_are_identical():
    bundle = (_ROOT / "messengers_router" / "prompts" / "appointment_change_validator.txt").read_text(encoding="utf-8")
    host = (_ROOT / "app_data" / "prompts" / "mr_appointment_change_validator.txt").read_text(encoding="utf-8")
    assert bundle == host
    assert "<<TEXT>>" in bundle
