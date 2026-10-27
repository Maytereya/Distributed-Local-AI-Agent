"""Срок готовности при слабом совпадении через модуль выбора (ход (в)) — через эндпоинт, мини-прод.

Выключатель `[MESSENGER_ROUTER] choose_readiness = off | on` (по умолчанию off), действует под
пониманием хода `on`.

Инварианты класса:
- слабое совпадение в вопросе о сроке: услугу выбирает модель, срок печатает код из справочника МИС
  (принцип 1) — модель срока не видит и не пишет;
- модель ничего не выбрала, сбой LLM, off — честный перевод, как раньше;
- сильное совпадение по данным — срок без вызова LLM.

Разбор реплики на фразы — LLM разметки (`turn_spans_llm = on`, прод с 24.10): без неё услугу не угадываем (решение 22.10 (а)),
и вопрос о сроке — честный перевод до выбора. Поэтому тесты дают разметке оракул (`tests/spans_oracle.py`): фраза услуги —
от разметки, выбор услуги срока — подделка теста.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import FIXTURE, run_dialog

_MARK = "сколько готовится анализ или исследование"
_HCG = "ХГЧ (общий) Хорионический гонадотропин человека (ХГЧ, бета-ХГЧ, б-ХГЧ, Human Chorionic)"
_HCG_PHRASE = ("анализ крови на ХГЧ",)  # фраза услуги в «Сколько готовится анализ крови на ХГЧ?» — ответ разметки


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


class _ReadinessLLM:
    def __init__(self, wanted: tuple[str, ...] = (), fail: bool = False) -> None:
        self.wanted, self.fail, self.prompts = wanted, fail, []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.prompts.append(seen)
        if self.fail:
            raise ConnectionError("ollama down")
        picks = []
        for line in prompt.splitlines():
            head, dot, rest = line.partition(". ")
            if dot and head.isdigit() and rest in self.wanted:
                picks.append(int(head))
        return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)


def _setup(monkeypatch, mode: str, llm: _ReadinessLLM | None = None, phrases: tuple[str, ...] | None = None) -> None:
    """Выключатели хода (в); разметка хода — оракул с фразами `phrases` (None — разметка недоступна), прочее — `llm`."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_CHOOSE_READINESS", mode, raising=False)
    if llm is not None or phrases is not None:
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(phrases, inner=llm))
        keep_llm_breaker_closed(monkeypatch)


def _ask(question: str, session: str, world_root) -> dict:
    with ev_world.FrozenWorld(world_root, "day"):
        [turn] = run_dialog([question], session)
    return turn


def test_weak_match_gets_the_deadline_the_code_reads_for_the_service_the_model_picked(mini_prod_world, monkeypatch):
    llm = _ReadinessLLM(wanted=(_HCG,))
    _setup(monkeypatch, "on", llm, phrases=_HCG_PHRASE)
    turn = _ask("Сколько готовится анализ крови на ХГЧ?", "ready-weak", mini_prod_world)
    assert turn["text"] == f"Срок готовности «{_HCG}» — 1–2 дня."
    assert turn["handoff"] is False
    assert len(llm.prompts) == 1 and "1-2" not in llm.prompts[0]  # срок модели не показывается


@pytest.mark.parametrize("llm", [_ReadinessLLM(wanted=()), _ReadinessLLM(fail=True)], ids=["none_fit", "failure"])
def test_no_pick_or_failure_is_an_honest_handoff(mini_prod_world, monkeypatch, llm):
    _setup(monkeypatch, "on", llm, phrases=_HCG_PHRASE)
    turn = _ask("Сколько готовится анализ крови на ХГЧ?", "ready-none", mini_prod_world)
    assert turn["handoff"] is True and "дня" not in turn["text"]
    assert llm.prompts  # дошли до выбора модели: перевод — её ответ или сбой, а не сбой разметки


def test_off_keeps_the_handoff(mini_prod_world, monkeypatch):
    llm = _ReadinessLLM(wanted=(_HCG,))
    _setup(monkeypatch, "off", llm, phrases=_HCG_PHRASE)
    turn = _ask("Сколько готовится анализ крови на ХГЧ?", "ready-off", mini_prod_world)
    assert turn["handoff"] is True and llm.prompts == []


def test_strong_match_needs_no_llm(mini_prod_world, monkeypatch):
    # Фраза разбора «ферритин» — дословное название: срок без выбора LLM (разметка — LLM хода, выбор срока не зовётся).
    llm = _ReadinessLLM(wanted=(_HCG,))
    _setup(monkeypatch, "on", llm, phrases=("ферритин",))
    turn = _ask("сколько готовится ферритин", "ready-strong", mini_prod_world)
    assert turn["text"].startswith("Срок готовности «Ферритин» — ")
    assert llm.prompts == []
