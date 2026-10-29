"""LLM-3 (ревью DLA, стенд 09.10): уточняющая классификация — не больше одного раза за ход.

Инварианты класса «один и тот же вызов LLM дважды за ход»:
- через эндпоинт любой ход зовёт refine-pass (`classifier_refine_patient`) не больше одного раза;
- ветка правила, чью тему решает LLM-классификатор (`topic_by_llm`), refine не зовёт вовсе: его итог выбросили бы, а вызов
  шёл вне лимита решений хода (Н-3, STEP-2026-10-30-REFINE-BY-LLM-BRANCH); та же ветка без `topic_by_llm` refine зовёт;
- ранний safety-гард (`early_guards`) от refine не зависит: refine уточняет только
  DOCTOR_SCHEDULE / DOCTOR_INFO / APPOINTMENT в несрочные метки, срочность он не даёт и не снимает;
- итог хода (текст ответа, метка, сущности) — тот же, что при прежнем двойном refine
  (характеризация на подделке LLM, отвечающей детерминированно по промпту).
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ollama import GenerateResponse

from messengers_router import classifier
from messengers_router import endpoint as endpoint_mod
from messengers_router import llm_runtime
from messengers_router.memory import MemoryStore
from messengers_router.services import Services

_REFINE_MARKER = "refine-pass"

# Реплики, которые детерминированный слой отдаёт в refine (серые DOCTOR_SCHEDULE / APPOINTMENT),
# и соседи без refine — счётчик не должен расти и у них. «Расскажи про врача» в refine больше не идёт:
# ветка `rule_doctor_info` снята, тему решает классификатор (STEP-2026-10-28-C6-RULES-NEVER-FIRED).
# Ветка `rule_schedule` — тему решает правило, пока ветки нет в `topic_by_llm`; ветка `rule_appointment` — всегда правило.
_SCHEDULE_TEXTS = [
    "покажи расписание кардиолога",
    "когда принимает невролог",
    "расписание врачей на завтра",
]
_APPOINTMENT_TEXTS = [
    "хочу записаться к кардиологу или лучше к терапевту",
    "запишите меня на прием, уточню у кого позже",
]
_REFINE_TEXTS = _SCHEDULE_TEXTS + _APPOINTMENT_TEXTS
_OTHER_TEXTS = [
    "сколько стоит общий анализ крови",
    "задыхаюсь",
    "привет",
    "где вы находитесь",
]


class _PromptDeterministicOllama:
    """Подделка ollama: ответ зависит только от промпта, вызовы refine считаются по маркеру."""

    def __init__(self) -> None:
        self.refine_calls = 0

    async def generate(self, **kwargs):
        prompt = str(kwargs.get("prompt") or "")
        if _REFINE_MARKER in prompt:
            self.refine_calls += 1
            allowed = re.search(r'"label":\s*"<one of: ([A-Z_]+)', prompt)
            label = allowed.group(1) if allowed else "OTHER"
            payload = {
                "label": label,
                "confidence": 0.9,
                "context_action": "continue",
                # Сущность, которой нет в правилах: без refine итог был бы другим.
                "entities": {"date_hint": "tomorrow"},
                "flags": [],
            }
            return GenerateResponse(model="m", response=json.dumps(payload, ensure_ascii=False), done=True)
        if kwargs.get("format") == "json":
            payload = {"label": "OTHER", "confidence": 0.3, "entities": {}, "flags": []}
            return GenerateResponse(model="m", response=json.dumps(payload), done=True)
        return GenerateResponse(model="m", response="Ответ.", done=True)


@pytest.fixture
def fake_llm(monkeypatch):
    from agent_logic_2.nayka_api import api_nayka, api_price

    from messengers_router.runtime_config import config as cfg

    # Расписание — правилом (как до `topic_by_llm`): набор ниже проверяет сам refine; ветку LLM — отдельные тесты.
    monkeypatch.setattr(cfg, "MR_TOPIC_BY_LLM", "", raising=False)
    fake = _PromptDeterministicOllama()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
    monkeypatch.setattr(api_nayka, "site_regions", lambda *a, **kw: [])
    # Мир хода офлайн: врачи — из последнего локального среза, priceUnits «на сегодня» не качаем.
    monkeypatch.setattr(api_nayka, "get_all_doctors", lambda: [])

    def _no_download(*_a, **_kw):
        raise ConnectionError("priceUnits на «сегодня» тест не качает")

    monkeypatch.setattr(api_price.SESSION, "get", _no_download)
    return fake


def _client() -> TestClient:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    memory = MemoryStore()
    app = FastAPI()
    app.include_router(endpoint_mod.router)
    app.dependency_overrides[endpoint_mod.get_memory_store] = lambda: memory
    app.dependency_overrides[endpoint_mod.get_services] = lambda: services
    return TestClient(app)


def _turn(text: str) -> dict:
    return _client().post(
        "/api/messenger-generate-once",
        json={"session_id": "tg_refine_once", "text": text, "debug": True},
    ).json()


def _outcome(body: dict) -> tuple:
    """Итог хода: текст ответа, перевод, метка, сущности и флаги решения из debug."""
    debug = (body.get("state_update") or {}).get("debug") or {}
    decision = debug.get("decision") or {}
    return (
        body.get("text"),
        body.get("handoff"),
        decision.get("label"),
        json.dumps(decision.get("entities") or {}, ensure_ascii=False, sort_keys=True),
        tuple(decision.get("flags") or ()),
    )


@pytest.mark.parametrize("text", _REFINE_TEXTS + _OTHER_TEXTS)
def test_refine_runs_at_most_once_per_turn_through_endpoint(fake_llm, text):
    body = _turn(text)

    assert body["text"]
    assert fake_llm.refine_calls <= 1, f"{text!r}: refine за ход {fake_llm.refine_calls} раз(а)"


def test_refine_texts_really_reach_refine(fake_llm):
    """Защита от пустого теста: набор действительно уходит в refine (иначе ≤1 — тавтология)."""
    reached = 0
    for text in _REFINE_TEXTS:
        fake_llm.refine_calls = 0
        _turn(text)
        reached += fake_llm.refine_calls == 1
    assert reached >= len(_REFINE_TEXTS) - 1


@pytest.mark.parametrize("text", _REFINE_TEXTS)
def test_single_refine_keeps_turn_outcome(fake_llm, monkeypatch, text):
    """Характеризация: итог хода совпадает с прежним поведением (refine и в safety-гарде)."""
    after = _outcome(_turn(text))

    original = classifier.deterministic_rule_decision

    async def legacy_always_refine(*args, **kwargs):
        kwargs["allow_refine"] = True
        return await original(*args, **kwargs)

    monkeypatch.setattr(classifier, "deterministic_rule_decision", legacy_always_refine)
    before = _outcome(_turn(text))

    assert before[2] is not None, "debug не вернул решение — сравнивать нечего"
    assert "llm_refine_used" in before[4], "реплика не дошла до refine — характеризация пустая"
    assert after == before


@pytest.mark.parametrize("text", _REFINE_TEXTS + _OTHER_TEXTS)
def test_refine_never_changes_safety_status_of_rule_pass(fake_llm, text):
    """Основание фикса: safety-гарду refine не нужен — он не меняет «срочное или нет»."""

    async def both():
        with_refine = await classifier.deterministic_rule_decision(text, {}, allow_refine=True)
        without = await classifier.deterministic_rule_decision(text, {}, allow_refine=False)
        return with_refine, without

    with_refine, without = asyncio.run(both())
    safety = {"URGENT", "COMPLAINT", "MEDICAL_ADVICE"}
    assert (getattr(with_refine, "label", None) in safety) == (getattr(without, "label", None) in safety)


@pytest.mark.parametrize("text", _SCHEDULE_TEXTS)
def test_branch_decided_by_llm_never_calls_refine(fake_llm, monkeypatch, text):
    """Н-3: тему ветки решает классификатор — уточнение правила не зовётся, классификатор зовётся (тема — у LLM)."""
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TOPIC_BY_LLM", "rule_schedule", raising=False)
    body = _turn(text)

    assert body["text"]
    assert fake_llm.refine_calls == 0, f"{text!r}: refine у ветки topic_by_llm — {fake_llm.refine_calls} раз(а)"
    flags = ((body.get("state_update") or {}).get("debug") or {}).get("decision", {}).get("flags") or []
    assert "topic_by_llm" in flags, "ветка не ушла в LLM — проверка refine пустая"


@pytest.mark.parametrize("text", _APPOINTMENT_TEXTS)
def test_branch_decided_by_rule_keeps_refine_under_topic_by_llm(fake_llm, monkeypatch, text):
    """Обратная сторона: ветка не из `topic_by_llm` уточнение зовёт и при включённом списке — снят только лишний вызов."""
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TOPIC_BY_LLM", "rule_schedule", raising=False)
    _turn(text)

    assert fake_llm.refine_calls == 1
