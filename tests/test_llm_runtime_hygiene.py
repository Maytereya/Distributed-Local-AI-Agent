"""Гигиена LLM-слоёв (ревью DLA: LLM-1, LLM-2; L-03; решение владельца 08.10, повестка №10).

Инварианты класса:
- сбой LLM (таймаут, очередь, ошибка) не кэшируется: следующий ход спрашивает модель снова,
  а минутная перегрузка не закрепляет вердикт до рестарта для всех пациентов (LLM-1);
- каждый выключатель LLM-слоя, который читает код, объявлен в `agent_logic_2/config.py` —
  иначе он всегда включён, и выключить слой в инциденте нельзя (LLM-1);
- параметры генерации не зависят от того, вызывался ли операторский эндпоинт: пустой кэш
  настроек инициализируется сам (LLM-2);
- короткие решения идут с явными параметрами: temperature 0 (L-03).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from agent_logic_2 import config as app_config
from agent_logic_2 import ollama_settings
from messengers_router import llm_runtime
from messengers_router.services import (
    _appointment_change_validator,
    _empty_evidence_reply,
    _existing_appointment_validator,
    _patient_name_validator,
    _result_timing_validator,
    _service_normalizer,
    _service_slot_validator,
)

_ROOT = Path(__file__).resolve().parents[1]
# Настоящие функции — до autouse-подмен conftest (часть из них подменяется прямо в модуле).
_REAL = {
    (m.__name__, f): getattr(m, f)
    for m, f in (
        (_appointment_change_validator, "is_own_appointment_change"),
        (_existing_appointment_validator, "is_own_existing_appointment"),
        (_empty_evidence_reply, "classify_empty_evidence_turn"),
        (_patient_name_validator, "is_patient_name_reply"),
        (_result_timing_validator, "is_result_timing_question"),
        (_service_slot_validator, "is_service_name_reply"),
        (_service_normalizer, "llm_normalize_service_query"),
    )
}

# (модуль, функция, реплика в допустимом диапазоне длины)
_NOT_CACHING_FAILURE = [
    (_patient_name_validator, "is_patient_name_reply", "Иванова Мария Петровна"),
    (_service_slot_validator, "is_service_name_reply", "холтер"),
    (_result_timing_validator, "is_result_timing_question", "через сколько готов анализ"),
    (_service_normalizer, "llm_normalize_service_query", "сколько стоит анализ на сахар"),
]


@pytest.mark.parametrize(("module", "func", "text"), _NOT_CACHING_FAILURE, ids=lambda x: getattr(x, "__name__", str(x))[:30])
def test_llm_failure_is_not_cached(monkeypatch, module, func, text):
    calls: list[str] = []

    async def failing_generate(prompt, **_kwargs):
        calls.append(prompt)
        raise TimeoutError

    monkeypatch.setattr(module, "_CACHE", {})
    monkeypatch.setattr(module, "_ENABLED", True, raising=False)
    monkeypatch.setattr(module, "generate_text", failing_generate)
    real = _REAL[(module.__name__, func)]

    for _ in range(2):
        asyncio.run(real(text))

    assert len(calls) == 2, f"{module.__name__}: сбой закэширован — второй ход не спросил LLM"
    assert module._CACHE == {}


def _kill_switch_names() -> set[str]:
    names: set[str] = set()
    for path in (_ROOT / "messengers_router").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        names |= set(re.findall(r'getattr\(_cfg, "(MR_[A-Z_]+)"', text))
        names |= set(re.findall(r'_runtime_bool\("(MR_[A-Z_]+_ENABLED)"', text))
    return names


def test_every_llm_kill_switch_read_by_code_is_declared_in_config():
    names = _kill_switch_names()
    assert names, "не нашли ни одного выключателя — регэксп сломался"
    missing = sorted(n for n in names if not hasattr(app_config, n))
    assert missing == [], f"выключатели не объявлены в agent_logic_2/config.py (всегда ВКЛ): {missing}"


def test_generation_options_initialise_themselves(monkeypatch):
    # До первого операторского запроса кэш настроек пуст — бот ходил с умолчаниями ollama.
    monkeypatch.setattr(ollama_settings, "_cached_opts", {})
    monkeypatch.setattr(ollama_settings, "init_options", lambda: ollama_settings.__dict__.__setitem__("_cached_opts", {"temperature": 0.2}))

    assert ollama_settings.options_set().temperature == 0.2


@pytest.mark.parametrize(
    ("module", "func", "text"),
    [
        (_appointment_change_validator, "is_own_appointment_change", "хочу отменить, заболела"),
        (_existing_appointment_validator, "is_own_existing_appointment", "Мои записи"),
        (_empty_evidence_reply, "classify_empty_evidence_turn", "Электромиография"),
        (_patient_name_validator, "is_patient_name_reply", "Иванова Мария Петровна"),
        (_result_timing_validator, "is_result_timing_question", "через сколько готов анализ"),
        (_service_slot_validator, "is_service_name_reply", "холтер"),
    ],
    ids=lambda x: getattr(x, "__name__", str(x))[:30],
)
def test_short_decisions_use_explicit_deterministic_options(monkeypatch, module, func, text):
    seen: list[dict] = []

    async def fake_generate(prompt, **kwargs):
        seen.append(kwargs.get("options") or {})
        return "OTHER"

    monkeypatch.setattr(module, "_CACHE", {})
    monkeypatch.setattr(module, "_ENABLED", True, raising=False)
    monkeypatch.setattr(module, "generate_text", fake_generate)

    asyncio.run(_REAL[(module.__name__, func)](text))

    assert seen and seen[0] == llm_runtime.DECISION_OPTIONS
    assert llm_runtime.DECISION_OPTIONS["temperature"] == 0
