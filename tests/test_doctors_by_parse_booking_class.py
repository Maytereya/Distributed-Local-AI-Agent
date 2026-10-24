"""Запись «по врачу» под `doctors_by_parse` — через эндпоинт (консилиум 25.10, МЕГА п. 1; STEP-2026-10-24-DOCTORS-BY-PARSE).

«запишите на <услугу>» → бот спрашивает филиал («…Или написать список врачей?») → «список врачей» → `doctors_info` с услугой
записи из контекста. Инварианты класса:
- на шаге «список врачей» реплика — выбор пункта, а не название услуги: её разбор не нужен, услуга — из записи; если это
  название строки МИС, врачи — по данным, **без вызовов LLM** (раньше путь `on` тратил +1 решение на разметку слов «список врачей»
  и ещё +1 на выбор — ходы записи упирались в лимит 3 решений вместе с `pending_answer` и `booking_specialty_llm`);
- какую услугу записи имел в виду пациент, данные и LLM не решили (LLM недоступна) — честное «уточните» с оффером, а не тихий
  уход к выбору филиала (`build_appointment_step_response` раньше при пустом списке молча возвращал к филиалам);
- под `off` — как было.

Мир — гейт (локальные срезы врачей, цен врачей, филиалов), как у `test_appointment_branch_reply_class`; LLM недоступна, тест
считает, какие места к ней обращались.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from messengers_router import llm_runtime
from messengers_router.response_builder import DOCTORS_SERVICE_CLARIFY_BOOKING_OFFER

_BRANCH_QUESTION = "Какой филиал вам удобен?"
_DOCTOR_LIST = "список врачей"
_KNOWN = "запишите на удаление серной пробки"  # строка МИС, её делают врачи Самары
# Названия записи нет у врачей Самары (код услуги — только у врачей другого города): что имел в виду пациент, решает выбор LLM.
_NOT_IN_DATA = "запишите на Ультразвуковое исследование щитовидной железы"


class _DownLLM:
    """LLM недоступна; записывает, какие места к ней шли (разметка слов, выбор услуги врачей)."""

    def __init__(self) -> None:
        self.sites: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs: Any):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if "Слова сообщения с номерами:" in seen:
            self.sites.append("turn_spans")
        elif "какие врачи делают услугу" in seen:
            self.sites.append("choose_doctors_service")
        raise ConnectionError("тест: LLM недоступна")


@pytest.fixture
def down_llm(monkeypatch) -> _DownLLM:
    llm = _DownLLM()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    return llm


def _switch(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_DOCTORS_BY_PARSE", mode, raising=False)


def _dialog(turns: tuple[str, ...], session_id: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append({"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff")), "entities": dict(memory.get(session_id).last_entities or {})})
        return out

    return asyncio.run(run())


@pytest.mark.parametrize("mode", ["off", "on"])
def test_doctor_list_of_a_booked_service_comes_from_data_without_llm(monkeypatch, down_llm, mode) -> None:
    _switch(monkeypatch, mode)
    first, second = _dialog((_KNOWN, _DOCTOR_LIST), f"dbp-booking-known-{mode}")

    assert _BRANCH_QUESTION in first["text"], first["text"]
    assert second["text"].startswith("1. "), second["text"]  # список врачей, а не снова филиалы
    assert not second["handoff"]
    if mode == "on":
        assert down_llm.sites == [], down_llm.sites  # ни разметки «список врачей», ни выбора: услуга записи — строка МИС


def test_booked_service_not_decided_asks_to_clarify_instead_of_silent_branches(monkeypatch, down_llm) -> None:
    _switch(monkeypatch, "on")
    first, second = _dialog((_NOT_IN_DATA, _DOCTOR_LIST), "dbp-booking-clarify")

    assert _BRANCH_QUESTION in first["text"], first["text"]
    assert second["text"] == DOCTORS_SERVICE_CLARIFY_BOOKING_OFFER, second["text"]
    assert _BRANCH_QUESTION not in second["text"]
    assert second["entities"].get("_operator_offer_pending") is True
    assert down_llm.sites == ["choose_doctors_service"], down_llm.sites  # реплику «список врачей» не размечали
