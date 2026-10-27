"""Ответ на вопрос «Какой филиал вам удобен?» дословно из списка бота — принят (BUG-2026-10-18-BRANCH-OPTION-VERBATIM).

«запишите на консультацию гинеколога» → бот показал «г. Самара, ул. Ново-Садовая, 106, кор. 82» → пациент повторил это
дословно → бот снова спросил филиал: разбор адреса (`policies.extract_branch_hint`) ограничен 6 словами, а вариант — 7.
Ответ, совпадающий с показанным вариантом, — сам вариант (класс 5: бот знает, чего ждёт), разбор не нужен.

Инварианты класса: любой показанный вариант, повторённый дословно (с другой пунктуацией и регистром), — этот филиал;
через эндпоинт — каждый показанный филиал ведёт к вопросу о времени.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from messengers_router.policies import extract_branch_hint


_GYN = "Прием (осмотр, консультация) врача-акушера-гинеколога"


class _PickGyn:
    """LLM выбора строки (суд услуги записи) отмечает строку «Прием (осмотр, консультация) врача-акушера-гинеколога»; прочие места — «недоступна»."""

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        from ollama import GenerateResponse

        if "ответить пациенту на вопрос о цене" not in f"{system}\n\n{prompt}":
            raise ConnectionError("тест: LLM недоступна для этого места")
        picks = []
        for line in prompt.splitlines():
            head, dot, rest = line.partition(". ")
            if dot and head.isdigit() and rest.rsplit(" — ", 1)[0].strip() == _GYN:
                picks.append(int(head))
        answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks, "reason": "none", "search_terms": []}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _gyn_llm(monkeypatch) -> None:
    """Разметка хода — оракул с фразой «консультацию гинеколога», суд услуги записи — `_PickGyn` (`turn_spans_llm = on`,
    `price_by_parse = on`, прод с 24/26.10). Без LLM суд записи не выбирает строку (Ш4): одно название правил уходит на
    переспрос «Похоже, вы имели в виду «…»? да/нет» — пациент подтверждает догадку лишним ходом (нейтрально по принципам),
    и тест не дошёл бы до своего предмета."""

    from messengers_router import llm_runtime
    from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(("консультацию гинеколога",), inner=_PickGyn()))
    keep_llm_breaker_closed(monkeypatch)

OPENINGS = ("запишите на консультацию гинеколога", "хочу записаться к урологу", "запишите к терапевту")


def _turns(turns: tuple[str, ...], session_id: str) -> list[tuple[str, dict[str, Any]]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[tuple[str, dict[str, Any]]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append((str(body.get("text") or ""), dict(memory.get(session_id).last_entities or {})))
        return out

    return asyncio.run(run())


@pytest.mark.parametrize("opening", OPENINGS)
def test_every_shown_branch_said_verbatim_is_accepted(opening, monkeypatch):
    if "гинеколог" in opening:
        _gyn_llm(monkeypatch)  # услуга записи — строка прайса, её выбирает LLM; специальность («к урологу») — без LLM
    [(_, entities)] = _turns((opening,), f"verbatim-probe-{opening}")
    options = entities.get("appointment_branch_options") or []
    assert options, entities
    for i, option in enumerate(options):
        assert extract_branch_hint(option, entities) == option
        assert extract_branch_hint(option.upper().replace(",", " ,"), entities) == option
        _, (text, _) = _turns((opening, option), f"verbatim-{opening}-{i}")
        assert "Когда вам удобно прийти?" in text, (option, text)


def test_long_option_without_shown_list_falls_back_to_parsing():
    # Списка не показывали — прежний разбор (сверка с показанным только дополняет его).
    assert extract_branch_hint("ул. Ново-Садовая, 106", {}) == "Ново-Садовая 106"
