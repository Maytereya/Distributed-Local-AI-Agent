"""Класс: «Похоже, вы имели в виду …?» — нечёткую догадку каталога судит LLM (BUG-2026-10-22-CATALOG-CONFIRM-BY-LETTERS).

Симптом (стенд, деплой 13): переспрос предлагал чужую услугу по буквам — «Записаться на анализ» → «Анализ крови на
аминокислоты (48 показателей)» (23 раза на 4 прогона), «кровь на фадиатоп» → «Кровь на ВИЧ», «УЗИ суставов ног» → «УЗИ
периферических нервов»; «да» пациента ведёт к чужой услуге и чужой цене.

Инварианты (выключатель `catalog_confirm_llm`):
- off — как было: догадка правил на переспрос, суд LLM не зовётся;
- on: LLM подтвердила — переспрос с догадкой; назвала другую — переспрос с её названием; «не услуга» / «ничего не подходит» /
  «слишком общее» — переспроса нет; LLM недоступна — как было;
- у цены переспроса нет и под on (строку выбирает LLM) — догадку не судим.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_logic_2 import config as cfg
from messengers_router import entity_grounder as G
from messengers_router.services._booking_service_select import BookingServiceVerdict


class _Services:
    async def _ensure_service_catalog_rows_loaded(self) -> list[dict]:
        return []


def _run(label: str, verdict: BookingServiceVerdict | None, monkeypatch, *, mode: str) -> tuple[dict, set, list]:
    calls: list[tuple] = []

    async def fake_judge(phrase, canonical, **kw):
        calls.append((phrase, canonical))
        return verdict

    monkeypatch.setattr(G, "judge_booking_service", fake_judge)
    monkeypatch.setattr(cfg, "MR_CATALOG_CONFIRM_LLM", mode, raising=False)
    out: dict = {}
    flags: set = set()
    asyncio.run(
        G.judge_fuzzy_service_candidate(
            label=label,
            candidate="Анализ крови на аминокислоты (48 показателей)",
            query="анализ",
            user_text="Записаться на анализ",
            services=_Services(),
            runtime_llm_mode="hybrid",
            out=out,
            flags=flags,
            candidate_flag="entity_catalog_service_fuzzy_candidate",
        )
    )
    return out, flags, calls


def test_off_keeps_letter_guess_without_llm(monkeypatch) -> None:
    out, _, calls = _run("APPOINTMENT", BookingServiceVerdict("refuse", status="none_fit"), monkeypatch, mode="off")
    assert out["_catalog_service_candidate"].startswith("Анализ крови на аминокислоты")
    assert calls == []


@pytest.mark.parametrize("label", ["APPOINTMENT", "ADDRESS", "TEST_ASSIST"])
@pytest.mark.parametrize("status", ["none_fit", "not_a_service"])
def test_on_llm_refuses_no_confirm(monkeypatch, label: str, status: str) -> None:
    out, flags, calls = _run(label, BookingServiceVerdict("refuse", status=status), monkeypatch, mode="on")
    assert "_catalog_service_candidate" not in out
    assert calls and f"catalog_confirm_judged_{status}" in flags


def test_on_too_general_no_yes_no_confirm(monkeypatch) -> None:
    verdict = BookingServiceVerdict("clarify", names=("Общий анализ крови", "Глюкоза"), status="too_general")
    out, _, _ = _run("TEST_ASSIST", verdict, monkeypatch, mode="on")
    assert "_catalog_service_candidate" not in out


def test_on_llm_names_other_service_confirm_with_it(monkeypatch) -> None:
    verdict = BookingServiceVerdict("replace", "Общий анализ крови", status="picked")
    out, _, _ = _run("APPOINTMENT", verdict, monkeypatch, mode="on")
    assert out["_catalog_service_candidate"] == "Общий анализ крови"
    assert out["_catalog_service_query"] == "анализ"


def test_on_llm_confirms_guess(monkeypatch) -> None:
    verdict = BookingServiceVerdict("keep", "Анализ крови на аминокислоты (48 показателей)", status="picked")
    out, _, _ = _run("APPOINTMENT", verdict, monkeypatch, mode="on")
    assert out["_catalog_service_candidate"].startswith("Анализ крови на аминокислоты")


def test_on_llm_unavailable_as_before(monkeypatch) -> None:
    verdict = BookingServiceVerdict("choose", names=("A", "B"), status="unavailable")
    out, _, _ = _run("ADDRESS", verdict, monkeypatch, mode="on")
    assert out["_catalog_service_candidate"].startswith("Анализ крови на аминокислоты")


def test_on_price_label_not_judged(monkeypatch) -> None:
    out, _, calls = _run("PRICE", BookingServiceVerdict("refuse", status="none_fit"), monkeypatch, mode="on")
    assert calls == []
    assert "_catalog_service_candidate" in out
