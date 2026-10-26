"""Класс: «Похоже, вы имели в виду …?» — нечёткую догадку каталога судит LLM (BUG-2026-10-22-CATALOG-CONFIRM-BY-LETTERS).

Симптом (стенд, деплой 13): переспрос предлагал чужую услугу по буквам — «Записаться на анализ» → «Анализ крови на
аминокислоты (48 показателей)» (23 раза на 4 прогона), «кровь на фадиатоп» → «Кровь на ВИЧ», «УЗИ суставов ног» → «УЗИ
периферических нервов»; «да» пациента ведёт к чужой услуге и чужой цене.

Инварианты (выключатель `catalog_confirm_llm` снят 29.10, путь `on` — единственный; тесты пути off удалены):
- LLM подтвердила — переспрос с догадкой; назвала другую — переспрос с её названием; «не услуга» / «ничего не подходит» /
  «слишком общее» — переспроса нет; LLM недоступна — как было;
- у цены переспроса нет (строку выбирает LLM) — догадку не судим; у подбора анализов (TEST_ASSIST) — тоже: под
  `test_assist_from_data = on` (прод с 24.10) догадка — кандидат ответа подбора, суд пропущен (`test_test_assist_decisions_class`).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router import entity_grounder as G
from messengers_router.services._booking_service_select import BookingServiceVerdict


class _Services:
    async def _ensure_service_catalog_rows_loaded(self) -> list[dict]:
        return []


def _run(label: str, verdict: BookingServiceVerdict | None, monkeypatch) -> tuple[dict, set, list]:
    calls: list[tuple] = []

    async def fake_judge(phrase, canonical, **kw):
        calls.append((phrase, canonical))
        return verdict

    monkeypatch.setattr(G, "judge_booking_service", fake_judge)
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


@pytest.mark.parametrize("label", ["APPOINTMENT", "ADDRESS"])
@pytest.mark.parametrize("status", ["none_fit", "not_a_service"])
def test_on_llm_refuses_no_confirm(monkeypatch, label: str, status: str) -> None:
    out, flags, calls = _run(label, BookingServiceVerdict("refuse", status=status), monkeypatch)
    assert "_catalog_service_candidate" not in out
    assert calls and f"catalog_confirm_judged_{status}" in flags


def test_on_too_general_no_yes_no_confirm(monkeypatch) -> None:
    verdict = BookingServiceVerdict("clarify", names=("Общий анализ крови", "Глюкоза"), status="too_general")
    out, _, _ = _run("APPOINTMENT", verdict, monkeypatch)
    assert "_catalog_service_candidate" not in out


def test_on_llm_names_other_service_confirm_with_it(monkeypatch) -> None:
    verdict = BookingServiceVerdict("replace", "Общий анализ крови", status="picked")
    out, _, _ = _run("APPOINTMENT", verdict, monkeypatch)
    assert out["_catalog_service_candidate"] == "Общий анализ крови"
    assert out["_catalog_service_query"] == "анализ"


def test_on_llm_confirms_guess(monkeypatch) -> None:
    verdict = BookingServiceVerdict("keep", "Анализ крови на аминокислоты (48 показателей)", status="picked")
    out, _, _ = _run("APPOINTMENT", verdict, monkeypatch)
    assert out["_catalog_service_candidate"].startswith("Анализ крови на аминокислоты")


def test_on_llm_unavailable_as_before(monkeypatch) -> None:
    verdict = BookingServiceVerdict("choose", names=("A", "B"), status="unavailable")
    out, _, _ = _run("ADDRESS", verdict, monkeypatch)
    assert out["_catalog_service_candidate"].startswith("Анализ крови на аминокислоты")


def test_on_price_label_not_judged(monkeypatch) -> None:
    out, _, calls = _run("PRICE", BookingServiceVerdict("refuse", status="none_fit"), monkeypatch)
    assert calls == []
    assert "_catalog_service_candidate" in out


def test_on_budget_exhausted_after_judgement_no_letter_confirm(monkeypatch) -> None:
    """Вторая фраза хода после суда первой: «недоступна» = лимит решений хода — догадку по буквам не выносим (g0196)."""

    async def fake_judge(phrase, canonical, **kw):
        return BookingServiceVerdict("choose", names=(canonical,), status="unavailable")

    monkeypatch.setattr(G, "judge_booking_service", fake_judge)
    out: dict = {}
    flags: set = {"catalog_confirm_judged_none_fit"}
    asyncio.run(
        G.judge_fuzzy_service_candidate(
            label="APPOINTMENT",  # у TEST_ASSIST суд пропущен (`test_assist_from_data`) — механизм лимита судим на записи
            candidate="Вирус клещевого энцефалита Ig G",
            query="инцифалит",
            user_text="есть ли исследование клеща на инцифалит?",
            services=_Services(),
            runtime_llm_mode="hybrid",
            out=out,
            flags=flags,
            candidate_flag="entity_catalog_service_fuzzy_candidate",
        )
    )
    assert "_catalog_service_candidate" not in out


def test_on_budget_exhausted_after_router_judgement_no_letter_confirm(monkeypatch) -> None:
    """Суд роутера (флаг в решении хода) до заземления — тоже «LLM в этом ходе была доступна»."""

    async def fake_judge(phrase, canonical, **kw):
        return BookingServiceVerdict("choose", names=(canonical,), status="unavailable")

    monkeypatch.setattr(G, "judge_booking_service", fake_judge)
    out: dict = {}
    asyncio.run(
        G.judge_fuzzy_service_candidate(
            label="APPOINTMENT",  # у TEST_ASSIST суд пропущен (`test_assist_from_data`) — механизм лимита судим на записи
            candidate="Вирус клещевого энцефалита Ig G",
            query="инцифалит",
            user_text="есть ли исследование клеща на инцифалит?",
            services=_Services(),
            runtime_llm_mode="hybrid",
            out=out,
            flags=set(),
            candidate_flag="entity_catalog_service_fuzzy_candidate",
            prior_flags={"catalog_confirm_judged_none_fit"},
        )
    )
    assert "_catalog_service_candidate" not in out


def test_on_budget_exhausted_by_other_decisions_no_letter_confirm(monkeypatch) -> None:
    """Лимит решений хода потратили другие суды (не переспроса) — «недоступна» не возвращает переспрос по буквам (g0287)."""

    from messengers_router import llm_runtime

    async def fake_judge(phrase, canonical, **kw):
        return BookingServiceVerdict("choose", names=(canonical,), status="unavailable")

    monkeypatch.setattr(G, "judge_booking_service", fake_judge)
    monkeypatch.setattr(llm_runtime, "turn_budget_exhausted", lambda: True)
    out: dict = {}
    asyncio.run(
        G.judge_fuzzy_service_candidate(
            label="APPOINTMENT",  # у TEST_ASSIST суд пропущен (`test_assist_from_data`) — механизм лимита судим на записи
            candidate="Анализ крови на аминокислоты (48 показателей)",
            query="анализ",
            user_text="Я смогу принести кал на анализ в 15.00?",
            services=_Services(),
            runtime_llm_mode="hybrid",
            out=out,
            flags=set(),
            candidate_flag="entity_catalog_service_fuzzy_candidate",
        )
    )
    assert "_catalog_service_candidate" not in out


def _run_query(query: str, verdict: BookingServiceVerdict, monkeypatch, *, label: str = "ADDRESS") -> tuple[dict, set, list]:
    calls: list[tuple] = []

    async def fake_judge(phrase, canonical, **kw):
        calls.append((phrase, canonical))
        return verdict

    monkeypatch.setattr(G, "judge_booking_service", fake_judge)
    out: dict = {}
    flags: set = set()
    asyncio.run(
        G.judge_fuzzy_service_candidate(
            label=label,
            candidate="Анализ крови на аминокислоты (48 показателей)",
            query=query,
            user_text="Анализ T-SPOT",
            services=_Services(),
            runtime_llm_mode="hybrid",
            out=out,
            flags=flags,
            candidate_flag="entity_catalog_service_fuzzy_candidate",
        )
    )
    return out, flags, calls


@pytest.mark.parametrize("label", ["APPOINTMENT", "ADDRESS"])
def test_on_phrase_names_no_service_no_letter_confirm(monkeypatch, label: str) -> None:
    """Фраза догадки не называет услугу («анализы» правила walk-in): выбор строки LLM не зовёт, «оставить» без решения —
    не подтверждение; переспроса по буквам нет (прод 22.10 после деплоя 15: «кровь ребенку 9 месяцев», «Анализ T-SPOT» →
    «Анализ крови на аминокислоты?» — нормализатор услуги переписал «анализы» в эту строку, ответ закэширован в процессе)."""

    out, flags, calls = _run_query("анализы", BookingServiceVerdict("keep", "Анализ крови на аминокислоты (48 показателей)"), monkeypatch, label=label)
    assert "_catalog_service_candidate" not in out
    assert calls == []
    assert "catalog_confirm_judged_not_a_service" in flags


@pytest.mark.parametrize("query", ["анализ", "T-SPOT", "фадиатоп", "УЗИ суставов ног"])
def test_on_phrase_names_service_still_judged(monkeypatch, query: str) -> None:
    """Встречный случай: фраза называет услугу — суд LLM как был (подтвердила — переспрос)."""

    verdict = BookingServiceVerdict("keep", "Анализ крови на аминокислоты (48 показателей)", status="picked")
    out, _, calls = _run_query(query, verdict, monkeypatch)
    assert calls
    assert out["_catalog_service_candidate"].startswith("Анализ крови на аминокислоты")
