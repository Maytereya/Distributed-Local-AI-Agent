"""Ход подбора анализов не тратит решения LLM на суд догадки каталога (`test_assist_from_data`, BUG-2026-10-21-WALKIN-LLM-FREE-TEXT).

Класс (прод 24.10, деплой 17; стенд `f17_dev`): «есть ли исследование клеща на инцифалит?» — разбор реплики (1 решение), суд
нечёткой догадки каталога в подстановке (`router._inject_catalog_candidates`, два круга выбора строки — 2 решения) и тот же суд
ещё раз в заземлении (`ground_decision_entities`) — лимит хода (3 / 20 с) кончился до ответа хода, `ta_answer = failed`, оффер
вместо ответа по данным; 21 с на проде. Суд догадки решает, переспросить ли «Похоже, вы имели в виду …?», а под
`test_assist_from_data` строку выбирает LLM ответа хода среди всех кандидатов — как у цены, где переспроса нет вовсе.

Инвариант под on: на ходе TEST_ASSIST суд догадки не зовётся (решения хода не тратятся) и «да/нет» по ней не ставится, а сама
догадка остаётся кандидатом ответа хода (без неё «уровень железа в крови» терял «Железо в сыворотке» — поиск по реплике
нашёл только «CA 15-3 (молочная железа)»); другие метки (запись, адреса) и off — как было.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from messengers_router import entity_grounder, test_assist_answer
from messengers_router.services._booking_service_select import BookingServiceVerdict


class _Services:
    async def _ensure_service_catalog_rows_loaded(self) -> list[dict[str, Any]]:
        return []


def judge(monkeypatch, *, label: str, switch: str) -> tuple[dict[str, Any], set[str], list[str]]:
    """Один вызов суда нечёткой догадки; возвращает сущности, флаги и сколько раз зван выбор строки LLM."""

    from messengers_router.runtime_config import config as cfg

    calls: list[str] = []

    async def fake_judge(query: str, candidate: str, **_kwargs: Any) -> BookingServiceVerdict:
        calls.append(query)
        return BookingServiceVerdict(action="keep", status="picked")

    monkeypatch.setattr(cfg, "MR_TEST_ASSIST_FROM_DATA", switch, raising=False)
    monkeypatch.setattr(cfg, "MR_CATALOG_CONFIRM_LLM", "on", raising=False)
    monkeypatch.setattr(entity_grounder, "judge_booking_service", fake_judge)
    monkeypatch.setattr(entity_grounder, "names_service", lambda _q: True)
    out: dict[str, Any] = {}
    flags: set[str] = set()
    asyncio.run(
        entity_grounder.judge_fuzzy_service_candidate(
            label=label,
            candidate="Исследование клеща на зараженность инфекциями",
            query="исследование клеща на инцифалит",
            user_text="есть ли исследование клеща на инцифалит?",
            services=_Services(),
            runtime_llm_mode="hybrid",
            out=out,
            flags=flags,
            candidate_flag="catalog_service_fuzzy_candidate",
        )
    )
    return out, flags, calls


def test_on_test_assist_guess_is_not_judged_and_not_confirmed(monkeypatch):
    out, flags, calls = judge(monkeypatch, label="TEST_ASSIST", switch="on")

    assert calls == []  # решение хода не потрачено
    assert "_catalog_service_candidate" not in out  # «Похоже, вы имели в виду …?» не будет — строку выберет ответ хода
    assert "catalog_confirm_skipped_test_assist_data" in flags
    # догадка — кандидат ответа хода, а не переспрос
    assert out[test_assist_answer.CATALOG_GUESS_KEY] == "Исследование клеща на зараженность инфекциями"


def test_off_test_assist_guess_is_judged_as_before(monkeypatch):
    out, flags, calls = judge(monkeypatch, label="TEST_ASSIST", switch="off")

    assert calls == ["исследование клеща на инцифалит"]
    assert out.get("_catalog_service_candidate") == "Исследование клеща на зараженность инфекциями"
    assert "catalog_confirm_judged_picked" in flags
    assert test_assist_answer.CATALOG_GUESS_KEY not in out


@pytest.mark.parametrize("label", ["APPOINTMENT", "ADDRESS"])
def test_on_other_labels_are_judged_as_before(label, monkeypatch):
    out, flags, calls = judge(monkeypatch, label=label, switch="on")

    assert calls == ["исследование клеща на инцифалит"]
    assert "catalog_confirm_skipped_test_assist_data" not in flags
    assert test_assist_answer.CATALOG_GUESS_KEY not in out


def test_guess_key_survives_grounding_as_control_key():
    # заземление отбрасывает ключи не из белого списка метки — догадка до роутера дожить должна
    assert entity_grounder._is_allowed_key(test_assist_answer.CATALOG_GUESS_KEY, "TEST_ASSIST", None)
