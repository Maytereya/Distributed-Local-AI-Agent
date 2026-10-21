"""Класс: корни текста подготовки кэшируются и не меняют подбор (STEP-2026-10-22-E-PREPARE-ROOTS).

Подбор памятки подготовки сравнивает вопрос со всеми записями справочника и до 22.10 на
каждом вопросе заново выделял корни из текста каждой памятки (PREPARE вне LLM p95 4,5 с).

Инварианты:
1. Корни из кэша = корням, посчитанным заново, на текстах всего справочника.
2. Наружу — копия: изменение результата не портит кэш.
3. Оценка релевантности не зависит от того, тёплый ли кэш.
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_service_info
from messengers_router.services import _prepare as P


@pytest.fixture(scope="module")
def texts() -> list[str]:
    out: list[str] = []
    for row in api_service_info.load_service_info() or []:
        if not isinstance(row, dict):
            continue
        for key in ("serviceName", "name", "preparation", "patientPreparation", "description"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
    assert len(out) >= 100, "нужен срез справочника МИС (≥100 текстов)"
    return out[:3000]


def test_cached_roots_equal_fresh(texts: list[str]) -> None:
    warm = [P._prepare_term_roots(t) for t in texts]
    P._prepare_term_roots_cached.cache_clear()
    cold = [P._prepare_term_roots(t) for t in texts]
    assert warm == cold


def test_returned_roots_are_private_copies(texts: list[str]) -> None:
    text = next(t for t in texts if P._prepare_term_roots(t))
    before = P._prepare_term_roots(text)
    P._prepare_term_roots(text).add("mutated")
    assert P._prepare_term_roots(text) == before


@pytest.mark.parametrize("query", ["подготовка к анализу на сахар", "как подготовиться к узи брюшной полости", "фгдс натощак"])
def test_relevance_is_unchanged_by_cache_state(texts: list[str], query: str) -> None:
    sample = texts[:400]
    warm = [P._prepare_fast_relevance_score(query, t, title=t[:60]) for t in sample]
    P._prepare_term_roots_cached.cache_clear()
    cold = [P._prepare_fast_relevance_score(query, t, title=t[:60]) for t in sample]
    assert warm == cold
