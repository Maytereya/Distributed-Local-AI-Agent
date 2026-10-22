"""Класс: индекс корней подготовки и кэш разбора специальности не меняют ответа (ход (е), STEP-2026-10-22-E-SPECIALTY-PREPARE-CACHE).

Инварианты: `_prepare_roots_match_indexed` = перебору `_prepare_roots_match` на корнях текстов справочника МИС × корнях
вопросов; кэш `extract_specialties_from_text` и `_is_prepare_content_actionable` — те же ответы с холодным и тёплым кэшем.
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_price, api_service_info
from messengers_router import specialty_parser as SP
from messengers_router.services import _prepare as P
from messengers_router.services import _prices_helpers as H


@pytest.fixture(scope="module")
def root_sets() -> list[frozenset[str]]:
    out = []
    for row in (api_service_info.load_service_info() or [])[:1500]:
        if isinstance(row, dict):
            for key in ("serviceName", "preparation", "patientPreparation"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    out.append(P._prepare_term_roots_cached(value))
    assert len(out) >= 100, "нужен срез справочника МИС"
    return out


_QUERIES = ("подготовка к анализу на сахар", "колоноскопия под наркозом", "узи брюшной полости натощак", "гастроскопия", "кровь из вены")


def test_indexed_match_equals_scan(root_sets) -> None:
    roots = {r for q in _QUERIES for r in P._prepare_term_roots(q)}
    roots |= {r for s in root_sets[:200] for r in list(s)[:3]}  # корни самих памяток — в том числе совпадения префиксом
    for cand in root_sets:
        for q in roots:
            assert P._prepare_roots_match_indexed(q, cand) == P._prepare_roots_match(q, set(cand)), (q, sorted(cand)[:5])


def test_specialty_and_actionable_cache_state_does_not_matter() -> None:
    names = [str(r.get("serviceName") or "") for r in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID)[:1500] if isinstance(r, dict)]
    assert len(names) >= 100
    warm = [SP.extract_specialties_from_text(n) for n in names]
    SP.extract_specialties_from_text.cache_clear()
    assert warm == [SP.extract_specialties_from_text(n) for n in names]
    texts = [str(r.get("preparation") or r.get("serviceName") or "") for r in (api_service_info.load_service_info() or [])[:800] if isinstance(r, dict)]
    warm_a = [P._is_prepare_content_actionable(t) for t in texts]
    P._is_prepare_content_actionable.cache_clear()
    assert warm_a == [P._is_prepare_content_actionable(t) for t in texts]
