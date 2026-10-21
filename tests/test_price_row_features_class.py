"""Класс: признаки строки прайса считаются один раз и не меняют оценку (STEP-2026-10-22-E-PRICE-ROW-FEATURES).

Ранжирование прайса оценивает каждую строку среза на каждый запрос. До 22.10 оно на каждом
запросе заново нормализовало название строки и флаги условий — 2–8 с на ход цены вне LLM.
Теперь признаки строки (`_row_features`) кэшируются по тексту названия и кода, а признаки
запроса (`_query_score_features`) считаются один раз на ранжирование.

Инварианты:
1. Оценка строки с готовыми признаками запроса = оценке без них, на всём срезе прайса.
2. Признаки из кэша = признакам, посчитанным заново (кэш не хранит устаревшего).
3. Наборы флагов, которые отдают `_lab_price_variant_flags` / `_row_nonbase_price_flags`,
   можно менять — это не портит кэш для следующего вызова.
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services import _prices_helpers as H

_QUERIES = (
    "сколько стоит ттг",
    "общий т3",
    "узи печени ребенку",
    "стоимость приема уролога",
    "повторный прием кардиолога",
    "анализ крови на витамин д",
    "ферритин cito",
    "выезд на дом взятие крови",
    "антитела к кори",
    "генетический анализ",
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    data = [r for r in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]
    assert len(data) >= 100, "нужен срез прайса (≥100 строк) — гард на маленьком прайсе ничего не доказывает"
    return data


def test_score_with_precomputed_query_features_equals_plain(rows: list[dict]) -> None:
    for text in _QUERIES:
        tokens = H._price_query_tokens(text)
        query = " ".join(tokens) or H._normalise_input(text)
        homecode_query = H._extract_homecode_query(text)
        qf = H._query_score_features(query)
        for row in rows:
            plain = H._price_row_score(row, query=query, tokens=tokens, homecode_query=homecode_query)
            fast = H._price_row_score(row, query=query, tokens=tokens, homecode_query=homecode_query, qf=qf)
            assert plain == fast, (text, row.get("serviceName"))


def test_cached_row_features_equal_fresh(rows: list[dict]) -> None:
    cached = [H._row_features(r) for r in rows]
    H._row_features_cached.cache_clear()
    fresh = [H._row_features(r) for r in rows]
    assert cached == fresh


def test_foreign_query_features_are_not_reused(rows: list[dict]) -> None:
    """`qf` от другого запроса не подменяет признаки этого (защита от ошибки вызывающего)."""

    text = "стоимость приема уролога"
    tokens = H._price_query_tokens(text)
    query = " ".join(tokens)
    alien = H._query_score_features("ферритин")
    for row in rows[:500]:
        assert H._price_row_score(row, query=query, tokens=tokens, homecode_query="", qf=alien) == H._price_row_score(
            row, query=query, tokens=tokens, homecode_query=""
        )


def test_returned_flag_sets_are_private_copies(rows: list[dict]) -> None:
    row = next(r for r in rows if H._row_features(r).nonbase_flags)
    before = H._row_nonbase_price_flags(row)
    H._row_nonbase_price_flags(row).add("mutated")
    H._lab_price_variant_flags(row).add("mutated")
    assert H._row_nonbase_price_flags(row) == before
    assert "mutated" not in H._lab_price_variant_flags(row)


def test_ranking_is_unchanged_by_cache_state(rows: list[dict]) -> None:
    for text in _QUERIES:
        warm = [r.get("serviceName") for r in H._rank_price_rows(rows, text, limit=10)]
        H._row_features_cached.cache_clear()
        cold = [r.get("serviceName") for r in H._rank_price_rows(rows, text, limit=10)]
        assert warm == cold, text
