"""BUG-D: «срок готовности» анализа — бот должен называть срок (поле `deadline`).

Корень (воспроизведён офлайн): запрос вида «срок готовности ОАК» / «сколько делается
ферритин» НЕ извлекал `test_name`, поэтому в матчер уходил ВЕСЬ текст с шумом
(«срок готовности»), а шумовые токены ломали лексический матч → `test_assist`
возвращал «no matches» → бот уходил в clarify/мисроутинг (BUG-D/BUG-C). При этом
`deadline` УЖЕ лежит в строках каталога, а renderer-промпт УЖЕ инструктирует
показывать «сроки готовности … буквальные значения из Данные».

Инвариант (класс): turnaround-фразировка («срок готовности», «сколько делается»,
«когда будет готов», «за сколько дней», «как долго делается») — это ШУМ вокруг
названия анализа; её надо снять перед матчингом, чтобы анализ нашёлся и его
`deadline` дошёл до пациента. Критерий по КЛАССУ фраз, не по инстансу.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services._prices_helpers import strip_readiness_phrasing


def run(coro):
    return asyncio.run(coro)


# turnaround-запросы: фраза снимается, остаётся название анализа, флаг=True
READINESS_CASES = [
    ("срок готовности ОАК", "оак"),
    ("сколько делается ферритин", "ферритин"),
    ("когда будет готов витамин B1", "витамин b1"),
    ("за сколько дней делается гемоглобин", "гемоглобин"),
    ("как долго делается ТТГ", "ттг"),
    ("сроки готовности анализа ферритин", "ферритин"),
    ("за сколько дней делается анализ на гемоглобин", "гемоглобин"),
]

# обычные запросы: НЕ readiness, текст не меняется
PLAIN_CASES = [
    "общий анализ крови",
    "ферритин",
    "как подготовиться к ФГДС",   # PREPARE ≠ срок готовности
    "результаты готовы?",          # TEST_RESULT (result-lookup) ≠ turnaround
]


@pytest.mark.parametrize("raw,expected", READINESS_CASES, ids=[c[0] for c in READINESS_CASES])
def test_strip_readiness_phrasing_extracts_analysis_name(raw, expected):
    cleaned, is_readiness = strip_readiness_phrasing(raw)
    assert is_readiness is True, f"{raw!r} not flagged readiness"
    assert cleaned.strip().lower() == expected, f"{raw!r} -> {cleaned!r}, expected {expected!r}"


@pytest.mark.parametrize("raw", PLAIN_CASES, ids=PLAIN_CASES)
def test_strip_readiness_phrasing_leaves_plain_queries_unchanged(raw):
    cleaned, is_readiness = strip_readiness_phrasing(raw)
    assert is_readiness is False, f"{raw!r} wrongly flagged readiness"
    assert cleaned.strip().lower() == raw.strip().lower(), f"{raw!r} wrongly modified -> {cleaned!r}"


@pytest.mark.parametrize(
    "query,name_fragments",
    [
        ("срок готовности ОАК", ("оак", "общий анализ крови")),
        ("сколько делается ферритин", ("ферритин",)),
    ],
    ids=["readiness_oak", "readiness_ferritin"],
)
def test_test_assist_readiness_query_matches_analysis_with_deadline(query, name_fragments):
    """e2e на реальном каталоге: turnaround-запрос находит анализ И несёт deadline.

    До фикса `test_assist` отдавал «no matches» (шум ломал матч).
    """
    svc = Services()
    payload = run(svc.test_assist(query, {}))
    tests = payload.get("tests") or []
    assert tests, f"{query!r}: no matches — BUG-D not fixed (note={payload.get('note')!r})"
    assert any(str(r.get("deadline") or "").strip() for r in tests), f"{query!r}: deadline missing in payload"
    assert any(
        any(frag in str(r.get("serviceName") or "").lower() for frag in name_fragments) for r in tests
    ), f"{query!r}: matched rows unrelated to requested analysis"


def test_test_assist_plain_query_unaffected_regression():
    """Регресс-гард: обычный запрос без turnaround-шума по-прежнему матчится."""
    svc = Services()
    payload = run(svc.test_assist("ферритин", {}))
    tests = payload.get("tests") or []
    assert tests, "plain 'ферритин' lost matches"
    assert any("ферритин" in str(r.get("serviceName") or "").lower() for r in tests)
