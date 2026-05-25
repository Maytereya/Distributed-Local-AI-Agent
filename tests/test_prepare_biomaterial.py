import asyncio

import pytest

from messengers_router import services as svc_mod
from messengers_router.services import Services
from messengers_router.services.prepare import _prepare_biomaterial


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("во сколько прийти сдать кровь", "blood"),
        ("общий анализ крови", "blood"),
        ("общий анализ мочи", "urine"),
        ("сдать мочу", "urine"),
        ("кал на копрологию", "feces"),
        ("ттг", None),
        ("подготовка к узи", None),
    ],
)
def test_prepare_biomaterial_detection(text, expected):
    assert _prepare_biomaterial(text) == expected


def _patch_meili(monkeypatch, seen):
    def fake_search(_index, query, *args, **kwargs):
        seen.append(query)
        return "Совпадений не найдено, cформулируйте запрос иначе"

    monkeypatch.setattr(svc_mod.meilisearch, "search_meili", fake_search)
    monkeypatch.setattr(svc_mod.html_cleaner, "strip_html", lambda s: s)


def test_prepare_drops_conflicting_stale_biomaterial(monkeypatch):
    """«сдать кровь» при stale service_name=«общий анализ мочи» не должен
    тянуть мочу в варианты подготовки (регрессия N5)."""
    svc = Services()

    async def no_api_candidates(*_a, **_k):
        return []

    monkeypatch.setattr(svc, "_prepare_candidates_from_analysis_api_cache", no_api_candidates)
    seen: list[str] = []
    _patch_meili(monkeypatch, seen)

    # «подготовка к сдаче крови» (не вопрос про время) идёт обычным prepare-путём,
    # где и работает дроп конфликтного stale-биоматериала.
    res = run(svc.test_prepare("подготовка к сдаче крови", {"service_name": "общий анализ мочи"}))

    assert not any("моч" in q.lower() for q in seen), seen
    assert any("кров" in q.lower() for q in seen), seen
    assert "prepare" in str(res.get("note") or "")


def test_prepare_keeps_consistent_stale_biomaterial(monkeypatch):
    """Совместимый stale (кровь) сохраняется — это не конфликт."""
    svc = Services()

    async def no_api_candidates(*_a, **_k):
        return []

    monkeypatch.setattr(svc, "_prepare_candidates_from_analysis_api_cache", no_api_candidates)
    seen: list[str] = []
    _patch_meili(monkeypatch, seen)

    run(svc.test_prepare("подготовка к сдаче крови", {"service_name": "общий анализ крови"}))

    assert any("общий анализ крови" in q.lower() for q in seen), seen
