import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services import prepare as prepare_mod
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


def _spy_prepare_inputs(monkeypatch, svc) -> list[dict]:
    """Что подготовка передала дальше: сущности для поиска памяток МИС и вопрос для LLM."""

    seen: list[dict] = []

    async def candidates(query, entities):
        seen.append({"entities": dict(entities)})
        return []

    async def select(question, memos, **kwargs):
        seen.append({"question": question})
        return ()

    monkeypatch.setattr(svc, "_prepare_candidates_from_analysis_api_cache", candidates)
    monkeypatch.setattr(prepare_mod, "select_prepare_memos", select)
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: ())
    return seen


def _entity_texts(seen: list[dict]) -> str:
    return " ".join(
        str(rec["entities"].get(key) or "") for rec in seen if "entities" in rec for key in ("test_name", "service_name")
    ).lower()


def test_prepare_drops_conflicting_stale_biomaterial(monkeypatch):
    """«сдать кровь» при stale service_name=«общий анализ мочи» не должен
    тянуть мочу в поиск подготовки (регрессия N5)."""
    svc = Services()
    seen = _spy_prepare_inputs(monkeypatch, svc)

    # «подготовка к сдаче крови» (не вопрос про время) идёт обычным prepare-путём,
    # где и работает дроп конфликтного stale-биоматериала.
    res = run(svc.test_prepare("подготовка к сдаче крови", {"service_name": "общий анализ мочи"}))

    assert any("entities" in rec for rec in seen), "поиск памяток не дошёл до МИС"
    assert "моч" not in _entity_texts(seen), seen
    assert all("моч" not in rec.get("question", "") for rec in seen), seen
    assert "prepare" in str(res.get("note") or "")


def test_prepare_keeps_consistent_stale_biomaterial(monkeypatch):
    """Совместимый stale (кровь) сохраняется — это не конфликт."""
    svc = Services()
    seen = _spy_prepare_inputs(monkeypatch, svc)

    run(svc.test_prepare("подготовка к сдаче крови", {"service_name": "общий анализ крови"}))

    assert "общий анализ крови" in _entity_texts(seen), seen
