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


def _spy_subjects(monkeypatch) -> list[tuple]:
    """Предметы подготовки, которые путь прода (`prepare_by_parse = on`, с 24.10) взял у хода: без разбора (вход — не реплика
    хода) предмет — услуга сущностей (`_subjects_of_turn`, источник `entity`); её и проверяет снятие stale-биоматериала."""

    seen: list[tuple] = []
    real = prepare_mod._subjects_of_turn

    def spy(raw_query, entities):
        out = real(raw_query, entities)
        seen.append(out)
        return out

    async def select(question, memos, **kwargs):
        return ()

    monkeypatch.setattr(prepare_mod, "_subjects_of_turn", spy)
    monkeypatch.setattr(prepare_mod, "select_prepare_memos_by_parse", select)
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: ())
    monkeypatch.setattr(prepare_mod.api_service_info, "load_service_info", lambda: [])
    return seen


def _subject_texts(seen: list[tuple]) -> str:
    return " ".join(phrase for subjects, _source in seen for phrase, _rows in subjects).lower()


def test_prepare_drops_conflicting_stale_biomaterial(monkeypatch):
    """«сдать кровь» при stale service_name=«общий анализ мочи» не должен
    тянуть мочу в поиск подготовки (регрессия N5)."""
    svc = Services()
    seen = _spy_subjects(monkeypatch)

    # «подготовка к сдаче крови» (не вопрос про время) идёт обычным prepare-путём,
    # где и работает дроп конфликтного stale-биоматериала.
    res = run(svc.test_prepare("подготовка к сдаче крови", {"service_name": "общий анализ мочи"}))

    assert seen, "подготовка не дошла до предметов"
    assert "моч" not in _subject_texts(seen), seen
    assert "prepare" in str(res.get("note") or "")


def test_prepare_keeps_consistent_stale_biomaterial(monkeypatch):
    """Совместимый stale (кровь) сохраняется — это не конфликт."""
    svc = Services()
    seen = _spy_subjects(monkeypatch)

    run(svc.test_prepare("подготовка к сдаче крови", {"service_name": "общий анализ крови"}))

    assert "общий анализ крови" in _subject_texts(seen), seen
