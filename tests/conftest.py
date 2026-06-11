"""Project-wide pytest fixtures.

ГЛАВНОЕ — изоляция last-good снапшота самарских филиалов (BUG-2026-06-11-01).

S1 (`bd4c5af`) добавил `persist_snapshot()` в `core._ensure_regions_loaded`: на каждой
ЗДОРОВОЙ загрузке `/regions` самарский срез пишется на диск в РЕАЛЬНЫЙ кэш-каталог
(`agent_logic_2/nayka_api/apidata/samara_branches_snapshot.json`). В тестах любой кейс,
мокающий `api_nayka.site_regions` здоровым срезом (BUG-A/BUG-E derive-city и т.п.),
запускает настоящий `_ensure_regions_loaded` → пишет реальный снапшот. Этот файл затем
читается через `fallback_branches()` в ДРУГОМ тесте (`test_address_walkin_degraded_guard`)
→ fallback отдаёт 2-3 чужих филиала вместо 31-филиального seed → ложное падение,
зависящее от порядка тестов и состояния диска (несамодостаточный гейт).

Фикс класса: уводим `_snapshot_path()` в per-test tmp. Реальный кэш doctors/prices
(тот же каталог, но ДРУГИЕ файлы) не трогаем — изолируем ТОЛЬКО снапшот.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_samara_snapshot(tmp_path, monkeypatch):
    """Снапшот самарских филиалов пишется/читается в per-test tmp, не в реальный apidata.

    Делает тест-гейт самодостаточным: persist одного теста не может заразить fallback
    другого. Покрывает ВЕСЬ класс (любой писатель через `_ensure_regions_loaded` и любой
    читатель через `fallback_branches`/`resolve_samara_branches`), а не один кейс.
    """
    from messengers_router.services import _samara_branches as _sb

    snap = tmp_path / "samara_branches_snapshot.json"
    monkeypatch.setattr(_sb, "_snapshot_path", lambda: snap)
    yield
