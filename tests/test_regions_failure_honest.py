"""Сбой справочника филиалов МИС (`/regions`) отличим от пустоты — ход (д), шаг 5, DATA-2 (`mis_honest_degradation`).

Было: `site_regions` при сбое (сеть, таймаут, 5xx, битый JSON) отдавал `[]` — как пустой справочник: хороший кэш
филиалов затирался, честная ветка `tech_unavailable` в адресах была недостижима (DATA-6), «где сдать анализы» без
`/regions` и срезов — «Не нашёл филиалы по этому городу. Уточните город» (замер стенда сбоев 17.10), расписание по
названному филиалу — «Регион '…' не найден».

Стало: `site_regions` отмечает сбой (`api_nayka.regions_failed()`; пустой ответ — тоже сбой: клиника без филиалов не
бывает), в логе — warning вместо print. Под `on`: хороший кэш филиалов при сбое не затирается, сбой виден
потребителям (`_regions_last_failed`) — адрес без данных переводит на оператора; расписание по филиалу при сбое
справочника — строка со сбоем → `ScheduleSourceUnavailable` → перевод.

Инвариант класса: сбой справочника филиалов не выдаётся за «такого филиала / города нет».
Выключатель `mis_honest_degradation` снят 28.10 (on на проде с 18.10): путь off удалён.
"""

from __future__ import annotations

import asyncio

import pytest

from evalbench import faults
from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_REGIONS = [{"id": 102, "name": "пр.Ленина, 5", "addressForSite": "г. Самара, пр.Ленина, 5", "city": "Самара", "parent": None}]

pytestmark = pytest.mark.real_mis_loader  # мир гейта подменяет `site_regions` пустышкой — нужен настоящий клиент


@pytest.mark.parametrize(("kind", "failed"), [("network", True), ("timeout", True), ("http_5xx", True), ("empty", True)])
def test_regions_failure_is_marked(kind, failed, monkeypatch):
    from agent_logic_2.nayka_api import api_nayka

    monkeypatch.setattr(api_nayka, "_session_get", faults.boundary(kind, []))
    assert api_nayka.site_regions() == []
    assert api_nayka.regions_failed() is failed

    class _Ok:
        def raise_for_status(self):
            return None

        def json(self):
            return [dict(r) for r in _REGIONS]

    monkeypatch.setattr(api_nayka, "_session_get", lambda *_a, **_kw: _Ok())
    assert api_nayka.site_regions() and api_nayka.regions_failed() is False


def test_good_branch_cache_survives_a_failure(monkeypatch):
    from agent_logic_2.nayka_api import api_nayka
    from messengers_router.services import Services

    svc = Services()
    monkeypatch.setattr(api_nayka, "site_regions", lambda *_a, **_kw: [dict(r) for r in _REGIONS])
    monkeypatch.setattr(api_nayka, "regions_failed", lambda: False)
    good = asyncio.run(svc._ensure_regions_loaded())
    assert good

    svc._regions_cache_loaded_at = 0.0  # TTL прошёл
    monkeypatch.setattr(api_nayka, "site_regions", lambda *_a, **_kw: [])
    monkeypatch.setattr(api_nayka, "regions_failed", lambda: True)
    after = asyncio.run(svc._ensure_regions_loaded())

    assert bool(after) is True
    assert svc._regions_last_failed is True


def test_address_without_branches_data_hands_off(mini_prod_world, monkeypatch):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        with faults.Faults(world, live="network", snapshots="none"):
            first, _second = run_dialog(["где сдать анализы?", "а на Ленина?"], "regions-down")

    assert first["handoff"] is True, first["text"]
    assert "Не нашёл филиалы" not in first["text"]


def test_schedule_by_branch_reports_a_branches_failure_not_a_missing_region(monkeypatch):
    from agent_logic_2.nayka_api import api_nayka

    monkeypatch.setattr(api_nayka, "_SCHEDULE_REFS_CACHE", {})
    monkeypatch.setattr(api_nayka, "_session_get", faults.boundary("network", []))

    text = api_nayka.find_doctor_schedule("Трубин", "Ленина")

    assert "Не удалось получить" in text
    assert api_nayka.is_api_error_message(text) is True
