"""Справочник филиалов не запрашивается у МИС в потоке цикла событий — ход (д), DATA-7, `get_branches` (18.10).

Было: `Services.get_branches` зовут синхронные функции хода — дозаполнение слотов (`flow_policy.quick_fill_entities_
from_text`, пять вызовов в роутере) и ответ шага записи (`response_builder`). При пустом кэше филиалов она сама
синхронно шла в МИС (`api_nayka.site_regions`, таймаут и повторы) — в цикле событий: весь бот ждал.

Стало (решение владельца 18.10 «делаем асинхрон»): ход в начале прогревает кэш (`Services.warm_branches` →
`_ensure_regions_loaded`, запрос в потоке); `get_branches` в цикле событий только читает кэш. Пустой ответ МИС — следующий
прогрев не раньше чем через 60 с. Вне цикла событий (скрипты) — прежний синхронный запрос.

Инвариант класса: в ходе пациента запрос справочника филиалов к МИС не выполняется в потоке цикла событий.
"""

from __future__ import annotations

import asyncio
import threading
import time

from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_REGIONS = [
    {"id": 101, "name": "Самара", "city": "Самара", "parent": None},
    {"id": 102, "name": "пр.Ленина, 5", "addressForSite": "г. Самара, пр.Ленина, 5", "city": "Самара", "parent": 101, "doctorService": True},
    {"id": 103, "name": "ул. Победы, 83", "addressForSite": "г. Самара, ул. Победы, 83", "city": "Самара", "parent": 101, "doctorService": True},
]


def test_appointment_dialog_never_asks_mis_for_branches_on_the_loop(mini_prod_world, monkeypatch):  # noqa: F811
    from agent_logic_2.nayka_api import api_nayka

    calls: list[bool] = []

    def site_regions(*_a, **_kw):
        calls.append(threading.current_thread() is threading.main_thread())
        return [dict(r) for r in _REGIONS]

    with ev_world.FrozenWorld(mini_prod_world, "day"), monkeypatch.context() as patch:
        patch.setattr(api_nayka, "site_regions", site_regions)
        run_dialog(["хочу записаться к урологу", "на Ленина"], "branches-off-loop")

    assert calls, "справочник филиалов в диалоге записи не запрашивался"
    assert not any(calls), calls  # asyncio.run держит цикл в главном потоке


def test_get_branches_reads_only_the_cache_inside_the_loop(monkeypatch):
    from agent_logic_2.nayka_api import api_nayka
    from messengers_router.services import Services

    monkeypatch.setattr(api_nayka, "site_regions", lambda *_a, **_kw: [dict(r) for r in _REGIONS])
    svc = Services()

    async def in_loop():
        empty = svc.get_branches()  # кэш пуст — в МИС не идёт
        await svc.warm_branches()
        return empty, svc.get_branches()

    empty, warm = asyncio.run(in_loop())
    assert empty == []
    assert warm  # после прогрева — филиалы из кэша
    assert [b["name"] for b in Services().get_branches()] == [b["name"] for b in warm]  # вне цикла — прежний путь


def test_warm_after_an_empty_answer_waits_a_minute(monkeypatch):
    from agent_logic_2.nayka_api import api_nayka
    from messengers_router.services import Services

    calls: list[int] = []
    monkeypatch.setattr(api_nayka, "site_regions", lambda *_a, **_kw: calls.append(1) or [])
    svc = Services()

    asyncio.run(svc.warm_branches())
    asyncio.run(svc.warm_branches())  # сбой был только что — МИС не дёргаем
    assert len(calls) == 1
    svc._branches_warm_failed_at = time.time() - 61
    asyncio.run(svc.warm_branches())
    assert len(calls) == 2
