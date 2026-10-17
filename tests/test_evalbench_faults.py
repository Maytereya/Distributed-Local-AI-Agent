"""Фабрика сбоев МИС на сетевой границе (`evalbench/faults.py`) — ход (д), шаг 2, TEST-3 (17.10).

Инварианты: сбой ставится на HTTP-границу клиентов МИС, а не исключением из подменённого загрузчика (которого прод не
бросает); каждый вид даёт то, что даёт настоящая граница (`ConnectionError`, `ReadTimeout`, 200 и `[]`, 503 →
`HTTPError`); запрос настоящего клиента справочника доходит до границы; срезы под `stale` / `none` видны в телеметрии
хода как `stale` с давностью / `unavailable`; сеть не открывается, мир не пишет срезов; выход откатывает всё.
"""

from __future__ import annotations

import pytest
import requests

from evalbench import faults
from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("network", requests.ConnectionError), ("timeout", requests.ReadTimeout), ("empty", None), ("http_5xx", requests.HTTPError)],
)
def test_boundary_answers_like_the_real_one(kind, expected):
    calls: list[str] = []
    get = faults.boundary(kind, calls)

    if kind in ("network", "timeout"):
        with pytest.raises(expected):
            get("https://mis.example/api/v1/site/regions")
    else:
        resp = get("https://mis.example/api/v1/site/regions")
        if expected is None:
            resp.raise_for_status()
            assert (resp.status_code, resp.json()) == (200, [])
        else:
            with pytest.raises(expected):
                resp.raise_for_status()
    assert calls == ["v1/site/regions"]


def test_unknown_kinds_are_rejected(mini_prod_world):  # noqa: F811
    with pytest.raises(ValueError):
        faults.boundary("slow", [])
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        with pytest.raises(ValueError):
            faults.Faults(world, snapshots="broken")
        with pytest.raises(ValueError):
            faults.Faults(world, snapshots="stale")


@pytest.mark.real_mis_loader  # мир гейта подменяет `site_regions` пустышкой — здесь нужен настоящий клиент
@pytest.mark.parametrize("kind", faults.LIVE_KINDS)
def test_live_fault_reaches_the_boundary_through_the_real_clients(kind, mini_prod_world):  # noqa: F811
    from agent_logic_2.nayka_api import api_nayka

    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        in_world = (api_nayka.site_regions, api_nayka.site_promotions, api_nayka._session_get)
        with faults.Faults(world, live=kind) as fault:
            api_nayka.site_regions()
            try:
                api_nayka.site_promotions()
            except requests.RequestException:
                pass  # клиент акций сбой пробрасывает — так и на проде
        assert any(call.endswith("/regions") for call in fault.calls), fault.calls
        assert any(call.endswith("/promotions") for call in fault.calls), fault.calls
        assert (api_nayka.site_regions, api_nayka.site_promotions, api_nayka._session_get) == in_world  # откат
        assert world.net.attempts == []


def _price_src(rows):
    return [(row.get("src") or {}).get("price_region_3") for row in rows]


@pytest.mark.parametrize(
    ("snapshots", "stale_days", "live", "expected"),
    [
        ("stale", 3, None, {"status": "stale", "age_days": 3}),
        ("stale", 9, "timeout", {"status": "stale", "age_days": 9}),
        ("none", 0, "network", {"status": "unavailable", "age_days": None}),
        ("none", 0, "empty", {"status": "unavailable", "age_days": None}),
    ],
    ids=["stale3", "stale9_timeout", "none_network", "none_empty"],
)
def test_snapshot_faults_show_up_in_turn_telemetry(snapshots, stale_days, live, expected, mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        with faults.Faults(world, live=live, snapshots=snapshots, stale_days=stale_days) as fault:
            run_dialog(["сколько стоит ферритин", "а цена?"], f"faults-{snapshots}-{live}")
            written = fault.written()
        rows = list(world.telemetry)
        assert world.net.attempts == []
    assert _price_src(rows)[0] == expected
    assert written == []  # пустой ответ МИС среза не пишет (`require_rows`), сбой — тем более
    if live is not None:
        assert any("priceByRegion" in call for call in fault.calls), fault.calls  # загрузчик дошёл до границы
