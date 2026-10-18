"""Давность прайса в ответе — ход (д), шаг 3 (`mis_honest_degradation`; решения владельца 08.10 и 18.10).

Было: при сбое скачивания бот отдавал последний срез прайса любой давности как свежий (DATA-9; замер стенда сбоев
17.10: прайс старше 8 дней — ответ как по свежему).

Стало под `on`: срез не старше 2 дней — молча; 3–7 — приписка в конце ответа «Цены актуальны для г. …, по данным на
ДД.ММ.» (вариант «Б»); старше 7 дней — перевод `service_error_prices`. Под `off` — как было.

Инвариант класса: цена из устаревшего среза не выдаётся пациенту за актуальную.
Проверка — через эндпоинт, два хода, мир мини-прода (срез 08.10), «сегодня» загрузчиков сдвинуто стендом сбоев.
"""

from __future__ import annotations

import pytest

from evalbench import faults
from evalbench import world as ev_world
from messengers_router import renderer
from messengers_router.policies import handoff_message
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_DIALOG = ["сколько стоит ферритин", "а цена?"]


def _play(world_dir, monkeypatch, honest: str, stale_days: int):
    from agent_logic_2.nayka_api import snapshots
    from messengers_router.runtime_config import config as cfg

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_MIS_HONEST_DEGRADATION", honest, raising=False)
        patch.setattr(snapshots, "_last", {})
        if stale_days:
            with faults.Faults(world, snapshots="stale", stale_days=stale_days):
                return run_dialog(_DIALOG, f"fresh-{honest}-{stale_days}")
        return run_dialog(_DIALOG, f"fresh-{honest}-{stale_days}")


@pytest.mark.parametrize(("stale_days", "as_of"), [(0, None), (2, None), (3, "08.10"), (7, "08.10")])
def test_price_age_up_to_a_week_is_answered_with_or_without_a_date(stale_days, as_of, mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _play(mini_prod_world, monkeypatch, "on", stale_days)

    assert first["handoff"] is False and "Ферритин — 530 руб." in first["text"], first["text"]
    if as_of:
        assert first["text"].rstrip().endswith(f"по данным на {as_of}._"), first["text"]
    else:
        assert "по данным на" not in first["text"] + second["text"]


def test_price_older_than_a_week_goes_to_the_operator(mini_prod_world, monkeypatch):  # noqa: F811
    first, _second = _play(mini_prod_world, monkeypatch, "on", 8)

    assert first["handoff"] is True and handoff_message("service_error_prices") in first["text"], first["text"]
    assert "530 руб" not in first["text"]


@pytest.mark.parametrize("stale_days", [3, 8])
def test_switch_off_answers_as_before(stale_days, mini_prod_world, monkeypatch):  # noqa: F811
    fresh = _play(mini_prod_world, monkeypatch, "off", 0)
    stale = _play(mini_prod_world, monkeypatch, "off", stale_days)

    assert [t["text"] for t in stale] == [t["text"] for t in fresh]


def test_disclaimer_line_with_a_date_is_one_line_and_idempotent():
    once = renderer._append_samara_disclaimer("1) Ферритин — 530 руб.", "08.10")
    assert once.endswith("_Цены актуальны для г. Самара, по данным на 08.10._")
    assert renderer._append_samara_disclaimer(once, "08.10") == once
    assert renderer._append_samara_disclaimer("1) Ферритин — 530 руб.").endswith("_Цены актуальны для г. Самара._")
