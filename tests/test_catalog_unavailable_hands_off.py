"""Каталог клиники пуст — перевод, а не «попробуйте через 5 минут или напишите «оператор»» (ход (д), принцип 3;
`mis_honest_degradation`, решение владельца 18.10 «сбой МИС → оператор»).

Было (замер стенда сбоев 17.10, «срезов нет»): «Сейчас временно недоступен каталог услуг. Я не смогу надежно подобрать
нужную услугу. Попробуйте повторить запрос через 5 минут или напишите «оператор».» — без перевода: пациент сам
должен звать оператора (принцип 3 запрещает «уточните у оператора» без перевода).

Стало под `on`: тот же ход — перевод с текстом сбоя по источнику (`service_error_prices` / `service_error_doctors_list`
/ `service_error`). Под `off` — как было.

Инвариант класса: пустой каталог МИС не оставляет пациента без оператора.
"""

from __future__ import annotations

import pytest

from evalbench import faults
from evalbench import world as ev_world
from messengers_router.policies import handoff_message
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура


@pytest.mark.parametrize(("honest", "handoff"), [("on", True), ("off", False)])
def test_empty_catalog_turn(honest, handoff, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    with ev_world.FrozenWorld(mini_prod_world, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_MIS_HONEST_DEGRADATION", honest, raising=False)
        with faults.Faults(world, snapshots="none", live="network"):
            first, second = run_dialog(["сколько стоит ферритин", "а цена?"], f"catalog-empty-{honest}")

    assert first["handoff"] is handoff, first["text"]
    if handoff:
        assert handoff_message("service_error_prices") in first["text"], first["text"]
        assert "напишите «оператор»" not in first["text"]
    else:
        assert "Попробуйте повторить запрос через 5 минут" in first["text"]
