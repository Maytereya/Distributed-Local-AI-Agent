"""Каталог клиники пуст — перевод, а не «попробуйте через 5 минут или напишите «оператор»» (ход (д), принцип 3;
`mis_honest_degradation`, решение владельца 18.10 «сбой МИС → оператор»).

Было (замер стенда сбоев 17.10, «срезов нет»): «Сейчас временно недоступен каталог услуг. Я не смогу надежно подобрать
нужную услугу. Попробуйте повторить запрос через 5 минут или напишите «оператор».» — без перевода: пациент сам
должен звать оператора (принцип 3 запрещает «уточните у оператора» без перевода).

Стало под `on`: тот же ход — перевод (`service_error_doctors_list`, если нужен только каталог врачей, иначе общий
`service_error`: каталог услуг нужен и цене, и записи).

Инвариант класса: пустой каталог МИС не оставляет пациента без оператора.
Выключатель `mis_honest_degradation` снят 28.10 (on на проде с 18.10): путь off удалён.
"""

from __future__ import annotations


from evalbench import faults
from evalbench import world as ev_world
from messengers_router.policies import handoff_message
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура


def test_empty_catalog_turn(mini_prod_world, monkeypatch):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        with faults.Faults(world, snapshots="none", live="network"):
            first, second = run_dialog(["сколько стоит ферритин", "а цена?"], "catalog-empty")

    assert first["handoff"] is True, first["text"]
    assert handoff_message("service_error") in first["text"], first["text"]
    assert "напишите «оператор»" not in first["text"]
