"""Запись при пустом списке врачей — перевод, а не все адреса подряд (ход (д), шаг 7б, DATA-5; `mis_honest_degradation`).

Было: в записи список филиалов фильтруется «где принимает врач»; пустой список врачей (сбой утренней загрузки, свежий
том) молча выключал фильтр — пациенту, записывающемуся к урологу, предлагались все адреса, включая филиалы без
уролога и пункты забора (пример с приватного мира 18.10: пять адресов вместо трёх), а note писал «filtered by
doctor-capable branches».

Стало (решение владельца 18.10, вариант «а»): пустой список врачей в записи — `log_degraded` (и под off); под `on` —
«Не смог проверить, в каких филиалах принимает врач, — передаю оператору.» и перевод.

Инвариант класса: сбой среза врачей не превращается в список филиалов без проверки, кто где принимает.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router.services.addresses import APPOINTMENT_BRANCHES_UNAVAILABLE_TEXT
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура


def _play(world_dir, monkeypatch, honest: str):
    from messengers_router.runtime_config import config as cfg

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:  # в мире мини-прода врачей нет
        patch.setattr(cfg, "MR_MIS_HONEST_DEGRADATION", honest, raising=False)
        return run_dialog(["хочу записаться к урологу", "на Ленина"], f"appt-no-doctors-{honest}")


def test_empty_doctors_list_hands_off_under_the_switch(mini_prod_world, monkeypatch):  # noqa: F811
    first, _second = _play(mini_prod_world, monkeypatch, "on")

    assert first["handoff"] is True and APPOINTMENT_BRANCHES_UNAVAILABLE_TEXT in first["text"], first["text"]


def test_switch_off_keeps_the_old_answer(mini_prod_world, monkeypatch):  # noqa: F811
    first, _second = _play(mini_prod_world, monkeypatch, "off")

    assert APPOINTMENT_BRANCHES_UNAVAILABLE_TEXT not in first["text"]


@pytest.mark.parametrize("honest", ["on", "off"])
def test_address_question_outside_appointment_is_untouched(honest, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    with ev_world.FrozenWorld(mini_prod_world, "day"), monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_MIS_HONEST_DEGRADATION", honest, raising=False)
        [turn] = run_dialog(["где сдать анализы?"], f"address-{honest}")

    assert APPOINTMENT_BRANCHES_UNAVAILABLE_TEXT not in turn["text"]
