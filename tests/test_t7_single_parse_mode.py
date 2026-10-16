"""T7 шаг 3б: разбор ценовой реплики на фразы (`turn_parse`, контракт C) — режим по умолчанию, как на проде с 15.10.

Было: умолчание кода `turn_parse = off`; прод — `on` из `config.ini`. Потеря ключа или опечатка откатывали прод на путь
правил, который T7 снимает (STEP-2026-10-15-DEFAULTS-B — тот же класс для других выключателей), а тесты без явного
режима проверяли путь, которого на проде нет.

Инвариант класса: без явной настройки ценовой ход идёт разбором (`tu_parse_status` в телеметрии, понимание — фразы
разбора), ответ — по эталону пути цены; второй короткий вопрос после корзины не берёт услугу, которой пациент не видел.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from tests.test_price_turns_golden import _golden, mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля


def test_config_default_is_on():
    from agent_logic_2 import config

    assert config.MR_TURN_PARSE == "on"


@pytest.mark.parametrize(("raw", "expected"), [(None, "on"), ("on", "on")])
def test_mode_without_the_key_is_on(raw, expected, monkeypatch):
    from messengers_router import turn_parse
    from messengers_router.runtime_config import config as cfg

    if raw is None:
        monkeypatch.delattr(cfg, "MR_TURN_PARSE", raising=False)
    else:
        monkeypatch.setattr(cfg, "MR_TURN_PARSE", raw, raising=False)
    assert turn_parse.mode() == expected


def test_price_turn_goes_through_the_parse_by_default_two_turns(mini_prod_world, monkeypatch):  # noqa: F811
    """Через эндпоинт, без явного режима: список — фразы разбора (обе позиции сильные по данным — без LLM), ответ — эталон
    `list_text_and`; «а сколько?» следом не называет цены строки, которой не было в ответе."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.delattr(cfg, "MR_TURN_PARSE", raising=False)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog(["сколько стоит ферритин и глюкоза", "а сколько?"], "t7-default-parse")
        rows = list(world.telemetry)
    assert first["text"] == _golden()["list_text_and"][0]["text"]
    assert rows[0].get("tu_parse_status") == "strong_all" and rows[0].get("tu_rule") == "parse_strong_all"
    assert "в моче" not in first["text"] and "в моче" not in second["text"]
