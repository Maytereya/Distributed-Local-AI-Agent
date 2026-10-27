"""T7 шаг 3б: разбор ценовой реплики на фразы (`turn_parse`, контракт C) — единственный режим, как на проде с 15.10.

Было: выключатель `[MESSENGER_ROUTER] turn_parse = off | shadow | on` с умолчанием кода `off` при проде `on`. Потеря
ключа или опечатка откатывали прод на путь правил, который T7 снимает (класс STEP-2026-10-15-DEFAULTS-B), а тесты без
явного режима проверяли путь, которого на проде нет. Шаг (а) — умолчание `on`; шаг (б) — режимы `off`/`shadow` и машина
тени (фоновая задача, `drain`, статус `late`) сняты, ключ не читается.

Инвариант класса: ценовой ход всегда идёт разбором (`tu_parse_status` в телеметрии, понимание — фразы разбора), что бы
ни лежало в конфиге; ответ — по эталону пути цены; второй короткий вопрос после корзины не берёт услугу, которой
пациент не видел.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля


def test_there_is_no_switch():
    from agent_logic_2 import config
    from messengers_router import turn_parse

    assert not hasattr(config, "MR_TURN_PARSE")
    assert not hasattr(config.settings.messenger_router, "turn_parse")
    assert not hasattr(turn_parse, "mode") and not hasattr(turn_parse, "drain")


@pytest.mark.parametrize("stale", [None, "off", "shadow"])
def test_price_turn_goes_through_the_parse_two_turns(stale, mini_prod_world, monkeypatch):  # noqa: F811
    """Через эндпоинт: список — фразы разбора (разметка `turn_spans` — оракул, прод с деплоя 17; обе позиции сильные по
    данным — строку LLM не выбирает), ответ — корзина ферритина и глюкозы; «а сколько?» следом не называет цены строки,
    которой не было в ответе. Старое значение ключа в конфиге (`off`, `shadow` — как в `config.ini` до шага) разбор не
    выключает. При лежащей LLM разметки нет — «Уточните название» (`price_by_parse = on`), разбор `unavailable`."""

    from messengers_router import llm_runtime
    from messengers_router.runtime_config import config as cfg
    from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed

    if stale is not None:
        monkeypatch.setattr(cfg, "MR_TURN_PARSE", stale, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(("ферритин", "глюкоза")))
    keep_llm_breaker_closed(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog(["сколько стоит ферритин и глюкоза", "а сколько?"], f"t7-parse-{stale}")
        rows = list(world.telemetry)
    assert "Ферритин — 530 руб." in first["text"] and "Глюкоза — 190 руб." in first["text"], first["text"]
    assert rows[0].get("tu_parse_status") == "picked" and rows[0].get("tu_rule") == "parse_picked"
    assert "в моче" not in first["text"] and "в моче" not in second["text"]
