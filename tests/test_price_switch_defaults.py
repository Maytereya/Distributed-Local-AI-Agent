"""Умолчания кода = прод для выключателей пути цены (STEP-2026-10-17-DEFAULTS-D, решение владельца 16.10, деплой 5).

На проде с 16.10: `choose_price = on`, `parse_price_guard = llm`, `p2_llm_judge = on`. Умолчания кода отставали — потеря
или опечатка ключа в `config.ini` молча вернула бы прежний вызов выбора строки и гарды П2 по словарю (инвариант
STEP-2026-10-15-DEFAULTS-B: умолчание = значение прода, проверенное деплоем).

Инварианты: нет атрибута конфига — значение прода; мусор в значении — прежний путь (как у `_get_enum` до переворота).
"""

from __future__ import annotations

import pytest


def _set(monkeypatch, key: str, raw: str | None) -> None:
    from messengers_router.runtime_config import config as cfg

    if raw is None:
        monkeypatch.delattr(cfg, key, raising=False)
    else:
        monkeypatch.setattr(cfg, key, raw, raising=False)


@pytest.mark.parametrize(("raw", "expected"), [(None, "on"), ("off", "off"), ("legacy", "legacy"), ("ON", "on"), ("x", "off")])
def test_choose_price_is_read_with_a_safe_default(raw, expected, monkeypatch):
    from messengers_router.services._price_select_llm import _choose_mode

    _set(monkeypatch, "MR_CHOOSE_PRICE", raw)
    assert _choose_mode() == expected


@pytest.mark.parametrize(("raw", "expected"), [(None, True), ("off", False), ("on", True), ("ON", True), ("x", False)])
def test_p2_llm_judge_is_read_with_a_safe_default(raw, expected, monkeypatch):
    from messengers_router.services import prices

    _set(monkeypatch, "MR_P2_LLM_JUDGE", raw)
    assert prices._p2_llm_judge_on() is expected


@pytest.mark.parametrize(("raw", "judged"), [(None, True), ("rules", False), ("llm", True), ("LLM", True), ("x", False)])
def test_parse_price_guard_is_read_with_a_safe_default(raw, judged, monkeypatch):
    """Под `llm` слабую фразу, выбранную LLM разбора одной услугой, судит выбор строки; иначе — гард П2 правил."""

    from messengers_router import turn_parse
    from messengers_router.choice import Strength
    from messengers_router.services import prices

    reply = "Сколько стоит анализ крови на концентрацию ферритина"
    phrase = turn_parse.ParsedPhrase(text="анализ крови на концентрацию ферритина", start=14, end=len(reply), strength=Strength.WEAK)
    parse = turn_parse.PhraseParse(reply=reply, status=turn_parse.ParseStatus.PICKED, source="llm", items=(phrase,))
    monkeypatch.setattr(prices.turn_parse, "acting_for", lambda _query: parse)
    _set(monkeypatch, "MR_PARSE_PRICE_GUARD", raw)
    assert prices._parse_phrase_judged_by_llm(reply) == (phrase.text if judged else "")


def test_config_defaults_are_the_prod_values():
    from agent_logic_2 import config

    assert (config.MR_CHOOSE_PRICE, config.MR_PARSE_PRICE_GUARD, config.MR_P2_LLM_JUDGE) == ("on", "llm", "on")
