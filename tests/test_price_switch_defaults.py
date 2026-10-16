"""Выключатели пути цены с одним значением на проде сняты (решение владельца 16.10, STEP-2026-10-17-ONE-PATH-*).

До 17.10: `choose_price = on`, `parse_price_guard = llm`, `p2_llm_judge = on` — значения прода с 16.10, с утра 17.10 —
умолчания кода (STEP-2026-10-17-DEFAULTS-D). Теперь путь один: ключей нет, прежние значения в `config.ini` прода (он держит
их до деплоя 6) не читаются.
"""

from __future__ import annotations

import pytest


def test_the_keys_are_gone_from_the_settings():
    from agent_logic_2 import config

    for attr, key in (("MR_CHOOSE_PRICE", "choose_price"), ("MR_PARSE_PRICE_GUARD", "parse_price_guard"), ("MR_P2_LLM_JUDGE", "p2_llm_judge")):
        assert not hasattr(config, attr) and not hasattr(config.settings.messenger_router, key), key


@pytest.mark.parametrize("raw", [None, "rules", "llm", "x"])
def test_parse_price_guard_key_is_not_read(raw, monkeypatch):
    """Слабую фразу, выбранную LLM разбора одной услугой, судит выбор строки, что бы ни лежало в ключе."""

    from messengers_router import turn_parse
    from messengers_router.choice import Strength
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import prices

    reply = "Сколько стоит анализ крови на концентрацию ферритина"
    phrase = turn_parse.ParsedPhrase(text="анализ крови на концентрацию ферритина", start=14, end=len(reply), strength=Strength.WEAK)
    parse = turn_parse.PhraseParse(reply=reply, status=turn_parse.ParseStatus.PICKED, source="llm", items=(phrase,))
    monkeypatch.setattr(prices.turn_parse, "acting_for", lambda _query: parse)
    if raw is not None:
        monkeypatch.setattr(cfg, "MR_PARSE_PRICE_GUARD", raw, raising=False)
    assert prices._parse_phrase_judged_by_llm(reply) == phrase.text


@pytest.mark.parametrize("raw", [None, "off", "on"])
def test_p2_llm_judge_key_is_not_read(raw, monkeypatch):
    """Название из сущности при отказе резолвера идёт к судье П2, что бы ни лежало в ключе."""

    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import prices

    if raw is not None:
        monkeypatch.setattr(cfg, "MR_P2_LLM_JUDGE", raw, raising=False)
    assert not hasattr(prices, "_p2_llm_judge_on")
    assert prices._p2_name_from_entity("узи щитовидной железы", None, "узи щитовидной железы", "сколько стоит узи щитовидной железы по омс")
