"""Политика сбоя под `turn_parse = on`: сколько вариантов кода у слабой позиции (L-15b, шаг 6, T5).

Поправка владельца 13.10: при недоступной LLM слабая позиция — несколько вариантов кода (до 3–5) с полными
названиями и ценами, форму решит замер шага 6. Замер `down` (dev 867): при 5 вариантах «вне разметки» 102 → 296.
Число — настройка `[MESSENGER_ROUTER] parse_fail_variants` (2–5), чтобы замерить 5 против 3 одним деревом
(`evalbench.run --cfg MR_PARSE_FAIL_VARIANTS=3`); по умолчанию 3 — как на проде (STEP-2026-10-15-DEFAULTS-B), запасное
при нечитаемом значении — 5. Инвариант: у слабой позиции вариантов не больше настройки и не меньше двух (одна строка —
не «та самая цена»).
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router.services import _basket_select as bs
from tests.test_price_one_or_many_class import CASES, _configure, _priced_sections
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля


@pytest.mark.parametrize(("raw", "expected"), [(None, 5), (5, 5), (3, 3), (2, 2), (1, 5), (9, 5), ("x", 5)])
def test_setting_is_read_with_a_safe_default(raw, expected, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    if raw is None:
        monkeypatch.delattr(cfg, "MR_PARSE_FAIL_VARIANTS", raising=False)
    else:
        monkeypatch.setattr(cfg, "MR_PARSE_FAIL_VARIANTS", raw, raising=False)
    assert bs.variants_max() == expected


def test_config_default_is_three():
    """Умолчание кода = прод (STEP-2026-10-15-DEFAULTS-B)."""

    from agent_logic_2 import config

    assert config.MR_PARSE_FAIL_VARIANTS == 3


@pytest.mark.parametrize("limit", [3, 5])
def test_weak_position_without_llm_shows_at_most_the_setting(limit, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    _configure(monkeypatch, "deploy")
    monkeypatch.setattr(cfg, "MR_PARSE_FAIL_VARIANTS", limit, raising=False)
    case = CASES["list_true_three"]  # «ферритин» сильная, «ттг» и «витамин д» — слабые
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        [turn] = run_dialog([case.text], f"parse-fail-variants-{limit}")
    sections = _priced_sections(turn["text"])
    weak = [section for section in sections if "Ферритин" not in section]
    assert weak, turn["text"]
    assert all(2 <= len(section) <= limit for section in weak), (limit, sections)
