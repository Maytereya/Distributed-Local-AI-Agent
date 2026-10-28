"""Акции при сбое МИС — честный перевод (ход (д), шаг 7а, DATA-14; `mis_honest_degradation`, решение владельца 18.10).

Было: сбой `/promotions` — «Сейчас не получилось загрузить список акций. Попробуйте чуть позже…» без оператора; сбой
справочника филиалов (`site_regions` отдаёт `[]`) — акции, привязанные к филиалам региона, отбрасывались молча, и при
отсутствии общих акций пациент слышал «Сейчас активных акций нет» (замер стенда сбоев 17.10).

Стало под `on`: «Не получилось загрузить всю информацию об акциях — передаю оператору.» и перевод; оба сбоя — в лог
(`log_degraded`).

Инвариант класса: сбой источника акций не выдаётся за «акций нет» и не обрывается без оператора.
Выключатель `mis_honest_degradation` снят 28.10 (on на проде с 18.10): путь off удалён.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router.services.news import PROMOTIONS_UNAVAILABLE_TEXT
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

_PROMO = {"id": 7, "title": "Скидка на анализы", "subtitle": "", "text": "Скидка 10%", "endDate": "2026-12-31", "regions": [11], "services": []}


def _play(world_dir, monkeypatch, *, promotions=None, regions=None):
    from agent_logic_2.nayka_api import api_nayka
    from messengers_router.services import news

    with ev_world.FrozenWorld(world_dir, "day"), monkeypatch.context() as patch:
        patch.setattr(api_nayka, "_SCHEDULE_REFS_CACHE", {})
        if promotions is not None:
            patch.setattr(api_nayka, "site_promotions", lambda *_a, **_kw: [dict(p) for p in promotions])
            if hasattr(news, "site_promotions"):
                patch.setattr(news, "site_promotions", lambda *_a, **_kw: [dict(p) for p in promotions])
        if regions is not None:
            patch.setattr(api_nayka, "site_regions", lambda *_a, **_kw: [dict(r) for r in regions])
        return run_dialog(["какие у вас акции?", "а на анализы есть скидки?"], "promo")


@pytest.mark.parametrize("case", ["promotions_down", "regions_empty"])
def test_promotions_failure_hands_off_under_the_switch(case, mini_prod_world, monkeypatch):  # noqa: F811
    kwargs = {} if case == "promotions_down" else {"promotions": [_PROMO], "regions": []}  # в мире акций нет — сбой

    first, second = _play(mini_prod_world, monkeypatch, **kwargs)

    assert first["handoff"] is True and PROMOTIONS_UNAVAILABLE_TEXT in first["text"], first["text"]
    assert "активных акций нет" not in (first["text"] + second["text"]).lower()
