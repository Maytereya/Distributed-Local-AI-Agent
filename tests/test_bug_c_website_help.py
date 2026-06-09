"""BUG-C (facet 1): «корзина на сайте» мисроутится в ADDRESS (адреса филиалов).

Прод-диалог Д193: «Где находится корзина на сайте?» → бот отдал список филиалов
(ADDRESS). Это про навигацию по САЙТУ (онлайн-корзина/личный кабинет), а не про
физический адрес. Бот клиники такими вопросами не занимается → честный ответ
«помогаю с услугами клиники, по сайту обратитесь в поддержку», НЕ адреса.

Архитектура: legacy_v2 — если `deterministic_rule_decision` вернул не-None, LLM
пропускается (`classifier.analyze: if rule is not None: return rule`), правило
драйвит прод. «Где находится…» срабатывает в ADDRESS-правиле → чиним правилом:
website-ветку ставим ПЕРЕД ADDRESS (как clarify, шаблон BUG-2026-06-04-02).

Инвариант (класс): вопрос про навигацию по сайту (корзина/личный кабинет на сайте,
«как пользоваться сайтом», «сайт не работает») → честный website-help (clarify,
handoff=False), НЕ ADDRESS. Специфично к САЙТУ: не задевает будущую «корзину
анализов» (bot-side) и запись «на сайте». Физические адреса филиалов — без изменений.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.classifier import deterministic_rule_decision
from messengers_router.policies import WEBSITE_HELP_TEXT, detect_website_help_intent


def run(coro):
    return asyncio.run(coro)


WEBSITE_POSITIVE = [
    "Где находится корзина на сайте?",
    "корзина на сайте",
    "где на сайте корзина",
    "личный кабинет на сайте",
    "как пользоваться сайтом",
    "сайт не работает",
]

WEBSITE_NEGATIVE = [
    "корзина анализов",            # будущая bot-side фича — не трогать
    "добавь в корзину анализ",     # bot-side корзина
    "можно записаться на сайте",    # это про запись, не про навигацию сайта
    "запись через сайт",
    "Где находится филиал на Победы 83?",  # физический адрес
    "адрес клиники",
    "сколько стоит ОАК",
]


@pytest.mark.parametrize("text", WEBSITE_POSITIVE, ids=[t[:24] for t in WEBSITE_POSITIVE])
def test_detect_website_help_positive(text):
    assert detect_website_help_intent(text) is True, f"{text!r} not detected as website-help"


@pytest.mark.parametrize("text", WEBSITE_NEGATIVE, ids=[t[:24] for t in WEBSITE_NEGATIVE])
def test_detect_website_help_negative(text):
    assert detect_website_help_intent(text) is False, f"{text!r} wrongly detected as website-help"


def test_deterministic_rule_decision_website_cart_clarifies_not_address():
    d = run(deterministic_rule_decision("Где находится корзина на сайте?", {}))
    assert d is not None
    assert d.label != "ADDRESS", f"website-cart wrongly routed to ADDRESS: {d.label}"
    assert d.clarify_needed is True
    assert d.clarify_reason == WEBSITE_HELP_TEXT
    assert d.needs_handoff is False
    assert "rule_website_help" in d.flags


def test_deterministic_rule_decision_physical_address_still_address():
    """Анти-регресс: физический адрес филиала по-прежнему ADDRESS."""
    d = run(deterministic_rule_decision("Где находится филиал на Победы 83?", {}))
    assert d is not None
    assert d.label == "ADDRESS"
    assert "rule_website_help" not in d.flags
