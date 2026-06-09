"""BUG-C (facet 3): «Код 5437» (homecode услуги) мисроутится в TEST_RESULT.

Прод-диалог (после деплоя): «Код 5437» → бот «укажите: фамилия, год, код, номер»
(TEST_RESULT), хотя 5437 = `serviceHomecode` услуги «Витамин B1(тиаминпирофосфат)».

Важно: price-слой УЖЕ резолвит по homecode — `price_info("Код 5437")` → Витамин B1
(через `_extract_homecode_query`). Значит баг чисто в NLU-роутинге: «Код 5437» →
rule=None → LLM→TEST_RESULT. Архитектура legacy_v2: правило не-None драйвит прод
(classifier.analyze short-circuit). → добавляем правило «код/услуга <цифры>» → PRICE,
и существующий price_info отдаёт цену услуги по коду.

Различающий сигнал: result-код — БУКВЫ («Вг»/«Бг»); service-homecode — ЦИФРЫ (5437).
Инвариант (класс): «код/услуга <3–6 цифр>» → PRICE (lookup услуги по homecode), НЕ
TEST_RESULT. Буквенные result-коды («Код Вг») не затрагиваются. Класс, не инстанс.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.classifier import deterministic_rule_decision
from messengers_router.policies import detect_service_code_lookup_intent
from messengers_router.services import Services


def run(coro):
    return asyncio.run(coro)


CODE_POSITIVE = ["Код 5437", "код 5437", "услуга 5437", "код услуги 5437", "по коду 5437", "услуга 502"]
CODE_NEGATIVE = [
    "Код Вг",                 # буквенный result-код
    "сколько стоит ОАК",
    "5437",                    # голые цифры без ключевого слова
    "год рождения 1989",
    "номер анализа 12345",     # это про результат, не услуга
    "записаться на завтра",
]


@pytest.mark.parametrize("text", CODE_POSITIVE, ids=[t[:18] for t in CODE_POSITIVE])
def test_detect_service_code_lookup_positive(text):
    assert detect_service_code_lookup_intent(text) is True, f"{text!r} not detected"


@pytest.mark.parametrize("text", CODE_NEGATIVE, ids=[t[:18] for t in CODE_NEGATIVE])
def test_detect_service_code_lookup_negative(text):
    assert detect_service_code_lookup_intent(text) is False, f"{text!r} wrongly detected"


def test_deterministic_rule_decision_service_code_routes_to_price():
    d = run(deterministic_rule_decision("Код 5437", {}))
    assert d is not None
    assert d.label == "PRICE", f"service code wrongly routed to {d.label} (expected PRICE)"
    assert d.label != "TEST_RESULT"
    assert "rule_service_code_lookup" in d.flags
    assert d.needs_handoff is False


def test_letter_result_code_not_service_lookup():
    """Анти-регресс: буквенный result-код «Код Вг» НЕ уходит в service-code PRICE."""
    d = run(deterministic_rule_decision("Код Вг", {}))
    if d is not None:
        assert "rule_service_code_lookup" not in d.flags


@pytest.mark.parametrize("query", ["Код 5437", "услуга 5437"], ids=["kod", "usluga"])
def test_price_info_resolves_service_by_homecode_e2e(query):
    """e2e: price_info по коду услуги отдаёт «Витамин B1» (homecode 5437)."""
    svc = Services()
    payload = run(svc.price_info(query, {}))
    rows = (payload.get("prices") or []) + (payload.get("family_variants") or [])
    names = " | ".join(str(r.get("serviceName") or r.get("name") or "") for r in rows).lower()
    assert "витамин b1" in names, f"{query!r} -> {names[:100]!r}"
