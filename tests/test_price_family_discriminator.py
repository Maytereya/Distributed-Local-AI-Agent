"""Класс-инвариант: различающий токен семейства анализов (гепатит С/В/А…).

Класс дефекта `family_discriminator_dropped` (SEVERE, дезинформация ценой):
токенизатор price-запроса выбрасывал 1-символьный дискриминатор (`с`/`в`/`b`),
запрос «гепатит с» схлопывался в `['гепатит']` и приземлялся на ПЕРВУЮ строку
семейства — «Гепатит В - HBsAg» (350₽) — то есть пациент получал цену ЧУЖОГО
гепатита. Аналогично family-mode показывал все гепатиты вперемешку, во главе — В.

Инвариант (проверяется на РЕАЛЬНОМ каталоге region 3, как соседние price-тесты):
когда запрос указывает семейство + букву-дискриминатор (и буква — последний
значимый токен), НИ single-резолвер, НИ family-mode не отдают строку, чей
буквенный маркер противоречит запросу. Механизм catalog-derived (без хардкода
названий болезней): «семейство» выводится из самого прайса (≥2 буквенных
варианта у одного ствола ≥6 символов).

Решение владельца (2026-07-22): при неоднозначности показывать список вариантов
ТОЛЬКО совпавшего семейства (гепатит С), а не единственную (возможно чужую) цену.
"""

from __future__ import annotations

import asyncio
import re

from agent_logic_2.nayka_api import api_price
from messengers_router.services import Services
from messengers_router.services._prices_helpers import (
    _build_price_family_payload,
    resolve_price_service_name_from_catalog,
)


def _run(coro):
    return asyncio.run(coro)


def _price_info_names(query: str) -> list[str]:
    """End-to-end: имена строк из пациентского payload price_info (prices+family)."""
    payload = _run(Services().price_info(query, {}))
    rows = (payload.get("prices") or []) + (payload.get("family_variants") or [])
    return [str(r.get("serviceName") or r.get("name") or "") for r in rows]


def _rows():
    return [r for r in api_price.load_price_by_region(3) if isinstance(r, dict)]


# --- Классификаторы строки по буквенному маркеру гепатита ------------------
# Опираемся на фактические маркеры прайса: С↔HCV, В↔HBsAg/HBe/HBc/HBs.

_HEP_B_RE = re.compile(r"гепатит\w*\s+в\b|hbsag|hbe|hbc|hbs|вирус\s+гепатита\s+b\b", re.I)
_HEP_C_RE = re.compile(r"гепатит\w*\s+с\b|hcv|вирус\s+гепатита\s+c\b", re.I)


def _is_hep_b(name: str) -> bool:
    return bool(_HEP_B_RE.search(name or ""))


def _is_hep_c(name: str) -> bool:
    return bool(_HEP_C_RE.search(name or ""))


def _family_names(payload) -> list[str]:
    if not payload:
        return []
    return [
        str(v.get("serviceName") or v.get("name") or "")
        for v in (payload.get("family_variants") or [])
    ]


# --- single-резолвер: чужой гепатит НЕ уходит ------------------------------

def test_resolve_hepatitis_c_never_returns_b_row():
    """«гепатит с» не должен вернуть В-строку (дезинформация ценой)."""
    resolved = resolve_price_service_name_from_catalog("гепатит с", rows=_rows())
    # Либо None (→ уточнение списком), либо честная С-строка. НИКОГДА не В.
    assert not (resolved and _is_hep_b(resolved)), f"вернулась чужая (B) строка: {resolved!r}"
    if resolved:
        assert _is_hep_c(resolved), f"ожидалась C-строка, получено: {resolved!r}"


def test_resolve_hepatitis_b_never_returns_c_row():
    resolved = resolve_price_service_name_from_catalog("гепатит в", rows=_rows())
    assert not (resolved and _is_hep_c(resolved)), f"вернулась чужая (C) строка: {resolved!r}"


def test_resolve_hepatitis_latin_b_letter_not_c():
    """Латинская буква-дискриминатор («гепатит b») тоже не должна давать C."""
    resolved = resolve_price_service_name_from_catalog("гепатит b", rows=_rows())
    assert not (resolved and _is_hep_c(resolved)), f"вернулась чужая (C) строка: {resolved!r}"


# --- family-mode: список ТОЛЬКО совпавшего семейства -----------------------

def test_family_hepatitis_c_lists_only_c_variants():
    payload = _build_price_family_payload("гепатит с", _rows())
    names = _family_names(payload)
    assert names, "family-mode не построил список для «гепатит с»"
    offending = [n for n in names if _is_hep_b(n)]
    assert not offending, f"в списке C оказались B-строки: {offending}"
    assert any(_is_hep_c(n) for n in names), f"нет ни одной C-строки: {names}"


def test_family_hepatitis_b_lists_only_b_variants():
    payload = _build_price_family_payload("гепатит в", _rows())
    names = _family_names(payload)
    assert names, "family-mode не построил список для «гепатит в»"
    offending = [n for n in names if _is_hep_c(n)]
    assert not offending, f"в списке B оказались C-строки: {offending}"


# --- Регрессии: существующее поведение НЕ ломается --------------------------

def test_regression_hepatitis_c_summary_still_resolves_c():
    """«гепатит с суммарные» и раньше резолвился верно (буква не последняя)."""
    resolved = resolve_price_service_name_from_catalog("гепатит с суммарные", rows=_rows())
    assert resolved and _is_hep_c(resolved), f"регрессия: {resolved!r}"


def test_regression_anti_hcv_resolves_c():
    resolved = resolve_price_service_name_from_catalog("анти-hcv", rows=_rows())
    assert resolved and _is_hep_c(resolved), f"регрессия: {resolved!r}"


def test_regression_bare_hepatitis_still_family():
    """Голое «гепатит» без буквы — по-прежнему семейство всех гепатитов."""
    payload = _build_price_family_payload("гепатит", _rows())
    names = _family_names(payload)
    assert len(names) >= 3, f"family-mode для голого «гепатит» сломан: {names}"


def test_regression_vitamin_d_unaffected():
    """Витамины идут своим путём (composite-token), не через новый гард."""
    resolved = resolve_price_service_name_from_catalog("витамин д", rows=_rows())
    assert resolved and "витамин" in resolved.lower() and "d" in resolved.lower(), (
        f"регрессия витамина D: {resolved!r}"
    )


# --- End-to-end price_info: малые семейства (single/lab-ветка, не family) ---
# Гепатит А/D/E имеют <3 вариантов → family-mode не триггерит, ответ идёт через
# single/lab-путь. Он тоже не должен показывать чужую букву (был баг: «гепатит А»
# → «Гепатит В - HBsAg» первой строкой).

def test_e2e_hepatitis_a_price_info_only_a():
    names = _price_info_names("гепатит а")
    assert names, "price_info вернул пусто на «гепатит а»"
    offending = [n for n in names if _is_hep_b(n) or _is_hep_c(n)]
    assert not offending, f"«гепатит а» показал чужую букву: {offending}"


def test_e2e_hepatitis_c_price_info_only_c():
    names = _price_info_names("гепатит с")
    assert names, "price_info вернул пусто на «гепатит с»"
    offending = [n for n in names if _is_hep_b(n)]
    assert not offending, f"«гепатит с» показал B-строки: {offending}"


def test_e2e_hepatitis_b_price_info_only_b():
    names = _price_info_names("гепатит в")
    assert names, "price_info вернул пусто на «гепатит в»"
    offending = [n for n in names if _is_hep_c(n)]
    assert not offending, f"«гепатит в» показал C-строки: {offending}"
