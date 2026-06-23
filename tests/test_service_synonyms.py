"""BUG-2026-06-23-01 — курируемые синонимы услуг к терминам каталога.

Класс: формулировка пациента расходится со словом каталога ("забор"≠"взятие",
"электромиография"≠"ЭМГ") → лексический матчер не находит / цепляет чужое. Узкий
детерминированный fast-path нормализует синоним ПЕРЕД матчем. Общий случай (любой
синоним/аббревиатура/опечатка) — LLM-нормализация (вариант B), отдельный трек.
"""

from __future__ import annotations

from messengers_router.services._prices_helpers import (
    _apply_service_synonyms,
    resolve_price_service_name_from_catalog,
)


def test_apply_service_synonyms_maps_known_only():
    assert _apply_service_synonyms("забор крови из вены") == "взятие крови из вены"
    assert _apply_service_synonyms("сколько стоит забора крови") == "сколько стоит взятие крови"
    assert _apply_service_synonyms("Электромиография") == "ЭМГ"
    assert _apply_service_synonyms("электромиографию цена") == "ЭМГ цена"
    # НЕ задевает чужое:
    assert _apply_service_synonyms("общий анализ крови") == "общий анализ крови"
    # ЭНМГ (электронейромиография) — другой тест, не должен схлопываться в ЭМГ:
    assert _apply_service_synonyms("электронейромиография") == "электронейромиография"


def test_resolve_blood_draw_synonym_finds_venipuncture():
    # «забор крови из вены» (синоним «взятия») → «Взятие крови из вены», а не None (был clarify-цикл).
    assert resolve_price_service_name_from_catalog("забор крови из вены") == "Взятие крови из вены"
    assert (
        resolve_price_service_name_from_catalog("сколько стоит забор крови из вены")
        == "Взятие крови из вены"
    )
    assert resolve_price_service_name_from_catalog("забор крови из пальца") == "Взятие крови из пальца"


def test_resolve_emg_abbreviation_not_wrong_coagulation():
    # «электромиография» (полное слово) → услуга ЭМГ, НЕ чужое «Электро…коагуляция» (по префиксу).
    res = resolve_price_service_name_from_catalog("электромиография")
    assert res is not None
    assert "ЭМГ" in res, res
    assert "коагуляц" not in res.lower(), res


def test_resolve_unrelated_services_unchanged_regression():
    # Регресс: запросы без забор/электромиограф не затронуты синонимами.
    assert resolve_price_service_name_from_catalog("взятие крови из вены") == "Взятие крови из вены"
    assert "ттг" in (resolve_price_service_name_from_catalog("ттг") or "").lower()
    assert resolve_price_service_name_from_catalog("общий анализ крови") is not None
