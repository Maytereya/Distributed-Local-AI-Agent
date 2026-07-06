"""П5 (вариант B, BUG-2026-06-23-01 «Открыто»): LLM-нормализация услуги, fallback-only.

Класс: пациентская формулировка вне курируемых синонимов и лексики каталога
(«коагулограмма» = гемостазиограмма, «сахар в крови» = глюкоза) → раньше miss →
clarify-цикл. Теперь: лексический exact и difflib-fuzzy ПРОМАХНУЛИСЬ → LLM
переписывает формулировку в термин каталога → результат ОБЯЗАТЕЛЬНО
верифицируется тем же лексическим resolve (LLM не может выдумать услугу) →
отдаётся как FUZZY-кандидат → существующий confirm-флоу («Вы имели в виду …?»).
Инварианты класса:
  1) рабочие запросы (exact/fuzzy без LLM) НИКОГДА не зовут LLM;
  2) не-услуги (запись/парковка, без price-intent) НИКОГДА не зовут LLM;
  3) любой сбой/мусор/NONE от LLM → прежний miss (fail-open, ноль регресса);
  4) LLM-выход без каталожной верификации НЕ попадает в ответ.
LLM в тестах замокан — детерминированная логика; живое поведение → remote eval.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services import Services
from messengers_router.services import _service_normalizer as SN


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_normalizer_cache(monkeypatch):
    monkeypatch.setattr(SN, "_CACHE", {})
    monkeypatch.setattr(SN, "_ENABLED", True)
    yield


def _mock_llm(monkeypatch, reply: str, calls: list | None = None):
    async def fake_generate_text(prompt: str, **kwargs):
        if calls is not None:
            calls.append(prompt)
        return reply

    monkeypatch.setattr(SN, "generate_text", fake_generate_text)


# --- юнит: сам нормализатор -------------------------------------------------

def test_normalizer_returns_sanitized_single_line(monkeypatch):
    _mock_llm(monkeypatch, '  "Гемостазиограмма"\nлишняя строка')
    out = run(SN.llm_normalize_service_query("коагулограмма"))
    assert out == "Гемостазиограмма"


@pytest.mark.parametrize("reply", ["NONE", "none", "", "   ", "х" * 200])
def test_normalizer_rejects_none_empty_and_overlong(monkeypatch, reply):
    _mock_llm(monkeypatch, reply)
    assert run(SN.llm_normalize_service_query("коагулограмма")) is None


def test_normalizer_rejects_echo_of_input(monkeypatch):
    # Эхо (LLM вернула то же слово) бесполезно: лексика по нему уже промахнулась.
    _mock_llm(monkeypatch, "коагулограмма")
    assert run(SN.llm_normalize_service_query("коагулограмма")) is None


def test_normalizer_fail_open_on_exception(monkeypatch):
    async def boom(prompt: str, **kwargs):
        raise TimeoutError("llm busy")

    monkeypatch.setattr(SN, "generate_text", boom)
    assert run(SN.llm_normalize_service_query("коагулограмма")) is None


def test_normalizer_caches_result(monkeypatch):
    calls: list = []
    _mock_llm(monkeypatch, "Гемостазиограмма", calls)
    assert run(SN.llm_normalize_service_query("коагулограмма")) == "Гемостазиограмма"
    assert run(SN.llm_normalize_service_query("коагулограмма")) == "Гемостазиограмма"
    assert len(calls) == 1, "повторный запрос должен идти из кэша"


def test_normalizer_disabled_flag(monkeypatch):
    calls: list = []
    _mock_llm(monkeypatch, "Гемостазиограмма", calls)
    monkeypatch.setattr(SN, "_ENABLED", False)
    assert run(SN.llm_normalize_service_query("коагулограмма")) is None
    assert not calls


# --- интеграция: match_catalog_service (реальный каталог, LLM замокан) -------

def test_match_catalog_llm_fallback_verified_as_fuzzy_candidate(monkeypatch):
    # Класс-инвариант: промах лексики+difflib → LLM-канон, ВЕРИФИЦИРОВАННЫЙ каталогом,
    # отдаётся как fuzzy-кандидат (→ downstream confirm «Вы имели в виду …?»).
    _mock_llm(monkeypatch, "Гемостазиограмма")
    svc = Services()
    res = run(svc.match_catalog_service("сколько стоит коагулограмма"))
    assert res["status"] == "fuzzy", res
    assert "гемостаз" in res["canonical"].lower(), res
    assert res.get("matched_key") == "llm_normalized"


def test_match_catalog_llm_garbage_and_unverified_stay_miss(monkeypatch):
    svc = Services()
    # Мусорный ответ LLM → resolve не подтверждает → прежний miss (инвариант 3/4).
    _mock_llm(monkeypatch, "Извините, я не понимаю ваш вопрос")
    res = run(svc.match_catalog_service("коагулограмма"))
    assert res["status"] == "miss", res
    # Несуществующая услуга от LLM → тоже miss.
    _mock_llm(monkeypatch, "Анализ лунного грунта")
    res2 = run(svc.match_catalog_service("коагулограмма"))
    assert res2["status"] == "miss", res2


def test_match_catalog_exact_path_never_calls_llm(monkeypatch):
    # Инвариант 1: рабочий запрос (ттг → exact) не должен трогать LLM вообще.
    calls: list = []
    _mock_llm(monkeypatch, "ЭТОГО НЕ ДОЛЖНО БЫТЬ", calls)
    svc = Services()
    res = run(svc.match_catalog_service("ттг"))
    assert res["status"] == "exact"
    assert not calls, "exact-путь дернул LLM — регресс горячего пути"


@pytest.mark.parametrize(
    "text",
    ["запишите меня на завтра", "перенесите запись", "у вас есть парковка возле клиники?"],
)
def test_match_catalog_non_service_text_never_calls_llm(monkeypatch, text):
    # Инвариант 2: не-услуги (нет фразы услуги и нет price-intent) не зовут LLM —
    # иначе спекулятивный prefetch замедлил бы обычные ходы записи.
    calls: list = []
    _mock_llm(monkeypatch, "Глюкоза", calls)
    svc = Services()
    res = run(svc.match_catalog_service(text))
    assert res["status"] in {"miss", "fuzzy", "exact"}
    assert not calls, f"LLM дернулся на не-услуге: {text!r}"


def test_match_catalog_llm_fallback_second_pair_sugar(monkeypatch):
    # Класс, не инстанс: вторая пара из bug-log — «сахар в крови» → Глюкоза.
    _mock_llm(monkeypatch, "Глюкоза")
    svc = Services()
    res = run(svc.match_catalog_service("сколько стоит сахар в крови"))
    assert res["status"] == "fuzzy", res
    assert "глюкоз" in res["canonical"].lower(), res
