"""Акции из CRM /promotions вместо meili-дайджестов «НОВОСТИ ЗА 05.11».

Прод-диалог 09.07: (1) «Акции» → бессодержательный список «НОВОСТИ ЗА <дата>»;
(2) «Уточни» после списка → LLM→OTHER-дефлект «нет информации»; (3) «Акция
почему нет сил» → снова дайджесты вместо честного промаха.

Инварианты (класс):
1. Источник — /promotions: настоящие названия/условия; base64-image в пайплайн
   не попадает.
2. Срок: истёкшие скрыты; endDate — ISO-datetime с TZ; без endDate = активна.
3. Регион: бот самарский — акции чужих регионов (Пенза) скрыты, «Все»/Самара/
   филиалы Самары показаны; /regions недоступен → консервативно только «Все».
4. Поиск по названию: совпадение → detail; промах → ЧЕСТНЫЙ «не нашёл» +
   актуальный список (не молча дайджесты).
5. Follow-up после списка («уточни», «2», название) продолжает NEWS —
   только сразу после NEWS-ответа (гард _last_label).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from agent_logic_2.nayka_api import api_nayka
from messengers_router.classifier import deterministic_rule_decision
from messengers_router.entity_grounder import _is_allowed_key
from messengers_router.mess_types import Evidence, SessionState
from messengers_router.renderer import format_news_for_patient
from messengers_router.response_builder import build_news_response
from messengers_router.services import news as news_mod
from messengers_router import evidence_keys as ek


def run(coro):
    return asyncio.run(coro)


_REGIONS = [
    {"id": 1, "name": "Все", "parent": None},
    {"id": 2, "name": "Самарская область", "parent": 1},
    {"id": 3, "name": "Самара", "parent": 2},
    {"id": 32, "name": "Пензенская область", "parent": 1},
    {"id": 9052, "name": "Пенза", "parent": 32},
    {"id": 290, "name": "Нефтегорск", "parent": 2},
    {"id": 8882, "name": "Ленина 5", "parent": 3},
]

_PROMOS = [
    {
        "id": 1, "title": "ЧЕКАП + ВИТАМИН D", "subtitle": "с выгодой 700 ₽",
        "text": "Уважаемые пациенты! Скидка 50% на витамин D при прохождении чекапа. Коды: 1079 Ежегодный чекап.",
        "endDate": "2026-08-30T20:00:00.000+00:00", "startDate": None,
        "regions": [1], "isAnalysis": True, "isDoctorService": False,
    },
    {
        "id": 2, "title": "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%", "subtitle": "до 31.12.2026 г.",
        "text": "Скидка 30% на Общий анализ крови, Глюкозу и другие популярные анализы.",
        "endDate": "2026-12-30T20:00:00.000+00:00", "startDate": None,
        "regions": [290, 3], "isAnalysis": True, "isDoctorService": False,
    },
    {
        "id": 3, "title": "Анализы перед операцией 3570 ₽", "subtitle": "Пенза",
        "text": "Пензенская цена профиля Госпитализация.",
        "endDate": "2026-12-30T20:00:00.000+00:00", "startDate": None,
        "regions": [9052], "isAnalysis": True, "isDoctorService": False,
    },
    {
        "id": 4, "title": "Истёкшая акция", "subtitle": "",
        "text": "Уже всё.", "endDate": "2026-06-29T20:00:00.000+00:00", "startDate": None,
        "regions": [1], "isAnalysis": False, "isDoctorService": False,
    },
    {
        "id": 5, "title": "Скидка в офисе на Ленина", "subtitle": "",
        "text": "Только в филиале пр. Ленина, 5.", "endDate": None, "startDate": None,
        "regions": [8882], "isAnalysis": False, "isDoctorService": True,
    },
]


class _FixedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 7, 10, 12, 0, tzinfo=tz or timezone.utc)


@pytest.fixture()
def promo_env(monkeypatch):
    monkeypatch.setattr(news_mod.api_nayka, "site_promotions", lambda *, realtime=False: [dict(p) for p in _PROMOS])
    monkeypatch.setattr(news_mod.api_nayka, "site_regions", lambda *, realtime=False: list(_REGIONS))
    monkeypatch.setattr(news_mod, "datetime", _FixedDT)


def _titles(payload):
    return [x["title"] for x in payload["news"]]


# --- фильтры: срок + регион -------------------------------------------------

def test_list_mode_filters_expired_and_foreign_regions(promo_env):
    payload = run(news_mod.news_info(None, "какие акции есть?", {}))
    assert payload["mode"] == "list"
    titles = _titles(payload)
    assert "ЧЕКАП + ВИТАМИН D" in titles          # regions=[1] «Все»
    assert "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%" in titles  # [290, 3] — есть Самара
    assert "Скидка в офисе на Ленина" in titles    # филиал Самары, без endDate
    assert "Анализы перед операцией 3570 ₽" not in titles  # Пенза-only
    assert "Истёкшая акция" not in titles


def test_regions_unavailable_keeps_only_explicit_vse(promo_env, monkeypatch):
    def _boom(*, realtime=False):
        raise RuntimeError("regions down")

    monkeypatch.setattr(news_mod.api_nayka, "site_regions", _boom)
    payload = run(news_mod.news_info(None, "акции", {}))
    titles = _titles(payload)
    assert titles == ["ЧЕКАП + ВИТАМИН D"], "при недоступном /regions — только явное «Все»"


def test_source_unavailable_is_soft(promo_env, monkeypatch):
    def _boom(*, realtime=False):
        raise RuntimeError("promotions down")

    monkeypatch.setattr(news_mod.api_nayka, "site_promotions", _boom)
    payload = run(news_mod.news_info(None, "акции", {}))
    assert payload["news"] == []
    assert payload["note"] == "news source unavailable"


# --- поиск по названию --------------------------------------------------------

def test_search_by_title_returns_detail(promo_env):
    payload = run(news_mod.news_info(None, "акция чекап", {}))
    assert payload["mode"] == "detail"
    assert _titles(payload)[0] == "ЧЕКАП + ВИТАМИН D"


def test_search_miss_is_honest(promo_env):
    payload = run(news_mod.news_info(None, "акция почему нет сил", {}))
    assert payload["mode"] == "miss"
    assert _titles(payload), "при промахе показываем актуальный список"


def test_pick_by_number_from_context(promo_env):
    ctx = {"titles": ["ЧЕКАП + ВИТАМИН D", "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%"]}
    payload = run(news_mod.news_info(None, "расскажи про 2", {"_promo_context": ctx}))
    assert payload["mode"] == "detail"
    assert _titles(payload) == ["ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%"]


def test_pick_by_ordinal_word(promo_env):
    ctx = {"titles": ["ЧЕКАП + ВИТАМИН D", "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%"]}
    payload = run(news_mod.news_info(None, "первая", {"_promo_context": ctx}))
    assert payload["mode"] == "detail"
    assert _titles(payload) == ["ЧЕКАП + ВИТАМИН D"]


# --- follow-up правило в классификаторе ---------------------------------------

_PROMO_LAST = {
    "_promo_context": {"titles": ["ЧЕКАП + ВИТАМИН D", "ПОПУЛЯРНЫЕ АНАЛИЗЫ СО СКИДКОЙ 30%"]},
    "_last_label": "NEWS",
}


@pytest.mark.parametrize("text", ["Уточни", "подробнее", "2", "про вторую", "чекап"])
def test_promo_followup_rule_continues_news(text):
    d = run(deterministic_rule_decision(text, dict(_PROMO_LAST)))
    assert d is not None
    assert d.label == "NEWS"
    assert "rule_news_promo_followup" in d.flags
    assert d.entities.get("promo_query") == text
    assert d.entities.get("_promo_context") == _PROMO_LAST["_promo_context"]


def test_promo_followup_does_not_hijack_price():
    d = run(deterministic_rule_decision("сколько стоит ОАК", dict(_PROMO_LAST)))
    assert d is None or "rule_news_promo_followup" not in d.flags


def test_promo_followup_requires_fresh_news_label():
    stale = dict(_PROMO_LAST, _last_label="PRICE")
    d = run(deterministic_rule_decision("Уточни", stale))
    assert d is None or "rule_news_promo_followup" not in d.flags


def test_news_intent_still_routes_first_turn():
    d = run(deterministic_rule_decision("какие акции сейчас действуют?", {}))
    assert d is not None and d.label == "NEWS"


def test_grounder_allows_promo_control_keys():
    assert _is_allowed_key("promo_query", "NEWS", None) is True
    assert _is_allowed_key("promo_pick_index", "NEWS", None) is True
    assert _is_allowed_key("_promo_context", "NEWS", None) is True


# --- рендерер ------------------------------------------------------------------

def test_render_detail_card(promo_env):
    payload = run(news_mod.news_info(None, "акция чекап", {}))
    text = format_news_for_patient(payload, {})
    assert "Акция «ЧЕКАП + ВИТАМИН D»" in text
    assert "Действует до 31.08.2026" in text  # 2026-08-30T20:00Z + 4ч = 31.08 по Самаре
    assert "витамин D" in text
    assert "НОВОСТИ ЗА" not in text


def test_render_list_numbered_with_hint(promo_env):
    payload = run(news_mod.news_info(None, "какие акции есть?", {}))
    text = format_news_for_patient(payload, {})
    assert "1. " in text
    assert "Напишите номер или название акции" in text


def test_render_miss_is_honest(promo_env):
    payload = run(news_mod.news_info(None, "акция почему нет сил", {}))
    text = format_news_for_patient(payload, {})
    assert "Не нашёл акцию" in text
    assert "Сейчас действуют:" in text


def test_render_source_unavailable(promo_env, monkeypatch):
    def _boom(*, realtime=False):
        raise RuntimeError("down")

    monkeypatch.setattr(news_mod.api_nayka, "site_promotions", _boom)
    payload = run(news_mod.news_info(None, "акции", {}))
    text = format_news_for_patient(payload, {})
    assert "не получилось загрузить" in text.lower()


# --- контекст для follow-up (response_builder) ----------------------------------

def _news_envelope(payload):
    ev = Evidence()
    ev.put(ek.NEWS, payload)
    state = SessionState(session_id="promo-ctx")
    resp = build_news_response("NEWS", ev, state)
    return resp, state


def test_list_response_stashes_promo_context(promo_env):
    payload = run(news_mod.news_info(None, "какие акции есть?", {}))
    resp, state = _news_envelope(payload)
    assert resp is not None
    ctx = state.last_entities.get("_promo_context")
    assert isinstance(ctx, dict) and ctx.get("titles")


def test_detail_response_clears_promo_context(promo_env):
    payload = run(news_mod.news_info(None, "акция чекап", {}))
    ev = Evidence()
    ev.put(ek.NEWS, payload)
    state = SessionState(session_id="promo-ctx2")
    state.last_entities["_promo_context"] = {"titles": ["x"]}
    build_news_response("NEWS", ev, state)
    assert "_promo_context" not in state.last_entities


# --- api_nayka: base64-image не тащим -------------------------------------------

def test_site_promotions_strips_image_and_caches(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"id": 1, "title": "T", "image": "/9j/AAAA...", "contentType": "image/jpeg"}]

    def fake_get(url, **kwargs):
        calls["n"] += 1
        assert url.endswith("/promotions")
        return _Resp()

    monkeypatch.setattr(api_nayka, "_session_get", fake_get)
    monkeypatch.setattr(api_nayka, "SCHEDULE_REFS_TTL_SECONDS", 300.0)
    out1 = api_nayka.site_promotions(realtime=True)
    out2 = api_nayka.site_promotions(realtime=True)
    assert calls["n"] == 1, "справочник акций кэшируется"
    assert out1 and "image" not in out1[0] and "contentType" not in out1[0]
    assert out2 == out1
