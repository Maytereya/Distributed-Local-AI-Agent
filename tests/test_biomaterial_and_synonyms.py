"""П3 (прод 10.08): упоминание БИОМАТЕРИАЛА ломало матч услуги.

Симптом: «Сколько стоит мазок на микрофлору с поверхности задней стенки глотки»
→ «Скажите название услуги» дважды, затем на «мазок» — список из шести
нерелевантных мазков.

Офлайн-репро на живых данных (прайс region 3 + `service_info` МИС) показал, что
всё ещё хуже: соседняя формулировка «сколько стоит мазок на микрофлору с задней
стенки глотки» приземлялась на **«Обработка задней стенки глотки
расфокусированным лучом СО2 лазера»** — упоминание материала не просто мешало
матчу, оно уводило в ЧУЖУЮ услугу (процедуру вместо анализа).

Корень: матч ломает именно упоминание биоматериала, а не отсутствие синонима —
«мазок на микрофлору» (материал снят) резолвится в «Посев на микрофлору» без
ошибок.

Инвариант (класс): фраза-биоматериал снимается с запроса (словарь — из
`biomatNames` МИС), остаток матчится обычным путём, и результат принимается
ТОЛЬКО если справочник МИС подтверждает: услуга этот материал принимает.
Биоматериал НЕ идентифицирует услугу — «соскоб с задней стенки глотки» подходит
к 44 услугам, «кровь из вены» — к 1103.

`serviceSynonyms` (высшее доверие) клиника заполняет; на срезе 14.08 поле пустое
у всех 1613 строк, поэтому механика проверяется на подставленном справочнике.
"""

from __future__ import annotations

import pytest

from agent_logic_2.nayka_api import api_price, api_service_info
from messengers_router.services import _biomaterial as bio
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    resolve_price_service_name_from_catalog,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]
INFO = [r for r in api_service_info.load_service_info() if isinstance(r, dict)]


def _resolve(q: str) -> str | None:
    return resolve_price_service_name_from_catalog(q, rows=ROWS)


# --- Гарды окружения ------------------------------------------------------

def test_mis_snapshot_has_biomat_and_no_synonyms_yet():
    """Фиксируем состояние справочника, на котором построен фикс: biomatNames
    заполнен почти везде, serviceSynonyms клиника ещё не заполнила."""
    with_biomat = sum(1 for r in INFO if r.get("biomatNames"))
    with_syn = sum(1 for r in INFO if r.get("serviceSynonyms"))
    assert with_biomat > len(INFO) * 0.9, (with_biomat, len(INFO))
    assert with_syn == 0, "serviceSynonyms заполнили — снять этот гард и добавить кейсы"


def test_biomaterial_does_not_identify_service():
    """Ключевое ограничение дизайна: по биоматериалу услугу определять нельзя."""
    def _count(biomat: str) -> int:
        return sum(
            1
            for r in INFO
            if bio.service_accepts_biomaterial(str(r.get("serviceName") or ""), biomat) is True
        )

    assert _count("соскоб с задней стенки глотки") > 20
    assert _count("кровь из вены") > 500


# --- Класс: биоматериал снимается, услуга находится ------------------------

@pytest.mark.parametrize(
    "query",
    [
        "мазок на микрофлору с поверхности задней стенки глотки",
        "сколько стоит мазок на микрофлору с задней стенки глотки",
        "мазок на микрофлору с задней стенки глотки",
    ],
    ids=["prod_10_08", "with_price_verb", "no_surface_word"],
)
def test_biomaterial_phrase_no_longer_breaks_match(query):
    resolved = _resolve(query)
    assert resolved is not None, f"{query!r} не резолвится"
    assert "посев на микрофлору" in resolved.lower(), resolved


def test_biomaterial_mention_does_not_land_on_foreign_procedure():
    """Худший из найденных вариантов: локативное совпадение уводило в лазерную
    обработку глотки — процедуру вместо анализа."""
    resolved = _resolve("сколько стоит мазок на микрофлору с задней стенки глотки")
    assert resolved is not None
    assert "лазер" not in resolved.lower(), resolved
    assert "обработка" not in resolved.lower(), resolved


def test_found_service_actually_accepts_the_biomaterial():
    """Шаг 3 механики: найденная услуга подтверждена справочником МИС."""
    head, tail = bio.split_biomaterial_tail(
        "мазок на микрофлору с поверхности задней стенки глотки"
    )
    assert head == "мазок на микрофлору", head
    assert "задней стенки глотки" in tail, tail
    resolved = _resolve("мазок на микрофлору с поверхности задней стенки глотки")
    assert bio.service_accepts_biomaterial(resolved or "", tail) is True


# --- Снятие хвоста: границы -----------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "взятие крови из вены",
        "забор крови из вены",
        "анализ крови из вены на ферритин",
        "общий анализ крови",
        "посев из зева на микрофлору",
        "спермограмма",
        "приём кардиолога",
    ],
    ids=["venous_draw", "venous_draw_syn", "ferritin", "oak", "throat_swab", "spermogram", "cardio"],
)
def test_normal_queries_unchanged(query):
    """Анти-over-trigger: слой не имеет права портить работающие запросы —
    в том числе те, где биоматериал ЧАСТЬ названия услуги («Взятие крови из вены»)."""
    assert _resolve(query) is not None, query


def test_single_word_tail_is_not_stripped():
    """«кал», «моча», «нос» сами бывают названием услуги/различающим токеном —
    хвост из одного слова не снимаем."""
    for text in ("кал", "анализ кал", "моча", "мазок нос"):
        head, tail = bio.split_biomaterial_tail(text)
        assert tail == "", (text, head, tail)


def test_tail_of_only_prepositions_is_not_stripped():
    head, tail = bio.split_biomaterial_tail("посев из")
    assert tail == "", (head, tail)


def test_strip_leaves_nonempty_head():
    """Если после снятия ничего не остаётся — не снимаем (иначе запрос исчезнет)."""
    head, tail = bio.split_biomaterial_tail("соскоб с задней стенки глотки")
    assert tail == "" or head, (head, tail)


# --- serviceSynonyms: механика подключена заранее --------------------------

def test_mis_synonym_takes_priority_when_filled(monkeypatch):
    """Когда клиника заполнит `serviceSynonyms`, синоним обязан выигрывать у
    лексического матча — это высшее доверие. Проверяем на подставленном
    справочнике, потому что живое поле ещё пустое."""
    fake = bio._MisVocabularies()
    fake.synonym_to_service = {"кровь на сахар": "Глюкоза"}
    monkeypatch.setattr(bio, "_vocabularies", lambda: fake)

    from messengers_router.services import _prices_helpers as ph

    monkeypatch.setattr(ph, "mis_synonym_service", bio.mis_synonym_service)
    assert ph.resolve_price_service_name_from_catalog("кровь на сахар", rows=ROWS) == "Глюкоза"


def test_mis_synonym_ignored_when_service_absent_from_price(monkeypatch):
    """Синоним, указывающий на услугу вне прайса региона, не подставляется."""
    fake = bio._MisVocabularies()
    fake.synonym_to_service = {"выдуманный анализ": "Услуга которой нет в прайсе"}
    monkeypatch.setattr(bio, "_vocabularies", lambda: fake)

    from messengers_router.services import _prices_helpers as ph

    monkeypatch.setattr(ph, "mis_synonym_service", bio.mis_synonym_service)
    assert ph.resolve_price_service_name_from_catalog("выдуманный анализ", rows=ROWS) is None


def test_layer_is_silent_without_mis_catalog(monkeypatch):
    """Справочник МИС недоступен — слой молчит, поведение прежнее (fail-open)."""
    monkeypatch.setattr(bio, "_vocabularies", lambda: bio._MisVocabularies())
    assert bio.split_biomaterial_tail("мазок на микрофлору с задней стенки глотки") == (
        "мазок на микрофлору с задней стенки глотки",
        "",
    )
    assert bio.mis_synonym_service("что угодно") is None
    assert bio.service_accepts_biomaterial("Глюкоза", "кровь из вены") is None
