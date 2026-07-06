"""П6 «корзина анализов» (BUG-B): мульти-список анализов распознаётся целиком.

Класс: пациент шлёт СПИСОК анализов (переносы строк, маркеры, или сплошная
строка CAPS без разделителей) → раньше сплиттер (только `, ; + / и`) видел одну
строку → single-путь → ответ про ПЕРВЫЙ анализ. Теперь:
  S1: `\\n` и `•`-маркеры — разделители; шумовые хвосты («Сколько будет стоить?»)
      отбрасываются молча (не попадают ни в услуги, ни в «не распознал»);
  S2: сплошной токен-ран сегментируется ПО КАТАЛОГУ (жадный longest-match:
      лексикон = сам прайс; «ОБЩИЙ БЕЛОК» — 2 токена, «ОАК» — 1);
  S3: нераспознанные позиции честно перечисляются пациенту («Не распознал: …»),
      а не замалчиваются (не дезинформируем полнотой).
Все проверки — на РЕАЛЬНОМ каталоге (region 3), как соседние price-тесты.
"""

from __future__ import annotations

from agent_logic_2.nayka_api import api_price
from messengers_router.renderer import format_price_for_patient
from messengers_router.services._prices_helpers import (
    _build_multi_price_payload,
    _resolve_multi_price_items,
    _split_price_query_items,
)


def _rows():
    return [r for r in api_price.load_price_by_region(3) if isinstance(r, dict)]


# --- S1: переносы строк и маркеры ------------------------------------------

def test_split_newline_list_yields_items_and_drops_price_noise():
    frags = _split_price_query_items("ОАК\nферритин\nвитамин д\nСколько будет стоить?")
    lowered = [f.lower() for f in frags]
    assert "оак" in lowered and "ферритин" in lowered, frags
    assert not any("сколько" in f for f in lowered), f"шумовой хвост попал в услуги: {frags}"


def test_split_bullet_list():
    frags = _split_price_query_items("стоимость • ОАМ • глюкоза • ферритин")
    lowered = [f.lower() for f in frags]
    assert {"оам", "глюкоза", "ферритин"} <= set(lowered), frags


def test_resolve_newline_list_end_to_end():
    items, unrecognized = _resolve_multi_price_items(
        "ОАК\nферритин\nвитамин д\nСколько будет стоить?", _rows()
    )
    names = " | ".join(i["service_name"].lower() for i in items)
    assert len(items) >= 3, names
    assert "общий анализ крови" in names and "ферритин" in names and "витамин d" in names
    assert not unrecognized, unrecognized


# --- S2: сплошная строка без разделителей (сегментация по каталогу) ---------

def test_segment_caps_run_bug_b_replica():
    # Реплика класса BUG-B: CAPS-список одной строкой, мульти-словная позиция
    # («ОБЩИЙ БЕЛОК»), опечатка («ЛАКТАТІ» → Лактат через гомоглиф-фолдинг),
    # шумовой хвост-вопрос.
    q = "ОАК ОАМ ОБЩИЙ БЕЛОК ГЛЮКОЗА ФЕРРИТИН ЛАКТАТІ Сколько будет стоить?"
    items, unrecognized = _resolve_multi_price_items(q, _rows())
    names = " | ".join(i["service_name"].lower() for i in items)
    assert len(items) >= 5, f"распозналось {len(items)}: {names}"
    for need in ("общий анализ крови", "общий анализ мочи", "общий белок", "глюкоза", "ферритин"):
        assert need in names, (need, names)
    assert not any("сколько" in str(u).lower() for u in unrecognized), unrecognized


def test_single_service_query_not_segmented():
    # Регресс: одиночный запрос НЕ должен уходить в multi (гейт ≥2 распознанных).
    items, _ = _resolve_multi_price_items("сколько стоит ттг", _rows())
    assert items == []


def test_non_service_sentence_not_segmented():
    # Регресс: обычная фраза записи не сегментируется в «услуги» по кускам.
    items, _ = _resolve_multi_price_items(
        "запишите меня к кардиологу на завтра после обеда", _rows()
    )
    assert items == []


# --- S3: честный «не распознал» ---------------------------------------------

def test_unrecognized_items_surface_honestly():
    q = "ОАК\nферритин\nКВАНТОВЫЙ АНАЛИЗ АУРЫ\nглюкоза"
    items, unrecognized = _resolve_multi_price_items(q, _rows())
    assert len(items) >= 3
    assert any("аур" in str(u).lower() for u in unrecognized), unrecognized

    payload = _build_multi_price_payload(q, _rows())
    assert payload is not None
    note = str(payload.get("unrecognized_note") or "")
    assert "не распознал" in note.lower() and "аур" in note.lower(), payload.get("unrecognized_note")

    rendered = format_price_for_patient(payload, {})
    assert "не распознал" in rendered.lower(), rendered[-300:]


def test_multi_payload_without_unrecognized_has_no_note():
    payload = _build_multi_price_payload("ОАК, ферритин", _rows())
    assert payload is not None
    assert not payload.get("unrecognized_note")
    rendered = format_price_for_patient(payload, {})
    assert "не распознал" not in rendered.lower()
